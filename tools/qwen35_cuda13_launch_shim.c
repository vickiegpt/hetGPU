#define _GNU_SOURCE

#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int cudaError_t;
typedef void *cudaStream_t;

typedef struct {
    unsigned int x;
    unsigned int y;
    unsigned int z;
} dim3;

typedef struct {
    dim3 gridDim;
    dim3 blockDim;
    size_t dynamicSmemBytes;
    cudaStream_t stream;
    void *attrs;
    unsigned int numAttrs;
} cudaLaunchConfig_t;

typedef void (*register_function_fn)(
    void **,
    const char *,
    char *,
    const char *,
    int,
    void *,
    void *,
    void *,
    void *,
    int *);
typedef cudaError_t (*launch_kernel_fn)(
    const void *, dim3, dim3, void **, size_t, cudaStream_t);
typedef cudaError_t (*launch_kernel_ex_fn)(const cudaLaunchConfig_t *, const void *, void **);
typedef cudaError_t (*func_get_name_fn)(const char **, const void *);
typedef int (*launch_named_fn)(
    const char *,
    unsigned int,
    unsigned int,
    unsigned int,
    unsigned int,
    unsigned int,
    unsigned int,
    unsigned int,
    void *,
    void **,
    void **);

typedef struct function_entry {
    const void *host_function;
    char *name;
    struct function_entry *next;
} function_entry;

static pthread_mutex_t registry_mutex = PTHREAD_MUTEX_INITIALIZER;
static function_entry *registry_head;

static const void *normalize_function(const void *function) {
    return (const void *)((uintptr_t)function & ~(uintptr_t)0x7);
}

static void *resolve_next(const char *symbol) {
    void *address = dlvsym(RTLD_NEXT, symbol, "libcudart.so.13");
    if (!address) {
        address = dlsym(RTLD_NEXT, symbol);
    }
    return address;
}

static void remember_function(const void *host_function, const char *name) {
    if (!host_function || !name || !name[0]) {
        return;
    }
    host_function = normalize_function(host_function);
    pthread_mutex_lock(&registry_mutex);
    for (function_entry *entry = registry_head; entry; entry = entry->next) {
        if (entry->host_function == host_function) {
            pthread_mutex_unlock(&registry_mutex);
            return;
        }
    }
    function_entry *entry = calloc(1, sizeof(*entry));
    if (entry) {
        entry->name = strdup(name);
        if (!entry->name) {
            free(entry);
        } else {
            entry->host_function = host_function;
            entry->next = registry_head;
            registry_head = entry;
        }
    }
    pthread_mutex_unlock(&registry_mutex);
}

static const char *registered_name(const void *host_function) {
    host_function = normalize_function(host_function);
    const char *name = NULL;
    pthread_mutex_lock(&registry_mutex);
    for (function_entry *entry = registry_head; entry; entry = entry->next) {
        if (entry->host_function == host_function) {
            name = entry->name;
            break;
        }
    }
    pthread_mutex_unlock(&registry_mutex);
    return name;
}

static const char *resolve_function_name_with(
    const void *function,
    func_get_name_fn runtime_lookup) {
    const char *name = registered_name(function);
    if (name || !runtime_lookup) {
        return name;
    }
    const char *runtime_name = NULL;
    if (runtime_lookup(&runtime_name, function) != 0 || !runtime_name || !runtime_name[0]) {
        return NULL;
    }
    remember_function(function, runtime_name);
    return registered_name(function);
}

static const char *resolve_function_name(const void *function) {
    static func_get_name_fn runtime_lookup;
    if (!runtime_lookup) {
        runtime_lookup = (func_get_name_fn)resolve_next("cudaFuncGetName");
    }
    return resolve_function_name_with(function, runtime_lookup);
}

#ifdef HETGPU_QWEN35_LAUNCH_SHIM_TEST
const char *hetgpu_qwen35_resolve_function_name_for_test(
    const void *function,
    func_get_name_fn runtime_lookup) {
    return resolve_function_name_with(function, runtime_lookup);
}
#endif

static int env_enabled(const char *name) {
    const char *value = getenv(name);
    return value && value[0] && strcmp(value, "0") != 0 && strcasecmp(value, "false") != 0 &&
           strcasecmp(value, "no") != 0 && strcasecmp(value, "off") != 0;
}

static cudaError_t try_named_launch(
    const void *function,
    dim3 grid,
    dim3 block,
    size_t shared_memory,
    cudaStream_t stream,
    void **arguments) {
    if (!env_enabled("HETGPU_CUDART_PRELAUNCH_NAMED_KERNEL")) {
        return 1;
    }
    const char *name = resolve_function_name(function);
    if (!name) {
        return 1;
    }
    launch_named_fn launch_named = (launch_named_fn)dlsym(RTLD_DEFAULT, "hetgpu_launch_named_kernel");
    if (!launch_named) {
        fprintf(stderr, "[Qwen CUDA13 launch shim] hetgpu_launch_named_kernel is unavailable\n");
        return 999;
    }
    return launch_named(
        name,
        grid.x,
        grid.y,
        grid.z,
        block.x,
        block.y,
        block.z,
        (unsigned int)shared_memory,
        stream,
        arguments,
        NULL);
}

void __cudaRegisterFunction(
    void **fat_binary_handle,
    const char *host_function,
    char *device_function,
    const char *device_name,
    int thread_limit,
    void *thread_id,
    void *block_id,
    void *block_dim,
    void *grid_dim,
    int *warp_size) {
    static register_function_fn real_function;
    if (!real_function) {
        real_function = (register_function_fn)resolve_next("__cudaRegisterFunction");
    }
    if (!real_function) {
        fprintf(stderr, "[Qwen CUDA13 launch shim] real __cudaRegisterFunction is unavailable\n");
        abort();
    }
    real_function(
        fat_binary_handle,
        host_function,
        device_function,
        device_name,
        thread_limit,
        thread_id,
        block_id,
        block_dim,
        grid_dim,
        warp_size);
    remember_function(host_function, device_name);
}

cudaError_t __cudaLaunchKernel(
    const void *function,
    dim3 grid,
    dim3 block,
    void **arguments,
    size_t shared_memory,
    cudaStream_t stream) {
    cudaError_t named = try_named_launch(function, grid, block, shared_memory, stream, arguments);
    if (named == 0 || named == 999) {
        return named;
    }
    static launch_kernel_fn real_function;
    if (!real_function) {
        real_function = (launch_kernel_fn)resolve_next("__cudaLaunchKernel");
    }
    if (!real_function) {
        fprintf(stderr, "[Qwen CUDA13 launch shim] real __cudaLaunchKernel is unavailable\n");
        return 999;
    }
    return real_function(function, grid, block, arguments, shared_memory, stream);
}

cudaError_t cudaLaunchKernel(
    const void *function,
    dim3 grid,
    dim3 block,
    void **arguments,
    size_t shared_memory,
    cudaStream_t stream) {
    cudaError_t named = try_named_launch(function, grid, block, shared_memory, stream, arguments);
    if (named == 0 || named == 999) {
        return named;
    }
    static launch_kernel_fn real_function;
    if (!real_function) {
        real_function = (launch_kernel_fn)resolve_next("cudaLaunchKernel");
    }
    if (!real_function) {
        fprintf(stderr, "[Qwen CUDA13 launch shim] real cudaLaunchKernel is unavailable\n");
        return 999;
    }
    return real_function(function, grid, block, arguments, shared_memory, stream);
}

cudaError_t cudaLaunchKernelExC(
    const cudaLaunchConfig_t *config,
    const void *function,
    void **arguments) {
    if (config) {
        cudaError_t named = try_named_launch(
            function,
            config->gridDim,
            config->blockDim,
            config->dynamicSmemBytes,
            config->stream,
            arguments);
        if (named == 0 || named == 999) {
            return named;
        }
    }
    static launch_kernel_ex_fn real_function;
    if (!real_function) {
        real_function = (launch_kernel_ex_fn)resolve_next("cudaLaunchKernelExC");
    }
    if (!real_function) {
        fprintf(stderr, "[Qwen CUDA13 launch shim] real cudaLaunchKernelExC is unavailable\n");
        return 999;
    }
    return real_function(config, function, arguments);
}
