#include <stddef.h>
#include <stdio.h>
#include <string.h>

typedef int (*func_get_name_fn)(const char **, const void *);

const char *hetgpu_qwen35_resolve_function_name_for_test(
    const void *function,
    func_get_name_fn fallback);

static unsigned int fallback_calls;

static int resolve_multi_token_iq1s(const char **name, const void *function) {
    fallback_calls++;
    if (function != (const void *) 0x1000) {
        return 1;
    }
    *name = "_Z9mul_mat_qIL9ggml_type19ELi16ELb1EEv";
    return 0;
}

int main(void) {
    const void *function = (const void *) 0x1000;
    const char *first = hetgpu_qwen35_resolve_function_name_for_test(
        function, resolve_multi_token_iq1s);
    if (!first || strcmp(first, "_Z9mul_mat_qIL9ggml_type19ELi16ELb1EEv") != 0) {
        fprintf(stderr, "runtime fallback did not resolve the multi-token IQ1_S kernel\n");
        return 1;
    }
    const char *second = hetgpu_qwen35_resolve_function_name_for_test(
        function, resolve_multi_token_iq1s);
    if (!second || strcmp(second, first) != 0 || fallback_calls != 1) {
        fprintf(stderr, "runtime-resolved kernel name was not cached\n");
        return 1;
    }
    if (hetgpu_qwen35_resolve_function_name_for_test(
            (const void *) 0x2000, resolve_multi_token_iq1s) != NULL) {
        fprintf(stderr, "failed cudaFuncGetName lookup did not remain unresolved\n");
        return 1;
    }
    return 0;
}
