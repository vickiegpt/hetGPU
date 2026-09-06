#[cfg(any(feature = "tenstorrent", feature = "nvidia", feature = "sifive"))]
use cuda_types::cuda::*;
#[cfg(feature = "amd")]
use hip_runtime_sys::*;
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
use nvidia_runtime_sys;
#[cfg(feature = "intel")]
use ze_runtime_sys::*;

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
use sifive_runtime_sys;
use std::ptr;
#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
use std::sync::{Mutex, OnceLock};
#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
static SIFIVE_RMSNORM_OFFLOAD_DISABLED_AFTER_FAILURE: AtomicBool = AtomicBool::new(false);
#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
static SIFIVE_MMVF_OFFLOAD_DISABLED_AFTER_FAILURE: AtomicBool = AtomicBool::new(false);
#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
static SIFIVE_DRIVER_KERNEL_NOOP_LAUNCH_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
static SIFIVE_DRIVER_KERNEL_NOOP_LOG_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
static SIFIVE_NAMED_FAILOPEN_LOG_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
static SIFIVE_GENERIC_FAST_SUCCESS_LOG_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
static SIFIVE_NAMED_ERROR_LOG_COUNT: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "intel")]
fn tmatmul_default_cocotb_dir() -> String {
    "/root/matmulfreellm/hardware/ternary_matmul/cocotb".to_string()
}

#[cfg(feature = "intel")]
fn tmatmul_ptx_looks_valid(ptx: &str) -> bool {
    let trimmed = ptx.trim_start();
    if trimmed.len() < 50 {
        return false;
    }
    if !(trimmed.starts_with(".version") || trimmed.starts_with("//")) {
        return false;
    }
    if !(trimmed.contains(".target ") && trimmed.contains(".address_size")) {
        return false;
    }
    if !trimmed.lines().any(|line| {
        let line = line.trim_start();
        line.starts_with(".visible .entry ") || line.starts_with(".entry ")
    }) {
        return false;
    }
    !trimmed
        .bytes()
        .any(|b| b == 0 || b == 0x7f || (b < 0x20 && !matches!(b, b'\n' | b'\r' | b'\t')))
}

#[cfg(feature = "intel")]
fn load_tmatmul_reference_ptx(cocotb_dir: &str) -> Option<(String, String)> {
    let mut candidates = Vec::new();
    if let Ok(path) = std::env::var("HETGPU_TMATMUL_REFERENCE_PTX") {
        if !path.trim().is_empty() {
            candidates.push(std::path::PathBuf::from(path));
        }
    }
    if let Ok(dir) = std::env::var("HETGPU_TMATMUL_REFERENCE_COCOTB_DIR") {
        if !dir.trim().is_empty() {
            candidates.push(std::path::Path::new(&dir).join("run/kernel.ptx"));
        }
    }
    candidates.push(std::path::Path::new(cocotb_dir).join("reference/kernel.ptx"));
    candidates.push(std::path::PathBuf::from(
        "/root/ternary_matmul/cocotb/run/kernel.ptx",
    ));
    candidates.push(std::path::PathBuf::from(
        "/home/victoryang00/ternary_matmul/cocotb/run/kernel.ptx",
    ));

    for path in candidates {
        if !path.is_file() {
            continue;
        }
        match std::fs::read_to_string(&path) {
            Ok(ptx) if tmatmul_ptx_looks_valid(&ptx) => {
                return Some((ptx, path.display().to_string()));
            }
            Ok(ptx) => {
                eprintln!(
                    "[TMatmul Backend] Ignoring invalid reference PTX {} ({} bytes)",
                    path.display(),
                    ptx.len()
                );
            }
            Err(e) => {
                eprintln!(
                    "[TMatmul Backend] Failed to read reference PTX {}: {}",
                    path.display(),
                    e
                );
            }
        }
    }
    None
}

#[cfg(feature = "intel")]
fn select_tmatmul_ptx(runtime_ptx: Option<&str>, cocotb_dir: &str) -> Option<(String, String)> {
    if let Some(ptx) = runtime_ptx {
        if tmatmul_ptx_looks_valid(ptx) {
            return Some((ptx.to_string(), "runtime module".to_string()));
        }
        eprintln!(
            "[TMatmul Backend] Runtime PTX failed validation ({} bytes, starts with {:?})",
            ptx.len(),
            ptx.chars().take(40).collect::<String>()
        );
    }

    load_tmatmul_reference_ptx(cocotb_dir)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[derive(Copy, Clone)]
struct SifiveCachedKernelHandles {
    device: usize,
    program: usize,
    kernel: usize,
}
#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
static SIFIVE_PYTORCH_SOFTMAX_ELF_KERNELS: std::sync::OnceLock<
    std::sync::Mutex<[Option<SifiveCachedKernelHandles>; 4]>,
> = std::sync::OnceLock::new();

#[cfg(feature = "amd")]
pub(crate) fn get_attribute(
    pi: &mut i32,
    cu_attrib: hipFunction_attribute,
    func: hipFunction_t,
) -> hipError_t {
    // TODO: implement HIP_FUNC_ATTRIBUTE_PTX_VERSION
    // TODO: implement HIP_FUNC_ATTRIBUTE_BINARY_VERSION
    unsafe { hipFuncGetAttribute(pi, cu_attrib, func) }?;
    if cu_attrib == hipFunction_attribute::HIP_FUNC_ATTRIBUTE_NUM_REGS {
        *pi = (*pi).max(1);
    }
    Ok(())
}

#[cfg(feature = "intel")]
pub(crate) fn get_attribute(
    pi: &mut i32,
    cu_attrib: ze_kernel_properties_t,
    func: ze_kernel_handle_t,
) -> ze_result_t {
    // For virtual devices or when Level Zero is not available, return sensible defaults
    // This prevents SIGFPE crashes in PyTorch's kernel launch configuration code
    if func.0.is_null() {
        // Virtual device - return reasonable defaults
        *pi = 32; // Default value for most attributes
        return ze_result_t::ZE_RESULT_SUCCESS;
    }

    let mut props = cu_attrib;
    let result = unsafe { zeKernelGetProperties(func, &mut props) };
    if result != ze_result_t::ZE_RESULT_SUCCESS {
        // If Level Zero call fails, return sensible defaults instead of failing
        eprintln!("[hetGPU] zeKernelGetProperties failed, returning defaults");
        *pi = 32; // Safe default
        return ze_result_t::ZE_RESULT_SUCCESS;
    }

    *pi = props.localMemSize as i32;
    ze_result_t::ZE_RESULT_SUCCESS
}

#[cfg(feature = "amd")]
pub(crate) fn launch_kernel(
    f: hipFunction_t,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    block_dim_y: ::core::ffi::c_uint,
    block_dim_z: ::core::ffi::c_uint,
    shared_mem_bytes: ::core::ffi::c_uint,
    stream: hipStream_t,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> hipError_t {
    // TODO: fix constants in extra
    unsafe {
        hipModuleLaunchKernel(
            f,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            block_dim_x,
            block_dim_y,
            block_dim_z,
            shared_mem_bytes,
            stream,
            kernel_params,
            extra,
        )
    }
}

#[cfg(feature = "intel")]
pub(crate) unsafe fn launch_kernel(
    f: &super::module::ZeKernel,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    block_dim_y: ::core::ffi::c_uint,
    block_dim_z: ::core::ffi::c_uint,
    shared_mem_bytes: ::core::ffi::c_uint,
    stream: ze_command_queue_handle_t,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> ze_result_t {
    // Check for checkpoint at launch point
    if super::checkpoint::check_checkpoint_at_launch() {
        // Checkpoint was triggered and pause was requested
        // Return success without executing the kernel
        return ze_result_t::ZE_RESULT_SUCCESS;
    }

    // Start tracking kernel execution for checkpoint support
    let exec_id = super::checkpoint::begin_kernel_execution(
        &f.name,
        (grid_dim_x, grid_dim_y, grid_dim_z),
        (block_dim_x, block_dim_y, block_dim_z),
        shared_mem_bytes,
        stream.0 as u64,
        f.module_handle,
        f.kernel.0 as u64,
    );

    // Record kernel launch for replay if recording is active
    let replay_seq_id = if super::replay::is_recording_active() {
        // Extract kernel arguments for recording
        let mut args = Vec::new();
        if !kernel_params.is_null() {
            // Note: We don't know the exact number of arguments without kernel metadata
            // For now, we record pointer arguments that look valid
            for i in 0..16 {
                let param_ptr = unsafe { *kernel_params.add(i) };
                if param_ptr.is_null() {
                    break;
                }
                // Try to detect if it's a device pointer (typical GPU memory range)
                let value = unsafe { *(param_ptr as *const u64) };
                let arg = if value > 0x1000 && value < 0x800000000000 {
                    // Likely a device pointer
                    super::replay::KernelArgument {
                        index: i as u32,
                        size: 8,
                        arg_type: super::replay::ArgumentType::DevicePointer,
                        device_addr: Some(value),
                        scalar_data: None,
                    }
                } else {
                    // Treat as scalar
                    super::replay::KernelArgument {
                        index: i as u32,
                        size: 8,
                        arg_type: super::replay::ArgumentType::Scalar,
                        device_addr: None,
                        scalar_data: Some(value.to_le_bytes().to_vec()),
                    }
                };
                args.push(arg);
            }
        }

        super::replay::record_kernel_pre_launch(
            &f.name,
            f.module_handle,
            f.kernel.0 as u64,
            (grid_dim_x, grid_dim_y, grid_dim_z),
            (block_dim_x, block_dim_y, block_dim_z),
            shared_mem_bytes,
            stream.0 as u64,
            args,
        )
    } else {
        0
    };
    let kernel_start_time = std::time::Instant::now();

    // Detect virtual backend (no real Level Zero device available)
    let mut virtual_backend = false;
    if let Ok(gs) = crate::r#impl::driver::global_state() {
        if let Some(dev0) = gs.devices.get(0) {
            let (ctx0, _handle0) = dev0.primary_context();
            if ctx0.device.0.is_null() {
                virtual_backend = true;
            }
        }
    }
    // Cocotb fallback: if enabled or kernel handle is null (virtual), execute staged assembly via make
    let use_cocotb = std::env::var("HETGPU_TMATMUL_COCOTB")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false);
    if use_cocotb || f.kernel.0.is_null() || virtual_backend {
        eprintln!(
            "[TMatmul Backend] Kernel launch: {} (grid={},{},{} block={},{},{})",
            f.name, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z
        );

        if block_dim_x == 0 || block_dim_y == 0 || block_dim_z == 0 {
            eprintln!("[TMatmul Backend] WARNING: zero block dimension detected");
        }
        if grid_dim_x == 0 || grid_dim_y == 0 || grid_dim_z == 0 {
            eprintln!("[TMatmul Backend] WARNING: zero grid dimension detected");
        }

        if tmatmul_pre_jit_named_fallback_enabled()
            && (super::cxl_tmatmul::cxl_tmatmul_enabled()
                || tmatmul_named_fallback_enabled()
                || tmatmul_hardware_matmul_enabled())
        {
            match execute_kernel_name_fallback(
                &f.name,
                kernel_params,
                grid_dim_x,
                grid_dim_y,
                grid_dim_z,
                block_dim_x,
                block_dim_y,
                block_dim_z,
            ) {
                KernelNameFallbackStatus::Handled => {
                    eprintln!(
                        "[TMatmul Backend] Kernel '{}' handled by pre-JIT named fallback",
                        f.name
                    );
                    super::checkpoint::end_kernel_execution(exec_id);
                    return ze_result_t::ZE_RESULT_SUCCESS;
                }
                KernelNameFallbackStatus::Rejected => {
                    eprintln!(
                        "[TMatmul Backend] Kernel '{}' rejected by pre-JIT fallback routing",
                        f.name
                    );
                    super::checkpoint::end_kernel_execution(exec_id);
                    return ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE;
                }
                KernelNameFallbackStatus::ContinueNative => {
                    eprintln!(
                        "[TMatmul Backend] Kernel '{}' not handled by pre-JIT named fallback; continuing to PTX JIT",
                        f.name
                    );
                }
            }
        }

        let runtime_ptx = f.ptx_source.as_deref().map(String::as_str);
        let lifted_ptx = if runtime_ptx.is_none() {
            super::module::recover_ptx_source_with_sass_fallback(
                None,
                f.cubin_binary.as_deref().map(Vec::as_slice),
                &f.name,
                super::module::lift_cubin_to_ptx_for_kernel,
            )
        } else {
            None
        };
        if let Some(ref ptx) = lifted_ptx {
            eprintln!(
                "[TMatmul Backend] recovered {} bytes of PTX for '{}' with SASS lifter",
                ptx.len(),
                f.name
            );
        }
        let ptx_source = runtime_ptx.or(lifted_ptx.as_deref());
        let mut continue_native_launch = false;
        let mut rejected_launch = false;
        match super::cxl_tmatmul::compile_ptx_to_tmatmul_assembly(ptx_source) {
            Ok(compiled) => {
                let ptx_source = ptx_source.expect("compile requires PTX source");
                eprintln!(
                    "[TMatmul Backend] PTX JIT lowered '{}' to tmatmul assembly (ptx={} bytes, asm={} bytes)",
                    f.name,
                    compiled.source_len,
                    compiled.assembly.len()
                );

                match super::cxl_tmatmul::stage_jit_artifacts(&f.name, ptx_source, &compiled) {
                    Ok(artifacts) => eprintln!(
                        "[TMatmul Backend] staged PTX={} ASM={}",
                        artifacts.ptx_path.display(),
                        artifacts.asm_path.display()
                    ),
                    Err(e) => eprintln!("[TMatmul Backend] artifact staging failed: {e}"),
                }

                let _ = std::fs::write("/tmp/tmatmul_kernel.S", compiled.assembly.as_bytes());

                if use_cocotb {
                    let cocotb_dir = std::env::var("HETGPU_TMATMUL_COCOTB_DIR")
                        .unwrap_or_else(|_| tmatmul_default_cocotb_dir());
                    let run_dir = std::path::Path::new(&cocotb_dir).join("run");
                    if let Err(e) = std::fs::create_dir_all(&run_dir) {
                        eprintln!(
                            "[TMatmul Backend] failed to create cocotb run dir {}: {}",
                            run_dir.display(),
                            e
                        );
                    } else {
                        let _ = std::fs::write(run_dir.join("kernel.ptx"), ptx_source.as_bytes());
                        let _ =
                            std::fs::write(run_dir.join("kernel.S"), compiled.assembly.as_bytes());
                    }
                }

                let interpreter_enabled = std::env::var("HETGPU_TMATMUL_INTERPRETER")
                    .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
                    .unwrap_or(false);
                if interpreter_enabled && !kernel_params.is_null() {
                    match super::tmatmul_interpreter::execute_assembly(
                        &compiled.assembly,
                        kernel_params,
                        (grid_dim_x, grid_dim_y, grid_dim_z),
                        (block_dim_x, block_dim_y, block_dim_z),
                    ) {
                        Ok(()) => eprintln!(
                            "[TMatmul Interpreter] Kernel '{}' executed from JIT assembly",
                            f.name
                        ),
                        Err(e) => eprintln!(
                            "[TMatmul Interpreter] JIT assembly execution failed for '{}': {}",
                            f.name, e
                        ),
                    }
                }

                let cxl_enabled = super::cxl_tmatmul::cxl_tmatmul_enabled();
                if cxl_enabled {
                    eprintln!(
                        "[CXL TMatmul] PTX JIT completed; trying kernel-name fallback while .S-to-AFU instruction encoding is unavailable"
                    );
                }
                if cxl_enabled
                    || tmatmul_named_fallback_enabled()
                    || tmatmul_hardware_matmul_enabled()
                {
                    match execute_kernel_name_fallback(
                        &f.name,
                        kernel_params,
                        grid_dim_x,
                        grid_dim_y,
                        grid_dim_z,
                        block_dim_x,
                        block_dim_y,
                        block_dim_z,
                    ) {
                        KernelNameFallbackStatus::Handled => {}
                        KernelNameFallbackStatus::Rejected => {
                            rejected_launch = true;
                        }
                        KernelNameFallbackStatus::ContinueNative => {
                            continue_native_launch = true;
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!(
                    "[TMatmul Backend] PTX JIT unavailable for '{}': {}; trying kernel-name fallback if enabled",
                    f.name, e
                );
                match execute_kernel_name_fallback(
                    &f.name,
                    kernel_params,
                    grid_dim_x,
                    grid_dim_y,
                    grid_dim_z,
                    block_dim_x,
                    block_dim_y,
                    block_dim_z,
                ) {
                    KernelNameFallbackStatus::Handled => {}
                    KernelNameFallbackStatus::Rejected => {
                        rejected_launch = true;
                    }
                    KernelNameFallbackStatus::ContinueNative => {
                        continue_native_launch = true;
                    }
                }
            }
        }

        if rejected_launch {
            eprintln!(
                "[TMatmul Backend] Kernel '{}' rejected by strict fallback routing",
                f.name
            );
            super::checkpoint::end_kernel_execution(exec_id);
            return ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE;
        }

        if !continue_native_launch {
            super::checkpoint::end_kernel_execution(exec_id);
            return ze_result_t::ZE_RESULT_SUCCESS;
        }

        if f.kernel.0.is_null() || virtual_backend {
            eprintln!(
                "[TMatmul Backend] Kernel '{}' requested native GPU path, but no native Level Zero kernel is available",
                f.name
            );
            super::checkpoint::end_kernel_execution(exec_id);
            return ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE;
        }

        eprintln!(
            "[TMatmul Backend] Kernel '{}' continuing on native Level Zero path",
            f.name
        );
    }

    // Set the group size (equivalent to CUDA block dimensions)
    let result = unsafe { zeKernelSetGroupSize(f.kernel, block_dim_x, block_dim_y, block_dim_z) };

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        super::checkpoint::end_kernel_execution(exec_id);
        return result;
    }

    // Set arguments from kernel_params if provided
    if !kernel_params.is_null() {
        let mut param_index = 0;
        let mut current_param = kernel_params;

        while !(*current_param).is_null() {
            unsafe {
                let param_value = *current_param;
                let result = zeKernelSetArgumentValue(
                    f.kernel,
                    param_index,
                    std::mem::size_of::<*mut ::core::ffi::c_void>(),
                    param_value as *const ::core::ffi::c_void,
                );

                if result != ze_result_t::ZE_RESULT_SUCCESS {
                    super::checkpoint::end_kernel_execution(exec_id);
                    return result;
                }

                param_index += 1;
                current_param = current_param.add(1);
            }
        }
    }

    // Process 'extra' parameters if provided (e.g., shared memory size)
    if !extra.is_null() {
        // 'extra' is typically of the form [KEY1, VALUE1, KEY2, VALUE2, ..., 0]
        unsafe {
            let mut i = 0;
            loop {
                let key = *extra.add(i);
                if key.is_null() {
                    break;
                }

                let key_value = key as usize;
                let value_ptr = extra.add(i + 1);
                let value = *value_ptr;

                if key_value == 1 { // CU_LAUNCH_PARAM_BUFFER_SHARED_MEMORY
                     // shared memory is already set via the shared_mem_bytes parameter
                }

                i += 2;
            }
        }
    }

    // Get or create a command list for this stream
    let command_list = unsafe {
        // In a real implementation, you'd have a way to get or create a command list for the given stream
        // For simplicity, we'll assume some function exists to do this
        get_or_create_command_list_for_stream(stream)
    };

    if command_list.0.is_null() {
        super::checkpoint::end_kernel_execution(exec_id);
        return ze_result_t::ZE_RESULT_ERROR_UNINITIALIZED;
    }

    // Prepare launch arguments for grid dimensions
    let dispatch_args = ze_group_count_t {
        groupCountX: grid_dim_x,
        groupCountY: grid_dim_y,
        groupCountZ: grid_dim_z,
    };

    // Launch the kernel
    let result = unsafe {
        zeCommandListAppendLaunchKernel(
            command_list,
            f.kernel,
            &dispatch_args,
            ze_event_handle_t(ptr::null_mut()), // No event to signal
            0,                                  // No events to wait on
            ptr::null_mut(),                    // No event to signal
        )
    };

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        super::checkpoint::end_kernel_execution(exec_id);
        return result;
    }

    // Close and execute the command list (in a real implementation, this might be deferred)
    let result = unsafe { zeCommandListClose(command_list) };

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        super::checkpoint::end_kernel_execution(exec_id);
        return result;
    }

    let result = unsafe {
        // Execute the command list
        zeCommandQueueExecuteCommandLists(
            stream,
            1,
            &command_list,
            ze_fence_handle_t(ptr::null_mut()),
        )
    };

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        super::checkpoint::end_kernel_execution(exec_id);
        return result;
    }

    // If this is a synchronous stream, synchronize immediately
    let is_synchronous = false; // In a real implementation, determine if stream is synchronous

    if is_synchronous {
        let result = unsafe { zeCommandQueueSynchronize(stream, u64::MAX) };

        if result != ze_result_t::ZE_RESULT_SUCCESS {
            super::checkpoint::end_kernel_execution(exec_id);
            return result;
        }
    }

    // End kernel execution tracking
    super::checkpoint::end_kernel_execution(exec_id);

    // Record post-launch for replay
    if replay_seq_id > 0 && super::replay::is_recording_active() {
        let execution_time_ns = kernel_start_time.elapsed().as_nanos() as u64;
        super::replay::record_kernel_post_launch(replay_seq_id, execution_time_ns);
    }

    ze_result_t::ZE_RESULT_SUCCESS
}

/// Check if a memory range is readable by writing to a pipe (forces kernel to copy_from_user).
/// Returns true if the start and end of the range are accessible.
#[cfg(feature = "intel")]
#[allow(dead_code)]
unsafe fn is_memory_readable(ptr: *const u8, len: usize) -> bool {
    if ptr.is_null() || len == 0 {
        return false;
    }
    let mut pipefd = [0i32; 2];
    if libc::pipe(pipefd.as_mut_ptr()) != 0 {
        return false;
    }
    // Set pipe write end to non-blocking to avoid hanging on large writes
    libc::fcntl(pipefd[1], libc::F_SETFL, libc::O_NONBLOCK);

    // Probe the first 4 bytes
    let probe_start = libc::write(pipefd[1], ptr as *const libc::c_void, 4.min(len));
    // Drain the pipe
    let mut drain_buf = [0u8; 4];
    let _ = libc::read(pipefd[0], drain_buf.as_mut_ptr() as *mut libc::c_void, 4);

    // Probe the last 4 bytes if range is larger than 4
    let probe_end = if len > 4 {
        let end_ptr = ptr.add(len - 4);
        let r = libc::write(pipefd[1], end_ptr as *const libc::c_void, 4);
        let _ = libc::read(pipefd[0], drain_buf.as_mut_ptr() as *mut libc::c_void, 4);
        r
    } else {
        probe_start
    };

    libc::close(pipefd[0]);
    libc::close(pipefd[1]);
    probe_start > 0 && probe_end > 0
}

/// Invoke the Python-based TMatmul emulator bridge to execute compiled assembly.
/// Writes tensor data to temp files, runs the Python bridge, and reads results back.
#[cfg(feature = "intel")]
#[allow(dead_code)]
unsafe fn invoke_emulator_bridge(
    asm_path: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    numel: usize,
    kernel_name: &str,
    num_pointer_params: usize, // Known count of pointer params (from assembly/kernel name)
) {
    use std::fmt::Write;

    // Maximum elements we'll try to read (256MB / 4 bytes = 64M elements)
    const MAX_ELEMENTS: usize = 64 * 1024 * 1024;

    // We KNOW the param layout:
    // - First `num_pointer_params` params are 64-bit device pointers
    // - We DON'T try to read scalar params (would crash if kernel_params is shorter)
    // - Instead, derive numel from allocation sizes or grid*block dimensions

    struct ParamEntry {
        value: u64,
        alloc_size: usize,
    }
    let mut params: Vec<ParamEntry> = Vec::new();

    // Read ONLY the pointer params - these are guaranteed to exist for the kernel
    for i in 0..num_pointer_params {
        let param_slot = kernel_params.add(i);
        let param_addr = *param_slot;
        if param_addr.is_null() {
            eprintln!("[TMatmul Emulator] Param {} is null, stopping", i);
            break;
        }
        // Sanity check: param_addr should be a valid stack/heap address
        if (param_addr as usize) < 0x1000 {
            eprintln!(
                "[TMatmul Emulator] Param {} addr too low ({:#x}), stopping",
                i, param_addr as usize
            );
            break;
        }

        let ptr_value = (param_addr as *const u64).read_unaligned();
        let alloc_size = super::memory::get_alloc_size(ptr_value as usize).unwrap_or(0);
        eprintln!(
            "[TMatmul Emulator] Param {}: ptr_value={:#x}, alloc_size={}",
            i, ptr_value, alloc_size
        );
        params.push(ParamEntry {
            value: ptr_value,
            alloc_size,
        });
    }

    if params.is_empty() || params.len() < 2 {
        eprintln!(
            "[TMatmul Emulator] Not enough pointer params ({}) for kernel '{}'",
            params.len(),
            kernel_name
        );
        return;
    }

    // Check if any params are tracked in our virtual allocation map
    let tracked_count = params.iter().filter(|p| p.alloc_size > 0).count();
    if tracked_count == 0 {
        eprintln!(
            "[TMatmul Emulator] No params found in virtual alloc map for '{}' - memory not managed by hetGPU",
            kernel_name
        );
        return;
    }

    // Use grid*block numel as primary size, capped by allocation size
    // (PyTorch caching allocator blocks are much larger than individual tensors)
    let mut actual_numel = numel;

    let min_alloc_elements: Option<usize> = params
        .iter()
        .filter(|p| p.alloc_size > 0)
        .map(|p| p.alloc_size / 4) // f32 = 4 bytes
        .min();

    if let Some(alloc_elements) = min_alloc_elements {
        // Use alloc_elements as upper bound - don't read past allocation
        actual_numel = actual_numel.min(alloc_elements);
    }

    // Safety cap
    if actual_numel > MAX_ELEMENTS {
        actual_numel = MAX_ELEMENTS;
    }
    if actual_numel == 0 {
        eprintln!(
            "[TMatmul Emulator] numel is 0, nothing to do for '{}'",
            kernel_name
        );
        return;
    }

    eprintln!(
        "[TMatmul Emulator] kernel='{}', numel={}, pointer_params={}",
        kernel_name,
        actual_numel,
        params.len()
    );

    // Build JSON and write data files - all params are pointers
    let mut params_json = String::from("[");
    let name_lower = kernel_name.to_lowercase();
    let matmul_layout = tmatmul_hardware_matmul_layout(&name_lower);
    let matrix_param = matmul_layout.map(|layout| layout.matrix_param);
    struct ParamInfo {
        host_ptr: *mut u8,
        file_path: String,
        count: usize,
    }
    let mut param_infos: Vec<ParamInfo> = Vec::new();

    for (i, p) in params.iter().enumerate() {
        if i > 0 {
            params_json.push(',');
        }

        let file_path = format!("/tmp/hetgpu_param_{}.bin", i);
        let requested_count = if matrix_param == Some(i) {
            let dim = tmatmul_env_usize("HETGPU_TMATMUL_NUMEL").unwrap_or(actual_numel);
            dim.checked_mul(dim)
                .unwrap_or(MAX_ELEMENTS)
                .min(MAX_ELEMENTS)
        } else {
            actual_numel
        };
        let max_elements = if p.alloc_size > 0 {
            p.alloc_size / 4
        } else {
            requested_count
        };
        let count = requested_count.min(max_elements).min(MAX_ELEMENTS);
        let size_bytes = count * 4;
        let data_ptr = p.value as *const u8;

        if matrix_param == Some(i) && super::cxl_tmatmul::matrix_stage_cuda_dax_enabled() {
            let dim = tmatmul_env_usize("HETGPU_TMATMUL_NUMEL").unwrap_or(actual_numel);
            let stage_bytes = super::cxl_tmatmul::matrix_bytes(dim).unwrap_or(size_bytes);
            match super::cxl_tmatmul::cuda_dax_bridge_param_json(p.value, stage_bytes, count) {
                Ok(json) => params_json.push_str(&json),
                Err(e) => {
                    eprintln!(
                        "[TMatmul Emulator] Param {} cuda_dax serialization failed: {}",
                        i, e
                    );
                    let _ = write!(params_json, r#"{{"file":"","count":0,"is_pointer":true}}"#);
                }
            }
            param_infos.push(ParamInfo {
                host_ptr: std::ptr::null_mut(),
                file_path: String::new(),
                count: 0,
            });
            continue;
        }

        // Validate: tracked allocations are preferred, but some ggml CUDA
        // params are readable host pointers that never enter VIRTUAL_ALLOC_MAP.
        let ptr_val = p.value as usize;
        let ptr_aligned = ptr_val % 4 == 0;
        let ptr_in_range = ptr_val >= 0x10000 && ptr_val < 0x7fff_ffff_ffff;
        if count == 0
            || data_ptr.is_null()
            || !ptr_aligned
            || !ptr_in_range
            || size_bytes > 256 * 1024 * 1024
        {
            eprintln!(
                "[TMatmul Emulator] Param {} skipped: count={}, ptr={:#x}, alloc_size={}, aligned={}, in_range={}",
                i, count, p.value, p.alloc_size, ptr_aligned, ptr_in_range
            );
            let _ = write!(params_json, r#"{{"file":"","count":0,"is_pointer":true}}"#);
            param_infos.push(ParamInfo {
                host_ptr: std::ptr::null_mut(),
                file_path: String::new(),
                count: 0,
            });
            continue;
        }

        // Validate memory is readable before attempting to copy
        if !is_memory_readable(data_ptr, size_bytes) {
            eprintln!(
                "[TMatmul Emulator] Param {} memory not readable: ptr={:#x}, size={}",
                i, p.value, size_bytes
            );
            let _ = write!(params_json, r#"{{"file":"","count":0,"is_pointer":true}}"#);
            param_infos.push(ParamInfo {
                host_ptr: std::ptr::null_mut(),
                file_path: String::new(),
                count: 0,
            });
            continue;
        }

        // Write tensor data to temp file
        let data_slice = std::slice::from_raw_parts(data_ptr, size_bytes);
        if let Err(e) = std::fs::write(&file_path, data_slice) {
            eprintln!("[TMatmul Emulator] Failed to write param {} data: {}", i, e);
            return;
        }

        let _ = write!(
            params_json,
            r#"{{"file":"{}","count":{},"is_pointer":true}}"#,
            file_path, count
        );

        param_infos.push(ParamInfo {
            host_ptr: data_ptr as *mut u8,
            file_path: file_path.clone(),
            count,
        });
    }

    params_json.push(']');
    let param_count = params.len();

    // Build the full JSON config
    let config_json = format!(
        r#"{{"assembly_file":"{}","params":{},"numel":{}}}"#,
        asm_path, params_json, actual_numel
    );

    let config_path = "/tmp/tmatmul_bridge_config.json";
    if let Err(e) = std::fs::write(config_path, &config_json) {
        eprintln!("[TMatmul Emulator] Failed to write config: {}", e);
        return;
    }

    eprintln!(
        "[TMatmul Emulator] Invoking bridge for '{}' (numel={}, params={})",
        kernel_name, actual_numel, param_count
    );

    // Invoke the Python bridge
    let bridge_script = std::env::var("HETGPU_TMATMUL_BRIDGE")
        .unwrap_or_else(|_| "/root/ternary_matmul/sw_utils/lib/hetgpu_bridge.py".to_string());

    let python = std::env::var("HETGPU_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let bridge_pythonpath = std::env::var("HETGPU_TMATMUL_PYTHONPATH")
        .or_else(|_| std::env::var("PYTHONPATH"))
        .unwrap_or_else(|_| "/root/ternary_matmul/sw_utils".to_string());

    #[cfg(test)]
    TMATMUL_BRIDGE_INVOKE_TEST_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

    match std::process::Command::new(&python)
        .arg(&bridge_script)
        .arg(config_path)
        .env("PYTHONPATH", bridge_pythonpath)
        .stderr(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .output()
    {
        Ok(output) => {
            if !output.stderr.is_empty() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                for line in stderr.lines() {
                    eprintln!("[TMatmul Emulator] {}", line);
                }
            }
            if output.status.success() {
                tmatmul_note_bridge_success();
                eprintln!(
                    "[TMatmul Emulator] Kernel '{}' executed successfully",
                    kernel_name
                );

                // Read back output data from files written by the bridge
                // The bridge writes modified PARAM_N data back to the same file paths
                for info in &param_infos {
                    if info.host_ptr.is_null() {
                        continue;
                    }
                    // Check if the bridge wrote an output file
                    let out_path = format!("{}.out", info.file_path);
                    if let Ok(data) = std::fs::read(&out_path) {
                        let expected_size = info.count * 4;
                        if data.len() == expected_size
                            && is_memory_readable(info.host_ptr, expected_size)
                        {
                            std::ptr::copy_nonoverlapping(
                                data.as_ptr(),
                                info.host_ptr,
                                expected_size,
                            );
                        }
                        let _ = std::fs::remove_file(&out_path);
                    }
                    let _ = std::fs::remove_file(&info.file_path);
                }
            } else {
                eprintln!(
                    "[TMatmul Emulator] Bridge exited with code {:?} for '{}'",
                    output.status.code(),
                    kernel_name
                );
                tmatmul_note_bridge_failure(kernel_name);
                // Clean up temp files
                for info in &param_infos {
                    let _ = std::fs::remove_file(&info.file_path);
                }
            }
        }
        Err(e) => {
            eprintln!("[TMatmul Emulator] Failed to invoke bridge: {}", e);
            tmatmul_note_bridge_failure(kernel_name);
        }
    }
}

/// Helper: log unhandled kernel launches in the fallback path.
/// The emulator bridge is only invoked when PTX compilation succeeds (not from this fallback).
#[cfg(feature = "intel")]
#[allow(dead_code)]
fn tmatmul_env_truthy(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

#[cfg(feature = "intel")]
#[allow(dead_code)]
fn tmatmul_env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
}

#[cfg(feature = "intel")]
#[allow(dead_code)]
fn tmatmul_env_enabled_default(name: &str, default_value: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default_value,
        },
        Err(_) => default_value,
    }
}

#[cfg(feature = "intel")]
#[allow(dead_code)]
fn tmatmul_named_fallback_enabled() -> bool {
    tmatmul_env_truthy("HETGPU_TMATMUL_NAMED_FALLBACK")
}

#[cfg(feature = "intel")]
#[allow(dead_code)]
fn tmatmul_pre_jit_named_fallback_enabled() -> bool {
    tmatmul_env_truthy("HETGPU_TMATMUL_PRE_JIT_NAMED_FALLBACK")
}

#[cfg(feature = "intel")]
#[allow(dead_code)]
fn tmatmul_hardware_matmul_enabled() -> bool {
    tmatmul_env_truthy("HETGPU_TMATMUL_HARDWARE_MATMUL")
}

#[cfg(feature = "intel")]
static TMATMUL_CXL_MATMUL_FAILURES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "intel")]
fn tmatmul_cxl_matmul_failure_limit() -> Option<usize> {
    tmatmul_env_usize("HETGPU_CXL_TMATMUL_MAX_FAILURES").filter(|limit| *limit > 0)
}

#[cfg(feature = "intel")]
fn tmatmul_cxl_matmul_demoted() -> bool {
    tmatmul_cxl_matmul_failure_limit().is_some_and(|limit| {
        TMATMUL_CXL_MATMUL_FAILURES.load(std::sync::atomic::Ordering::SeqCst) >= limit
    })
}

#[cfg(feature = "intel")]
fn tmatmul_note_cxl_matmul_failure(kernel_name: &str) {
    let Some(limit) = tmatmul_cxl_matmul_failure_limit() else {
        return;
    };
    let failures =
        TMATMUL_CXL_MATMUL_FAILURES.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
    if failures >= limit {
        eprintln!(
            "[CXL TMatmul] Demoting future matmul CXL submits after {} failure(s); last='{}'",
            failures, kernel_name
        );
    }
}

#[cfg(feature = "intel")]
fn tmatmul_note_cxl_matmul_success() {
    if tmatmul_cxl_matmul_failure_limit().is_some() {
        TMATMUL_CXL_MATMUL_FAILURES.store(0, std::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(feature = "intel")]
static TMATMUL_BRIDGE_FAILURES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "intel")]
fn tmatmul_bridge_failure_limit() -> Option<usize> {
    tmatmul_env_usize("HETGPU_TMATMUL_BRIDGE_MAX_FAILURES").filter(|limit| *limit > 0)
}

#[cfg(feature = "intel")]
fn tmatmul_bridge_demoted() -> bool {
    tmatmul_bridge_failure_limit().is_some_and(|limit| {
        TMATMUL_BRIDGE_FAILURES.load(std::sync::atomic::Ordering::SeqCst) >= limit
    })
}

#[cfg(feature = "intel")]
fn tmatmul_note_bridge_failure(kernel_name: &str) {
    let Some(limit) = tmatmul_bridge_failure_limit() else {
        return;
    };
    let failures = TMATMUL_BRIDGE_FAILURES.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
    if failures >= limit {
        eprintln!(
            "[TMatmul Emulator] Demoting future bridge invocations after {} failure(s); last='{}'",
            failures, kernel_name
        );
    }
}

#[cfg(feature = "intel")]
fn tmatmul_note_bridge_success() {
    if tmatmul_bridge_failure_limit().is_some() {
        TMATMUL_BRIDGE_FAILURES.store(0, std::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(feature = "intel")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TmatmulHardwareMatmulLayout {
    name: &'static str,
    matrix_param: usize,
    vector_param: usize,
    output_param: usize,
    pointer_params: usize,
}

#[cfg(feature = "intel")]
impl TmatmulHardwareMatmulLayout {
    fn with_env_overrides(mut self) -> Self {
        if let Some(value) = tmatmul_env_usize("HETGPU_TMATMUL_MATRIX_PARAM") {
            self.matrix_param = value;
        }
        if let Some(value) = tmatmul_env_usize("HETGPU_TMATMUL_VECTOR_PARAM") {
            self.vector_param = value;
        }
        if let Some(value) = tmatmul_env_usize("HETGPU_TMATMUL_OUTPUT_PARAM") {
            self.output_param = value;
        }
        if let Some(value) = tmatmul_env_usize("HETGPU_TMATMUL_POINTER_PARAMS") {
            self.pointer_params = value;
        }
        self.pointer_params = self
            .pointer_params
            .max(self.matrix_param + 1)
            .max(self.vector_param + 1)
            .max(self.output_param + 1);
        self
    }
}

#[cfg(feature = "intel")]
#[allow(dead_code)]
fn tmatmul_is_matmul_kernel_name(name_lower: &str) -> bool {
    if name_lower.contains("mul_mat_q_stream_k_fixup") || name_lower.contains("stream_k_fixup") {
        return false;
    }
    name_lower.contains("mul_mat")
        || name_lower.contains("gemm")
        || name_lower.contains("matmul")
        || name_lower.contains("mm_")
        || name_lower.contains("dot")
        || name_lower.contains("cublas")
}

#[cfg(feature = "intel")]
#[allow(dead_code)]
fn tmatmul_hardware_matmul_layout(name_lower: &str) -> Option<TmatmulHardwareMatmulLayout> {
    if !tmatmul_is_matmul_kernel_name(name_lower) {
        return None;
    }

    let layout = if name_lower.contains("mul_mat_f_ids") {
        TmatmulHardwareMatmulLayout {
            name: "mul_mat_f_ids",
            matrix_param: 0,
            vector_param: 1,
            output_param: 5,
            pointer_params: 6,
        }
    } else if name_lower.contains("mul_mat_vec_f") {
        TmatmulHardwareMatmulLayout {
            name: "mul_mat_vec_f",
            matrix_param: 0,
            vector_param: 1,
            output_param: 2,
            pointer_params: 3,
        }
    } else if name_lower.contains("mul_mat_f") {
        TmatmulHardwareMatmulLayout {
            name: "mul_mat_f",
            matrix_param: 0,
            vector_param: 1,
            output_param: 3,
            pointer_params: 4,
        }
    } else if name_lower.contains("mul_mat_vec_q_moe") {
        TmatmulHardwareMatmulLayout {
            name: "mul_mat_vec_q_moe",
            matrix_param: 0,
            vector_param: 1,
            output_param: 3,
            pointer_params: 4,
        }
    } else if name_lower.contains("mul_mat_vec_q") {
        TmatmulHardwareMatmulLayout {
            name: "mul_mat_vec_q",
            matrix_param: 0,
            vector_param: 1,
            output_param: 2,
            pointer_params: 3,
        }
    } else if name_lower.contains("mul_mat_q") {
        TmatmulHardwareMatmulLayout {
            name: "mul_mat_q",
            matrix_param: 0,
            vector_param: 1,
            output_param: 2,
            pointer_params: 4,
        }
    } else {
        TmatmulHardwareMatmulLayout {
            name: "generic",
            matrix_param: 0,
            vector_param: 1,
            output_param: 2,
            pointer_params: 3,
        }
    };

    Some(layout.with_env_overrides())
}

#[cfg(feature = "intel")]
#[allow(dead_code)]
fn tmatmul_generate_hardware_matmul_assembly(
    kernel_name: &str,
    layout: TmatmulHardwareMatmulLayout,
) -> String {
    format!(
        "; IA-780I hardware matmul fallback generated from kernel name
         ; kernel: {kernel_name}
         ; layout: {layout_name} matrix=PARAM_{matrix} vector=PARAM_{vector} output=PARAM_{output}
         ; BIND PARAM_{matrix} matrix
         ; BIND PARAM_{vector} vector
         ; BIND PARAM_{output} output
         ldv v0,PARAM_{vector}
         tmatmul_import v0
         tmatmul_go PARAM_{matrix}
         tmatmul_export v1
         sv v1,PARAM_{output}
         stall
",
        kernel_name = kernel_name,
        layout_name = layout.name,
        matrix = layout.matrix_param,
        vector = layout.vector_param,
        output = layout.output_param,
    )
}

#[cfg(feature = "intel")]
#[allow(dead_code)]
fn tmatmul_expected_vector_bytes() -> usize {
    let dim = tmatmul_env_usize("HETGPU_TMATMUL_NUMEL").unwrap_or(2048);
    super::cxl_tmatmul::vector_bytes(dim).unwrap_or(4096)
}

#[cfg(feature = "intel")]
#[allow(dead_code)]
unsafe fn tmatmul_read_device_pointer_param(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
    kernel_name: &str,
    allow_unreadable_cuda_device: bool,
) -> Result<(usize, usize), super::cxl_tmatmul::CxlTmatmulError> {
    if kernel_params.is_null() {
        return Err(super::cxl_tmatmul::CxlTmatmulError::Device(format!(
            "kernel '{kernel_name}' has null kernel_params"
        )));
    }

    let param_slot = kernel_params.add(index);
    let param_addr = *param_slot;
    if param_addr.is_null() || (param_addr as usize) < 0x1000 {
        return Err(super::cxl_tmatmul::CxlTmatmulError::Device(format!(
            "kernel '{kernel_name}' PARAM_{index} has invalid parameter slot {:#x}",
            param_addr as usize
        )));
    }
    if !is_memory_readable(param_addr as *const u8, std::mem::size_of::<u64>()) {
        return Err(super::cxl_tmatmul::CxlTmatmulError::Device(format!(
            "kernel '{kernel_name}' PARAM_{index} parameter slot is not readable"
        )));
    }

    let ptr_value = (param_addr as *const u64).read_unaligned() as usize;
    let alloc_size = super::memory::get_alloc_size(ptr_value).unwrap_or(0);
    if ptr_value < 0x1000 {
        return Err(super::cxl_tmatmul::CxlTmatmulError::Device(format!(
            "kernel '{kernel_name}' PARAM_{index} has invalid pointer value {ptr_value:#x}"
        )));
    }
    if alloc_size == 0 {
        if allow_unreadable_cuda_device {
            let alloc_hint = tmatmul_env_usize("HETGPU_TMATMUL_MATRIX_BYTES").unwrap_or(usize::MAX);
            return Ok((ptr_value, alloc_hint));
        }
        if index == 1 {
            let expected_bytes = tmatmul_expected_vector_bytes();
            if is_memory_readable(ptr_value as *const u8, expected_bytes) {
                return Ok((ptr_value, expected_bytes));
            }
            return Err(super::cxl_tmatmul::CxlTmatmulError::Device(format!(
                "kernel '{kernel_name}' PARAM_{index} ptr={ptr_value:#x} is not a tracked hetGPU allocation and is not readable for {expected_bytes} bytes"
            )));
        }
        return Err(super::cxl_tmatmul::CxlTmatmulError::Device(format!(
            "kernel '{kernel_name}' PARAM_{index} ptr={ptr_value:#x} is not a tracked hetGPU allocation"
        )));
    }
    if !is_memory_readable(ptr_value as *const u8, alloc_size.min(4096)) {
        if allow_unreadable_cuda_device {
            return Ok((ptr_value, alloc_size));
        }
        return Err(super::cxl_tmatmul::CxlTmatmulError::Device(format!(
            "kernel '{kernel_name}' PARAM_{index} ptr={ptr_value:#x} allocation is not readable"
        )));
    }

    Ok((ptr_value, alloc_size))
}

#[cfg(feature = "intel")]
#[allow(dead_code)]
unsafe fn submit_cxl_hardware_matmul_fallback(
    kernel_name: &str,
    layout: TmatmulHardwareMatmulLayout,
    assembly: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Result<super::cxl_tmatmul::CxlTmatmulRunStatus, super::cxl_tmatmul::CxlTmatmulError> {
    #[cfg(test)]
    TMATMUL_CXL_SUBMIT_TEST_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

    let allow_cuda_dax_matrix = super::cxl_tmatmul::matrix_stage_cuda_dax_enabled();
    let (matrix_ptr, matrix_alloc) = tmatmul_read_device_pointer_param(
        kernel_params,
        layout.matrix_param,
        kernel_name,
        allow_cuda_dax_matrix,
    )?;
    let (input_ptr, input_alloc) =
        tmatmul_read_device_pointer_param(kernel_params, layout.vector_param, kernel_name, false)?;
    let (output_ptr, output_alloc) =
        tmatmul_read_device_pointer_param(kernel_params, layout.output_param, kernel_name, false)?;

    let matrix_offset = super::cxl_tmatmul::matrix_dpa_offset()?;
    let labels = std::collections::HashMap::from([
        (format!("PARAM_{}", layout.matrix_param), matrix_offset),
        (
            format!("PARAM_{}", layout.vector_param),
            super::cxl_tmatmul::TMATMUL_DPA_INPUT,
        ),
        (
            format!("PARAM_{}", layout.output_param),
            super::cxl_tmatmul::TMATMUL_DPA_OUTPUT,
        ),
    ]);
    let timeout_ms = tmatmul_env_usize("HETGPU_CXL_TMATMUL_TIMEOUT_MS")
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(0);

    eprintln!(
        "[CXL TMatmul] staging '{}' matrix=PARAM_{} ptr={:#x}/{} vector=PARAM_{} ptr={:#x}/{} output=PARAM_{} ptr={:#x}/{}",
        kernel_name,
        layout.matrix_param,
        matrix_ptr,
        matrix_alloc,
        layout.vector_param,
        input_ptr,
        input_alloc,
        layout.output_param,
        output_ptr,
        output_alloc,
    );

    super::cxl_tmatmul::submit_hardware_matmul_from_ptrs(
        assembly,
        &labels,
        matrix_ptr as *const u8,
        matrix_alloc,
        input_ptr as *const u8,
        input_alloc,
        output_ptr as *mut u8,
        output_alloc,
        timeout_ms,
    )
}

#[cfg(all(test, feature = "intel"))]
static TMATMUL_EMULATOR_FALLBACK_TEST_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(all(test, feature = "intel"))]
static TMATMUL_NAMED_FALLBACK_TEST_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(all(test, feature = "intel"))]
static TMATMUL_CXL_SUBMIT_TEST_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(all(test, feature = "intel"))]
static TMATMUL_BRIDGE_INVOKE_TEST_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "intel")]
fn note_tmatmul_emulator_fallback_for_test() {
    #[cfg(test)]
    TMATMUL_EMULATOR_FALLBACK_TEST_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(feature = "intel")]
fn note_tmatmul_named_fallback_for_test() {
    #[cfg(test)]
    TMATMUL_NAMED_FALLBACK_TEST_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(feature = "intel")]
#[allow(dead_code)]
unsafe fn execute_tmatmul_hardware_matmul_fallback(
    kernel_name: &str,
    name_lower: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    strict_cxl_submit_failure: bool,
) -> KernelNameFallbackStatus {
    let Some(layout) = tmatmul_hardware_matmul_layout(name_lower) else {
        eprintln!(
            "[TMatmul HW Matmul] Kernel '{}' is not a supported matmul layout, skipping",
            kernel_name
        );
        return if strict_cxl_submit_failure {
            KernelNameFallbackStatus::Rejected
        } else {
            KernelNameFallbackStatus::ContinueNative
        };
    };

    let asm_path = std::env::var("HETGPU_TMATMUL_ASM_PATH")
        .unwrap_or_else(|_| "/tmp/tmatmul_kernel.S".to_string());
    let assembly = tmatmul_generate_hardware_matmul_assembly(kernel_name, layout);
    if let Err(err) = std::fs::write(&asm_path, assembly.as_bytes()) {
        eprintln!(
            "[TMatmul HW Matmul] Failed to write assembly '{}': {}",
            asm_path, err
        );
        return if strict_cxl_submit_failure {
            KernelNameFallbackStatus::Rejected
        } else {
            KernelNameFallbackStatus::ContinueNative
        };
    }

    let numel = tmatmul_env_usize("HETGPU_TMATMUL_NUMEL").unwrap_or(2048);
    eprintln!(
        "[TMatmul HW Matmul] Launching '{}' as {} via {} (numel={}, pointer_params={})",
        kernel_name, layout.name, asm_path, numel, layout.pointer_params
    );

    if super::cxl_tmatmul::cxl_tmatmul_enabled() && !tmatmul_cxl_matmul_demoted() {
        match submit_cxl_hardware_matmul_fallback(kernel_name, layout, &assembly, kernel_params) {
            Ok(status) => {
                tmatmul_note_cxl_matmul_success();
                eprintln!(
                    "[CXL TMatmul] Kernel '{}' executed via RUN_CSR_ONLY: {:?}",
                    kernel_name, status
                );
                return KernelNameFallbackStatus::Handled;
            }
            Err(err) => {
                eprintln!(
                    "[CXL TMatmul] Kernel '{}' RUN_CSR_ONLY submit failed: {}",
                    kernel_name, err
                );
                if strict_cxl_submit_failure {
                    return KernelNameFallbackStatus::Rejected;
                }
                tmatmul_note_cxl_matmul_failure(kernel_name);
                eprintln!(
                    "[CXL TMatmul] Falling back to TMatmul emulator for '{}'",
                    kernel_name
                );
            }
        }
    } else if super::cxl_tmatmul::cxl_tmatmul_enabled() {
        eprintln!(
            "[CXL TMatmul] Skipping CXL submit for '{}' after prior matmul failure limit",
            kernel_name
        );
    }

    note_tmatmul_emulator_fallback_for_test();
    if tmatmul_bridge_demoted() {
        eprintln!(
            "[TMatmul Emulator] Skipping bridge for '{}' after prior bridge failure limit",
            kernel_name
        );
        return KernelNameFallbackStatus::Handled;
    }
    invoke_emulator_bridge(
        &asm_path,
        kernel_params,
        numel,
        kernel_name,
        layout.pointer_params,
    );
    KernelNameFallbackStatus::Handled
}

#[cfg(all(test, feature = "intel"))]
mod tmatmul_hardware_matmul_tests {
    use super::*;
    use std::sync::Mutex;

    static FALLBACK_TEST_MUTEX: Mutex<()> = Mutex::new(());

    struct EnvGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        previous: Vec<(&'static str, Option<String>)>,
    }

    impl EnvGuard {
        fn set(vars: &[(&'static str, Option<&str>)]) -> Self {
            let lock = super::super::test_env::lock();
            let previous = vars
                .iter()
                .map(|(name, _)| (*name, std::env::var(name).ok()))
                .collect::<Vec<_>>();
            for (name, value) in vars {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
            Self {
                _lock: lock,
                previous,
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (name, value) in self.previous.drain(..) {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }

    struct VirtualAllocGuard {
        keys: Vec<usize>,
    }

    impl VirtualAllocGuard {
        fn insert(entries: &[(usize, usize)]) -> Self {
            let mut map = super::super::memory::VIRTUAL_ALLOC_MAP.lock().unwrap();
            for &(base, size) in entries {
                map.insert(base, size);
            }
            Self {
                keys: entries.iter().map(|&(base, _)| base).collect(),
            }
        }
    }

    impl Drop for VirtualAllocGuard {
        fn drop(&mut self) {
            let mut map = super::super::memory::VIRTUAL_ALLOC_MAP.lock().unwrap();
            for key in self.keys.drain(..) {
                map.remove(&key);
            }
        }
    }

    const SIMPLE_PTX: &str = r#"
.version 7.0
.target sm_80
.address_size 64

.visible .entry simple_tmatmul_jit(
    .param .u64 input,
    .param .u64 output
) {
    .reg .f32 %f<4>;
    .reg .u64 %rd<3>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.global.f32 %f0, [%rd1];
    mul.f32 %f1, %f0, %f0;
    st.global.f32 [%rd2], %f1;

    ret;
}
"#;

    #[test]
    fn hardware_matmul_env_gate_accepts_truthy_values() {
        let _guard = EnvGuard::set(&[("HETGPU_TMATMUL_HARDWARE_MATMUL", Some("1"))]);
        assert!(tmatmul_hardware_matmul_enabled());
    }

    #[test]
    fn matmul_name_filter_skips_stream_k_fixup() {
        assert!(tmatmul_is_matmul_kernel_name("_z9mul_mat_q"));
        assert!(tmatmul_is_matmul_kernel_name("_z13mul_mat_vec_q"));
        assert!(!tmatmul_is_matmul_kernel_name(
            "_z26mul_mat_q_stream_k_fixup"
        ));
    }

    #[test]
    fn ggml_iq1s_mul_mat_q_layout_uses_param2_output() {
        let layout = tmatmul_hardware_matmul_layout("_z9mul_mat_qil10ggml_type41eevpkc...iq1_s")
            .expect("mul_mat_q should have a hardware fallback layout");
        assert_eq!(layout.name, "mul_mat_q");
        assert_eq!(layout.matrix_param, 0);
        assert_eq!(layout.vector_param, 1);
        assert_eq!(layout.output_param, 2);
        assert_eq!(layout.pointer_params, 4);

        let asm = tmatmul_generate_hardware_matmul_assembly("kernel", layout);
        assert!(asm.contains("ldv v0,PARAM_1"));
        assert!(asm.contains("tmatmul_go PARAM_0"));
        assert!(asm.contains("sv v1,PARAM_2"));
    }

    #[test]
    fn ggml_mul_mat_vec_q_layout_uses_param2_output() {
        let layout =
            tmatmul_hardware_matmul_layout("_Z13mul_mat_vec_qIL9ggml_type12ELi1EEvPKvS2_Pfiiii")
                .expect("mul_mat_vec_q should have a hardware fallback layout");
        assert_eq!(layout.name, "mul_mat_vec_q");
        assert_eq!(layout.matrix_param, 0);
        assert_eq!(layout.vector_param, 1);
        assert_eq!(layout.output_param, 2);
        assert_eq!(layout.pointer_params, 3);

        let asm = tmatmul_generate_hardware_matmul_assembly("kernel", layout);
        assert!(asm.contains("ldv v0,PARAM_1"));
        assert!(asm.contains("tmatmul_go PARAM_0"));
        assert!(asm.contains("sv v1,PARAM_2"));
    }

    #[test]
    fn hardware_matmul_param1_accepts_readable_untracked_vector_only() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[("HETGPU_TMATMUL_NUMEL", Some("2048"))]);
        let vector = vec![0x5au8; 4096];
        let output = vec![0xa5u8; 4096];
        let mut vector_param = vector.as_ptr() as u64;
        let mut output_param = output.as_ptr() as u64;
        let mut kernel_params = [
            std::ptr::null_mut(),
            (&mut vector_param as *mut u64).cast::<::core::ffi::c_void>(),
            (&mut output_param as *mut u64).cast::<::core::ffi::c_void>(),
        ];

        let vector_resolved = unsafe {
            tmatmul_read_device_pointer_param(
                kernel_params.as_mut_ptr(),
                1,
                "_Z13mul_mat_vec_qIL9ggml_type20ELi1EEvPKvS2_Pfiiii",
                false,
            )
        }
        .expect("readable host vector pointer should be accepted");
        assert_eq!(vector_resolved, (vector.as_ptr() as usize, vector.len()));

        let output_err = unsafe {
            tmatmul_read_device_pointer_param(
                kernel_params.as_mut_ptr(),
                2,
                "_Z13mul_mat_vec_qIL9ggml_type20ELi1EEvPKvS2_Pfiiii",
                false,
            )
        }
        .expect_err("untracked output pointer must remain rejected");
        assert!(output_err.to_string().contains("PARAM_2"));
        assert!(output_err.to_string().contains("not a tracked"));
    }

    fn cxl_submit_failure_env<'a>(asm_path: &'a str) -> [(&'static str, Option<&'a str>); 6] {
        [
            ("HETGPU_CXL_TMATMUL", Some("1")),
            ("HETGPU_TMATMUL_CXL", None),
            ("HETGPU_TMATMUL_ASM_PATH", Some(asm_path)),
            ("HETGPU_TMATMUL_NUMEL", Some("1")),
            ("HETGPU_CXL_TMATMUL_DEV", None),
            ("HETGPU_CXL_TMATMUL_DEVICE", None),
        ]
    }

    #[test]
    fn cxl_submit_failure_falls_back_to_emulator_by_default() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let asm_path = dir.path().join("kernel.S");
        let asm_text = asm_path.to_string_lossy().to_string();
        let _guard = EnvGuard::set(&cxl_submit_failure_env(&asm_text));
        let mut kernel_params = [std::ptr::null_mut(); 3];
        let before = TMATMUL_EMULATOR_FALLBACK_TEST_COUNT.load(std::sync::atomic::Ordering::SeqCst);

        unsafe {
            execute_tmatmul_hardware_matmul_fallback(
                "layer_0_ffn_gate_mul_mat",
                "layer_0_ffn_gate_mul_mat",
                kernel_params.as_mut_ptr(),
                false,
            );
        }

        let after = TMATMUL_EMULATOR_FALLBACK_TEST_COUNT.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(after, before + 1);
    }

    #[test]
    fn strict_cxl_submit_failure_skips_emulator_fallback() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let asm_path = dir.path().join("kernel.S");
        let asm_text = asm_path.to_string_lossy().to_string();
        let _guard = EnvGuard::set(&cxl_submit_failure_env(&asm_text));
        let mut kernel_params = [std::ptr::null_mut(); 3];
        let before = TMATMUL_EMULATOR_FALLBACK_TEST_COUNT.load(std::sync::atomic::Ordering::SeqCst);

        unsafe {
            execute_tmatmul_hardware_matmul_fallback(
                "layer_0_ffn_gate_mul_mat",
                "layer_0_ffn_gate_mul_mat",
                kernel_params.as_mut_ptr(),
                true,
            );
        }

        let after = TMATMUL_EMULATOR_FALLBACK_TEST_COUNT.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(after, before);
    }

    #[test]
    fn cxl_submit_failure_limit_demotes_later_matmul_launches() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let asm_path = dir.path().join("kernel.S");
        let asm_text = asm_path.to_string_lossy().to_string();
        let mut vars = cxl_submit_failure_env(&asm_text).to_vec();
        vars.push(("HETGPU_CXL_TMATMUL_MAX_FAILURES", Some("1")));
        let _guard = EnvGuard::set(&vars);
        let mut kernel_params = [std::ptr::null_mut(); 3];
        let before = TMATMUL_CXL_SUBMIT_TEST_COUNT.load(std::sync::atomic::Ordering::SeqCst);

        unsafe {
            execute_tmatmul_hardware_matmul_fallback(
                "layer_0_ffn_gate_mul_mat",
                "layer_0_ffn_gate_mul_mat",
                kernel_params.as_mut_ptr(),
                false,
            );
            execute_tmatmul_hardware_matmul_fallback(
                "layer_1_ffn_gate_mul_mat",
                "layer_1_ffn_gate_mul_mat",
                kernel_params.as_mut_ptr(),
                false,
            );
        }

        let after = TMATMUL_CXL_SUBMIT_TEST_COUNT.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(after - before, 1);
    }

    #[test]
    fn bridge_failure_limit_skips_later_bridge_invocations() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let asm_path = dir.path().join("kernel.S");
        let asm_text = asm_path.to_string_lossy().to_string();
        let _guard = EnvGuard::set(&[
            ("HETGPU_CXL_TMATMUL", None),
            ("HETGPU_TMATMUL_CXL", None),
            ("HETGPU_TMATMUL_ASM_PATH", Some(&asm_text)),
            ("HETGPU_TMATMUL_NUMEL", Some("1")),
            ("HETGPU_PYTHON", Some("/bin/false")),
            ("HETGPU_TMATMUL_BRIDGE", Some("/bin/false")),
            ("HETGPU_TMATMUL_BRIDGE_MAX_FAILURES", Some("1")),
        ]);

        let mut matrix = vec![1.0f32; 1];
        let mut vector = vec![2.0f32; 1];
        let mut output = vec![0.0f32; 1];
        let mut matrix_ptr = matrix.as_mut_ptr() as u64;
        let mut vector_ptr = vector.as_mut_ptr() as u64;
        let mut output_ptr = output.as_mut_ptr() as u64;
        let _alloc_guard = VirtualAllocGuard::insert(&[
            (
                matrix.as_ptr() as usize,
                matrix.len() * std::mem::size_of::<f32>(),
            ),
            (
                output.as_ptr() as usize,
                output.len() * std::mem::size_of::<f32>(),
            ),
        ]);
        let mut kernel_params = [
            &mut matrix_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut vector_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut output_ptr as *mut _ as *mut ::core::ffi::c_void,
        ];
        let before = TMATMUL_BRIDGE_INVOKE_TEST_COUNT.load(std::sync::atomic::Ordering::SeqCst);

        unsafe {
            execute_tmatmul_hardware_matmul_fallback(
                "layer_0_ffn_gate_mul_mat",
                "layer_0_ffn_gate_mul_mat",
                kernel_params.as_mut_ptr(),
                false,
            );
            execute_tmatmul_hardware_matmul_fallback(
                "layer_1_ffn_gate_mul_mat",
                "layer_1_ffn_gate_mul_mat",
                kernel_params.as_mut_ptr(),
                false,
            );
        }

        let after = TMATMUL_BRIDGE_INVOKE_TEST_COUNT.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(after - before, 1);
    }

    #[test]
    fn bitnet_fallback_route_continues_to_named_fallback() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", Some("1")),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
            ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
            ("HETGPU_BITNET_FFN_CXL", None),
            ("HETGPU_TMATMUL_BITNET_DISAGGREGATE", None),
            ("HETGPU_BITNET_DISAGG_STRICT", None),
            ("HETGPU_BITNET_CXL_KERNELS", None),
            ("HETGPU_BITNET_GPU_KERNELS", None),
            ("HETGPU_BITNET_ROUTE_MANIFEST", None),
            ("HETGPU_BITNET_ROUTE_LOG", None),
        ]);
        let mut kernel_params = [std::ptr::null_mut(); 16];
        let before = TMATMUL_NAMED_FALLBACK_TEST_COUNT.load(std::sync::atomic::Ordering::SeqCst);

        unsafe {
            execute_kernel_name_fallback(
                "unknown_matmul_probe",
                kernel_params.as_mut_ptr(),
                1,
                1,
                1,
                1,
                1,
                1,
            );
        }

        let after = TMATMUL_NAMED_FALLBACK_TEST_COUNT.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(after, before + 1);
    }

    #[test]
    fn bitnet_gpu_route_requests_native_continuation() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", Some("1")),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
            ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
            ("HETGPU_BITNET_FFN_CXL", None),
            ("HETGPU_TMATMUL_BITNET_DISAGGREGATE", None),
            ("HETGPU_BITNET_DISAGG_STRICT", None),
            ("HETGPU_BITNET_CXL_KERNELS", None),
            ("HETGPU_BITNET_GPU_KERNELS", None),
            ("HETGPU_BITNET_ROUTE_MANIFEST", None),
            ("HETGPU_BITNET_ROUTE_LOG", None),
        ]);
        let mut kernel_params = [std::ptr::null_mut(); 16];
        let before = TMATMUL_NAMED_FALLBACK_TEST_COUNT.load(std::sync::atomic::Ordering::SeqCst);

        let status = unsafe {
            execute_kernel_name_fallback(
                "_z13flash_attn_mul_mat_q",
                kernel_params.as_mut_ptr(),
                1,
                1,
                1,
                1,
                1,
                1,
            )
        };

        assert_eq!(status, KernelNameFallbackStatus::ContinueNative);
        let after = TMATMUL_NAMED_FALLBACK_TEST_COUNT.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(after, before);
    }

    #[test]
    fn bitnet_gpu_non_matmul_route_requests_native_continuation() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", Some("1")),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
            ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
            ("HETGPU_BITNET_FFN_CXL", None),
            ("HETGPU_TMATMUL_BITNET_DISAGGREGATE", None),
            ("HETGPU_BITNET_DISAGG_STRICT", None),
            ("HETGPU_BITNET_CXL_KERNELS", None),
            ("HETGPU_BITNET_GPU_KERNELS", None),
            ("HETGPU_BITNET_ROUTE_MANIFEST", None),
            ("HETGPU_BITNET_ROUTE_LOG", None),
        ]);
        let mut kernel_params = [std::ptr::null_mut(); 16];
        let before = TMATMUL_NAMED_FALLBACK_TEST_COUNT.load(std::sync::atomic::Ordering::SeqCst);

        let status = unsafe {
            execute_kernel_name_fallback(
                "_Z12soft_max_f32ILb1ELi32ELi32EfEvPKfPKT2_Pfiiffffj",
                kernel_params.as_mut_ptr(),
                1,
                1,
                1,
                1,
                1,
                1,
            )
        };

        assert_eq!(status, KernelNameFallbackStatus::ContinueNative);
        let after = TMATMUL_NAMED_FALLBACK_TEST_COUNT.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(after, before);
    }

    #[test]
    fn ggml_softmax_f32_fallback_uses_named_param_layout() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", None),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
        ]);

        let mut input = vec![1.0f32, 2.0, 3.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0f32; input.len()];
        let mut x = input.as_mut_ptr() as u64;
        let mut mask = 0u64;
        let mut sinks = 0u64;
        let mut dst = output.as_mut_ptr() as u64;
        let mut params = TmatmulSoftMaxParams {
            nheads: 1,
            n_head_log2: 0,
            _pad0: 0,
            ncols: 3,
            nrows_x: 2,
            nrows_y: 2,
            ne00: 3,
            ne01: 2,
            ne02: 1,
            ne03: 1,
            nb11: 0,
            nb12: 0,
            nb13: 0,
            ne12: 1,
            ne13: 1,
            scale: 1.0,
            max_bias: 0.0,
            m0: 0.0,
            m1: 0.0,
        };
        let _alloc_guard = VirtualAllocGuard::insert(&[
            (
                input.as_ptr() as usize,
                input.len() * std::mem::size_of::<f32>(),
            ),
            (
                output.as_ptr() as usize,
                output.len() * std::mem::size_of::<f32>(),
            ),
        ]);
        let mut kernel_params = [
            &mut x as *mut _ as *mut ::core::ffi::c_void,
            &mut mask as *mut _ as *mut ::core::ffi::c_void,
            &mut sinks as *mut _ as *mut ::core::ffi::c_void,
            &mut dst as *mut _ as *mut ::core::ffi::c_void,
            &mut params as *mut _ as *mut ::core::ffi::c_void,
        ];

        unsafe {
            execute_softmax_kernel_fallback(
                "_Z12soft_max_f32ILb1ELi32ELi32EfEvPKfPKT2_Pfiiffffj",
                "soft_max_f32",
                kernel_params.as_mut_ptr(),
            );
        }

        assert!(output[2] > output[1]);
        assert!(output[1] > output[0]);
        assert!((output[0] + output[1] + output[2] - 1.0).abs() < 1e-5);
        assert!((output[3] - 1.0 / 3.0).abs() < 1e-5);
        assert!((output[4] - 1.0 / 3.0).abs() < 1e-5);
        assert!((output[5] - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn ggml_bin_bcast_add_f32_fallback_handles_broadcast_strides() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", None),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
        ]);

        let mut src0 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut src1 = vec![10.0f32, 20.0, 30.0];
        let mut dst = vec![0.0f32; src0.len()];
        let mut p0 = src0.as_mut_ptr() as u64;
        let mut p1 = src1.as_mut_ptr() as u64;
        let mut p2 = dst.as_mut_ptr() as u64;
        let mut ne0 = 3i32;
        let mut ne1 = 2i32;
        let mut ne2 = 1i32;
        let mut ne3 = 1i32;
        let mut ne10 = 3i32;
        let mut ne11 = 1i32;
        let mut ne12 = 1i32;
        let mut ne13 = 1i32;
        let mut s1 = 3i32;
        let mut s2 = 6i32;
        let mut s3 = 6i32;
        let mut s01 = 3i32;
        let mut s02 = 6i32;
        let mut s03 = 6i32;
        let mut s11 = 0i32;
        let mut s12 = 0i32;
        let mut s13 = 0i32;
        let _alloc_guard = VirtualAllocGuard::insert(&[
            (
                src0.as_ptr() as usize,
                src0.len() * std::mem::size_of::<f32>(),
            ),
            (
                src1.as_ptr() as usize,
                src1.len() * std::mem::size_of::<f32>(),
            ),
            (
                dst.as_ptr() as usize,
                dst.len() * std::mem::size_of::<f32>(),
            ),
        ]);
        let mut kernel_params = [
            &mut p0 as *mut _ as *mut ::core::ffi::c_void,
            &mut p1 as *mut _ as *mut ::core::ffi::c_void,
            &mut p2 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne0 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne1 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne2 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne3 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne10 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne11 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne12 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne13 as *mut _ as *mut ::core::ffi::c_void,
            &mut s1 as *mut _ as *mut ::core::ffi::c_void,
            &mut s2 as *mut _ as *mut ::core::ffi::c_void,
            &mut s3 as *mut _ as *mut ::core::ffi::c_void,
            &mut s01 as *mut _ as *mut ::core::ffi::c_void,
            &mut s02 as *mut _ as *mut ::core::ffi::c_void,
            &mut s03 as *mut _ as *mut ::core::ffi::c_void,
            &mut s11 as *mut _ as *mut ::core::ffi::c_void,
            &mut s12 as *mut _ as *mut ::core::ffi::c_void,
            &mut s13 as *mut _ as *mut ::core::ffi::c_void,
        ];

        let handled = unsafe {
            execute_tmatmul_bin_bcast_f32_fallback(
                "_Z11k_bin_bcastIXadL_ZN42_INTERNAL_8e457c15_11_binbcast_cu_6840010b6op_addEffEEfffEvPKT0_PKT1_PT2_iiiiiiiiiiiiiiiii",
                kernel_params.as_mut_ptr(),
            )
        };

        assert!(handled);
        assert_eq!(dst, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn ggml_rms_norm_f32_fallback_uses_named_param_layout() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", None),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
            ("HETGPU_BITNET_DISAGGREGATE", None),
        ]);

        let mut input = vec![3.0f32, 4.0, 0.0, 0.0];
        let mut output = vec![0.0f32; input.len()];
        let mut input_ptr = input.as_mut_ptr() as u64;
        let mut output_ptr = output.as_mut_ptr() as u64;
        let mut ncols = 2i32;
        let mut eps = 1e-5f32;
        let _alloc_guard = VirtualAllocGuard::insert(&[
            (
                input.as_ptr() as usize,
                input.len() * std::mem::size_of::<f32>(),
            ),
            (
                output.as_ptr() as usize,
                output.len() * std::mem::size_of::<f32>(),
            ),
        ]);
        let mut kernel_params = [
            &mut input_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut output_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut ncols as *mut _ as *mut ::core::ffi::c_void,
            &mut eps as *mut _ as *mut ::core::ffi::c_void,
        ];

        let status = unsafe {
            execute_kernel_name_fallback(
                "_Z12rms_norm_f32ILi1024EEvPKfPfif",
                kernel_params.as_mut_ptr(),
                2,
                1,
                1,
                1024,
                1,
                1,
            )
        };

        assert_eq!(status, KernelNameFallbackStatus::Handled);
        let scale = (12.5f32 + eps).sqrt();
        assert!((output[0] - 3.0 / scale).abs() < 1e-5);
        assert!((output[1] - 4.0 / scale).abs() < 1e-5);
        assert_eq!(output[2], 0.0);
        assert_eq!(output[3], 0.0);
    }

    #[test]
    fn ggml_quantize_mmq_q8_1_ds4_fallback_writes_mmq_layout() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", None),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
            ("HETGPU_BITNET_DISAGGREGATE", None),
        ]);

        let mut input = vec![1.0f32; 128];
        input.extend(std::iter::repeat(2.0f32).take(128));
        let mut output = vec![0u8; 2 * 144];
        let mut input_ptr = input.as_mut_ptr() as u64;
        let mut output_ptr = output.as_mut_ptr() as u64;
        let mut kx0 = 128i64;
        let mut kx1 = 2i64;
        let mut kx0_padded = 128i64;
        let _alloc_guard = VirtualAllocGuard::insert(&[
            (
                input.as_ptr() as usize,
                input.len() * std::mem::size_of::<f32>(),
            ),
            (output.as_ptr() as usize, output.len()),
        ]);
        let mut kernel_params = [
            &mut input_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut output_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut kx0 as *mut _ as *mut ::core::ffi::c_void,
            &mut kx1 as *mut _ as *mut ::core::ffi::c_void,
            &mut kx0_padded as *mut _ as *mut ::core::ffi::c_void,
        ];

        let handled = unsafe {
            execute_tmatmul_quantize_mmq_q8_1_fallback(
                "_Z17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout1EEvPKfPvlll",
                kernel_params.as_mut_ptr(),
                1,
                2,
                1,
            )
        };

        assert!(handled);
        let row0_d0 = f16_to_f32(u16::from_le_bytes([output[0], output[1]]));
        let row0_sum0 = f16_to_f32(u16::from_le_bytes([output[2], output[3]]));
        let row1_d0 = f16_to_f32(u16::from_le_bytes([output[144], output[145]]));
        let row1_sum0 = f16_to_f32(u16::from_le_bytes([output[146], output[147]]));
        assert!((row0_d0 - 1.0 / 127.0).abs() < 1e-5);
        assert!((row0_sum0 - 32.0).abs() < 1e-3);
        assert!((row1_d0 - 2.0 / 127.0).abs() < 1e-5);
        assert!((row1_sum0 - 64.0).abs() < 1e-3);
        assert!(output[16..144].iter().all(|&v| v as i8 == 127));
        assert!(output[160..288].iter().all(|&v| v as i8 == 127));
    }

    #[test]
    fn ggml_quantize_q8_1_fallback_writes_block_q8_1_layout() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", None),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
            ("HETGPU_BITNET_DISAGGREGATE", None),
        ]);

        let mut input = (1..=32).map(|v| v as f32).collect::<Vec<_>>();
        let mut output = vec![0u8; 36];
        let mut input_ptr = input.as_mut_ptr() as u64;
        let mut output_ptr = output.as_mut_ptr() as u64;
        let mut kx = 32i64;
        let mut kx0_padded = 32i64;
        let _alloc_guard = VirtualAllocGuard::insert(&[
            (
                input.as_ptr() as usize,
                input.len() * std::mem::size_of::<f32>(),
            ),
            (output.as_ptr() as usize, output.len()),
        ]);
        let mut kernel_params = [
            &mut input_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut output_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut kx as *mut _ as *mut ::core::ffi::c_void,
            &mut kx0_padded as *mut _ as *mut ::core::ffi::c_void,
        ];

        let handled = unsafe {
            execute_tmatmul_quantize_q8_1_fallback(
                "_Z13quantize_q8_1PKfPvll",
                kernel_params.as_mut_ptr(),
                1,
            )
        };

        assert!(handled);
        let d = f16_to_f32(u16::from_le_bytes([output[32], output[33]]));
        let sum = f16_to_f32(u16::from_le_bytes([output[34], output[35]]));
        assert!((d - (32.0 / 127.0)).abs() < 1e-4);
        assert!((sum - 528.0).abs() < 1e-2);
        assert_eq!(i8::from_ne_bytes([output[31]]), 127);
    }

    #[test]
    fn ggml_cpy_f32_f16_f32_to_f32_fallback_copies_contiguous_values() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", None),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
            ("HETGPU_BITNET_DISAGGREGATE", None),
        ]);

        let mut input = vec![1.25f32, -2.5, 3.75, 4.5];
        let mut output = vec![0.0f32; input.len()];
        let mut src = input.as_mut_ptr() as u64;
        let mut dst = output.as_mut_ptr() as u64;
        let mut ne = input.len() as i32;
        let mut ne00 = input.len() as i32;
        let mut ne01 = 1i32;
        let mut ne02 = 1i32;
        let mut nb00 = std::mem::size_of::<f32>() as i32;
        let mut nb01 = (input.len() * std::mem::size_of::<f32>()) as i32;
        let mut nb02 = nb01;
        let mut nb03 = nb01;
        let mut ne10 = ne00;
        let mut ne11 = 1i32;
        let mut ne12 = 1i32;
        let mut nb10 = nb00;
        let mut nb11 = nb01;
        let mut nb12 = nb01;
        let mut nb13 = nb01;
        let _alloc_guard = VirtualAllocGuard::insert(&[
            (
                input.as_ptr() as usize,
                input.len() * std::mem::size_of::<f32>(),
            ),
            (
                output.as_ptr() as usize,
                output.len() * std::mem::size_of::<f32>(),
            ),
        ]);
        let mut kernel_params = [
            &mut src as *mut _ as *mut ::core::ffi::c_void,
            &mut dst as *mut _ as *mut ::core::ffi::c_void,
            &mut ne as *mut _ as *mut ::core::ffi::c_void,
            &mut ne00 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne01 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne02 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb00 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb01 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb02 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb03 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne10 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne11 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne12 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb10 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb11 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb12 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb13 as *mut _ as *mut ::core::ffi::c_void,
        ];

        let handled = unsafe {
            execute_tmatmul_cpy_f32_f16_fallback(
                "_Z11cpy_f32_f16IXadL_ZN36_INTERNAL_a6e895be_6_cpy_cu_bfba63e213cpy_1_f32_f32EPKcPcEEEvS2_S3_iiiiiiiiiiiiiii",
                kernel_params.as_mut_ptr(),
            )
        };

        assert!(handled);
        assert_eq!(output, input);
    }

    #[test]
    fn ggml_rope_norm_f32_fallback_rotates_rows() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", None),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
            ("HETGPU_BITNET_DISAGGREGATE", None),
        ]);

        let mut input = vec![1.0f32, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; input.len()];
        let mut pos = vec![0i32, 1i32];
        let mut corr_dims = [0.0f32, 0.0f32];
        let mut src = input.as_mut_ptr() as u64;
        let mut dst = output.as_mut_ptr() as u64;
        let mut ne0 = 4i32;
        let mut n_dims = 4i32;
        let mut pos_ptr = pos.as_mut_ptr() as u64;
        let mut freq_scale = 1.0f32;
        let mut p_delta_rows = 1i32;
        let mut ext_factor = 0.0f32;
        let mut attn_factor = 1.0f32;
        let mut theta_scale = 1.0f32;
        let mut freq_factors = 0u64;
        let _alloc_guard = VirtualAllocGuard::insert(&[
            (
                input.as_ptr() as usize,
                input.len() * std::mem::size_of::<f32>(),
            ),
            (
                output.as_ptr() as usize,
                output.len() * std::mem::size_of::<f32>(),
            ),
            (
                pos.as_ptr() as usize,
                pos.len() * std::mem::size_of::<i32>(),
            ),
        ]);
        let mut kernel_params = [
            &mut src as *mut _ as *mut ::core::ffi::c_void,
            &mut dst as *mut _ as *mut ::core::ffi::c_void,
            &mut ne0 as *mut _ as *mut ::core::ffi::c_void,
            &mut n_dims as *mut _ as *mut ::core::ffi::c_void,
            &mut pos_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut freq_scale as *mut _ as *mut ::core::ffi::c_void,
            &mut p_delta_rows as *mut _ as *mut ::core::ffi::c_void,
            &mut ext_factor as *mut _ as *mut ::core::ffi::c_void,
            &mut attn_factor as *mut _ as *mut ::core::ffi::c_void,
            corr_dims.as_mut_ptr() as *mut ::core::ffi::c_void,
            &mut theta_scale as *mut _ as *mut ::core::ffi::c_void,
            &mut freq_factors as *mut _ as *mut ::core::ffi::c_void,
        ];

        let status = unsafe {
            execute_kernel_name_fallback(
                "_Z9rope_normIfLb0EEvPKT_PS0_iiPKififf14rope_corr_dimsfPKf",
                kernel_params.as_mut_ptr(),
                2,
                1,
                1,
                1,
                256,
                1,
            )
        };

        assert_eq!(status, KernelNameFallbackStatus::Handled);
        assert!((output[0] - 1.0).abs() < 1e-6);
        assert!((output[1] - 0.0).abs() < 1e-6);
        assert!((output[2] - 0.0).abs() < 1e-6);
        assert!((output[3] - 1.0).abs() < 1e-6);

        let cos_theta = 1.0f32.cos();
        let sin_theta = 1.0f32.sin();
        assert!((output[4] - (2.0 * cos_theta - 3.0 * sin_theta)).abs() < 1e-5);
        assert!((output[5] - (2.0 * sin_theta + 3.0 * cos_theta)).abs() < 1e-5);
        assert!((output[6] - (4.0 * cos_theta - 5.0 * sin_theta)).abs() < 1e-5);
        assert!((output[7] - (4.0 * sin_theta + 5.0 * cos_theta)).abs() < 1e-5);
    }

    #[test]
    fn ggml_convert_unary_f32_f16_fallback_roundtrips_contiguous_values() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", None),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
            ("HETGPU_BITNET_DISAGGREGATE", None),
        ]);

        let mut input = vec![1.25f32, -2.5, 0.0, 3.5];
        let mut halfs = vec![0u16; input.len()];
        let mut output = vec![0.0f32; input.len()];
        let mut src = input.as_mut_ptr() as u64;
        let mut half_dst = halfs.as_mut_ptr() as u64;
        let mut ne = input.len() as i64;
        let _alloc_guard = VirtualAllocGuard::insert(&[
            (
                input.as_ptr() as usize,
                input.len() * std::mem::size_of::<f32>(),
            ),
            (
                halfs.as_ptr() as usize,
                halfs.len() * std::mem::size_of::<u16>(),
            ),
            (
                output.as_ptr() as usize,
                output.len() * std::mem::size_of::<f32>(),
            ),
        ]);
        let mut to_half_params = [
            &mut src as *mut _ as *mut ::core::ffi::c_void,
            &mut half_dst as *mut _ as *mut ::core::ffi::c_void,
            &mut ne as *mut _ as *mut ::core::ffi::c_void,
        ];

        let to_half = unsafe {
            execute_kernel_name_fallback(
                "_Z13convert_unaryIf6__halfEvPKvPT0_l",
                to_half_params.as_mut_ptr(),
                1,
                1,
                1,
                256,
                1,
                1,
            )
        };

        assert_eq!(to_half, KernelNameFallbackStatus::Handled);
        assert!(halfs.iter().any(|&value| value != 0));
        let mut half_src = halfs.as_mut_ptr() as u64;
        let mut dst = output.as_mut_ptr() as u64;
        let mut from_half_params = [
            &mut half_src as *mut _ as *mut ::core::ffi::c_void,
            &mut dst as *mut _ as *mut ::core::ffi::c_void,
            &mut ne as *mut _ as *mut ::core::ffi::c_void,
        ];

        let from_half = unsafe {
            execute_kernel_name_fallback(
                "_Z13convert_unaryI6__halffEvPKvPT0_l",
                from_half_params.as_mut_ptr(),
                1,
                1,
                1,
                256,
                1,
                1,
            )
        };

        assert_eq!(from_half, KernelNameFallbackStatus::Handled);
        for (actual, expected) in output.iter().zip(input.iter()) {
            assert!((actual - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn ggml_sigmoid_f32_fallback_matches_cuda_unary_kernel() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", None),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
            ("HETGPU_BITNET_DISAGGREGATE", None),
        ]);

        let mut input = vec![-2.0f32, 0.0, 2.0];
        let mut output = vec![0.0f32; input.len()];
        let mut input_ptr = input.as_mut_ptr() as u64;
        let mut output_ptr = output.as_mut_ptr() as u64;
        let mut k = input.len() as i32;
        let _alloc_guard = VirtualAllocGuard::insert(&[
            (
                input.as_ptr() as usize,
                input.len() * std::mem::size_of::<f32>(),
            ),
            (
                output.as_ptr() as usize,
                output.len() * std::mem::size_of::<f32>(),
            ),
        ]);
        let mut kernel_params = [
            &mut input_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut output_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut k as *mut _ as *mut ::core::ffi::c_void,
        ];

        let status = unsafe {
            execute_kernel_name_fallback(
                "_Z11sigmoid_f32PKfPfi",
                kernel_params.as_mut_ptr(),
                1,
                1,
                1,
                256,
                1,
                1,
            )
        };

        assert_eq!(status, KernelNameFallbackStatus::Handled);
        for (actual, input) in output.iter().zip(input.iter()) {
            let expected = 1.0 / (1.0 + (-*input).exp());
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn ggml_get_rows_float_fallback_gathers_indexed_rows() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", None),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
            ("HETGPU_BITNET_DISAGGREGATE", None),
        ]);

        let mut src0 = vec![
            1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0,
        ];
        let mut src1 = vec![2i32, 0i32];
        let mut dst = vec![0.0f32; 8];
        let mut src0_ptr = src0.as_mut_ptr() as u64;
        let mut src1_ptr = src1.as_mut_ptr() as u64;
        let mut dst_ptr = dst.as_mut_ptr() as u64;
        let mut ne00 = 4i64;
        let mut ne12 = 1i64;
        let mut s1 = 4u64;
        let mut s2 = 8u64;
        let mut s3 = 8u64;
        let mut nb01 = (4 * std::mem::size_of::<f32>()) as u64;
        let mut nb02 = src0.len() as u64 * std::mem::size_of::<f32>() as u64;
        let mut nb03 = nb02;
        let mut s10 = 1u64;
        let mut s11 = src1.len() as u64;
        let mut s12 = src1.len() as u64;
        let _alloc_guard = VirtualAllocGuard::insert(&[
            (
                src0.as_ptr() as usize,
                src0.len() * std::mem::size_of::<f32>(),
            ),
            (
                src1.as_ptr() as usize,
                src1.len() * std::mem::size_of::<i32>(),
            ),
            (
                dst.as_ptr() as usize,
                dst.len() * std::mem::size_of::<f32>(),
            ),
        ]);
        let mut kernel_params = [
            &mut src0_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut src1_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut dst_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut ne00 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne12 as *mut _ as *mut ::core::ffi::c_void,
            &mut s1 as *mut _ as *mut ::core::ffi::c_void,
            &mut s2 as *mut _ as *mut ::core::ffi::c_void,
            &mut s3 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb01 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb02 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb03 as *mut _ as *mut ::core::ffi::c_void,
            &mut s10 as *mut _ as *mut ::core::ffi::c_void,
            &mut s11 as *mut _ as *mut ::core::ffi::c_void,
            &mut s12 as *mut _ as *mut ::core::ffi::c_void,
        ];

        let status = unsafe {
            execute_kernel_name_fallback(
                "_Z16k_get_rows_floatIffEvPKT_PKiPT0_llmmmmmmmmm",
                kernel_params.as_mut_ptr(),
                1,
                2,
                1,
                256,
                1,
                1,
            )
        };

        assert_eq!(status, KernelNameFallbackStatus::Handled);
        assert_eq!(dst, vec![100.0, 200.0, 300.0, 400.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn ggml_concat_f32_dim0_fallback_concatenates_rows() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", None),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
            ("HETGPU_BITNET_DISAGGREGATE", None),
        ]);

        let mut x = vec![1.0f32, 2.0, 10.0, 20.0];
        let mut y = vec![3.0f32, 4.0, 5.0, 30.0, 40.0, 50.0];
        let mut dst = vec![0.0f32; 10];
        let mut x_ptr = x.as_mut_ptr() as u64;
        let mut y_ptr = y.as_mut_ptr() as u64;
        let mut dst_ptr = dst.as_mut_ptr() as u64;
        let mut ne0 = 5i32;
        let mut ne00 = 2i32;
        let _alloc_guard = VirtualAllocGuard::insert(&[
            (x.as_ptr() as usize, x.len() * std::mem::size_of::<f32>()),
            (y.as_ptr() as usize, y.len() * std::mem::size_of::<f32>()),
            (
                dst.as_ptr() as usize,
                dst.len() * std::mem::size_of::<f32>(),
            ),
        ]);
        let mut kernel_params = [
            &mut x_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut y_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut dst_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut ne0 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne00 as *mut _ as *mut ::core::ffi::c_void,
        ];

        let status = unsafe {
            execute_kernel_name_fallback(
                "_Z15concat_f32_dim0PKfS0_Pfii",
                kernel_params.as_mut_ptr(),
                1,
                2,
                1,
                256,
                1,
                1,
            )
        };

        assert_eq!(status, KernelNameFallbackStatus::Handled);
        assert_eq!(
            dst,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        );
    }

    #[test]
    fn pre_jit_named_fallback_handles_concat_without_staging_artifacts() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let artifact_dir = tempfile::tempdir().unwrap();
        let artifact_dir_text = artifact_dir.path().to_string_lossy().to_string();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", None),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
            ("HETGPU_TMATMUL_PRE_JIT_NAMED_FALLBACK", Some("1")),
            ("HETGPU_TMATMUL_INTERPRETER", None),
            ("HETGPU_CXL_TMATMUL", None),
            ("HETGPU_TMATMUL_CXL", None),
            ("HETGPU_TMATMUL_ARTIFACT_DIR", Some(&artifact_dir_text)),
            ("HETGPU_BITNET_DISAGGREGATE", None),
        ]);

        let mut x = vec![1.0f32, 2.0, 10.0, 20.0];
        let mut y = vec![3.0f32, 4.0, 5.0, 30.0, 40.0, 50.0];
        let mut dst = vec![0.0f32; 10];
        let mut x_ptr = x.as_mut_ptr() as u64;
        let mut y_ptr = y.as_mut_ptr() as u64;
        let mut dst_ptr = dst.as_mut_ptr() as u64;
        let mut ne0 = 5i32;
        let mut ne00 = 2i32;
        let _alloc_guard = VirtualAllocGuard::insert(&[
            (x.as_ptr() as usize, x.len() * std::mem::size_of::<f32>()),
            (y.as_ptr() as usize, y.len() * std::mem::size_of::<f32>()),
            (
                dst.as_ptr() as usize,
                dst.len() * std::mem::size_of::<f32>(),
            ),
        ]);
        let mut kernel_params = [
            &mut x_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut y_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut dst_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut ne0 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne00 as *mut _ as *mut ::core::ffi::c_void,
        ];
        let kernel = super::super::module::ZeKernel {
            context: ze_context_handle_t(std::ptr::null_mut()),
            device: ze_device_handle_t(std::ptr::null_mut()),
            module: ze_module_handle_t(std::ptr::null_mut()),
            kernel: ze_kernel_handle_t(std::ptr::null_mut()),
            name: "_Z15concat_f32_dim0PKfS0_Pfii".to_string(),
            ptx_source: Some(std::sync::Arc::new(SIMPLE_PTX.to_string())),
            cubin_binary: None,
            module_handle: 0,
        };

        let result = unsafe {
            launch_kernel(
                &kernel,
                1,
                2,
                1,
                256,
                1,
                1,
                0,
                ze_command_queue_handle_t(std::ptr::null_mut()),
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        assert_eq!(result, ze_result_t::ZE_RESULT_SUCCESS);
        assert_eq!(
            dst,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        );
        assert_eq!(artifact_dir.path().read_dir().unwrap().count(), 0);
    }

    #[test]
    fn ggml_concat_f32_non_cont_fallback_concatenates_dim1() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", None),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
            ("HETGPU_BITNET_DISAGGREGATE", None),
        ]);

        let mut src0 = vec![1.0f32, 2.0];
        let mut src1 = vec![10.0f32, 20.0, 30.0, 40.0];
        let mut dst = vec![0.0f32; 6];
        let mut src0_ptr = src0.as_mut_ptr() as u64;
        let mut src1_ptr = src1.as_mut_ptr() as u64;
        let mut dst_ptr = dst.as_mut_ptr() as u64;
        let mut ne00 = 2i64;
        let mut ne01 = 1i64;
        let mut ne02 = 1i64;
        let mut ne03 = 1i64;
        let mut nb00 = 4u64;
        let mut nb01 = 8u64;
        let mut nb02 = 8u64;
        let mut nb03 = 8u64;
        let mut ne10 = 2i64;
        let mut ne11 = 2i64;
        let mut ne12 = 1i64;
        let mut ne13 = 1i64;
        let mut nb10 = 4u64;
        let mut nb11 = 8u64;
        let mut nb12 = 16u64;
        let mut nb13 = 16u64;
        let mut ne0 = 2i64;
        let mut ne1 = 3i64;
        let mut ne2 = 1i64;
        let mut ne3 = 1i64;
        let mut nb0 = 4u64;
        let mut nb1 = 8u64;
        let mut nb2 = 24u64;
        let mut nb3 = 24u64;
        let mut dim = 1i32;
        let _alloc_guard = VirtualAllocGuard::insert(&[
            (
                src0.as_ptr() as usize,
                src0.len() * std::mem::size_of::<f32>(),
            ),
            (
                src1.as_ptr() as usize,
                src1.len() * std::mem::size_of::<f32>(),
            ),
            (
                dst.as_ptr() as usize,
                dst.len() * std::mem::size_of::<f32>(),
            ),
        ]);
        let mut kernel_params = [
            &mut src0_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut src1_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut dst_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut ne00 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne01 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne02 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne03 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb00 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb01 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb02 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb03 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne10 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne11 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne12 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne13 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb10 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb11 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb12 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb13 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne0 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne1 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne2 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne3 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb0 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb1 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb2 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb3 as *mut _ as *mut ::core::ffi::c_void,
            &mut dim as *mut _ as *mut ::core::ffi::c_void,
        ];

        let status = unsafe {
            execute_kernel_name_fallback(
                "_Z19concat_f32_non_contPKcS0_Pcllllmmmmllllmmmmllllmmmmi",
                kernel_params.as_mut_ptr(),
                3,
                1,
                1,
                256,
                1,
                1,
            )
        };

        assert_eq!(status, KernelNameFallbackStatus::Handled);
        assert_eq!(dst, vec![1.0, 2.0, 10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn emulator_bridge_serializes_matmul_matrix_param_at_square_extent() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_PYTHON", Some("/bin/true")),
            ("HETGPU_TMATMUL_BRIDGE", Some("/bin/true")),
            ("HETGPU_TMATMUL_NUMEL", Some("4")),
        ]);
        let _ = std::fs::remove_file("/tmp/tmatmul_bridge_config.json");

        let mut matrix = vec![1.0f32; 16];
        let mut vector = vec![2.0f32; 4];
        let mut output = vec![0.0f32; 4];
        let mut matrix_ptr = matrix.as_mut_ptr() as u64;
        let mut vector_ptr = vector.as_mut_ptr() as u64;
        let mut output_ptr = output.as_mut_ptr() as u64;
        let _alloc_guard = VirtualAllocGuard::insert(&[
            (
                matrix.as_ptr() as usize,
                matrix.len() * std::mem::size_of::<f32>(),
            ),
            (
                output.as_ptr() as usize,
                output.len() * std::mem::size_of::<f32>(),
            ),
        ]);
        let mut kernel_params = [
            &mut matrix_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut vector_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut output_ptr as *mut _ as *mut ::core::ffi::c_void,
        ];

        unsafe {
            invoke_emulator_bridge(
                "/tmp/tmatmul_kernel.S",
                kernel_params.as_mut_ptr(),
                4,
                "_Z13mul_mat_vec_qIL9ggml_type12ELi1EEvPKvS2_Pfiiii",
                3,
            );
        }

        let config = std::fs::read_to_string("/tmp/tmatmul_bridge_config.json")
            .expect("bridge config should be written");
        assert!(config.contains(r#""numel":4"#), "{config}");
        assert!(
            config.contains(r#""file":"/tmp/hetgpu_param_0.bin","count":16"#),
            "{config}"
        );
        assert!(
            config.contains(r#""file":"/tmp/hetgpu_param_1.bin","count":4"#),
            "{config}"
        );
        assert!(
            config.contains(r#""file":"/tmp/hetgpu_param_2.bin","count":4"#),
            "{config}"
        );
        let _ = std::fs::remove_file("/tmp/tmatmul_bridge_config.json");
    }

    #[test]
    fn emulator_bridge_serializes_cuda_dax_matrix_param_without_host_read() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_PYTHON", Some("/bin/true")),
            ("HETGPU_TMATMUL_BRIDGE", Some("/bin/true")),
            ("HETGPU_TMATMUL_NUMEL", Some("4")),
            ("HETGPU_TMATMUL_MATRIX_STAGE", Some("cuda_dax")),
            ("HETGPU_TMATMUL_MATRIX_CXL_OFFSET", Some("0x400000")),
            ("HETGPU_TMATMUL_CUDA_GPU", Some("2")),
            ("CXL_DAX_PATH", Some("/dev/dax12.0")),
        ]);
        let _ = std::fs::remove_file("/tmp/tmatmul_bridge_config.json");

        let mut matrix_ptr = 0x7f00_0000_0000u64;
        let mut vector = vec![2.0f32; 4];
        let mut output = vec![0.0f32; 4];
        let mut vector_ptr = vector.as_mut_ptr() as u64;
        let mut output_ptr = output.as_mut_ptr() as u64;
        let _alloc_guard = VirtualAllocGuard::insert(&[
            (
                vector.as_ptr() as usize,
                vector.len() * std::mem::size_of::<f32>(),
            ),
            (
                output.as_ptr() as usize,
                output.len() * std::mem::size_of::<f32>(),
            ),
        ]);
        let mut kernel_params = [
            &mut matrix_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut vector_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut output_ptr as *mut _ as *mut ::core::ffi::c_void,
        ];

        unsafe {
            invoke_emulator_bridge(
                "/tmp/tmatmul_kernel.S",
                kernel_params.as_mut_ptr(),
                4,
                "_Z13mul_mat_vec_qIL9ggml_type12ELi1EEvPKvS2_Pfiiii",
                3,
            );
        }

        let config = std::fs::read_to_string("/tmp/tmatmul_bridge_config.json")
            .expect("bridge config should be written");
        assert!(config.contains(r#""stage":"cuda_dax""#), "{config}");
        assert!(
            config.contains(r#""cuda_device_ptr":"0x7f0000000000""#),
            "{config}"
        );
        assert!(config.contains(r#""bytes":4"#), "{config}");
        assert!(config.contains(r#""gpu":2"#), "{config}");
        assert!(config.contains(r#""dax_path":"/dev/dax12.0""#), "{config}");
        assert!(config.contains(r#""cxl_offset":4194304"#), "{config}");
        assert!(
            !config.contains(r#""file":"/tmp/hetgpu_param_0.bin""#),
            "{config}"
        );
        let _ = std::fs::remove_file("/tmp/tmatmul_bridge_config.json");
    }

    #[test]
    fn ggml_mul_mat_q_stream_k_fixup_adds_partial_tile() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", None),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
            ("HETGPU_BITNET_DISAGGREGATE", None),
        ]);

        let mmq_x = 16usize;
        let mmq_y = 128usize;
        let mut dst = vec![5.0f32; mmq_x * mmq_y];
        let mut tmp = vec![0.0f32; 3 * mmq_x * mmq_y];
        for i in 0..(mmq_x * mmq_y) {
            tmp[mmq_x * mmq_y + i] = 1000.0 + i as f32;
        }
        let mut dst_ptr = dst.as_mut_ptr() as u64;
        let mut tmp_ptr = tmp.as_mut_ptr() as u64;
        let mut ne00 = 512i32;
        let mut ne01 = 128i32;
        let mut ne11 = 16i32;
        let mut ne0 = 128i32;
        let mut block_num_mmq = 3i32;
        let _alloc_guard = VirtualAllocGuard::insert(&[
            (
                dst.as_ptr() as usize,
                dst.len() * std::mem::size_of::<f32>(),
            ),
            (
                tmp.as_ptr() as usize,
                tmp.len() * std::mem::size_of::<f32>(),
            ),
        ]);
        let mut kernel_params = [
            &mut dst_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut tmp_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut ne00 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne01 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne11 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne0 as *mut _ as *mut ::core::ffi::c_void,
            &mut block_num_mmq as *mut _ as *mut ::core::ffi::c_void,
        ];

        let status = unsafe {
            execute_kernel_name_fallback(
                "_Z24mul_mat_q_stream_k_fixupIL9ggml_type12ELi16ELi8ELb0EEvPfPKfiiiii",
                kernel_params.as_mut_ptr(),
                1,
                1,
                1,
                32,
                8,
                1,
            )
        };

        assert_eq!(status, KernelNameFallbackStatus::Handled);
        assert_eq!(dst[0], 1005.0);
        assert_eq!(dst[7 * mmq_y + 31], 5.0 + 1000.0 + (7 * mmq_y + 31) as f32);
        assert_eq!(
            dst[15 * mmq_y + 127],
            5.0 + 1000.0 + (15 * mmq_y + 127) as f32
        );
    }

    #[test]
    fn ggml_mul_mat_q_stream_k_fixup_handles_iq4_nl_qk32() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", None),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
            ("HETGPU_BITNET_DISAGGREGATE", None),
        ]);

        let mmq_x = 16usize;
        let mmq_y = 128usize;
        let mut dst = vec![7.0f32; mmq_x * mmq_y];
        let mut tmp = vec![0.0f32; 3 * mmq_x * mmq_y];
        for i in 0..(mmq_x * mmq_y) {
            tmp[mmq_x * mmq_y + i] = 2000.0 + i as f32;
        }
        let mut dst_ptr = dst.as_mut_ptr() as u64;
        let mut tmp_ptr = tmp.as_mut_ptr() as u64;
        let mut ne00 = 64i32;
        let mut ne01 = 128i32;
        let mut ne11 = 16i32;
        let mut ne0 = 128i32;
        let mut block_num_mmq = 3i32;
        let _alloc_guard = VirtualAllocGuard::insert(&[
            (
                dst.as_ptr() as usize,
                dst.len() * std::mem::size_of::<f32>(),
            ),
            (
                tmp.as_ptr() as usize,
                tmp.len() * std::mem::size_of::<f32>(),
            ),
        ]);
        let mut kernel_params = [
            &mut dst_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut tmp_ptr as *mut _ as *mut ::core::ffi::c_void,
            &mut ne00 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne01 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne11 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne0 as *mut _ as *mut ::core::ffi::c_void,
            &mut block_num_mmq as *mut _ as *mut ::core::ffi::c_void,
        ];

        let status = unsafe {
            execute_kernel_name_fallback(
                "_Z24mul_mat_q_stream_k_fixupIL9ggml_type20ELi16ELi8ELb0EEvPfPKfiiiii",
                kernel_params.as_mut_ptr(),
                1,
                1,
                1,
                32,
                8,
                1,
            )
        };

        assert_eq!(status, KernelNameFallbackStatus::Handled);
        assert!(dst.iter().all(|value| *value == 7.0));
    }

    #[test]
    fn bitnet_gpu_route_does_not_succeed_in_virtual_fallback() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", Some("1")),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
            ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
            ("HETGPU_BITNET_FFN_CXL", None),
            ("HETGPU_TMATMUL_BITNET_DISAGGREGATE", None),
            ("HETGPU_BITNET_DISAGG_STRICT", None),
            ("HETGPU_BITNET_CXL_KERNELS", None),
            ("HETGPU_BITNET_GPU_KERNELS", None),
            ("HETGPU_BITNET_ROUTE_MANIFEST", None),
            ("HETGPU_BITNET_ROUTE_LOG", None),
        ]);
        let kernel = super::super::module::ZeKernel {
            context: ze_context_handle_t(std::ptr::null_mut()),
            device: ze_device_handle_t(std::ptr::null_mut()),
            module: ze_module_handle_t(std::ptr::null_mut()),
            kernel: ze_kernel_handle_t(std::ptr::null_mut()),
            name: "_z13flash_attn_mul_mat_q".to_string(),
            ptx_source: None,
            cubin_binary: None,
            module_handle: 0,
        };
        let mut kernel_params = [std::ptr::null_mut(); 16];

        let result = unsafe {
            launch_kernel(
                &kernel,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                ze_command_queue_handle_t(std::ptr::null_mut()),
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        assert_eq!(result, ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
    }

    #[test]
    fn jit_success_still_reaches_named_fallback_before_native_continuation() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", None),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
            ("HETGPU_TMATMUL_INTERPRETER", None),
            ("HETGPU_CXL_TMATMUL", Some("1")),
            ("HETGPU_TMATMUL_CXL", None),
            ("HETGPU_TMATMUL_ARTIFACT_DIR", None),
        ]);
        let kernel = super::super::module::ZeKernel {
            context: ze_context_handle_t(std::ptr::null_mut()),
            device: ze_device_handle_t(std::ptr::null_mut()),
            module: ze_module_handle_t(std::ptr::null_mut()),
            kernel: ze_kernel_handle_t(std::ptr::null_mut()),
            name: "unhandled_elementwise_probe".to_string(),
            ptx_source: Some(std::sync::Arc::new(SIMPLE_PTX.to_string())),
            cubin_binary: None,
            module_handle: 0,
        };
        let mut kernel_params = [std::ptr::null_mut(); 16];
        let before = TMATMUL_NAMED_FALLBACK_TEST_COUNT.load(std::sync::atomic::Ordering::SeqCst);

        let result = unsafe {
            launch_kernel(
                &kernel,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                ze_command_queue_handle_t(std::ptr::null_mut()),
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        assert_eq!(result, ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
        let after = TMATMUL_NAMED_FALLBACK_TEST_COUNT.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(after, before + 1);
    }

    #[test]
    fn jit_success_continue_native_does_not_succeed_in_virtual_fallback() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", None),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
            ("HETGPU_TMATMUL_INTERPRETER", None),
            ("HETGPU_CXL_TMATMUL", Some("1")),
            ("HETGPU_TMATMUL_CXL", None),
            ("HETGPU_TMATMUL_ARTIFACT_DIR", None),
        ]);
        let kernel = super::super::module::ZeKernel {
            context: ze_context_handle_t(std::ptr::null_mut()),
            device: ze_device_handle_t(std::ptr::null_mut()),
            module: ze_module_handle_t(std::ptr::null_mut()),
            kernel: ze_kernel_handle_t(std::ptr::null_mut()),
            name: "unhandled_copy_probe".to_string(),
            ptx_source: Some(std::sync::Arc::new(SIMPLE_PTX.to_string())),
            cubin_binary: None,
            module_handle: 0,
        };
        let mut kernel_params = [std::ptr::null_mut(); 16];

        let result = unsafe {
            launch_kernel(
                &kernel,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                ze_command_queue_handle_t(std::ptr::null_mut()),
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        assert_eq!(result, ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
    }

    #[test]
    fn compute_batched_ptrs_fills_tables_without_named_fallback_gate() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", None),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", None),
        ]);
        let src0 = vec![0u8; 1024];
        let src1 = vec![0u8; 1024];
        let dst = vec![0u8; 1024];
        let mut ptrs_src = vec![0u64; 12];
        let mut ptrs_dst = vec![0u64; 6];
        let _alloc_guard = VirtualAllocGuard::insert(&[
            (src0.as_ptr() as usize, src0.len()),
            (src1.as_ptr() as usize, src1.len()),
            (dst.as_ptr() as usize, dst.len()),
            (
                ptrs_src.as_mut_ptr() as usize,
                64 * std::mem::size_of::<u64>(),
            ),
            (
                ptrs_dst.as_mut_ptr() as usize,
                ptrs_dst.len() * std::mem::size_of::<u64>(),
            ),
        ]);

        let mut p0 = src0.as_ptr() as u64;
        let mut p1 = src1.as_ptr() as u64;
        let mut p2 = dst.as_ptr() as u64;
        let mut p3 = ptrs_src.as_mut_ptr() as u64;
        let mut p4 = ptrs_dst.as_mut_ptr() as u64;
        let mut ne12 = 2i64;
        let mut ne13 = 3i64;
        let mut ne23 = 6i64;
        let mut nb02 = 10u64;
        let mut nb03 = 100u64;
        let mut nb12 = 20u64;
        let mut nb13 = 200u64;
        let mut nbd2 = 30u64;
        let mut nbd3 = 300u64;
        let mut r2 = 1i64;
        let mut r3 = 1i64;
        let mut kernel_params = [
            &mut p0 as *mut _ as *mut ::core::ffi::c_void,
            &mut p1 as *mut _ as *mut ::core::ffi::c_void,
            &mut p2 as *mut _ as *mut ::core::ffi::c_void,
            &mut p3 as *mut _ as *mut ::core::ffi::c_void,
            &mut p4 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne12 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne13 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne23 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb02 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb03 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb12 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb13 as *mut _ as *mut ::core::ffi::c_void,
            &mut nbd2 as *mut _ as *mut ::core::ffi::c_void,
            &mut nbd3 as *mut _ as *mut ::core::ffi::c_void,
            &mut r2 as *mut _ as *mut ::core::ffi::c_void,
            &mut r3 as *mut _ as *mut ::core::ffi::c_void,
        ];

        let status = unsafe {
            execute_kernel_name_fallback(
                "_Z22k_compute_batched_ptrsPK6__halfS1_PcPPKvPPvlllmmmmmmll",
                kernel_params.as_mut_ptr(),
                1,
                1,
                1,
                1,
                1,
                1,
            )
        };

        assert_eq!(status, KernelNameFallbackStatus::Handled);
        assert_eq!(ptrs_src[0], src0.as_ptr() as u64);
        assert_eq!(ptrs_src[5], src0.as_ptr() as u64 + 210);
        assert_eq!(ptrs_src[6], src1.as_ptr() as u64);
        assert_eq!(ptrs_src[11], src1.as_ptr() as u64 + 420);
        assert_eq!(ptrs_dst[0], dst.as_ptr() as u64);
        assert_eq!(ptrs_dst[5], dst.as_ptr() as u64 + 630);
    }

    #[test]
    fn compute_batched_ptrs_reads_ne12_as_kernel_i64() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", None),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", None),
        ]);
        let src0 = vec![0u8; 1024];
        let src1 = vec![0u8; 1024];
        let dst = vec![0u8; 1024];
        let mut ptrs_src = vec![0u64; 128];
        let mut ptrs_dst = vec![0u64; 64];
        let _alloc_guard = VirtualAllocGuard::insert(&[
            (src0.as_ptr() as usize, src0.len()),
            (src1.as_ptr() as usize, src1.len()),
            (dst.as_ptr() as usize, dst.len()),
            (
                ptrs_src.as_mut_ptr() as usize,
                ptrs_src.len() * std::mem::size_of::<u64>(),
            ),
            (
                ptrs_dst.as_mut_ptr() as usize,
                ptrs_dst.len() * std::mem::size_of::<u64>(),
            ),
        ]);

        let mut p0 = src0.as_ptr() as u64;
        let mut p1 = src1.as_ptr() as u64;
        let mut p2 = dst.as_ptr() as u64;
        let mut p3 = ptrs_src.as_mut_ptr() as u64;
        let mut p4 = ptrs_dst.as_mut_ptr() as u64;
        let mut ne12_storage = [64i64, 1i64];
        let mut ne13 = 1i64;
        let mut ne23 = 64i64;
        let mut nb02 = 4u64;
        let mut nb03 = 400u64;
        let mut nb12 = 8u64;
        let mut nb13 = 800u64;
        let mut nbd2 = 12u64;
        let mut nbd3 = 1200u64;
        let mut r2 = 1i64;
        let mut r3 = 1i64;
        let mut kernel_params = [
            &mut p0 as *mut _ as *mut ::core::ffi::c_void,
            &mut p1 as *mut _ as *mut ::core::ffi::c_void,
            &mut p2 as *mut _ as *mut ::core::ffi::c_void,
            &mut p3 as *mut _ as *mut ::core::ffi::c_void,
            &mut p4 as *mut _ as *mut ::core::ffi::c_void,
            ne12_storage.as_mut_ptr() as *mut ::core::ffi::c_void,
            &mut ne13 as *mut _ as *mut ::core::ffi::c_void,
            &mut ne23 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb02 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb03 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb12 as *mut _ as *mut ::core::ffi::c_void,
            &mut nb13 as *mut _ as *mut ::core::ffi::c_void,
            &mut nbd2 as *mut _ as *mut ::core::ffi::c_void,
            &mut nbd3 as *mut _ as *mut ::core::ffi::c_void,
            &mut r2 as *mut _ as *mut ::core::ffi::c_void,
            &mut r3 as *mut _ as *mut ::core::ffi::c_void,
        ];

        let status = unsafe {
            execute_kernel_name_fallback(
                "_Z22k_compute_batched_ptrsPK6__halfS1_PcPPKvPPvlllmmmmmmll",
                kernel_params.as_mut_ptr(),
                1,
                1,
                1,
                1,
                64,
                1,
            )
        };

        assert_eq!(status, KernelNameFallbackStatus::Handled);
        assert_eq!(ptrs_src[63], src0.as_ptr() as u64 + 63 * 4);
        assert_eq!(ptrs_src[127], src1.as_ptr() as u64 + 63 * 8);
        assert_eq!(ptrs_dst[63], dst.as_ptr() as u64 + 63 * 12);
    }

    #[test]
    fn bitnet_strict_reject_does_not_succeed_in_virtual_fallback() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", Some("1")),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
            ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
            ("HETGPU_BITNET_DISAGG_STRICT", Some("1")),
            ("HETGPU_BITNET_FFN_CXL", None),
            ("HETGPU_TMATMUL_BITNET_DISAGGREGATE", None),
            ("HETGPU_BITNET_CXL_KERNELS", None),
            ("HETGPU_BITNET_GPU_KERNELS", None),
            ("HETGPU_BITNET_ROUTE_MANIFEST", None),
            ("HETGPU_BITNET_ROUTE_LOG", None),
        ]);
        let kernel = super::super::module::ZeKernel {
            context: ze_context_handle_t(std::ptr::null_mut()),
            device: ze_device_handle_t(std::ptr::null_mut()),
            module: ze_module_handle_t(std::ptr::null_mut()),
            kernel: ze_kernel_handle_t(std::ptr::null_mut()),
            name: "unknown_matmul_probe".to_string(),
            ptx_source: None,
            cubin_binary: None,
            module_handle: 0,
        };
        let mut kernel_params = [std::ptr::null_mut(); 16];

        let result = unsafe {
            launch_kernel(
                &kernel,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                ze_command_queue_handle_t(std::ptr::null_mut()),
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        assert_eq!(result, ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
    }

    #[test]
    fn strict_cxl_submit_failure_does_not_succeed_in_virtual_fallback() {
        let _lock = FALLBACK_TEST_MUTEX.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let asm_path = dir.path().join("kernel.S");
        let asm_text = asm_path.to_string_lossy().to_string();
        let _guard = EnvGuard::set(&[
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", Some("1")),
            ("HETGPU_TMATMUL_NAMED_FALLBACK", Some("1")),
            ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
            ("HETGPU_BITNET_DISAGG_STRICT", Some("1")),
            ("HETGPU_BITNET_FFN_CXL", None),
            ("HETGPU_TMATMUL_BITNET_DISAGGREGATE", None),
            ("HETGPU_BITNET_CXL_KERNELS", None),
            ("HETGPU_BITNET_GPU_KERNELS", None),
            ("HETGPU_BITNET_ROUTE_MANIFEST", None),
            ("HETGPU_BITNET_ROUTE_LOG", None),
            ("HETGPU_CXL_TMATMUL", Some("1")),
            ("HETGPU_TMATMUL_CXL", None),
            ("HETGPU_TMATMUL_ASM_PATH", Some(&asm_text)),
            ("HETGPU_TMATMUL_NUMEL", Some("1")),
            ("HETGPU_CXL_TMATMUL_DEV", None),
            ("HETGPU_CXL_TMATMUL_DEVICE", None),
        ]);
        let kernel = super::super::module::ZeKernel {
            context: ze_context_handle_t(std::ptr::null_mut()),
            device: ze_device_handle_t(std::ptr::null_mut()),
            module: ze_module_handle_t(std::ptr::null_mut()),
            kernel: ze_kernel_handle_t(std::ptr::null_mut()),
            name: "layer_0_ffn_gate_mul_mat".to_string(),
            ptx_source: None,
            cubin_binary: None,
            module_handle: 0,
        };
        let mut kernel_params = [std::ptr::null_mut(); 16];

        let result = unsafe {
            launch_kernel(
                &kernel,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                ze_command_queue_handle_t(std::ptr::null_mut()),
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        assert_eq!(result, ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
    }

    #[test]
    fn bitnet_default_disaggregation_keeps_attention_on_gpu() {
        let mut cfg = super::super::bitnet_disagg::BitnetRouteConfig::default();
        cfg.enabled = true;

        assert_eq!(
            super::super::bitnet_disagg::classify_kernel_name(
                "_z13mul_mat_vec_q_bitnet_ffn_gate_iq1_s",
                &cfg
            )
            .route,
            super::super::bitnet_disagg::BitnetRoute::CxlTmatmul
        );
        assert_eq!(
            super::super::bitnet_disagg::classify_kernel_name("_z13flash_attn_mul_mat_q", &cfg)
                .route,
            super::super::bitnet_disagg::BitnetRoute::GpuNative
        );
        assert_eq!(
            super::super::bitnet_disagg::classify_kernel_name("_z17rope_neox_attention", &cfg)
                .route,
            super::super::bitnet_disagg::BitnetRoute::GpuNative
        );
    }

    #[test]
    fn bitnet_disaggregation_config_markers_override_default_routing() {
        let mut cfg = super::super::bitnet_disagg::BitnetRouteConfig::default();
        cfg.enabled = true;
        cfg.cxl_markers = vec!["ffn_gate".to_string(), "mlp_up".to_string()];
        cfg.gpu_markers = vec!["force_gpu".to_string()];

        assert_eq!(
            super::super::bitnet_disagg::classify_kernel_name("layer_03_ffn_gate_mul_mat", &cfg)
                .route,
            super::super::bitnet_disagg::BitnetRoute::CxlTmatmul
        );
        assert_eq!(
            super::super::bitnet_disagg::classify_kernel_name("layer_04_mlp_up_mul_mat", &cfg)
                .route,
            super::super::bitnet_disagg::BitnetRoute::CxlTmatmul
        );
        assert_eq!(
            super::super::bitnet_disagg::classify_kernel_name("layer_05_mul_mat_q", &cfg).route,
            super::super::bitnet_disagg::BitnetRoute::CxlTmatmul
        );
        assert_eq!(
            super::super::bitnet_disagg::classify_kernel_name("layer_06_force_gpu_ffn_gate", &cfg)
                .route,
            super::super::bitnet_disagg::BitnetRoute::GpuNative
        );
    }
}

#[cfg(feature = "intel")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KernelNameFallbackStatus {
    Handled,
    ContinueNative,
    Rejected,
}

#[cfg(feature = "intel")]
unsafe fn execute_kernel_name_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    _block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
) -> KernelNameFallbackStatus {
    if kernel_params.is_null() {
        eprintln!(
            "[TMatmul Fallback] Kernel '{}' - null kernel_params, skipping",
            kernel_name
        );
        return KernelNameFallbackStatus::ContinueNative;
    }

    let name_lower = kernel_name.to_lowercase();

    if name_lower.contains("compute_batched_ptrs") {
        return execute_tmatmul_compute_batched_ptrs_fallback(kernel_name, kernel_params);
    }

    if !tmatmul_named_fallback_enabled() && !tmatmul_hardware_matmul_enabled() {
        return KernelNameFallbackStatus::ContinueNative;
    }

    if super::bitnet_disagg::enabled_from_env() && !tmatmul_is_matmul_kernel_name(&name_lower) {
        let route_config = super::bitnet_disagg::config_from_env();
        let decision = super::bitnet_disagg::classify_kernel_name(kernel_name, &route_config);
        let cxl_enabled = super::cxl_tmatmul::cxl_tmatmul_enabled();
        if let Err(err) = super::bitnet_disagg::append_route_log_from_env(
            &decision,
            super::bitnet_disagg::RouteHardware::cxl(
                cxl_enabled,
                tmatmul_hardware_matmul_enabled(),
            ),
        ) {
            eprintln!(
                "[BitNet Disagg] route log failed for '{}': {}",
                kernel_name, err
            );
            if decision.strict && decision.route == super::bitnet_disagg::BitnetRoute::CxlTmatmul {
                return KernelNameFallbackStatus::Rejected;
            }
        }

        match decision.route {
            super::bitnet_disagg::BitnetRoute::GpuNative => {
                eprintln!(
                    "[BitNet Disagg] Keeping '{}' on GPU path via {}",
                    kernel_name,
                    decision.source.as_str()
                );
                return KernelNameFallbackStatus::ContinueNative;
            }
            super::bitnet_disagg::BitnetRoute::Reject => {
                eprintln!(
                    "[BitNet Disagg] Strict mode rejected '{}' via {}",
                    kernel_name,
                    decision.source.as_str()
                );
                return KernelNameFallbackStatus::Rejected;
            }
            super::bitnet_disagg::BitnetRoute::CxlTmatmul => {
                eprintln!(
                    "[BitNet Disagg] Non-matmul kernel '{}' matched CXL route via {}; leaving on fallback/native path",
                    kernel_name,
                    decision.source.as_str()
                );
            }
            super::bitnet_disagg::BitnetRoute::Fallback => {}
        }
    }

    if tmatmul_hardware_matmul_enabled() && tmatmul_is_matmul_kernel_name(&name_lower) {
        let mut strict_cxl_submit_failure = false;
        let mut run_hardware_matmul = true;
        if super::bitnet_disagg::enabled_from_env() {
            let route_config = super::bitnet_disagg::config_from_env();
            let decision = super::bitnet_disagg::classify_kernel_name(kernel_name, &route_config);
            let cxl_enabled = super::cxl_tmatmul::cxl_tmatmul_enabled();
            if let Err(err) = super::bitnet_disagg::append_route_log_from_env(
                &decision,
                super::bitnet_disagg::RouteHardware::cxl(cxl_enabled, true),
            ) {
                eprintln!(
                    "[BitNet Disagg] route log failed for '{}': {}",
                    kernel_name, err
                );
                if decision.strict
                    && decision.route == super::bitnet_disagg::BitnetRoute::CxlTmatmul
                {
                    return KernelNameFallbackStatus::Rejected;
                }
            }

            match decision.route {
                super::bitnet_disagg::BitnetRoute::CxlTmatmul => {
                    strict_cxl_submit_failure = decision.strict;
                    eprintln!(
                        "[BitNet Disagg] Routing '{}' to CXL tmatmul candidate via {}",
                        kernel_name,
                        decision.source.as_str()
                    );
                }
                super::bitnet_disagg::BitnetRoute::GpuNative => {
                    eprintln!(
                        "[BitNet Disagg] Keeping '{}' on GPU path via {}",
                        kernel_name,
                        decision.source.as_str()
                    );
                    return KernelNameFallbackStatus::ContinueNative;
                }
                super::bitnet_disagg::BitnetRoute::Fallback => {
                    eprintln!(
                        "[BitNet Disagg] Leaving '{}' on fallback/native path via {}",
                        kernel_name,
                        decision.source.as_str()
                    );
                    run_hardware_matmul = false;
                }
                super::bitnet_disagg::BitnetRoute::Reject => {
                    eprintln!(
                        "[BitNet Disagg] Strict mode rejected '{}' via {}",
                        kernel_name,
                        decision.source.as_str()
                    );
                    return KernelNameFallbackStatus::Rejected;
                }
            }
        }
        if run_hardware_matmul {
            let status = execute_tmatmul_hardware_matmul_fallback(
                kernel_name,
                &name_lower,
                kernel_params,
                strict_cxl_submit_failure,
            );
            match status {
                KernelNameFallbackStatus::Handled | KernelNameFallbackStatus::Rejected => {
                    return status;
                }
                KernelNameFallbackStatus::ContinueNative => {}
            }
        }
    }

    if !tmatmul_named_fallback_enabled() {
        return KernelNameFallbackStatus::ContinueNative;
    }
    note_tmatmul_named_fallback_for_test();

    // Handle different kernel types
    if name_lower.contains("reduce_kernel") {
        execute_reduce_kernel_fallback(kernel_name, &name_lower, kernel_params);
        return KernelNameFallbackStatus::Handled;
    }
    if name_lower.contains("softmax") || name_lower.contains("soft_max") {
        execute_softmax_kernel_fallback(kernel_name, &name_lower, kernel_params);
        return KernelNameFallbackStatus::Handled;
    }
    if name_lower.contains("indexselect") || name_lower.contains("index_select") {
        execute_indexselect_kernel_fallback(kernel_name, kernel_params);
        return KernelNameFallbackStatus::Handled;
    }
    if name_lower.contains("gemm") || name_lower.contains("matmul") || name_lower.contains("cublas")
    {
        execute_matmul_kernel_fallback(kernel_name, kernel_params);
        return KernelNameFallbackStatus::Handled;
    }
    if name_lower.contains("layernorm")
        || name_lower.contains("layer_norm")
        || name_lower.contains("rmsnorm")
        || name_lower.contains("rms_norm")
        || name_lower.contains("welford")
    {
        execute_norm_kernel_fallback(
            kernel_name,
            &name_lower,
            kernel_params,
            grid_dim_x,
            block_dim_y,
        );
        return KernelNameFallbackStatus::Handled;
    }
    if name_lower.contains("k_bin_bcast") {
        if execute_tmatmul_bin_bcast_f32_fallback(kernel_name, kernel_params) {
            return KernelNameFallbackStatus::Handled;
        }
        eprintln!(
            "[TMatmul Fallback] k_bin_bcast '{}' not handled by f32 fallback",
            kernel_name
        );
        return KernelNameFallbackStatus::ContinueNative;
    }
    if name_lower.contains("mul_mat_q_stream_k_fixup") {
        if execute_tmatmul_mul_mat_q_stream_k_fixup_fallback(
            kernel_name,
            kernel_params,
            grid_dim_x,
            grid_dim_y,
        ) {
            return KernelNameFallbackStatus::Handled;
        }
        eprintln!(
            "[TMatmul Fallback] mul_mat_q_stream_k_fixup '{}' not handled by host fallback",
            kernel_name
        );
        return KernelNameFallbackStatus::ContinueNative;
    }
    if name_lower.contains("quantize_mmq_q8_1") {
        if execute_tmatmul_quantize_mmq_q8_1_fallback(
            kernel_name,
            kernel_params,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
        ) {
            return KernelNameFallbackStatus::Handled;
        }
        eprintln!(
            "[TMatmul Fallback] quantize_mmq_q8_1 '{}' not handled by host fallback",
            kernel_name
        );
        return KernelNameFallbackStatus::ContinueNative;
    }
    if name_lower.contains("quantize_q8_1") {
        if execute_tmatmul_quantize_q8_1_fallback(kernel_name, kernel_params, grid_dim_y) {
            return KernelNameFallbackStatus::Handled;
        }
        eprintln!(
            "[TMatmul Fallback] quantize_q8_1 '{}' not handled by host fallback",
            kernel_name
        );
        return KernelNameFallbackStatus::ContinueNative;
    }
    if name_lower.contains("cpy_f32_f16") {
        if execute_tmatmul_cpy_f32_f16_fallback(kernel_name, kernel_params) {
            return KernelNameFallbackStatus::Handled;
        }
        eprintln!(
            "[TMatmul Fallback] cpy_f32_f16 '{}' not handled by host fallback",
            kernel_name
        );
        return KernelNameFallbackStatus::ContinueNative;
    }
    if name_lower.contains("convert_unary") {
        if execute_tmatmul_convert_unary_fallback(kernel_name, kernel_params) {
            return KernelNameFallbackStatus::Handled;
        }
        eprintln!(
            "[TMatmul Fallback] convert_unary '{}' not handled by host fallback",
            kernel_name
        );
        return KernelNameFallbackStatus::ContinueNative;
    }
    if tmatmul_direct_unary_f32_op(&name_lower).is_some() {
        if execute_tmatmul_direct_unary_f32_fallback(kernel_name, &name_lower, kernel_params) {
            return KernelNameFallbackStatus::Handled;
        }
        eprintln!(
            "[TMatmul Fallback] direct unary f32 '{}' not handled by host fallback",
            kernel_name
        );
        return KernelNameFallbackStatus::ContinueNative;
    }
    if name_lower.contains("k_get_rows_float") {
        if execute_tmatmul_get_rows_float_fallback(
            kernel_name,
            kernel_params,
            grid_dim_y,
            grid_dim_z,
            block_dim_y,
            block_dim_z,
        ) {
            return KernelNameFallbackStatus::Handled;
        }
        eprintln!(
            "[TMatmul Fallback] k_get_rows_float '{}' not handled by host fallback",
            kernel_name
        );
        return KernelNameFallbackStatus::ContinueNative;
    }
    if name_lower.contains("concat_f32_non_cont") {
        if execute_tmatmul_concat_f32_non_cont_fallback(kernel_name, kernel_params) {
            return KernelNameFallbackStatus::Handled;
        }
        eprintln!(
            "[TMatmul Fallback] concat_f32_non_cont '{}' not handled by host fallback",
            kernel_name
        );
        return KernelNameFallbackStatus::ContinueNative;
    }
    if name_lower.contains("concat_f32_dim") {
        if execute_tmatmul_concat_f32_dim_fallback(
            kernel_name,
            &name_lower,
            kernel_params,
            grid_dim_y,
            grid_dim_z,
        ) {
            return KernelNameFallbackStatus::Handled;
        }
        eprintln!(
            "[TMatmul Fallback] concat_f32_dim '{}' not handled by host fallback",
            kernel_name
        );
        return KernelNameFallbackStatus::ContinueNative;
    }
    if name_lower.contains("rope_norm") || name_lower.contains("rope_neox") {
        if execute_tmatmul_rope_f32_fallback(kernel_name, &name_lower, kernel_params, grid_dim_x) {
            return KernelNameFallbackStatus::Handled;
        }
        eprintln!(
            "[TMatmul Fallback] rope '{}' not handled by host fallback",
            kernel_name
        );
        return KernelNameFallbackStatus::ContinueNative;
    }

    // Handle vectorized_elementwise_kernel from PyTorch.
    // Signature: vectorized_elementwise_kernel(int N, Functor f, std::array<char*, K> data)
    // kernel_params[0] = &N (int32)
    // kernel_params[1] = &functor
    // kernel_params[2] = &data_array (K sequential char* pointers)
    if !name_lower.contains("vectorized_elementwise_kernel")
        && !name_lower.contains("unrolled_elementwise_kernel")
        && !name_lower.contains("elementwise_kernel")
    {
        eprintln!(
            "[TMatmul Fallback] Unhandled kernel '{}' - no-op",
            kernel_name
        );
        return KernelNameFallbackStatus::ContinueNative;
    }

    // Determine the operation and number of data pointers
    let (op, num_ptrs) = detect_vectorized_op(kernel_name);
    let op = match op {
        Some(o) => o,
        None => {
            eprintln!(
                "[TMatmul Fallback] Unrecognized op in '{}' - no-op",
                kernel_name
            );
            return KernelNameFallbackStatus::Handled;
        }
    };

    // Read numel from kernel_params[0]
    let numel_param = *kernel_params.add(0);
    if numel_param.is_null() {
        eprintln!("[TMatmul Fallback] kernel_params[0] is null");
        return KernelNameFallbackStatus::Handled;
    }
    let numel = (numel_param as *const i32).read_unaligned() as usize;
    if numel == 0 || numel > 64 * 1024 * 1024 {
        eprintln!(
            "[TMatmul Fallback] Invalid numel={} for '{}'",
            numel, kernel_name
        );
        return KernelNameFallbackStatus::Handled;
    }

    // Read tensor pointers from the std::array at kernel_params[2]
    let data_param = *kernel_params.add(2);
    if data_param.is_null() {
        eprintln!("[TMatmul Fallback] kernel_params[2] (data array) is null");
        return KernelNameFallbackStatus::Handled;
    }

    let mut data_ptrs: Vec<*mut u8> = Vec::new();
    for i in 0..num_ptrs {
        let ptr_val = (data_param as *const u64).add(i).read_unaligned();
        // Verify this pointer is in our allocation map
        if let Some(_size) = super::memory::get_alloc_size(ptr_val as usize) {
            data_ptrs.push(ptr_val as *mut u8);
        } else {
            eprintln!(
                "[TMatmul Fallback] data_ptrs[{}] = {:#x} not in alloc map",
                i, ptr_val
            );
            data_ptrs.push(std::ptr::null_mut());
        }
    }

    // Validate we have at least output and one input
    if data_ptrs.len() < 2 || data_ptrs[0].is_null() || data_ptrs[1].is_null() {
        eprintln!(
            "[TMatmul Fallback] Missing output or input pointer for '{}'",
            kernel_name
        );
        return KernelNameFallbackStatus::Handled;
    }

    // Detect element size from kernel name (float16=2, float32/float=4, double=8)
    let elem_size: usize = if name_lower.contains("float16")
        || name_lower.contains("half")
        || name_lower.contains("f16")
        || name_lower.contains("bf16")
        || name_lower.contains("bfloat16")
    {
        2
    } else if name_lower.contains("double") || name_lower.contains("float64") {
        8
    } else {
        4 // default f32
    };

    eprintln!(
        "[TMatmul Fallback] Executing {} on {} elements ({} ptrs, {}B each) for '{}'",
        op, numel, num_ptrs, elem_size, kernel_name
    );

    // Read functor data from kernel_params[1] (used by some ops for alpha/scalar)
    let functor_param = *kernel_params.add(1);
    let functor_f32 = if !functor_param.is_null() {
        (functor_param as *const f32).read_unaligned()
    } else {
        1.0f32
    };

    // Helper closures for reading/writing elements with dtype conversion
    // For f16: read u16 bits, convert to f32 for math, convert back
    #[inline(always)]
    unsafe fn read_f16(ptr: *const u8) -> f32 {
        let bits = (ptr as *const u16).read_unaligned();
        f16_to_f32(bits)
    }
    #[inline(always)]
    unsafe fn write_f16(ptr: *mut u8, val: f32) {
        (ptr as *mut u16).write_unaligned(f32_to_f16(val));
    }
    #[inline(always)]
    unsafe fn read_elem(ptr: *const u8, offset: usize, elem_size: usize) -> f32 {
        match elem_size {
            2 => read_f16(ptr.add(offset * 2)),
            8 => (ptr.add(offset * 8) as *const f64).read_unaligned() as f32,
            _ => (ptr.add(offset * 4) as *const f32).read_unaligned(),
        }
    }
    #[inline(always)]
    unsafe fn write_elem(ptr: *mut u8, offset: usize, elem_size: usize, val: f32) {
        match elem_size {
            2 => write_f16(ptr.add(offset * 2), val),
            8 => (ptr.add(offset * 8) as *mut f64).write_unaligned(val as f64),
            _ => (ptr.add(offset * 4) as *mut f32).write_unaligned(val),
        }
    }

    // Execute the operation directly on the tensor data
    match op {
        "add" => {
            if data_ptrs.len() < 3 || data_ptrs[2].is_null() {
                eprintln!("[TMatmul Fallback] add needs 3 valid pointers");
                return KernelNameFallbackStatus::Handled;
            }
            let alpha = functor_f32;
            let out = data_ptrs[0];
            let in1 = data_ptrs[1];
            let in2 = data_ptrs[2];
            for i in 0..numel {
                let a = read_elem(in1, i, elem_size);
                let b = read_elem(in2, i, elem_size);
                write_elem(out, i, elem_size, a + alpha * b);
            }
        }
        "mul" | "div" => {
            if data_ptrs.len() < 3 || data_ptrs[2].is_null() {
                eprintln!("[TMatmul Fallback] Binary op needs 3 valid pointers");
                return KernelNameFallbackStatus::Handled;
            }
            let out = data_ptrs[0];
            let in1 = data_ptrs[1];
            let in2 = data_ptrs[2];
            for i in 0..numel {
                let a = read_elem(in1, i, elem_size);
                let b = read_elem(in2, i, elem_size);
                let result = match op {
                    "mul" => a * b,
                    "div" => {
                        if b != 0.0 {
                            a / b
                        } else {
                            0.0
                        }
                    }
                    _ => unreachable!(),
                };
                write_elem(out, i, elem_size, result);
            }
        }
        "abs" => {
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                write_elem(out, i, elem_size, x.abs());
            }
        }
        "add_scalar" => {
            let scalar = functor_f32;
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                write_elem(out, i, elem_size, x + scalar);
            }
        }
        "clamp" => {
            let min_val = functor_f32;
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                write_elem(out, i, elem_size, if x < min_val { min_val } else { x });
            }
        }
        "rsqrt" => {
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                write_elem(out, i, elem_size, 1.0 / x.sqrt());
            }
        }
        "pow" => {
            let exponent = functor_f32;
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                write_elem(out, i, elem_size, x.powf(exponent));
            }
        }
        "mul_scalar" => {
            let scalar = functor_f32;
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                write_elem(out, i, elem_size, x * scalar);
            }
        }
        "sigmoid" => {
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                write_elem(out, i, elem_size, 1.0 / (1.0 + (-x).exp()));
            }
        }
        "silu" => {
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                let sig = 1.0 / (1.0 + (-x).exp());
                write_elem(out, i, elem_size, x * sig);
            }
        }
        "relu" => {
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                write_elem(out, i, elem_size, if x > 0.0 { x } else { 0.0 });
            }
        }
        "tanh" => {
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                write_elem(out, i, elem_size, x.tanh());
            }
        }
        "copy" => {
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            std::ptr::copy_nonoverlapping(inp, out, numel * elem_size);
        }
        "neg" => {
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                write_elem(out, i, elem_size, -x);
            }
        }
        "exp" => {
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                write_elem(out, i, elem_size, x.exp());
            }
        }
        "gelu" => {
            let out = data_ptrs[0];
            let inp = data_ptrs[1];
            for i in 0..numel {
                let x = read_elem(inp, i, elem_size);
                let c = 0.7978845608_f32; // sqrt(2/pi)
                let inner = c * (x + 0.044715 * x * x * x);
                write_elem(out, i, elem_size, 0.5 * x * (1.0 + inner.tanh()));
            }
        }
        _ => {
            eprintln!("[TMatmul Fallback] Unimplemented op '{}' - no-op", op);
        }
    }
    eprintln!(
        "[TMatmul Fallback] Kernel '{}' executed successfully ({} elements, {}B)",
        kernel_name, numel, elem_size
    );
    KernelNameFallbackStatus::Handled
}

/// Execute PyTorch arange_cuda_out fallback.
#[cfg(feature = "intel")]
unsafe fn execute_arange_kernel_fallback(
    kernel_name: &str,
    name_lower: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) {
    let numel_param = *kernel_params.add(0);
    let mut numel = 0usize;
    if !numel_param.is_null() {
        let n_i32 = (numel_param as *const i32).read_unaligned();
        if n_i32 > 0 {
            numel = n_i32 as usize;
        } else {
            let n_u64 = (numel_param as *const u64).read_unaligned();
            if n_u64 > 0 && n_u64 <= 64 * 1024 * 1024 {
                numel = n_u64 as usize;
            }
        }
    }

    let mut all_ptrs: Vec<(usize, u64, usize)> = Vec::new();
    let functor_param = *kernel_params.add(1);
    if !functor_param.is_null() {
        if let Some((ptr, size)) = read_alloc_pointer_from_param(functor_param) {
            all_ptrs.push((1, ptr, size));
        }
        let inner = scan_for_alloc_pointers(functor_param as *const u8, 128);
        all_ptrs.extend(
            inner
                .into_iter()
                .map(|(off, ptr, sz)| (1000 + off, ptr, sz)),
        );
    }
    let out_param = *kernel_params.add(2);
    if !out_param.is_null() {
        if let Some((ptr, size)) = read_alloc_pointer_from_param(out_param) {
            all_ptrs.push((2, ptr, size));
        }
    }

    all_ptrs.sort_by_key(|&(_, ptr, _)| ptr);
    all_ptrs.dedup_by_key(|a| a.1);

    if all_ptrs.is_empty() {
        eprintln!(
            "[TMatmul Fallback] arange: found no output pointer for '{}'",
            kernel_name
        );
        return;
    }

    let selected = all_ptrs
        .iter()
        .copied()
        .filter(|&(_, _, size)| {
            numel == 0 || (size >= numel && size % numel == 0 && size / numel <= 8)
        })
        .max_by_key(|&(_, _, size)| size)
        .unwrap_or_else(|| {
            all_ptrs
                .iter()
                .copied()
                .max_by_key(|&(_, _, size)| size)
                .unwrap()
        });

    let (_, out_ptr, out_size) = selected;
    if numel == 0 {
        numel = if out_size % 8 == 0 {
            out_size / 8
        } else {
            out_size / 4
        };
    }
    if numel == 0 || numel > 64 * 1024 * 1024 {
        eprintln!(
            "[TMatmul Fallback] arange: invalid numel={} for '{}'",
            numel, kernel_name
        );
        return;
    }

    let force_f32_arange = name_lower.contains("e5_clev")
        || name_lower.contains("float32")
        || name_lower.contains("float");
    let elem_size = if force_f32_arange {
        4
    } else if out_size >= numel && out_size % numel == 0 {
        match out_size / numel {
            1 | 2 | 4 | 8 => out_size / numel,
            n if n > 8 => 8,
            _ => 4,
        }
    } else if out_size >= numel * 8 {
        8
    } else if out_size >= numel * 4 {
        4
    } else if out_size >= numel * 2 {
        2
    } else {
        1
    };

    let out = out_ptr as *mut u8;
    let is_float64 = name_lower.contains("double") || name_lower.contains("float64");
    let is_float32 = elem_size == 4
        && !name_lower.contains("int32")
        && !name_lower.contains("uint32")
        && !name_lower.contains("long");

    for i in 0..numel {
        match elem_size {
            8 if is_float64 => (out.add(i * 8) as *mut f64).write_unaligned(i as f64),
            8 => (out.add(i * 8) as *mut i64).write_unaligned(i as i64),
            4 if is_float32 => (out.add(i * 4) as *mut f32).write_unaligned(i as f32),
            4 => (out.add(i * 4) as *mut i32).write_unaligned(i as i32),
            2 => (out.add(i * 2) as *mut u16).write_unaligned(f32_to_f16(i as f32)),
            _ => out.add(i).write_unaligned(i as u8),
        }
    }

    eprintln!(
        "[TMatmul Fallback] arange executed ({} elements, {}B each) for '{}'",
        numel, elem_size, kernel_name
    );
}

// IEEE 754 half-precision (f16) <-> f32 conversion
#[cfg(feature = "intel")]
#[inline]
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: convert to normalized f32
            let mut m = mant;
            let mut e: i32 = -14;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let f32_exp = ((e + 127) as u32) & 0xFF;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        if mant == 0 {
            f32::from_bits((sign << 31) | 0x7F800000)
        } else {
            f32::from_bits((sign << 31) | 0x7FC00000 | (mant << 13))
        }
    } else {
        // Normalized
        let f32_exp = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }
}

#[cfg(feature = "intel")]
#[inline]
fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 255 {
        // Inf or NaN
        if mant == 0 {
            (sign << 15) | 0x7C00
        } else {
            (sign << 15) | 0x7E00 // quiet NaN
        }
    } else if exp > 142 {
        // Overflow -> Inf
        (sign << 15) | 0x7C00
    } else if exp < 103 {
        // Underflow -> zero
        sign << 15
    } else if exp < 113 {
        // Subnormal
        let shift = 113 - exp;
        let m = (mant | 0x800000) >> (shift + 13);
        (sign << 15) | (m as u16)
    } else {
        // Normalized
        let h_exp = ((exp - 112) as u16) & 0x1F;
        let h_mant = (mant >> 13) as u16;
        (sign << 15) | (h_exp << 10) | h_mant
    }
}

/// Detect the operation and number of data pointers from a vectorized_elementwise_kernel name
#[cfg(feature = "intel")]
fn detect_vectorized_op(kernel_name: &str) -> (Option<&'static str>, usize) {
    let name_lower = kernel_name.to_lowercase();

    // Binary ops: std::array<char*, 3>
    // CUDAFunctor_add handles both add and sub (sub uses alpha=-1.0)
    if name_lower.contains("functor_add")
        || (name_lower.contains("add") && name_lower.contains("lm3"))
    {
        return (Some("add"), 3);
    }
    if name_lower.contains("mulfunctor")
        || (name_lower.contains("mul")
            && !name_lower.contains("cumul")
            && name_lower.contains("lm3"))
    {
        return (Some("mul"), 3);
    }
    if name_lower.contains("divfunctor")
        || (name_lower.contains("div") && name_lower.contains("lm3"))
    {
        return (Some("div"), 3);
    }

    // Unary ops: std::array<char*, 2>
    if name_lower.contains("sigmoid") {
        return (Some("sigmoid"), 2);
    }
    if name_lower.contains("silu") {
        return (Some("silu"), 2);
    }
    if name_lower.contains("tanh") && !name_lower.contains("atanh") {
        return (Some("tanh"), 2);
    }
    if name_lower.contains("gelu") {
        return (Some("gelu"), 2);
    }
    if name_lower.contains("absfunctor") {
        return (Some("abs"), 2);
    }
    if name_lower.contains("clamp") {
        return (Some("clamp"), 2);
    }
    if name_lower.contains("functoronselfadd") || name_lower.contains("functoronself_add") {
        return (Some("add_scalar"), 2);
    }
    if name_lower.contains("rsqrt") {
        return (Some("rsqrt"), 2);
    }
    if name_lower.contains("pow") {
        return (Some("pow"), 2);
    }
    if name_lower.contains("exp") && !name_lower.contains("export") {
        return (Some("exp"), 2);
    }
    if name_lower.contains("neg") {
        return (Some("neg"), 2);
    }
    if name_lower.contains("copy") || name_lower.contains("contiguous") {
        return (Some("copy"), 2);
    }
    if name_lower.contains("relu") {
        return (Some("relu"), 2);
    }
    if name_lower.contains("functoronselfmul") || name_lower.contains("functoronself_mul") {
        return (Some("mul_scalar"), 2);
    }

    // Check array size from template: "Lm3" means 3, "Lm2" means 2
    let num_ptrs = if name_lower.contains("lm3") {
        3
    } else if name_lower.contains("lm2") {
        2
    } else {
        2
    }; // default

    // Try to detect from functor name patterns
    if name_lower.contains("add") && num_ptrs == 3 {
        return (Some("add"), 3);
    }

    (None, num_ptrs)
}

/// Scan a memory region for pointers that match entries in the virtual alloc map.
/// Returns a list of (offset, pointer_value, alloc_size) tuples.
#[cfg(feature = "intel")]
unsafe fn scan_for_alloc_pointers(base: *const u8, scan_bytes: usize) -> Vec<(usize, u64, usize)> {
    let mut found = Vec::new();
    // Scan at 8-byte aligned offsets for 64-bit pointers
    let num_slots = scan_bytes / 8;
    for i in 0..num_slots {
        let ptr_val = (base as *const u64).add(i).read_unaligned();
        // Quick filter: valid user-space pointer range
        if ptr_val < 0x10000 || ptr_val > 0x7fff_ffff_ffff {
            continue;
        }
        // Check if this pointer is in our alloc map
        if let Some(size) = super::memory::get_alloc_size(ptr_val as usize) {
            found.push((i * 8, ptr_val, size));
        }
    }
    found
}

#[cfg(feature = "intel")]
unsafe fn resolve_alloc_pointer_value(ptr_val: u64) -> Option<(u64, usize)> {
    if let Some(size) = super::memory::get_alloc_size(ptr_val as usize) {
        return Some((ptr_val, size));
    }

    if ptr_val < 0x10000 || ptr_val > 0x7fff_ffff_ffff {
        return None;
    }
    let ptr = ptr_val as *const u8;
    if !is_memory_readable(ptr, 8) {
        return None;
    }
    let nested = (ptr as *const u64).read_unaligned();
    super::memory::get_alloc_size(nested as usize).map(|size| (nested, size))
}

#[cfg(feature = "intel")]
unsafe fn read_alloc_pointer_from_param(param: *mut ::core::ffi::c_void) -> Option<(u64, usize)> {
    if param.is_null() {
        return None;
    }
    let ptr_val = (param as *const u64).read_unaligned();
    resolve_alloc_pointer_value(ptr_val)
}

#[cfg(feature = "intel")]
unsafe fn tmatmul_read_param_u64(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<u64> {
    if kernel_params.is_null() {
        return None;
    }
    let param = *kernel_params.add(index);
    if param.is_null() || (param as usize) < 0x1000 {
        return None;
    }
    Some((param as *const u64).read_unaligned())
}

#[cfg(feature = "intel")]
unsafe fn tmatmul_read_param_i64(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<i64> {
    if kernel_params.is_null() {
        return None;
    }
    let param = *kernel_params.add(index);
    if param.is_null() || (param as usize) < 0x1000 {
        return None;
    }
    Some((param as *const i64).read_unaligned())
}

#[cfg(feature = "intel")]
unsafe fn tmatmul_read_param_i32(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<i32> {
    if kernel_params.is_null() {
        return None;
    }
    let param = *kernel_params.add(index);
    if param.is_null() || (param as usize) < 0x1000 {
        return None;
    }
    Some((param as *const i32).read_unaligned())
}

#[cfg(feature = "intel")]
unsafe fn tmatmul_read_param_f32(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<f32> {
    if kernel_params.is_null() {
        return None;
    }
    let param = *kernel_params.add(index);
    if param.is_null() || (param as usize) < 0x1000 {
        return None;
    }
    Some((param as *const f32).read_unaligned())
}

#[cfg(feature = "intel")]
unsafe fn tmatmul_read_param_uint3_z(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<u32> {
    if kernel_params.is_null() {
        return None;
    }
    let param = *kernel_params.add(index);
    if param.is_null() || (param as usize) < 0x1000 {
        return None;
    }
    Some((param as *const u32).add(2).read_unaligned())
}

#[cfg(feature = "intel")]
fn tmatmul_virtual_alloc_has_bytes(addr: u64, bytes: usize) -> bool {
    if bytes == 0 {
        return true;
    }
    super::memory::get_alloc_size(addr as usize)
        .map(|remaining| remaining >= bytes)
        .unwrap_or(false)
}

#[cfg(feature = "intel")]
fn tmatmul_virtual_alloc_has_elems<T>(ptr: *const T, elems: usize) -> bool {
    elems
        .checked_mul(std::mem::size_of::<T>())
        .map(|bytes| tmatmul_virtual_alloc_has_bytes(ptr as u64, bytes))
        .unwrap_or(false)
}

#[cfg(feature = "intel")]
fn tmatmul_host_or_virtual_alloc_has_bytes(addr: u64, bytes: usize, need_write: bool) -> bool {
    if tmatmul_virtual_alloc_has_bytes(addr, bytes) {
        return true;
    }
    let Ok(host_addr) = usize::try_from(addr) else {
        return false;
    };
    tmatmul_process_range_has_perms(host_addr, bytes, need_write)
}

#[cfg(feature = "intel")]
fn tmatmul_process_range_has_perms(addr: usize, len: usize, need_write: bool) -> bool {
    if len == 0 {
        return true;
    }
    let Some(end_addr) = addr.checked_add(len) else {
        return false;
    };
    let Ok(maps) = std::fs::read_to_string("/proc/self/maps") else {
        return false;
    };
    for line in maps.lines() {
        let mut parts = line.split_whitespace();
        let Some(range) = parts.next() else {
            continue;
        };
        let Some(perms) = parts.next() else {
            continue;
        };
        if !perms.starts_with('r') || (need_write && perms.as_bytes().get(1).copied() != Some(b'w'))
        {
            continue;
        }
        let Some((start_hex, end_hex)) = range.split_once('-') else {
            continue;
        };
        let Ok(start) = usize::from_str_radix(start_hex, 16) else {
            continue;
        };
        let Ok(end) = usize::from_str_radix(end_hex, 16) else {
            continue;
        };
        if addr >= start && end_addr <= end {
            return true;
        }
    }
    false
}

#[cfg(feature = "intel")]
fn tmatmul_pointer_table_has_elems<T>(ptr: *const T, elems: usize) -> bool {
    elems
        .checked_mul(std::mem::size_of::<T>())
        .map(|bytes| {
            tmatmul_virtual_alloc_has_bytes(ptr as u64, bytes)
                || tmatmul_process_range_has_perms(ptr as usize, bytes, true)
        })
        .unwrap_or(false)
}

#[cfg(feature = "intel")]
unsafe fn execute_tmatmul_compute_batched_ptrs_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> KernelNameFallbackStatus {
    let Some(src0) = tmatmul_read_param_u64(kernel_params, 0) else {
        eprintln!(
            "[TMatmul Fallback] compute_batched_ptrs '{}' missing src0",
            kernel_name
        );
        return KernelNameFallbackStatus::Rejected;
    };
    let Some(src1) = tmatmul_read_param_u64(kernel_params, 1) else {
        eprintln!(
            "[TMatmul Fallback] compute_batched_ptrs '{}' missing src1",
            kernel_name
        );
        return KernelNameFallbackStatus::Rejected;
    };
    let Some(dst) = tmatmul_read_param_u64(kernel_params, 2) else {
        eprintln!(
            "[TMatmul Fallback] compute_batched_ptrs '{}' missing dst",
            kernel_name
        );
        return KernelNameFallbackStatus::Rejected;
    };
    let Some(ptrs_src) = tmatmul_read_param_u64(kernel_params, 3) else {
        eprintln!(
            "[TMatmul Fallback] compute_batched_ptrs '{}' missing ptrs_src",
            kernel_name
        );
        return KernelNameFallbackStatus::Rejected;
    };
    let Some(ptrs_dst) = tmatmul_read_param_u64(kernel_params, 4) else {
        eprintln!(
            "[TMatmul Fallback] compute_batched_ptrs '{}' missing ptrs_dst",
            kernel_name
        );
        return KernelNameFallbackStatus::Rejected;
    };
    let Some(ne12) = tmatmul_read_param_i64(kernel_params, 5).map(|v| v.max(0) as usize) else {
        eprintln!(
            "[TMatmul Fallback] compute_batched_ptrs '{}' missing ne12",
            kernel_name
        );
        return KernelNameFallbackStatus::Rejected;
    };
    let Some(ne13) = tmatmul_read_param_i64(kernel_params, 6).map(|v| v.max(0) as usize) else {
        eprintln!(
            "[TMatmul Fallback] compute_batched_ptrs '{}' missing ne13",
            kernel_name
        );
        return KernelNameFallbackStatus::Rejected;
    };
    let Some(ne23) = tmatmul_read_param_i64(kernel_params, 7).map(|v| v.max(0) as usize) else {
        eprintln!(
            "[TMatmul Fallback] compute_batched_ptrs '{}' missing ne23",
            kernel_name
        );
        return KernelNameFallbackStatus::Rejected;
    };
    let Some(nb02) = tmatmul_read_param_u64(kernel_params, 8).map(|v| v as usize) else {
        return KernelNameFallbackStatus::Rejected;
    };
    let Some(nb03) = tmatmul_read_param_u64(kernel_params, 9).map(|v| v as usize) else {
        return KernelNameFallbackStatus::Rejected;
    };
    let Some(nb12) = tmatmul_read_param_u64(kernel_params, 10).map(|v| v as usize) else {
        return KernelNameFallbackStatus::Rejected;
    };
    let Some(nb13) = tmatmul_read_param_u64(kernel_params, 11).map(|v| v as usize) else {
        return KernelNameFallbackStatus::Rejected;
    };
    let Some(nbd2) = tmatmul_read_param_u64(kernel_params, 12).map(|v| v as usize) else {
        return KernelNameFallbackStatus::Rejected;
    };
    let Some(nbd3) = tmatmul_read_param_u64(kernel_params, 13).map(|v| v as usize) else {
        return KernelNameFallbackStatus::Rejected;
    };
    let Some(r2) = tmatmul_read_param_i64(kernel_params, 14).map(|v| v.max(1) as usize) else {
        return KernelNameFallbackStatus::Rejected;
    };
    let Some(r3) = tmatmul_read_param_i64(kernel_params, 15).map(|v| v.max(1) as usize) else {
        return KernelNameFallbackStatus::Rejected;
    };

    if ne12 == 0 || ne13 == 0 || ne23 == 0 {
        return KernelNameFallbackStatus::Handled;
    }

    let ptrs_src_host = ptrs_src as *mut u64;
    let ptrs_dst_host = ptrs_dst as *mut u64;
    if ptrs_src_host.is_null() || ptrs_dst_host.is_null() {
        eprintln!(
            "[TMatmul Fallback] compute_batched_ptrs '{}' has null pointer tables",
            kernel_name
        );
        return KernelNameFallbackStatus::Rejected;
    }

    let Some(table_count) = ne12.checked_mul(ne13) else {
        eprintln!(
            "[TMatmul Fallback] compute_batched_ptrs '{}' table size overflow ne12={} ne13={}",
            kernel_name, ne12, ne13
        );
        return KernelNameFallbackStatus::Rejected;
    };
    let Some(ptrs_src_count) = ne23.checked_add(table_count) else {
        eprintln!(
            "[TMatmul Fallback] compute_batched_ptrs '{}' src table size overflow ne23={} table_count={}",
            kernel_name, ne23, table_count
        );
        return KernelNameFallbackStatus::Rejected;
    };

    if !tmatmul_pointer_table_has_elems(ptrs_src_host as *const u64, ptrs_src_count)
        || !tmatmul_pointer_table_has_elems(ptrs_dst_host as *const u64, table_count)
    {
        eprintln!(
            "[TMatmul Fallback] compute_batched_ptrs '{}' rejected pointer table ranges ptrs_src={:p} ptrs_dst={:p} src_count={} dst_count={} ne12={} ne13={} ne23={}",
            kernel_name,
            ptrs_src_host,
            ptrs_dst_host,
            ptrs_src_count,
            table_count,
            ne12,
            ne13,
            ne23
        );
        return KernelNameFallbackStatus::Rejected;
    }

    for i13 in 0..ne13 {
        for i12 in 0..ne12 {
            let i03 = i13 / r3;
            let i02 = i12 / r2;
            let index = i12 + i13 * ne12;
            ptrs_src_host
                .add(index)
                .write_unaligned(src0.saturating_add((i02 * nb02 + i03 * nb03) as u64));
            ptrs_src_host
                .add(ne23 + index)
                .write_unaligned(src1.saturating_add((i12 * nb12 + i13 * nb13) as u64));
            ptrs_dst_host
                .add(index)
                .write_unaligned(dst.saturating_add((i12 * nbd2 + i13 * nbd3) as u64));
        }
    }

    if std::env::var("HETGPU_TMATMUL_LOG_COMPUTE_BATCHED_PTRS")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!(
            "[TMatmul Fallback] handled compute_batched_ptrs '{}' ne12={} ne13={} ne23={}",
            kernel_name, ne12, ne13, ne23
        );
    }
    KernelNameFallbackStatus::Handled
}

/// Execute a reduce_kernel fallback (sum, mean, max, var).
/// The reduce_kernel has a single ReduceOp struct parameter containing embedded pointers.
#[cfg(feature = "intel")]
unsafe fn execute_reduce_kernel_fallback(
    kernel_name: &str,
    name_lower: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) {
    // reduce_kernel takes a single ReduceOp parameter
    // kernel_params[0] = &ReduceOp (struct containing input/output pointers)
    let op_param = *kernel_params.add(0);
    if op_param.is_null() {
        eprintln!("[TMatmul Fallback] reduce_kernel: null parameter");
        return;
    }

    // Scan the ReduceOp struct for valid allocation pointers
    // The struct is typically < 512 bytes
    let scan_size = 512;
    let found_ptrs = scan_for_alloc_pointers(op_param as *const u8, scan_size);

    if found_ptrs.len() < 2 {
        eprintln!(
            "[TMatmul Fallback] reduce_kernel: found only {} alloc pointers (need >=2)",
            found_ptrs.len()
        );
        return;
    }

    // Heuristic: in PyTorch's ReduceOp struct layout, output pointer comes before input pointer.
    // The output is smaller (reduced dimension), input is larger.
    // Sort by alloc_size to identify: smallest = output, largest = input
    let mut sorted = found_ptrs.clone();
    sorted.sort_by_key(|&(_, _, size)| size);

    let (_, out_ptr, out_size) = sorted[0]; // smallest alloc = output
    let (_, in_ptr, in_size) = sorted[sorted.len() - 1]; // largest alloc = input

    let in_elements = in_size / 4; // f32
    let out_elements = out_size / 4;

    if out_elements == 0 || in_elements == 0 {
        eprintln!(
            "[TMatmul Fallback] reduce_kernel: zero elements (in={}, out={})",
            in_elements, out_elements
        );
        return;
    }

    // Determine reduction size: in_elements / out_elements
    let reduce_size = if out_elements > 0 {
        in_elements / out_elements
    } else {
        in_elements
    };

    let inp = in_ptr as *const f32;
    let out = out_ptr as *mut f32;

    // Determine which reduction operation
    let op_type = if name_lower.contains("sum_functor") || name_lower.contains("sum") {
        "sum"
    } else if name_lower.contains("meanops") || name_lower.contains("mean") {
        "mean"
    } else if name_lower.contains("maxops") || name_lower.contains("max") {
        "max"
    } else if name_lower.contains("minops") || name_lower.contains("min") {
        "min"
    } else if name_lower.contains("normops") || name_lower.contains("var") {
        "var"
    } else {
        "sum" // default
    };

    eprintln!(
        "[TMatmul Fallback] reduce_kernel: op={}, in_elements={}, out_elements={}, reduce_size={}",
        op_type, in_elements, out_elements, reduce_size
    );

    for row in 0..out_elements {
        let base = row * reduce_size;
        let end = (base + reduce_size).min(in_elements);
        let count = end - base;
        if count == 0 {
            continue;
        }

        match op_type {
            "sum" => {
                let mut sum = 0.0f32;
                for i in base..end {
                    sum += inp.add(i).read_unaligned();
                }
                out.add(row).write_unaligned(sum);
            }
            "mean" => {
                let mut sum = 0.0f32;
                for i in base..end {
                    sum += inp.add(i).read_unaligned();
                }
                out.add(row).write_unaligned(sum / count as f32);
            }
            "max" => {
                let mut max_val = f32::NEG_INFINITY;
                for i in base..end {
                    let v = inp.add(i).read_unaligned();
                    if v > max_val {
                        max_val = v;
                    }
                }
                out.add(row).write_unaligned(max_val);
            }
            "min" => {
                let mut min_val = f32::INFINITY;
                for i in base..end {
                    let v = inp.add(i).read_unaligned();
                    if v < min_val {
                        min_val = v;
                    }
                }
                out.add(row).write_unaligned(min_val);
            }
            "var" => {
                let mut sum = 0.0f32;
                for i in base..end {
                    sum += inp.add(i).read_unaligned();
                }
                let mean = sum / count as f32;
                let mut var_sum = 0.0f32;
                for i in base..end {
                    let diff = inp.add(i).read_unaligned() - mean;
                    var_sum += diff * diff;
                }
                // Unbiased variance (Bessel's correction)
                let divisor = if count > 1 {
                    (count - 1) as f32
                } else {
                    count as f32
                };
                out.add(row).write_unaligned(var_sum / divisor);
            }
            _ => {}
        }
    }
    eprintln!(
        "[TMatmul Fallback] reduce_kernel '{}' executed ({} -> {} elements)",
        op_type, in_elements, out_elements
    );
}

/// Execute a softmax kernel fallback.
#[cfg(feature = "intel")]
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct TmatmulSoftMaxParams {
    nheads: i64,
    n_head_log2: u32,
    _pad0: u32,
    ncols: i64,
    nrows_x: i64,
    nrows_y: i64,
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb11: i64,
    nb12: i64,
    nb13: i64,
    ne12: i64,
    ne13: i64,
    scale: f32,
    max_bias: f32,
    m0: f32,
    m1: f32,
}

#[cfg(feature = "intel")]
fn tmatmul_alibi_slope(max_bias: f32, h: u32, n_head_log2: u32, m0: f32, m1: f32) -> f32 {
    if max_bias <= 0.0 {
        return 1.0;
    }
    let (base, exp) = if h < n_head_log2 {
        (m0, h + 1)
    } else {
        (m1, 2 * (h - n_head_log2) + 1)
    };
    base.powi(exp as i32)
}

#[cfg(feature = "intel")]
unsafe fn try_execute_ggml_softmax_f32_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> bool {
    let Some(x_addr) = tmatmul_read_param_u64(kernel_params, 0) else {
        return false;
    };
    let mask_addr = tmatmul_read_param_u64(kernel_params, 1).unwrap_or(0);
    let sinks_addr = tmatmul_read_param_u64(kernel_params, 2).unwrap_or(0);
    let Some(dst_addr) = tmatmul_read_param_u64(kernel_params, 3) else {
        return false;
    };
    if kernel_params.is_null() {
        return false;
    }
    let params_ptr = *kernel_params.add(4) as *const TmatmulSoftMaxParams;
    if params_ptr.is_null()
        || !tmatmul_process_range_has_perms(
            params_ptr as usize,
            std::mem::size_of::<TmatmulSoftMaxParams>(),
            false,
        )
    {
        return false;
    }
    let p = params_ptr.read_unaligned();
    if p.ncols < 0 || p.ne01 < 0 || p.ne02 < 0 || p.ne03 < 0 {
        return false;
    }
    let ncols = match usize::try_from(p.ncols) {
        Ok(value) => value,
        Err(_) => return false,
    };
    let ne01 = match usize::try_from(p.ne01) {
        Ok(value) => value,
        Err(_) => return false,
    };
    let ne02 = match usize::try_from(p.ne02) {
        Ok(value) => value,
        Err(_) => return false,
    };
    let ne03 = match usize::try_from(p.ne03) {
        Ok(value) => value,
        Err(_) => return false,
    };
    let Some(rows) = ne01
        .checked_mul(ne02)
        .and_then(|value| value.checked_mul(ne03))
    else {
        return false;
    };
    if ncols == 0 || rows == 0 {
        return true;
    }
    let Some(bytes) = rows
        .checked_mul(ncols)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
    else {
        return false;
    };
    if x_addr == 0
        || dst_addr == 0
        || !tmatmul_host_or_virtual_alloc_has_bytes(x_addr, bytes, false)
        || !tmatmul_host_or_virtual_alloc_has_bytes(dst_addr, bytes, true)
    {
        eprintln!(
            "[TMatmul Fallback] ggml softmax '{}' rejected x/dst range x=0x{:x} dst=0x{:x} rows={} cols={}",
            kernel_name, x_addr, dst_addr, rows, ncols
        );
        return false;
    }

    if sinks_addr != 0 {
        let Some(sinks_bytes) = ne02.checked_mul(std::mem::size_of::<f32>()) else {
            return false;
        };
        if !tmatmul_host_or_virtual_alloc_has_bytes(sinks_addr, sinks_bytes, false) {
            eprintln!(
                "[TMatmul Fallback] ggml softmax '{}' rejected sinks range sinks=0x{:x} ne02={}",
                kernel_name, sinks_addr, ne02
            );
            return false;
        }
    }

    let mask_is_f16 = kernel_name.contains("__half") || kernel_name.contains("6__half");
    let mask_elem_size = if mask_is_f16 { 2usize } else { 4usize };
    if mask_addr != 0 {
        if p.nb11 < 0 || p.nb12 < 0 || p.nb13 < 0 || p.ne12 < 0 || p.ne13 < 0 {
            return false;
        }
        let ne12 = usize::try_from(p.ne12).ok().unwrap_or(0).max(1);
        let ne13 = usize::try_from(p.ne13).ok().unwrap_or(0).max(1);
        let Some(max_mask_off) = ((ne01.saturating_sub(1)) as i128)
            .checked_mul(p.nb11 as i128)
            .and_then(|value| {
                value.checked_add(((ne12.saturating_sub(1)) as i128) * p.nb12 as i128)
            })
            .and_then(|value| {
                value.checked_add(((ne13.saturating_sub(1)) as i128) * p.nb13 as i128)
            })
        else {
            return false;
        };
        if max_mask_off < 0 {
            return false;
        }
        let Some(mask_bytes) = usize::try_from(max_mask_off)
            .ok()
            .and_then(|value| value.checked_add(ncols.checked_mul(mask_elem_size)?))
        else {
            return false;
        };
        if !tmatmul_host_or_virtual_alloc_has_bytes(mask_addr, mask_bytes, false) {
            eprintln!(
                "[TMatmul Fallback] ggml softmax '{}' rejected mask range mask=0x{:x} bytes={}",
                kernel_name, mask_addr, mask_bytes
            );
            return false;
        }
    }

    let x = x_addr as *const f32;
    let dst = dst_addr as *mut f32;
    let sinks = sinks_addr as *const f32;
    let mask = mask_addr as *const u8;
    let mut vals = vec![0.0f32; ncols];

    for i03 in 0..ne03 {
        for i02 in 0..ne02 {
            for i01 in 0..ne01 {
                let rowx = i01 + i02 * ne01 + i03 * ne01 * ne02;
                let x_row = x.add(rowx * ncols);
                let dst_row = dst.add(rowx * ncols);
                let mut max_val = if sinks_addr == 0 {
                    f32::NEG_INFINITY
                } else {
                    sinks.add(i02).read_unaligned()
                };
                let slope = tmatmul_alibi_slope(p.max_bias, i02 as u32, p.n_head_log2, p.m0, p.m1);
                let mask_row = if mask_addr == 0 {
                    std::ptr::null()
                } else {
                    let ne12 = usize::try_from(p.ne12).ok().unwrap_or(0).max(1);
                    let ne13 = usize::try_from(p.ne13).ok().unwrap_or(0).max(1);
                    let i12 = i02 % ne12;
                    let i13 = i03 % ne13;
                    let Some(byte_off) = (i01 as i64)
                        .checked_mul(p.nb11)
                        .and_then(|value| value.checked_add((i12 as i64).checked_mul(p.nb12)?))
                        .and_then(|value| value.checked_add((i13 as i64).checked_mul(p.nb13)?))
                    else {
                        return false;
                    };
                    if byte_off < 0 {
                        return false;
                    }
                    mask.add(byte_off as usize)
                };

                for col in 0..ncols {
                    let mask_v = if mask_row.is_null() {
                        0.0
                    } else if mask_is_f16 {
                        f16_to_f32((mask_row as *const u16).add(col).read_unaligned())
                    } else {
                        (mask_row as *const f32).add(col).read_unaligned()
                    };
                    let val = x_row.add(col).read_unaligned() * p.scale + slope * mask_v;
                    vals[col] = val;
                    max_val = max_val.max(val);
                }

                let mut sum = if sinks_addr == 0 {
                    0.0
                } else {
                    (sinks.add(i02).read_unaligned() - max_val).exp()
                };
                for value in &mut vals {
                    *value = (*value - max_val).exp();
                    sum += *value;
                }
                let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for (col, value) in vals.iter().enumerate() {
                    dst_row.add(col).write_unaligned(*value * inv_sum);
                }
            }
        }
    }

    eprintln!(
        "[TMatmul Fallback] ggml softmax '{}' executed rows={} cols={} mask={} sinks={}",
        kernel_name,
        rows,
        ncols,
        mask_addr != 0,
        sinks_addr != 0
    );
    true
}

#[cfg(feature = "intel")]
unsafe fn execute_softmax_kernel_fallback(
    kernel_name: &str,
    name_lower: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) {
    if name_lower.contains("soft_max_f32") || name_lower.contains("softmax_f32") {
        if try_execute_ggml_softmax_f32_fallback(kernel_name, kernel_params) {
            return;
        }
        eprintln!(
            "[TMatmul Fallback] ggml softmax '{}' not handled; refusing legacy softmax parser",
            kernel_name
        );
        return;
    }

    // softmax_warp_forward signature:
    //   (output_t *dst, const input_t *src, int batch_size, int stride, int element_count, ...)
    // kernel_params[0] = &dst, [1] = &src, [2] = &batch_size, [3] = &stride, [4] = &element_count

    let param0 = *kernel_params.add(0);
    let param1 = *kernel_params.add(1);
    if param0.is_null() || param1.is_null() {
        eprintln!("[TMatmul Fallback] softmax: null dst or src parameter");
        return;
    }

    let dst_ptr_val = (param0 as *const u64).read_unaligned();
    let src_ptr_val = (param1 as *const u64).read_unaligned();

    let dst_size = super::memory::get_alloc_size(dst_ptr_val as usize);
    let src_size = super::memory::get_alloc_size(src_ptr_val as usize);

    if dst_size.is_none() || src_size.is_none() {
        eprintln!(
            "[TMatmul Fallback] softmax: dst or src not in alloc map (dst={:#x}, src={:#x})",
            dst_ptr_val, src_ptr_val
        );
        return;
    }

    // Read dimension params
    let param2 = *kernel_params.add(2);
    let param3 = *kernel_params.add(3);
    let param4 = *kernel_params.add(4);

    let batch_size = if !param2.is_null() {
        (param2 as *const i32).read_unaligned() as usize
    } else {
        1
    };
    let stride = if !param3.is_null() {
        (param3 as *const i32).read_unaligned() as usize
    } else {
        1
    };
    let element_count = if !param4.is_null() {
        (param4 as *const i32).read_unaligned() as usize
    } else {
        // Fallback: infer from allocation size
        let total = src_size.unwrap() / 4;
        if batch_size > 0 {
            total / batch_size
        } else {
            total
        }
    };

    let rows = batch_size;
    let cols = element_count;

    if rows == 0 || cols == 0 {
        eprintln!(
            "[TMatmul Fallback] softmax: invalid dims (rows={}, cols={})",
            rows, cols
        );
        return;
    }

    execute_softmax_on_data(
        src_ptr_val as *const f32,
        dst_ptr_val as *mut f32,
        rows,
        cols,
    );
    eprintln!(
        "[TMatmul Fallback] softmax executed ({}x{} = {} elements) for '{}'",
        rows,
        cols,
        rows * cols,
        kernel_name
    );
}

#[cfg(feature = "intel")]
unsafe fn execute_softmax_on_data(inp: *const f32, out: *mut f32, rows: usize, cols: usize) {
    for row in 0..rows {
        let base = row * cols;
        // Find max for numerical stability
        let mut max_val = f32::NEG_INFINITY;
        for c in 0..cols {
            let v = inp.add(base + c).read_unaligned();
            if v > max_val {
                max_val = v;
            }
        }
        // Compute exp(x - max) and sum
        let mut sum = 0.0f32;
        for c in 0..cols {
            let v = inp.add(base + c).read_unaligned();
            let e = (v - max_val).exp();
            out.add(base + c).write_unaligned(e);
            sum += e;
        }
        // Normalize
        if sum > 0.0 {
            for c in 0..cols {
                let e = out.add(base + c).read_unaligned();
                out.add(base + c).write_unaligned(e / sum);
            }
        }
    }
}

#[cfg(feature = "intel")]
unsafe fn execute_tmatmul_bin_bcast_f32_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> bool {
    #[derive(Copy, Clone)]
    enum Op {
        Repeat,
        Add,
        Sub,
        Mul,
        Div,
    }

    let op = if kernel_name.contains("op_repeat") {
        Op::Repeat
    } else if kernel_name.contains("op_add") {
        Op::Add
    } else if kernel_name.contains("op_sub") {
        Op::Sub
    } else if kernel_name.contains("op_mul") {
        Op::Mul
    } else if kernel_name.contains("op_div") {
        Op::Div
    } else {
        return false;
    };

    if kernel_name.contains("k_bin_bcast_unravel") {
        return false;
    }

    let src0_addr = tmatmul_read_param_u64(kernel_params, 0).unwrap_or(0);
    let Some(src1_addr) = tmatmul_read_param_u64(kernel_params, 1) else {
        return false;
    };
    let Some(dst_addr) = tmatmul_read_param_u64(kernel_params, 2) else {
        return false;
    };

    let Some(ne0) = tmatmul_read_param_i32(kernel_params, 3).map(|v| v.max(0) as usize) else {
        return false;
    };
    let Some(ne1) = tmatmul_read_param_i32(kernel_params, 4).map(|v| v.max(0) as usize) else {
        return false;
    };
    let Some(ne2) = tmatmul_read_param_i32(kernel_params, 5).map(|v| v.max(0) as usize) else {
        return false;
    };
    let Some(ne3) = tmatmul_read_param_i32(kernel_params, 6).map(|v| v.max(0) as usize) else {
        return false;
    };
    let Some(ne10) = tmatmul_read_param_i32(kernel_params, 7).map(|v| v.max(1) as usize) else {
        return false;
    };
    let Some(ne11) = tmatmul_read_param_i32(kernel_params, 8).map(|v| v.max(1) as usize) else {
        return false;
    };
    let Some(ne12) = tmatmul_read_param_i32(kernel_params, 9).map(|v| v.max(1) as usize) else {
        return false;
    };
    let Some(ne13) = tmatmul_read_param_i32(kernel_params, 10).map(|v| v.max(1) as usize) else {
        return false;
    };

    let read_stride = |index| -> Option<usize> {
        tmatmul_read_param_i32(kernel_params, index).map(|v| v.max(0) as usize)
    };
    let Some(s1) = read_stride(11) else {
        return false;
    };
    let Some(s2) = read_stride(12) else {
        return false;
    };
    let Some(s3) = read_stride(13) else {
        return false;
    };
    let Some(s01) = read_stride(14) else {
        return false;
    };
    let Some(s02) = read_stride(15) else {
        return false;
    };
    let Some(s03) = read_stride(16) else {
        return false;
    };
    let Some(s11) = read_stride(17) else {
        return false;
    };
    let Some(s12) = read_stride(18) else {
        return false;
    };
    let Some(s13) = read_stride(19) else {
        return false;
    };
    let s00 = 1usize;
    let s10 = 1usize;

    if ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0 {
        return true;
    }

    let checked_extent =
        |a: usize, sa: usize, b: usize, sb: usize, c: usize, sc: usize, d: usize, sd: usize| {
            a.checked_sub(1)
                .and_then(|v| v.checked_mul(sa))
                .and_then(|v| {
                    b.checked_sub(1)
                        .and_then(|w| w.checked_mul(sb))
                        .and_then(|w| v.checked_add(w))
                })
                .and_then(|v| {
                    c.checked_sub(1)
                        .and_then(|w| w.checked_mul(sc))
                        .and_then(|w| v.checked_add(w))
                })
                .and_then(|v| {
                    d.checked_sub(1)
                        .and_then(|w| w.checked_mul(sd))
                        .and_then(|w| v.checked_add(w))
                })
                .and_then(|v| v.checked_add(1))
        };
    let Some(dst_elems) = checked_extent(ne3, s3, ne2, s2, ne1, s1, ne0, 1) else {
        return false;
    };
    let Some(src1_elems) =
        checked_extent(ne13, s13, ne12, s12, ne11, s11, ne0.min(ne10).max(1), s10)
    else {
        return false;
    };
    let src0_elems = if src0_addr == 0 {
        0
    } else {
        let Some(extent) = checked_extent(ne3, s03, ne2, s02, ne1, s01, ne0, s00) else {
            return false;
        };
        extent
    };

    let elem_size = std::mem::size_of::<f32>();
    let Some(dst_bytes) = dst_elems.checked_mul(elem_size) else {
        return false;
    };
    let Some(src1_bytes) = src1_elems.checked_mul(elem_size) else {
        return false;
    };
    let Some(src0_bytes) = src0_elems.checked_mul(elem_size) else {
        return false;
    };
    if dst_addr == 0
        || src1_addr == 0
        || !tmatmul_host_or_virtual_alloc_has_bytes(dst_addr, dst_bytes, true)
        || !tmatmul_host_or_virtual_alloc_has_bytes(src1_addr, src1_bytes, false)
        || (src0_addr != 0
            && !tmatmul_host_or_virtual_alloc_has_bytes(src0_addr, src0_bytes, false))
    {
        eprintln!(
            "[TMatmul Fallback] bin_bcast '{}' rejected range dst=0x{:x}/{} src0=0x{:x}/{} src1=0x{:x}/{}",
            kernel_name, dst_addr, dst_bytes, src0_addr, src0_bytes, src1_addr, src1_bytes
        );
        return false;
    }

    let src0 = src0_addr as *const f32;
    let src1 = src1_addr as *const f32;
    let dst = dst_addr as *mut f32;
    let apply = |op: Op, a: f32, b: f32| -> f32 {
        match op {
            Op::Repeat => b,
            Op::Add => a + b,
            Op::Sub => a - b,
            Op::Mul => a * b,
            Op::Div => a / b,
        }
    };

    for row in 0..(ne1 * ne2 * ne3) {
        let i1 = row % ne1;
        let t = row / ne1;
        let i2 = t % ne2;
        let i3 = t / ne2;
        let i11 = i1 % ne11;
        let i12 = i2 % ne12;
        let i13 = i3 % ne13;
        let src0_base = i3 * s03 + i2 * s02 + i1 * s01;
        let src1_base = i13 * s13 + i12 * s12 + i11 * s11;
        let dst_base = i3 * s3 + i2 * s2 + i1 * s1;
        for i0 in 0..ne0 {
            let i10 = i0 % ne10;
            let src1_off = src1_base + i10 * s10;
            let left = if src0_addr == 0 {
                0.0
            } else {
                src0.add(src0_base + i0 * s00).read_unaligned()
            };
            let right = src1.add(src1_off).read_unaligned();
            dst.add(dst_base + i0)
                .write_unaligned(apply(op, left, right));
        }
    }

    eprintln!(
        "[TMatmul Fallback] bin_bcast '{}' executed ne=({}, {}, {}, {})",
        kernel_name, ne0, ne1, ne2, ne3
    );
    true
}

#[cfg(feature = "intel")]
fn tmatmul_parse_digits_after(haystack: &str, marker: &str) -> Option<usize> {
    let start = haystack.find(marker)?.checked_add(marker.len())?;
    let bytes = haystack.as_bytes();
    let mut end = start;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    (end > start)
        .then(|| haystack[start..end].parse::<usize>().ok())
        .flatten()
}

#[cfg(feature = "intel")]
fn tmatmul_ggml_qk_from_kernel_name(kernel_name: &str) -> Option<usize> {
    match tmatmul_parse_digits_after(kernel_name, "ggml_type")? {
        // Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 use 32-element blocks.
        2 | 3 | 6 | 7 | 8 | 9 => Some(32),
        // IQ4_NL is a non-linear 4-bit type with QK4_NL=32.
        20 => Some(32),
        // K-quants and the remaining IQ quants in this BitNet llama.cpp tree use QK_K=256.
        10..=19 | 21..=23 => Some(256),
        _ => None,
    }
}

#[cfg(feature = "intel")]
fn tmatmul_stream_k_fixup_template_params(kernel_name: &str) -> Option<(usize, usize, bool)> {
    let template_start = kernel_name.find("ggml_type")?;
    let template = &kernel_name[template_start..];
    let mut li_values = Vec::with_capacity(2);
    let mut rest = template;
    while let Some(pos) = rest.find("Li") {
        let start = pos + 2;
        let bytes = rest.as_bytes();
        let mut end = start;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
        if end > start {
            if let Ok(value) = rest[start..end].parse::<usize>() {
                li_values.push(value);
                if li_values.len() == 2 {
                    break;
                }
            }
        }
        rest = &rest[end.min(rest.len())..];
    }
    let mmq_x = *li_values.get(0)?;
    let nwarps = *li_values.get(1)?;
    if mmq_x == 0 || nwarps == 0 {
        return None;
    }
    Some((mmq_x, nwarps, template.contains("ELb1")))
}

#[cfg(feature = "intel")]
unsafe fn execute_tmatmul_mul_mat_q_stream_k_fixup_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: u32,
    grid_dim_y: u32,
) -> bool {
    const MMQ_ITER_K: usize = 256;
    const MMQ_Y_SM120: usize = 128;

    let Some(qk) = tmatmul_ggml_qk_from_kernel_name(kernel_name) else {
        return false;
    };
    let Some((mmq_x, _nwarps, need_check)) = tmatmul_stream_k_fixup_template_params(kernel_name)
    else {
        return false;
    };
    if qk == 0 || MMQ_ITER_K % qk != 0 {
        return false;
    }
    let blocks_per_iter = MMQ_ITER_K / qk;
    let mmq_y = MMQ_Y_SM120;
    let grid_x = grid_dim_x.max(1) as usize;
    let grid_y = grid_dim_y.max(1) as usize;

    let Some(dst_addr) = tmatmul_read_param_u64(kernel_params, 0) else {
        return false;
    };
    let Some(tmp_addr) = tmatmul_read_param_u64(kernel_params, 1) else {
        return false;
    };
    let Some(ne00) = tmatmul_read_param_i32(kernel_params, 2) else {
        return false;
    };
    let Some(ne01) = tmatmul_read_param_i32(kernel_params, 3) else {
        return false;
    };
    let Some(ne11) = tmatmul_read_param_i32(kernel_params, 4) else {
        return false;
    };
    let Some(ne0) = tmatmul_read_param_i32(kernel_params, 5) else {
        return false;
    };
    let Some(block_num_mmq) = tmatmul_read_param_i32(kernel_params, 6) else {
        return false;
    };
    if dst_addr == 0 || tmp_addr == 0 || ne00 <= 0 || ne0 <= 0 || block_num_mmq <= 0 {
        return false;
    }
    if ne01 <= 0 || ne11 <= 0 {
        return true;
    }
    let Ok(ne00) = usize::try_from(ne00) else {
        return false;
    };
    let Ok(ne01) = usize::try_from(ne01) else {
        return false;
    };
    let Ok(ne11) = usize::try_from(ne11) else {
        return false;
    };
    let Ok(ne0) = usize::try_from(ne0) else {
        return false;
    };
    let Ok(block_num_mmq) = usize::try_from(block_num_mmq) else {
        return false;
    };
    let blocks_per_ne00 = ne00 / qk;
    if blocks_per_ne00 == 0 {
        return false;
    }
    let Some(ntx) = ne11.checked_add(mmq_x - 1).map(|v| v / mmq_x) else {
        return false;
    };
    let Some(nty) = ne01.checked_add(mmq_y - 1).map(|v| v / mmq_y) else {
        return false;
    };
    if ntx == 0 || nty == 0 {
        return true;
    }
    let Some(tile_elems) = mmq_x.checked_mul(mmq_y) else {
        return false;
    };
    let Some(tmp_bytes) = block_num_mmq
        .checked_mul(tile_elems)
        .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
    else {
        return false;
    };
    if !tmatmul_host_or_virtual_alloc_has_bytes(tmp_addr, tmp_bytes, false) {
        eprintln!(
            "[TMatmul Fallback] stream_k_fixup '{}' rejected tmp range tmp=0x{:x}/{}",
            kernel_name, tmp_addr, tmp_bytes
        );
        return false;
    }

    let mut max_dst_elems = 0usize;
    for block_y in 0..grid_y {
        let row_start = block_y.saturating_mul(mmq_x);
        if row_start >= ne11 {
            continue;
        }
        let rows = mmq_x.min(ne11 - row_start);
        for block_x in 0..grid_x {
            let col_start = block_x.saturating_mul(mmq_y);
            let cols = if need_check {
                if col_start >= ne01 {
                    continue;
                }
                mmq_y.min(ne01 - col_start)
            } else {
                mmq_y
            };
            let Some(end_elem) = row_start
                .checked_mul(ne0)
                .and_then(|v| v.checked_add(col_start))
                .and_then(|v| v.checked_add((rows - 1).saturating_mul(ne0)))
                .and_then(|v| v.checked_add(cols))
            else {
                return false;
            };
            max_dst_elems = max_dst_elems.max(end_elem);
        }
    }
    let Some(dst_bytes) = max_dst_elems.checked_mul(std::mem::size_of::<f32>()) else {
        return false;
    };
    if !tmatmul_host_or_virtual_alloc_has_bytes(dst_addr, dst_bytes, true) {
        eprintln!(
            "[TMatmul Fallback] stream_k_fixup '{}' rejected dst range dst=0x{:x}/{}",
            kernel_name, dst_addr, dst_bytes
        );
        return false;
    }

    let dst = dst_addr as *mut f32;
    let tmp = tmp_addr as *const f32;
    let denom = grid_y.checked_mul(grid_x).unwrap_or(0);
    if denom == 0 {
        return false;
    }
    let mut fixup_tiles = 0usize;
    let mut blocks_with_fixup = 0usize;

    for block_y in 0..grid_y {
        for block_x in 0..grid_x {
            let Some(tile_linear) = block_y
                .checked_mul(nty)
                .and_then(|v| v.checked_add(block_x))
            else {
                return false;
            };
            let bidx_start = tile_linear.saturating_mul(block_num_mmq) / denom;
            let bidx_stop = tile_linear
                .checked_add(1)
                .and_then(|v| v.checked_mul(block_num_mmq))
                .and_then(|v| v.checked_add(denom - 1))
                .map(|v| v / denom)
                .unwrap_or(block_num_mmq)
                .min(block_num_mmq);
            let mut kbc_stop_0 = bidx_start
                .checked_mul(blocks_per_ne00)
                .and_then(|v| v.checked_mul(ntx))
                .and_then(|v| v.checked_mul(nty))
                .map(|v| v / block_num_mmq)
                .unwrap_or(0);
            let mut sums = vec![0.0f32; tile_elems];
            let mut any_fixup = false;

            for bidx in bidx_start..bidx_stop {
                let kbc_0 = kbc_stop_0;
                let Some(next_kbc_stop_0) = bidx
                    .checked_add(1)
                    .and_then(|v| v.checked_mul(blocks_per_ne00))
                    .and_then(|v| v.checked_mul(ntx))
                    .and_then(|v| v.checked_mul(nty))
                    .map(|v| v / block_num_mmq)
                else {
                    return false;
                };
                kbc_stop_0 = next_kbc_stop_0;

                let kbc = kbc_0 - (kbc_0 % blocks_per_ne00) % blocks_per_iter;
                let kbc_stop = kbc_stop_0 - (kbc_stop_0 % blocks_per_ne00) % blocks_per_iter;
                if kbc == kbc_stop || kbc_stop % blocks_per_ne00 == 0 {
                    continue;
                }

                let jt = kbc_stop / (blocks_per_ne00 * nty);
                let it = (kbc_stop - jt * (blocks_per_ne00 * nty)) / blocks_per_ne00;
                if it != block_x || jt != block_y {
                    continue;
                }

                any_fixup = true;
                fixup_tiles += 1;
                let tmp_tile = tmp.add(bidx * tile_elems);
                for j in 0..mmq_x {
                    for i in 0..mmq_y {
                        let off = j * mmq_y + i;
                        sums[off] += tmp_tile.add(off).read_unaligned();
                    }
                }
            }

            if !any_fixup {
                continue;
            }
            blocks_with_fixup += 1;
            let row_start = block_y * mmq_x;
            if row_start >= ne11 {
                continue;
            }
            let j_limit = mmq_x.min(ne11 - row_start);
            let col_start = block_x * mmq_y;
            let i_limit = if need_check {
                if col_start >= ne01 {
                    0
                } else {
                    mmq_y.min(ne01 - col_start)
                }
            } else {
                mmq_y
            };
            let dst_base = row_start * ne0 + col_start;
            for j in 0..j_limit {
                for i in 0..i_limit {
                    let dst_off = dst_base + j * ne0 + i;
                    let old = dst.add(dst_off).read_unaligned();
                    dst.add(dst_off).write_unaligned(old + sums[j * mmq_y + i]);
                }
            }
        }
    }

    eprintln!(
        "[TMatmul Fallback] stream_k_fixup '{}' executed grid=({}, {}) qk={} mmq=({}, {}) fixup_tiles={} blocks={}",
        kernel_name, grid_x, grid_y, qk, mmq_x, mmq_y, fixup_tiles, blocks_with_fixup
    );
    true
}

#[cfg(feature = "intel")]
unsafe fn execute_tmatmul_quantize_mmq_q8_1_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    _grid_dim_x: u32,
    _grid_dim_y: u32,
    grid_dim_z: u32,
) -> bool {
    #[derive(Clone, Copy, Eq, PartialEq)]
    enum Layout {
        D4,
        Ds4,
        D2s6,
    }

    let layout = if kernel_name.contains("ds_layout0") {
        Layout::D4
    } else if kernel_name.contains("ds_layout1") {
        Layout::Ds4
    } else if kernel_name.contains("ds_layout2") {
        Layout::D2s6
    } else {
        return false;
    };

    let Some(x_addr) = tmatmul_read_param_u64(kernel_params, 0) else {
        return false;
    };
    let Some(y_addr) = tmatmul_read_param_u64(kernel_params, 1) else {
        return false;
    };
    let Some(kx0) = tmatmul_read_param_i64(kernel_params, 2) else {
        return false;
    };
    let Some(kx1) = tmatmul_read_param_i64(kernel_params, 3) else {
        return false;
    };
    let Some(kx0_padded) = tmatmul_read_param_i64(kernel_params, 4) else {
        return false;
    };

    if x_addr == 0 || y_addr == 0 || kx0 < 0 || kx1 <= 0 || kx0_padded <= 0 {
        return false;
    }
    let Ok(kx0) = usize::try_from(kx0) else {
        return false;
    };
    let Ok(kx1) = usize::try_from(kx1) else {
        return false;
    };
    let Ok(kx0_padded) = usize::try_from(kx0_padded) else {
        return false;
    };
    let channels = grid_dim_z.max(1) as usize;
    let Some(blocks_per_row) = kx0_padded.checked_add(127).map(|v| v / 128) else {
        return false;
    };
    let block_bytes = 144usize;
    let Some(x_bytes) = channels
        .checked_mul(kx1)
        .and_then(|v| v.checked_mul(kx0))
        .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
    else {
        return false;
    };
    let Some(y_bytes) = channels
        .checked_mul(kx1)
        .and_then(|v| v.checked_mul(blocks_per_row))
        .and_then(|v| v.checked_mul(block_bytes))
    else {
        return false;
    };
    if !tmatmul_host_or_virtual_alloc_has_bytes(x_addr, x_bytes, false)
        || !tmatmul_host_or_virtual_alloc_has_bytes(y_addr, y_bytes, true)
    {
        eprintln!(
            "[TMatmul Fallback] quantize_mmq_q8_1 '{}' rejected range x=0x{:x}/{} y=0x{:x}/{}",
            kernel_name, x_addr, x_bytes, y_addr, y_bytes
        );
        return false;
    }

    let x = x_addr as *const f32;
    let y = y_addr as *mut u8;
    let quantize = |value: f32, d: f32| -> i8 {
        if d == 0.0 {
            0
        } else {
            (value / d).round().clamp(-128.0, 127.0) as i8
        }
    };
    let read_value = |row: usize, col: usize| -> f32 {
        if col < kx0 {
            x.add(row * kx0 + col).read_unaligned()
        } else {
            0.0
        }
    };

    for channel in 0..channels {
        for row in 0..kx1 {
            let src_row = channel * kx1 + row;
            for block_x in 0..blocks_per_row {
                let block_index = channel * kx1 * blocks_per_row + block_x * kx1 + row;
                let block = y.add(block_index * block_bytes);
                if layout == Layout::D2s6 {
                    for scale_group in 0..2usize {
                        let start = block_x * 128 + scale_group * 64;
                        let mut amax = 0.0f32;
                        for lane in 0..64usize {
                            amax = amax.max(read_value(src_row, start + lane).abs());
                        }
                        let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
                        (block as *mut u16)
                            .add(scale_group)
                            .write_unaligned(f32_to_f16(d));
                        for lane in 0..64usize {
                            let q = quantize(read_value(src_row, start + lane), d);
                            block
                                .add(16 + scale_group * 64 + lane)
                                .write_unaligned(q as u8);
                        }
                    }
                    for sum_group in 0..6usize {
                        let start = block_x * 128 + sum_group * 16;
                        let mut sum = 0.0f32;
                        for lane in 0..16usize {
                            sum += read_value(src_row, start + lane);
                        }
                        (block as *mut u16)
                            .add(2 + sum_group)
                            .write_unaligned(f32_to_f16(sum));
                    }
                    continue;
                }

                for group in 0..4usize {
                    let start = block_x * 128 + group * 32;
                    let mut amax = 0.0f32;
                    let mut sum = 0.0f32;
                    let mut values = [0.0f32; 32];
                    for lane in 0..32usize {
                        let value = read_value(src_row, start + lane);
                        values[lane] = value;
                        amax = amax.max(value.abs());
                        sum += value;
                    }
                    let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
                    for lane in 0..32usize {
                        let q = quantize(values[lane], d);
                        block.add(16 + group * 32 + lane).write_unaligned(q as u8);
                    }
                    match layout {
                        Layout::D4 => {
                            (block as *mut f32).add(group).write_unaligned(d);
                        }
                        Layout::Ds4 => {
                            let meta = block.add(group * 4) as *mut u16;
                            meta.write_unaligned(f32_to_f16(d));
                            meta.add(1).write_unaligned(f32_to_f16(sum));
                        }
                        Layout::D2s6 => unreachable!(),
                    }
                }
            }
        }
    }

    eprintln!(
        "[TMatmul Fallback] quantize_mmq_q8_1 '{}' executed kx0={} kx1={} channels={} layout={}",
        kernel_name,
        kx0,
        kx1,
        channels,
        match layout {
            Layout::D4 => "d4",
            Layout::Ds4 => "ds4",
            Layout::D2s6 => "d2s6",
        }
    );
    true
}

#[cfg(feature = "intel")]
unsafe fn execute_tmatmul_quantize_q8_1_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_y: u32,
) -> bool {
    let Some(x_addr) = tmatmul_read_param_u64(kernel_params, 0) else {
        return false;
    };
    let Some(y_addr) = tmatmul_read_param_u64(kernel_params, 1) else {
        return false;
    };
    let Some(kx) = tmatmul_read_param_i64(kernel_params, 2) else {
        return false;
    };
    let Some(kx0_padded) = tmatmul_read_param_i64(kernel_params, 3) else {
        return false;
    };

    if x_addr == 0 || y_addr == 0 || kx < 0 || kx0_padded <= 0 {
        return false;
    }
    let Ok(kx) = usize::try_from(kx) else {
        return false;
    };
    let Ok(kx0_padded) = usize::try_from(kx0_padded) else {
        return false;
    };
    if kx0_padded % 32 != 0 {
        return false;
    }
    let rows = grid_dim_y.max(1) as usize;
    let Some(x_bytes) = rows
        .checked_mul(kx)
        .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
    else {
        return false;
    };
    let Some(y_bytes) = rows
        .checked_mul(kx0_padded / 32)
        .and_then(|v| v.checked_mul(36))
    else {
        return false;
    };
    if !tmatmul_host_or_virtual_alloc_has_bytes(x_addr, x_bytes, false)
        || !tmatmul_host_or_virtual_alloc_has_bytes(y_addr, y_bytes, true)
    {
        eprintln!(
            "[TMatmul Fallback] quantize_q8_1 '{}' rejected range x=0x{:x}/{} y=0x{:x}/{}",
            kernel_name, x_addr, x_bytes, y_addr, y_bytes
        );
        return false;
    }

    let x = x_addr as *const f32;
    let y = y_addr as *mut u8;
    for row in 0..rows {
        for block_col in (0..kx0_padded).step_by(32) {
            let block_index = (row * kx0_padded + block_col) / 32;
            let block = y.add(block_index * 36);
            let mut values = [0.0f32; 32];
            let mut amax = 0.0f32;
            let mut sum = 0.0f32;
            for lane in 0..32usize {
                let col = block_col + lane;
                let value = if col < kx {
                    x.add(row * kx + col).read_unaligned()
                } else {
                    0.0
                };
                values[lane] = value;
                amax = amax.max(value.abs());
                sum += value;
            }
            let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
            for lane in 0..32usize {
                let q = if d == 0.0 {
                    0i8
                } else {
                    (values[lane] / d).round().clamp(-128.0, 127.0) as i8
                };
                block.add(lane).write_unaligned(q as u8);
            }
            (block.add(32) as *mut u16).write_unaligned(f32_to_f16(d));
            (block.add(34) as *mut u16).write_unaligned(f32_to_f16(sum));
        }
    }

    eprintln!(
        "[TMatmul Fallback] quantize_q8_1 '{}' executed kx={} kx0_padded={} rows={}",
        kernel_name, kx, kx0_padded, rows
    );
    true
}

#[cfg(feature = "intel")]
unsafe fn execute_tmatmul_cpy_f32_f16_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> bool {
    let Some(src_addr) = tmatmul_read_param_u64(kernel_params, 0) else {
        return false;
    };
    let Some(dst_addr) = tmatmul_read_param_u64(kernel_params, 1) else {
        return false;
    };
    let Some(ne) = tmatmul_read_param_i32(kernel_params, 2) else {
        return false;
    };
    let Some(ne00) = tmatmul_read_param_i32(kernel_params, 3) else {
        return false;
    };
    let Some(ne01) = tmatmul_read_param_i32(kernel_params, 4) else {
        return false;
    };
    let Some(ne02) = tmatmul_read_param_i32(kernel_params, 5) else {
        return false;
    };
    let Some(nb00) = tmatmul_read_param_i32(kernel_params, 6) else {
        return false;
    };
    let Some(nb01) = tmatmul_read_param_i32(kernel_params, 7) else {
        return false;
    };
    let Some(nb02) = tmatmul_read_param_i32(kernel_params, 8) else {
        return false;
    };
    let Some(nb03) = tmatmul_read_param_i32(kernel_params, 9) else {
        return false;
    };
    let Some(ne10) = tmatmul_read_param_i32(kernel_params, 10) else {
        return false;
    };
    let Some(ne11) = tmatmul_read_param_i32(kernel_params, 11) else {
        return false;
    };
    let Some(ne12) = tmatmul_read_param_i32(kernel_params, 12) else {
        return false;
    };
    let Some(nb10) = tmatmul_read_param_i32(kernel_params, 13) else {
        return false;
    };
    let Some(nb11) = tmatmul_read_param_i32(kernel_params, 14) else {
        return false;
    };
    let Some(nb12) = tmatmul_read_param_i32(kernel_params, 15) else {
        return false;
    };
    let Some(nb13) = tmatmul_read_param_i32(kernel_params, 16) else {
        return false;
    };

    if src_addr == 0
        || dst_addr == 0
        || ne < 0
        || ne00 <= 0
        || ne01 <= 0
        || ne02 <= 0
        || ne10 <= 0
        || ne11 <= 0
        || ne12 <= 0
        || nb00 < 0
        || nb01 < 0
        || nb02 < 0
        || nb03 < 0
        || nb10 < 0
        || nb11 < 0
        || nb12 < 0
        || nb13 < 0
    {
        return false;
    }
    let ne = ne as usize;
    let ne00 = ne00 as usize;
    let ne01 = ne01 as usize;
    let ne02 = ne02 as usize;
    let ne10 = ne10 as usize;
    let ne11 = ne11 as usize;
    let ne12 = ne12 as usize;
    let nb00 = nb00 as usize;
    let nb01 = nb01 as usize;
    let nb02 = nb02 as usize;
    let nb03 = nb03 as usize;
    let nb10 = nb10 as usize;
    let nb11 = nb11 as usize;
    let nb12 = nb12 as usize;
    let nb13 = nb13 as usize;

    let (src_elem, dst_elem) = if kernel_name.contains("cpy_1_f32_f32") {
        (4usize, 4usize)
    } else if kernel_name.contains("cpy_1_f32_f16") {
        (4usize, 2usize)
    } else if kernel_name.contains("cpy_1_f16_f32") {
        (2usize, 4usize)
    } else if kernel_name.contains("cpy_1_f16_f16") {
        (2usize, 2usize)
    } else {
        return false;
    };

    let index_offsets = |i: usize,
                         n0: usize,
                         n1: usize,
                         n2: usize,
                         b0: usize,
                         b1: usize,
                         b2: usize,
                         b3: usize|
     -> Option<usize> {
        let plane = n0.checked_mul(n1)?.checked_mul(n2)?;
        let row = n0.checked_mul(n1)?;
        let i3 = i / plane;
        let rem3 = i.checked_sub(i3.checked_mul(plane)?)?;
        let i2 = rem3 / row;
        let rem2 = rem3.checked_sub(i2.checked_mul(row)?)?;
        let i1 = rem2 / n0;
        let i0 = rem2.checked_sub(i1.checked_mul(n0)?)?;
        i0.checked_mul(b0)?
            .checked_add(i1.checked_mul(b1)?)?
            .checked_add(i2.checked_mul(b2)?)?
            .checked_add(i3.checked_mul(b3)?)
    };

    let mut max_src_end = 0usize;
    let mut max_dst_end = 0usize;
    for i in 0..ne {
        let Some(src_off) = index_offsets(i, ne00, ne01, ne02, nb00, nb01, nb02, nb03) else {
            return false;
        };
        let Some(dst_off) = index_offsets(i, ne10, ne11, ne12, nb10, nb11, nb12, nb13) else {
            return false;
        };
        max_src_end = max_src_end.max(src_off.saturating_add(src_elem));
        max_dst_end = max_dst_end.max(dst_off.saturating_add(dst_elem));
    }
    if !tmatmul_host_or_virtual_alloc_has_bytes(src_addr, max_src_end, false)
        || !tmatmul_host_or_virtual_alloc_has_bytes(dst_addr, max_dst_end, true)
    {
        eprintln!(
            "[TMatmul Fallback] cpy_f32_f16 '{}' rejected ranges src=0x{:x}/{} dst=0x{:x}/{}",
            kernel_name, src_addr, max_src_end, dst_addr, max_dst_end
        );
        return false;
    }

    let src = src_addr as *const u8;
    let dst = dst_addr as *mut u8;
    for i in 0..ne {
        let Some(src_off) = index_offsets(i, ne00, ne01, ne02, nb00, nb01, nb02, nb03) else {
            return false;
        };
        let Some(dst_off) = index_offsets(i, ne10, ne11, ne12, nb10, nb11, nb12, nb13) else {
            return false;
        };
        match (src_elem, dst_elem) {
            (4, 4) => {
                let value = (src.add(src_off) as *const f32).read_unaligned();
                (dst.add(dst_off) as *mut f32).write_unaligned(value);
            }
            (4, 2) => {
                let value = (src.add(src_off) as *const f32).read_unaligned();
                (dst.add(dst_off) as *mut u16).write_unaligned(f32_to_f16(value));
            }
            (2, 4) => {
                let value = f16_to_f32((src.add(src_off) as *const u16).read_unaligned());
                (dst.add(dst_off) as *mut f32).write_unaligned(value);
            }
            (2, 2) => {
                let value = (src.add(src_off) as *const u16).read_unaligned();
                (dst.add(dst_off) as *mut u16).write_unaligned(value);
            }
            _ => return false,
        }
    }

    eprintln!(
        "[TMatmul Fallback] cpy_f32_f16 '{}' executed ne={} src_elem={} dst_elem={}",
        kernel_name, ne, src_elem, dst_elem
    );
    true
}

#[cfg(feature = "intel")]
fn tmatmul_parse_mangled_scalar_size(name: &str, offset: &mut usize) -> Option<usize> {
    let rest = name.get(*offset..)?;
    if rest.starts_with('f') {
        *offset += 1;
        Some(std::mem::size_of::<f32>())
    } else if rest.starts_with("6__half") {
        *offset += "6__half".len();
        Some(std::mem::size_of::<u16>())
    } else {
        None
    }
}

#[cfg(feature = "intel")]
fn tmatmul_parse_convert_unary_element_sizes(kernel_name: &str) -> Option<(usize, usize)> {
    let marker = "convert_unaryI";
    let mut offset = kernel_name.find(marker)? + marker.len();
    let src = tmatmul_parse_mangled_scalar_size(kernel_name, &mut offset)?;
    let dst = tmatmul_parse_mangled_scalar_size(kernel_name, &mut offset)?;
    Some((src, dst))
}

#[cfg(feature = "intel")]
unsafe fn execute_tmatmul_convert_unary_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> bool {
    let Some((src_elem, dst_elem)) = tmatmul_parse_convert_unary_element_sizes(kernel_name) else {
        return false;
    };
    if !matches!(src_elem, 2 | 4) || !matches!(dst_elem, 2 | 4) {
        return false;
    }
    let Some(src_addr) = tmatmul_read_param_u64(kernel_params, 0) else {
        return false;
    };
    let Some(dst_addr) = tmatmul_read_param_u64(kernel_params, 1) else {
        return false;
    };
    let Some(ne) = tmatmul_read_param_i64(kernel_params, 2) else {
        return false;
    };

    if src_addr == 0 || dst_addr == 0 || ne < 0 {
        return false;
    }
    let Ok(ne) = usize::try_from(ne) else {
        return false;
    };
    let Some(src_bytes) = ne.checked_mul(src_elem) else {
        return false;
    };
    let Some(dst_bytes) = ne.checked_mul(dst_elem) else {
        return false;
    };
    if !tmatmul_host_or_virtual_alloc_has_bytes(src_addr, src_bytes, false)
        || !tmatmul_host_or_virtual_alloc_has_bytes(dst_addr, dst_bytes, true)
    {
        eprintln!(
            "[TMatmul Fallback] convert_unary '{}' rejected ranges src=0x{:x}/{} dst=0x{:x}/{}",
            kernel_name, src_addr, src_bytes, dst_addr, dst_bytes
        );
        return false;
    }

    let src = src_addr as *const u8;
    let dst = dst_addr as *mut u8;
    for i in 0..ne {
        let value = match src_elem {
            4 => (src.add(i * 4) as *const f32).read_unaligned(),
            2 => f16_to_f32((src.add(i * 2) as *const u16).read_unaligned()),
            _ => return false,
        };
        match dst_elem {
            4 => (dst.add(i * 4) as *mut f32).write_unaligned(value),
            2 => (dst.add(i * 2) as *mut u16).write_unaligned(f32_to_f16(value)),
            _ => return false,
        }
    }

    eprintln!(
        "[TMatmul Fallback] convert_unary '{}' executed ne={} src_elem={} dst_elem={}",
        kernel_name, ne, src_elem, dst_elem
    );
    true
}

#[cfg(feature = "intel")]
fn tmatmul_direct_unary_f32_op(name_lower: &str) -> Option<&'static str> {
    if name_lower.contains("hardsigmoid_f32") {
        Some("hardsigmoid")
    } else if name_lower.contains("hardswish_f32") {
        Some("hardswish")
    } else if name_lower.contains("gelu_quick_f32") {
        Some("gelu_quick")
    } else if name_lower.contains("sigmoid_f32") {
        Some("sigmoid")
    } else if name_lower.contains("silu_f32") {
        Some("silu")
    } else if name_lower.contains("gelu_f32") {
        Some("gelu")
    } else if name_lower.contains("relu_f32") {
        Some("relu")
    } else if name_lower.contains("tanh_f32") {
        Some("tanh")
    } else if name_lower.contains("exp_f32") {
        Some("exp")
    } else if name_lower.contains("sqrt_f32") {
        Some("sqrt")
    } else if name_lower.contains("sqr_f32") {
        Some("sqr")
    } else if name_lower.contains("sin_f32") {
        Some("sin")
    } else if name_lower.contains("cos_f32") {
        Some("cos")
    } else if name_lower.contains("neg_f32") {
        Some("neg")
    } else if name_lower.contains("step_f32") {
        Some("step")
    } else {
        None
    }
}

#[cfg(feature = "intel")]
unsafe fn execute_tmatmul_direct_unary_f32_fallback(
    kernel_name: &str,
    name_lower: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> bool {
    let Some(op) = tmatmul_direct_unary_f32_op(name_lower) else {
        return false;
    };
    let Some(src_addr) = tmatmul_read_param_u64(kernel_params, 0) else {
        return false;
    };
    let Some(dst_addr) = tmatmul_read_param_u64(kernel_params, 1) else {
        return false;
    };
    let Some(k) = tmatmul_read_param_i32(kernel_params, 2) else {
        return false;
    };
    if src_addr == 0 || dst_addr == 0 || k < 0 {
        return false;
    }
    let Ok(ne) = usize::try_from(k) else {
        return false;
    };
    let Some(bytes) = ne.checked_mul(std::mem::size_of::<f32>()) else {
        return false;
    };
    if !tmatmul_host_or_virtual_alloc_has_bytes(src_addr, bytes, false)
        || !tmatmul_host_or_virtual_alloc_has_bytes(dst_addr, bytes, true)
    {
        eprintln!(
            "[TMatmul Fallback] direct unary f32 '{}' rejected ranges src=0x{:x}/{} dst=0x{:x}/{}",
            kernel_name, src_addr, bytes, dst_addr, bytes
        );
        return false;
    }

    let src = src_addr as *const f32;
    let dst = dst_addr as *mut f32;
    for i in 0..ne {
        let x = src.add(i).read_unaligned();
        let y = match op {
            "hardsigmoid" => ((x + 3.0) / 6.0).clamp(0.0, 1.0),
            "hardswish" => x * ((x + 3.0) / 6.0).clamp(0.0, 1.0),
            "gelu_quick" => x / (1.0 + (-1.702 * x).exp()),
            "sigmoid" => 1.0 / (1.0 + (-x).exp()),
            "silu" => x / (1.0 + (-x).exp()),
            "gelu" => {
                let c = 0.7978845608028654_f32;
                0.5 * x * (1.0 + (c * x * (1.0 + 0.044715 * x * x)).tanh())
            }
            "relu" => x.max(0.0),
            "tanh" => x.tanh(),
            "exp" => x.exp(),
            "sqrt" => x.sqrt(),
            "sqr" => x * x,
            "sin" => x.sin(),
            "cos" => x.cos(),
            "neg" => -x,
            "step" => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            _ => return false,
        };
        dst.add(i).write_unaligned(y);
    }

    eprintln!(
        "[TMatmul Fallback] direct unary f32 '{}' executed op={} ne={}",
        kernel_name, op, ne
    );
    true
}

#[cfg(feature = "intel")]
fn tmatmul_parse_get_rows_float_element_sizes(kernel_name: &str) -> Option<(usize, usize)> {
    let marker = "k_get_rows_floatI";
    let mut offset = kernel_name.find(marker)? + marker.len();
    let src = tmatmul_parse_mangled_scalar_size(kernel_name, &mut offset)?;
    let dst = tmatmul_parse_mangled_scalar_size(kernel_name, &mut offset)?;
    Some((src, dst))
}

#[cfg(feature = "intel")]
unsafe fn execute_tmatmul_get_rows_float_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_y: u32,
    grid_dim_z: u32,
    block_dim_y: u32,
    block_dim_z: u32,
) -> bool {
    let Some((src_elem, dst_elem)) = tmatmul_parse_get_rows_float_element_sizes(kernel_name) else {
        return false;
    };
    if !matches!(src_elem, 2 | 4) || !matches!(dst_elem, 2 | 4) {
        return false;
    }

    let Some(src0_addr) = tmatmul_read_param_u64(kernel_params, 0) else {
        return false;
    };
    let Some(src1_addr) = tmatmul_read_param_u64(kernel_params, 1) else {
        return false;
    };
    let Some(dst_addr) = tmatmul_read_param_u64(kernel_params, 2) else {
        return false;
    };
    let Some(ne00) = tmatmul_read_param_i64(kernel_params, 3) else {
        return false;
    };
    let Some(ne12) = tmatmul_read_param_i64(kernel_params, 4) else {
        return false;
    };
    let Some(s1) = tmatmul_read_param_u64(kernel_params, 5) else {
        return false;
    };
    let Some(s2) = tmatmul_read_param_u64(kernel_params, 6) else {
        return false;
    };
    let Some(s3) = tmatmul_read_param_u64(kernel_params, 7) else {
        return false;
    };
    let Some(nb01) = tmatmul_read_param_u64(kernel_params, 8) else {
        return false;
    };
    let Some(nb02) = tmatmul_read_param_u64(kernel_params, 9) else {
        return false;
    };
    let Some(nb03) = tmatmul_read_param_u64(kernel_params, 10) else {
        return false;
    };
    let Some(s10) = tmatmul_read_param_u64(kernel_params, 11) else {
        return false;
    };
    let Some(s11) = tmatmul_read_param_u64(kernel_params, 12) else {
        return false;
    };
    let Some(s12) = tmatmul_read_param_u64(kernel_params, 13) else {
        return false;
    };

    if src0_addr == 0 || src1_addr == 0 || dst_addr == 0 || ne00 < 0 || ne12 <= 0 {
        return false;
    }
    let Ok(ne00) = usize::try_from(ne00) else {
        return false;
    };
    let Ok(ne12) = usize::try_from(ne12) else {
        return false;
    };
    let Some(ne10) = (grid_dim_y.max(1) as usize).checked_mul(block_dim_y.max(1) as usize) else {
        return false;
    };
    let Some(z_lanes) = (grid_dim_z.max(1) as usize).checked_mul(block_dim_z.max(1) as usize)
    else {
        return false;
    };
    if ne00 == 0 || ne10 == 0 || z_lanes == 0 {
        eprintln!(
            "[TMatmul Fallback] k_get_rows_float '{}' executed empty gather ne00={} ne10={} z_lanes={}",
            kernel_name, ne00, ne10, z_lanes
        );
        return true;
    }

    let Some(src_row_bytes) = ne00.checked_mul(src_elem) else {
        return false;
    };
    let Some(max_i10) = ne10.checked_sub(1) else {
        return false;
    };
    let mut max_idx_off = 0usize;
    let mut max_dst_end = 0usize;
    for linear_z in 0..z_lanes {
        let i11 = linear_z / ne12;
        let i12 = linear_z % ne12;
        let Some(idx_base) = (|| -> Option<usize> {
            Some(
                i11.checked_mul(s11 as usize)?
                    .checked_add(i12.checked_mul(s12 as usize)?)?,
            )
        })() else {
            return false;
        };
        let Some(idx_off) = max_i10
            .checked_mul(s10 as usize)
            .and_then(|value| value.checked_add(idx_base))
        else {
            return false;
        };
        max_idx_off = max_idx_off.max(idx_off);

        let Some(dst_base) = (|| -> Option<usize> {
            Some(
                max_i10
                    .checked_mul(s1 as usize)?
                    .checked_add(i11.checked_mul(s2 as usize)?)?
                    .checked_add(i12.checked_mul(s3 as usize)?)?,
            )
        })() else {
            return false;
        };
        let Some(dst_end) = dst_base
            .checked_add(ne00)
            .and_then(|elems| elems.checked_mul(dst_elem))
        else {
            return false;
        };
        max_dst_end = max_dst_end.max(dst_end);
    }

    let Some(idx_bytes) = max_idx_off
        .checked_add(1)
        .and_then(|elems| elems.checked_mul(std::mem::size_of::<i32>()))
    else {
        return false;
    };
    if !tmatmul_host_or_virtual_alloc_has_bytes(src1_addr, idx_bytes, false)
        || !tmatmul_host_or_virtual_alloc_has_bytes(dst_addr, max_dst_end, true)
    {
        eprintln!(
            "[TMatmul Fallback] k_get_rows_float '{}' rejected index/dst ranges src1=0x{:x}/{} dst=0x{:x}/{}",
            kernel_name, src1_addr, idx_bytes, dst_addr, max_dst_end
        );
        return false;
    }

    let src0 = src0_addr as *const u8;
    let src1 = src1_addr as *const i32;
    let dst = dst_addr as *mut u8;
    let clamp_oob = tmatmul_env_enabled_default("HETGPU_TMATMUL_GET_ROWS_FLOAT_CLAMP_OOB", true);
    let mut clamped_rows = 0usize;

    for linear_z in 0..z_lanes {
        let i11 = linear_z / ne12;
        let i12 = linear_z % ne12;
        let Some(z_idx_base) = (|| -> Option<usize> {
            Some(
                i11.checked_mul(s11 as usize)?
                    .checked_add(i12.checked_mul(s12 as usize)?)?,
            )
        })() else {
            return false;
        };
        let Some(z_dst_base) = (|| -> Option<usize> {
            Some(
                i11.checked_mul(s2 as usize)?
                    .checked_add(i12.checked_mul(s3 as usize)?)?,
            )
        })() else {
            return false;
        };
        let Some(z_src_base) = (|| -> Option<usize> {
            Some(
                i11.checked_mul(nb02 as usize)?
                    .checked_add(i12.checked_mul(nb03 as usize)?)?,
            )
        })() else {
            return false;
        };

        for i10 in 0..ne10 {
            let Some(idx_off) = i10
                .checked_mul(s10 as usize)
                .and_then(|value| value.checked_add(z_idx_base))
            else {
                return false;
            };
            let i01 = src1.add(idx_off).read_unaligned().max(0) as usize;
            let Some(dst_row_byte) = i10
                .checked_mul(s1 as usize)
                .and_then(|value| value.checked_add(z_dst_base))
                .and_then(|value| value.checked_mul(dst_elem))
            else {
                return false;
            };
            let Some(src_row_byte) = i01
                .checked_mul(nb01 as usize)
                .and_then(|value| value.checked_add(z_src_base))
            else {
                return false;
            };
            let src_row_addr = src0_addr.saturating_add(src_row_byte as u64);
            let src_row = if tmatmul_host_or_virtual_alloc_has_bytes(
                src_row_addr,
                src_row_bytes,
                false,
            ) {
                src0.add(src_row_byte)
            } else if clamp_oob {
                clamped_rows = clamped_rows.saturating_add(1);
                let fallback_addr = src0_addr.saturating_add(z_src_base as u64);
                if tmatmul_host_or_virtual_alloc_has_bytes(fallback_addr, src_row_bytes, false) {
                    src0.add(z_src_base)
                } else {
                    std::ptr::null()
                }
            } else {
                eprintln!(
                    "[TMatmul Fallback] k_get_rows_float '{}' rejected source row outside allocation src0=0x{:x} row=0x{:x} idx={} ne00={}",
                    kernel_name, src0_addr, src_row_addr, i01, ne00
                );
                return false;
            };
            let dst_row = dst.add(dst_row_byte);
            for i00 in 0..ne00 {
                let value = if src_row.is_null() {
                    0.0
                } else {
                    match src_elem {
                        4 => (src_row.add(i00 * 4) as *const f32).read_unaligned(),
                        2 => f16_to_f32((src_row.add(i00 * 2) as *const u16).read_unaligned()),
                        _ => return false,
                    }
                };
                match dst_elem {
                    4 => (dst_row.add(i00 * 4) as *mut f32).write_unaligned(value),
                    2 => (dst_row.add(i00 * 2) as *mut u16).write_unaligned(f32_to_f16(value)),
                    _ => return false,
                }
            }
        }
    }

    eprintln!(
        "[TMatmul Fallback] k_get_rows_float '{}' executed ne00={} ne10={} z_lanes={} ne12={} src_elem={} dst_elem={} clamped_rows={}",
        kernel_name, ne00, ne10, z_lanes, ne12, src_elem, dst_elem, clamped_rows
    );
    true
}

#[cfg(feature = "intel")]
unsafe fn execute_tmatmul_concat_f32_dim_fallback(
    kernel_name: &str,
    name_lower: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_y: u32,
    grid_dim_z: u32,
) -> bool {
    let dim = if name_lower.contains("concat_f32_dim0") {
        0usize
    } else if name_lower.contains("concat_f32_dim1") {
        1usize
    } else if name_lower.contains("concat_f32_dim2") {
        2usize
    } else {
        return false;
    };
    let Some(x_addr) = tmatmul_read_param_u64(kernel_params, 0) else {
        return false;
    };
    let Some(y_addr) = tmatmul_read_param_u64(kernel_params, 1) else {
        return false;
    };
    let Some(dst_addr) = tmatmul_read_param_u64(kernel_params, 2) else {
        return false;
    };
    let Some(ne0) = tmatmul_read_param_i32(kernel_params, 3) else {
        return false;
    };
    let Some(split) = tmatmul_read_param_i32(kernel_params, 4) else {
        return false;
    };

    if x_addr == 0 || y_addr == 0 || dst_addr == 0 || ne0 <= 0 || split < 0 {
        return false;
    }
    let ne0 = ne0 as usize;
    let split = split as usize;
    let ne1 = grid_dim_y.max(1) as usize;
    let ne2 = grid_dim_z.max(1) as usize;
    let Some((x_elems, y_elems, dst_elems)) = (|| -> Option<(usize, usize, usize)> {
        match dim {
            0 if split <= ne0 => Some((
                split.checked_mul(ne1)?.checked_mul(ne2)?,
                ne0.checked_sub(split)?.checked_mul(ne1)?.checked_mul(ne2)?,
                ne0.checked_mul(ne1)?.checked_mul(ne2)?,
            )),
            1 if split <= ne1 => Some((
                ne0.checked_mul(split)?.checked_mul(ne2)?,
                ne0.checked_mul(ne1.checked_sub(split)?)?.checked_mul(ne2)?,
                ne0.checked_mul(ne1)?.checked_mul(ne2)?,
            )),
            2 if split <= ne2 => Some((
                ne0.checked_mul(ne1)?.checked_mul(split)?,
                ne0.checked_mul(ne1)?.checked_mul(ne2.checked_sub(split)?)?,
                ne0.checked_mul(ne1)?.checked_mul(ne2)?,
            )),
            _ => None,
        }
    })() else {
        return false;
    };
    let elem_size = std::mem::size_of::<f32>();
    let Some(x_bytes) = x_elems.checked_mul(elem_size) else {
        return false;
    };
    let Some(y_bytes) = y_elems.checked_mul(elem_size) else {
        return false;
    };
    let Some(dst_bytes) = dst_elems.checked_mul(elem_size) else {
        return false;
    };
    if !tmatmul_host_or_virtual_alloc_has_bytes(x_addr, x_bytes, false)
        || !tmatmul_host_or_virtual_alloc_has_bytes(y_addr, y_bytes, false)
        || !tmatmul_host_or_virtual_alloc_has_bytes(dst_addr, dst_bytes, true)
    {
        eprintln!(
            "[TMatmul Fallback] concat_f32_dim '{}' rejected ranges x=0x{:x}/{} y=0x{:x}/{} dst=0x{:x}/{}",
            kernel_name, x_addr, x_bytes, y_addr, y_bytes, dst_addr, dst_bytes
        );
        return false;
    }

    let x = x_addr as *const f32;
    let y = y_addr as *const f32;
    let dst = dst_addr as *mut f32;
    for i2 in 0..ne2 {
        for i1 in 0..ne1 {
            for i0 in 0..ne0 {
                let dst_index = i0 + i1 * ne0 + i2 * ne0 * ne1;
                let value = match dim {
                    0 if i0 < split => {
                        let src_index = i0 + i1 * split + i2 * split * ne1;
                        x.add(src_index).read_unaligned()
                    }
                    0 => {
                        let src_ne0 = ne0 - split;
                        let src_index = (i0 - split) + i1 * src_ne0 + i2 * src_ne0 * ne1;
                        y.add(src_index).read_unaligned()
                    }
                    1 if i1 < split => {
                        let src_index = i0 + i1 * ne0 + i2 * ne0 * split;
                        x.add(src_index).read_unaligned()
                    }
                    1 => {
                        let src_ne1 = ne1 - split;
                        let src_index = i0 + (i1 - split) * ne0 + i2 * ne0 * src_ne1;
                        y.add(src_index).read_unaligned()
                    }
                    2 if i2 < split => {
                        let src_index = i0 + i1 * ne0 + i2 * ne0 * ne1;
                        x.add(src_index).read_unaligned()
                    }
                    2 => {
                        let src_index = i0 + i1 * ne0 + (i2 - split) * ne0 * ne1;
                        y.add(src_index).read_unaligned()
                    }
                    _ => return false,
                };
                dst.add(dst_index).write_unaligned(value);
            }
        }
    }

    eprintln!(
        "[TMatmul Fallback] concat_f32_dim '{}' executed dim={} ne0={} split={} grid_y={} grid_z={}",
        kernel_name, dim, ne0, split, ne1, ne2
    );
    true
}

#[cfg(feature = "intel")]
unsafe fn execute_tmatmul_concat_f32_non_cont_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> bool {
    let Some(src0_addr) = tmatmul_read_param_u64(kernel_params, 0) else {
        return false;
    };
    let Some(src1_addr) = tmatmul_read_param_u64(kernel_params, 1) else {
        return false;
    };
    let Some(dst_addr) = tmatmul_read_param_u64(kernel_params, 2) else {
        return false;
    };
    let Some(ne00) = tmatmul_read_param_i64(kernel_params, 3) else {
        return false;
    };
    let Some(ne01) = tmatmul_read_param_i64(kernel_params, 4) else {
        return false;
    };
    let Some(ne02) = tmatmul_read_param_i64(kernel_params, 5) else {
        return false;
    };
    let Some(ne03) = tmatmul_read_param_i64(kernel_params, 6) else {
        return false;
    };
    let Some(nb00) = tmatmul_read_param_u64(kernel_params, 7) else {
        return false;
    };
    let Some(nb01) = tmatmul_read_param_u64(kernel_params, 8) else {
        return false;
    };
    let Some(nb02) = tmatmul_read_param_u64(kernel_params, 9) else {
        return false;
    };
    let Some(nb03) = tmatmul_read_param_u64(kernel_params, 10) else {
        return false;
    };
    let Some(ne10) = tmatmul_read_param_i64(kernel_params, 11) else {
        return false;
    };
    let Some(ne11) = tmatmul_read_param_i64(kernel_params, 12) else {
        return false;
    };
    let Some(ne12) = tmatmul_read_param_i64(kernel_params, 13) else {
        return false;
    };
    let Some(ne13) = tmatmul_read_param_i64(kernel_params, 14) else {
        return false;
    };
    let Some(nb10) = tmatmul_read_param_u64(kernel_params, 15) else {
        return false;
    };
    let Some(nb11) = tmatmul_read_param_u64(kernel_params, 16) else {
        return false;
    };
    let Some(nb12) = tmatmul_read_param_u64(kernel_params, 17) else {
        return false;
    };
    let Some(nb13) = tmatmul_read_param_u64(kernel_params, 18) else {
        return false;
    };
    let Some(ne0) = tmatmul_read_param_i64(kernel_params, 19) else {
        return false;
    };
    let Some(ne1) = tmatmul_read_param_i64(kernel_params, 20) else {
        return false;
    };
    let Some(ne2) = tmatmul_read_param_i64(kernel_params, 21) else {
        return false;
    };
    let Some(ne3) = tmatmul_read_param_i64(kernel_params, 22) else {
        return false;
    };
    let Some(nb0) = tmatmul_read_param_u64(kernel_params, 23) else {
        return false;
    };
    let Some(nb1) = tmatmul_read_param_u64(kernel_params, 24) else {
        return false;
    };
    let Some(nb2) = tmatmul_read_param_u64(kernel_params, 25) else {
        return false;
    };
    let Some(nb3) = tmatmul_read_param_u64(kernel_params, 26) else {
        return false;
    };
    let Some(dim) = tmatmul_read_param_i32(kernel_params, 27) else {
        return false;
    };

    if src0_addr == 0 || src1_addr == 0 || dst_addr == 0 || !(0..=3).contains(&dim) {
        return false;
    }
    let dims = [
        ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3,
    ];
    if dims.iter().any(|&value| value < 0) {
        return false;
    }
    if ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0 {
        return true;
    }

    let Ok(ne00) = usize::try_from(ne00) else {
        return false;
    };
    let Ok(ne01) = usize::try_from(ne01) else {
        return false;
    };
    let Ok(ne02) = usize::try_from(ne02) else {
        return false;
    };
    let Ok(ne03) = usize::try_from(ne03) else {
        return false;
    };
    let Ok(ne10) = usize::try_from(ne10) else {
        return false;
    };
    let Ok(ne11) = usize::try_from(ne11) else {
        return false;
    };
    let Ok(ne12) = usize::try_from(ne12) else {
        return false;
    };
    let Ok(ne13) = usize::try_from(ne13) else {
        return false;
    };
    let Ok(ne0) = usize::try_from(ne0) else {
        return false;
    };
    let Ok(ne1) = usize::try_from(ne1) else {
        return false;
    };
    let Ok(ne2) = usize::try_from(ne2) else {
        return false;
    };
    let Ok(ne3) = usize::try_from(ne3) else {
        return false;
    };
    let dim = dim as usize;
    let nb00 = nb00 as usize;
    let nb01 = nb01 as usize;
    let nb02 = nb02 as usize;
    let nb03 = nb03 as usize;
    let nb10 = nb10 as usize;
    let nb11 = nb11 as usize;
    let nb12 = nb12 as usize;
    let nb13 = nb13 as usize;
    let nb0 = nb0 as usize;
    let nb1 = nb1 as usize;
    let nb2 = nb2 as usize;
    let nb3 = nb3 as usize;

    fn tensor_bytes(
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ne3: usize,
        nb0: usize,
        nb1: usize,
        nb2: usize,
        nb3: usize,
    ) -> Option<usize> {
        if ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0 {
            return Some(0);
        }
        ne3.checked_sub(1)?
            .checked_mul(nb3)?
            .checked_add(ne2.checked_sub(1)?.checked_mul(nb2)?)?
            .checked_add(ne1.checked_sub(1)?.checked_mul(nb1)?)?
            .checked_add(ne0.checked_sub(1)?.checked_mul(nb0)?)?
            .checked_add(std::mem::size_of::<f32>())
    }

    let Some(src0_bytes) = tensor_bytes(ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03) else {
        return false;
    };
    let Some(src1_bytes) = tensor_bytes(ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13) else {
        return false;
    };
    let Some(dst_bytes) = tensor_bytes(ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3) else {
        return false;
    };
    if !tmatmul_host_or_virtual_alloc_has_bytes(src0_addr, src0_bytes, false)
        || !tmatmul_host_or_virtual_alloc_has_bytes(src1_addr, src1_bytes, false)
        || !tmatmul_host_or_virtual_alloc_has_bytes(dst_addr, dst_bytes, true)
    {
        eprintln!(
            "[TMatmul Fallback] concat_f32_non_cont '{}' rejected ranges src0=0x{:x}/{} src1=0x{:x}/{} dst=0x{:x}/{}",
            kernel_name, src0_addr, src0_bytes, src1_addr, src1_bytes, dst_addr, dst_bytes
        );
        return false;
    }

    let src0 = src0_addr as *const u8;
    let src1 = src1_addr as *const u8;
    let dst = dst_addr as *mut u8;
    let src0_ne = [ne00, ne01, ne02, ne03];

    for i3 in 0..ne3 {
        for i2 in 0..ne2 {
            for i1 in 0..ne1 {
                for i0 in 0..ne0 {
                    let in_src0 = i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03;
                    let Some(src_off) = (if in_src0 {
                        i3.checked_mul(nb03)
                            .and_then(|v| v.checked_add(i2.checked_mul(nb02)?))
                            .and_then(|v| v.checked_add(i1.checked_mul(nb01)?))
                            .and_then(|v| v.checked_add(i0.checked_mul(nb00)?))
                    } else {
                        let i = [i0, i1, i2, i3];
                        let nb = [nb10, nb11, nb12, nb13];
                        let mut src_i = [i0, i1, i2, i3];
                        let Some(adjusted_dim) = i[dim].checked_sub(src0_ne[dim]) else {
                            return false;
                        };
                        src_i[dim] = adjusted_dim;
                        src_i[3]
                            .checked_mul(nb[3])
                            .and_then(|v| v.checked_add(src_i[2].checked_mul(nb[2])?))
                            .and_then(|v| v.checked_add(src_i[1].checked_mul(nb[1])?))
                            .and_then(|v| v.checked_add(src_i[0].checked_mul(nb[0])?))
                    }) else {
                        return false;
                    };
                    let Some(dst_off) = i3
                        .checked_mul(nb3)
                        .and_then(|v| v.checked_add(i2.checked_mul(nb2)?))
                        .and_then(|v| v.checked_add(i1.checked_mul(nb1)?))
                        .and_then(|v| v.checked_add(i0.checked_mul(nb0)?))
                    else {
                        return false;
                    };
                    let src_base = if in_src0 { src0 } else { src1 };
                    let value = (src_base.add(src_off) as *const f32).read_unaligned();
                    (dst.add(dst_off) as *mut f32).write_unaligned(value);
                }
            }
        }
    }

    eprintln!(
        "[TMatmul Fallback] concat_f32_non_cont '{}' executed dim={} ne=({}, {}, {}, {})",
        kernel_name, dim, ne0, ne1, ne2, ne3
    );
    true
}

#[cfg(feature = "intel")]
unsafe fn execute_tmatmul_rope_f32_fallback(
    kernel_name: &str,
    name_lower: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: u32,
) -> bool {
    if !name_lower.contains("rope_norm") && !name_lower.contains("rope_neox") {
        return false;
    }
    if !kernel_name.contains("If") {
        return false;
    }

    let Some(src_addr) = tmatmul_read_param_u64(kernel_params, 0) else {
        return false;
    };
    let Some(dst_addr) = tmatmul_read_param_u64(kernel_params, 1) else {
        return false;
    };
    let Some(ne0) = tmatmul_read_param_i32(kernel_params, 2) else {
        return false;
    };
    let Some(n_dims) = tmatmul_read_param_i32(kernel_params, 3) else {
        return false;
    };
    let Some(pos_addr) = tmatmul_read_param_u64(kernel_params, 4) else {
        return false;
    };
    let Some(freq_scale) = tmatmul_read_param_f32(kernel_params, 5) else {
        return false;
    };
    let Some(p_delta_rows) = tmatmul_read_param_i32(kernel_params, 6) else {
        return false;
    };
    let Some(ext_factor) = tmatmul_read_param_f32(kernel_params, 7) else {
        return false;
    };
    let Some(attn_factor) = tmatmul_read_param_f32(kernel_params, 8) else {
        return false;
    };
    let corr_param = *kernel_params.add(9);
    if corr_param.is_null() || (corr_param as usize) < 0x1000 {
        return false;
    }
    let corr0 = (corr_param as *const f32).read_unaligned();
    let corr1 = (corr_param as *const f32).add(1).read_unaligned();
    let Some(theta_scale) = tmatmul_read_param_f32(kernel_params, 10) else {
        return false;
    };
    let freq_factors_addr = tmatmul_read_param_u64(kernel_params, 11).unwrap_or(0);

    if src_addr == 0
        || dst_addr == 0
        || pos_addr == 0
        || ne0 <= 0
        || n_dims <= 0
        || n_dims > ne0
        || ne0 % 2 != 0
        || n_dims % 2 != 0
        || p_delta_rows <= 0
    {
        return false;
    }

    let ne0 = ne0 as usize;
    let n_dims = n_dims as usize;
    let rows = grid_dim_x.max(1) as usize;
    let p_delta_rows = p_delta_rows as usize;
    let Some(data_bytes) = rows
        .checked_mul(ne0)
        .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
    else {
        return false;
    };
    let pos_count = rows.div_ceil(p_delta_rows).max(1);
    let Some(pos_bytes) = pos_count.checked_mul(std::mem::size_of::<i32>()) else {
        return false;
    };
    let freq_bytes = (n_dims / 2).saturating_mul(std::mem::size_of::<f32>());

    if !tmatmul_host_or_virtual_alloc_has_bytes(src_addr, data_bytes, false)
        || !tmatmul_host_or_virtual_alloc_has_bytes(dst_addr, data_bytes, true)
        || !tmatmul_host_or_virtual_alloc_has_bytes(pos_addr, pos_bytes, false)
        || (freq_factors_addr != 0
            && !tmatmul_host_or_virtual_alloc_has_bytes(freq_factors_addr, freq_bytes, false))
    {
        eprintln!(
            "[TMatmul Fallback] rope '{}' rejected ranges src=0x{:x}/{} dst=0x{:x}/{} pos=0x{:x}/{}",
            kernel_name, src_addr, data_bytes, dst_addr, data_bytes, pos_addr, pos_bytes
        );
        return false;
    }

    let src = src_addr as *const f32;
    let dst = dst_addr as *mut f32;
    let pos = pos_addr as *const i32;
    let freq_factors = freq_factors_addr as *const f32;
    let has_freq_factors = freq_factors_addr != 0 && !freq_factors.is_null();
    let is_neox = name_lower.contains("rope_neox");
    let forward = !name_lower.contains("rope_back");

    for row in 0..rows {
        let pos_index = row / p_delta_rows;
        let pos_val = pos.add(pos_index).read_unaligned() as f32;
        for i0 in (0..ne0).step_by(2) {
            let base = row * ne0;
            if i0 >= n_dims {
                if is_neox {
                    let half = n_dims / 2;
                    let ix0 = base + i0 / 2;
                    let ix1 = ix0 + half;
                    dst.add(ix0).write_unaligned(src.add(ix0).read_unaligned());
                    dst.add(ix1).write_unaligned(src.add(ix1).read_unaligned());
                } else {
                    let ix0 = base + i0;
                    let ix1 = ix0 + 1;
                    dst.add(ix0).write_unaligned(src.add(ix0).read_unaligned());
                    dst.add(ix1).write_unaligned(src.add(ix1).read_unaligned());
                }
                continue;
            }

            let freq_factor = if has_freq_factors {
                freq_factors.add(i0 / 2).read_unaligned()
            } else {
                1.0
            };
            let theta_extrap = pos_val * theta_scale.powf(i0 as f32 / 2.0) / freq_factor;
            let theta_interp = freq_scale * theta_extrap;
            let mut theta = theta_interp;
            let mut mscale = attn_factor;
            if ext_factor != 0.0 {
                let y = (i0 as f32 / 2.0 - corr0) / 0.001f32.max(corr1 - corr0);
                let ramp = 1.0 - y.clamp(0.0, 1.0);
                let ramp_mix = ramp * ext_factor;
                theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
                mscale *= 1.0 + 0.1 * (1.0 / freq_scale).ln();
            }
            let cos_theta = theta.cos() * mscale;
            let mut sin_theta = theta.sin() * mscale;
            if !forward {
                sin_theta = -sin_theta;
            }

            if is_neox {
                let half = n_dims / 2;
                let ix0 = base + i0 / 2;
                let ix1 = ix0 + half;
                let x0 = src.add(ix0).read_unaligned();
                let x1 = src.add(ix1).read_unaligned();
                dst.add(ix0)
                    .write_unaligned(x0 * cos_theta - x1 * sin_theta);
                dst.add(ix1)
                    .write_unaligned(x0 * sin_theta + x1 * cos_theta);
            } else {
                let ix0 = base + i0;
                let ix1 = ix0 + 1;
                let x0 = src.add(ix0).read_unaligned();
                let x1 = src.add(ix1).read_unaligned();
                dst.add(ix0)
                    .write_unaligned(x0 * cos_theta - x1 * sin_theta);
                dst.add(ix1)
                    .write_unaligned(x0 * sin_theta + x1 * cos_theta);
            }
        }
    }

    eprintln!(
        "[TMatmul Fallback] rope '{}' executed rows={} ne0={} n_dims={} neox={}",
        kernel_name, rows, ne0, n_dims, is_neox
    );
    true
}

#[cfg(feature = "intel")]
unsafe fn execute_ggml_norm_f32_fallback(
    kernel_name: &str,
    name_lower: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: u32,
    block_dim_y: u32,
) -> bool {
    if name_lower.contains("group_norm") {
        return false;
    }
    let Some(x_addr) = tmatmul_read_param_u64(kernel_params, 0) else {
        return false;
    };
    let Some(dst_addr) = tmatmul_read_param_u64(kernel_params, 1) else {
        return false;
    };
    let Some(ncols) = tmatmul_read_param_i32(kernel_params, 2).map(|v| v.max(0) as usize) else {
        return false;
    };
    let eps = tmatmul_read_param_f32(kernel_params, 3).unwrap_or(1e-5);
    let rows = (grid_dim_x.max(1) as usize).saturating_mul(block_dim_y.max(1) as usize);
    if x_addr == 0 || dst_addr == 0 || ncols == 0 || rows == 0 {
        return false;
    }
    let Some(bytes) = rows
        .checked_mul(ncols)
        .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
    else {
        return false;
    };
    if !tmatmul_host_or_virtual_alloc_has_bytes(x_addr, bytes, false)
        || !tmatmul_host_or_virtual_alloc_has_bytes(dst_addr, bytes, true)
    {
        eprintln!(
            "[TMatmul Fallback] ggml norm '{}' rejected range x=0x{:x}/{} dst=0x{:x}/{}",
            kernel_name, x_addr, bytes, dst_addr, bytes
        );
        return false;
    }

    execute_norm_on_data(
        x_addr as *const f32,
        dst_addr as *mut f32,
        std::ptr::null(),
        std::ptr::null(),
        rows,
        ncols,
        name_lower.contains("rms_norm_f32") || name_lower.contains("rmsnorm_f32"),
        eps,
    );
    eprintln!(
        "[TMatmul Fallback] ggml norm '{}' executed rows={} ncols={} eps={}",
        kernel_name, rows, ncols, eps
    );
    true
}

/// Execute indexSelect kernel fallback (used by nn.Embedding).
#[cfg(feature = "intel")]
unsafe fn execute_indexselect_kernel_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) {
    // indexSelectSmallIndex takes TensorInfo structs for output, input, and indices
    // Scan first param for pointers
    let param0 = *kernel_params.add(0);
    if param0.is_null() {
        eprintln!("[TMatmul Fallback] indexSelect: null parameter");
        return;
    }

    // indexSelectSmallIndex has 7 params:
    //   TensorInfo output, TensorInfo input, TensorInfo indices,
    //   int dim, int numIndices, IndexType innerSize, int64_t outerSizeI
    // Only scan the first 3 TensorInfo params for pointers (each contains data ptr at offset 0)
    let mut all_ptrs: Vec<(usize, u64, usize)> = Vec::new();
    for pi in 0..3 {
        let p = *kernel_params.add(pi);
        if p.is_null() {
            continue;
        }
        // TensorInfo has data pointer at offset 0
        let data_ptr = (p as *const u64).read_unaligned();
        if let Some(size) = super::memory::get_alloc_size(data_ptr as usize) {
            all_ptrs.push((pi, data_ptr, size));
        }
    }

    // Deduplicate by pointer value
    all_ptrs.sort_by_key(|&(_, ptr, _)| ptr);
    all_ptrs.dedup_by_key(|a| a.1);

    if all_ptrs.len() < 3 {
        eprintln!(
            "[TMatmul Fallback] indexSelect: need output, input, indices (found {})",
            all_ptrs.len()
        );
        return;
    }

    // Sort by size: indices (smallest, int64/int32), output (medium), weight table (largest)
    all_ptrs.sort_by_key(|&(_, _, size)| size);

    let (_, indices_ptr, indices_size) = all_ptrs[0]; // smallest = indices
    let (_, out_ptr, _out_size) = all_ptrs[1]; // medium = output
    let (_, weight_ptr, weight_size) = all_ptrs[all_ptrs.len() - 1]; // largest = weight table

    // Determine dimensions
    // indices are typically int64 (8 bytes each) or int32 (4 bytes each)
    // Try int64 first
    let num_indices = indices_size / 8;
    // embedding_dim = weight_size / vocab_size, but we don't know vocab_size
    // Infer from output: output_size = num_indices * embedding_dim
    // So embedding_dim = out_size / num_indices
    let out_elements = _out_size / 4; // f32
    let embedding_dim = if num_indices > 0 {
        out_elements / num_indices
    } else {
        0
    };

    if embedding_dim == 0 || num_indices == 0 {
        eprintln!(
            "[TMatmul Fallback] indexSelect: can't determine dimensions (indices={}, emb_dim={})",
            num_indices, embedding_dim
        );
        return;
    }

    let weight = weight_ptr as *const f32;
    let indices = indices_ptr as *const i64;
    let output = out_ptr as *mut f32;
    let vocab_size = (weight_size / 4) / embedding_dim;

    eprintln!(
        "[TMatmul Fallback] indexSelect/embedding: vocab={}, dim={}, seq={}",
        vocab_size, embedding_dim, num_indices
    );

    for i in 0..num_indices {
        let idx = indices.add(i).read_unaligned() as usize;
        if idx >= vocab_size {
            eprintln!(
                "[TMatmul Fallback] indexSelect: index {} out of bounds (vocab={})",
                idx, vocab_size
            );
            continue;
        }
        // Copy embedding vector
        let src_base = idx * embedding_dim;
        let dst_base = i * embedding_dim;
        for d in 0..embedding_dim {
            let val = weight.add(src_base + d).read_unaligned();
            output.add(dst_base + d).write_unaligned(val);
        }
    }
    eprintln!(
        "[TMatmul Fallback] indexSelect executed for '{}'",
        kernel_name
    );
}

/// Execute matmul/GEMM kernel fallback.
#[cfg(feature = "intel")]
unsafe fn execute_matmul_kernel_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) {
    // GEMM kernels have various parameter layouts
    // Scan params for allocation pointers
    let mut all_ptrs: Vec<(usize, u64, usize)> = Vec::new();
    for pi in 0..16 {
        let p = *kernel_params.add(pi);
        if p.is_null() {
            break;
        }
        let as_u64 = (p as *const u64).read_unaligned();
        if let Some(size) = super::memory::get_alloc_size(as_u64 as usize) {
            all_ptrs.push((pi, as_u64, size));
        }
        let inner = scan_for_alloc_pointers(p as *const u8, 128);
        all_ptrs.extend(
            inner
                .into_iter()
                .map(|(off, ptr, sz)| (pi * 1000 + off, ptr, sz)),
        );
    }

    all_ptrs.sort_by_key(|&(_, ptr, _)| ptr);
    all_ptrs.dedup_by_key(|a| a.1);

    if all_ptrs.len() < 3 {
        eprintln!(
            "[TMatmul Fallback] matmul: need A, B, C pointers (found {})",
            all_ptrs.len()
        );
        return;
    }

    // Sort by alloc size: C (output, M*N), A (M*K), B (K*N) or similar
    all_ptrs.sort_by_key(|&(_, _, size)| size);

    // For simple matmul C = A @ B:
    // Try to infer dimensions from sizes
    // A: M*K, B: K*N, C: M*N (all in f32 = 4 bytes)
    let sizes: Vec<usize> = all_ptrs.iter().map(|&(_, _, s)| s / 4).collect();

    // Heuristic: try common dimension arrangements
    // If all same size, assume square matrices
    let (a_ptr, b_ptr, c_ptr, m, n, k);

    if all_ptrs.len() == 3 {
        let s0 = sizes[0]; // smallest
        let s1 = sizes[1];
        let s2 = sizes[2]; // largest

        // Try to factor: assume one common dimension K
        // A=M*K, B=K*N, C=M*N
        // If s0=s1=s2, all square (M=N=K=sqrt(size))
        if s0 == s1 && s1 == s2 {
            let side = (s0 as f64).sqrt() as usize;
            if side * side == s0 {
                m = side;
                n = side;
                k = side;
            } else {
                m = 1;
                n = s0;
                k = 1; // fallback: vector
            }
        } else {
            // Try: smallest is output (M*N), middle is one input, largest is other
            // Or: try to find K such that sizes work out
            // Simple heuristic: assume M=rows of smallest dimension
            let total = s2; // largest
            let side = (total as f64).sqrt() as usize;
            m = if s0 > 0 && total % s0 == 0 { s0 } else { side };
            n = if m > 0 { s2 / m } else { side };
            k = if m > 0 && s1 > 0 { s1 / m } else { side };
        }

        // C = largest, A and B are the others
        c_ptr = all_ptrs[2].1 as *mut f32;
        a_ptr = all_ptrs[0].1 as *const f32;
        b_ptr = all_ptrs[1].1 as *const f32;
    } else {
        eprintln!(
            "[TMatmul Fallback] matmul: unexpected number of pointers ({})",
            all_ptrs.len()
        );
        return;
    }

    if m == 0 || n == 0 || k == 0 {
        eprintln!(
            "[TMatmul Fallback] matmul: zero dimensions M={}, N={}, K={}",
            m, n, k
        );
        return;
    }

    eprintln!("[TMatmul Fallback] matmul: M={}, N={}, K={}", m, n, k);

    // C = A @ B (row-major)
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                let a_val = a_ptr.add(i * k + p).read_unaligned();
                let b_val = b_ptr.add(p * n + j).read_unaligned();
                sum += a_val * b_val;
            }
            c_ptr.add(i * n + j).write_unaligned(sum);
        }
    }
    eprintln!("[TMatmul Fallback] matmul executed for '{}'", kernel_name);
}

/// Execute layernorm/rmsnorm kernel fallback.
#[cfg(feature = "intel")]
unsafe fn execute_norm_kernel_fallback(
    kernel_name: &str,
    name_lower: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: u32,
    block_dim_y: u32,
) {
    let is_rmsnorm = name_lower.contains("rmsnorm") || name_lower.contains("rms_norm");

    if name_lower.contains("rms_norm_f32") || name_lower.contains("norm_f32") {
        if execute_ggml_norm_f32_fallback(
            kernel_name,
            name_lower,
            kernel_params,
            grid_dim_x,
            block_dim_y,
        ) {
            return;
        }
    }

    // vectorized_layer_norm_kernel signature:
    //   (int N, T_ACC epsilon, const T* X, const T* gamma, const T* beta, T* Y, T_ACC* mean, T_ACC* rstd)
    // kernel_params[0] = &N, [1] = &epsilon, [2] = &X, [3] = &gamma, [4] = &beta,
    // [5] = &Y, [6] = &mean, [7] = &rstd
    //
    // For welford/other norm kernels, scan only pointer params safely
    let param0 = *kernel_params.add(0);
    if param0.is_null() {
        eprintln!("[TMatmul Fallback] norm: null param0");
        return;
    }

    // Read hidden_size (N) from param 0
    let hidden_size = (param0 as *const i32).read_unaligned() as usize;
    if hidden_size == 0 || hidden_size > 65536 {
        eprintln!(
            "[TMatmul Fallback] norm: invalid hidden_size={}",
            hidden_size
        );
        return;
    }

    // Read epsilon from param 1
    let eps_param = *kernel_params.add(1);
    let epsilon = if !eps_param.is_null() {
        (eps_param as *const f32).read_unaligned() as f64
    } else {
        1e-5
    };

    // Read tensor pointers from params 2-5
    let mut ptrs: Vec<(*mut u8, usize)> = Vec::new(); // (ptr, alloc_size)
    for pi in 2..8 {
        let p = *kernel_params.add(pi);
        if p.is_null() {
            ptrs.push((std::ptr::null_mut(), 0));
            continue;
        }
        let ptr_val = (p as *const u64).read_unaligned();
        if let Some(size) = super::memory::get_alloc_size(ptr_val as usize) {
            ptrs.push((ptr_val as *mut u8, size));
        } else {
            ptrs.push((std::ptr::null_mut(), 0));
        }
    }

    // ptrs[0] = X (input), ptrs[1] = gamma (weight), ptrs[2] = beta (bias),
    // ptrs[3] = Y (output), ptrs[4] = mean, ptrs[5] = rstd
    let x_ptr = ptrs[0].0 as *const f32;
    let gamma_ptr = ptrs[1].0 as *const f32;
    let beta_ptr = ptrs[2].0 as *const f32;
    let y_ptr = ptrs[3].0 as *mut f32;

    if x_ptr.is_null() || y_ptr.is_null() {
        eprintln!("[TMatmul Fallback] norm: missing input or output pointer");
        return;
    }

    // Determine batch size from allocation
    let x_size = ptrs[0].1;
    let total_elements = x_size / 4; // f32
    let batch_size = if hidden_size > 0 {
        total_elements / hidden_size
    } else {
        1
    };

    let weight_ptr = if !gamma_ptr.is_null() {
        gamma_ptr
    } else {
        std::ptr::null()
    };
    let bias_ptr = if !beta_ptr.is_null() {
        beta_ptr
    } else {
        std::ptr::null()
    };

    let inp_ptr = x_ptr as u64;
    let out_ptr = y_ptr as u64;

    execute_norm_on_data(
        inp_ptr as *const f32,
        out_ptr as *mut f32,
        weight_ptr,
        bias_ptr,
        batch_size,
        hidden_size,
        is_rmsnorm,
        epsilon as f32,
    );
    eprintln!(
        "[TMatmul Fallback] norm executed (batch={}, hidden={}, eps={}, rmsnorm={}) for '{}'",
        batch_size, hidden_size, epsilon, is_rmsnorm, kernel_name
    );
}

#[cfg(feature = "intel")]
unsafe fn execute_norm_on_data(
    inp: *const f32,
    out: *mut f32,
    weight: *const f32,
    bias: *const f32,
    batch_size: usize,
    hidden_size: usize,
    is_rmsnorm: bool,
    eps: f32,
) {
    for b in 0..batch_size {
        let base = b * hidden_size;

        if is_rmsnorm {
            // RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight
            let mut sq_sum = 0.0f32;
            for h in 0..hidden_size {
                let x = inp.add(base + h).read_unaligned();
                sq_sum += x * x;
            }
            let rms = (sq_sum / hidden_size as f32 + eps).sqrt();
            for h in 0..hidden_size {
                let x = inp.add(base + h).read_unaligned();
                let w = if !weight.is_null() {
                    weight.add(h).read_unaligned()
                } else {
                    1.0
                };
                out.add(base + h).write_unaligned(x / rms * w);
            }
        } else {
            // LayerNorm: out = (x - mean) / sqrt(var + eps) * weight + bias
            let mut sum = 0.0f32;
            for h in 0..hidden_size {
                sum += inp.add(base + h).read_unaligned();
            }
            let mean = sum / hidden_size as f32;

            let mut var_sum = 0.0f32;
            for h in 0..hidden_size {
                let diff = inp.add(base + h).read_unaligned() - mean;
                var_sum += diff * diff;
            }
            let std_dev = (var_sum / hidden_size as f32 + eps).sqrt();

            for h in 0..hidden_size {
                let x = inp.add(base + h).read_unaligned();
                let normalized = (x - mean) / std_dev;
                let w = if !weight.is_null() {
                    weight.add(h).read_unaligned()
                } else {
                    1.0
                };
                let b = if !bias.is_null() {
                    bias.add(h).read_unaligned()
                } else {
                    0.0
                };
                out.add(base + h).write_unaligned(normalized * w + b);
            }
        }
    }
}

// Implement cuLaunchKernelEx by unwrapping the CUlaunchConfig and delegating to launch_kernel
#[cfg(feature = "intel")]
pub(crate) fn cuLaunchKernelEx(
    config: *const cuda_types::cuda::CUlaunchConfig,
    f: &super::module::ZeKernel,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> ze_result_t {
    if config.is_null() {
        return ze_result_t::ZE_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    let cfg = unsafe { &*config };
    let grid_x = cfg.gridDimX;
    let grid_y = cfg.gridDimY;
    let grid_z = cfg.gridDimZ;
    let block_x = cfg.blockDimX;
    let block_y = cfg.blockDimY;
    let block_z = cfg.blockDimZ;
    let shmem = cfg.sharedMemBytes;
    // In virtual backend, stream is usually null; pass a placeholder
    let stream = ze_command_queue_handle_t(::core::ptr::null_mut());
    unsafe {
        launch_kernel(
            f,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            shmem,
            stream,
            kernel_params,
            extra,
        )
    }
}

// Normalized name expected by cuda_normalize_fn!(function::launch_kernel_ex)
#[cfg(feature = "intel")]
pub(crate) fn launch_kernel_ex(
    config: *const cuda_types::cuda::CUlaunchConfig,
    f: &super::module::ZeKernel,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> ze_result_t {
    cuLaunchKernelEx(config, f, kernel_params, extra)
}

// Helper function to get or create a command list for a stream
#[cfg(feature = "intel")]
unsafe fn get_or_create_command_list_for_stream(
    stream: ze_command_queue_handle_t,
) -> ze_command_list_handle_t {
    // In a real implementation, you'd have a way to track command lists per stream
    // For now, we'll create a new one (this would leak in a real implementation)

    // Get the device and context from the stream
    let device = get_device_from_stream(stream);
    let context = get_context_from_stream(stream);

    let desc = ze_command_list_desc_t {
        stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        pNext: ptr::null(),
        commandQueueGroupOrdinal: 0, // Default queue group
        flags: 0,
    };

    let mut command_list = ze_command_list_handle_t(ptr::null_mut());
    let result = zeCommandListCreate(context, device, &desc, &mut command_list);

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return ze_command_list_handle_t(ptr::null_mut());
    }

    command_list
}

#[cfg(feature = "intel")]
unsafe fn get_device_from_stream(stream: ze_command_queue_handle_t) -> ze_device_handle_t {
    // Get device from global state
    // If stream is null or we can't find the device, use the primary device
    if let Ok(gs) = crate::r#impl::driver::global_state() {
        if let Some(dev0) = gs.devices.get(0) {
            let (ctx, _raw_ctx) = dev0.primary_context();
            return ctx.device;
        }
    }
    ze_device_handle_t(ptr::null_mut())
}

#[cfg(feature = "intel")]
unsafe fn get_context_from_stream(stream: ze_command_queue_handle_t) -> ze_context_handle_t {
    // Get context from global state
    // If stream is null or we can't find the context, use the primary context
    if let Ok(gs) = crate::r#impl::driver::global_state() {
        if let Some(dev0) = gs.devices.get(0) {
            let (ctx, _raw_ctx) = dev0.primary_context();
            return ctx.context;
        }
    }
    ze_context_handle_t(ptr::null_mut())
}

// Tenstorrent function implementations
#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn get_attribute(
    pi: *mut i32,
    attrib: CUfunction_attribute,
    func: *mut crate::r#impl::module::TtKernel,
) -> CUresult {
    if pi.is_null() || func.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    // For Tenstorrent, return placeholder values for function attributes
    let result = match attrib {
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK => 1024,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES => 0,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES => 0,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES => 0,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS => 32,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_PTX_VERSION => 75,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_BINARY_VERSION => 75,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_CACHE_MODE_CA => 0,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES => 65536,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT => 0,
        _ => return Err(CUerror::INVALID_VALUE),
    };

    unsafe { *pi = result };
    Ok(())
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
pub(crate) fn launch_kernel(
    f: *mut crate::r#impl::module::TtKernel,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    block_dim_y: ::core::ffi::c_uint,
    block_dim_z: ::core::ffi::c_uint,
    shared_mem_bytes: ::core::ffi::c_uint,
    stream: *mut ::core::ffi::c_void,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> CUresult {
    if f.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    // For Tenstorrent, implement kernel launch
    // In a real implementation, this would:
    // 1. Set up the kernel parameters
    // 2. Configure the grid and block dimensions
    // 3. Launch the kernel on the Tenstorrent device
    // 4. Handle synchronization based on the stream

    let _kernel = unsafe { &*f };

    // Process kernel parameters if provided
    if !kernel_params.is_null() {
        unsafe {
            let mut param_index = 0;
            let mut current_param = kernel_params;

            while !(*current_param).is_null() {
                let _param_value = *current_param;
                // In a real implementation, set kernel argument at param_index

                param_index += 1;
                current_param = current_param.add(1);
            }
        }
    }

    // Process extra parameters if provided
    if !extra.is_null() {
        unsafe {
            let mut i = 0;
            loop {
                let key = *extra.add(i);
                if key.is_null() {
                    break;
                }

                let _key_value = key as usize;
                let _value_ptr = extra.add(i + 1);
                let _value = *_value_ptr;

                // Process extra parameters as needed

                i += 2;
            }
        }
    }

    // Placeholder for actual Tenstorrent kernel launch
    // This would interface with the tt_runtime_sys to launch the kernel

    // Suppress unused parameter warnings
    let _ = (grid_dim_x, grid_dim_y, grid_dim_z);
    let _ = (block_dim_x, block_dim_y, block_dim_z);
    let _ = shared_mem_bytes;
    let _ = stream;

    Ok(())
}

// NVIDIA backend function implementations - passthrough to real libcuda.so
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get_attribute(
    pi: &mut ::core::ffi::c_int,
    attrib: cuda_types::cuda::CUfunction_attribute,
    hfunc: &super::module::NvidiaKernel,
) -> CUresult {
    let result = nvidia_runtime_sys::cuFuncGetAttribute(pi, attrib, hfunc.cuda_function);
    if result != 0 {
        return Err(CUerror::UNKNOWN);
    }
    Ok(())
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn set_attribute(
    hfunc: &super::module::NvidiaKernel,
    attrib: cuda_types::cuda::CUfunction_attribute,
    value: ::core::ffi::c_int,
) -> CUresult {
    let result = nvidia_runtime_sys::cuFuncSetAttribute(hfunc.cuda_function, attrib, value);
    if result != 0 {
        return Err(CUerror::UNKNOWN);
    }
    Ok(())
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_log_bitnet_route_for_native_launch(kernel_name: &str) {
    if !super::bitnet_disagg::enabled_from_env() {
        return;
    }

    let route_config = super::bitnet_disagg::config_from_env();
    let decision = super::bitnet_disagg::classify_kernel_name(kernel_name, &route_config);
    // The NVIDIA backend currently observes the BitNet split while forwarding
    // launches to native CUDA. The Intel/tmatmul backend owns actual CXL
    // matmul diversion, so keep this field false here to avoid claiming a
    // hardware matmul submit happened in the NVIDIA pass-through path.
    let hardware_matmul_enabled = false;
    let hardware = match super::bitnet_disagg::tmatmul_backend_from_env() {
        Ok(Some(super::bitnet_disagg::TmatmulBackend::Xrt)) => {
            super::bitnet_disagg::RouteHardware::xrt(hardware_matmul_enabled)
        }
        Ok(Some(super::bitnet_disagg::TmatmulBackend::Cxl)) | Ok(None) => {
            let cxl_enabled =
                nvidia_env_truthy("HETGPU_CXL_TMATMUL") || nvidia_env_truthy("HETGPU_TMATMUL_CXL");
            super::bitnet_disagg::RouteHardware::cxl(cxl_enabled, hardware_matmul_enabled)
        }
        Err(err) => {
            eprintln!("[BitNet Disagg][NVIDIA] invalid tmatmul backend: {err}");
            return;
        }
    };
    if let Err(err) = super::bitnet_disagg::append_route_log_from_env(&decision, hardware) {
        eprintln!(
            "[BitNet Disagg][NVIDIA] route log failed for '{}': {}",
            kernel_name, err
        );
    }

    if decision.route == super::bitnet_disagg::BitnetRoute::CxlTmatmul
        && nvidia_env_truthy("HETGPU_BITNET_ROUTE_TRACE")
    {
        eprintln!(
            "[BitNet Disagg][NVIDIA] '{}' classified as CXL tmatmul candidate via {}; native CUDA pass-through remains active",
            kernel_name,
            decision.source.as_str()
        );
    }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_env_truthy(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|value| {
        let trimmed = value.trim();
        if let Some(hex) = trimmed
            .strip_prefix("0x")
            .or_else(|| trimmed.strip_prefix("0X"))
        {
            usize::from_str_radix(hex, 16).ok()
        } else {
            trimmed.parse::<usize>().ok()
        }
    })
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_env_u64_any(names: &[&str]) -> Option<u64> {
    names.iter().find_map(|name| {
        let value = std::env::var(name).ok()?;
        let trimmed = value.trim();
        if let Some(hex) = trimmed
            .strip_prefix("0x")
            .or_else(|| trimmed.strip_prefix("0X"))
        {
            u64::from_str_radix(hex, 16).ok()
        } else {
            trimmed.parse::<u64>().ok()
        }
    })
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_kernel_param_u64_snapshot(
    kernel_params: *mut *mut ::core::ffi::c_void,
    num_params: usize,
) -> Vec<u64> {
    if kernel_params.is_null() {
        return Vec::new();
    }

    let mut params = Vec::with_capacity(num_params);
    unsafe {
        for index in 0..num_params {
            let slot = *kernel_params.add(index);
            if slot.is_null() || (slot as usize) < 0x1000 {
                params.push(0);
            } else {
                params.push((slot as *const u64).read_unaligned());
            }
        }
    }
    params
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_kimi_concordia_param_snapshot(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Vec<u64> {
    let name = kernel_name.to_ascii_lowercase();
    if !super::kimi_concordia::is_kimi_stateful_kernel_name(kernel_name)
        && !name.contains("lora")
        && !name.contains("adapter")
    {
        return Vec::new();
    }
    nvidia_kernel_param_u64_snapshot(kernel_params, 8)
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NvidiaCxlMatmulLayout {
    name: &'static str,
    matrix_param: usize,
    vector_param: usize,
    output_param: usize,
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NvidiaCxlMmvqShape {
    ncols_x: i32,
    nrows_x: i32,
    nrows_y: i32,
    nrows_dst: i32,
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NvidiaCudaUint3 {
    x: u32,
    y: u32,
    z: u32,
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NvidiaModernMmvqLayout {
    ncols_x: u32,
    nchannels_y: NvidiaCudaUint3,
    stride_row_x: u32,
    stride_col_y: u32,
    stride_col_dst: u32,
    stride_channel_x: u32,
    stride_channel_y: u32,
    stride_channel_dst: u32,
    sample_ratio: NvidiaCudaUint3,
    stride_sample_x: u32,
    stride_sample_y: u32,
    stride_sample_dst: u32,
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NvidiaModernMoeMmvqLayout {
    ncols_x: u32,
    nchannels_y: NvidiaCudaUint3,
    nrows_x: u32,
    stride_row_x: u32,
    stride_col_y: u32,
    stride_col_dst: u32,
    stride_channel_x: u32,
    stride_channel_y: u32,
    stride_channel_dst: u32,
    ncols_dst: u32,
    ids_stride: u32,
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[derive(Clone, Debug)]
struct NvidiaModernMmvqPlan {
    matrix_ptr: usize,
    activation_ptr: usize,
    output_ptr: usize,
    signature: super::iq1s_tmatmul::GgmlType19VecSignature,
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NvidiaCxlMmqShape {
    ne00: i32,
    ne01: i32,
    stride01: i32,
    ne10: i32,
    ne11: i32,
    stride11: i32,
    ne0: i32,
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_cxl_mmvq_contract_error(shape: NvidiaCxlMmvqShape) -> Option<String> {
    const DIM: i32 = 2048;
    if shape.ncols_x != DIM
        || shape.nrows_x != DIM
        || shape.nrows_y != DIM
        || shape.nrows_dst != DIM
    {
        return Some(format!(
            "mul_mat_vec_q shape ncols_x={} nrows_x={} nrows_y={} nrows_dst={} does not match the current tmatmul square contract, which requires {DIM} for every dimension",
            shape.ncols_x, shape.nrows_x, shape.nrows_y, shape.nrows_dst
        ));
    }

    Some(
        "mul_mat_vec_q uses the GGML block-quantized matrix/Q8_1 activation ABI; tmatmul_go_nvint8 requires a dense NVINT8 matrix and a 16-bit activation vector"
            .to_string(),
    )
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_cxl_mmq_contract_error(shape: NvidiaCxlMmqShape) -> String {
    const DIM: i32 = 2048;
    if shape.ne00 != DIM
        || shape.ne01 != DIM
        || shape.ne10 != DIM
        || shape.ne11 != 1
        || shape.ne0 != DIM
    {
        return format!(
            "mul_mat_q shape ne00={} ne01={} stride01={} ne10={} ne11={} stride11={} ne0={} does not match the current tmatmul contract, which requires 2048x2048 matrix-vector operations",
            shape.ne00,
            shape.ne01,
            shape.stride01,
            shape.ne10,
            shape.ne11,
            shape.stride11,
            shape.ne0,
        );
    }

    format!(
        "mul_mat_q shape ne00={} ne01={} stride01={} ne10={} ne11={} stride11={} ne0={} uses the GGML packed quantized matrix/Q8_1 activation ABI; tmatmul_go_nvint8 requires a dense NVINT8 matrix and a 16-bit activation vector",
        shape.ne00,
        shape.ne01,
        shape.stride01,
        shape.ne10,
        shape.ne11,
        shape.stride11,
        shape.ne0,
    )
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_cxl_matmul_layout(name_lower: &str) -> Option<NvidiaCxlMatmulLayout> {
    if name_lower.contains("mul_mat_q_stream_k_fixup") || name_lower.contains("stream_k_fixup") {
        return None;
    }
    if name_lower.contains("mul_mat_vec_q") {
        return Some(NvidiaCxlMatmulLayout {
            name: "mul_mat_vec_q",
            matrix_param: nvidia_env_usize("HETGPU_TMATMUL_MATRIX_PARAM").unwrap_or(0),
            vector_param: nvidia_env_usize("HETGPU_TMATMUL_VECTOR_PARAM").unwrap_or(1),
            output_param: nvidia_env_usize("HETGPU_TMATMUL_OUTPUT_PARAM").unwrap_or(2),
        });
    }
    if !name_lower.contains("mul_mat_q") {
        return None;
    }

    Some(NvidiaCxlMatmulLayout {
        name: "mul_mat_q",
        matrix_param: nvidia_env_usize("HETGPU_TMATMUL_MATRIX_PARAM").unwrap_or(0),
        vector_param: nvidia_env_usize("HETGPU_TMATMUL_VECTOR_PARAM").unwrap_or(1),
        output_param: nvidia_env_usize("HETGPU_TMATMUL_OUTPUT_PARAM").unwrap_or(2),
    })
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_cxl_generate_matmul_assembly(kernel_name: &str, layout: NvidiaCxlMatmulLayout) -> String {
    format!(
        "; IA-780I hardware matmul fallback generated from NVIDIA named kernel
         ; kernel: {kernel_name}
         ; layout: {layout_name} matrix=PARAM_{matrix} vector=PARAM_{vector} output=PARAM_{output}
         ; BIND PARAM_{matrix} matrix
         ; BIND PARAM_{vector} vector
         ; BIND PARAM_{output} output
         ldv v0,PARAM_{vector}
         tmatmul_import v0
         tmatmul_go_nvint8 PARAM_{matrix},4
         tmatmul_export v1
         sv v1,PARAM_{output}
         stall
",
        kernel_name = kernel_name,
        layout_name = layout.name,
        matrix = layout.matrix_param,
        vector = layout.vector_param,
        output = layout.output_param,
    )
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn nvidia_cxl_read_pointer_param(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
    kernel_name: &str,
) -> Result<usize, String> {
    if kernel_params.is_null() {
        return Err(format!("kernel '{kernel_name}' has null kernel_params"));
    }
    let slot = *kernel_params.add(index);
    if slot.is_null() || (slot as usize) < 0x1000 {
        return Err(format!(
            "kernel '{kernel_name}' PARAM_{index} has invalid parameter slot {:#x}",
            slot as usize
        ));
    }
    let ptr_value = (slot as *const u64).read_unaligned() as usize;
    if ptr_value < 0x1000 {
        return Err(format!(
            "kernel '{kernel_name}' PARAM_{index} has invalid pointer value {ptr_value:#x}"
        ));
    }
    Ok(ptr_value)
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn nvidia_cxl_read_i32_param(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
    kernel_name: &str,
) -> Result<i32, String> {
    if kernel_params.is_null() {
        return Err(format!("kernel '{kernel_name}' has null kernel_params"));
    }
    let slot = *kernel_params.add(index);
    if slot.is_null() || (slot as usize) < 0x1000 {
        return Err(format!(
            "kernel '{kernel_name}' PARAM_{index} has invalid scalar slot {:#x}",
            slot as usize
        ));
    }
    Ok((slot as *const i32).read_unaligned())
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn nvidia_cxl_read_u32_param(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
    kernel_name: &str,
) -> Result<u32, String> {
    if kernel_params.is_null() {
        return Err(format!("kernel '{kernel_name}' has null kernel_params"));
    }
    let slot = *kernel_params.add(index);
    if slot.is_null() || (slot as usize) < 0x1000 {
        return Err(format!(
            "kernel '{kernel_name}' PARAM_{index} has invalid scalar slot {:#x}",
            slot as usize
        ));
    }
    Ok((slot as *const u32).read_unaligned())
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn nvidia_cxl_read_uint3_param(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
    kernel_name: &str,
) -> Result<NvidiaCudaUint3, String> {
    if kernel_params.is_null() {
        return Err(format!("kernel '{kernel_name}' has null kernel_params"));
    }
    let slot = *kernel_params.add(index);
    if slot.is_null() || (slot as usize) < 0x1000 {
        return Err(format!(
            "kernel '{kernel_name}' PARAM_{index} has invalid uint3 slot {:#x}",
            slot as usize
        ));
    }
    let words = slot as *const u32;
    Ok(NvidiaCudaUint3 {
        x: words.read_unaligned(),
        y: words.add(1).read_unaligned(),
        z: words.add(2).read_unaligned(),
    })
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_modern_iq1s_mmvq_has_fusion(kernel_name: &str) -> bool {
    kernel_name
        .to_ascii_lowercase()
        .contains("ggml_type19eli1elb1")
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_is_modern_iq1s_mmvq(kernel_name: &str) -> bool {
    let name = kernel_name.to_ascii_lowercase();
    name.contains("mul_mat_vec_q")
        && name.contains("ggml_type19eli1el")
        && name.contains("ggml_cuda_mm_fusion_args_device")
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_is_modern_iq1s_moe_mmvq(kernel_name: &str) -> bool {
    let name = kernel_name.to_ascii_lowercase();
    name.contains("mul_mat_vec_q_moe")
        && name.contains("ggml_type19")
        && name.contains("eli2e")
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_checked_pointer_offset(
    base: usize,
    elements: u64,
    element_bytes: usize,
    field: &str,
) -> Result<usize, String> {
    let elements = usize::try_from(elements).map_err(|_| format!("{field} does not fit usize"))?;
    let bytes = elements
        .checked_mul(element_bytes)
        .ok_or_else(|| format!("{field} byte offset overflow"))?;
    base.checked_add(bytes)
        .ok_or_else(|| format!("{field} pointer overflow"))
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_plan_modern_iq1s_mmvq(
    layout: NvidiaModernMmvqLayout,
    matrix_base: usize,
    activation_base: usize,
    output_base: usize,
    expert_ids: &[i32],
    grid: (u32, u32, u32),
) -> Result<Vec<NvidiaModernMmvqPlan>, String> {
    use super::iq1s_tmatmul::{IQ1S_BLOCK_BYTES, IQ1S_BLOCK_VALUES, Q8_1_BLOCK_BYTES};

    if matrix_base == 0 || activation_base == 0 || output_base == 0 {
        return Err("modern IQ1_S MMVQ contains a null data pointer".to_string());
    }
    if grid.0 == 0 || grid.1 == 0 || grid.2 == 0 {
        return Err("modern IQ1_S MMVQ grid dimensions must be positive".to_string());
    }
    if expert_ids.len() != grid.1 as usize {
        return Err(format!(
            "modern IQ1_S MMVQ has {} expert ids for grid.y={}",
            expert_ids.len(),
            grid.1
        ));
    }
    if grid.2 != 1 {
        return Err(format!(
            "modern single-token IQ1_S MMVQ requires grid.z=1, got {}",
            grid.2
        ));
    }
    if layout.ncols_x == 0 || layout.stride_col_dst == 0 || layout.stride_channel_dst == 0 {
        return Err("modern IQ1_S MMVQ dimensions must be positive".to_string());
    }
    let expected_stride = layout.ncols_x / IQ1S_BLOCK_VALUES as u32;
    if !layout.ncols_x.is_multiple_of(IQ1S_BLOCK_VALUES as u32)
        || layout.stride_row_x != expected_stride
    {
        return Err(format!(
            "modern IQ1_S MMVQ stride_row_x={} is not the contiguous IQ1_S row stride {} for ncols_x={}",
            layout.stride_row_x, expected_stride, layout.ncols_x
        ));
    }
    if layout.nchannels_y.z == 0 || layout.sample_ratio.z == 0 {
        return Err("modern IQ1_S MMVQ fast-divisor metadata has a zero divisor".to_string());
    }
    if layout.stride_col_y == 0 {
        return Err("modern IQ1_S MMVQ stride_col_y must be positive".to_string());
    }
    let nrows_x = layout
        .stride_channel_x
        .checked_div(layout.stride_row_x)
        .ok_or("modern IQ1_S MMVQ stride_row_x must divide stride_channel_x")?;
    if nrows_x
        .checked_mul(layout.stride_row_x)
        .filter(|span| *span == layout.stride_channel_x)
        .is_none()
        || nrows_x != layout.stride_channel_dst
    {
        return Err(format!(
            "modern IQ1_S MMVQ per-expert row layout is inconsistent: stride_channel_x={}, stride_row_x={}, stride_channel_dst={}",
            layout.stride_channel_x, layout.stride_row_x, layout.stride_channel_dst
        ));
    }
    // In llama.cpp's single-token MUL_MAT_ID kernel, selected experts are
    // separated by stride_channel_dst. stride_col_dst is the pitch between
    // token columns and may span the complete destination tensor, including
    // channels not selected by this launch.
    if layout.stride_col_dst < nrows_x {
        return Err(format!(
            "modern IQ1_S MMVQ stride_col_dst={} is smaller than one {}-row output column",
            layout.stride_col_dst, nrows_x
        ));
    }
    if layout.stride_sample_x == 0
        || !layout
            .stride_sample_x
            .is_multiple_of(layout.stride_channel_x)
    {
        return Err(format!(
            "modern IQ1_S MMVQ stride_sample_x={} does not contain a whole number of expert matrices with stride_channel_x={}",
            layout.stride_sample_x, layout.stride_channel_x
        ));
    }
    let expert_count = layout.stride_sample_x / layout.stride_channel_x;

    let mut plans = Vec::with_capacity(expert_ids.len() * grid.2 as usize);
    for sample_dst in 0..grid.2 {
        let sample_x = sample_dst / layout.sample_ratio.z;
        for (channel_dst, expert_id) in expert_ids.iter().copied().enumerate() {
            let expert = u32::try_from(expert_id)
                .map_err(|_| format!("modern IQ1_S MMVQ expert id {expert_id} is negative"))?;
            if expert >= expert_count {
                return Err(format!(
                    "modern IQ1_S MMVQ expert id {expert} exceeds expert count {expert_count}"
                ));
            }
            let channel_dst = channel_dst as u32;
            let channel_y = channel_dst % layout.nchannels_y.z;
            let matrix_blocks = u64::from(sample_x)
                .checked_mul(u64::from(layout.stride_sample_x))
                .and_then(|offset| {
                    u64::from(expert)
                        .checked_mul(u64::from(layout.stride_channel_x))
                        .and_then(|expert_offset| offset.checked_add(expert_offset))
                })
                .ok_or("modern IQ1_S MMVQ matrix offset overflow")?;
            let activation_blocks = u64::from(sample_dst)
                .checked_mul(u64::from(layout.stride_sample_y))
                .and_then(|offset| {
                    u64::from(channel_y)
                        .checked_mul(u64::from(layout.stride_channel_y))
                        .and_then(|channel_offset| offset.checked_add(channel_offset))
                })
                .ok_or("modern IQ1_S MMVQ activation offset overflow")?;
            let output_elements = u64::from(sample_dst)
                .checked_mul(u64::from(layout.stride_sample_dst))
                .and_then(|offset| {
                    u64::from(channel_dst)
                        .checked_mul(u64::from(layout.stride_channel_dst))
                        .and_then(|channel_offset| offset.checked_add(channel_offset))
                })
                .ok_or("modern IQ1_S MMVQ output offset overflow")?;
            plans.push(NvidiaModernMmvqPlan {
                matrix_ptr: nvidia_checked_pointer_offset(
                    matrix_base,
                    matrix_blocks,
                    IQ1S_BLOCK_BYTES,
                    "matrix",
                )?,
                activation_ptr: nvidia_checked_pointer_offset(
                    activation_base,
                    activation_blocks,
                    Q8_1_BLOCK_BYTES,
                    "activation",
                )?,
                output_ptr: nvidia_checked_pointer_offset(
                    output_base,
                    output_elements,
                    std::mem::size_of::<f32>(),
                    "output",
                )?,
                signature: super::iq1s_tmatmul::GgmlType19VecSignature {
                    kernel: "mul_mat_vec_q_ggml_type19".to_string(),
                    ncols_x: u64::from(layout.ncols_x),
                    nrows_x: u64::from(nrows_x),
                    nrows_y: u64::from(layout.ncols_x),
                    nrows_dst: u64::from(nrows_x),
                }
                .validate()?,
            });
        }
    }
    Ok(plans)
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_plan_modern_iq1s_moe_mmvq(
    layout: NvidiaModernMoeMmvqLayout,
    matrix_base: usize,
    activation_base: usize,
    output_base: usize,
    expert_ids: &[i32],
    grid: (u32, u32, u32),
) -> Result<Vec<NvidiaModernMmvqPlan>, String> {
    use super::iq1s_tmatmul::{IQ1S_BLOCK_BYTES, IQ1S_BLOCK_VALUES, Q8_1_BLOCK_BYTES};

    if matrix_base < 0x1000 || activation_base < 0x1000 || output_base < 0x1000 {
        return Err("modern IQ1_S MoE MMVQ contains an invalid data pointer".to_string());
    }
    if layout.ncols_x == 0
        || layout.nrows_x == 0
        || layout.nchannels_y.z == 0
        || layout.ncols_dst < 2
        || layout.ncols_dst > 16
    {
        return Err("modern IQ1_S MoE MMVQ dimensions or active-token count are invalid".to_string());
    }
    if grid.1 == 0 || grid.2 != 1 {
        return Err("modern IQ1_S MoE MMVQ grid.y must be positive and grid.z must equal one".to_string());
    }
    let expected_grid_x = layout.nrows_x.div_ceil(2);
    if grid.0 != expected_grid_x {
        return Err(format!(
            "modern IQ1_S MoE MMVQ grid.x={} does not match two-row kernel grid.x={expected_grid_x}",
            grid.0
        ));
    }
    let expected_row_stride = layout.ncols_x / IQ1S_BLOCK_VALUES as u32;
    if !layout
        .ncols_x
        .is_multiple_of(IQ1S_BLOCK_VALUES as u32)
        || layout.stride_row_x != expected_row_stride
    {
        return Err("modern IQ1_S MoE MMVQ has a noncontiguous IQ1_S row stride".to_string());
    }
    let expected_expert_stride = layout
        .nrows_x
        .checked_mul(layout.stride_row_x)
        .ok_or("modern IQ1_S MoE MMVQ expert stride overflow")?;
    if layout.stride_channel_x != expected_expert_stride
        || layout.stride_channel_dst != layout.nrows_x
    {
        return Err("modern IQ1_S MoE MMVQ matrix/output channel strides do not match nrows_x".to_string());
    }
    let expected_q8_channel_stride = layout.ncols_x / 32;
    let expected_q8_column_stride = layout
        .nchannels_y
        .z
        .checked_mul(expected_q8_channel_stride)
        .ok_or("modern IQ1_S MoE MMVQ activation column stride overflow")?;
    if layout.stride_col_y != expected_q8_column_stride
        || layout.stride_channel_y != expected_q8_channel_stride
    {
        return Err(format!(
            "modern IQ1_S MoE MMVQ activation strides col={} channel={} do not match Q8_1 storage col={} channel={}",
            layout.stride_col_y,
            layout.stride_channel_y,
            expected_q8_column_stride,
            expected_q8_channel_stride,
        ));
    }
    if layout.ids_stride < grid.1 {
        return Err("modern IQ1_S MoE MMVQ ids_stride is smaller than grid.y".to_string());
    }
    let required_ids = layout
        .ncols_dst
        .checked_sub(1)
        .and_then(|token| token.checked_mul(layout.ids_stride))
        .and_then(|base| base.checked_add(grid.1))
        .ok_or("modern IQ1_S MoE MMVQ ids extent overflow")?;
    if expert_ids.len() < required_ids as usize {
        return Err(format!(
            "modern IQ1_S MoE MMVQ has {} ids but its strided layout requires {required_ids}",
            expert_ids.len()
        ));
    }

    let mut plans = Vec::with_capacity(layout.ncols_dst as usize * grid.1 as usize);
    for token in 0..layout.ncols_dst {
        for channel_dst in 0..grid.1 {
            let id_index = token
                .checked_mul(layout.ids_stride)
                .and_then(|base| base.checked_add(channel_dst))
                .ok_or("modern IQ1_S MoE MMVQ id index overflow")?;
            let expert_id = expert_ids[id_index as usize];
            let expert = u64::try_from(expert_id)
                .map_err(|_| format!("modern IQ1_S MoE MMVQ expert id {expert_id} is negative"))?;
            let channel_y = channel_dst % layout.nchannels_y.z;
            let matrix_blocks = expert
                .checked_mul(u64::from(layout.stride_channel_x))
                .ok_or("modern IQ1_S MoE MMVQ matrix offset overflow")?;
            let activation_blocks = u64::from(token)
                .checked_mul(u64::from(layout.stride_col_y))
                .and_then(|offset| {
                    u64::from(channel_y)
                        .checked_mul(u64::from(layout.stride_channel_y))
                        .and_then(|channel_offset| offset.checked_add(channel_offset))
                })
                .ok_or("modern IQ1_S MoE MMVQ activation offset overflow")?;
            let output_elements = u64::from(token)
                .checked_mul(u64::from(layout.stride_col_dst))
                .and_then(|offset| {
                    u64::from(channel_dst)
                        .checked_mul(u64::from(layout.stride_channel_dst))
                        .and_then(|channel_offset| offset.checked_add(channel_offset))
                })
                .ok_or("modern IQ1_S MoE MMVQ output offset overflow")?;
            plans.push(NvidiaModernMmvqPlan {
                matrix_ptr: nvidia_checked_pointer_offset(
                    matrix_base,
                    matrix_blocks,
                    IQ1S_BLOCK_BYTES,
                    "MoE matrix",
                )?,
                activation_ptr: nvidia_checked_pointer_offset(
                    activation_base,
                    activation_blocks,
                    Q8_1_BLOCK_BYTES,
                    "MoE activation",
                )?,
                output_ptr: nvidia_checked_pointer_offset(
                    output_base,
                    output_elements,
                    std::mem::size_of::<f32>(),
                    "MoE output",
                )?,
                signature: super::iq1s_tmatmul::GgmlType19VecSignature {
                    kernel: "mul_mat_vec_q_moe_ggml_type19".to_string(),
                    ncols_x: u64::from(layout.ncols_x),
                    nrows_x: u64::from(layout.nrows_x),
                    nrows_y: u64::from(layout.ncols_x),
                    nrows_dst: u64::from(layout.nrows_x),
                }
                .validate()?,
            });
        }
    }
    Ok(plans)
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn nvidia_cxl_read_mmvq_shape(
    kernel_params: *mut *mut ::core::ffi::c_void,
    kernel_name: &str,
) -> Result<NvidiaCxlMmvqShape, String> {
    Ok(NvidiaCxlMmvqShape {
        ncols_x: nvidia_cxl_read_i32_param(kernel_params, 3, kernel_name)?,
        nrows_x: nvidia_cxl_read_i32_param(kernel_params, 4, kernel_name)?,
        nrows_y: nvidia_cxl_read_i32_param(kernel_params, 5, kernel_name)?,
        nrows_dst: nvidia_cxl_read_i32_param(kernel_params, 6, kernel_name)?,
    })
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn nvidia_cxl_read_mmq_shape(
    kernel_params: *mut *mut ::core::ffi::c_void,
    kernel_name: &str,
) -> Result<NvidiaCxlMmqShape, String> {
    Ok(NvidiaCxlMmqShape {
        ne00: nvidia_cxl_read_i32_param(kernel_params, 4, kernel_name)?,
        ne01: nvidia_cxl_read_i32_param(kernel_params, 5, kernel_name)?,
        stride01: nvidia_cxl_read_i32_param(kernel_params, 6, kernel_name)?,
        ne10: nvidia_cxl_read_i32_param(kernel_params, 7, kernel_name)?,
        ne11: nvidia_cxl_read_i32_param(kernel_params, 8, kernel_name)?,
        stride11: nvidia_cxl_read_i32_param(kernel_params, 9, kernel_name)?,
        ne0: nvidia_cxl_read_i32_param(kernel_params, 10, kernel_name)?,
    })
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_iq1s_u64_field(field: &str, value: i32) -> Result<u64, String> {
    u64::try_from(value).map_err(|_| format!("{field} has negative value {value}"))
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_iq1s_signature(
    kernel_name: &str,
    shape: NvidiaCxlMmqShape,
) -> Result<super::iq1s_tmatmul::GgmlType19Signature, String> {
    let name_lower = kernel_name.to_ascii_lowercase();
    if !name_lower.contains("mul_mat_q")
        || name_lower.contains("mul_mat_vec_q")
        || !name_lower.contains("ggml_type19")
    {
        return Err(format!(
            "kernel '{kernel_name}' is not the qualified IQ1_S mul_mat_q symbol"
        ));
    }

    super::iq1s_tmatmul::GgmlType19Signature {
        kernel: kernel_name.to_string(),
        ne00: nvidia_iq1s_u64_field("ne00", shape.ne00)?,
        ne01: nvidia_iq1s_u64_field("ne01", shape.ne01)?,
        stride01: nvidia_iq1s_u64_field("stride01", shape.stride01)?,
        ne10: nvidia_iq1s_u64_field("ne10", shape.ne10)?,
        ne11: nvidia_iq1s_u64_field("ne11", shape.ne11)?,
        stride11: nvidia_iq1s_u64_field("stride11", shape.stride11)?,
        ne0: nvidia_iq1s_u64_field("ne0", shape.ne0)?,
    }
    .validate()
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_iq1s_vec_signature(
    kernel_name: &str,
    shape: NvidiaCxlMmvqShape,
) -> Result<super::iq1s_tmatmul::GgmlType19VecSignature, String> {
    let name_lower = kernel_name.to_ascii_lowercase();
    if !name_lower.contains("mul_mat_vec_q") || !name_lower.contains("ggml_type19") {
        return Err(format!(
            "kernel '{kernel_name}' is not the qualified IQ1_S mul_mat_vec_q symbol"
        ));
    }
    let marker = "ggml_type19eli";
    let marker_start = name_lower
        .find(marker)
        .ok_or_else(|| format!("kernel '{kernel_name}' has no IQ1_S vector batch arity"))?
        + marker.len();
    let digits_end = name_lower[marker_start..]
        .find(|byte: char| !byte.is_ascii_digit())
        .map(|offset| marker_start + offset)
        .unwrap_or(name_lower.len());
    let ncols_y = name_lower[marker_start..digits_end]
        .parse::<u32>()
        .map_err(|_| format!("kernel '{kernel_name}' has an invalid vector batch arity"))?;
    if ncols_y != 1 {
        return Err(format!(
            "kernel '{kernel_name}' uses ncols_y={ncols_y}; only one-token IQ1_S vector launches are supported"
        ));
    }

    super::iq1s_tmatmul::GgmlType19VecSignature {
        kernel: kernel_name.to_string(),
        ncols_x: shape.ncols_x as u64,
        nrows_x: shape.nrows_x as u64,
        nrows_y: shape.nrows_y as u64,
        nrows_dst: shape.nrows_dst as u64,
    }
    .validate()
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[derive(serde::Serialize)]
struct NvidiaIq1sCompletedEvidence<'a> {
    event: &'static str,
    status: &'static str,
    kernel: &'a str,
    logical_batch: u64,
    descriptor_count: u64,
    unique_submission_count: u64,
    lane_mask: u64,
    per_lane_completion_counts: &'a [u64],
    total_accelerator_cycles: u64,
    total_matrix_bytes_read: u64,
    total_input_bytes_read: u64,
    total_output_bytes_written: u64,
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_iq1s_completed_evidence<'a>(
    kernel: &'a str,
    logical_batch: u64,
    execution: &'a super::iq1s_tmatmul::ExecutionResult,
) -> NvidiaIq1sCompletedEvidence<'a> {
    let scheduler = &execution.scheduler;
    NvidiaIq1sCompletedEvidence {
        event: "ternip_v3_iq1s_execution",
        status: "completed",
        kernel,
        logical_batch,
        descriptor_count: scheduler.descriptor_count(),
        unique_submission_count: scheduler.unique_submission_count(),
        lane_mask: scheduler.lane_mask(),
        per_lane_completion_counts: scheduler.per_lane_completion_counts(),
        total_accelerator_cycles: scheduler.total_accelerator_cycles(),
        total_matrix_bytes_read: scheduler.total_matrix_bytes_read(),
        total_input_bytes_read: scheduler.total_input_bytes_read(),
        total_output_bytes_written: scheduler.total_output_bytes_written(),
    }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_iq1s_completed_evidence_json(evidence: NvidiaIq1sCompletedEvidence<'_>) -> String {
    serde_json::to_string(&evidence)
        .expect("primitive IQ1_S completion evidence must serialize as JSON")
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[derive(serde::Serialize)]
struct GpuAttentionCompletedEvidence<'a> {
    event: &'static str,
    status: &'static str,
    kernel: &'a str,
    launch_count: u64,
    device: &'static str,
    cpu_fallback: bool,
    emulator_fallback: bool,
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_gpu_attention_completed_evidence<'a>(
    kernel: &'a str,
    config: &super::bitnet_disagg::BitnetRouteConfig,
) -> Option<GpuAttentionCompletedEvidence<'a>> {
    let decision = super::bitnet_disagg::classify_kernel_name(kernel, config);
    if decision.route != super::bitnet_disagg::BitnetRoute::GpuNative {
        return None;
    }
    let matched = decision.matched.as_deref()?.to_ascii_lowercase();
    if !["attention", "attn", "flash"]
        .iter()
        .any(|marker| matched.contains(marker))
    {
        return None;
    }
    Some(GpuAttentionCompletedEvidence {
        event: "hetgpu_gpu_attention_execution",
        status: "completed",
        kernel,
        launch_count: 1,
        device: "cuda",
        cpu_fallback: false,
        emulator_fallback: false,
    })
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_gpu_attention_completed_evidence_json(
    evidence: GpuAttentionCompletedEvidence<'_>,
) -> String {
    serde_json::to_string(&evidence)
        .expect("primitive GPU attention evidence must serialize as JSON")
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_emit_gpu_attention_completed(kernel: &str) {
    let config = super::bitnet_disagg::config_from_env();
    let Some(evidence) = nvidia_gpu_attention_completed_evidence(kernel, &config) else {
        return;
    };
    eprintln!("{}", nvidia_gpu_attention_completed_evidence_json(evidence));
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_iq1s_post_execution_route(
    kernel_name: &str,
    logical_batch: u64,
    strict: bool,
    execution: Result<super::iq1s_tmatmul::ExecutionResult, String>,
    mut emit_completed: impl FnMut(&str),
) -> Option<Result<(), String>> {
    match execution {
        Ok(execution) => {
            let evidence = nvidia_iq1s_completed_evidence(kernel_name, logical_batch, &execution);
            let record = nvidia_iq1s_completed_evidence_json(evidence);
            emit_completed(&record);
            Some(Ok(()))
        }
        Err(error) => nvidia_iq1s_route_failure(kernel_name, strict, error),
    }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_iq1s_route_failure(
    kernel_name: &str,
    strict: bool,
    error: impl Into<String>,
) -> Option<Result<(), String>> {
    let message = format!(
        "kernel '{kernel_name}' IQ1_S CXL V3 execution failed: {}",
        error.into()
    );
    if strict {
        Some(Err(message))
    } else {
        eprintln!("[CXL TMatmul][NVIDIA] {message}; continuing native");
        None
    }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn nvidia_try_launch_iq1s_cxl_tmatmul(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    strict: bool,
) -> Option<Result<(), String>> {
    let shape = match nvidia_cxl_read_mmq_shape(kernel_params, kernel_name) {
        Ok(shape) => shape,
        Err(error) => return nvidia_iq1s_route_failure(kernel_name, strict, error),
    };
    let signature = match nvidia_iq1s_signature(kernel_name, shape) {
        Ok(signature) => signature,
        Err(error) => return nvidia_iq1s_route_failure(kernel_name, strict, error),
    };
    let matrix_ptr = match nvidia_cxl_read_pointer_param(kernel_params, 0, kernel_name) {
        Ok(value) => value,
        Err(error) => return nvidia_iq1s_route_failure(kernel_name, strict, error),
    };
    let activation_ptr = match nvidia_cxl_read_pointer_param(kernel_params, 1, kernel_name) {
        Ok(value) => value,
        Err(error) => return nvidia_iq1s_route_failure(kernel_name, strict, error),
    };
    let output_ptr = match nvidia_cxl_read_pointer_param(kernel_params, 2, kernel_name) {
        Ok(value) => value,
        Err(error) => return nvidia_iq1s_route_failure(kernel_name, strict, error),
    };

    let launch = super::iq1s_tmatmul::LogicalLaunch {
        matrix_ptr,
        activation_ptr,
        output_ptr,
        allocation_generation: nvidia_env_u64_any(&[
            "HETGPU_TMATMUL_ALLOCATION_GENERATION",
            "HETGPU_CXL_TMATMUL_ALLOCATION_GENERATION",
        ])
        .unwrap_or(0),
        // capture_launch fills this from the copied matrix bytes.  This keeps
        // the component cache safe when a CUDA allocation is reused.
        content_hash: [0; 32],
        signature,
    };
    let control_path = std::env::var("HETGPU_CXL_TMATMUL_DEVICE")
        .or_else(|_| std::env::var("HETGPU_TMATMUL_DEVICE"))
        .or_else(|_| std::env::var("CXL_TMATMUL_DEVICE"))
        .unwrap_or_else(|_| "/dev/cxl_tmatmul3b001".to_string());
    let dax_path = std::env::var("HETGPU_CXL_TMATMUL_DAX")
        .or_else(|_| std::env::var("HETGPU_TMATMUL_DAX"))
        .or_else(|_| std::env::var("CXL_DAX_PATH"))
        .unwrap_or_else(|_| "/dev/dax6.0".to_string());
    let base_dpa = nvidia_env_u64_any(&[
        "HETGPU_CXL_TMATMUL_V3_BASE_DPA",
        "HETGPU_TMATMUL_V3_BASE_DPA",
        "HETGPU_TMATMUL_DATA_BASE",
    ])
    .unwrap_or(0x0100_0000);

    eprintln!(
        "[CXL TMatmul][NVIDIA] launching IQ1_S '{}' via TernIP V3 matrix={:#x} activation={:#x} output={:#x} control={} dax={} base_dpa={:#x}",
        kernel_name,
        matrix_ptr,
        activation_ptr,
        output_ptr,
        control_path,
        dax_path,
        base_dpa,
    );

    let captured = match super::iq1s_tmatmul::capture_launch(launch) {
        Ok(captured) => captured,
        Err(error) => return nvidia_iq1s_route_failure(kernel_name, strict, error),
    };
    let logical_batch = captured.launch.signature.ne11;
    let execution = super::iq1s_tmatmul::execute_captured(
        &captured,
        std::path::Path::new(&control_path),
        std::path::Path::new(&dax_path),
        base_dpa,
    );
    nvidia_iq1s_post_execution_route(kernel_name, logical_batch, strict, execution, |record| {
        eprintln!("{record}")
    })
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn nvidia_try_launch_iq1s_vec_cxl_tmatmul(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    strict: bool,
) -> Option<Result<(), String>> {
    let shape = match nvidia_cxl_read_mmvq_shape(kernel_params, kernel_name) {
        Ok(shape) => shape,
        Err(error) => return nvidia_iq1s_route_failure(kernel_name, strict, error),
    };
    let signature = match nvidia_iq1s_vec_signature(kernel_name, shape) {
        Ok(signature) => signature,
        Err(error) => return nvidia_iq1s_route_failure(kernel_name, strict, error),
    };
    let matrix_ptr = match nvidia_cxl_read_pointer_param(kernel_params, 0, kernel_name) {
        Ok(value) => value,
        Err(error) => return nvidia_iq1s_route_failure(kernel_name, strict, error),
    };
    let activation_ptr = match nvidia_cxl_read_pointer_param(kernel_params, 1, kernel_name) {
        Ok(value) => value,
        Err(error) => return nvidia_iq1s_route_failure(kernel_name, strict, error),
    };
    let output_ptr = match nvidia_cxl_read_pointer_param(kernel_params, 2, kernel_name) {
        Ok(value) => value,
        Err(error) => return nvidia_iq1s_route_failure(kernel_name, strict, error),
    };
    let control_path = std::env::var("HETGPU_CXL_TMATMUL_DEVICE")
        .or_else(|_| std::env::var("HETGPU_TMATMUL_DEVICE"))
        .or_else(|_| std::env::var("CXL_TMATMUL_DEVICE"))
        .unwrap_or_else(|_| "/dev/cxl_tmatmul3b001".to_string());
    let dax_path = std::env::var("HETGPU_CXL_TMATMUL_DAX")
        .or_else(|_| std::env::var("HETGPU_TMATMUL_DAX"))
        .or_else(|_| std::env::var("CXL_DAX_PATH"))
        .unwrap_or_else(|_| "/dev/dax6.0".to_string());
    let base_dpa = nvidia_env_u64_any(&[
        "HETGPU_CXL_TMATMUL_V3_BASE_DPA",
        "HETGPU_TMATMUL_V3_BASE_DPA",
        "HETGPU_TMATMUL_DATA_BASE",
    ])
    .unwrap_or(0x0100_0000);

    eprintln!(
        "[CXL TMatmul][NVIDIA] launching IQ1_S vector '{}' via TernIP V3 matrix={:#x} activation={:#x} output={:#x} control={} dax={} base_dpa={:#x}",
        kernel_name,
        matrix_ptr,
        activation_ptr,
        output_ptr,
        control_path,
        dax_path,
        base_dpa,
    );

    let captured = match super::iq1s_tmatmul::capture_vec_launch(
        matrix_ptr,
        activation_ptr,
        output_ptr,
        nvidia_env_u64_any(&[
            "HETGPU_TMATMUL_ALLOCATION_GENERATION",
            "HETGPU_CXL_TMATMUL_ALLOCATION_GENERATION",
        ])
        .unwrap_or(0),
        [0; 32],
        signature,
    ) {
        Ok(captured) => captured,
        Err(error) => return nvidia_iq1s_route_failure(kernel_name, strict, error),
    };
    let logical_batch = captured.launch.signature.ne11;
    let execution = super::iq1s_tmatmul::execute_captured(
        &captured,
        std::path::Path::new(&control_path),
        std::path::Path::new(&dax_path),
        base_dpa,
    );
    nvidia_iq1s_post_execution_route(kernel_name, logical_batch, strict, execution, |record| {
        eprintln!("{record}")
    })
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[derive(Clone, Copy)]
enum NvidiaIq1sXrtKernel {
    Mmq,
    Mmvq,
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_iq1s_xrt_kernel(kernel_name: &str) -> Option<NvidiaIq1sXrtKernel> {
    let name = kernel_name.to_ascii_lowercase();
    if !name.contains("ggml_type19") || name.contains("stream_k_fixup") {
        return None;
    }
    if name.contains("mul_mat_vec_q") {
        Some(NvidiaIq1sXrtKernel::Mmvq)
    } else if name.contains("mul_mat_q") {
        Some(NvidiaIq1sXrtKernel::Mmq)
    } else {
        None
    }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_xrt_route_failure(
    kernel_name: &str,
    strict: bool,
    error: impl Into<String>,
) -> Option<Result<(), String>> {
    let message = format!(
        "kernel '{kernel_name}' IQ1_S AU250 XRT execution failed: {}",
        error.into()
    );
    if strict {
        Some(Err(message))
    } else {
        eprintln!("[XRT TMatmul][NVIDIA] {message}; continuing native");
        None
    }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_xrt_post_execution_route(
    kernel_name: &str,
    strict: bool,
    execution: Result<super::iq1s_xrt::XrtIq1sResult, String>,
    mut on_success: impl FnMut(&super::iq1s_xrt::XrtIq1sResult),
) -> Option<Result<(), String>> {
    match execution {
        Ok(execution) => {
            on_success(&execution);
            Some(Ok(()))
        }
        Err(error) => nvidia_xrt_route_failure(kernel_name, strict, error),
    }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_xrt_compare_max_launches() -> Result<u64, String> {
    match std::env::var("HETGPU_XRT_COMPARE_MAX_LAUNCHES") {
        Ok(text) => text.parse::<u64>().map_err(|error| {
            format!(
                "HETGPU_XRT_COMPARE_MAX_LAUNCHES={text:?} is not a nonnegative integer: {error}"
            )
        }),
        Err(std::env::VarError::NotPresent) => Ok(0),
        Err(error) => Err(format!("read HETGPU_XRT_COMPARE_MAX_LAUNCHES: {error}")),
    }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_xrt_output_hash(outputs: &[f32]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    outputs.len().hash(&mut hasher);
    for value in outputs {
        value.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct NvidiaIq1sCaptureIdentity {
    tensor_name: String,
    expert: u64,
    allocation_generation: u64,
    model_hash: [u8; 32],
    content_hash: [u8; 32],
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_resolve_registered_iq1s_launch(
    registry: &super::iq1s_weight_registry::Iq1sWeightRegistry,
    matrix_ptr: usize,
    matrix_bytes: u64,
    ncols: u64,
    nrows: u64,
    row_stride_bytes: u64,
) -> Result<NvidiaIq1sCaptureIdentity, String> {
    let resolved = registry.resolve_launch(
        matrix_ptr,
        matrix_bytes,
        ncols,
        nrows,
        row_stride_bytes,
    )?;
    Ok(NvidiaIq1sCaptureIdentity {
        tensor_name: resolved.identity.name,
        expert: resolved.expert,
        allocation_generation: resolved.allocation_generation,
        model_hash: resolved.identity.model_sha256,
        content_hash: resolved.content_sha256,
    })
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_qwen_iq1s_capture_identity(
    matrix_ptr: usize,
    matrix_bytes: usize,
    ncols: u64,
    nrows: u64,
    row_stride_bytes: u64,
) -> Result<Option<NvidiaIq1sCaptureIdentity>, String> {
    if !nvidia_env_truthy("HETGPU_QWEN_IQ1S_STRICT") {
        return Ok(None);
    }
    nvidia_resolve_registered_iq1s_launch(
        super::iq1s_weight_registry::global_registry(),
        matrix_ptr,
        u64::try_from(matrix_bytes).map_err(|_| "IQ1_S matrix bytes do not fit u64")?,
        ncols,
        nrows,
        row_stride_bytes,
    )
    .map(Some)
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_xrt_emit_success_records(
    kernel_name: &str,
    result: &super::iq1s_xrt::XrtIq1sResult,
    comparison: Option<(u64, &super::iq1s_xrt::SampledOutputComparison)>,
) {
    let completed = serde_json::json!({
        "event": "au250_xrt_iq1s_completed",
        "kernel": kernel_name,
        "evidence": &result.evidence,
    });
    eprintln!("{completed}");

    if let Some((ordinal, comparison)) = comparison {
        let comparison = serde_json::json!({
            "event": "captured_layer_comparison",
            "backend": "xrt",
            "kernel": kernel_name,
            "launch_ordinal": ordinal,
            "output_hash": format!("{:016x}", nvidia_xrt_output_hash(&result.outputs)),
            "reference_checked_components": result.evidence.reference_checked_components,
            "comparison_status": comparison.status,
            "comparison": comparison,
        });
        eprintln!("{comparison}");
    }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_xrt_reserve_comparison(compare_max_launches: u64) -> Option<u64> {
    static SUCCESSFUL_XRT_LAUNCHES: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(0);
    let ordinal = SUCCESSFUL_XRT_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    (ordinal < compare_max_launches).then_some(ordinal)
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn nvidia_capture_iq1s_xrt_mmq(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Result<Vec<super::iq1s_tmatmul::CapturedLaunch>, String> {
    let shape = nvidia_cxl_read_mmq_shape(kernel_params, kernel_name)?;
    let signature = nvidia_iq1s_signature(kernel_name, shape)?;
    let matrix_ptr = nvidia_cxl_read_pointer_param(kernel_params, 0, kernel_name)?;
    let activation_ptr = nvidia_cxl_read_pointer_param(kernel_params, 1, kernel_name)?;
    let output_ptr = nvidia_cxl_read_pointer_param(kernel_params, 2, kernel_name)?;
    let matrix_bytes = signature.matrix_storage_bytes()?;
    let row_stride_bytes = signature
        .stride01
        .checked_mul(super::iq1s_tmatmul::IQ1S_BLOCK_BYTES as u64)
        .ok_or("IQ1_S MMQ row stride overflow")?;
    let identity = nvidia_qwen_iq1s_capture_identity(
        matrix_ptr,
        matrix_bytes,
        signature.ne00,
        signature.ne01,
        row_stride_bytes,
    )?;
    super::iq1s_tmatmul::capture_launch(super::iq1s_tmatmul::LogicalLaunch {
        matrix_ptr,
        activation_ptr,
        output_ptr,
        allocation_generation: identity
            .as_ref()
            .map(|identity| identity.allocation_generation)
            .unwrap_or_else(|| {
                nvidia_env_u64_any(&[
                    "HETGPU_TMATMUL_ALLOCATION_GENERATION",
                    "HETGPU_XRT_TMATMUL_ALLOCATION_GENERATION",
                ])
                .unwrap_or(0)
            }),
        content_hash: identity
            .as_ref()
            .map(|identity| identity.content_hash)
            .unwrap_or([0; 32]),
        signature,
    })
    .map(|captured| vec![captured])
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn nvidia_capture_iq1s_xrt_mmvq(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Result<Vec<super::iq1s_tmatmul::CapturedLaunch>, String> {
    let shape = nvidia_cxl_read_mmvq_shape(kernel_params, kernel_name)?;
    let signature = nvidia_iq1s_vec_signature(kernel_name, shape)?;
    let matrix_ptr = nvidia_cxl_read_pointer_param(kernel_params, 0, kernel_name)?;
    let activation_ptr = nvidia_cxl_read_pointer_param(kernel_params, 1, kernel_name)?;
    let output_ptr = nvidia_cxl_read_pointer_param(kernel_params, 2, kernel_name)?;
    let matrix_bytes = signature.matrix_storage_bytes()?;
    let row_stride_bytes = signature
        .ncols_x
        .checked_div(super::iq1s_tmatmul::IQ1S_BLOCK_VALUES as u64)
        .and_then(|blocks| {
            blocks.checked_mul(super::iq1s_tmatmul::IQ1S_BLOCK_BYTES as u64)
        })
        .ok_or("IQ1_S MMVQ row stride overflow")?;
    let identity = nvidia_qwen_iq1s_capture_identity(
        matrix_ptr,
        matrix_bytes,
        signature.ncols_x,
        signature.nrows_x,
        row_stride_bytes,
    )?;
    super::iq1s_tmatmul::capture_vec_launch(
        matrix_ptr,
        activation_ptr,
        output_ptr,
        identity
            .as_ref()
            .map(|identity| identity.allocation_generation)
            .unwrap_or_else(|| {
                nvidia_env_u64_any(&[
                    "HETGPU_TMATMUL_ALLOCATION_GENERATION",
                    "HETGPU_XRT_TMATMUL_ALLOCATION_GENERATION",
                ])
                .unwrap_or(0)
            }),
        identity
            .as_ref()
            .map(|identity| identity.content_hash)
            .unwrap_or([0; 32]),
        signature,
    )
    .map(|captured| vec![captured])
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn nvidia_capture_modern_iq1s_xrt_mmvq(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid: (u32, u32, u32),
) -> Result<Vec<super::iq1s_tmatmul::CapturedLaunch>, String> {
    if nvidia_modern_iq1s_mmvq_has_fusion(kernel_name) {
        return Err(format!(
            "kernel '{kernel_name}' has fused MMVQ semantics; set HETGPU_QWEN_IQ1S_DISABLE_CUDA_FUSION=1 in the patched llama.cpp runtime"
        ));
    }
    let matrix_base = nvidia_cxl_read_pointer_param(kernel_params, 0, kernel_name)?;
    let activation_base = nvidia_cxl_read_pointer_param(kernel_params, 1, kernel_name)?;
    let ids_ptr = nvidia_cxl_read_pointer_param(kernel_params, 2, kernel_name)?;
    let output_base = nvidia_cxl_read_pointer_param(kernel_params, 4, kernel_name)?;
    let layout = NvidiaModernMmvqLayout {
        ncols_x: nvidia_cxl_read_u32_param(kernel_params, 5, kernel_name)?,
        nchannels_y: nvidia_cxl_read_uint3_param(kernel_params, 6, kernel_name)?,
        stride_row_x: nvidia_cxl_read_u32_param(kernel_params, 7, kernel_name)?,
        stride_col_y: nvidia_cxl_read_u32_param(kernel_params, 8, kernel_name)?,
        stride_col_dst: nvidia_cxl_read_u32_param(kernel_params, 9, kernel_name)?,
        stride_channel_x: nvidia_cxl_read_u32_param(kernel_params, 11, kernel_name)?,
        stride_channel_y: nvidia_cxl_read_u32_param(kernel_params, 12, kernel_name)?,
        stride_channel_dst: nvidia_cxl_read_u32_param(kernel_params, 13, kernel_name)?,
        sample_ratio: nvidia_cxl_read_uint3_param(kernel_params, 14, kernel_name)?,
        stride_sample_x: nvidia_cxl_read_u32_param(kernel_params, 15, kernel_name)?,
        stride_sample_y: nvidia_cxl_read_u32_param(kernel_params, 16, kernel_name)?,
        stride_sample_dst: nvidia_cxl_read_u32_param(kernel_params, 17, kernel_name)?,
    };
    let ids_bytes = super::cxl_tmatmul::copy_cuda_to_host(
        ids_ptr,
        (grid.1 as usize)
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or("modern IQ1_S MMVQ ids size overflow")?,
    )
    .map_err(|error| error.to_string())?;
    let expert_ids = ids_bytes
        .chunks_exact(std::mem::size_of::<i32>())
        .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("four-byte ids chunk")))
        .collect::<Vec<_>>();
    let plans = nvidia_plan_modern_iq1s_mmvq(
        layout,
        matrix_base,
        activation_base,
        output_base,
        &expert_ids,
        grid,
    )?;
    let legacy_generation = nvidia_env_u64_any(&[
        "HETGPU_TMATMUL_ALLOCATION_GENERATION",
        "HETGPU_XRT_TMATMUL_ALLOCATION_GENERATION",
    ])
    .unwrap_or(0);
    plans
        .into_iter()
        .map(|mut plan| {
            plan.signature.kernel = kernel_name.to_string();
            let matrix_bytes = plan.signature.matrix_storage_bytes()?;
            let row_stride_bytes = plan
                .signature
                .ncols_x
                .checked_div(super::iq1s_tmatmul::IQ1S_BLOCK_VALUES as u64)
                .and_then(|blocks| {
                    blocks.checked_mul(super::iq1s_tmatmul::IQ1S_BLOCK_BYTES as u64)
                })
                .ok_or("modern IQ1_S MMVQ row stride overflow")?;
            let identity = nvidia_qwen_iq1s_capture_identity(
                plan.matrix_ptr,
                matrix_bytes,
                plan.signature.ncols_x,
                plan.signature.nrows_x,
                row_stride_bytes,
            )?;
            super::iq1s_tmatmul::capture_vec_launch(
                plan.matrix_ptr,
                plan.activation_ptr,
                plan.output_ptr,
                identity
                    .as_ref()
                    .map(|identity| identity.allocation_generation)
                    .unwrap_or(legacy_generation),
                identity
                    .as_ref()
                    .map(|identity| identity.content_hash)
                    .unwrap_or([0; 32]),
                plan.signature,
            )
        })
        .collect()
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn nvidia_plan_modern_iq1s_xrt_moe_mmvq_launch(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid: (u32, u32, u32),
) -> Result<Vec<(NvidiaModernMmvqPlan, u64, [u8; 32])>, String> {
    if !nvidia_is_modern_iq1s_moe_mmvq(kernel_name) {
        return Err(format!(
            "kernel '{kernel_name}' is not the pinned two-row IQ1_S MoE MMVQ ABI"
        ));
    }
    let matrix_base = nvidia_cxl_read_pointer_param(kernel_params, 0, kernel_name)?;
    let activation_base = nvidia_cxl_read_pointer_param(kernel_params, 1, kernel_name)?;
    let ids_ptr = nvidia_cxl_read_pointer_param(kernel_params, 2, kernel_name)?;
    let output_base = nvidia_cxl_read_pointer_param(kernel_params, 3, kernel_name)?;
    let layout = NvidiaModernMoeMmvqLayout {
        ncols_x: nvidia_cxl_read_u32_param(kernel_params, 4, kernel_name)?,
        nchannels_y: nvidia_cxl_read_uint3_param(kernel_params, 5, kernel_name)?,
        nrows_x: nvidia_cxl_read_u32_param(kernel_params, 6, kernel_name)?,
        stride_row_x: nvidia_cxl_read_u32_param(kernel_params, 7, kernel_name)?,
        stride_col_y: nvidia_cxl_read_u32_param(kernel_params, 8, kernel_name)?,
        stride_col_dst: nvidia_cxl_read_u32_param(kernel_params, 9, kernel_name)?,
        stride_channel_x: nvidia_cxl_read_u32_param(kernel_params, 10, kernel_name)?,
        stride_channel_y: nvidia_cxl_read_u32_param(kernel_params, 11, kernel_name)?,
        stride_channel_dst: nvidia_cxl_read_u32_param(kernel_params, 12, kernel_name)?,
        ncols_dst: nvidia_cxl_read_u32_param(kernel_params, 13, kernel_name)?,
        ids_stride: nvidia_cxl_read_u32_param(kernel_params, 14, kernel_name)?,
    };
    let ids_count = layout
        .ncols_dst
        .checked_sub(1)
        .and_then(|token| token.checked_mul(layout.ids_stride))
        .and_then(|base| base.checked_add(grid.1))
        .ok_or("modern IQ1_S MoE MMVQ ids extent overflow")?;
    let ids_bytes = super::cxl_tmatmul::copy_cuda_to_host(
        ids_ptr,
        usize::try_from(ids_count)
            .map_err(|_| "modern IQ1_S MoE MMVQ ids count does not fit usize")?
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or("modern IQ1_S MoE MMVQ ids size overflow")?,
    )
    .map_err(|error| error.to_string())?;
    let expert_ids = ids_bytes
        .chunks_exact(std::mem::size_of::<i32>())
        .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("four-byte ids chunk")))
        .collect::<Vec<_>>();
    let plans = nvidia_plan_modern_iq1s_moe_mmvq(
        layout,
        matrix_base,
        activation_base,
        output_base,
        &expert_ids,
        grid,
    )?;
    let legacy_generation = nvidia_env_u64_any(&[
        "HETGPU_TMATMUL_ALLOCATION_GENERATION",
        "HETGPU_XRT_TMATMUL_ALLOCATION_GENERATION",
    ])
    .unwrap_or(0);
    plans
        .into_iter()
        .map(|mut plan| {
            plan.signature.kernel = kernel_name.to_string();
            let matrix_bytes = plan.signature.matrix_storage_bytes()?;
            let row_stride_bytes = plan
                .signature
                .ncols_x
                .checked_div(super::iq1s_tmatmul::IQ1S_BLOCK_VALUES as u64)
                .and_then(|blocks| {
                    blocks.checked_mul(super::iq1s_tmatmul::IQ1S_BLOCK_BYTES as u64)
                })
                .ok_or("modern IQ1_S MoE MMVQ row stride overflow")?;
            let identity = nvidia_qwen_iq1s_capture_identity(
                plan.matrix_ptr,
                matrix_bytes,
                plan.signature.ncols_x,
                plan.signature.nrows_x,
                row_stride_bytes,
            )?;
            Ok((
                plan,
                identity
                    .as_ref()
                    .map(|identity| identity.allocation_generation)
                    .unwrap_or(legacy_generation),
                identity
                    .as_ref()
                    .map(|identity| identity.content_hash)
                    .unwrap_or([0; 32]),
            ))
        })
        .collect()
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn nvidia_capture_modern_iq1s_xrt_moe_mmvq(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid: (u32, u32, u32),
) -> Result<Vec<super::iq1s_tmatmul::CapturedLaunch>, String> {
    nvidia_plan_modern_iq1s_xrt_moe_mmvq_launch(kernel_name, kernel_params, grid)?
        .into_iter()
        .map(|(plan, allocation_generation, content_hash)| {
            super::iq1s_tmatmul::capture_vec_launch(
                plan.matrix_ptr,
                plan.activation_ptr,
                plan.output_ptr,
                allocation_generation,
                content_hash,
                plan.signature,
            )
        })
        .collect()
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn nvidia_capture_modern_iq1s_xrt_moe_activations(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid: (u32, u32, u32),
) -> Result<Vec<super::iq1s_tmatmul::CapturedActivationLaunch>, String> {
    nvidia_plan_modern_iq1s_xrt_moe_mmvq_launch(kernel_name, kernel_params, grid)?
        .into_iter()
        .map(|(plan, allocation_generation, content_hash)| {
            super::iq1s_tmatmul::capture_vec_activation(
                plan.matrix_ptr,
                plan.activation_ptr,
                plan.output_ptr,
                allocation_generation,
                content_hash,
                plan.signature,
            )
        })
        .collect()
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
trait NvidiaIq1sLayerCaptureSink {
    fn has_open_transaction(&self, stream: usize) -> Result<bool, String>;
    fn capture_projection(
        &self,
        stream: usize,
        captured: Vec<super::iq1s_tmatmul::CapturedActivationLaunch>,
    ) -> Result<(), String>;
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
struct NvidiaIq1sGlobalLayerCaptureSink;

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl NvidiaIq1sLayerCaptureSink for NvidiaIq1sGlobalLayerCaptureSink {
    fn has_open_transaction(&self, stream: usize) -> Result<bool, String> {
        super::iq1s_layer::has_open_transaction(stream)
    }

    fn capture_projection(
        &self,
        stream: usize,
        captured: Vec<super::iq1s_tmatmul::CapturedActivationLaunch>,
    ) -> Result<(), String> {
        let role = super::iq1s_layer::resolve_captured_role(&captured)?;
        super::iq1s_layer::capture_projection(stream, role, captured)
    }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[derive(Debug)]
enum NvidiaIq1sDispatch<T> {
    Buffered,
    Executed(Vec<(super::iq1s_tmatmul::CapturedLaunch, T)>),
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[derive(Debug)]
enum NvidiaIq1sDispatchError {
    Capture(String),
    Execute(String),
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl std::fmt::Display for NvidiaIq1sDispatchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Capture(error) => write!(formatter, "capture: {error}"),
            Self::Execute(error) => write!(formatter, "execute: {error}"),
        }
    }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn nvidia_dispatch_iq1s_capture<E, A, F>(
    modern_moe: bool,
    strict_persistent: bool,
    stream: usize,
    layer_sink: &impl NvidiaIq1sLayerCaptureSink,
    capture_activations: A,
    capture_full: F,
    executor: &mut E,
) -> Result<NvidiaIq1sDispatch<E::Output>, NvidiaIq1sDispatchError>
where
    E: super::iq1s_xrt::CapturedLaunchExecutor,
    A: FnOnce() -> Result<Vec<super::iq1s_tmatmul::CapturedActivationLaunch>, String>,
    F: FnOnce() -> Result<Vec<super::iq1s_tmatmul::CapturedLaunch>, String>,
{
    let transaction_open = modern_moe
        && layer_sink
            .has_open_transaction(stream)
            .map_err(NvidiaIq1sDispatchError::Capture)?;
    if strict_persistent && (!modern_moe || !transaction_open) {
        eprintln!(
            "[hetgpu-iq1s-layer] eligible_direct_route=1 modern_moe={} transaction_open={}",
            u8::from(modern_moe),
            u8::from(transaction_open)
        );
        return Err(NvidiaIq1sDispatchError::Capture(
            "strict persistent IQ1_S eligible_direct_route=1 requires an open layer lifecycle"
                .to_string(),
        ));
    }
    if transaction_open {
        let captured = capture_activations().map_err(NvidiaIq1sDispatchError::Capture)?;
        layer_sink
            .capture_projection(stream, captured)
            .map_err(NvidiaIq1sDispatchError::Capture)?;
        return Ok(NvidiaIq1sDispatch::Buffered);
    }

    let captured = capture_full().map_err(NvidiaIq1sDispatchError::Capture)?;
    let mut executed = Vec::with_capacity(captured.len());
    for component in captured {
        let output = executor
            .execute(&component)
            .map_err(NvidiaIq1sDispatchError::Execute)?;
        executed.push((component, output));
    }
    Ok(NvidiaIq1sDispatch::Executed(executed))
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) unsafe fn nvidia_try_launch_named_xrt_tmatmul(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid: (u32, u32, u32),
    stream: cuda_types::cuda::CUstream,
) -> Option<Result<(), String>> {
    let kernel_kind = nvidia_iq1s_xrt_kernel(kernel_name)?;
    if !super::bitnet_disagg::enabled_from_env()
        || !nvidia_env_truthy("HETGPU_TMATMUL_HARDWARE_MATMUL")
    {
        return None;
    }

    let route_config = super::bitnet_disagg::config_from_env();
    let decision = super::bitnet_disagg::classify_kernel_name(kernel_name, &route_config);
    match decision.route {
        super::bitnet_disagg::BitnetRoute::GpuNative
        | super::bitnet_disagg::BitnetRoute::Fallback => return None,
        super::bitnet_disagg::BitnetRoute::Reject => {
            return Some(Err(format!(
                "strict route rejected '{}' via {}",
                kernel_name,
                decision.source.as_str()
            )));
        }
        super::bitnet_disagg::BitnetRoute::CxlTmatmul => {}
    }
    let modern_moe = matches!(kernel_kind, NvidiaIq1sXrtKernel::Mmvq)
        && nvidia_is_modern_iq1s_moe_mmvq(kernel_name);
    let modern_multi = matches!(kernel_kind, NvidiaIq1sXrtKernel::Mmvq)
        && (modern_moe || nvidia_is_modern_iq1s_mmvq(kernel_name));
    if !modern_multi {
        if let Err(error) = super::bitnet_disagg::append_route_log_from_env(
            &decision,
            super::bitnet_disagg::RouteHardware::xrt(true),
        ) {
            eprintln!("[BitNet Disagg][NVIDIA XRT] route log failed for '{kernel_name}': {error}");
            if decision.strict {
                return Some(Err(format!("route log failed: {error}")));
            }
        }
    }
    let strict_persistent = nvidia_env_truthy("HETGPU_QWEN_IQ1S_PERSISTENT");
    let mut executor = super::iq1s_xrt::DirectCapturedLaunchExecutor;
    let dispatched = match nvidia_dispatch_iq1s_capture(
        modern_moe,
        strict_persistent,
        stream.0 as usize,
        &NvidiaIq1sGlobalLayerCaptureSink,
        || {
            if !modern_moe {
                return Err("activation-only capture requires a modern IQ1_S MoE launch".into());
            }
            nvidia_capture_modern_iq1s_xrt_moe_activations(kernel_name, kernel_params, grid)
        },
        || match kernel_kind {
            NvidiaIq1sXrtKernel::Mmq => nvidia_capture_iq1s_xrt_mmq(kernel_name, kernel_params),
            NvidiaIq1sXrtKernel::Mmvq if modern_moe => {
                nvidia_capture_modern_iq1s_xrt_moe_mmvq(kernel_name, kernel_params, grid)
            }
            NvidiaIq1sXrtKernel::Mmvq if modern_multi => {
                nvidia_capture_modern_iq1s_xrt_mmvq(kernel_name, kernel_params, grid)
            }
            NvidiaIq1sXrtKernel::Mmvq => nvidia_capture_iq1s_xrt_mmvq(kernel_name, kernel_params),
        },
        &mut executor,
    ) {
        Ok(dispatched) => dispatched,
        Err(NvidiaIq1sDispatchError::Capture(error)) => {
            if strict_persistent {
                super::iq1s_persistent_runtime::poison_global_persistent_runtime(&error);
            }
            return Some(Err(format!(
                "IQ1_S layer transaction capture failed without fallback: {error}"
            )));
        }
        Err(NvidiaIq1sDispatchError::Execute(error)) => {
            return nvidia_xrt_route_failure(kernel_name, decision.strict, error)
        }
    };
    let NvidiaIq1sDispatch::Executed(executed) = dispatched else {
        return Some(Ok(()));
    };
    eprintln!(
        "[XRT TMatmul][NVIDIA] launching {} IQ1_S component(s) from '{}' on the AU250 direct path",
        executed.len(),
        kernel_name
    );
    let compare_max_launches = match nvidia_xrt_compare_max_launches() {
        Ok(value) => value,
        Err(error) => return nvidia_xrt_route_failure(kernel_name, decision.strict, error),
    };
    for _component in &executed {
        if modern_multi {
            if let Err(error) = super::bitnet_disagg::append_route_log_from_env(
                &decision,
                super::bitnet_disagg::RouteHardware::xrt(true),
            ) {
                eprintln!(
                    "[BitNet Disagg][NVIDIA XRT] route log failed for '{kernel_name}': {error}"
                );
                if decision.strict {
                    return Some(Err(format!("route log failed: {error}")));
                }
            }
        }
    }
    for (component, result) in executed {
        let comparison_ordinal = nvidia_xrt_reserve_comparison(compare_max_launches);
        let comparison = match comparison_ordinal {
            Some(ordinal) => match super::iq1s_xrt::compare_outputs_with_reference(
                &component,
                &result.outputs,
                1.0e-4,
                1.0e-3,
            ) {
                Ok(comparison) => Some((ordinal, comparison)),
                Err(error) => {
                    return nvidia_xrt_route_failure(
                        kernel_name,
                        decision.strict,
                        format!("sampled IQ1_S output comparison failed: {error}"),
                    )
                }
            },
            None => None,
        };
        nvidia_xrt_emit_success_records(
            kernel_name,
            &result,
            comparison
                .as_ref()
                .map(|(ordinal, record)| (*ordinal, record)),
        )
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) unsafe fn nvidia_try_launch_named_cxl_tmatmul(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<Result<(), String>> {
    if !super::bitnet_disagg::enabled_from_env()
        || !nvidia_env_truthy("HETGPU_TMATMUL_HARDWARE_MATMUL")
    {
        return None;
    }

    if !super::cxl_tmatmul::cxl_tmatmul_enabled() {
        return None;
    }

    let route_config = super::bitnet_disagg::config_from_env();
    let decision = super::bitnet_disagg::classify_kernel_name(kernel_name, &route_config);
    match decision.route {
        super::bitnet_disagg::BitnetRoute::GpuNative
        | super::bitnet_disagg::BitnetRoute::Fallback => return None,
        super::bitnet_disagg::BitnetRoute::Reject => {
            return Some(Err(format!(
                "strict route rejected '{}' via {}",
                kernel_name,
                decision.source.as_str()
            )));
        }
        super::bitnet_disagg::BitnetRoute::CxlTmatmul => {}
    }

    if let Err(err) = super::bitnet_disagg::append_route_log_from_env(
        &decision,
        super::bitnet_disagg::RouteHardware::cxl(true, true),
    ) {
        eprintln!(
            "[BitNet Disagg][NVIDIA CXL] route log failed for '{}': {}",
            kernel_name, err
        );
        if decision.strict {
            return Some(Err(format!("route log failed: {err}")));
        }
    }

    match super::cxl_tmatmul::matrix_stage_mode() {
        Ok(super::cxl_tmatmul::MatrixStageMode::CudaDax)
        | Ok(super::cxl_tmatmul::MatrixStageMode::CudaHost) => {}
        Ok(_) => {
            let msg = "NVIDIA CXL named matmul requires HETGPU_TMATMUL_MATRIX_STAGE=cuda_dax or cuda_host";
            if decision.strict {
                return Some(Err(msg.to_string()));
            }
            eprintln!("[CXL TMatmul][NVIDIA] {msg}; continuing native for '{kernel_name}'");
            return None;
        }
        Err(err) => {
            if decision.strict {
                return Some(Err(err.to_string()));
            }
            eprintln!("[CXL TMatmul][NVIDIA] {err}; continuing native for '{kernel_name}'");
            return None;
        }
    }
    match super::cxl_tmatmul::io_stage_mode() {
        Ok(super::cxl_tmatmul::IoStageMode::CudaDax)
        | Ok(super::cxl_tmatmul::IoStageMode::CudaHost) => {}
        Ok(_) => {
            let msg =
                "NVIDIA CXL named matmul requires HETGPU_TMATMUL_IO_STAGE=cuda_dax or cuda_host";
            if decision.strict {
                return Some(Err(msg.to_string()));
            }
            eprintln!("[CXL TMatmul][NVIDIA] {msg}; continuing native for '{kernel_name}'");
            return None;
        }
        Err(err) => {
            if decision.strict {
                return Some(Err(err.to_string()));
            }
            eprintln!("[CXL TMatmul][NVIDIA] {err}; continuing native for '{kernel_name}'");
            return None;
        }
    }

    let name_lower = kernel_name.to_ascii_lowercase();
    if name_lower.contains("mul_mat_q")
        && !name_lower.contains("mul_mat_vec_q")
        && name_lower.contains("ggml_type19")
    {
        if let Some(result) =
            nvidia_try_launch_iq1s_cxl_tmatmul(kernel_name, kernel_params, decision.strict)
        {
            return Some(result);
        }
    }
    if name_lower.contains("mul_mat_vec_q") && name_lower.contains("ggml_type19") {
        if let Some(result) =
            nvidia_try_launch_iq1s_vec_cxl_tmatmul(kernel_name, kernel_params, decision.strict)
        {
            return Some(result);
        }
    }

    let Some(layout) = nvidia_cxl_matmul_layout(&name_lower) else {
        let msg = format!("kernel '{kernel_name}' is not a supported NVIDIA CXL matmul layout");
        if decision.strict {
            return Some(Err(msg));
        }
        eprintln!("[CXL TMatmul][NVIDIA] {msg}; continuing native");
        return None;
    };

    if layout.name == "mul_mat_vec_q" {
        let shape = match nvidia_cxl_read_mmvq_shape(kernel_params, kernel_name) {
            Ok(shape) => shape,
            Err(err) => return Some(Err(err)),
        };
        let msg = nvidia_cxl_mmvq_contract_error(shape)
            .expect("mul_mat_vec_q is not a dense NVINT8 tmatmul ABI");
        if decision.strict {
            return Some(Err(msg));
        }
        eprintln!("[CXL TMatmul][NVIDIA] {msg}; continuing native for '{kernel_name}'");
        return None;
    }

    if layout.name == "mul_mat_q" {
        let shape = match nvidia_cxl_read_mmq_shape(kernel_params, kernel_name) {
            Ok(shape) => shape,
            Err(err) => return Some(Err(err)),
        };
        let msg = nvidia_cxl_mmq_contract_error(shape);
        if decision.strict {
            return Some(Err(msg));
        }
        eprintln!("[CXL TMatmul][NVIDIA] {msg}; continuing native for '{kernel_name}'");
        return None;
    }

    let matrix_ptr =
        match nvidia_cxl_read_pointer_param(kernel_params, layout.matrix_param, kernel_name) {
            Ok(value) => value,
            Err(err) => return Some(Err(err)),
        };
    let vector_ptr =
        match nvidia_cxl_read_pointer_param(kernel_params, layout.vector_param, kernel_name) {
            Ok(value) => value,
            Err(err) => return Some(Err(err)),
        };
    let output_ptr =
        match nvidia_cxl_read_pointer_param(kernel_params, layout.output_param, kernel_name) {
            Ok(value) => value,
            Err(err) => return Some(Err(err)),
        };

    let assembly = nvidia_cxl_generate_matmul_assembly(kernel_name, layout);
    if let Ok(path) = std::env::var("HETGPU_TMATMUL_ASM_PATH") {
        let _ = std::fs::write(path, assembly.as_bytes());
    }
    let matrix_offset = match super::cxl_tmatmul::matrix_dpa_offset() {
        Ok(value) => value,
        Err(err) => return Some(Err(err.to_string())),
    };
    let labels = std::collections::HashMap::from([
        (format!("PARAM_{}", layout.matrix_param), matrix_offset),
        (
            format!("PARAM_{}", layout.vector_param),
            super::cxl_tmatmul::TMATMUL_DPA_INPUT,
        ),
        (
            format!("PARAM_{}", layout.output_param),
            super::cxl_tmatmul::TMATMUL_DPA_OUTPUT,
        ),
    ]);
    let timeout_ms = nvidia_env_usize("HETGPU_CXL_TMATMUL_TIMEOUT_MS")
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(0);

    eprintln!(
        "[CXL TMatmul][NVIDIA] launching '{}' as {} matrix=PARAM_{}:{:#x} vector=PARAM_{}:{:#x} output=PARAM_{}:{:#x}",
        kernel_name,
        layout.name,
        layout.matrix_param,
        matrix_ptr,
        layout.vector_param,
        vector_ptr,
        layout.output_param,
        output_ptr,
    );

    match super::cxl_tmatmul::submit_hardware_matmul_from_ptrs(
        &assembly,
        &labels,
        matrix_ptr as *const u8,
        usize::MAX,
        vector_ptr as *const u8,
        usize::MAX,
        output_ptr as *mut u8,
        usize::MAX,
        timeout_ms,
    ) {
        Ok(status) => {
            eprintln!(
                "[CXL TMatmul][NVIDIA] Kernel '{}' executed via RUN_CSR_ONLY: {:?}",
                kernel_name, status
            );
            Some(Ok(()))
        }
        Err(err) => {
            let msg = format!("kernel '{kernel_name}' RUN_CSR_ONLY submit failed: {err}");
            if decision.strict {
                Some(Err(msg))
            } else {
                eprintln!("[CXL TMatmul][NVIDIA] {msg}; continuing native");
                None
            }
        }
    }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) unsafe fn nvidia_try_launch_named_tmatmul(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid: (u32, u32, u32),
    stream: cuda_types::cuda::CUstream,
) -> Option<Result<(), String>> {
    if !super::bitnet_disagg::enabled_from_env()
        || !nvidia_env_truthy("HETGPU_TMATMUL_HARDWARE_MATMUL")
    {
        return None;
    }
    let backend = match super::bitnet_disagg::tmatmul_backend_from_env() {
        Ok(Some(backend)) => backend,
        Ok(None) => super::bitnet_disagg::TmatmulBackend::Cxl,
        Err(error) => return Some(Err(error)),
    };
    match backend {
        super::bitnet_disagg::TmatmulBackend::Cxl => {
            nvidia_try_launch_named_cxl_tmatmul(kernel_name, kernel_params)
        }
        super::bitnet_disagg::TmatmulBackend::Xrt => {
            nvidia_try_launch_named_xrt_tmatmul(kernel_name, kernel_params, grid, stream)
        }
    }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn nvidia_try_launch_tmatmul_before_native(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    stream: cuda_types::cuda::CUstream,
) -> Option<CUresult> {
    if let Some(result) =
        super::nvint4_tmatmul::try_launch(kernel_name, kernel_params, grid, block, stream)
    {
        return Some(result.map_err(|err| {
            eprintln!("[NVINT4 TMatmul] strict launch failed: {err}");
            CUerror::UNKNOWN
        }));
    }

    if let Some(result) = nvidia_try_launch_named_tmatmul(kernel_name, kernel_params, grid, stream)
    {
        return Some(result.map_err(|err| {
            eprintln!("[TMatmul][NVIDIA] named launch failed: {err}");
            CUerror::UNKNOWN
        }));
    }

    None
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn launch_kernel(
    f: &super::module::NvidiaKernel,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    block_dim_y: ::core::ffi::c_uint,
    block_dim_z: ::core::ffi::c_uint,
    shared_mem_bytes: ::core::ffi::c_uint,
    h_stream: cuda_types::cuda::CUstream,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> CUresult {
    if let Some(result) = unsafe {
        nvidia_try_launch_tmatmul_before_native(
            &f.function_name,
            kernel_params,
            (grid_dim_x, grid_dim_y, grid_dim_z),
            (block_dim_x, block_dim_y, block_dim_z),
            h_stream,
        )
    } {
        return result;
    }
    nvidia_log_bitnet_route_for_native_launch(&f.function_name);
    let concordia_ptrs = nvidia_kimi_concordia_param_snapshot(&f.function_name, kernel_params);
    super::kimi_concordia::prepare_kernel_launch(&f.function_name, &concordia_ptrs);

    if super::persistent_router::try_route(
        &f.function_name,
        kernel_params,
        8,
        (grid_dim_x, grid_dim_y, grid_dim_z),
        (block_dim_x, block_dim_y, block_dim_z),
    ) {
        return Ok(());
    }

    let result = nvidia_runtime_sys::cuLaunchKernel(
        f.cuda_function,
        grid_dim_x,
        grid_dim_y,
        grid_dim_z,
        block_dim_x,
        block_dim_y,
        block_dim_z,
        shared_mem_bytes,
        h_stream,
        kernel_params,
        extra,
    );
    if result != 0 {
        eprintln!(
            "[NVIDIA Backend] cuLaunchKernel failed with error {}",
            result
        );
        return Err(CUerror::UNKNOWN);
    }
    nvidia_emit_gpu_attention_completed(&f.function_name);
    super::kimi_concordia::observe_kernel_launch(&f.function_name, h_stream, &concordia_ptrs);
    Ok(())
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn launch_kernel_ex(
    config: &cuda_types::cuda::CUlaunchConfig,
    f: &super::module::NvidiaKernel,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> CUresult {
    if let Some(result) = unsafe {
        nvidia_try_launch_tmatmul_before_native(
            &f.function_name,
            kernel_params,
            (config.gridDimX, config.gridDimY, config.gridDimZ),
            (config.blockDimX, config.blockDimY, config.blockDimZ),
            config.hStream,
        )
    } {
        return result;
    }
    nvidia_log_bitnet_route_for_native_launch(&f.function_name);
    let concordia_ptrs = nvidia_kimi_concordia_param_snapshot(&f.function_name, kernel_params);
    super::kimi_concordia::prepare_kernel_launch(&f.function_name, &concordia_ptrs);

    // cuLaunchKernelEx wraps cuLaunchKernel with additional config
    let result = nvidia_runtime_sys::cuLaunchKernel(
        f.cuda_function,
        config.gridDimX,
        config.gridDimY,
        config.gridDimZ,
        config.blockDimX,
        config.blockDimY,
        config.blockDimZ,
        config.sharedMemBytes,
        config.hStream,
        kernel_params,
        extra,
    );
    if result != 0 {
        eprintln!(
            "[NVIDIA Backend] cuLaunchKernelEx failed with error {}",
            result
        );
        return Err(CUerror::UNKNOWN);
    }
    nvidia_emit_gpu_attention_completed(&f.function_name);
    super::kimi_concordia::observe_kernel_launch(&f.function_name, config.hStream, &concordia_ptrs);
    Ok(())
}

#[cfg(all(
    test,
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
mod nvidia_bitnet_route_tests {
    use core::ffi::c_void;
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    static NVIDIA_ROUTE_TEST_MUTEX: Mutex<()> = Mutex::new(());

    struct EnvGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        previous: Vec<(&'static str, Option<String>)>,
    }

    impl EnvGuard {
        fn set(vars: &[(&'static str, Option<&str>)]) -> Self {
            let lock = super::super::test_env::lock();
            let previous = vars
                .iter()
                .map(|(name, _)| (*name, std::env::var(name).ok()))
                .collect::<Vec<_>>();
            for (name, value) in vars {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
            Self {
                _lock: lock,
                previous,
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (name, value) in self.previous.drain(..) {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }

    #[test]
    fn nvidia_native_launch_records_bitnet_route_before_passthrough() {
        let _mutex = NVIDIA_ROUTE_TEST_MUTEX.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let route_log = dir.path().join("routes.jsonl");
        let route_log_text = route_log.to_string_lossy().to_string();
        let _guard = EnvGuard::set(&[
            ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
            ("HETGPU_BITNET_FFN_CXL", None),
            ("HETGPU_TMATMUL_BITNET_DISAGGREGATE", None),
            ("HETGPU_BITNET_DISAGG_STRICT", None),
            ("HETGPU_BITNET_CXL_KERNELS", Some("ffn_gate")),
            ("HETGPU_BITNET_GPU_KERNELS", None),
            ("HETGPU_BITNET_ROUTE_MANIFEST", None),
            ("HETGPU_BITNET_ROUTE_LOG", Some(&route_log_text)),
            ("HETGPU_CXL_TMATMUL", Some("1")),
            ("HETGPU_TMATMUL_CXL", None),
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", Some("1")),
        ]);

        super::nvidia_log_bitnet_route_for_native_launch("layer_0_ffn_gate_mul_mat");

        let logged = std::fs::read_to_string(&route_log).unwrap();
        assert!(logged.contains(r#""kernel":"layer_0_ffn_gate_mul_mat""#));
        assert!(logged.contains(r#""route":"cxl_tmatmul""#));
        assert!(logged.contains(r#""source":"explicit_cxl_env""#));
        assert!(logged.contains(r#""cxl_enabled":true"#));
        assert!(logged.contains(r#""hardware_matmul_enabled":false"#));
    }

    #[test]
    fn nvidia_named_cuda13_attention_launch_logs_gpu_route_and_passes_through() {
        let _mutex = NVIDIA_ROUTE_TEST_MUTEX.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let route_log = dir.path().join("routes.jsonl");
        let route_log_text = route_log.to_string_lossy().to_string();
        let _guard = EnvGuard::set(&[
            ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
            ("HETGPU_BITNET_DISAGG_STRICT", Some("1")),
            ("HETGPU_BITNET_CXL_KERNELS", Some("ggml_type19")),
            ("HETGPU_BITNET_GPU_KERNELS", Some("flash,attn,attention")),
            ("HETGPU_BITNET_ROUTE_MANIFEST", None),
            ("HETGPU_BITNET_ROUTE_LOG", Some(&route_log_text)),
            ("HETGPU_TMATMUL_BACKEND", Some("xrt")),
            ("HETGPU_CXL_TMATMUL", None),
            ("HETGPU_TMATMUL_CXL", None),
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", Some("1")),
        ]);
        let kernel_name = std::ffi::CString::new("flash_attn_fwd").unwrap();

        let result = unsafe {
            super::launch_named_kernel_c(
                kernel_name.as_ptr(),
                1,
                1,
                1,
                32,
                1,
                1,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };

        assert_eq!(result, 1, "attention must continue through native CUDA");
        let logged = std::fs::read_to_string(&route_log).unwrap();
        assert!(logged.contains(r#""kernel":"flash_attn_fwd""#));
        assert!(logged.contains(r#""route":"gpu""#));
        assert!(logged.contains(r#""backend":"xrt""#));
        assert!(logged.contains(r#""hardware_matmul_enabled":false"#));
    }

    #[test]
    fn nvidia_named_cxl_candidate_rejects_before_native_when_strict() {
        let _mutex = NVIDIA_ROUTE_TEST_MUTEX.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let route_log = dir.path().join("routes.jsonl");
        let route_log_text = route_log.to_string_lossy().to_string();
        let _guard = EnvGuard::set(&[
            ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
            ("HETGPU_BITNET_FFN_CXL", None),
            ("HETGPU_TMATMUL_BITNET_DISAGGREGATE", None),
            ("HETGPU_BITNET_DISAGG_STRICT", Some("1")),
            ("HETGPU_BITNET_CXL_KERNELS", Some("mul_mat_q")),
            ("HETGPU_BITNET_GPU_KERNELS", None),
            ("HETGPU_BITNET_ROUTE_MANIFEST", None),
            ("HETGPU_BITNET_ROUTE_LOG", Some(&route_log_text)),
            ("HETGPU_CXL_TMATMUL", Some("1")),
            ("HETGPU_TMATMUL_CXL", None),
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", Some("1")),
            ("HETGPU_TMATMUL_MATRIX_STAGE", Some("cuda_dax")),
            ("HETGPU_TMATMUL_IO_STAGE", Some("cuda_dax")),
        ]);

        let result = unsafe {
            super::nvidia_try_launch_named_cxl_tmatmul(
                "_Z9mul_mat_qIL9ggml_type20ELi32ELi8ELb0EEvPKcS2_PfS3_iiiiiii",
                std::ptr::null_mut(),
            )
        };

        assert!(matches!(result, Some(Err(_))));
        let logged = std::fs::read_to_string(&route_log).unwrap();
        assert!(logged.contains(r#""route":"cxl_tmatmul""#));
        assert!(logged.contains(r#""hardware_matmul_enabled":true"#));
    }

    #[test]
    fn nvidia_named_cxl_candidate_accepts_cuda_host_ioctl_staging() {
        let _mutex = NVIDIA_ROUTE_TEST_MUTEX.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let route_log = dir.path().join("routes.jsonl");
        let route_log_text = route_log.to_string_lossy().to_string();
        let _guard = EnvGuard::set(&[
            ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
            ("HETGPU_BITNET_DISAGG_STRICT", Some("1")),
            ("HETGPU_BITNET_CXL_KERNELS", Some("mul_mat_q")),
            ("HETGPU_BITNET_ROUTE_LOG", Some(&route_log_text)),
            ("HETGPU_CXL_TMATMUL", Some("1")),
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", Some("1")),
            ("HETGPU_TMATMUL_MATRIX_STAGE", Some("cuda_host")),
            ("HETGPU_TMATMUL_IO_STAGE", Some("cuda_host")),
            ("HETGPU_CXL_TMATMUL_STAGING", Some("ioctl")),
        ]);

        let error = unsafe {
            super::nvidia_try_launch_named_cxl_tmatmul(
                "_Z9mul_mat_qIL9ggml_type19ELi32ELi8ELb0EEvPKcS2_PfS3_iiiiiii",
                std::ptr::null_mut(),
            )
        }
        .expect("strict CXL route must be consumed")
        .expect_err("null IQ1_S parameters must fail after staging selection");

        assert!(!error.contains("requires HETGPU_TMATMUL_MATRIX_STAGE"));
        assert!(!error.contains("requires HETGPU_TMATMUL_IO_STAGE"));
        assert!(error.contains("IQ1_S") || error.contains("kernel_params"));
        let logged = std::fs::read_to_string(&route_log).unwrap();
        assert!(logged.contains(r#""route":"cxl_tmatmul""#));
    }

    #[test]
    fn nvidia_named_xrt_candidate_is_selected_without_cxl_staging() {
        let _mutex = NVIDIA_ROUTE_TEST_MUTEX.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let route_log = dir.path().join("routes.jsonl");
        let route_log_text = route_log.to_string_lossy().to_string();
        let _guard = EnvGuard::set(&[
            ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
            ("HETGPU_BITNET_DISAGG_STRICT", Some("1")),
            ("HETGPU_BITNET_CXL_KERNELS", Some("mul_mat_q")),
            ("HETGPU_BITNET_ROUTE_LOG", Some(&route_log_text)),
            ("HETGPU_TMATMUL_BACKEND", Some("xrt")),
            ("HETGPU_CXL_TMATMUL", None),
            ("HETGPU_TMATMUL_CXL", None),
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", Some("1")),
            ("HETGPU_TMATMUL_MATRIX_STAGE", None),
            ("HETGPU_TMATMUL_IO_STAGE", None),
        ]);

        let result = unsafe {
            super::nvidia_try_launch_named_tmatmul(
                "_Z9mul_mat_qIL9ggml_type19ELi32ELi8ELb0EEvPKcS2_PfS3_iiiiiii",
                std::ptr::null_mut(),
                (1, 1, 1),
                cuda_types::cuda::CUstream(std::ptr::null_mut()),
            )
        }
        .expect("selected XRT route must be consumed")
        .expect_err("null launch parameters must fail before XRT execution");

        assert!(
            result.contains("PARAM_4") || result.contains("IQ1_S"),
            "{result}"
        );
        let logged = std::fs::read_to_string(&route_log).unwrap();
        assert!(logged.contains(r#""route":"cxl_tmatmul""#));
        assert!(logged.contains(r#""backend":"xrt""#));
        assert!(logged.contains(r#""xrt_enabled":true"#));
        assert!(logged.contains(r#""cxl_enabled":false"#));
    }

    #[test]
    fn nvidia_named_route_rejects_invalid_physical_backend() {
        let _mutex = NVIDIA_ROUTE_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", Some("1")),
            ("HETGPU_TMATMUL_BACKEND", Some("cuda")),
            ("HETGPU_CXL_TMATMUL", None),
        ]);

        let error = unsafe {
            super::nvidia_try_launch_named_tmatmul(
                "ffn_mul_mat_q",
                std::ptr::null_mut(),
                (1, 1, 1),
                cuda_types::cuda::CUstream(std::ptr::null_mut()),
            )
        }
        .expect("invalid backend must be consumed")
        .expect_err("invalid backend must fail closed");
        assert!(error.contains("cuda"), "{error}");
    }

    #[test]
    fn nvidia_named_xrt_does_not_claim_attention_or_non_iq1s_kernels() {
        let _mutex = NVIDIA_ROUTE_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[
            ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
            ("HETGPU_BITNET_DISAGG_STRICT", Some("1")),
            ("HETGPU_BITNET_CXL_KERNELS", Some("mul_mat_q,flash_attn")),
            ("HETGPU_TMATMUL_BACKEND", Some("xrt")),
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", Some("1")),
        ]);

        let attention = unsafe {
            super::nvidia_try_launch_named_tmatmul(
                "flash_attn_fwd",
                std::ptr::null_mut(),
                (1, 1, 1),
                cuda_types::cuda::CUstream(std::ptr::null_mut()),
            )
        };
        let q4 = unsafe {
            super::nvidia_try_launch_named_tmatmul(
                "_Z9mul_mat_qIL9ggml_type12ELi32ELi8ELb0EEvPKcS2_PfS3_iiiiiii",
                std::ptr::null_mut(),
                (1, 1, 1),
                cuda_types::cuda::CUstream(std::ptr::null_mut()),
            )
        };

        assert!(attention.is_none(), "attention must remain on the GPU");
        assert!(q4.is_none(), "only exact IQ1_S matmul may use XRT");
        assert!(
            super::nvidia_iq1s_xrt_kernel("_Z17mul_mat_vec_q_moeIL9ggml_type19ELi2EEvPKvS2_PKiPf")
                .is_some(),
            "the pinned multi-token MoE ABI must be intercepted for continuous batching"
        );
    }

    #[test]
    fn nvidia_xrt_strict_execution_failure_is_consumed() {
        let mut copied = false;
        let result = super::nvidia_xrt_post_execution_route(
            "qualified_iq1s_kernel",
            true,
            Err("AU250 completion missing request id 7".to_string()),
            |_| copied = true,
        );

        let error = result
            .expect("strict XRT failure must be consumed")
            .expect_err("strict XRT failure must not launch the native kernel");
        assert!(error.contains("missing request id 7"), "{error}");
        assert!(!copied, "failed execution must not copy an output to CUDA");
    }

    #[test]
    fn nvidia_xrt_compare_limit_accepts_nonnegative_u64_and_rejects_negative() {
        let _mutex = NVIDIA_ROUTE_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set(&[("HETGPU_XRT_COMPARE_MAX_LAUNCHES", Some("7"))]);
        assert_eq!(super::nvidia_xrt_compare_max_launches().unwrap(), 7);
        drop(_guard);

        let _guard = EnvGuard::set(&[("HETGPU_XRT_COMPARE_MAX_LAUNCHES", Some("-1"))]);
        let error = super::nvidia_xrt_compare_max_launches().unwrap_err();
        assert!(error.contains("nonnegative integer"), "{error}");
    }

    #[test]
    fn nvidia_pre_native_launch_consumes_named_cxl_candidate_when_strict() {
        let _mutex = NVIDIA_ROUTE_TEST_MUTEX.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let route_log = dir.path().join("routes.jsonl");
        let route_log_text = route_log.to_string_lossy().to_string();
        let _guard = EnvGuard::set(&[
            ("HETGPU_NVINT4_TMATMUL", None),
            ("HETGPU_NVINT4_BITLINEAR_HOOK", None),
            ("HETGPU_NVINT4_GPU_FALLBACK", None),
            ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
            ("HETGPU_BITNET_FFN_CXL", None),
            ("HETGPU_TMATMUL_BITNET_DISAGGREGATE", None),
            ("HETGPU_BITNET_DISAGG_STRICT", Some("1")),
            ("HETGPU_BITNET_CXL_KERNELS", Some("mul_mat_q")),
            ("HETGPU_BITNET_GPU_KERNELS", None),
            ("HETGPU_BITNET_ROUTE_MANIFEST", None),
            ("HETGPU_BITNET_ROUTE_LOG", Some(&route_log_text)),
            ("HETGPU_CXL_TMATMUL", Some("1")),
            ("HETGPU_TMATMUL_CXL", None),
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", Some("1")),
            ("HETGPU_TMATMUL_MATRIX_STAGE", Some("cuda_dax")),
            ("HETGPU_TMATMUL_IO_STAGE", Some("cuda_dax")),
        ]);

        let result = unsafe {
            super::nvidia_try_launch_tmatmul_before_native(
                "_Z9mul_mat_qIL9ggml_type19ELi32ELi8ELb0EEvPKcS2_PfS3_iiiiiii",
                std::ptr::null_mut(),
                (1, 1, 1),
                (1, 1, 1),
                cuda_types::cuda::CUstream(std::ptr::null_mut()),
            )
        };

        assert_eq!(result, Some(Err(cuda_types::cuda::CUerror::UNKNOWN)));
        let logged = std::fs::read_to_string(&route_log).unwrap();
        assert!(logged.contains(r#""route":"cxl_tmatmul""#));
        assert!(logged.contains(r#""hardware_matmul_enabled":true"#));
    }

    #[test]
    fn nvidia_mmvq_contract_rejects_qwen05b_shape_and_block_quantized_abi() {
        let shape = super::NvidiaCxlMmvqShape {
            ncols_x: 896,
            nrows_x: 896,
            nrows_y: 896,
            nrows_dst: 896,
        };

        let err = super::nvidia_cxl_mmvq_contract_error(shape).unwrap();
        assert!(err.contains("ncols_x=896"), "unexpected error: {err}");
        assert!(err.contains("requires 2048"), "unexpected error: {err}");

        let square = super::NvidiaCxlMmvqShape {
            ncols_x: 2048,
            nrows_x: 2048,
            nrows_y: 2048,
            nrows_dst: 2048,
        };
        let err = super::nvidia_cxl_mmvq_contract_error(square).unwrap();
        assert!(
            err.contains("GGML block-quantized"),
            "unexpected error: {err}"
        );
        assert!(err.contains("dense NVINT8"), "unexpected error: {err}");
    }

    #[test]
    fn nvidia_modern_iq1s_mmvq_plans_selected_expert_offsets() {
        let layout = super::NvidiaModernMmvqLayout {
            ncols_x: 4096,
            nchannels_y: super::NvidiaCudaUint3 { x: 0, y: 0, z: 8 },
            stride_row_x: 16,
            stride_col_y: 128,
            stride_col_dst: 10240,
            stride_channel_x: 16384,
            stride_channel_y: 128,
            stride_channel_dst: 1024,
            sample_ratio: super::NvidiaCudaUint3 { x: 0, y: 0, z: 1 },
            stride_sample_x: 512 * 16384,
            stride_sample_y: 1024,
            stride_sample_dst: 10240,
        };
        let plans = super::nvidia_plan_modern_iq1s_mmvq(
            layout,
            0x1000_0000,
            0x2000_0000,
            0x3000_0000,
            &[7, 3],
            (1024, 2, 1),
        )
        .unwrap();
        assert_eq!(plans.len(), 2);
        assert_eq!(plans[0].matrix_ptr, 0x1000_0000 + 7 * 16384 * 50);
        assert_eq!(plans[1].matrix_ptr, 0x1000_0000 + 3 * 16384 * 50);
        assert_eq!(plans[0].activation_ptr, 0x2000_0000);
        assert_eq!(plans[1].activation_ptr, 0x2000_0000 + 128 * 36);
        assert_eq!(plans[0].output_ptr, 0x3000_0000);
        assert_eq!(plans[1].output_ptr, 0x3000_0000 + 1024 * 4);
        assert_eq!(plans[0].signature.ncols_x, 4096);
        assert_eq!(plans[0].signature.nrows_x, 1024);
    }

    #[test]
    fn nvidia_iq1s_capture_identity_comes_from_registered_weight_not_environment() {
        use crate::r#impl::iq1s_weight_registry::{
            Iq1sDeviceBinding, Iq1sExpertRole, Iq1sTensorRegistration, Iq1sWeightRegistry,
        };

        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&vec![0x5c; 200]).unwrap();
        file.flush().unwrap();
        let registry = Iq1sWeightRegistry::default();
        let source = registry
            .register(Iq1sTensorRegistration {
                path: file.path().to_path_buf(),
                file_offset: 0,
                nbytes: 200,
                name: "blk.4.ffn_down_exps.weight".to_string(),
                ne: [256, 2, 2, 1],
                nb: [50, 50, 100, 200],
                role: Iq1sExpertRole::Down,
                ggml_type: 19,
                model_sha256: [0x33; 32],
            })
            .unwrap();
        registry
            .bind(Iq1sDeviceBinding {
                name: source.identity.name.clone(),
                base_ptr: 0x70_000,
                allocation_bytes: 200,
                allocation_generation: 27,
            })
            .unwrap();

        let identity = super::nvidia_resolve_registered_iq1s_launch(
            &registry,
            0x70_000 + 100,
            100,
            256,
            2,
            50,
        )
        .unwrap();
        assert_eq!(identity.allocation_generation, 27);
        assert_eq!(identity.content_hash, source.identity.content_sha256);
        assert_eq!(identity.tensor_name, source.identity.name);
        assert_eq!(identity.expert, 1);

        let error = super::nvidia_resolve_registered_iq1s_launch(
            &registry,
            0x80_000,
            100,
            256,
            2,
            50,
        )
        .unwrap_err();
        assert!(error.contains("registered device binding"), "{error}");
    }

    #[test]
    fn nvidia_modern_iq1s_mmvq_rejects_fused_or_noncontiguous_layouts() {
        let kernel = "_Z13mul_mat_vec_qIL9ggml_type19ELi1ELb1ELb0ELb0EEv";
        assert!(super::nvidia_modern_iq1s_mmvq_has_fusion(kernel));

        let layout = super::NvidiaModernMmvqLayout {
            ncols_x: 4096,
            nchannels_y: super::NvidiaCudaUint3 { x: 0, y: 0, z: 1 },
            stride_row_x: 17,
            stride_col_y: 128,
            stride_col_dst: 4096,
            stride_channel_x: 65536,
            stride_channel_y: 128,
            stride_channel_dst: 4096,
            sample_ratio: super::NvidiaCudaUint3 { x: 0, y: 0, z: 1 },
            stride_sample_x: 0,
            stride_sample_y: 128,
            stride_sample_dst: 4096,
        };
        let error =
            super::nvidia_plan_modern_iq1s_mmvq(layout, 0x1000, 0x2000, 0x3000, &[0], (4, 1, 1))
                .unwrap_err();
        assert!(error.contains("stride_row_x"));
    }

    #[test]
    fn nvidia_modern_iq1s_moe_plans_every_token_and_selected_expert() {
        let layout = super::NvidiaModernMoeMmvqLayout {
            ncols_x: 4096,
            nchannels_y: super::NvidiaCudaUint3 { x: 0, y: 0, z: 8 },
            nrows_x: 1024,
            stride_row_x: 16,
            stride_col_y: 1024,
            stride_col_dst: 8192,
            stride_channel_x: 16384,
            stride_channel_y: 128,
            stride_channel_dst: 1024,
            ncols_dst: 2,
            ids_stride: 8,
        };
        let plans = super::nvidia_plan_modern_iq1s_moe_mmvq(
            layout,
            0x1000_0000,
            0x2000_0000,
            0x3000_0000,
            &[7, 3, 0, 0, 0, 0, 0, 0, 5, 1],
            (512, 2, 1),
        )
        .unwrap();

        assert_eq!(plans.len(), 4);
        assert_eq!(plans[0].matrix_ptr, 0x1000_0000 + 7 * 16384 * 50);
        assert_eq!(plans[1].matrix_ptr, 0x1000_0000 + 3 * 16384 * 50);
        assert_eq!(plans[2].matrix_ptr, 0x1000_0000 + 5 * 16384 * 50);
        assert_eq!(plans[3].matrix_ptr, 0x1000_0000 + 1 * 16384 * 50);
        assert_eq!(plans[0].activation_ptr, 0x2000_0000);
        assert_eq!(plans[1].activation_ptr, 0x2000_0000 + 128 * 36);
        assert_eq!(plans[2].activation_ptr, 0x2000_0000 + 1024 * 36);
        assert_eq!(plans[3].activation_ptr, 0x2000_0000 + (1024 + 128) * 36);
        assert_eq!(plans[0].output_ptr, 0x3000_0000);
        assert_eq!(plans[1].output_ptr, 0x3000_0000 + 1024 * 4);
        assert_eq!(plans[2].output_ptr, 0x3000_0000 + 8192 * 4);
        assert_eq!(plans[3].output_ptr, 0x3000_0000 + (8192 + 1024) * 4);
    }

    #[test]
    fn nvidia_modern_iq1s_moe_rejects_nonexact_grid_and_ids_layout() {
        let layout = super::NvidiaModernMoeMmvqLayout {
            ncols_x: 4096,
            nchannels_y: super::NvidiaCudaUint3 { x: 0, y: 0, z: 1 },
            nrows_x: 1024,
            stride_row_x: 16,
            stride_col_y: 128,
            stride_col_dst: 4096,
            stride_channel_x: 16384,
            stride_channel_y: 128,
            stride_channel_dst: 1024,
            ncols_dst: 2,
            ids_stride: 1,
        };
        let error = super::nvidia_plan_modern_iq1s_moe_mmvq(
            layout,
            0x1000,
            0x2000,
            0x3000,
            &[0, 1],
            (511, 1, 1),
        )
        .unwrap_err();
        assert!(error.contains("grid.x"), "{error}");

        let error = super::nvidia_plan_modern_iq1s_moe_mmvq(
            layout,
            0x1000,
            0x2000,
            0x3000,
            &[0],
            (512, 1, 1),
        )
        .unwrap_err();
        assert!(error.contains("ids"), "{error}");
    }

    #[test]
    fn nvidia_mmq_contract_rejects_kimi_iq1s_shape_and_block_quantized_abi() {
        let shape = super::NvidiaCxlMmqShape {
            ne00: 7168,
            ne01: 2048,
            stride01: 1400,
            ne10: 7168,
            ne11: 1,
            stride11: 8064,
            ne0: 2048,
        };

        let err = super::nvidia_cxl_mmq_contract_error(shape);
        assert!(err.contains("ne00=7168"), "unexpected error: {err}");
        assert!(
            err.contains("requires 2048x2048"),
            "unexpected error: {err}"
        );

        let square = super::NvidiaCxlMmqShape {
            ne00: 2048,
            ne01: 2048,
            stride01: 400,
            ne10: 2048,
            ne11: 1,
            stride11: 2304,
            ne0: 2048,
        };
        let err = super::nvidia_cxl_mmq_contract_error(square);
        assert!(
            err.contains("packed quantized matrix/Q8_1 activation ABI"),
            "unexpected error: {err}"
        );
        assert!(err.contains("dense NVINT8"), "unexpected error: {err}");
    }

    #[test]
    fn nvidia_iq1s_signature_rejects_negative_signed_fields_before_conversion() {
        const KERNEL: &str = "_Z9mul_mat_qIL9ggml_type19ELi32ELi8ELb0EEvPKcS2_PfS3_iiiiiii";
        let valid = super::NvidiaCxlMmqShape {
            ne00: 256,
            ne01: 1,
            stride01: 1,
            ne10: 256,
            ne11: 1,
            stride11: 1,
            ne0: 1,
        };

        let mut negative_stride = valid;
        negative_stride.stride11 = -1;
        let error = super::nvidia_iq1s_signature(KERNEL, negative_stride).unwrap_err();
        assert!(error.contains("stride11"), "unexpected error: {error}");
        assert!(error.contains("negative"), "unexpected error: {error}");

        let mut negative_dimension = valid;
        negative_dimension.ne10 = -256;
        let error = super::nvidia_iq1s_signature(KERNEL, negative_dimension).unwrap_err();
        assert!(error.contains("ne10"), "unexpected error: {error}");
        assert!(error.contains("negative"), "unexpected error: {error}");
    }

    #[test]
    fn nvidia_iq1s_completed_evidence_is_stable_single_line_json() {
        let per_lane_completion_counts = [2, 1, 0, 1];
        let evidence = super::NvidiaIq1sCompletedEvidence {
            event: "ternip_v3_iq1s_execution",
            status: "completed",
            kernel: "qualified_iq1s_kernel",
            logical_batch: 4,
            descriptor_count: 4,
            unique_submission_count: 2,
            lane_mask: 0b1011,
            per_lane_completion_counts: &per_lane_completion_counts,
            total_accelerator_cycles: 12_345,
            total_matrix_bytes_read: 4_096,
            total_input_bytes_read: 512,
            total_output_bytes_written: 256,
        };

        let record = super::nvidia_iq1s_completed_evidence_json(evidence);

        assert_eq!(
            record,
            r#"{"event":"ternip_v3_iq1s_execution","status":"completed","kernel":"qualified_iq1s_kernel","logical_batch":4,"descriptor_count":4,"unique_submission_count":2,"lane_mask":11,"per_lane_completion_counts":[2,1,0,1],"total_accelerator_cycles":12345,"total_matrix_bytes_read":4096,"total_input_bytes_read":512,"total_output_bytes_written":256}"#
        );
        assert!(!record.contains('\n'));
    }

    #[test]
    fn nvidia_iq1s_completed_gpu_attention_evidence_is_native_cuda_only() {
        let evidence = super::GpuAttentionCompletedEvidence {
            event: "hetgpu_gpu_attention_execution",
            status: "completed",
            kernel: "flash_attn_fwd",
            launch_count: 1,
            device: "cuda",
            cpu_fallback: false,
            emulator_fallback: false,
        };
        let record = super::nvidia_gpu_attention_completed_evidence_json(evidence);
        assert_eq!(
            record,
            r#"{"event":"hetgpu_gpu_attention_execution","status":"completed","kernel":"flash_attn_fwd","launch_count":1,"device":"cuda","cpu_fallback":false,"emulator_fallback":false}"#
        );
        assert!(!record.contains('\n'));

        let mut config = crate::r#impl::bitnet_disagg::BitnetRouteConfig::default();
        config.enabled = true;
        assert!(
            super::nvidia_gpu_attention_completed_evidence("flash_attn_fwd", &config).is_some()
        );
        assert!(
            super::nvidia_gpu_attention_completed_evidence("layer_3_attention", &config).is_some()
        );
        assert!(super::nvidia_gpu_attention_completed_evidence(
            "layer_3_ffn_gate_mul_mat",
            &config
        )
        .is_none());

        config.cxl_markers = vec!["flash_attn".to_string()];
        assert!(
            super::nvidia_gpu_attention_completed_evidence("flash_attn_fwd", &config).is_none()
        );
    }

    fn nvidia_iq1s_test_execution_result() -> crate::r#impl::iq1s_tmatmul::ExecutionResult {
        use crate::r#impl::cxl_tmatmul_v3::{CompletedTaskV3, CompletionV3, TaskV3};

        fn completed(
            submission_id: u64,
            request_id: u64,
            lane_used: u32,
            accelerator_cycles: u64,
        ) -> CompletedTaskV3 {
            let mut task = TaskV3::default();
            task.request_id = request_id;
            task.batch = 1;
            let mut completion = CompletionV3::default();
            completion.request_id = request_id;
            completion.lane_used = lane_used;
            completion.accelerator_cycles = accelerator_cycles;
            completion.matrix_bytes_read = 1_024;
            completion.input_bytes_read = 128;
            completion.output_bytes_written = 64;
            CompletedTaskV3::test_only_new_unchecked(submission_id, task, completion)
        }

        let completions = [
            completed(41, 1, 0, 100),
            completed(41, 2, 1, 200),
            completed(42, 3, 3, 300),
            completed(42, 4, 0, 400),
        ];
        let scheduler =
            crate::r#impl::batch_scheduler::SchedulerReport::from_completions(&completions, 4)
                .unwrap();
        crate::r#impl::iq1s_tmatmul::ExecutionResult {
            outputs: Vec::new(),
            physical: Vec::new(),
            raw_components: Vec::new(),
            scheduler,
        }
    }

    #[test]
    fn nvidia_iq1s_completed_route_success_emits_one_parseable_record() {
        let kernel = "qualified\"iq1s\\kernel\nline";
        let mut emitted = Vec::new();

        let result = super::nvidia_iq1s_post_execution_route(
            kernel,
            4,
            true,
            Ok(nvidia_iq1s_test_execution_result()),
            |record| emitted.push(record.to_string()),
        );

        assert!(matches!(result, Some(Ok(()))));
        assert_eq!(emitted.len(), 1);
        assert!(!emitted[0].contains('\n'));
        assert!(!emitted[0].contains('\r'));
        let parsed: serde_json::Value = serde_json::from_str(&emitted[0]).unwrap();
        assert_eq!(parsed["event"], "ternip_v3_iq1s_execution");
        assert_eq!(parsed["status"], "completed");
        assert_eq!(parsed["kernel"].as_str(), Some(kernel));
        assert_eq!(parsed["logical_batch"], 4);
        assert_eq!(parsed["descriptor_count"], 4);
        assert_eq!(parsed["unique_submission_count"], 2);
        assert_eq!(parsed["lane_mask"], 0b1011);
        assert_eq!(
            parsed["per_lane_completion_counts"],
            serde_json::json!([2, 1, 0, 1])
        );
        assert_eq!(parsed["total_accelerator_cycles"], 1_000);
        assert_eq!(parsed["total_matrix_bytes_read"], 4_096);
        assert_eq!(parsed["total_input_bytes_read"], 512);
        assert_eq!(parsed["total_output_bytes_written"], 256);
    }

    #[test]
    fn nvidia_iq1s_completed_route_strict_error_emits_no_record() {
        let mut emitted = Vec::new();

        let result = super::nvidia_iq1s_post_execution_route(
            "qualified_iq1s_kernel",
            4,
            true,
            Err("output copy failed".to_string()),
            |record| emitted.push(record.to_string()),
        );

        let error = result
            .expect("strict route must consume the failed execution")
            .expect_err("strict route must return the execution error");
        assert!(error.contains("output copy failed"));
        assert!(emitted.is_empty());
    }

    #[test]
    fn nvidia_iq1s_completed_route_nonstrict_error_emits_no_record() {
        let mut emitted = Vec::new();

        let result = super::nvidia_iq1s_post_execution_route(
            "qualified_iq1s_kernel",
            4,
            false,
            Err("lease cleanup failed".to_string()),
            |record| emitted.push(record.to_string()),
        );

        assert!(result.is_none());
        assert!(emitted.is_empty());
    }

    #[test]
    fn nvidia_named_q4k_strict_gpu_route_falls_back_before_hardware_staging() {
        let _mutex = NVIDIA_ROUTE_TEST_MUTEX.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let route_log = dir.path().join("routes.jsonl");
        let route_log_text = route_log.to_string_lossy().to_string();
        let _guard = EnvGuard::set(&[
            ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
            ("HETGPU_BITNET_DISAGG_STRICT", Some("1")),
            ("HETGPU_BITNET_CXL_KERNELS", None),
            ("HETGPU_BITNET_GPU_KERNELS", Some("ggml_type12")),
            ("HETGPU_BITNET_ROUTE_MANIFEST", None),
            ("HETGPU_BITNET_ROUTE_LOG", Some(&route_log_text)),
            ("HETGPU_CXL_TMATMUL", Some("1")),
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", Some("1")),
            ("HETGPU_TMATMUL_MATRIX_STAGE", Some("cuda_dax")),
            ("HETGPU_TMATMUL_IO_STAGE", Some("cuda_dax")),
        ]);

        let mut ncols_x = 896i32;
        let mut nrows_x = 896i32;
        let mut nrows_y = 896i32;
        let mut nrows_dst = 896i32;
        let mut params = [
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            (&mut ncols_x as *mut i32).cast::<c_void>(),
            (&mut nrows_x as *mut i32).cast::<c_void>(),
            (&mut nrows_y as *mut i32).cast::<c_void>(),
            (&mut nrows_dst as *mut i32).cast::<c_void>(),
        ];
        let result = unsafe {
            super::nvidia_try_launch_named_cxl_tmatmul(
                "_Z13mul_mat_vec_qIL9ggml_type12ELi3EEvPKvS2_Pfiiii",
                params.as_mut_ptr(),
            )
        };

        assert!(result.is_none(), "incompatible GGML ABI must stay on GPU");
    }

    #[test]
    fn nvidia_named_iq1s_mmq_selects_exact_adapter_before_generic_abi_guard() {
        let _mutex = NVIDIA_ROUTE_TEST_MUTEX.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let route_log = dir.path().join("routes.jsonl");
        let route_log_text = route_log.to_string_lossy().to_string();
        let _guard = EnvGuard::set(&[
            ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
            ("HETGPU_BITNET_DISAGG_STRICT", Some("1")),
            ("HETGPU_BITNET_CXL_KERNELS", Some("mul_mat_q")),
            ("HETGPU_BITNET_GPU_KERNELS", None),
            ("HETGPU_BITNET_ROUTE_MANIFEST", None),
            ("HETGPU_BITNET_ROUTE_LOG", Some(&route_log_text)),
            ("HETGPU_CXL_TMATMUL", Some("1")),
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", Some("1")),
            ("HETGPU_TMATMUL_MATRIX_STAGE", Some("cuda_dax")),
            ("HETGPU_TMATMUL_IO_STAGE", Some("cuda_dax")),
        ]);

        let mut ne00 = 7168i32;
        let mut ne01 = 2048i32;
        let mut stride01 = 1400i32;
        let mut ne10 = 7168i32;
        let mut ne11 = 1i32;
        let mut stride11 = 8064i32;
        let mut ne0 = 2048i32;
        let mut params = [
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            (&mut ne00 as *mut i32).cast::<c_void>(),
            (&mut ne01 as *mut i32).cast::<c_void>(),
            (&mut stride01 as *mut i32).cast::<c_void>(),
            (&mut ne10 as *mut i32).cast::<c_void>(),
            (&mut ne11 as *mut i32).cast::<c_void>(),
            (&mut stride11 as *mut i32).cast::<c_void>(),
            (&mut ne0 as *mut i32).cast::<c_void>(),
        ];
        let result = unsafe {
            super::nvidia_try_launch_named_cxl_tmatmul(
                "_Z9mul_mat_qIL9ggml_type19ELi32ELi8ELb0EEvPKcS2_PfS3_iiiiiii",
                params.as_mut_ptr(),
            )
        };

        let error = result
            .expect("strict IQ1_S route must be consumed before native launch")
            .expect_err("null pointers must fail inside the IQ1_S adapter");
        assert!(
            error.contains("PARAM_0") || error.contains("IQ1_S"),
            "unexpected pre-adapter error: {error}"
        );
        assert!(
            !error.contains("packed quantized matrix/Q8_1 activation ABI"),
            "the generic dense-NVINT8 guard ran before the IQ1_S adapter: {error}"
        );
        let logged = std::fs::read_to_string(&route_log).unwrap();
        assert!(logged.contains(r#""route":"cxl_tmatmul""#));
        assert!(logged.contains(r#""hardware_matmul_enabled":true"#));
    }

    #[test]
    fn nvidia_named_iq1s_vec_selects_exact_adapter_before_generic_abi_guard() {
        let _mutex = NVIDIA_ROUTE_TEST_MUTEX.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let route_log = dir.path().join("routes.jsonl");
        let route_log_text = route_log.to_string_lossy().to_string();
        let _guard = EnvGuard::set(&[
            ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
            ("HETGPU_BITNET_DISAGG_STRICT", Some("1")),
            ("HETGPU_BITNET_CXL_KERNELS", Some("mul_mat_vec_q")),
            ("HETGPU_BITNET_GPU_KERNELS", None),
            ("HETGPU_BITNET_ROUTE_MANIFEST", None),
            ("HETGPU_BITNET_ROUTE_LOG", Some(&route_log_text)),
            ("HETGPU_CXL_TMATMUL", Some("1")),
            ("HETGPU_TMATMUL_HARDWARE_MATMUL", Some("1")),
            ("HETGPU_TMATMUL_MATRIX_STAGE", Some("cuda_dax")),
            ("HETGPU_TMATMUL_IO_STAGE", Some("cuda_dax")),
        ]);

        let mut ncols_x = 2048i32;
        let mut nrows_x = 2048i32;
        let mut nrows_y = 2048i32;
        let mut nrows_dst = 2048i32;
        let mut params = [
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            (&mut ncols_x as *mut i32).cast::<c_void>(),
            (&mut nrows_x as *mut i32).cast::<c_void>(),
            (&mut nrows_y as *mut i32).cast::<c_void>(),
            (&mut nrows_dst as *mut i32).cast::<c_void>(),
        ];
        let result = unsafe {
            super::nvidia_try_launch_named_cxl_tmatmul(
                "_Z13mul_mat_vec_qIL9ggml_type19ELi1EEvPKvS2_Pfiiii",
                params.as_mut_ptr(),
            )
        };

        let error = result
            .expect("strict IQ1_S vector route must be consumed before native launch")
            .expect_err("null pointers must fail inside the IQ1_S vector adapter");
        assert!(
            error.contains("PARAM_0") || error.contains("IQ1_S"),
            "unexpected pre-adapter error: {error}"
        );
        assert!(
            !error.contains("packed quantized matrix/Q8_1 activation ABI"),
            "the generic dense-NVINT8 guard ran before the IQ1_S vector adapter: {error}"
        );
        let logged = std::fs::read_to_string(&route_log).unwrap();
        assert!(logged.contains(r#""route":"cxl_tmatmul"#));
        assert!(logged.contains(r#""hardware_matmul_enabled":true"#));
    }

    #[derive(Default)]
    struct FakeLayerCaptureSink {
        open: bool,
        capture_error: Option<&'static str>,
        captured: Arc<AtomicUsize>,
    }

    impl super::NvidiaIq1sLayerCaptureSink for FakeLayerCaptureSink {
        fn has_open_transaction(&self, _stream: usize) -> Result<bool, String> {
            Ok(self.open)
        }

        fn capture_projection(
            &self,
            _stream: usize,
            captured: Vec<crate::r#impl::iq1s_tmatmul::CapturedActivationLaunch>,
        ) -> Result<(), String> {
            if let Some(error) = self.capture_error {
                return Err(error.to_string());
            }
            self.captured.fetch_add(captured.len(), Ordering::SeqCst);
            Ok(())
        }
    }

    struct FakeCapturedExecutor {
        executions: Arc<AtomicUsize>,
    }

    impl crate::r#impl::iq1s_xrt::CapturedLaunchExecutor for FakeCapturedExecutor {
        type Output = ();

        fn execute(
            &mut self,
            _captured: &crate::r#impl::iq1s_tmatmul::CapturedLaunch,
        ) -> Result<Self::Output, String> {
            self.executions.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn nvidia_modern_iq1s_captured_fixture() -> crate::r#impl::iq1s_tmatmul::CapturedLaunch {
        use crate::r#impl::iq1s_tmatmul::{
            capture_from_host, GgmlType19Signature, GridTable, LogicalLaunch, GRID_ENTRIES,
            IQ1S_BLOCK_BYTES, Q8_1_MMQ_BYTES,
        };

        let signature = GgmlType19Signature {
            kernel: "mul_mat_q".to_string(),
            ne00: 256,
            ne01: 1,
            stride01: 1,
            ne10: 256,
            ne11: 1,
            stride11: 1,
            ne0: 1,
        };
        let launch = LogicalLaunch {
            matrix_ptr: 0x10_0000,
            activation_ptr: 0x20_0000,
            output_ptr: 0x30_0000,
            allocation_generation: 7,
            content_hash: [0x5a; 32],
            signature,
        };
        let matrix = [0u8; IQ1S_BLOCK_BYTES];
        let activations = vec![0u8; 2 * Q8_1_MMQ_BYTES];
        let grid: GridTable = [[0; 8]; GRID_ENTRIES];
        capture_from_host(launch, &matrix, &activations, &grid).unwrap()
    }

    fn nvidia_modern_iq1s_activation_fixture(
    ) -> crate::r#impl::iq1s_tmatmul::CapturedActivationLaunch {
        use crate::r#impl::iq1s_tmatmul::{
            capture_activation_from_host, GgmlType19Signature, LogicalLaunch, Q8_1_MMQ_BYTES,
        };

        let launch = LogicalLaunch {
            matrix_ptr: 0x10_0000,
            activation_ptr: 0x20_0000,
            output_ptr: 0x30_0000,
            allocation_generation: 7,
            content_hash: [0x5a; 32],
            signature: GgmlType19Signature {
                kernel: "mul_mat_q".to_string(),
                ne00: 256,
                ne01: 1,
                stride01: 1,
                ne10: 256,
                ne11: 1,
                stride11: 1,
                ne0: 1,
            },
        };
        let activations = vec![0u8; 2 * Q8_1_MMQ_BYTES];
        capture_activation_from_host(launch, &activations).unwrap()
    }

    #[test]
    fn nvidia_modern_iq1s_strict_persistent_captures_before_direct_execution() {
        let lightweight_captures = Arc::new(AtomicUsize::new(0));
        let full_captures = Arc::new(AtomicUsize::new(0));
        let captured_count = Arc::new(AtomicUsize::new(0));
        let execution_count = Arc::new(AtomicUsize::new(0));
        let sink = FakeLayerCaptureSink {
            open: true,
            capture_error: None,
            captured: captured_count.clone(),
        };
        let mut executor = FakeCapturedExecutor {
            executions: execution_count.clone(),
        };

        let disposition = super::nvidia_dispatch_iq1s_capture(
            true,
            true,
            0xabc0,
            &sink,
            || {
                lightweight_captures.fetch_add(10, Ordering::SeqCst);
                Ok(vec![nvidia_modern_iq1s_activation_fixture(); 10])
            },
            || {
                full_captures.fetch_add(10, Ordering::SeqCst);
                Ok(vec![nvidia_modern_iq1s_captured_fixture(); 10])
            },
            &mut executor,
        )
        .unwrap();

        assert!(matches!(disposition, super::NvidiaIq1sDispatch::Buffered));
        assert_eq!(lightweight_captures.load(Ordering::SeqCst), 10);
        assert_eq!(full_captures.load(Ordering::SeqCst), 0);
        assert_eq!(captured_count.load(Ordering::SeqCst), 10);
        assert_eq!(execution_count.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn nvidia_modern_iq1s_strict_persistent_requires_transaction_before_capture() {
        let lightweight_captures = Arc::new(AtomicUsize::new(0));
        let full_captures = Arc::new(AtomicUsize::new(0));
        let execution_count = Arc::new(AtomicUsize::new(0));
        let sink = FakeLayerCaptureSink {
            open: false,
            capture_error: None,
            captured: Arc::new(AtomicUsize::new(0)),
        };
        let mut executor = FakeCapturedExecutor {
            executions: execution_count.clone(),
        };

        let error = super::nvidia_dispatch_iq1s_capture(
            true,
            true,
            0xabc0,
            &sink,
            || {
                lightweight_captures.fetch_add(1, Ordering::SeqCst);
                Ok(vec![nvidia_modern_iq1s_activation_fixture()])
            },
            || {
                full_captures.fetch_add(1, Ordering::SeqCst);
                Ok(vec![nvidia_modern_iq1s_captured_fixture()])
            },
            &mut executor,
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("no layer transaction"), "{error}");
        assert_eq!(lightweight_captures.load(Ordering::SeqCst), 0);
        assert_eq!(full_captures.load(Ordering::SeqCst), 0);
        assert_eq!(execution_count.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn qwen_iq1s_layer_integration_strict_persistent_rejects_direct_executor_entry() {
        let lightweight_captures = Arc::new(AtomicUsize::new(0));
        let full_captures = Arc::new(AtomicUsize::new(0));
        let execution_count = Arc::new(AtomicUsize::new(0));
        let sink = FakeLayerCaptureSink {
            open: true,
            capture_error: None,
            captured: Arc::new(AtomicUsize::new(0)),
        };
        let mut executor = FakeCapturedExecutor {
            executions: execution_count.clone(),
        };

        let error = super::nvidia_dispatch_iq1s_capture(
            false,
            true,
            0xabc0,
            &sink,
            || {
                lightweight_captures.fetch_add(1, Ordering::SeqCst);
                Ok(vec![nvidia_modern_iq1s_activation_fixture()])
            },
            || {
                full_captures.fetch_add(1, Ordering::SeqCst);
                Ok(vec![nvidia_modern_iq1s_captured_fixture()])
            },
            &mut executor,
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("eligible_direct_route=1"), "{error}");
        assert_eq!(lightweight_captures.load(Ordering::SeqCst), 0);
        assert_eq!(full_captures.load(Ordering::SeqCst), 0);
        assert_eq!(execution_count.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn nvidia_modern_iq1s_open_transaction_buffers_ten_without_execution() {
        let captured_count = Arc::new(AtomicUsize::new(0));
        let execution_count = Arc::new(AtomicUsize::new(0));
        let sink = FakeLayerCaptureSink {
            open: true,
            capture_error: None,
            captured: captured_count.clone(),
        };
        let mut executor = FakeCapturedExecutor {
            executions: execution_count.clone(),
        };

        let disposition = super::nvidia_dispatch_iq1s_capture(
            true,
            false,
            0xabc0,
            &sink,
            || Ok(vec![nvidia_modern_iq1s_activation_fixture(); 10]),
            || Ok(vec![nvidia_modern_iq1s_captured_fixture(); 10]),
            &mut executor,
        )
        .unwrap();

        assert!(matches!(disposition, super::NvidiaIq1sDispatch::Buffered));
        assert_eq!(captured_count.load(Ordering::SeqCst), 10);
        assert_eq!(execution_count.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn nvidia_modern_iq1s_capture_errors_never_fall_back_to_execution() {
        for expected in ["unregistered matrix pointer", "expert ID mismatch"] {
            let captured_count = Arc::new(AtomicUsize::new(0));
            let execution_count = Arc::new(AtomicUsize::new(0));
            let sink = FakeLayerCaptureSink {
                open: true,
                capture_error: Some(expected),
                captured: captured_count.clone(),
            };
            let mut executor = FakeCapturedExecutor {
                executions: execution_count.clone(),
            };
            let error = super::nvidia_dispatch_iq1s_capture(
                true,
                false,
                0xabc0,
                &sink,
                || Ok(vec![nvidia_modern_iq1s_activation_fixture(); 10]),
                || Ok(vec![nvidia_modern_iq1s_captured_fixture(); 10]),
                &mut executor,
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains(expected), "{error}");
            assert_eq!(captured_count.load(Ordering::SeqCst), 0);
            assert_eq!(execution_count.load(Ordering::SeqCst), 0);
        }
    }

    #[test]
    fn nvidia_modern_iq1s_legacy_and_nontransaction_paths_still_execute() {
        for (modern_moe, open) in [(false, true), (true, false)] {
            let execution_count = Arc::new(AtomicUsize::new(0));
            let sink = FakeLayerCaptureSink {
                open,
                capture_error: None,
                captured: Arc::new(AtomicUsize::new(0)),
            };
            let mut executor = FakeCapturedExecutor {
                executions: execution_count.clone(),
            };
            let disposition = super::nvidia_dispatch_iq1s_capture(
                modern_moe,
                false,
                0xabc0,
                &sink,
                || Ok(vec![nvidia_modern_iq1s_activation_fixture(); 10]),
                || Ok(vec![nvidia_modern_iq1s_captured_fixture(); 10]),
                &mut executor,
            )
            .unwrap();
            assert!(matches!(
                disposition,
                super::NvidiaIq1sDispatch::Executed(_)
            ));
            assert_eq!(execution_count.load(Ordering::SeqCst), 10);
        }
    }
}

// ============================================================================
// SIFIVE function implementations (SiFive Intelligence XM / RISC-V IME)
// ============================================================================

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn get_attribute(
    pi: *mut i32,
    attrib: cuda_types::cuda::CUfunction_attribute,
    func: *mut crate::r#impl::module::SifiveKernel,
) -> cuda_types::cuda::CUresult {
    use cuda_types::cuda::*;
    if pi.is_null() || func.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    let result = match attrib {
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK => 1024,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES => 0,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES => 0,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES => 0,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS => 32,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_PTX_VERSION => 75,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_BINARY_VERSION => 75,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_CACHE_MODE_CA => 0,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES => 65536,
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT => 0,
        _ => return Err(CUerror::INVALID_VALUE),
    };
    unsafe { *pi = result };
    Ok(())
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn launch_kernel(
    f: *mut crate::r#impl::module::SifiveKernel,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    block_dim_y: ::core::ffi::c_uint,
    block_dim_z: ::core::ffi::c_uint,
    shared_mem_bytes: ::core::ffi::c_uint,
    stream: *mut ::core::ffi::c_void,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> cuda_types::cuda::CUresult {
    use cuda_types::cuda::*;
    if f.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }

    let kernel = unsafe { &*f };

    if std::env::var("HETGPU_SIFIVE_LOG_KERNEL_LAUNCHES")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!(
            "[SIFIVE Backend] Launching kernel '{}' grid=({},{},{}) block=({},{},{})",
            kernel.kernel_name,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            block_dim_x,
            block_dim_y,
            block_dim_z
        );
    }

    if sifive_driver_kernel_noop_enabled() {
        let launch_index =
            SIFIVE_DRIVER_KERNEL_NOOP_LAUNCH_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        let should_submit = {
            let first = sifive_driver_kernel_noop_first();
            let every = sifive_driver_kernel_noop_every();
            launch_index <= first || every <= 1 || (launch_index % every) == 0
        };

        if !should_submit {
            if SIFIVE_DRIVER_KERNEL_NOOP_LOG_COUNT.fetch_add(1, Ordering::Relaxed) < 5 {
                eprintln!(
                    "[SIFIVE Backend] driver KERNEL_SIFIVE_NOOP sampled out launch #{} for '{}'; success",
                    launch_index, kernel.kernel_name
                );
            }
            return Ok(());
        }

        let device_id = current_sifive_device_id_or_zero().max(0) as u32;
        let c_name = std::ffi::CString::new(kernel.kernel_name.as_str())
            .unwrap_or_else(|_| std::ffi::CString::new("<invalid>").unwrap());
        let rc = unsafe {
            sifive_runtime_sys::hetgpu_sifive_launch_kernel_noop(
                device_id,
                c_name.as_ptr(),
                grid_dim_x,
                grid_dim_y,
                grid_dim_z,
                block_dim_x,
                block_dim_y,
                block_dim_z,
            )
        };
        if rc == sifive_runtime_sys::sifive_Result_Success {
            if SIFIVE_DRIVER_KERNEL_NOOP_LOG_COUNT.fetch_add(1, Ordering::Relaxed) < 20 {
                eprintln!(
                    "[SIFIVE Backend] driver KERNEL_SIFIVE_NOOP submitted '{}' to sifive{} grid=({},{},{}) block=({},{},{})",
                    kernel.kernel_name,
                    device_id,
                    grid_dim_x,
                    grid_dim_y,
                    grid_dim_z,
                    block_dim_x,
                    block_dim_y,
                    block_dim_z
                );
            }
            return Ok(());
        }

        if sifive_named_fail_open_enabled() {
            eprintln!(
                "[SIFIVE Backend] driver KERNEL_SIFIVE_NOOP submit failed for '{}' on sifive{} rc={}; fail-open success",
                kernel.kernel_name, device_id, rc
            );
            return Ok(());
        }
        return Err(CUerror::UNKNOWN);
    }

    let strict = std::env::var("HETGPU_SIFIVE_STRICT").ok().as_deref() == Some("1");
    let allow_failed_kernel_skip = !strict
        && match std::env::var("HETGPU_SIFIVE_ALLOW_FAILED_KERNEL_SKIP")
            .ok()
            .map(|value| value.trim().to_ascii_lowercase())
        {
            Some(value) if value == "0" || value == "false" || value == "no" || value == "off" => {
                false
            }
            Some(_) => true,
            None => sifive_named_fail_open_enabled(),
        };

    if let Some(result) = unsafe {
        try_offload_named_sifive_kernel(
            &kernel.kernel_name,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            kernel_params,
        )
    } {
        return result;
    }

    if sifive_generic_kernel_fast_success_enabled() {
        sifive_log_limited(
            &SIFIVE_GENERIC_FAST_SUCCESS_LOG_COUNT,
            "HETGPU_CUDART_GENERIC_KERNEL_FAST_SUCCESS_LOG_LIMIT",
            8,
            || {
                eprintln!(
                    "[SIFIVE Backend] generic cudart kernel fast-success for '{}' grid=({},{},{}) block=({},{},{})",
                    kernel.kernel_name,
                    grid_dim_x,
                    grid_dim_y,
                    grid_dim_z,
                    block_dim_x,
                    block_dim_y,
                    block_dim_z
                );
            },
        );
        let _ = (shared_mem_bytes, stream, kernel_params, extra);
        return Ok(());
    }

    if kernel.kernel_ptr.is_null() {
        crate::r#impl::hetgpu_debug!(
            "[SIFIVE Backend] Missing SIFIVE kernel handle for '{}'",
            kernel.kernel_name
        );
        if strict || !allow_failed_kernel_skip {
            eprintln!(
                "[SIFIVE Backend] missing SIFIVE kernel handle for '{}'; refusing to skip kernel",
                kernel.kernel_name
            );
            return Err(CUerror::UNKNOWN);
        }
    } else {
        if unsafe { sifive_kernel_has_nonempty_elf(kernel.kernel_ptr) } {
            let abi_result = unsafe {
                configure_sifive_launch_abi(
                    kernel.kernel_ptr,
                    &kernel.kernel_name,
                    grid_dim_x,
                    grid_dim_y,
                    grid_dim_z,
                    kernel_params,
                    extra,
                )
            };
            if abi_result != sifive_runtime_sys::sifive_Result_Success {
                crate::r#impl::hetgpu_debug!(
                    "[SIFIVE Backend] configure_sifive_launch_abi failed: {}",
                    abi_result
                );
                if strict || !allow_failed_kernel_skip {
                    eprintln!(
                        "[SIFIVE Backend] configure_sifive_launch_abi failed for '{}' with rc={}; refusing to launch with stale/empty args",
                        kernel.kernel_name, abi_result
                    );
                    return Err(CUerror::UNKNOWN);
                }
            }
        } else if std::env::var("HETGPU_SIFIVE_LOG_KERNEL_LAUNCHES")
            .ok()
            .as_deref()
            == Some("1")
        {
            eprintln!(
                "[SIFIVE Backend] skipping launch ABI for '{}' because kernel ELF is empty",
                kernel.kernel_name
            );
        }
        let result = unsafe {
            sifive_runtime_sys::sifive_LaunchKernel(
                kernel.kernel_ptr,
                grid_dim_x,
                grid_dim_y,
                grid_dim_z,
                block_dim_x,
                block_dim_y,
                block_dim_z,
            )
        };
        if result != sifive_runtime_sys::sifive_Result_Success {
            crate::r#impl::hetgpu_debug!("[SIFIVE Backend] sifive_LaunchKernel failed: {}", result);
            if strict || !allow_failed_kernel_skip {
                eprintln!(
                    "[SIFIVE Backend] sifive_LaunchKernel failed for '{}' with rc={}; refusing to report success",
                    kernel.kernel_name, result
                );
                return Err(CUerror::UNKNOWN);
            }
        }
    }

    let _ = (shared_mem_bytes, stream, kernel_params, extra);
    Ok(())
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_max_launch_params() -> usize {
    std::env::var("HETGPU_SIFIVE_MAX_KERNEL_PARAMS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0 && n <= 256)
        .unwrap_or(32)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_known_kernel_param_count(kernel_name: &str) -> Option<usize> {
    let name = kernel_name.to_lowercase();

    if name.contains("deep_ep") {
        if name.contains("get_dispatch_layout") {
            return Some(9);
        }
        if name.contains("cached_notify_dispatch") {
            return Some(5);
        }
        if name.contains("notify_dispatch") {
            return Some(15);
        }
        if name.contains("cached_notify_combine") {
            return Some(7);
        }
        if name.contains("dispatch") {
            return Some(25);
        }
        if name.contains("combine") {
            return Some(17);
        }
        if name.contains("barrier") {
            return Some(3);
        }
    }

    // ggml-cuda direct launch kernels are passed through cudaLaunchKernel with a
    // plain `void **kernelParams` array. That array is not null-terminated, so
    // our generic "scan until NULL" path can walk past the end and segfault.
    // For the hot kernels we know today, pin the exact arity from the CUDA
    // kernel signatures and only read that many entries.
    if name.contains("mul_mat_vec_q_moe") {
        return Some(15);
    }
    if name.contains("topk_moe_cuda") {
        return Some(9);
    }
    if name.contains("mul_mat_vec_q") || name.contains("mul_mat_vec_f") {
        return Some(19);
    }
    if name.contains("mul_mat_q_stream_k_fixup") {
        return Some(13);
    }
    if name.contains("mul_mat_q") {
        return Some(23);
    }
    if name.contains("l2_norm_f32") {
        return Some(7);
    }
    if name.contains("rms_norm_back_f32") {
        return Some(5);
    }
    if name.contains("scale_f32") {
        return Some(5);
    }
    if name.contains("k_argsort_f32_i32") {
        return Some(4);
    }
    if name.contains("k_get_rows") {
        return Some(15);
    }
    if name.contains("compute_batched_ptrs") {
        return Some(16);
    }
    if name.contains("softmax") || name.contains("soft_max") {
        return Some(5);
    }
    if name.contains("quantize_q8_1") {
        return Some(9);
    }
    if name.contains("dequantize_block_q8_0_f16") {
        return Some(3);
    }
    if name.contains("convert_unary") {
        return Some(9);
    }
    if name.contains("concat_f32_non_cont") {
        return Some(27);
    }
    if name.contains("concat_f32_dim") {
        return Some(5);
    }
    if name.contains("cpy_scalar_contiguous") {
        return Some(3);
    }
    if name.contains("cpy_scalar") {
        return Some(17);
    }
    if name.contains("k_set_rows_quant") || name.contains("k_set_rows") {
        return Some(22);
    }
    if name.contains("k_bin_bcast_unravel") {
        return Some(24 + sifive_bin_bcast_fuse_count(kernel_name));
    }
    if name.contains("k_bin_bcast") {
        return Some(22 + sifive_bin_bcast_fuse_count(kernel_name));
    }
    if name.contains("rope_norm") || name.contains("rope_neox") || name.contains("rope_multi") {
        return Some(21);
    }
    if name.contains("unary_op_kernel") {
        return Some(3);
    }
    if name.contains("unary_gated_op_kernel") {
        return Some(7);
    }
    if name.contains("ssm_conv_f32") || name.contains("ssm_conv_long_token_f32") {
        return Some(12);
    }
    if name.contains("gated_delta_net_cuda") {
        return Some(22);
    }
    if name.contains("vectorized_gather_kernel") {
        return Some(9);
    }
    if name.contains("direct_copy_kernel_cuda")
        || (name.contains("unrolled_elementwise_kernel")
            && name.contains("loadwithcast")
            && name.contains("storewithcast"))
    {
        return Some(6);
    }
    if name.contains("bfloat16_copy_kernel_cuda") {
        return Some(3);
    }
    if name.contains("arange_cuda_out") || name.contains("elementwise_kernel_with_index") {
        return Some(4);
    }
    if name.contains("vectorized_elementwise_kernel") && name.contains("cudafunctoronself_add") {
        return Some(3);
    }
    if name.contains("vectorized_elementwise_kernel") && name.contains("sigmoid_kernel_cuda") {
        return Some(3);
    }
    if name.contains("vectorized_elementwise_kernel") && name.contains("silu_kernel") {
        return Some(3);
    }
    if name.contains("distribution_elementwise_grid_stride_kernel")
        && name.contains("uniform_kernel")
    {
        return Some(7);
    }
    if name.contains("vectorized_elementwise_kernel")
        && (name.contains("exp_kernel_cuda")
            || name.contains("log_kernel_cuda")
            || name.contains("softplus_kernel_cuda")
            || name.contains("softplus_kernel")
            || name.contains("neg_kernel_cuda"))
    {
        return Some(3);
    }
    if name.contains("vectorized_elementwise_kernel")
        && name.contains("aunaryfunctor")
        && name.contains("mulfunctor")
    {
        return Some(6);
    }
    if name.contains("elementwise_kernel")
        && (name.contains("cudafunctor_add")
            || name.contains("cudafunctor_mul")
            || name.contains("cudafunctor_div")
            || name.contains("mulfunctor"))
    {
        return Some(3);
    }
    if name.contains("vectorized_elementwise_kernel") && name.contains("fillfunctor") {
        return Some(3);
    }
    if name.contains("elementwise_kernel") && name.contains("comparefunctor") && name.contains("il")
    {
        return Some(4);
    }

    None
}

#[cfg(all(
    any(feature = "sifive", feature = "nvidia"),
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) unsafe fn launch_named_kernel_c(
    kernel_name: *const std::os::raw::c_char,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    block_dim_y: ::core::ffi::c_uint,
    block_dim_z: ::core::ffi::c_uint,
    shared_mem_bytes: ::core::ffi::c_uint,
    stream: *mut ::core::ffi::c_void,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> i32 {
    let _ = (
        block_dim_x,
        block_dim_y,
        block_dim_z,
        shared_mem_bytes,
        stream,
        extra,
    );
    if kernel_name.is_null() {
        return 1;
    }

    let kernel_name = match std::ffi::CStr::from_ptr(kernel_name).to_str() {
        Ok(name) => name,
        Err(_) => return 1,
    };

    #[cfg(all(
        feature = "nvidia",
        not(feature = "amd"),
        not(feature = "intel"),
        not(feature = "tenstorrent"),
        not(feature = "tmatmul")
    ))]
    if let Some(result) = nvidia_try_launch_tmatmul_before_native(
        kernel_name,
        kernel_params,
        (grid_dim_x, grid_dim_y, grid_dim_z),
        (block_dim_x, block_dim_y, block_dim_z),
        cuda_types::cuda::CUstream(stream.cast()),
    ) {
        return match result {
            Ok(()) => 0,
            Err(_) => 999,
        };
    }

    #[cfg(all(
        feature = "nvidia",
        not(feature = "amd"),
        not(feature = "intel"),
        not(feature = "tenstorrent"),
        not(feature = "tmatmul")
    ))]
    nvidia_log_bitnet_route_for_native_launch(kernel_name);

    #[cfg(feature = "sifive")]
    {
        match try_offload_named_sifive_kernel(
            kernel_name,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            kernel_params,
        ) {
            Some(Ok(())) => 0,
            Some(Err(_)) => 999,
            None => 1,
        }
    }

    #[cfg(not(feature = "sifive"))]
    {
        let _ = (grid_dim_x, grid_dim_y, grid_dim_z);
        1
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_looks_like_pointer(value: u64) -> bool {
    value > 0x1000 && value < 0x0000_8000_0000_0000
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_looks_like_host_param_addr(addr: usize) -> bool {
    if !(addr >= 0x1_0000 && addr < 0x0000_8000_0000_0000usize && (addr & 0x3) == 0) {
        return false;
    }

    if std::env::var("HETGPU_SIFIVE_VALIDATE_PARAM_ADDRS")
        .ok()
        .as_deref()
        != Some("1")
    {
        return true;
    }

    let Ok(maps) = std::fs::read_to_string("/proc/self/maps") else {
        return false;
    };

    for line in maps.lines() {
        let mut parts = line.split_whitespace();
        let Some(range) = parts.next() else {
            continue;
        };
        let Some(perms) = parts.next() else {
            continue;
        };
        if !perms.starts_with('r') {
            continue;
        }
        let Some((start_hex, end_hex)) = range.split_once('-') else {
            continue;
        };
        let Ok(start) = usize::from_str_radix(start_hex, 16) else {
            continue;
        };
        let Ok(end) = usize::from_str_radix(end_hex, 16) else {
            continue;
        };
        if addr >= start && addr.saturating_add(std::mem::size_of::<u64>()) <= end {
            return true;
        }
    }

    false
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[derive(Clone, Copy)]
struct SifiveHostMapRange {
    start: usize,
    end: usize,
    read: bool,
    write: bool,
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
static SIFIVE_HOST_MAPS_CACHE: OnceLock<Mutex<Vec<SifiveHostMapRange>>> = OnceLock::new();

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn parse_sifive_host_maps() -> Vec<SifiveHostMapRange> {
    let Ok(maps) = std::fs::read_to_string("/proc/self/maps") else {
        return Vec::new();
    };
    let mut ranges = Vec::new();
    for line in maps.lines() {
        let mut parts = line.split_whitespace();
        let Some(range) = parts.next() else {
            continue;
        };
        let Some(perms) = parts.next() else {
            continue;
        };
        let Some((start_hex, end_hex)) = range.split_once('-') else {
            continue;
        };
        let Ok(start) = usize::from_str_radix(start_hex, 16) else {
            continue;
        };
        let Ok(end) = usize::from_str_radix(end_hex, 16) else {
            continue;
        };
        ranges.push(SifiveHostMapRange {
            start,
            end,
            read: perms.starts_with('r'),
            write: perms.as_bytes().get(1).copied() == Some(b'w'),
        });
    }
    ranges
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_host_maps_contains(
    ranges: &[SifiveHostMapRange],
    addr: usize,
    end_addr: usize,
    need_write: bool,
) -> bool {
    ranges.iter().any(|range| {
        range.read && (!need_write || range.write) && addr >= range.start && end_addr <= range.end
    })
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_host_range_has_perms(addr: usize, len: usize, need_write: bool) -> bool {
    if len == 0 {
        return true;
    }
    let Some(end_addr) = addr.checked_add(len) else {
        return false;
    };
    let cache = SIFIVE_HOST_MAPS_CACHE.get_or_init(|| Mutex::new(parse_sifive_host_maps()));
    if let Ok(mut ranges) = cache.lock() {
        if ranges.is_empty() {
            *ranges = parse_sifive_host_maps();
        }
        if sifive_host_maps_contains(&ranges, addr, end_addr, need_write) {
            return true;
        }
    }
    false
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_kernel_has_nonempty_elf(
    kernel_ptr: *mut sifive_runtime_sys::sifive_Kernel,
) -> bool {
    if kernel_ptr.is_null() {
        return false;
    }
    let program = (*kernel_ptr).program;
    if program.is_null() {
        return false;
    }
    !(*program).elf_bytes.is_empty()
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn parse_sifive_launch_extra_blob(extra: *mut *mut ::core::ffi::c_void) -> Option<Vec<u8>> {
    use cuda_types::cuda::{
        CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT, CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT,
    };

    if extra.is_null() {
        return None;
    }

    let mut raw_ptr: *const u8 = std::ptr::null();
    let mut raw_size: usize = 0;
    let mut i = 0usize;
    loop {
        let key = *extra.add(i);
        if key.is_null() {
            break;
        }
        let value = *extra.add(i + 1);
        let key_usize = key as usize;
        if key_usize == CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT as usize {
            raw_ptr = value as *const u8;
        } else if key_usize == CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT as usize {
            if !value.is_null() {
                raw_size = (value as *const usize).read_unaligned();
            }
        }
        i += 2;
    }

    if raw_ptr.is_null() || raw_size == 0 {
        return None;
    }

    Some(std::slice::from_raw_parts(raw_ptr, raw_size).to_vec())
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn configure_sifive_launch_abi(
    kernel_ptr: *mut sifive_runtime_sys::sifive_Kernel,
    kernel_name: &str,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> sifive_runtime_sys::sifive_Result {
    if kernel_ptr.is_null() {
        return sifive_runtime_sys::sifive_Result_Error;
    }

    let clear = sifive_runtime_sys::sifive_KernelClearLaunchState(kernel_ptr);
    if clear != sifive_runtime_sys::sifive_Result_Success {
        return clear;
    }

    let mut raw_param_blob = parse_sifive_launch_extra_blob(extra).unwrap_or_default();

    if kernel_name.starts_with("lanxin_sifive_mul_mat_") {
        let m_v = read_param_i32(kernel_params, 0).unwrap_or(0).max(0) as u32;
        let n_v = read_param_i32(kernel_params, 1).unwrap_or(0).max(0) as u32;
        let k_v = read_param_i32(kernel_params, 2).unwrap_or(0).max(0) as u32;
        let a = read_param_u64(kernel_params, 3).unwrap_or(0) as *const ::core::ffi::c_void;
        let b = read_param_u64(kernel_params, 4).unwrap_or(0) as *const ::core::ffi::c_void;
        let c = read_param_u64(kernel_params, 5).unwrap_or(0) as *mut ::core::ffi::c_void;
        return sifive_runtime_sys::sifive_KernelConfigureLanxinMulMatTile(
            kernel_ptr, m_v, n_v, k_v, a, 0, b, 0, c, 0,
        );
    }

    if kernel_params.is_null() {
        if !raw_param_blob.is_empty() {
            let rc = sifive_runtime_sys::sifive_KernelSetRawParamBlob(
                kernel_ptr,
                raw_param_blob.as_ptr() as *const _,
                raw_param_blob.len() as u64,
            );
            if rc != sifive_runtime_sys::sifive_Result_Success {
                return rc;
            }
        }
        return sifive_runtime_sys::sifive_Result_Success;
    }

    let max_params =
        sifive_known_kernel_param_count(kernel_name).unwrap_or_else(sifive_max_launch_params);
    let log_launches = std::env::var("HETGPU_SIFIVE_LOG_KERNEL_LAUNCHES")
        .ok()
        .as_deref()
        == Some("1");
    let log_arg_records = log_launches
        && (kernel_name.contains("k_bin_bcast")
            || std::env::var("HETGPU_SIFIVE_LOG_KERNEL_ARGS")
                .ok()
                .as_deref()
                == Some("1"));
    let _ = (grid_dim_x, grid_dim_y);
    let mut pushed = 0usize;
    let mut pointer_like = 0usize;

    for i in 0..max_params {
        let param = *kernel_params.add(i);
        if param.is_null() {
            if sifive_known_kernel_param_count(kernel_name).is_some() {
                if log_launches {
                    eprintln!(
                        "[SIFIVE Backend] launch ABI '{}' hit unexpected null param at index {} of {}",
                        kernel_name, i, max_params
                    );
                }
                return sifive_runtime_sys::sifive_Result_Error;
            }
            break;
        }

        let arg_size = sifive_kernel_arg_size(kernel_name, i);
        let inline_immediate = if arg_size == 1 {
            let addr = param as usize;
            !(addr >= 0x1_0000 && addr < 0x0000_8000_0000_0000usize)
        } else {
            !sifive_looks_like_host_param_addr(param as usize)
        };
        let mut record_flags = 0u32;
        let (value, value_hi) = if !inline_immediate && arg_size > 16 {
            let offset = (raw_param_blob.len() + 7) & !7;
            raw_param_blob.resize(offset, 0);
            raw_param_blob
                .extend_from_slice(std::slice::from_raw_parts(param as *const u8, arg_size));
            record_flags |= sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_INLINE_BLOB;
            (offset as u64, 0)
        } else if inline_immediate {
            if arg_size > 16 {
                return sifive_runtime_sys::sifive_Result_Error;
            }
            (param as usize as u64, 0)
        } else {
            let mut lo = 0u64;
            let mut hi = 0u64;
            let lo_len = arg_size.min(std::mem::size_of::<u64>());
            ptr::copy_nonoverlapping(
                param as *const u8,
                (&mut lo as *mut u64).cast::<u8>(),
                lo_len,
            );
            if arg_size > std::mem::size_of::<u64>() {
                let hi_len =
                    (arg_size - std::mem::size_of::<u64>()).min(std::mem::size_of::<u64>());
                ptr::copy_nonoverlapping(
                    (param as *const u8).add(std::mem::size_of::<u64>()),
                    (&mut hi as *mut u64).cast::<u8>(),
                    hi_len,
                );
            }
            (lo, hi)
        };
        let can_be_pointer = !inline_immediate
            && arg_size == 8
            && sifive_looks_like_pointer(value)
            && sifive_kernel_arg_can_be_pointer(kernel_name, i);
        let binding_metadata = if can_be_pointer {
            sifive_kernel_binding_metadata(
                kernel_name,
                kernel_params,
                grid_dim_x,
                grid_dim_y,
                grid_dim_z,
                i,
            )
        } else {
            None
        };
        let is_pointer = can_be_pointer
            && (binding_metadata.is_some()
                || super::memory::sifive_allocation_remaining_addr(value).is_some());
        let record = sifive_runtime_sys::SifiveKernelArgRecord {
            kind: if is_pointer {
                sifive_runtime_sys::SIFIVE_KERNEL_ARG_KIND_POINTER
            } else {
                sifive_runtime_sys::SIFIVE_KERNEL_ARG_KIND_SCALAR
            },
            size: arg_size as u32,
            flags: record_flags,
            reserved: 0,
            value,
            value_hi,
        };
        let rc = sifive_runtime_sys::sifive_KernelPushArgRecord(kernel_ptr, &record);
        if rc != sifive_runtime_sys::sifive_Result_Success {
            return rc;
        }

        if log_arg_records {
            eprintln!(
                "[SIFIVE Backend] launch arg kernel='{}' idx={} param={:p} inline={} size={} kind={} flags=0x{:x} value=0x{:x} hi=0x{:x}",
                kernel_name,
                i,
                param,
                inline_immediate,
                record.size,
                record.kind,
                record.flags,
                record.value,
                record.value_hi,
            );
        }

        if is_pointer {
            let (size, flags) = binding_metadata
                .or_else(|| {
                    let remaining = super::memory::sifive_allocation_remaining_addr(value)? as u64;
                    Some((
                        remaining,
                        sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT
                            | sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
                    ))
                })
                .unwrap_or((0, 0));
            let (addr, flags) =
                if let Some(phys) = super::memory::sifive_shared_ddr_physical_addr(value) {
                    (
                        phys,
                        flags | sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_DEVICE_PHYS,
                    )
                } else {
                    (value, flags)
                };
            let binding = sifive_runtime_sys::SifiveKernelBufferBinding {
                arg_index: i as u32,
                flags,
                addr,
                size,
            };
            if log_arg_records {
                eprintln!(
                    "[SIFIVE Backend] launch binding kernel='{}' bind={} arg={} host=0x{:x} addr=0x{:x} size={} flags=0x{:x} direct_shared_ddr={}",
                    kernel_name,
                    pointer_like,
                    i,
                    value,
                    binding.addr,
                    binding.size,
                    binding.flags,
                    (binding.flags & sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_DEVICE_PHYS) != 0,
                );
            }
            let rc = sifive_runtime_sys::sifive_KernelAddBufferBinding(kernel_ptr, &binding);
            if rc != sifive_runtime_sys::sifive_Result_Success {
                return rc;
            }
            pointer_like += 1;
        }

        pushed += 1;
    }

    if !raw_param_blob.is_empty() {
        let rc = sifive_runtime_sys::sifive_KernelSetRawParamBlob(
            kernel_ptr,
            raw_param_blob.as_ptr() as *const _,
            raw_param_blob.len() as u64,
        );
        if rc != sifive_runtime_sys::sifive_Result_Success {
            return rc;
        }
    }

    if log_launches {
        eprintln!(
            "[SIFIVE Backend] launch ABI prepared for '{}' args={} pointer_like={}",
            kernel_name, pushed, pointer_like
        );
    }

    sifive_runtime_sys::sifive_Result_Success
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_kernel_arg_can_be_pointer(kernel_name: &str, index: usize) -> bool {
    let name = kernel_name.to_ascii_lowercase();
    if name.contains("deep_ep") {
        if name.contains("get_dispatch_layout") {
            return index <= 4;
        }
        if name.contains("cached_notify_dispatch") {
            return matches!(index, 0 | 2 | 3);
        }
        if name.contains("notify_dispatch") {
            return matches!(index, 0 | 1 | 2 | 3 | 7 | 8 | 9 | 12 | 13);
        }
        if name.contains("cached_notify_combine") {
            return matches!(index, 0 | 1 | 5);
        }
        if name.contains("dispatch") {
            return index <= 12 || index == 21;
        }
        if name.contains("combine") {
            return index <= 9 || index == 14;
        }
        if name.contains("barrier") {
            return index == 0;
        }
    }
    if name.contains("mul_mat_vec_q_moe") {
        return index <= 3;
    }
    if name.contains("topk_moe_cuda") {
        return index <= 3;
    }
    if (name.contains("mul_mat_vec_q") || name.contains("mul_mat_vec_f")) && index == 3 {
        return false;
    }
    if name.contains("gated_delta_net_cuda") {
        return index <= 6;
    }
    if name.contains("compute_batched_ptrs") {
        return (3..=4).contains(&index);
    }
    if name.contains("softmax") || name.contains("soft_max") {
        return index <= 3;
    }
    if name.contains("k_get_rows") {
        return index <= 2;
    }
    if name.contains("convert_unary") {
        return index <= 1;
    }
    if name.contains("cpy_scalar") {
        return index <= 1;
    }
    if name.contains("k_set_rows_quant") || name.contains("k_set_rows") {
        return index <= 2;
    }
    if name.contains("concat_f32") {
        return index <= 2;
    }
    if name.contains("op_clamp_kernel") {
        return index <= 1;
    }
    true
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_kernel_arg_size(kernel_name: &str, index: usize) -> usize {
    let name = kernel_name.to_ascii_lowercase();
    if name.contains("deep_ep") {
        if name.contains("get_dispatch_layout") {
            if matches!(index, 5..=8) {
                return 4;
            }
        } else if name.contains("cached_notify_dispatch") {
            if matches!(index, 1 | 4) {
                return 4;
            }
        } else if name.contains("notify_dispatch") {
            if matches!(index, 4..=6 | 10 | 11 | 14) {
                return 4;
            }
        } else if name.contains("cached_notify_combine") {
            if matches!(index, 2..=4 | 6) {
                return 4;
            }
        } else if name.contains("dispatch") {
            if matches!(index, 13..=20 | 22..=24) {
                return 4;
            }
        } else if name.contains("combine") {
            if matches!(index, 10..=13 | 15..=16) {
                return 4;
            }
        } else if name.contains("barrier") && index == 1 {
            return 4;
        }
    } else if name.contains("mul_mat_vec_q_moe") {
        if index == 5 {
            return 12;
        }
        if matches!(index, 4 | 6..=14) {
            return 4;
        }
    } else if name.contains("topk_moe_cuda") {
        if matches!(index, 4..=7) {
            return 4;
        }
        if index == 8 {
            return 3;
        }
    } else if name.contains("mul_mat_vec_q") || name.contains("mul_mat_vec_f") {
        if matches!(index, 6 | 10 | 14) {
            return 12;
        }
        if matches!(index, 5 | 7..=9 | 11..=18) {
            return 4;
        }
    } else if name.contains("k_bin_bcast_unravel") {
        if matches!(index, 3..=5 | 7..=12) {
            return 12;
        }
        if matches!(index, 6 | 13..=23) {
            return 4;
        }
    } else if name.contains("k_bin_bcast") {
        if matches!(index, 3..=5 | 11..=21) {
            return 4;
        }
        if matches!(index, 6..=10) {
            return 12;
        }
    } else if name.contains("rope_norm") || name.contains("rope_neox") {
        if matches!(index, 2..=11 | 13..=15 | 17 | 20) {
            return 4;
        }
        if index == 16 {
            return 8;
        }
    } else if name.contains("rope_multi") {
        if matches!(index, 2..=11 | 13..=15 | 17) {
            return 4;
        }
        if index == 16 {
            return 8;
        }
        if index == 19 {
            return 16;
        }
        if index == 20 {
            return 1;
        }
    } else if name.contains("scale_f32") && matches!(index, 2 | 3) {
        return 4;
    } else if name.contains("k_argsort_f32_i32") && matches!(index, 2 | 3) {
        return 4;
    } else if name.contains("concat_f32_dim") && matches!(index, 3 | 4) {
        return 4;
    } else if name.contains("op_clamp_kernel") {
        if matches!(index, 2 | 3) {
            return sifive_parse_op_clamp_element_size(kernel_name).unwrap_or(4) as usize;
        }
        if index == 4 {
            return 4;
        }
    } else if name.contains("unary_op_kernel") && index == 2 {
        return 4;
    } else if (name.contains("ssm_conv_f32") || name.contains("ssm_conv_long_token_f32"))
        && matches!(index, 3..=6 | 8..=10)
    {
        return 4;
    } else if name.contains("gated_delta_net_cuda") {
        if matches!(index, 19 | 20) {
            return 12;
        }
        if index == 21 {
            return 4;
        }
    } else if name.contains("convert_unary") && index == 5 {
        return 12;
    } else if (name.contains("softmax") || name.contains("soft_max")) && index == 4 {
        return std::mem::size_of::<SifiveSoftMaxParams>();
    } else if (name.contains("k_set_rows_quant") || name.contains("k_set_rows"))
        && matches!(index, 17..=21)
    {
        return 12;
    }
    8
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_kernel_binding_metadata(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    index: usize,
) -> Option<(u64, u32)> {
    let name = kernel_name.to_ascii_lowercase();
    if name.contains("deep_ep") && name.contains("get_dispatch_layout") {
        return sifive_deepep_layout_binding_metadata(kernel_params, index);
    }
    if name.contains("mul_mat_vec_q_moe") {
        return sifive_mul_mat_vec_q_moe_binding_metadata(
            kernel_name,
            kernel_params,
            grid_dim_y,
            index,
        );
    }
    if name.contains("topk_moe_cuda") {
        return sifive_topk_moe_binding_metadata(kernel_name, kernel_params, index);
    }
    if name.contains("mul_mat_vec_q") {
        return sifive_mul_mat_vec_q_binding_metadata(
            kernel_name,
            kernel_params,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            index,
        );
    }
    if name.contains("mul_mat_vec_f") {
        return sifive_mul_mat_vec_f_binding_metadata(
            kernel_name,
            kernel_params,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            index,
        );
    }
    if name.contains("scale_f32") {
        return sifive_scale_f32_binding_metadata(kernel_params, index);
    }
    if name.contains("k_argsort_f32_i32") {
        return sifive_argsort_f32_i32_binding_metadata(kernel_params, grid_dim_x, index);
    }
    if name.contains("k_get_rows_float") {
        return sifive_get_rows_float_binding_metadata(kernel_params, grid_dim_x, index);
    }
    if name.contains("k_get_rows") && name.contains("dequantize_q8_0") {
        return sifive_get_rows_q8_0_binding_metadata(
            kernel_name,
            kernel_params,
            grid_dim_x,
            index,
        );
    }
    if name.contains("l2_norm_f32") {
        return sifive_l2_norm_f32_binding_metadata(
            kernel_params,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            index,
        );
    }
    if name.contains("compute_batched_ptrs") {
        return sifive_compute_batched_ptrs_binding_metadata(kernel_params, index);
    }
    if name.contains("softmax") || name.contains("soft_max") {
        return sifive_softmax_binding_metadata(kernel_name, kernel_params, index);
    }
    if name.contains("quantize_q8_1") {
        return sifive_quantize_q8_1_binding_metadata(kernel_params, grid_dim_z, index);
    }
    if name.contains("dequantize_block_q8_0_f16") {
        return sifive_dequantize_block_q8_0_f16_binding_metadata(kernel_params, index);
    }
    if name.contains("convert_unary") {
        return sifive_convert_unary_binding_metadata(kernel_name, kernel_params, index);
    }
    if name.contains("concat_f32_non_cont") {
        return sifive_concat_non_cont_binding_metadata(
            kernel_params,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            index,
        );
    }
    if name.contains("concat_f32_dim") {
        return sifive_concat_dim_binding_metadata(
            kernel_name,
            kernel_params,
            grid_dim_y,
            grid_dim_z,
            index,
        );
    }
    if name.contains("op_clamp_kernel") {
        return sifive_op_clamp_binding_metadata(kernel_name, kernel_params, index);
    }
    if name.contains("cpy_scalar") {
        return sifive_cpy_scalar_binding_metadata(kernel_name, kernel_params, index);
    }
    if name.contains("k_set_rows") && !name.contains("k_set_rows_quant") {
        return sifive_set_rows_binding_metadata(kernel_name, kernel_params, index);
    }
    if name.contains("rope_norm") || name.contains("rope_neox") || name.contains("rope_multi") {
        return sifive_rope_multi_binding_metadata(kernel_name, kernel_params, grid_dim_z, index);
    }
    if name.contains("unary_op_kernel") {
        return sifive_unary_op_binding_metadata(kernel_name, kernel_params, index);
    }
    if name.contains("unary_gated_op_kernel") {
        return sifive_unary_gated_op_binding_metadata(kernel_name, kernel_params, index);
    }
    if name.contains("k_bin_bcast_unravel") {
        return sifive_bin_bcast_binding_metadata(kernel_name, kernel_params, index, true);
    }
    if name.contains("k_bin_bcast") {
        return sifive_bin_bcast_binding_metadata(kernel_name, kernel_params, index, false);
    }
    if name.contains("ssm_conv_f32") || name.contains("ssm_conv_long_token_f32") {
        return sifive_ssm_conv_binding_metadata(
            kernel_name,
            kernel_params,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            index,
        );
    }
    if name.contains("gated_delta_net_cuda") {
        return sifive_gated_delta_net_binding_metadata(kernel_name, kernel_params, index);
    }
    None
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_deepep_layout_binding_metadata(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<(u64, u32)> {
    let num_tokens = read_param_i32(kernel_params, 5)?.max(0) as u64;
    let num_topk = read_param_i32(kernel_params, 6)?.max(0) as u64;
    let num_ranks = read_param_i32(kernel_params, 7)?.max(0) as u64;
    let num_experts = read_param_i32(kernel_params, 8)?.max(0) as u64;

    let i32_bytes = std::mem::size_of::<i32>() as u64;
    let topk_bytes = std::mem::size_of::<i64>() as u64;
    let (bytes, flags) = match index {
        0 => (
            num_tokens
                .saturating_mul(num_topk)
                .saturating_mul(topk_bytes),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        1 => (
            num_ranks.saturating_mul(i32_bytes),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        ),
        2 => {
            let rdma_ranks = if num_ranks > 8 && num_ranks % 8 == 0 {
                num_ranks / 8
            } else {
                1
            };
            (
                rdma_ranks.saturating_mul(i32_bytes),
                sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
            )
        }
        3 => (
            num_experts.saturating_mul(i32_bytes),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        ),
        4 => (
            num_tokens.saturating_mul(num_ranks),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        ),
        _ => return None,
    };

    sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes).map(|clamped| (clamped, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_bin_bcast_fuse_count(kernel_name: &str) -> usize {
    let marker = match kernel_name.find("EvPKT0_") {
        Some(pos) => pos,
        None => return 0,
    };
    let template = &kernel_name[..marker];
    let Some(pack_start) = template.rfind('J') else {
        return 0;
    };
    let pack = template[pack_start + 1..]
        .strip_suffix('E')
        .unwrap_or(&template[pack_start + 1..]);
    sifive_count_mangled_type_pack(pack)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_count_mangled_type_pack(mut pack: &str) -> usize {
    let mut count = 0usize;
    while !pack.is_empty() {
        if let Some(rest) = pack.strip_prefix("PK") {
            pack = rest;
            if let Some(rest) = sifive_skip_mangled_scalar_type(pack) {
                pack = rest;
                count += 1;
                continue;
            }
        } else if let Some(rest) = pack.strip_prefix('P') {
            pack = rest;
            if let Some(rest) = sifive_skip_mangled_scalar_type(pack) {
                pack = rest;
                count += 1;
                continue;
            }
        } else if pack.starts_with('S') {
            if let Some(end) = pack.find('_') {
                pack = &pack[end + 1..];
                count += 1;
                continue;
            }
        }
        break;
    }
    count
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_skip_mangled_scalar_type(pack: &str) -> Option<&str> {
    if let Some(rest) = pack.strip_prefix('f') {
        Some(rest)
    } else if let Some(rest) = pack.strip_prefix("6__half") {
        Some(rest)
    } else if let Some(rest) = pack.strip_prefix("13__nv_bfloat16") {
        Some(rest)
    } else {
        None
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_parse_cpy_scalar_size(
    name: &str,
    offset: &mut usize,
    previous: Option<u64>,
) -> Option<u64> {
    let rest = name.get(*offset..)?;
    if rest.starts_with('f') || rest.starts_with('i') {
        *offset += 1;
        Some(4)
    } else if rest.starts_with("6__half") {
        *offset += "6__half".len();
        Some(2)
    } else if rest.starts_with("13__nv_bfloat16") {
        *offset += "13__nv_bfloat16".len();
        Some(2)
    } else if rest.starts_with("S0_") || rest.starts_with("S1_") || rest.starts_with("S2_") {
        *offset += 3;
        previous
    } else {
        None
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_cpy_scalar_element_sizes(kernel_name: &str) -> Option<(u64, u64)> {
    if let Some(start) = kernel_name.find("cpy_scalar_transposeI") {
        let mut offset = start + "cpy_scalar_transposeI".len();
        let elem = sifive_parse_cpy_scalar_size(kernel_name, &mut offset, None)?;
        return Some((elem, elem));
    }

    let marker = if let Some(start) = kernel_name.find("cpy_scalar_contiguousI") {
        (start, "cpy_scalar_contiguousI")
    } else if let Some(start) = kernel_name.find("cpy_1_scalarI") {
        (start, "cpy_1_scalarI")
    } else {
        return None;
    };

    let mut offset = marker.0 + marker.1.len();
    let src = sifive_parse_cpy_scalar_size(kernel_name, &mut offset, None)?;
    let dst = sifive_parse_cpy_scalar_size(kernel_name, &mut offset, Some(src))?;
    Some((src, dst))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_div_ceil_u64(numer: u64, denom: u64) -> u64 {
    if denom == 0 {
        0
    } else {
        numer.saturating_add(denom.saturating_sub(1)) / denom
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_parse_mangled_scalar_size(name: &str, offset: &mut usize) -> Option<u64> {
    let rest = name.get(*offset..)?;
    if rest.starts_with('f') {
        *offset += 1;
        Some(std::mem::size_of::<f32>() as u64)
    } else if rest.starts_with("6__half") {
        *offset += "6__half".len();
        Some(2)
    } else {
        None
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_parse_convert_unary_element_sizes(kernel_name: &str) -> Option<(u64, u64)> {
    let marker = "convert_unaryI";
    let mut offset = kernel_name.find(marker)? + marker.len();
    let src = sifive_parse_mangled_scalar_size(kernel_name, &mut offset)?;
    let dst = sifive_parse_mangled_scalar_size(kernel_name, &mut offset)?;
    Some((src, dst))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_parse_op_clamp_element_size(kernel_name: &str) -> Option<u64> {
    let marker = "op_clamp_kernelI";
    let mut offset = kernel_name.find(marker)? + marker.len();
    sifive_parse_mangled_scalar_size(kernel_name, &mut offset)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_parse_tagged_number(name: &str, offset: &mut usize, tag: &str) -> Option<u64> {
    let rest = name.get(*offset..)?;
    if !rest.starts_with(tag) {
        return None;
    }
    *offset += tag.len();
    let digits_start = *offset;
    while let Some(byte) = name.as_bytes().get(*offset) {
        if !byte.is_ascii_digit() {
            break;
        }
        *offset += 1;
    }
    if digits_start == *offset || name.as_bytes().get(*offset) != Some(&b'E') {
        return None;
    }
    let value = name.get(digits_start..*offset)?.parse().ok()?;
    *offset += 1;
    Some(value)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_parse_ssm_conv_template(kernel_name: &str) -> Option<(u64, u64, u64)> {
    let (marker, has_split_n_t) = if kernel_name.contains("ssm_conv_long_token_f32I") {
        ("ssm_conv_long_token_f32I", true)
    } else {
        ("ssm_conv_f32I", false)
    };
    let mut offset = kernel_name.find(marker)? + marker.len();
    sifive_parse_tagged_number(kernel_name, &mut offset, "Lb")?;
    let split_d_inner = sifive_parse_tagged_number(kernel_name, &mut offset, "Lm")?;
    let d_conv = sifive_parse_tagged_number(kernel_name, &mut offset, "Lm")?;
    let split_n_t = if has_split_n_t {
        sifive_parse_tagged_number(kernel_name, &mut offset, "Ll")?
    } else {
        0
    };
    Some((split_d_inner, d_conv, split_n_t))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_parse_gated_delta_net_template(kernel_name: &str) -> Option<(u64, bool)> {
    let marker = "gated_delta_net_cudaI";
    let mut offset = kernel_name.find(marker)? + marker.len();
    let s_v = sifive_parse_tagged_number(kernel_name, &mut offset, "Li")?;
    let kda = sifive_parse_tagged_number(kernel_name, &mut offset, "Lb")? != 0;
    Some((s_v, kda))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_bin_bcast_element_sizes(kernel_name: &str) -> (u64, u64, u64) {
    let default = std::mem::size_of::<f32>() as u64;
    let Some(type_start) = kernel_name.find("EE").map(|pos| pos + 2) else {
        return (default, default, default);
    };
    let mut offset = type_start;
    let src0 = sifive_parse_mangled_scalar_size(kernel_name, &mut offset).unwrap_or(default);
    let src1 = sifive_parse_mangled_scalar_size(kernel_name, &mut offset).unwrap_or(default);
    let dst = sifive_parse_mangled_scalar_size(kernel_name, &mut offset).unwrap_or(default);
    (src0, src1, dst)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_parse_set_rows_scalar_size(name: &str, offset: &mut usize) -> Option<u64> {
    let rest = name.get(*offset..)?;
    if rest.starts_with('f') {
        *offset += 1;
        Some(std::mem::size_of::<f32>() as u64)
    } else if rest.starts_with('i') {
        *offset += 1;
        Some(std::mem::size_of::<i32>() as u64)
    } else if rest.starts_with('l') {
        *offset += 1;
        Some(std::mem::size_of::<i64>() as u64)
    } else if rest.starts_with("6__half") {
        *offset += "6__half".len();
        Some(2)
    } else if rest.starts_with("13__nv_bfloat16") {
        *offset += "13__nv_bfloat16".len();
        Some(2)
    } else {
        None
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_set_rows_element_sizes(kernel_name: &str) -> Option<(u64, u64, u64)> {
    let marker = "k_set_rowsI";
    let mut offset = kernel_name.find(marker)? + marker.len();
    let src = sifive_parse_set_rows_scalar_size(kernel_name, &mut offset)?;
    let idx = sifive_parse_set_rows_scalar_size(kernel_name, &mut offset)?;
    let dst = sifive_parse_set_rows_scalar_size(kernel_name, &mut offset)?;
    Some((src, idx, dst))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_strided_extent_bytes(dims: [u64; 4], strides: [u64; 4], elem_size: u64) -> u64 {
    if elem_size == 0 || dims.iter().any(|&dim| dim == 0) {
        return 0;
    }
    let max_elem = dims
        .into_iter()
        .zip(strides)
        .fold(0u64, |acc, (dim, stride)| {
            acc.saturating_add(dim.saturating_sub(1).saturating_mul(stride))
        });
    max_elem.saturating_add(1).saturating_mul(elem_size)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_strided_extent_bytes_from_byte_strides(
    dims: [u64; 4],
    strides: [u64; 4],
    elem_size: u64,
) -> u64 {
    if elem_size == 0 || dims.iter().any(|&dim| dim == 0) {
        return 0;
    }
    let max_byte_offset = dims
        .into_iter()
        .zip(strides)
        .fold(0u64, |acc, (dim, stride)| {
            acc.saturating_add(dim.saturating_sub(1).saturating_mul(stride))
        });
    max_byte_offset.saturating_add(elem_size)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_binding_bytes_for_host_ptr(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
    bytes: u64,
) -> Option<u64> {
    let ptr = read_param_u64(kernel_params, index)?;
    if let Some(remaining) = super::memory::sifive_allocation_remaining_addr(ptr) {
        let remaining = remaining as u64;
        if remaining == 0 {
            None
        } else {
            Some(bytes.min(remaining))
        }
    } else if bytes <= usize::MAX as u64
        && sifive_host_range_has_perms(ptr as usize, bytes as usize, false)
    {
        Some(bytes)
    } else {
        None
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_max_nonnegative_i32_from_host_ptr(ptr: u64, elem_count: u64) -> Option<u64> {
    let remaining = super::memory::sifive_allocation_remaining_addr(ptr)
        .map(|remaining| remaining as u64)
        .or_else(|| {
            elem_count
                .checked_mul(std::mem::size_of::<i32>() as u64)
                .filter(|&bytes| {
                    bytes <= usize::MAX as u64
                        && sifive_host_range_has_perms(ptr as usize, bytes as usize, false)
                })
        })?;
    let count = elem_count
        .min(remaining / std::mem::size_of::<i32>() as u64)
        .min(usize::MAX as u64) as usize;
    if count == 0 {
        return None;
    }

    let values = unsafe { std::slice::from_raw_parts(ptr as *const i32, count) };
    values
        .iter()
        .copied()
        .filter(|&value| value >= 0)
        .map(|value| value as u64)
        .max()
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_scale_f32_binding_metadata(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<(u64, u32)> {
    let nelements = read_param_i64(kernel_params, 4)?.max(0) as u64;
    let bytes = nelements.saturating_mul(std::mem::size_of::<f32>() as u64);
    let flags = match index {
        0 => sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        1 => sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        _ => return None,
    };
    sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes).map(|clamped| (clamped, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_argsort_f32_i32_binding_metadata(
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: u32,
    index: usize,
) -> Option<(u64, u32)> {
    let rows = grid_dim_x.max(1) as u64;
    let ncols = read_param_i32(kernel_params, 2)?.max(0) as u64;
    let ncols_pad = read_param_i32(kernel_params, 3)?.max(0) as u64;
    let bytes = match index {
        0 => rows
            .saturating_mul(ncols)
            .saturating_mul(std::mem::size_of::<f32>() as u64),
        1 => rows
            .saturating_mul(ncols_pad.max(ncols))
            .saturating_mul(std::mem::size_of::<i32>() as u64),
        _ => return None,
    };
    let flags = match index {
        0 => sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        1 => sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        _ => return None,
    };
    sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes).map(|clamped| (clamped, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_op_clamp_binding_metadata(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<(u64, u32)> {
    let elem_size = sifive_parse_op_clamp_element_size(kernel_name)?;
    let nelements = read_param_i32(kernel_params, 4)?.max(0) as u64;
    let bytes = nelements.saturating_mul(elem_size);
    let flags = match index {
        0 => sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        1 => sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        _ => return None,
    };
    sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes).map(|clamped| (clamped, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_l2_norm_f32_binding_metadata(
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    index: usize,
) -> Option<(u64, u32)> {
    let ncols = read_param_i32(kernel_params, 2)?.max(0) as u64;
    let stride_row = read_param_i64(kernel_params, 3)?.max(0) as u64;
    let stride_channel = read_param_i64(kernel_params, 4)?.max(0) as u64;
    let stride_sample = read_param_i64(kernel_params, 5)?.max(0) as u64;
    let nrows = grid_dim_x.max(1) as u64;
    let nchannels = grid_dim_y.max(1) as u64;
    let nsamples = grid_dim_z.max(1) as u64;
    let elem_size = std::mem::size_of::<f32>() as u64;

    let (bytes, flags) = match index {
        0 => {
            let max_elem = nsamples
                .saturating_sub(1)
                .saturating_mul(stride_sample)
                .saturating_add(nchannels.saturating_sub(1).saturating_mul(stride_channel))
                .saturating_add(nrows.saturating_sub(1).saturating_mul(stride_row))
                .saturating_add(ncols.saturating_sub(1));
            (
                max_elem.saturating_add(1).saturating_mul(elem_size),
                sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
            )
        }
        1 => (
            nsamples
                .saturating_mul(nchannels)
                .saturating_mul(nrows)
                .saturating_mul(ncols)
                .saturating_mul(elem_size),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        ),
        _ => return None,
    };
    sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes).map(|clamped| (clamped, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_get_rows_float_binding_metadata(
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: u32,
    index: usize,
) -> Option<(u64, u32)> {
    let ne00 = read_param_i64(kernel_params, 3)?.max(0) as u64;
    let ne11 = read_param_i64(kernel_params, 4)?.max(0) as u64;
    let ne12 = u64::from(read_param_uint3_z(kernel_params, 5)?);
    let s1 = read_param_u64(kernel_params, 6)?;
    let s2 = read_param_u64(kernel_params, 7)?;
    let s3 = read_param_u64(kernel_params, 8)?;
    let nb01 = read_param_u64(kernel_params, 9)?;
    let nb02 = read_param_u64(kernel_params, 10)?;
    let nb03 = read_param_u64(kernel_params, 11)?;
    let s10 = read_param_u64(kernel_params, 12)?;
    let s11 = read_param_u64(kernel_params, 13)?;
    let s12 = read_param_u64(kernel_params, 14)?;
    let ne10 = grid_dim_x.max(1) as u64;
    let elem_size = std::mem::size_of::<f32>() as u64;

    let idx_max_off = ne10
        .saturating_sub(1)
        .saturating_mul(s10)
        .saturating_add(ne11.saturating_sub(1).saturating_mul(s11))
        .saturating_add(ne12.saturating_sub(1).saturating_mul(s12));
    let max_row_index = {
        let ids_ptr = read_param_u64(kernel_params, 1).unwrap_or(0);
        sifive_max_nonnegative_i32_from_host_ptr(ids_ptr, idx_max_off.saturating_add(1))?
    };

    let (bytes, flags) = match index {
        0 => {
            let max_byte = max_row_index
                .saturating_mul(nb01)
                .saturating_add(ne11.saturating_sub(1).saturating_mul(nb02))
                .saturating_add(ne12.saturating_sub(1).saturating_mul(nb03))
                .saturating_add(ne00.saturating_sub(1).saturating_mul(elem_size));
            (
                max_byte.saturating_add(elem_size),
                sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
            )
        }
        1 => (
            idx_max_off
                .saturating_add(1)
                .saturating_mul(std::mem::size_of::<i32>() as u64),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        2 => {
            let max_elem = ne10
                .saturating_sub(1)
                .saturating_mul(s1)
                .saturating_add(ne11.saturating_sub(1).saturating_mul(s2))
                .saturating_add(ne12.saturating_sub(1).saturating_mul(s3))
                .saturating_add(ne00.saturating_sub(1));
            (
                max_elem.saturating_add(1).saturating_mul(elem_size),
                sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
            )
        }
        _ => return None,
    };
    sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes).map(|clamped| (clamped, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_get_rows_q8_0_dst_elem_size(kernel_name: &str) -> u64 {
    if kernel_name.contains("13__nv_bfloat16") || kernel_name.contains("6__half") {
        2
    } else {
        std::mem::size_of::<f32>() as u64
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_get_rows_q8_0_binding_metadata(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: u32,
    index: usize,
) -> Option<(u64, u32)> {
    let ne00 = read_param_i64(kernel_params, 3)?.max(0) as u64;
    let ne11 = read_param_i64(kernel_params, 4)?.max(0) as u64;
    let ne12 = u64::from(read_param_uint3_z(kernel_params, 5)?);
    let s1 = read_param_u64(kernel_params, 6)?;
    let s2 = read_param_u64(kernel_params, 7)?;
    let s3 = read_param_u64(kernel_params, 8)?;
    let nb01 = read_param_u64(kernel_params, 9)?;
    let nb02 = read_param_u64(kernel_params, 10)?;
    let nb03 = read_param_u64(kernel_params, 11)?;
    let s10 = read_param_u64(kernel_params, 12)?;
    let s11 = read_param_u64(kernel_params, 13)?;
    let s12 = read_param_u64(kernel_params, 14)?;
    let ne10 = grid_dim_x.max(1) as u64;
    let dst_elem_size = sifive_get_rows_q8_0_dst_elem_size(kernel_name);
    let q8_0_block_bytes = (std::mem::size_of::<u16>() + 32) as u64;
    let src_row_bytes = sifive_div_ceil_u64(ne00, 32).saturating_mul(q8_0_block_bytes);

    let idx_max_off = ne10
        .saturating_sub(1)
        .saturating_mul(s10)
        .saturating_add(ne11.saturating_sub(1).saturating_mul(s11))
        .saturating_add(ne12.saturating_sub(1).saturating_mul(s12));
    let ids_ptr = read_param_u64(kernel_params, 1).unwrap_or(0);
    let max_row_index =
        sifive_max_nonnegative_i32_from_host_ptr(ids_ptr, idx_max_off.saturating_add(1))?;

    let (bytes, flags) = match index {
        0 => {
            let max_byte = max_row_index
                .saturating_mul(nb01)
                .saturating_add(ne11.saturating_sub(1).saturating_mul(nb02))
                .saturating_add(ne12.saturating_sub(1).saturating_mul(nb03))
                .saturating_add(src_row_bytes.saturating_sub(1));
            (
                max_byte.saturating_add(1),
                sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
            )
        }
        1 => (
            idx_max_off
                .saturating_add(1)
                .saturating_mul(std::mem::size_of::<i32>() as u64),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        2 => {
            let max_elem = ne10
                .saturating_sub(1)
                .saturating_mul(s1)
                .saturating_add(ne11.saturating_sub(1).saturating_mul(s2))
                .saturating_add(ne12.saturating_sub(1).saturating_mul(s3))
                .saturating_add(ne00.saturating_sub(1));
            (
                max_elem.saturating_add(1).saturating_mul(dst_elem_size),
                sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
            )
        }
        _ => return None,
    };
    sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes).map(|clamped| (clamped, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_compute_batched_ptrs_binding_metadata(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<(u64, u32)> {
    let ne12 = read_param_i64(kernel_params, 5)?.max(0) as u64;
    let ne13 = read_param_i64(kernel_params, 6)?.max(0) as u64;
    let ne23 = read_param_i64(kernel_params, 7)?.max(0) as u64;
    let table_count = ne12.saturating_mul(ne13);
    let ptr_size = std::mem::size_of::<u64>() as u64;
    let (bytes, flags) = match index {
        3 => (
            ne23.saturating_add(table_count).saturating_mul(ptr_size),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        ),
        4 => (
            table_count.saturating_mul(ptr_size),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        ),
        _ => return None,
    };
    sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes).map(|clamped| (clamped, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_cpy_scalar_binding_metadata(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<(u64, u32)> {
    if index > 1 {
        return None;
    }

    let (src_elem_size, dst_elem_size) = sifive_cpy_scalar_element_sizes(kernel_name)?;
    let ne = read_param_u64(kernel_params, 2)?;
    let flags = if index == 0 {
        sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT
    } else {
        sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT
    };
    if ne == 0 {
        return Some((0, flags));
    }

    let bytes = if kernel_name
        .to_ascii_lowercase()
        .contains("cpy_scalar_contiguous")
    {
        ne.saturating_mul(if index == 0 {
            src_elem_size
        } else {
            dst_elem_size
        })
    } else if index == 0 {
        let ne0 = read_param_u64(kernel_params, 3)?.max(1);
        let ne1 = read_param_u64(kernel_params, 4)?.max(1);
        let ne2 = read_param_u64(kernel_params, 5)?.max(1);
        let ne012 = ne0.saturating_mul(ne1).saturating_mul(ne2).max(1);
        let dims = [ne0, ne1, ne2, sifive_div_ceil_u64(ne, ne012).max(1)];
        let strides = [
            read_param_u64(kernel_params, 6)?,
            read_param_u64(kernel_params, 7)?,
            read_param_u64(kernel_params, 8)?,
            read_param_u64(kernel_params, 9)?,
        ];
        sifive_strided_extent_bytes_from_byte_strides(dims, strides, src_elem_size)
    } else {
        let ne0 = read_param_u64(kernel_params, 10)?.max(1);
        let ne1 = read_param_u64(kernel_params, 11)?.max(1);
        let ne2 = read_param_u64(kernel_params, 12)?.max(1);
        let ne012 = ne0.saturating_mul(ne1).saturating_mul(ne2).max(1);
        let dims = [ne0, ne1, ne2, sifive_div_ceil_u64(ne, ne012).max(1)];
        let strides = [
            read_param_u64(kernel_params, 13)?,
            read_param_u64(kernel_params, 14)?,
            read_param_u64(kernel_params, 15)?,
            read_param_u64(kernel_params, 16)?,
        ];
        sifive_strided_extent_bytes_from_byte_strides(dims, strides, dst_elem_size)
    };

    sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes).map(|clamped| (clamped, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_rope_element_sizes(kernel_name: &str) -> (u64, u64) {
    if kernel_name.contains("E6__halfS0_") || kernel_name.contains("E13__nv_bfloat16S0_") {
        (2, 2)
    } else if kernel_name.contains("Ef6__half") || kernel_name.contains("Ef13__nv_bfloat16") {
        (4, 2)
    } else if kernel_name.contains("E6__halff") || kernel_name.contains("E13__nv_bfloat16f") {
        (2, 4)
    } else if kernel_name.contains("6__half") || kernel_name.contains("13__nv_bfloat16") {
        (2, 2)
    } else {
        (4, 4)
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_rope_element_size(kernel_name: &str) -> u64 {
    let (src_elem_size, dst_elem_size) = sifive_rope_element_sizes(kernel_name);
    src_elem_size.max(dst_elem_size)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_rope_is_forward(kernel_name: &str) -> bool {
    !kernel_name.contains("rope_normILb0") && !kernel_name.contains("rope_neoxILb0")
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_rope_has_freq_factors(kernel_name: &str, freq_factors: u64) -> bool {
    freq_factors != 0
        || kernel_name.contains("rope_normILb1ELb1")
        || kernel_name.contains("rope_neoxILb1ELb1")
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | if frac == 0 { 0x7c00 } else { 0x7e00 };
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 0x1f {
        sign | 0x7c00
    } else if half_exp <= 0 {
        if half_exp < -10 {
            sign
        } else {
            let mant = frac | 0x80_0000;
            let shift = (14 - half_exp) as u32;
            sign | ((mant >> shift) as u16)
        }
    } else {
        sign | ((half_exp as u16) << 10) | ((frac >> 13) as u16)
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_read_elem_as_f32(base: *const u8, elem_index: i64, elem_size: u64) -> f32 {
    if elem_size == 2 {
        let ptr = base.offset((elem_index * 2) as isize) as *const u16;
        sifive_f16_to_f32(ptr.read_unaligned())
    } else {
        let ptr = base.offset((elem_index * 4) as isize) as *const f32;
        ptr.read_unaligned()
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_write_elem_from_f32(base: *mut u8, elem_index: i64, elem_size: u64, value: f32) {
    if elem_size == 2 {
        let ptr = base.offset((elem_index * 2) as isize) as *mut u16;
        ptr.write_unaligned(sifive_f32_to_f16(value));
    } else {
        let ptr = base.offset((elem_index * 4) as isize) as *mut f32;
        ptr.write_unaligned(value);
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits.wrapping_add(0x7fff).wrapping_add((bits >> 16) & 1)) >> 16) as u16
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[inline]
fn sifive_bf16_to_f32(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[inline]
unsafe fn sifive_read_f16_bf16_or_f32(base: *const u8, elem: i64, x_type: u32) -> f32 {
    if x_type == 2 {
        let xh = base.offset((elem * 2) as isize) as *const u16;
        sifive_f16_to_f32(xh.read_unaligned())
    } else if x_type == 3 {
        let xb = base.offset((elem * 2) as isize) as *const u16;
        sifive_bf16_to_f32(xb.read_unaligned())
    } else {
        (base as *const f32).offset(elem as isize).read_unaligned()
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_write_elem_from_f32_typed(
    base: *mut u8,
    elem_index: i64,
    elem_size: u64,
    value: f32,
    is_bf16: bool,
) {
    if elem_size == 2 && is_bf16 {
        let ptr = base.offset((elem_index * 2) as isize) as *mut u16;
        ptr.write_unaligned(sifive_f32_to_bf16(value));
    } else {
        sifive_write_elem_from_f32(base, elem_index, elem_size, value);
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_set_rows_host_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    use cuda_types::cuda::*;

    if !kernel_name.contains("k_set_rows") || kernel_name.contains("k_set_rows_quant") {
        return None;
    }
    if std::env::var("HETGPU_SIFIVE_SET_ROWS_HOST_FALLBACK")
        .ok()
        .as_deref()
        == Some("0")
    {
        return Some(Err(CUerror::UNKNOWN));
    }

    let (src_elem, idx_elem, dst_elem) = sifive_set_rows_element_sizes(kernel_name)?;
    if !matches!(src_elem, 2 | 4) || !matches!(idx_elem, 4 | 8) || !matches!(dst_elem, 2 | 4) {
        return None;
    }

    let src0 = read_param_u64(kernel_params, 0)?;
    let src1 = read_param_u64(kernel_params, 1)?;
    let dst = read_param_u64(kernel_params, 2)?;
    let ne_total_i = read_param_i64(kernel_params, 3)?;
    if ne_total_i <= 0 {
        return Some(Ok(()));
    }
    let ne_total = ne_total_i as u64;
    let ne10 = read_param_i64(kernel_params, 4)?.max(0) as u64;
    let _ne11 = read_param_i64(kernel_params, 5)?.max(0) as u64;
    let _ne12 = read_param_i64(kernel_params, 6)?.max(0) as u64;
    let s01 = read_param_i64(kernel_params, 8)?;
    let s02 = read_param_i64(kernel_params, 9)?;
    let s03 = read_param_i64(kernel_params, 10)?;
    let s10 = read_param_i64(kernel_params, 11)?;
    let s11 = read_param_i64(kernel_params, 12)?;
    let s12 = read_param_i64(kernel_params, 13)?;
    let s1 = read_param_i64(kernel_params, 14)?;
    let s2 = read_param_i64(kernel_params, 15)?;
    let s3 = read_param_i64(kernel_params, 16)?;
    let ne00 = read_param_uint3_z(kernel_params, 17)?.max(1) as u64;
    let ne01 = read_param_uint3_z(kernel_params, 18)?.max(1) as u64;
    let ne02 = read_param_uint3_z(kernel_params, 19)?.max(1) as u64;
    let ne11_fd = read_param_uint3_z(kernel_params, 20)?.max(1) as u64;
    let ne12_fd = read_param_uint3_z(kernel_params, 21)?.max(1) as u64;

    if [s01, s02, s03, s10, s11, s12, s1, s2, s3]
        .iter()
        .any(|&stride| stride < 0)
    {
        eprintln!(
            "[SIFIVE Backend] host-fallback k_set_rows '{}' rejected invalid shape/stride",
            kernel_name
        );
        return Some(Err(CUerror::UNKNOWN));
    }

    let ne012 = ne00.saturating_mul(ne01).saturating_mul(ne02).max(1);
    let ne03 = sifive_div_ceil_u64(ne_total, ne012);
    let src0_bytes = sifive_strided_extent_bytes(
        [ne00, ne01, ne02, ne03],
        [1, s01 as u64, s02 as u64, s03 as u64],
        src_elem,
    );
    let src1_bytes = sifive_strided_extent_bytes(
        [ne01, ne11_fd, ne12_fd, 1],
        [s10 as u64, s11 as u64, s12 as u64, 0],
        idx_elem,
    );
    let Some(src0_len) = usize::try_from(src0_bytes).ok() else {
        return Some(Err(CUerror::UNKNOWN));
    };
    let Some(src1_len) = usize::try_from(src1_bytes).ok() else {
        return Some(Err(CUerror::UNKNOWN));
    };
    if !sifive_host_or_cuda_alloc_has_bytes(src0, src0_len, false)
        || !sifive_host_or_cuda_alloc_has_bytes(src1, src1_len, false)
    {
        eprintln!(
            "[SIFIVE Backend] host-fallback k_set_rows '{}' rejected input ranges src0=0x{:x}/{} src1=0x{:x}/{}",
            kernel_name, src0, src0_bytes, src1, src1_bytes
        );
        return Some(Err(CUerror::UNKNOWN));
    }

    let src0_base = src0 as *const u8;
    let src1_base = src1 as *const u8;
    let dst_base = dst as *mut u8;
    let dst_is_bf16 = kernel_name.contains("13__nv_bfloat16");
    let mut max_dst_index = 0u64;

    for linear in 0..ne_total {
        let mut tmp = (linear as u32) as u64;
        let i00 = tmp % ne00;
        tmp /= ne00;
        let i01 = tmp % ne01;
        tmp /= ne01;
        let i02 = tmp % ne02;
        let i03 = tmp / ne02;
        let i12 = i03 % ne12_fd;
        let i11 = i02 % ne11_fd;
        let i10 = i01;

        let idx_index = i10
            .saturating_mul(s10 as u64)
            .saturating_add(i11.saturating_mul(s11 as u64))
            .saturating_add(i12.saturating_mul(s12 as u64));
        let dst_row = if idx_elem == 8 {
            (src1_base.add((idx_index * 8) as usize) as *const i64).read_unaligned()
        } else {
            (src1_base.add((idx_index * 4) as usize) as *const i32).read_unaligned() as i64
        };
        if dst_row < 0 {
            eprintln!(
                "[SIFIVE Backend] host-fallback k_set_rows '{}' rejected dst_row={}",
                kernel_name, dst_row
            );
            return Some(Err(CUerror::UNKNOWN));
        }
        let dst_index = i00 as i64 + dst_row * s1 + i02 as i64 * s2 + i03 as i64 * s3;
        if dst_index < 0 {
            return Some(Err(CUerror::UNKNOWN));
        }
        max_dst_index = max_dst_index.max(dst_index as u64);
    }

    let dst_bytes = max_dst_index.saturating_add(1).saturating_mul(dst_elem);
    let Some(dst_len) = usize::try_from(dst_bytes).ok() else {
        return Some(Err(CUerror::UNKNOWN));
    };
    if !sifive_host_or_cuda_alloc_has_bytes(dst, dst_len, true) {
        eprintln!(
            "[SIFIVE Backend] host-fallback k_set_rows '{}' rejected dst range dst=0x{:x}/{} max_dst_index={} src1_ne10={}",
            kernel_name, dst, dst_bytes, max_dst_index, ne10
        );
        return Some(Err(CUerror::UNKNOWN));
    }

    for linear in 0..ne_total {
        let mut tmp = (linear as u32) as u64;
        let i00 = tmp % ne00;
        tmp /= ne00;
        let i01 = tmp % ne01;
        tmp /= ne01;
        let i02 = tmp % ne02;
        let i03 = tmp / ne02;
        let i12 = i03 % ne12_fd;
        let i11 = i02 % ne11_fd;
        let i10 = i01;

        let idx_index = i10
            .saturating_mul(s10 as u64)
            .saturating_add(i11.saturating_mul(s11 as u64))
            .saturating_add(i12.saturating_mul(s12 as u64));
        let dst_row = if idx_elem == 8 {
            (src1_base.add((idx_index * 8) as usize) as *const i64).read_unaligned()
        } else {
            (src1_base.add((idx_index * 4) as usize) as *const i32).read_unaligned() as i64
        };
        if dst_row < 0 {
            eprintln!(
                "[SIFIVE Backend] host-fallback k_set_rows '{}' rejected dst_row={}",
                kernel_name, dst_row
            );
            return Some(Err(CUerror::UNKNOWN));
        }

        let src_index = i00 as i64 + i01 as i64 * s01 + i02 as i64 * s02 + i03 as i64 * s03;
        let dst_index = i00 as i64 + dst_row * s1 + i02 as i64 * s2 + i03 as i64 * s3;
        let value = sifive_read_elem_as_f32(src0_base, src_index, src_elem);
        sifive_write_elem_from_f32_typed(dst_base, dst_index, dst_elem, value, dst_is_bf16);
    }

    eprintln!(
        "[SIFIVE Backend] host-fallback k_set_rows '{}' ne_total={} ne00={} ne01={} ne02={} ne03={} idx_elem={} dst_elem={}",
        kernel_name, ne_total, ne00, ne01, ne02, ne03, idx_elem, dst_elem
    );
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_cpy_scalar_host_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    use cuda_types::cuda::*;

    if !kernel_name.contains("cpy_scalar") {
        return None;
    }
    let (src_elem, dst_elem) = sifive_cpy_scalar_element_sizes(kernel_name)?;
    if !matches!(src_elem, 2 | 4) || !matches!(dst_elem, 2 | 4) {
        return None;
    }

    let src = read_param_u64(kernel_params, 0)?;
    let dst = read_param_u64(kernel_params, 1)?;
    let ne = read_param_u64(kernel_params, 2)?;
    if ne == 0 {
        return Some(Ok(()));
    }

    let contiguous = kernel_name
        .to_ascii_lowercase()
        .contains("cpy_scalar_contiguous");
    let (src_len, dst_len) = if contiguous {
        (ne.saturating_mul(src_elem), ne.saturating_mul(dst_elem))
    } else {
        let src_ne0 = read_param_u64(kernel_params, 3)?.max(1);
        let src_ne1 = read_param_u64(kernel_params, 4)?.max(1);
        let src_ne2 = read_param_u64(kernel_params, 5)?.max(1);
        let src_ne012 = src_ne0
            .saturating_mul(src_ne1)
            .saturating_mul(src_ne2)
            .max(1);
        let src_dims = [
            src_ne0,
            src_ne1,
            src_ne2,
            sifive_div_ceil_u64(ne, src_ne012).max(1),
        ];
        let src_strides = [
            read_param_u64(kernel_params, 6)?,
            read_param_u64(kernel_params, 7)?,
            read_param_u64(kernel_params, 8)?,
            read_param_u64(kernel_params, 9)?,
        ];

        let dst_ne0 = read_param_u64(kernel_params, 10)?.max(1);
        let dst_ne1 = read_param_u64(kernel_params, 11)?.max(1);
        let dst_ne2 = read_param_u64(kernel_params, 12)?.max(1);
        let dst_ne012 = dst_ne0
            .saturating_mul(dst_ne1)
            .saturating_mul(dst_ne2)
            .max(1);
        let dst_dims = [
            dst_ne0,
            dst_ne1,
            dst_ne2,
            sifive_div_ceil_u64(ne, dst_ne012).max(1),
        ];
        let dst_strides = [
            read_param_u64(kernel_params, 13)?,
            read_param_u64(kernel_params, 14)?,
            read_param_u64(kernel_params, 15)?,
            read_param_u64(kernel_params, 16)?,
        ];

        (
            sifive_strided_extent_bytes_from_byte_strides(src_dims, src_strides, src_elem),
            sifive_strided_extent_bytes_from_byte_strides(dst_dims, dst_strides, dst_elem),
        )
    };

    let Some(src_len_usize) = usize::try_from(src_len).ok() else {
        return Some(Err(CUerror::UNKNOWN));
    };
    let Some(dst_len_usize) = usize::try_from(dst_len).ok() else {
        return Some(Err(CUerror::UNKNOWN));
    };
    if !sifive_host_or_cuda_alloc_has_bytes(src, src_len_usize, false)
        || !sifive_host_or_cuda_alloc_has_bytes(dst, dst_len_usize, true)
    {
        eprintln!(
            "[SIFIVE Backend] host-fallback cpy_scalar '{}' rejected ranges src=0x{:x}/{} dst=0x{:x}/{}",
            kernel_name, src, src_len, dst, dst_len
        );
        return Some(Err(CUerror::UNKNOWN));
    }

    let src_base = src as *const u8;
    let dst_base = dst as *mut u8;
    let dst_is_bf16 = kernel_name.contains("13__nv_bfloat16");
    if contiguous {
        if src_elem == dst_elem && !dst_is_bf16 {
            std::ptr::copy_nonoverlapping(src_base, dst_base, dst_len_usize);
        } else {
            for i in 0..ne {
                let value = sifive_read_elem_as_f32(src_base, i as i64, src_elem);
                sifive_write_elem_from_f32_typed(dst_base, i as i64, dst_elem, value, dst_is_bf16);
            }
        }
    } else {
        let src_ne0 = read_param_u64(kernel_params, 3)?.max(1);
        let src_ne1 = read_param_u64(kernel_params, 4)?.max(1);
        let src_ne2 = read_param_u64(kernel_params, 5)?.max(1);
        let src_ne012 = src_ne0
            .saturating_mul(src_ne1)
            .saturating_mul(src_ne2)
            .max(1);
        let src_ne03 = sifive_div_ceil_u64(ne, src_ne012).max(1);
        let src_strides = [
            read_param_u64(kernel_params, 6)? / src_elem,
            read_param_u64(kernel_params, 7)? / src_elem,
            read_param_u64(kernel_params, 8)? / src_elem,
            read_param_u64(kernel_params, 9)? / src_elem,
        ];

        let dst_ne0 = read_param_u64(kernel_params, 10)?.max(1);
        let dst_ne1 = read_param_u64(kernel_params, 11)?.max(1);
        let dst_ne2 = read_param_u64(kernel_params, 12)?.max(1);
        let dst_ne012 = dst_ne0
            .saturating_mul(dst_ne1)
            .saturating_mul(dst_ne2)
            .max(1);
        let dst_ne03 = sifive_div_ceil_u64(ne, dst_ne012).max(1);
        let dst_strides = [
            read_param_u64(kernel_params, 13)? / dst_elem,
            read_param_u64(kernel_params, 14)? / dst_elem,
            read_param_u64(kernel_params, 15)? / dst_elem,
            read_param_u64(kernel_params, 16)? / dst_elem,
        ];

        for i in 0..ne {
            let src_i0 = i % src_ne0;
            let src_i1 = (i / src_ne0) % src_ne1;
            let src_i2 = (i / src_ne0 / src_ne1) % src_ne2;
            let src_i3 = (i / src_ne012).min(src_ne03.saturating_sub(1));
            let dst_i0 = i % dst_ne0;
            let dst_i1 = (i / dst_ne0) % dst_ne1;
            let dst_i2 = (i / dst_ne0 / dst_ne1) % dst_ne2;
            let dst_i3 = (i / dst_ne012).min(dst_ne03.saturating_sub(1));
            let src_index = src_i0
                .saturating_mul(src_strides[0])
                .saturating_add(src_i1.saturating_mul(src_strides[1]))
                .saturating_add(src_i2.saturating_mul(src_strides[2]))
                .saturating_add(src_i3.saturating_mul(src_strides[3]));
            let dst_index = dst_i0
                .saturating_mul(dst_strides[0])
                .saturating_add(dst_i1.saturating_mul(dst_strides[1]))
                .saturating_add(dst_i2.saturating_mul(dst_strides[2]))
                .saturating_add(dst_i3.saturating_mul(dst_strides[3]));
            let value = sifive_read_elem_as_f32(src_base, src_index as i64, src_elem);
            sifive_write_elem_from_f32_typed(
                dst_base,
                dst_index as i64,
                dst_elem,
                value,
                dst_is_bf16,
            );
        }
    }

    if sifive_env_truthy("HETGPU_SIFIVE_CPY_SCALAR_TRACE") {
        eprintln!(
            "[SIFIVE Backend] host-fallback cpy_scalar '{}' ne={} src_elem={} dst_elem={} contiguous={}",
            kernel_name, ne, src_elem, dst_elem, contiguous
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_convert_unary_host_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    use cuda_types::cuda::*;

    if !kernel_name.contains("convert_unary") {
        return None;
    }
    let (src_elem, dst_elem) = sifive_parse_convert_unary_element_sizes(kernel_name)?;
    if !matches!(src_elem, 2 | 4) || !matches!(dst_elem, 2 | 4) {
        return None;
    }

    let src = read_param_u64(kernel_params, 0)?;
    let dst = read_param_u64(kernel_params, 1)?;
    let ne00 = read_param_i64(kernel_params, 2)?.max(0) as u64;
    let ne01 = read_param_i64(kernel_params, 3)?.max(0) as u64;
    let ne0203 = read_param_i64(kernel_params, 4)?.max(0) as u64;
    if ne00 == 0 || ne01 == 0 || ne0203 == 0 {
        return Some(Ok(()));
    }
    let ne02 = read_param_uint3_z(kernel_params, 5)?.max(1) as u64;
    let ne03 = sifive_div_ceil_u64(ne0203, ne02).max(1);
    let s01_i = read_param_i64(kernel_params, 6)?;
    let s02_i = read_param_i64(kernel_params, 7)?;
    let s03_i = read_param_i64(kernel_params, 8)?;
    if s01_i < 0 || s02_i < 0 || s03_i < 0 {
        eprintln!(
            "[SIFIVE Backend] host-fallback convert_unary '{}' rejected negative stride s01={} s02={} s03={}",
            kernel_name, s01_i, s02_i, s03_i
        );
        return Some(Err(CUerror::UNKNOWN));
    }
    let s01 = s01_i as u64;
    let s02 = s02_i as u64;
    let s03 = s03_i as u64;

    let src_bytes =
        sifive_strided_extent_bytes([ne00, ne01, ne02, ne03], [1, s01, s02, s03], src_elem);
    let dst_bytes = ne00
        .saturating_mul(ne01)
        .saturating_mul(ne0203)
        .saturating_mul(dst_elem);
    let Some(src_len) = usize::try_from(src_bytes).ok() else {
        return Some(Err(CUerror::UNKNOWN));
    };
    let Some(dst_len) = usize::try_from(dst_bytes).ok() else {
        return Some(Err(CUerror::UNKNOWN));
    };
    if !sifive_host_or_cuda_alloc_has_bytes(src, src_len, false)
        || !sifive_host_or_cuda_alloc_has_bytes(dst, dst_len, true)
    {
        eprintln!(
            "[SIFIVE Backend] host-fallback convert_unary '{}' rejected ranges src=0x{:x}/{} dst=0x{:x}/{}",
            kernel_name, src, src_bytes, dst, dst_bytes
        );
        return Some(Err(CUerror::UNKNOWN));
    }

    let src_base = src as *const u8;
    let dst_base = dst as *mut u8;
    let dst_is_bf16 = kernel_name.contains("13__nv_bfloat16");
    for i0203 in 0..ne0203 {
        let i02 = i0203 % ne02;
        let i03 = i0203 / ne02;
        for i01 in 0..ne01 {
            for i00 in 0..ne00 {
                let src_index = i03
                    .saturating_mul(s03)
                    .saturating_add(i02.saturating_mul(s02))
                    .saturating_add(i01.saturating_mul(s01))
                    .saturating_add(i00);
                let dst_index = i0203
                    .saturating_mul(ne01)
                    .saturating_add(i01)
                    .saturating_mul(ne00)
                    .saturating_add(i00);
                let value = sifive_read_elem_as_f32(src_base, src_index as i64, src_elem);
                sifive_write_elem_from_f32_typed(
                    dst_base,
                    dst_index as i64,
                    dst_elem,
                    value,
                    dst_is_bf16,
                );
            }
        }
    }

    eprintln!(
        "[SIFIVE Backend] host-fallback convert_unary '{}' ne00={} ne01={} ne0203={} src_elem={} dst_elem={}",
        kernel_name, ne00, ne01, ne0203, src_elem, dst_elem
    );
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_parse_mmvq_type(kernel_name: &str) -> Option<u32> {
    let marker = "L9ggml_type";
    let start = kernel_name.find(marker)? + marker.len();
    let digits = kernel_name[start..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .count();
    if digits == 0 {
        return None;
    }
    kernel_name[start..start + digits].parse::<u32>().ok()
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_parse_mmvq_ncols_dst(kernel_name: &str) -> Option<u32> {
    let type_marker = "L9ggml_type";
    let type_pos = kernel_name.find(type_marker)?;
    let after_type = &kernel_name[type_pos + type_marker.len()..];
    let pos = after_type.find("ELi")? + type_pos + type_marker.len() + 3;
    let digits = kernel_name[pos..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .count();
    if digits == 0 {
        return None;
    }
    kernel_name[pos..pos + digits].parse::<u32>().ok()
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_parse_mmvq_small_k(kernel_name: &str) -> bool {
    let type_marker = "L9ggml_type";
    let Some(type_pos) = kernel_name.find(type_marker) else {
        return false;
    };
    let after_type = &kernel_name[type_pos + type_marker.len()..];
    let Some(ncols_marker_pos) = after_type.find("ELi") else {
        return false;
    };
    let mut pos = type_pos + type_marker.len() + ncols_marker_pos + 3;
    while matches!(kernel_name.as_bytes().get(pos), Some(b'0'..=b'9')) {
        pos += 1;
    }

    let Some(after_fusion) = kernel_name.get(pos..).and_then(|s| s.strip_prefix("ELb")) else {
        return false;
    };
    let Some(after_fusion_value) = after_fusion.get(1..) else {
        return false;
    };
    let Some(after_small_k) = after_fusion_value.strip_prefix("ELb") else {
        return false;
    };
    after_small_k.starts_with('1')
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_mmvq_rows_per_block(kernel_name: &str, ncols_dst: u64) -> u64 {
    match ncols_dst {
        1 if sifive_parse_mmvq_small_k(kernel_name) => 4,
        1 => 1,
        2..=8 => 2,
        _ => 1,
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_parse_topk_moe_experts(kernel_name: &str) -> Option<u64> {
    let marker = "topk_moe_cudaILi";
    let start = kernel_name.find(marker)? + marker.len();
    let digits = kernel_name[start..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .count();
    if digits == 0 {
        return None;
    }
    kernel_name[start..start + digits].parse::<u64>().ok()
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_topk_moe_has_bias(kernel_name: &str) -> bool {
    kernel_name.contains("ELb1EEvPKfPfPi")
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_mmvq_type_layout(ggml_type: u32) -> Option<(u64, u64)> {
    match ggml_type {
        // GGML_TYPE_Q8_0: QK8_0 elements per block, one fp16 scale and 32 i8 quants.
        8 => Some((32, 34)),
        _ => None,
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_mul_mat_vec_q_binding_metadata(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    index: usize,
) -> Option<(u64, u32)> {
    let (qk, x_block_bytes) = sifive_mmvq_type_layout(sifive_parse_mmvq_type(kernel_name)?)?;
    let ncols_dst = sifive_parse_mmvq_ncols_dst(kernel_name)?.max(1) as u64;
    let rows_per_block = sifive_mmvq_rows_per_block(kernel_name, ncols_dst);
    let ncols_x = read_param_u32(kernel_params, 5)? as u64;
    let nchannels_y = read_param_uint3_z(kernel_params, 6)?.max(1) as u64;
    let stride_row_x = read_param_u32(kernel_params, 7)? as u64;
    let stride_col_y = read_param_u32(kernel_params, 8)? as u64;
    let stride_col_dst = read_param_u32(kernel_params, 9)? as u64;
    let channel_ratio = read_param_uint3_z(kernel_params, 10)?.max(1) as u64;
    let stride_channel_x = read_param_u32(kernel_params, 11)? as u64;
    let stride_channel_y = read_param_u32(kernel_params, 12)? as u64;
    let stride_channel_dst = read_param_u32(kernel_params, 13)? as u64;
    let sample_ratio = read_param_uint3_z(kernel_params, 14)?.max(1) as u64;
    let stride_sample_x = read_param_u32(kernel_params, 15)? as u64;
    let stride_sample_y = read_param_u32(kernel_params, 16)? as u64;
    let stride_sample_dst = read_param_u32(kernel_params, 17)? as u64;
    let ids_stride = read_param_u32(kernel_params, 18)? as u64;

    let grid_x = grid_dim_x.max(1) as u64;
    let grid_y = grid_dim_y.max(1) as u64;
    let grid_z = grid_dim_z.max(1) as u64;
    let rows_x = grid_x.saturating_mul(rows_per_block);
    let blocks_per_row_x = ncols_x.saturating_add(qk - 1) / qk;
    let q8_1_block_bytes = 36u64;
    let ids_ptr = read_param_u64(kernel_params, 2).unwrap_or(0);
    let has_ids =
        ids_ptr != 0 && super::memory::sifive_allocation_remaining_addr(ids_ptr).is_some();

    let (bytes, flags) = match index {
        0 => {
            let max_sample_x = grid_z.saturating_sub(1) / sample_ratio;
            let max_channel_x = if has_ids && ncols_dst == 1 {
                sifive_max_nonnegative_i32_from_host_ptr(ids_ptr, grid_y)
                    .unwrap_or_else(|| grid_y.saturating_sub(1))
            } else {
                grid_y.saturating_sub(1) / channel_ratio
            };
            let max_row = rows_x.saturating_sub(1);
            let max_block = blocks_per_row_x.saturating_sub(1);
            let max_off = max_sample_x
                .saturating_mul(stride_sample_x)
                .saturating_add(max_channel_x.saturating_mul(stride_channel_x))
                .saturating_add(max_row.saturating_mul(stride_row_x))
                .saturating_add(max_block);
            (
                max_off.saturating_add(1).saturating_mul(x_block_bytes),
                sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
            )
        }
        1 => {
            let max_sample_y = grid_z.saturating_sub(1);
            let max_channel_y = if has_ids && ncols_dst == 1 {
                grid_y.min(nchannels_y).saturating_sub(1)
            } else {
                grid_y.saturating_sub(1)
            };
            let max_col = ncols_dst.saturating_sub(1);
            let max_block = blocks_per_row_x.saturating_sub(1);
            let kby_mul = qk / 32;
            let max_kby = max_block.saturating_mul(kby_mul.max(1));
            let max_off = max_sample_y
                .saturating_mul(stride_sample_y)
                .saturating_add(max_channel_y.saturating_mul(stride_channel_y))
                .saturating_add(max_col.saturating_mul(stride_col_y))
                .saturating_add(max_kby);
            (
                max_off.saturating_add(1).saturating_mul(q8_1_block_bytes),
                sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
            )
        }
        2 if has_ids => {
            let max_off = if ncols_dst == 1 {
                grid_y.saturating_sub(1)
            } else {
                grid_y
                    .saturating_sub(1)
                    .saturating_add(grid_z.saturating_sub(1).saturating_mul(ids_stride))
            };
            (
                max_off.saturating_add(1).saturating_mul(4),
                sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
            )
        }
        4 => {
            let max_row = rows_x.saturating_sub(1);
            let max_col = ncols_dst.saturating_sub(1);
            let max_off = grid_z
                .saturating_sub(1)
                .saturating_mul(stride_sample_dst)
                .saturating_add(grid_y.saturating_sub(1).saturating_mul(stride_channel_dst))
                .saturating_add(max_col.saturating_mul(stride_col_dst))
                .saturating_add(max_row);
            (
                max_off.saturating_add(1).saturating_mul(4),
                sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
            )
        }
        _ => return None,
    };
    let bytes = sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes)?;
    Some((bytes, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_set_rows_binding_metadata(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<(u64, u32)> {
    let (src_elem, idx_elem, dst_elem) = sifive_set_rows_element_sizes(kernel_name)?;
    let ne_total = read_param_i64(kernel_params, 3)?.max(0) as u64;
    let ne10 = read_param_i64(kernel_params, 4)?.max(0) as u64;
    let ne11 = read_param_i64(kernel_params, 5)?.max(0) as u64;
    let ne12 = read_param_i64(kernel_params, 6)?.max(0) as u64;
    let s01 = read_param_i64(kernel_params, 8)?.max(0) as u64;
    let s02 = read_param_i64(kernel_params, 9)?.max(0) as u64;
    let s03 = read_param_i64(kernel_params, 10)?.max(0) as u64;
    let s10 = read_param_i64(kernel_params, 11)?.max(0) as u64;
    let s11 = read_param_i64(kernel_params, 12)?.max(0) as u64;
    let s12 = read_param_i64(kernel_params, 13)?.max(0) as u64;
    let s1 = read_param_i64(kernel_params, 14)?.max(0) as u64;
    let s2 = read_param_i64(kernel_params, 15)?.max(0) as u64;
    let s3 = read_param_i64(kernel_params, 16)?.max(0) as u64;
    let ne00 = read_param_uint3_z(kernel_params, 17)?.max(1) as u64;
    let ne01 = read_param_uint3_z(kernel_params, 18)?.max(1) as u64;
    let ne02 = read_param_uint3_z(kernel_params, 19)?.max(1) as u64;
    let ne11_fd = read_param_uint3_z(kernel_params, 20)?.max(1) as u64;
    let ne12_fd = read_param_uint3_z(kernel_params, 21)?.max(1) as u64;
    let ne012 = ne00.saturating_mul(ne01).saturating_mul(ne02).max(1);
    let ne03 = sifive_div_ceil_u64(ne_total, ne012);

    let src0_bytes =
        sifive_strided_extent_bytes([ne00, ne01, ne02, ne03], [1, s01, s02, s03], src_elem);
    let src1_bytes =
        sifive_strided_extent_bytes([ne01, ne11_fd, ne12_fd, 1], [s10, s11, s12, 0], idx_elem);
    let dst_bytes = sifive_strided_extent_bytes(
        [ne00, ne10, ne11.max(1), ne12.max(1)],
        [1, s1, s2, s3],
        dst_elem,
    );

    let (bytes, flags) = match index {
        0 => (
            src0_bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        1 => (
            src1_bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        2 => (
            dst_bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        ),
        _ => return None,
    };
    let bytes = sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes)?;
    Some((bytes, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_topk_moe_binding_metadata(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<(u64, u32)> {
    let n_experts = sifive_parse_topk_moe_experts(kernel_name)?.max(1);
    let n_rows = read_param_i32(kernel_params, 4)?.max(0) as u64;
    let n_expert_used = read_param_i32(kernel_params, 5)?.max(0) as u64;

    let (bytes, flags) = match index {
        0 => (
            n_rows
                .saturating_mul(n_experts)
                .saturating_mul(std::mem::size_of::<f32>() as u64),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        1 => (
            n_rows
                .saturating_mul(n_expert_used)
                .saturating_mul(std::mem::size_of::<f32>() as u64),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        ),
        2 => (
            n_rows
                .saturating_mul(n_experts)
                .saturating_mul(std::mem::size_of::<i32>() as u64),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        ),
        3 if sifive_topk_moe_has_bias(kernel_name) => (
            n_experts.saturating_mul(std::mem::size_of::<f32>() as u64),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        _ => return None,
    };

    let bytes = sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes)?;
    Some((bytes, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_mul_mat_vec_q_moe_binding_metadata(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_y: u32,
    index: usize,
) -> Option<(u64, u32)> {
    let (qk, x_block_bytes) = sifive_mmvq_type_layout(sifive_parse_mmvq_type(kernel_name)?)?;
    let ncols_x = read_param_u32(kernel_params, 4)? as u64;
    let nchannels_y = read_param_uint3_z(kernel_params, 5)?.max(1) as u64;
    let nrows_x = read_param_u32(kernel_params, 6)?.max(1) as u64;
    let stride_row_x = read_param_u32(kernel_params, 7)? as u64;
    let stride_col_y = read_param_u32(kernel_params, 8)? as u64;
    let stride_col_dst = read_param_u32(kernel_params, 9)? as u64;
    let stride_channel_x = read_param_u32(kernel_params, 10)? as u64;
    let stride_channel_y = read_param_u32(kernel_params, 11)? as u64;
    let stride_channel_dst = read_param_u32(kernel_params, 12)? as u64;
    let ncols_dst = read_param_u32(kernel_params, 13)?.max(1) as u64;
    let ids_stride = read_param_u32(kernel_params, 14)? as u64;

    let grid_y = (grid_dim_y as u64).max(1);
    let blocks_per_row_x = ncols_x.saturating_add(qk - 1) / qk;
    let q8_1_block_bytes = 36u64;
    let ids_ptr = read_param_u64(kernel_params, 2).unwrap_or(0);
    let has_ids =
        ids_ptr != 0 && super::memory::sifive_allocation_remaining_addr(ids_ptr).is_some();
    let ids_max_off = grid_y
        .saturating_sub(1)
        .saturating_add(ncols_dst.saturating_sub(1).saturating_mul(ids_stride));

    let (bytes, flags) = match index {
        0 => {
            let max_channel_x = if has_ids {
                sifive_max_nonnegative_i32_from_host_ptr(ids_ptr, ids_max_off.saturating_add(1))
                    .unwrap_or_else(|| nchannels_y.saturating_sub(1))
            } else {
                nchannels_y.saturating_sub(1)
            };
            let max_off = max_channel_x
                .saturating_mul(stride_channel_x)
                .saturating_add(nrows_x.saturating_sub(1).saturating_mul(stride_row_x))
                .saturating_add(blocks_per_row_x.saturating_sub(1));
            (
                max_off.saturating_add(1).saturating_mul(x_block_bytes),
                sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
            )
        }
        1 => {
            let max_off = grid_y
                .min(nchannels_y)
                .saturating_sub(1)
                .saturating_mul(stride_channel_y)
                .saturating_add(ncols_dst.saturating_sub(1).saturating_mul(stride_col_y))
                .saturating_add(blocks_per_row_x.saturating_sub(1));
            (
                max_off.saturating_add(1).saturating_mul(q8_1_block_bytes),
                sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
            )
        }
        2 => (
            ids_max_off
                .saturating_add(1)
                .saturating_mul(std::mem::size_of::<i32>() as u64),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        3 => {
            let max_off = grid_y
                .saturating_sub(1)
                .saturating_mul(stride_channel_dst)
                .saturating_add(ncols_dst.saturating_sub(1).saturating_mul(stride_col_dst))
                .saturating_add(nrows_x.saturating_sub(1));
            (
                max_off
                    .saturating_add(1)
                    .saturating_mul(std::mem::size_of::<f32>() as u64),
                sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
            )
        }
        _ => return None,
    };

    let bytes = sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes)?;
    Some((bytes, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_parse_mmvf_template(kernel_name: &str) -> Option<(u64, bool, bool)> {
    let marker = "mul_mat_vec_f";
    let after = &kernel_name[kernel_name.find(marker)? + marker.len()..];
    let li = after.find("Li")? + 2;
    let digits = after[li..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .count();
    if digits == 0 {
        return None;
    }
    let ncols_dst = after[li..li + digits].parse::<u64>().ok()?.max(1);

    let after_ncols = &after[li + digits..];
    let first_bool = after_ncols.find("ELb")? + 3;
    let has_fusion = after_ncols.as_bytes().get(first_bool).copied() == Some(b'1');
    let after_first_bool = &after_ncols[first_bool + 1..];
    let second_bool = after_first_bool.find("ELb")? + 3;
    let is_multi_token_id = after_first_bool.as_bytes().get(second_bool).copied() == Some(b'1');

    Some((ncols_dst, has_fusion, is_multi_token_id))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_mmvf_x_elem_bytes(kernel_name: &str) -> u64 {
    if kernel_name.contains("mul_mat_vec_fI6__half")
        || kernel_name.contains("mul_mat_vec_fI11nv_bfloat16")
        || kernel_name.contains("mul_mat_vec_fI12__nv_bfloat16")
        || kernel_name.contains("mul_mat_vec_fI13__nv_bfloat16")
    {
        2
    } else {
        4
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_mmvf_x_type(kernel_name: &str) -> Option<u32> {
    if kernel_name.contains("mul_mat_vec_fI6__half") {
        Some(2)
    } else if kernel_name.contains("mul_mat_vec_fI11nv_bfloat16")
        || kernel_name.contains("mul_mat_vec_fI12__nv_bfloat16")
        || kernel_name.contains("mul_mat_vec_fI13__nv_bfloat16")
        || kernel_name.contains("__nv_bfloat16")
        || kernel_name.contains("nv_bfloat16")
    {
        Some(3)
    } else if kernel_name.contains("mul_mat_vec_fIff") {
        Some(1)
    } else {
        None
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[repr(C)]
#[derive(Copy, Clone, Default)]
struct SifiveMmvFusionArgs {
    x_bias: u64,
    gate: u64,
    gate_bias: u64,
    glu_op: i32,
    _pad: u32,
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn read_mmvf_fusion_args(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<SifiveMmvFusionArgs> {
    if kernel_params.is_null() {
        return None;
    }
    let param = *kernel_params.add(index);
    if param.is_null()
        || (param as usize) < 0x1_0000
        || !sifive_host_range_has_perms(
            param as usize,
            std::mem::size_of::<SifiveMmvFusionArgs>(),
            false,
        )
    {
        return None;
    }
    Some((param as *const SifiveMmvFusionArgs).read_unaligned())
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_gelu(value: f32) -> f32 {
    const GELU_COEF_A: f32 = 0.044715;
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    0.5 * value * (1.0 + (SQRT_2_OVER_PI * value * (1.0 + GELU_COEF_A * value * value)).tanh())
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_swiglu_oai(x: f32, gate: f32) -> f32 {
    let x = x.min(7.0);
    let gate = gate.clamp(-7.0, 7.0);
    x * gate / (1.0 + (-1.702 * gate).exp())
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn read_param_uint3_value(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<sifive_runtime_sys::HetgpuSifiveUint3> {
    if kernel_params.is_null() {
        return None;
    }
    let param = *kernel_params.add(index);
    if param.is_null() || (param as usize) < 0x1_0000 {
        return None;
    }
    let p = param as *const u32;
    Some(sifive_runtime_sys::HetgpuSifiveUint3 {
        x: p.read_unaligned(),
        y: p.add(1).read_unaligned(),
        z: p.add(2).read_unaligned(),
    })
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_mul_mat_vec_f_binding_metadata(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    index: usize,
) -> Option<(u64, u32)> {
    let (ncols_dst, _has_fusion, is_multi_token_id) = sifive_parse_mmvf_template(kernel_name)?;
    let x_elem_bytes = sifive_mmvf_x_elem_bytes(kernel_name);
    let ncols2 = read_param_u32(kernel_params, 5)? as u64;
    let nchannels_y = read_param_uint3_z(kernel_params, 6)?.max(1) as u64;
    let stride_row = read_param_u32(kernel_params, 7)? as u64;
    let stride_col_y2 = read_param_u32(kernel_params, 8)? as u64;
    let stride_col_dst = read_param_u32(kernel_params, 9)? as u64;
    let channel_ratio = read_param_uint3_z(kernel_params, 10)?.max(1) as u64;
    let stride_channel_x = read_param_u32(kernel_params, 11)? as u64;
    let stride_channel_y = read_param_u32(kernel_params, 12)? as u64;
    let stride_channel_dst = read_param_u32(kernel_params, 13)? as u64;
    let sample_ratio = read_param_uint3_z(kernel_params, 14)?.max(1) as u64;
    let stride_sample_x = read_param_u32(kernel_params, 15)? as u64;
    let stride_sample_y = read_param_u32(kernel_params, 16)? as u64;
    let stride_sample_dst = read_param_u32(kernel_params, 17)? as u64;
    let ids_stride = read_param_u32(kernel_params, 18)? as u64;

    let grid_x = grid_dim_x.max(1) as u64;
    let grid_y = grid_dim_y.max(1) as u64;
    let grid_z = grid_dim_z.max(1) as u64;
    let ids_ptr = read_param_u64(kernel_params, 2).unwrap_or(0);
    let has_ids = ids_ptr != 0 && sifive_host_or_cuda_alloc_has_bytes(ids_ptr, 4, false);
    let col_elems = ncols2.saturating_mul(2);

    let (bytes, flags) = match index {
        0 => {
            let max_sample_x = if has_ids {
                0
            } else {
                grid_z.saturating_sub(1) / sample_ratio
            };
            let max_channel_x = if has_ids {
                let max_off = grid_y
                    .saturating_sub(1)
                    .saturating_add(grid_z.saturating_sub(1).saturating_mul(ids_stride));
                sifive_max_nonnegative_i32_from_host_ptr(ids_ptr, max_off.saturating_add(1))
                    .unwrap_or_else(|| grid_y.saturating_sub(1))
            } else {
                grid_y.saturating_sub(1) / channel_ratio
            };
            let max_off = max_sample_x
                .saturating_mul(stride_sample_x)
                .saturating_add(max_channel_x.saturating_mul(stride_channel_x))
                .saturating_add(grid_x.saturating_sub(1).saturating_mul(stride_row))
                .saturating_add(col_elems.saturating_sub(1));
            (
                max_off.saturating_add(1).saturating_mul(x_elem_bytes),
                sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
            )
        }
        1 => {
            let max_sample_y = if has_ids { 0 } else { grid_z.saturating_sub(1) };
            let max_channel_y = if has_ids {
                nchannels_y.saturating_sub(1)
            } else {
                grid_y.saturating_sub(1)
            };
            let max_col_or_token = if has_ids && is_multi_token_id {
                grid_z.saturating_sub(1)
            } else {
                ncols_dst.saturating_sub(1)
            };
            let max_off = max_sample_y
                .saturating_mul(stride_sample_y)
                .saturating_add(max_channel_y.saturating_mul(stride_channel_y))
                .saturating_add(
                    max_col_or_token
                        .saturating_mul(stride_col_y2)
                        .saturating_mul(2),
                )
                .saturating_add(col_elems.saturating_sub(1));
            (
                max_off.saturating_add(1).saturating_mul(4),
                sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
            )
        }
        2 if has_ids => {
            let max_off = grid_y
                .saturating_sub(1)
                .saturating_add(grid_z.saturating_sub(1).saturating_mul(ids_stride));
            (
                max_off.saturating_add(1).saturating_mul(4),
                sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
            )
        }
        4 => {
            let max_sample_dst = if has_ids { 0 } else { grid_z.saturating_sub(1) };
            let token_col = if is_multi_token_id {
                grid_z.saturating_sub(1)
            } else {
                0
            };
            let max_off = max_sample_dst
                .saturating_mul(stride_sample_dst)
                .saturating_add(grid_y.saturating_sub(1).saturating_mul(stride_channel_dst))
                .saturating_add(
                    token_col
                        .saturating_add(ncols_dst.saturating_sub(1))
                        .saturating_mul(stride_col_dst),
                )
                .saturating_add(grid_x.saturating_sub(1));
            (
                max_off.saturating_add(1).saturating_mul(4),
                sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
            )
        }
        _ => return None,
    };

    let bytes = sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes)?;
    Some((bytes, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_mmvf_host_fallback(
    kernel_name: &str,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    let (ncols_dst, has_fusion, is_multi_token_id) = sifive_parse_mmvf_template(kernel_name)?;
    if ncols_dst == 0 || ncols_dst > 8 || (has_fusion && ncols_dst != 1) {
        return None;
    }
    let x_type = sifive_mmvf_x_type(kernel_name)?;
    let x_host = read_param_u64(kernel_params, 0)?;
    let y_host = read_param_u64(kernel_params, 1)?;
    let ids_host = read_param_u64(kernel_params, 2).unwrap_or(0);
    let fusion = if has_fusion {
        read_mmvf_fusion_args(kernel_params, 3)?
    } else {
        SifiveMmvFusionArgs::default()
    };
    let dst_host = read_param_u64(kernel_params, 4)?;
    let x_bytes = sifive_mul_mat_vec_f_binding_metadata(
        kernel_name,
        kernel_params,
        grid_dim_x,
        grid_dim_y,
        grid_dim_z,
        0,
    )?
    .0 as usize;
    let y_bytes = sifive_mul_mat_vec_f_binding_metadata(
        kernel_name,
        kernel_params,
        grid_dim_x,
        grid_dim_y,
        grid_dim_z,
        1,
    )?
    .0 as usize;
    let ids_bytes = if ids_host != 0 {
        sifive_mul_mat_vec_f_binding_metadata(
            kernel_name,
            kernel_params,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            2,
        )?
        .0 as usize
    } else {
        0
    };
    let dst_bytes = sifive_mul_mat_vec_f_binding_metadata(
        kernel_name,
        kernel_params,
        grid_dim_x,
        grid_dim_y,
        grid_dim_z,
        4,
    )?
    .0 as usize;
    if x_bytes == 0
        || y_bytes == 0
        || dst_bytes == 0
        || !sifive_host_or_cuda_alloc_has_bytes(x_host, x_bytes, false)
        || !sifive_host_or_cuda_alloc_has_bytes(y_host, y_bytes, false)
        || !sifive_host_or_cuda_alloc_has_bytes(dst_host, dst_bytes, true)
        || (fusion.gate != 0 && !sifive_host_or_cuda_alloc_has_bytes(fusion.gate, x_bytes, false))
        || (fusion.x_bias != 0
            && !sifive_host_or_cuda_alloc_has_bytes(fusion.x_bias, dst_bytes, false))
        || (fusion.gate_bias != 0
            && !sifive_host_or_cuda_alloc_has_bytes(fusion.gate_bias, dst_bytes, false))
        || (ids_host != 0 && !sifive_host_or_cuda_alloc_has_bytes(ids_host, ids_bytes, false))
    {
        eprintln!(
            "[SIFIVE Backend] host-fallback MMVF '{}' rejected ranges x=0x{:x}/{} y=0x{:x}/{} ids=0x{:x}/{} dst=0x{:x}/{} gate=0x{:x} x_bias=0x{:x} gate_bias=0x{:x}",
            kernel_name,
            x_host,
            x_bytes,
            y_host,
            y_bytes,
            ids_host,
            ids_bytes,
            dst_host,
            dst_bytes,
            fusion.gate,
            fusion.x_bias,
            fusion.gate_bias
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }

    let ncols2 = read_param_u32(kernel_params, 5).unwrap_or(0) as i32;
    if ncols2 <= 0 {
        return Some(Ok(()));
    }
    let nchannels_y = read_param_uint3_value(kernel_params, 6)?;
    let stride_row = read_param_u32(kernel_params, 7).unwrap_or(0) as i64;
    let stride_col_y2 = read_param_u32(kernel_params, 8).unwrap_or(0) as i64;
    let stride_col_dst = read_param_u32(kernel_params, 9).unwrap_or(0) as i64;
    let channel_ratio = read_param_uint3_value(kernel_params, 10)?;
    let stride_channel_x = read_param_u32(kernel_params, 11).unwrap_or(0) as i64;
    let stride_channel_y = read_param_u32(kernel_params, 12).unwrap_or(0) as i64;
    let stride_channel_dst = read_param_u32(kernel_params, 13).unwrap_or(0) as i64;
    let sample_ratio = read_param_uint3_value(kernel_params, 14)?;
    let stride_sample_x = read_param_u32(kernel_params, 15).unwrap_or(0) as i64;
    let stride_sample_y = read_param_u32(kernel_params, 16).unwrap_or(0) as i64;
    let stride_sample_dst = read_param_u32(kernel_params, 17).unwrap_or(0) as i64;
    let ids_stride = read_param_u32(kernel_params, 18).unwrap_or(0) as i64;
    let grid_x = grid_dim_x.max(1) as u64;
    let grid_y = grid_dim_y.max(1) as u64;
    let grid_z = grid_dim_z.max(1) as u64;
    let work_items = grid_x.checked_mul(grid_y)?.checked_mul(grid_z)?;
    let x_elem_size = if x_type == 2 || x_type == 3 {
        2usize
    } else {
        4usize
    };
    let channel_ratio_z = channel_ratio.z.max(1) as u64;
    let sample_ratio_z = sample_ratio.z.max(1) as u64;
    let _ = nchannels_y;

    let workers = std::env::var("HETGPU_SIFIVE_MMVF_HOST_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4)
        .max(1)
        .min(work_items.max(1) as usize);
    let x_addr = x_host as usize;
    let y_addr = y_host as usize;
    let ids_addr = ids_host as usize;
    let dst_addr = dst_host as usize;
    let gate_addr = fusion.gate as usize;
    let x_bias_addr = fusion.x_bias as usize;
    let gate_bias_addr = fusion.gate_bias as usize;
    let glu_op = fusion.glu_op;
    let has_ids = ids_host != 0;
    let use_gate = has_fusion && fusion.gate != 0;
    let use_bias = has_fusion && fusion.x_bias != 0;
    let use_gate_bias = use_gate && fusion.gate_bias != 0;
    let ncols_dst_u = ncols_dst as u64;
    let ncols2_i = ncols2 as i64;

    std::thread::scope(|scope| {
        for worker in 0..workers {
            let begin = work_items * worker as u64 / workers as u64;
            let end = work_items * (worker as u64 + 1) / workers as u64;
            scope.spawn(move || {
                let x_base_ptr = x_addr as *const u8;
                let y_base_ptr = y_addr as *const f32;
                let ids_base_ptr = ids_addr as *const i32;
                let dst_base_ptr = dst_addr as *mut f32;
                let gate_base_ptr = gate_addr as *const u8;
                let x_bias_base_ptr = x_bias_addr as *const f32;
                let gate_bias_base_ptr = gate_bias_addr as *const f32;
                for idx in begin..end {
                    let row = idx % grid_x;
                    let t = idx / grid_x;
                    let channel_dst = t % grid_y;
                    let token_or_sample = t / grid_y;

                    let (token_idx, channel_x, channel_y, sample_dst) = if has_ids {
                        let token_idx = token_or_sample;
                        let ids_off =
                            channel_dst.saturating_add(token_idx.saturating_mul(ids_stride as u64));
                        let channel_x = ids_base_ptr
                            .offset(ids_off as isize)
                            .read_unaligned()
                            .max(0) as u64;
                        let channel_y = channel_dst % (nchannels_y.z.max(1) as u64);
                        (token_idx, channel_x, channel_y, 0)
                    } else {
                        let sample_dst = token_or_sample;
                        (0, channel_dst / channel_ratio_z, channel_dst, sample_dst)
                    };

                    let sample_x = sample_dst / sample_ratio_z;
                    let sample_y = sample_dst;

                    let x_base_elem = sample_x as i64 * stride_sample_x
                        + channel_x as i64 * stride_channel_x
                        + row as i64 * stride_row;
                    let y_base_elem = sample_y as i64 * stride_sample_y
                        + channel_y as i64 * stride_channel_y
                        + if has_ids && is_multi_token_id {
                            token_idx as i64 * stride_col_y2 * 2
                        } else {
                            0
                        };
                    let dst_base_elem = sample_dst as i64 * stride_sample_dst
                        + channel_dst as i64 * stride_channel_dst
                        + if has_ids && is_multi_token_id {
                            token_idx as i64 * stride_col_dst
                        } else {
                            0
                        };
                    let channel_bias = if has_ids { channel_x } else { channel_dst };
                    let bias_base_elem = sample_dst as i64 * stride_sample_dst
                        + channel_bias as i64 * stride_channel_dst;

                    if (x_type == 2 || x_type == 3) && ncols_dst_u == 1 {
                        let yf = y_base_ptr.offset(y_base_elem as isize);
                        let mut sum = 0.0f32;
                        let mut gate_sum = 0.0f32;
                        let total = ncols2_i * 2;
                        for i in 0..total {
                            let yv = yf.offset(i as isize).read_unaligned();
                            sum += sifive_read_f16_bf16_or_f32(x_base_ptr, x_base_elem + i, x_type)
                                * yv;
                            if use_gate {
                                gate_sum += sifive_read_f16_bf16_or_f32(
                                    gate_base_ptr,
                                    x_base_elem + i,
                                    x_type,
                                ) * yv;
                            }
                        }
                        if use_bias {
                            sum += x_bias_base_ptr
                                .offset((bias_base_elem + row as i64) as isize)
                                .read_unaligned();
                        }
                        if use_gate {
                            if use_gate_bias {
                                gate_sum += gate_bias_base_ptr
                                    .offset((bias_base_elem + row as i64) as isize)
                                    .read_unaligned();
                            }
                            sum = match glu_op {
                                1 => sum * sifive_gelu(gate_sum),
                                3 => sifive_swiglu_oai(gate_sum, sum),
                                _ => sum * sifive_silu(gate_sum),
                            };
                        }
                        dst_base_ptr
                            .offset((dst_base_elem + row as i64) as isize)
                            .write_unaligned(sum);
                        continue;
                    }

                    for j in 0..ncols_dst_u {
                        let mut sum = 0.0f32;
                        let mut gate_sum = 0.0f32;
                        for col2 in 0..ncols2_i {
                            let x0 = sifive_read_f16_bf16_or_f32(
                                x_base_ptr,
                                x_base_elem + col2 * 2,
                                x_type,
                            );
                            let x1 = sifive_read_f16_bf16_or_f32(
                                x_base_ptr,
                                x_base_elem + col2 * 2 + 1,
                                x_type,
                            );
                            let y2 = y_base_ptr.offset(
                                (y_base_elem + ((j as i64) * stride_col_y2 + col2) * 2) as isize,
                            );
                            sum += x0 * y2.read_unaligned() + x1 * y2.add(1).read_unaligned();
                            if use_gate {
                                let gx0 = sifive_read_f16_bf16_or_f32(
                                    gate_base_ptr,
                                    x_base_elem + col2 * 2,
                                    x_type,
                                );
                                let gx1 = sifive_read_f16_bf16_or_f32(
                                    gate_base_ptr,
                                    x_base_elem + col2 * 2 + 1,
                                    x_type,
                                );
                                gate_sum +=
                                    gx0 * y2.read_unaligned() + gx1 * y2.add(1).read_unaligned();
                            }
                        }
                        if use_bias {
                            sum += x_bias_base_ptr
                                .offset(
                                    (bias_base_elem + j as i64 * stride_col_dst + row as i64)
                                        as isize,
                                )
                                .read_unaligned();
                        }
                        if use_gate {
                            if use_gate_bias {
                                gate_sum += gate_bias_base_ptr
                                    .offset(
                                        (bias_base_elem + j as i64 * stride_col_dst + row as i64)
                                            as isize,
                                    )
                                    .read_unaligned();
                            }
                            sum = match glu_op {
                                1 => sum * sifive_gelu(gate_sum),
                                3 => sifive_swiglu_oai(gate_sum, sum),
                                _ => sum * sifive_silu(gate_sum),
                            };
                        }
                        dst_base_ptr
                            .offset(
                                (dst_base_elem + j as i64 * stride_col_dst + row as i64) as isize,
                            )
                            .write_unaligned(sum);
                    }
                }
            });
        }
    });

    if std::env::var("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!(
            "[SIFIVE Backend] host-fallback MMVF '{}' work={} ncols2={} ncols_dst={} x_type={} fusion={} workers={}",
            kernel_name, work_items, ncols2, ncols_dst, x_type, has_fusion, workers
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn try_offload_mmvf_named_sifive_kernel(
    kernel_name: &str,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    use cuda_types::cuda::*;
    let mmvf_trace = sifive_env_truthy("HETGPU_SIFIVE_MMVF_TRACE")
        || sifive_env_truthy("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS");
    macro_rules! trace_mmvf {
        ($($arg:tt)*) => {
            if mmvf_trace {
                eprintln!($($arg)*);
            }
        };
    }

    unsafe fn finish_without_direct_sifive(
        reason: &str,
        kernel_name: &str,
        grid_dim_x: ::core::ffi::c_uint,
        grid_dim_y: ::core::ffi::c_uint,
        grid_dim_z: ::core::ffi::c_uint,
        kernel_params: *mut *mut ::core::ffi::c_void,
    ) -> Option<cuda_types::cuda::CUresult> {
        if sifive_env_truthy("HETGPU_SIFIVE_MMVF_HOST_FALLBACK")
            || sifive_env_truthy("HETGPU_SIFIVE_ALLOW_NAMED_HOST_FALLBACK")
        {
            if let Some(result) = execute_mmvf_host_fallback(
                kernel_name,
                grid_dim_x,
                grid_dim_y,
                grid_dim_z,
                kernel_params,
            ) {
                return Some(result);
            }
        }
        if sifive_named_fail_open_enabled() {
            return sifive_named_assume_success(reason, kernel_name);
        }
        Some(Err(CUerror::UNKNOWN))
    }

    if sifive_env_truthy("HETGPU_SIFIVE_FUSE_GEMV_TO_GEMM")
        || std::env::var("HETGPU_SIFIVE_MMVF_SUBMIT")
            .ok()
            .map(|value| value.trim() == "0")
            .unwrap_or(false)
    {
        trace_mmvf!(
            "[SIFIVE Backend] MMVF '{}' direct submit disabled by GEMV->GEMM policy",
            kernel_name
        );
        return finish_without_direct_sifive(
            "MMVF direct submit disabled by GEMV->GEMM policy",
            kernel_name,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            kernel_params,
        );
    }

    if std::env::var("HETGPU_SIFIVE_MMVF_NAMED_OFFLOAD")
        .ok()
        .as_deref()
        == Some("0")
    {
        trace_mmvf!(
            "[SIFIVE Backend] MMVF '{}' direct offload disabled by HETGPU_SIFIVE_MMVF_NAMED_OFFLOAD=0",
            kernel_name
        );
        return None;
    }
    if sifive_env_truthy("HETGPU_SIFIVE_MMVF_FAST_SUCCESS")
        || sifive_env_truthy("HETGPU_SIFIVE_MMVF_NAMED_FAIL_OPEN")
    {
        return sifive_named_assume_success("MMVF named fast-success requested", kernel_name);
    }

    let (ncols_dst, has_fusion, is_multi_token_id) = match sifive_parse_mmvf_template(kernel_name) {
        Some(parsed) => parsed,
        None => {
            trace_mmvf!(
                "[SIFIVE Backend] MMVF '{}' template parse failed grid={}x{}x{}",
                kernel_name,
                grid_dim_x,
                grid_dim_y,
                grid_dim_z
            );
            return None;
        }
    };
    trace_mmvf!(
        "[SIFIVE Backend] MMVF '{}' parsed grid={}x{}x{} ncols_dst={} fusion={} multi_token_id={}",
        kernel_name,
        grid_dim_x,
        grid_dim_y,
        grid_dim_z,
        ncols_dst,
        has_fusion,
        is_multi_token_id
    );
    if is_multi_token_id || ncols_dst == 0 || ncols_dst > 8 || (has_fusion && ncols_dst != 1) {
        trace_mmvf!(
            "[SIFIVE Backend] MMVF '{}' unsupported template ncols_dst={} fusion={} multi_token_id={}",
            kernel_name,
            ncols_dst,
            has_fusion,
            is_multi_token_id
        );
        return finish_without_direct_sifive(
            "MMVF template unsupported by direct SIFIVE offload",
            kernel_name,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            kernel_params,
        );
    }
    let x_type = match sifive_mmvf_x_type(kernel_name) {
        Some(value) => value,
        None => {
            trace_mmvf!(
                "[SIFIVE Backend] MMVF '{}' x_type parse failed",
                kernel_name
            );
            return None;
        }
    };
    let mmvf_host_fallback = sifive_env_truthy("HETGPU_SIFIVE_MMVF_HOST_FALLBACK");
    if mmvf_host_fallback {
        trace_mmvf!(
            "[SIFIVE Backend] MMVF '{}' using explicit host fallback",
            kernel_name
        );
        return execute_mmvf_host_fallback(
            kernel_name,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            kernel_params,
        );
    }
    if SIFIVE_MMVF_OFFLOAD_DISABLED_AFTER_FAILURE.load(Ordering::Relaxed) {
        trace_mmvf!(
            "[SIFIVE Backend] MMVF '{}' rejected: offload disabled after prior failure",
            kernel_name
        );
        if sifive_named_fail_open_enabled() {
            return sifive_named_assume_success(
                "MMVF offload disabled after prior failure",
                kernel_name,
            );
        }
        return Some(Err(CUerror::UNKNOWN));
    }

    let x_host = match read_param_u64(kernel_params, 0) {
        Some(value) => value,
        None => {
            trace_mmvf!("[SIFIVE Backend] MMVF '{}' missing x param[0]", kernel_name);
            return None;
        }
    };
    let y_host = match read_param_u64(kernel_params, 1) {
        Some(value) => value,
        None => {
            trace_mmvf!("[SIFIVE Backend] MMVF '{}' missing y param[1]", kernel_name);
            return None;
        }
    };
    let ids_host = read_param_u64(kernel_params, 2).unwrap_or(0);
    let dst_host = match read_param_u64(kernel_params, 4) {
        Some(value) => value,
        None => {
            trace_mmvf!(
                "[SIFIVE Backend] MMVF '{}' missing dst param[4]",
                kernel_name
            );
            return None;
        }
    };
    trace_mmvf!(
        "[SIFIVE Backend] MMVF '{}' host ptrs x=0x{:x} y=0x{:x} ids=0x{:x} dst=0x{:x}",
        kernel_name,
        x_host,
        y_host,
        ids_host,
        dst_host
    );
    if ids_host != 0 {
        trace_mmvf!(
            "[SIFIVE Backend] MMVF '{}' rejected: ids ptr is nonzero 0x{:x}",
            kernel_name,
            ids_host
        );
        return finish_without_direct_sifive(
            "MMVF ids input is not supported by direct SIFIVE offload",
            kernel_name,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            kernel_params,
        );
    }

    let x_bytes = match sifive_mul_mat_vec_f_binding_metadata(
        kernel_name,
        kernel_params,
        grid_dim_x,
        grid_dim_y,
        grid_dim_z,
        0,
    ) {
        Some((bytes, _flags)) => bytes,
        None => {
            trace_mmvf!(
                "[SIFIVE Backend] MMVF '{}' failed to derive x binding bytes",
                kernel_name
            );
            return None;
        }
    };
    let y_bytes = match sifive_mul_mat_vec_f_binding_metadata(
        kernel_name,
        kernel_params,
        grid_dim_x,
        grid_dim_y,
        grid_dim_z,
        1,
    ) {
        Some((bytes, _flags)) => bytes,
        None => {
            trace_mmvf!(
                "[SIFIVE Backend] MMVF '{}' failed to derive y binding bytes",
                kernel_name
            );
            return None;
        }
    };
    let dst_bytes = match sifive_mul_mat_vec_f_binding_metadata(
        kernel_name,
        kernel_params,
        grid_dim_x,
        grid_dim_y,
        grid_dim_z,
        4,
    ) {
        Some((bytes, _flags)) => bytes,
        None => {
            trace_mmvf!(
                "[SIFIVE Backend] MMVF '{}' failed to derive dst binding bytes",
                kernel_name
            );
            return None;
        }
    };
    let nchannels_y = match read_param_uint3_value(kernel_params, 6) {
        Some(value) => value,
        None => {
            trace_mmvf!(
                "[SIFIVE Backend] MMVF '{}' missing nchannels_y param[6]",
                kernel_name
            );
            return None;
        }
    };
    let channel_ratio = match read_param_uint3_value(kernel_params, 10) {
        Some(value) => value,
        None => {
            trace_mmvf!(
                "[SIFIVE Backend] MMVF '{}' missing channel_ratio param[10]",
                kernel_name
            );
            return None;
        }
    };
    let sample_ratio = match read_param_uint3_value(kernel_params, 14) {
        Some(value) => value,
        None => {
            trace_mmvf!(
                "[SIFIVE Backend] MMVF '{}' missing sample_ratio param[14]",
                kernel_name
            );
            return None;
        }
    };

    let ncols2_raw = read_param_u32(kernel_params, 5).unwrap_or(0);
    let stride_row_raw = read_param_u32(kernel_params, 7).unwrap_or(0);
    let stride_col_y2_raw = read_param_u32(kernel_params, 8).unwrap_or(0);
    let stride_col_dst_raw = read_param_u32(kernel_params, 9).unwrap_or(0);
    let simple_small_n = ids_host == 0
        && !has_fusion
        && !is_multi_token_id
        && grid_dim_y.max(1) == 1
        && grid_dim_z.max(1) == 1
        && ncols_dst != 0
        && ncols_dst <= 8
        && ncols2_raw != 0
        && stride_row_raw != 0
        && stride_col_y2_raw != 0
        && stride_col_dst_raw != 0
        && sifive_host_or_cuda_alloc_has_bytes(x_host, x_bytes as usize, false)
        && sifive_host_or_cuda_alloc_has_bytes(y_host, y_bytes as usize, false)
        && sifive_host_or_cuda_alloc_has_bytes(dst_host, dst_bytes as usize, true);
    trace_mmvf!(
        "[SIFIVE Backend] MMVF {} small-N candidate simple={} ids=0x{:x} fusion={} multi_id={} grid={}x{}x{} ncols_dst={} ncols2={} stride_row={} stride_y2={} stride_dst={} bytes={}/{}/{} ranges={}/{}/{}",
        kernel_name,
        simple_small_n,
        ids_host,
        has_fusion,
        is_multi_token_id,
        grid_dim_x.max(1),
        grid_dim_y.max(1),
        grid_dim_z.max(1),
        ncols_dst,
        ncols2_raw,
        stride_row_raw,
        stride_col_y2_raw,
        stride_col_dst_raw,
        x_bytes,
        y_bytes,
        dst_bytes,
        sifive_host_or_cuda_alloc_has_bytes(x_host, x_bytes as usize, false),
        sifive_host_or_cuda_alloc_has_bytes(y_host, y_bytes as usize, false),
        sifive_host_or_cuda_alloc_has_bytes(dst_host, dst_bytes as usize, true)
    );
    if simple_small_n {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let atype = if x_type == 3 {
            sifive_runtime_sys::SifiveDataType::Bfloat16 as i32
        } else if x_type == 2 {
            sifive_runtime_sys::SifiveDataType::Float16 as i32
        } else {
            sifive_runtime_sys::SifiveDataType::Float32 as i32
        };
        let rc = if sifive_env_truthy("HETGPU_SIFIVE_MMVF_USE_GEMM_STAGED") {
            sifive_runtime_sys::hetgpu_sifive_submit_gemm_staged_tiled(
                1,
                0,
                grid_dim_x.max(1) as i32,
                ncols_dst as i32,
                ncols2_raw.saturating_mul(2) as i32,
                (&alpha as *const f32).cast(),
                x_host as *const ::core::ffi::c_void,
                atype,
                stride_row_raw as i32,
                0,
                y_host as *const ::core::ffi::c_void,
                sifive_runtime_sys::SifiveDataType::Float32 as i32,
                stride_col_y2_raw.saturating_mul(2) as i32,
                0,
                (&beta as *const f32).cast(),
                dst_host as *mut ::core::ffi::c_void,
                sifive_runtime_sys::SifiveDataType::Float32 as i32,
                stride_col_dst_raw as i32,
                0,
                1,
                0,
                2048,
                ncols_dst.max(1) as i32,
                ncols2_raw.saturating_mul(2) as i32,
            )
        } else {
            sifive_runtime_sys::hetgpu_sifive_submit_gemm_mmvf_small_n(
                1,
                0,
                grid_dim_x.max(1) as i32,
                ncols_dst as i32,
                ncols2_raw.saturating_mul(2) as i32,
                (&alpha as *const f32).cast(),
                x_host as *const ::core::ffi::c_void,
                atype,
                stride_row_raw as i32,
                0,
                y_host as *const ::core::ffi::c_void,
                sifive_runtime_sys::SifiveDataType::Float32 as i32,
                stride_col_y2_raw.saturating_mul(2) as i32,
                0,
                (&beta as *const f32).cast(),
                dst_host as *mut ::core::ffi::c_void,
                sifive_runtime_sys::SifiveDataType::Float32 as i32,
                stride_col_dst_raw as i32,
                0,
                1,
                0,
            )
        };
        if rc == 0 {
            trace_mmvf!(
                "[SIFIVE Backend] offloaded MMVF {} via staged small-N m={} n={} k={} atype={} lda={} ldb={} ldc={}",
                kernel_name,
                grid_dim_x.max(1),
                ncols_dst,
                ncols2_raw.saturating_mul(2),
                atype,
                stride_row_raw,
                stride_col_y2_raw.saturating_mul(2),
                stride_col_dst_raw
            );
            return Some(Ok(()));
        }
        trace_mmvf!(
            "[SIFIVE Backend] MMVF {} staged small-N returned rc={} m={} n={} k={}",
            kernel_name,
            rc,
            grid_dim_x.max(1),
            ncols_dst,
            ncols2_raw.saturating_mul(2)
        );
    }

    let x_addr = match super::memory::sifive_driver_physical_addr(x_host) {
        Some(addr) => addr,
        None => {
            trace_mmvf!(
                "[SIFIVE Backend] MMVF '{}' x host ptr 0x{:x} has no SIFIVE physical address",
                kernel_name,
                x_host
            );
            return finish_without_direct_sifive(
                "MMVF x allocation has no SIFIVE physical address",
                kernel_name,
                grid_dim_x,
                grid_dim_y,
                grid_dim_z,
                kernel_params,
            );
        }
    };
    let y_addr = match super::memory::sifive_driver_physical_addr(y_host) {
        Some(addr) => addr,
        None => {
            trace_mmvf!(
                "[SIFIVE Backend] MMVF '{}' y host ptr 0x{:x} has no SIFIVE physical address",
                kernel_name,
                y_host
            );
            return finish_without_direct_sifive(
                "MMVF y allocation has no SIFIVE physical address",
                kernel_name,
                grid_dim_x,
                grid_dim_y,
                grid_dim_z,
                kernel_params,
            );
        }
    };
    let dst_addr = match super::memory::sifive_driver_physical_addr(dst_host) {
        Some(addr) => addr,
        None => {
            trace_mmvf!(
                "[SIFIVE Backend] MMVF '{}' dst host ptr 0x{:x} has no SIFIVE physical address",
                kernel_name,
                dst_host
            );
            return finish_without_direct_sifive(
                "MMVF dst allocation has no SIFIVE physical address",
                kernel_name,
                grid_dim_x,
                grid_dim_y,
                grid_dim_z,
                kernel_params,
            );
        }
    };

    let job = sifive_runtime_sys::HetgpuSifiveMmvfJob {
        x_addr,
        y_addr,
        ids_addr: 0,
        dst_addr,
        x_bytes,
        y_bytes,
        dst_bytes,
        grid_x: grid_dim_x.max(1),
        grid_y: grid_dim_y.max(1),
        grid_z: grid_dim_z.max(1),
        ncols_dst: ncols_dst as u32,
        x_type,
        reserved0: 0,
        ncols2: read_param_u32(kernel_params, 5).unwrap_or(0) as i32,
        nchannels_y,
        stride_row: read_param_u32(kernel_params, 7).unwrap_or(0) as i32,
        stride_col_y2: read_param_u32(kernel_params, 8).unwrap_or(0) as i32,
        stride_col_dst: read_param_u32(kernel_params, 9).unwrap_or(0) as i32,
        channel_ratio,
        stride_channel_x: read_param_u32(kernel_params, 11).unwrap_or(0) as i32,
        stride_channel_y: read_param_u32(kernel_params, 12).unwrap_or(0) as i32,
        stride_channel_dst: read_param_u32(kernel_params, 13).unwrap_or(0) as i32,
        sample_ratio,
        stride_sample_x: read_param_u32(kernel_params, 15).unwrap_or(0) as i32,
        stride_sample_y: read_param_u32(kernel_params, 16).unwrap_or(0) as i32,
        stride_sample_dst: read_param_u32(kernel_params, 17).unwrap_or(0) as i32,
        ids_stride: read_param_u32(kernel_params, 18).unwrap_or(0) as i32,
    };

    if job.ncols2 <= 0 || job.x_bytes == 0 || job.y_bytes == 0 || job.dst_bytes == 0 {
        trace_mmvf!(
            "[SIFIVE Backend] MMVF '{}' empty metadata ncols2={} x_bytes={} y_bytes={} dst_bytes={}",
            kernel_name,
            job.ncols2,
            job.x_bytes,
            job.y_bytes,
            job.dst_bytes
        );
        return finish_without_direct_sifive(
            "MMVF binding metadata is empty",
            kernel_name,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            kernel_params,
        );
    }

    let dev_id = current_sifive_device_id_or_zero();
    trace_mmvf!(
        "[SIFIVE Backend] MMVF '{}' submit dev={} x_addr=0x{:x} y_addr=0x{:x} dst_addr=0x{:x} bytes={}/{}/{} grid={}x{}x{} ncols2={} ncols_dst={} x_type={} stride_row={} stride_col_y2={} stride_col_dst={}",
        kernel_name,
        dev_id,
        job.x_addr,
        job.y_addr,
        job.dst_addr,
        job.x_bytes,
        job.y_bytes,
        job.dst_bytes,
        job.grid_x,
        job.grid_y,
        job.grid_z,
        job.ncols2,
        job.ncols_dst,
        job.x_type,
        job.stride_row,
        job.stride_col_y2,
        job.stride_col_dst
    );
    let rc = sifive_runtime_sys::hetgpu_sifive_submit_mmvf_on(dev_id, &job as *const _);
    if rc == 0 {
        if mmvf_trace {
            eprintln!(
                "[SIFIVE Backend] offloaded MMVF '{}' dev={} grid={}x{}x{} ncols2={} ncols_dst={} x_type={}",
                kernel_name,
                dev_id,
                job.grid_x,
                job.grid_y,
                job.grid_z,
                job.ncols2,
                job.ncols_dst,
                job.x_type
            );
        }
        return Some(Ok(()));
    }
    trace_mmvf!(
        "[SIFIVE Backend] MMVF '{}' submit returned rc={} dev={} seq path failed",
        kernel_name,
        rc,
        dev_id
    );

    if !SIFIVE_MMVF_OFFLOAD_DISABLED_AFTER_FAILURE.swap(true, Ordering::Relaxed) {
        sifive_log_limited(
            &SIFIVE_NAMED_ERROR_LOG_COUNT,
            "HETGPU_SIFIVE_NAMED_ERROR_LOG_LIMIT",
            64,
            || {
                eprintln!(
                    "[SIFIVE Backend] MMVF '{}' offload failed with rc={}; disabling MMVF offload for this process",
                    kernel_name, rc
                );
            },
        );
    }
    if sifive_named_fail_open_enabled() {
        return sifive_named_assume_success("MMVF SIFIVE offload failed", kernel_name);
    }
    if std::env::var("HETGPU_SIFIVE_ALLOW_NAMED_HOST_FALLBACK")
        .ok()
        .as_deref()
        == Some("1")
    {
        if let Some(result) = execute_mmvf_host_fallback(
            kernel_name,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            kernel_params,
        ) {
            return Some(result);
        }
    }
    Some(Err(CUerror::UNKNOWN))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_parse_mmf_rows_per_block(kernel_name: &str) -> Option<u64> {
    let marker = "mul_mat_f";
    let after = &kernel_name[kernel_name.find(marker)? + marker.len()..];
    let li = after.find("Li")? + 2;
    let digits = after[li..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .count();
    if digits == 0 {
        return None;
    }
    after[li..li + digits].parse::<u64>().ok()
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn try_offload_mul_mat_f_named_sifive_kernel(
    kernel_name: &str,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    use cuda_types::cuda::*;

    if !sifive_env_enabled_default("HETGPU_SIFIVE_MUL_MAT_F_OFFLOAD", true) {
        return None;
    }

    let name_lower = kernel_name.to_ascii_lowercase();
    let is_bf16 = name_lower.contains("bfloat162") || name_lower.contains("nv_bfloat162");
    if !is_bf16 || !kernel_name.contains("ELb0") {
        return None;
    }

    let rows_per_block = sifive_parse_mmf_rows_per_block(kernel_name)
        .unwrap_or(32)
        .max(1);
    let x_host = read_param_u64(kernel_params, 0)?;
    let y_host = read_param_u64(kernel_params, 1)?;
    let ids_host = read_param_u64(kernel_params, 2).unwrap_or(0);
    let dst_host = read_param_u64(kernel_params, 3)?;
    if ids_host != 0 || x_host == 0 || y_host == 0 || dst_host == 0 {
        return None;
    }

    let ncols2 = read_param_i32(kernel_params, 4)?.max(0) as u64;
    let ncols_dst = read_param_i32(kernel_params, 5)?.max(0) as u64;
    let nchannels_dst = read_param_i32(kernel_params, 6)?.max(0) as u64;
    let stride_row2 = read_param_i32(kernel_params, 7)?.max(0) as u64;
    let stride_col_y2 = read_param_i32(kernel_params, 8)?.max(0) as u64;
    let stride_col_dst = read_param_i32(kernel_params, 9)?.max(0) as u64;
    let channel_ratio = read_param_i32(kernel_params, 12).unwrap_or(1).max(1) as u64;
    let stride_channel_x2 = read_param_i32(kernel_params, 13)?.max(0) as u64;
    let stride_channel_y = read_param_i32(kernel_params, 14)?.max(0) as u64;
    let stride_channel_dst = read_param_i32(kernel_params, 15)?.max(0) as u64;
    let sample_ratio = read_param_i32(kernel_params, 16).unwrap_or(1).max(1) as u64;
    let stride_sample_x2 = read_param_i32(kernel_params, 17)?.max(0) as u64;
    let stride_sample_y = read_param_i32(kernel_params, 18)?.max(0) as u64;
    let stride_sample_dst = read_param_i32(kernel_params, 19)?.max(0) as u64;

    let rows = (grid_dim_x.max(1) as u64).checked_mul(rows_per_block)?;
    let grid_y = grid_dim_y.max(1) as u64;
    let grid_z = grid_dim_z.max(1) as u64;
    let n = ncols_dst.max(1);
    let k = ncols2.checked_mul(2)?;
    if rows == 0
        || n == 0
        || k == 0
        || stride_row2 == 0
        || stride_col_y2 == 0
        || stride_col_dst == 0
        || (nchannels_dst != 0 && grid_y > nchannels_dst)
    {
        return None;
    }

    let m_i32 = i32::try_from(rows).ok()?;
    let n_i32 = i32::try_from(n).ok()?;
    let k_i32 = i32::try_from(k).ok()?;
    let lda_i32 = i32::try_from(stride_row2.checked_mul(2)?).ok()?;
    let ldb_i32 = i32::try_from(stride_col_y2.checked_mul(2)?).ok()?;
    let ldc_i32 = i32::try_from(stride_col_dst).ok()?;
    let max_m = i32::try_from(
        sifive_parse_env_u64_default("HETGPU_SIFIVE_MUL_MAT_F_MAX_M", rows.min(2048)).max(1),
    )
    .unwrap_or(i32::MAX);
    let max_n =
        i32::try_from(sifive_parse_env_u64_default("HETGPU_SIFIVE_MUL_MAT_F_MAX_N", 1).max(1))
            .unwrap_or(i32::MAX);
    let max_k = i32::try_from(
        sifive_parse_env_u64_default("HETGPU_SIFIVE_MUL_MAT_F_MAX_K", k.min(256)).max(1),
    )
    .unwrap_or(i32::MAX);

    let skinny_host_first =
        sifive_env_enabled_default("HETGPU_SIFIVE_MUL_MAT_F_SKINNY_HOST_FIRST", true)
            && n <= sifive_parse_env_u64_default("HETGPU_SIFIVE_MUL_MAT_F_SKINNY_MAX_N", 1).max(1)
            && k >= sifive_parse_env_u64_default("HETGPU_SIFIVE_MUL_MAT_F_SKINNY_MIN_K", 64).max(1);

    let atype = sifive_runtime_sys::SifiveDataType::Bfloat16 as i32;
    let btype = sifive_runtime_sys::SifiveDataType::Float32 as i32;
    let ctype = sifive_runtime_sys::SifiveDataType::Float32 as i32;
    let mut rc = if skinny_host_first { -2 } else { 0 };
    if !skinny_host_first {
        for sample_dst in 0..grid_z {
            let sample_x = sample_dst / sample_ratio;
            let sample_y = sample_dst;
            for channel_dst in 0..grid_y {
                let channel_x = channel_dst / channel_ratio;
                let channel_y = channel_dst;
                let x_off_pairs = sample_x
                    .checked_mul(stride_sample_x2)?
                    .checked_add(channel_x.checked_mul(stride_channel_x2)?)?;
                let y_off_f32 = sample_y
                    .checked_mul(stride_sample_y)?
                    .checked_add(channel_y.checked_mul(stride_channel_y)?)?;
                let dst_off_f32 = sample_dst
                    .checked_mul(stride_sample_dst)?
                    .checked_add(channel_dst.checked_mul(stride_channel_dst)?)?;
                let x_ptr = (x_host as *const u16)
                    .add(usize::try_from(x_off_pairs.checked_mul(2)?).ok()?)
                    .cast::<::core::ffi::c_void>();
                let y_ptr = (y_host as *const f32)
                    .add(usize::try_from(y_off_f32).ok()?)
                    .cast::<::core::ffi::c_void>();
                let dst_ptr = (dst_host as *mut f32)
                    .add(usize::try_from(dst_off_f32).ok()?)
                    .cast::<::core::ffi::c_void>();
                rc = if std::env::var("HETGPU_SIFIVE_MUL_MAT_F_STAGED_ON")
                    .ok()
                    .as_deref()
                    == Some("1")
                {
                    sifive_runtime_sys::hetgpu_sifive_submit_gemm_staged_on(
                        -1,
                        -1,
                        1,
                        0,
                        m_i32,
                        n_i32,
                        k_i32,
                        std::ptr::null(),
                        x_ptr,
                        atype,
                        lda_i32,
                        0,
                        y_ptr,
                        btype,
                        ldb_i32,
                        0,
                        std::ptr::null(),
                        dst_ptr,
                        ctype,
                        ldc_i32,
                        0,
                        1,
                        ctype,
                    )
                } else {
                    sifive_runtime_sys::hetgpu_sifive_submit_gemm_staged_tiled(
                        1,
                        0,
                        m_i32,
                        n_i32,
                        k_i32,
                        std::ptr::null(),
                        x_ptr,
                        atype,
                        lda_i32,
                        0,
                        y_ptr,
                        btype,
                        ldb_i32,
                        0,
                        std::ptr::null(),
                        dst_ptr,
                        ctype,
                        ldc_i32,
                        0,
                        1,
                        ctype,
                        max_m,
                        max_n,
                        max_k,
                    )
                };
                if rc != 0 {
                    break;
                }
            }
            if rc != 0 {
                break;
            }
        }
    }

    if rc == 0 {
        if std::env::var("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS")
            .ok()
            .as_deref()
            == Some("1")
        {
            eprintln!(
                "[SIFIVE Backend] offloaded mul_mat_f BF16 '{}' via staged GEMM grid={}x{}x{} m={} n={} k={} lda={} ldb={} ldc={}",
                kernel_name,
                grid_dim_x,
                grid_dim_y,
                grid_dim_z,
                rows,
                n,
                k,
                lda_i32,
                ldb_i32,
                ldc_i32
            );
        }
        return Some(Ok(()));
    }

    if skinny_host_first {
        if sifive_env_truthy("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS") {
            eprintln!(
                "[SIFIVE Backend] mul_mat_f BF16 '{}' skinny n={} k={} uses host path before SIFIVE submit",
                kernel_name, n, k
            );
        }
    } else {
        sifive_log_limited(
            &SIFIVE_NAMED_ERROR_LOG_COUNT,
            "HETGPU_SIFIVE_NAMED_ERROR_LOG_LIMIT",
            64,
            || {
                eprintln!(
                    "[SIFIVE Backend] mul_mat_f BF16 '{}' staged GEMM offload failed rc={} grid={}x{}x{} m={} n={} k={}",
                    kernel_name, rc, grid_dim_x, grid_dim_y, grid_dim_z, rows, n, k
                );
            },
        );
    }
    if sifive_env_enabled_default("HETGPU_SIFIVE_MUL_MAT_F_HOST_FALLBACK", true)
        || sifive_env_truthy("HETGPU_SIFIVE_ALLOW_NAMED_HOST_FALLBACK")
    {
        let work_items = rows.checked_mul(grid_y)?.checked_mul(grid_z)?;
        let workers = std::env::var("HETGPU_SIFIVE_MUL_MAT_F_HOST_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(4)
            .max(1)
            .min(work_items.max(1) as usize);
        let x_addr = x_host as usize;
        let y_addr = y_host as usize;
        let dst_addr = dst_host as usize;
        let rows_u = rows;
        let grid_y_u = grid_y;
        let n_u = n;
        let ncols2_i = i64::try_from(ncols2).ok()?;
        let channel_ratio_u = channel_ratio.max(1);
        let sample_ratio_u = sample_ratio.max(1);
        let stride_row2_i = i64::try_from(stride_row2).ok()?;
        let stride_col_y2_i = i64::try_from(stride_col_y2).ok()?;
        let stride_col_dst_i = i64::try_from(stride_col_dst).ok()?;
        let stride_channel_x2_i = i64::try_from(stride_channel_x2).ok()?;
        let stride_channel_y_i = i64::try_from(stride_channel_y).ok()?;
        let stride_channel_dst_i = i64::try_from(stride_channel_dst).ok()?;
        let stride_sample_x2_i = i64::try_from(stride_sample_x2).ok()?;
        let stride_sample_y_i = i64::try_from(stride_sample_y).ok()?;
        let stride_sample_dst_i = i64::try_from(stride_sample_dst).ok()?;

        std::thread::scope(|scope| {
            for worker in 0..workers {
                let begin = work_items * worker as u64 / workers as u64;
                let end = work_items * (worker as u64 + 1) / workers as u64;
                scope.spawn(move || {
                    let x_base_ptr = x_addr as *const u8;
                    let y_base_ptr = y_addr as *const f32;
                    let dst_base_ptr = dst_addr as *mut f32;
                    for idx in begin..end {
                        let row = idx % rows_u;
                        let t = idx / rows_u;
                        let channel_dst = t % grid_y_u;
                        let sample_dst = t / grid_y_u;
                        let sample_x = sample_dst / sample_ratio_u;
                        let sample_y = sample_dst;
                        let channel_x = channel_dst / channel_ratio_u;
                        let channel_y = channel_dst;
                        let x_base_elem = ((sample_x as i64 * stride_sample_x2_i
                            + channel_x as i64 * stride_channel_x2_i
                            + row as i64 * stride_row2_i)
                            * 2) as i64;
                        let y_base_elem = sample_y as i64 * stride_sample_y_i
                            + channel_y as i64 * stride_channel_y_i;
                        let dst_base_elem = sample_dst as i64 * stride_sample_dst_i
                            + channel_dst as i64 * stride_channel_dst_i;
                        for j in 0..n_u {
                            let mut sum = 0.0f32;
                            for col2 in 0..ncols2_i {
                                let x0 = sifive_read_f16_bf16_or_f32(
                                    x_base_ptr,
                                    x_base_elem + col2 * 2,
                                    3,
                                );
                                let x1 = sifive_read_f16_bf16_or_f32(
                                    x_base_ptr,
                                    x_base_elem + col2 * 2 + 1,
                                    3,
                                );
                                let y2 = y_base_ptr.offset(
                                    (y_base_elem + (j as i64 * stride_col_y2_i + col2) * 2)
                                        as isize,
                                );
                                sum += x0 * y2.read_unaligned() + x1 * y2.add(1).read_unaligned();
                            }
                            dst_base_ptr
                                .offset(
                                    (dst_base_elem + j as i64 * stride_col_dst_i + row as i64)
                                        as isize,
                                )
                                .write_unaligned(sum);
                        }
                    }
                });
            }
        });

        if sifive_env_truthy("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS") {
            eprintln!(
                "[SIFIVE Backend] host-fallback mul_mat_f BF16 '{}' grid={}x{}x{} rows={} n={} k={} workers={}",
                kernel_name,
                grid_dim_x,
                grid_dim_y,
                grid_dim_z,
                rows,
                n,
                k,
                workers
            );
        }
        return Some(Ok(()));
    }
    if sifive_env_truthy("HETGPU_SIFIVE_MUL_MAT_F_FAIL_OPEN") {
        return sifive_named_assume_success("mul_mat_f BF16 SIFIVE offload failed", kernel_name);
    }
    Some(Err(CUerror::UNKNOWN))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_rope_multi_binding_metadata(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: u32,
    index: usize,
) -> Option<(u64, u32)> {
    let (src_elem_size, dst_elem_size) = sifive_rope_element_sizes(kernel_name);
    let ne00 = read_param_i32(kernel_params, 2)?.max(0) as u64;
    let ne01 = read_param_i32(kernel_params, 3)?.max(0) as u64;
    let ne02 = read_param_i32(kernel_params, 4)?.max(0) as u64;
    let s01 = read_param_i32(kernel_params, 5)?.max(0) as u64;
    let s02 = read_param_i32(kernel_params, 6)?.max(0) as u64;
    let s03 = read_param_i32(kernel_params, 7)?.max(0) as u64;
    let s1 = read_param_i32(kernel_params, 8)?.max(0) as u64;
    let s2 = read_param_i32(kernel_params, 9)?.max(0) as u64;
    let s3 = read_param_i32(kernel_params, 10)?.max(0) as u64;
    let rows = (grid_dim_x as u64).max(1);
    let plane = ne01.saturating_mul(ne02).max(1);
    let ne03 = rows.saturating_add(plane - 1) / plane;
    let src_bytes =
        sifive_strided_extent_bytes([ne00, ne01, ne02, ne03], [1, s01, s02, s03], src_elem_size);
    let dst_bytes =
        sifive_strided_extent_bytes([ne00, ne01, ne02, ne03], [1, s1, s2, s3], dst_elem_size);
    let pos_bytes = ne02.saturating_mul(4).max(1).saturating_mul(4);
    let freq_bytes = ne00.saturating_add(1) / 2 * std::mem::size_of::<f32>() as u64;
    let row_indices_bytes = ne02
        .max(1)
        .saturating_mul(std::mem::size_of::<i64>() as u64);

    let (bytes, flags) = match index {
        0 => (
            src_bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        1 => (
            dst_bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        ),
        12 => (
            pos_bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        18 => (
            freq_bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        19 if kernel_name.contains("rope_norm") || kernel_name.contains("rope_neox") => (
            row_indices_bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        _ => return None,
    };
    let bytes = sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes)?;
    Some((bytes, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_rope_host_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: ::core::ffi::c_uint,
) -> Option<cuda_types::cuda::CUresult> {
    use cuda_types::cuda::*;

    if !(kernel_name.contains("rope_norm") || kernel_name.contains("rope_neox")) {
        return None;
    }
    if std::env::var("HETGPU_SIFIVE_ROPE_HOST_FALLBACK")
        .ok()
        .as_deref()
        == Some("0")
    {
        return Some(Err(CUerror::UNKNOWN));
    }

    let src = read_param_u64(kernel_params, 0)?;
    let dst = read_param_u64(kernel_params, 1)?;
    let ne00 = read_param_i32(kernel_params, 2)?.max(0) as u64;
    let ne01 = read_param_i32(kernel_params, 3)?.max(0) as u64;
    let ne02 = read_param_i32(kernel_params, 4)?.max(0) as u64;
    let s01 = read_param_i32(kernel_params, 5)? as i64;
    let s02 = read_param_i32(kernel_params, 6)? as i64;
    let s03 = read_param_i32(kernel_params, 7)? as i64;
    let s1 = read_param_i32(kernel_params, 8)? as i64;
    let s2 = read_param_i32(kernel_params, 9)? as i64;
    let s3 = read_param_i32(kernel_params, 10)? as i64;
    let n_dims = read_param_i32(kernel_params, 11)?.max(0) as u64;
    let pos = read_param_u64(kernel_params, 12)?;
    let freq_scale = read_param_f32(kernel_params, 13)?;
    let ext_factor = read_param_f32(kernel_params, 14)?;
    let attn_factor = read_param_f32(kernel_params, 15)?;
    let corr_param = *kernel_params.add(16);
    if corr_param.is_null() || (corr_param as usize) < 0x1_0000 {
        return None;
    }
    let corr0 = (corr_param as *const f32).read_unaligned();
    let corr1 = (corr_param as *const f32).add(1).read_unaligned();
    let theta_scale = read_param_f32(kernel_params, 17)?;
    let freq_factors = read_param_u64(kernel_params, 18).unwrap_or(0);
    let row_indices = read_param_u64(kernel_params, 19).unwrap_or(0);
    let set_rows_stride = read_param_i32(kernel_params, 20).unwrap_or(0) as i64;
    if ne00 == 0 || ne01 == 0 || ne02 == 0 || n_dims == 0 || n_dims > ne00 {
        return Some(Err(CUerror::UNKNOWN));
    }

    let rows = grid_dim_x.max(1) as u64;
    let plane = ne01.saturating_mul(ne02).max(1);
    let ne03 = rows.saturating_add(plane - 1) / plane;
    let (src_elem_size, dst_elem_size) = sifive_rope_element_sizes(kernel_name);
    let src_bytes = sifive_strided_extent_bytes(
        [ne00, ne01, ne02, ne03],
        [1, s01 as u64, s02 as u64, s03 as u64],
        src_elem_size,
    );
    let dst_bytes = sifive_strided_extent_bytes(
        [ne00, ne01, ne02, ne03],
        [1, s1 as u64, s2 as u64, s3 as u64],
        dst_elem_size,
    );
    let pos_bytes = ne02
        .max(1)
        .saturating_mul(std::mem::size_of::<i32>() as u64);
    let freq_bytes = n_dims
        .saturating_add(1)
        .saturating_div(2)
        .saturating_mul(std::mem::size_of::<f32>() as u64);
    let row_indices_bytes = ne02
        .max(1)
        .saturating_mul(std::mem::size_of::<i64>() as u64);
    if !sifive_host_or_cuda_alloc_has_bytes(src, src_bytes as usize, false)
        || !sifive_host_or_cuda_alloc_has_bytes(dst, dst_bytes as usize, true)
        || !sifive_host_or_cuda_alloc_has_bytes(pos, pos_bytes as usize, false)
        || (freq_factors != 0
            && !sifive_host_or_cuda_alloc_has_bytes(freq_factors, freq_bytes as usize, false))
        || (set_rows_stride != 0
            && !sifive_host_or_cuda_alloc_has_bytes(row_indices, row_indices_bytes as usize, false))
    {
        eprintln!(
            "[SIFIVE Backend] host-fallback ROPE '{}' rejected ranges src=0x{:x}/{} dst=0x{:x}/{} pos=0x{:x}/{}",
            kernel_name, src, src_bytes, dst, dst_bytes, pos, pos_bytes
        );
        return Some(Err(CUerror::UNKNOWN));
    }

    let forward = sifive_rope_is_forward(kernel_name);
    let is_neox = kernel_name.contains("rope_neox");
    let has_ff = sifive_rope_has_freq_factors(kernel_name, freq_factors);
    let src_base = src as *const u8;
    let dst_base = dst as *mut u8;
    let pos_base = pos as *const i32;
    let freq_base = freq_factors as *const f32;
    let rows_base = row_indices as *const i64;

    for row_dst in 0..rows {
        let i3 = row_dst / plane;
        let rem = row_dst - i3 * plane;
        let i2 = rem / ne01;
        let i1 = rem - i2 * ne01;
        let pos_val = pos_base.add(i2 as usize).read_unaligned() as f32;
        for i0 in (0..ne00).step_by(2) {
            let ix = if is_neox {
                (i0 / 2) as i64 + i1 as i64 * s01 + i2 as i64 * s02 + i3 as i64 * s03
            } else {
                i0 as i64 + i1 as i64 * s01 + i2 as i64 * s02 + i3 as i64 * s03
            };
            let mut idst = if is_neox {
                (i0 / 2) as i64 + i1 as i64 * s1 + i2 as i64 * s2 + i3 as i64 * s3
            } else {
                i0 as i64 + i1 as i64 * s1 + i2 as i64 * s2 + i3 as i64 * s3
            };
            if set_rows_stride != 0 {
                idst = i1 as i64 * s1 + if is_neox { (i0 / 2) as i64 } else { i0 as i64 };
                idst += rows_base.add(i2 as usize).read_unaligned() * set_rows_stride;
            }

            if i0 >= n_dims {
                if is_neox {
                    let x0 = sifive_read_elem_as_f32(src_base, ix + (i0 / 2) as i64, src_elem_size);
                    let x1 =
                        sifive_read_elem_as_f32(src_base, ix + (i0 / 2 + 1) as i64, src_elem_size);
                    sifive_write_elem_from_f32(dst_base, idst + (i0 / 2) as i64, dst_elem_size, x0);
                    sifive_write_elem_from_f32(
                        dst_base,
                        idst + (i0 / 2 + 1) as i64,
                        dst_elem_size,
                        x1,
                    );
                } else {
                    let x0 = sifive_read_elem_as_f32(src_base, ix, src_elem_size);
                    let x1 = sifive_read_elem_as_f32(src_base, ix + 1, src_elem_size);
                    sifive_write_elem_from_f32(dst_base, idst, dst_elem_size, x0);
                    sifive_write_elem_from_f32(dst_base, idst + 1, dst_elem_size, x1);
                }
                continue;
            }

            let freq_factor = if has_ff && freq_factors != 0 {
                freq_base.add((i0 / 2) as usize).read_unaligned()
            } else {
                1.0
            };
            let theta_extrap = pos_val * theta_scale.powf(i0 as f32 / 2.0) / freq_factor;
            let theta_interp = freq_scale * theta_extrap;
            let mut theta = theta_interp;
            let mut mscale = attn_factor;
            if ext_factor != 0.0 {
                let y = (i0 as f32 / 2.0 - corr0) / 0.001f32.max(corr1 - corr0);
                let ramp = 1.0 - y.clamp(0.0, 1.0);
                let ramp_mix = ramp * ext_factor;
                theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
                mscale *= 1.0 + 0.1 * (1.0 / freq_scale).ln();
            }
            let cos_theta = theta.cos() * mscale;
            let mut sin_theta = theta.sin() * mscale;
            if !forward {
                sin_theta = -sin_theta;
            }

            if is_neox {
                let x0 = sifive_read_elem_as_f32(src_base, ix, src_elem_size);
                let x1 = sifive_read_elem_as_f32(src_base, ix + (n_dims / 2) as i64, src_elem_size);
                sifive_write_elem_from_f32(
                    dst_base,
                    idst,
                    dst_elem_size,
                    x0 * cos_theta - x1 * sin_theta,
                );
                sifive_write_elem_from_f32(
                    dst_base,
                    idst + (n_dims / 2) as i64,
                    dst_elem_size,
                    x0 * sin_theta + x1 * cos_theta,
                );
            } else {
                let x0 = sifive_read_elem_as_f32(src_base, ix, src_elem_size);
                let x1 = sifive_read_elem_as_f32(src_base, ix + 1, src_elem_size);
                sifive_write_elem_from_f32(
                    dst_base,
                    idst,
                    dst_elem_size,
                    x0 * cos_theta - x1 * sin_theta,
                );
                sifive_write_elem_from_f32(
                    dst_base,
                    idst + 1,
                    dst_elem_size,
                    x0 * sin_theta + x1 * cos_theta,
                );
            }
        }
    }

    if std::env::var("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!(
            "[SIFIVE Backend] host-fallback ROPE '{}' rows={} ne00={} n_dims={} neox={} src_elem={} dst_elem={}",
            kernel_name, rows, ne00, n_dims, is_neox, src_elem_size, dst_elem_size
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_unary_op_binding_metadata(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<(u64, u32)> {
    let elem_size = sifive_rope_element_size(kernel_name);
    let k = read_param_i32(kernel_params, 2)?.max(0) as u64;
    let bytes = k.saturating_mul(elem_size);
    let flags = match index {
        0 => sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        1 => sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        _ => return None,
    };
    let bytes = sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes)?;
    Some((bytes, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_unary_gated_access_bytes(k: u64, n: u64, stride: u64, elem_size: u64) -> u64 {
    if k == 0 || n == 0 || elem_size == 0 {
        return 0;
    }
    let last_i = k - 1;
    let last_group = last_i / n;
    let last_col = (n - 1).min(last_i % n);
    last_group
        .saturating_mul(stride)
        .saturating_add(last_col)
        .saturating_add(1)
        .saturating_mul(elem_size)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_unary_gated_op_binding_metadata(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<(u64, u32)> {
    let elem_size = sifive_rope_element_size(kernel_name);
    let k = read_param_i64(kernel_params, 3)?.max(0) as u64;
    let n = read_param_i64(kernel_params, 4)?.max(0) as u64;
    let o0 = read_param_i64(kernel_params, 5)?.max(0) as u64;
    let o1 = read_param_i64(kernel_params, 6)?.max(0) as u64;
    let (bytes, flags) = match index {
        0 => (
            sifive_unary_gated_access_bytes(k, n, o0, elem_size),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        1 => (
            sifive_unary_gated_access_bytes(k, n, o1, elem_size),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        2 => (
            k.saturating_mul(elem_size),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        ),
        _ => return None,
    };
    let bytes = sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes)?;
    Some((bytes, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_bin_bcast_binding_metadata(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
    unravel: bool,
) -> Option<(u64, u32)> {
    let (src0_elem, src1_elem, dst_elem) = sifive_bin_bcast_element_sizes(kernel_name);
    let (ne0, ne1, ne2, ne3, ne10, ne11, ne12, ne13, stride_base) = if unravel {
        (
            read_param_uint3_z(kernel_params, 3)? as u64,
            read_param_uint3_z(kernel_params, 4)? as u64,
            read_param_uint3_z(kernel_params, 5)? as u64,
            read_param_u32(kernel_params, 6)? as u64,
            read_param_uint3_z(kernel_params, 9)? as u64,
            read_param_uint3_z(kernel_params, 10)? as u64,
            read_param_uint3_z(kernel_params, 11)? as u64,
            read_param_uint3_z(kernel_params, 12)? as u64,
            13usize,
        )
    } else {
        (
            read_param_i32(kernel_params, 3)?.max(0) as u64,
            read_param_i32(kernel_params, 4)?.max(0) as u64,
            read_param_i32(kernel_params, 5)?.max(0) as u64,
            read_param_uint3_z(kernel_params, 6)? as u64,
            read_param_uint3_z(kernel_params, 7)? as u64,
            read_param_uint3_z(kernel_params, 8)? as u64,
            read_param_uint3_z(kernel_params, 9)? as u64,
            read_param_uint3_z(kernel_params, 10)? as u64,
            11usize,
        )
    };

    let s1 = read_param_i32(kernel_params, stride_base)?.max(0) as u64;
    let s2 = read_param_i32(kernel_params, stride_base + 1)?.max(0) as u64;
    let s3 = read_param_i32(kernel_params, stride_base + 2)?.max(0) as u64;
    let s00 = read_param_i32(kernel_params, stride_base + 3)?.max(0) as u64;
    let s01 = read_param_i32(kernel_params, stride_base + 4)?.max(0) as u64;
    let s02 = read_param_i32(kernel_params, stride_base + 5)?.max(0) as u64;
    let s03 = read_param_i32(kernel_params, stride_base + 6)?.max(0) as u64;
    let s10 = read_param_i32(kernel_params, stride_base + 7)?.max(0) as u64;
    let s11 = read_param_i32(kernel_params, stride_base + 8)?.max(0) as u64;
    let s12 = read_param_i32(kernel_params, stride_base + 9)?.max(0) as u64;
    let s13 = read_param_i32(kernel_params, stride_base + 10)?.max(0) as u64;

    let src0_bytes =
        sifive_strided_extent_bytes([ne0, ne1, ne2, ne3], [s00, s01, s02, s03], src0_elem);
    let src1_bytes = sifive_strided_extent_bytes(
        [ne0.min(ne10), ne1.min(ne11), ne2.min(ne12), ne3.min(ne13)],
        [s10, s11, s12, s13],
        src1_elem,
    );
    let dst_bytes = sifive_strided_extent_bytes([ne0, ne1, ne2, ne3], [1, s1, s2, s3], dst_elem);

    let (bytes, flags) = match index {
        0 => (
            src0_bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        1 => (
            src1_bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        2 => (
            dst_bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        ),
        i if i >= stride_base + 11 => (
            src1_bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        _ => return None,
    };
    let bytes = sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes)?;
    Some((bytes, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_parse_concat_dim(kernel_name: &str) -> Option<u32> {
    if kernel_name.contains("concat_f32_dim0") || kernel_name.contains("concat_f32_non_contILi0") {
        Some(0)
    } else if kernel_name.contains("concat_f32_dim1")
        || kernel_name.contains("concat_f32_non_contILi1")
    {
        Some(1)
    } else if kernel_name.contains("concat_f32_dim2")
        || kernel_name.contains("concat_f32_non_contILi2")
    {
        Some(2)
    } else if kernel_name.contains("concat_f32_non_contILi3") {
        Some(3)
    } else {
        None
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_concat_dim_binding_metadata(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_y: u32,
    grid_dim_z: u32,
    index: usize,
) -> Option<(u64, u32)> {
    let dim = sifive_parse_concat_dim(kernel_name)?;
    let ne0 = read_param_i32(kernel_params, 3)?.max(0) as u64;
    let split = read_param_i32(kernel_params, 4)?.max(0) as u64;
    let ne1 = grid_dim_y.max(1) as u64;
    let ne2 = grid_dim_z.max(1) as u64;
    let elem_size = std::mem::size_of::<f32>() as u64;

    let (src0_elems, src1_elems, dst_elems) = match dim {
        0 => {
            let src0_ne0 = split.min(ne0);
            let src1_ne0 = ne0.saturating_sub(src0_ne0);
            (
                src0_ne0.saturating_mul(ne1).saturating_mul(ne2),
                src1_ne0.saturating_mul(ne1).saturating_mul(ne2),
                ne0.saturating_mul(ne1).saturating_mul(ne2),
            )
        }
        1 => {
            let src0_ne1 = split.min(ne1);
            let src1_ne1 = ne1.saturating_sub(src0_ne1);
            (
                ne0.saturating_mul(src0_ne1).saturating_mul(ne2),
                ne0.saturating_mul(src1_ne1).saturating_mul(ne2),
                ne0.saturating_mul(ne1).saturating_mul(ne2),
            )
        }
        2 => {
            let src0_ne2 = split.min(ne2);
            let src1_ne2 = ne2.saturating_sub(src0_ne2);
            (
                ne0.saturating_mul(ne1).saturating_mul(src0_ne2),
                ne0.saturating_mul(ne1).saturating_mul(src1_ne2),
                ne0.saturating_mul(ne1).saturating_mul(ne2),
            )
        }
        _ => return None,
    };

    let (bytes, flags) = match index {
        0 => (
            src0_elems.saturating_mul(elem_size),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        1 => (
            src1_elems.saturating_mul(elem_size),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        2 => (
            dst_elems.saturating_mul(elem_size),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        ),
        _ => return None,
    };
    let bytes = sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes)?;
    Some((bytes, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_concat_non_cont_binding_metadata(
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    index: usize,
) -> Option<(u64, u32)> {
    let elem_size = std::mem::size_of::<f32>() as u64;
    let ne00 = read_param_i64(kernel_params, 3)?.max(0) as u64;
    let ne01 = read_param_i64(kernel_params, 4)?.max(0) as u64;
    let ne02 = read_param_i64(kernel_params, 5)?.max(0) as u64;
    let ne03 = read_param_i64(kernel_params, 6)?.max(0) as u64;
    let nb00 = read_param_u64(kernel_params, 7)?;
    let nb01 = read_param_u64(kernel_params, 8)?;
    let nb02 = read_param_u64(kernel_params, 9)?;
    let nb03 = read_param_u64(kernel_params, 10)?;
    let ne10 = read_param_i64(kernel_params, 11)?.max(0) as u64;
    let ne11 = read_param_i64(kernel_params, 12)?.max(0) as u64;
    let ne12 = read_param_i64(kernel_params, 13)?.max(0) as u64;
    let ne13 = read_param_i64(kernel_params, 14)?.max(0) as u64;
    let nb10 = read_param_u64(kernel_params, 15)?;
    let nb11 = read_param_u64(kernel_params, 16)?;
    let nb12 = read_param_u64(kernel_params, 17)?;
    let nb13 = read_param_u64(kernel_params, 18)?;
    let ne0 = read_param_i64(kernel_params, 19)
        .unwrap_or(grid_dim_x as i64)
        .max(0) as u64;
    let ne1 = read_param_i64(kernel_params, 20)
        .unwrap_or(grid_dim_x as i64)
        .max(0) as u64;
    let ne2 = read_param_i64(kernel_params, 21)
        .unwrap_or(grid_dim_y as i64)
        .max(0) as u64;
    let ne3 = read_param_i64(kernel_params, 22)
        .unwrap_or(grid_dim_z as i64)
        .max(0) as u64;
    let nb0 = read_param_u64(kernel_params, 23)?;
    let nb1 = read_param_u64(kernel_params, 24)?;
    let nb2 = read_param_u64(kernel_params, 25)?;
    let nb3 = read_param_u64(kernel_params, 26)?;

    let src0_bytes = sifive_strided_extent_bytes_from_byte_strides(
        [ne00, ne01, ne02, ne03],
        [nb00, nb01, nb02, nb03],
        elem_size,
    );
    let src1_bytes = sifive_strided_extent_bytes_from_byte_strides(
        [ne10, ne11, ne12, ne13],
        [nb10, nb11, nb12, nb13],
        elem_size,
    );
    let dst_bytes = sifive_strided_extent_bytes_from_byte_strides(
        [ne0, ne1, ne2, ne3],
        [nb0, nb1, nb2, nb3],
        elem_size,
    );

    let (bytes, flags) = match index {
        0 => (
            src0_bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        1 => (
            src1_bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        2 => (
            dst_bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        ),
        _ => return None,
    };
    let bytes = sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes)?;
    Some((bytes, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_quantize_q8_1_binding_metadata(
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_z: u32,
    index: usize,
) -> Option<(u64, u32)> {
    let ne00 = read_param_i64(kernel_params, 2)?.max(0) as u64;
    let s01 = read_param_i64(kernel_params, 3)?.max(0) as u64;
    let s02 = read_param_i64(kernel_params, 4)?.max(0) as u64;
    let s03 = read_param_i64(kernel_params, 5)?.max(0) as u64;
    let ne0 = read_param_i64(kernel_params, 6)?.max(0) as u64;
    let ne1 = read_param_u32(kernel_params, 7)? as u64;
    let ne2 = read_param_uint3_z(kernel_params, 8)? as u64;
    let grid_z = grid_dim_z as u64;

    match index {
        0 => {
            let ne3 = if ne2 == 0 {
                1
            } else {
                grid_z.max(1).div_ceil(ne2)
            };
            let max_i0 = ne00.min(ne0).saturating_sub(1);
            let max_off = max_i0
                .saturating_add(ne1.saturating_sub(1).saturating_mul(s01))
                .saturating_add(ne2.saturating_sub(1).saturating_mul(s02))
                .saturating_add(ne3.saturating_sub(1).saturating_mul(s03));
            let bytes = max_off
                .saturating_add(1)
                .saturating_mul(std::mem::size_of::<f32>() as u64);
            Some((
                bytes,
                sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
            ))
        }
        1 => {
            let total = ne0.saturating_mul(ne1).saturating_mul(grid_z.max(1));
            let blocks = total.saturating_add(31) / 32;
            let block_q8_1_bytes = 36u64;
            Some((
                blocks.saturating_mul(block_q8_1_bytes),
                sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
            ))
        }
        _ => None,
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_quantize_q8_1_host_fallback(
    kernel_name: &str,
    grid_dim_z: ::core::ffi::c_uint,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    let x_addr = read_param_u64(kernel_params, 0)?;
    let y_addr = read_param_u64(kernel_params, 1)?;
    let ne00 = read_param_i64(kernel_params, 2)?;
    let s01 = read_param_i64(kernel_params, 3)?;
    let s02 = read_param_i64(kernel_params, 4)?;
    let s03 = read_param_i64(kernel_params, 5)?;
    let ne0 = read_param_i64(kernel_params, 6)?;
    let ne1 = read_param_u32(kernel_params, 7)? as i64;
    let ne2 = read_param_uint3_z(kernel_params, 8)? as i64;
    let grid_z = (grid_dim_z as i64).max(1);

    if ne00 < 0 || s01 < 0 || s02 < 0 || s03 < 0 || ne0 <= 0 || ne1 <= 0 || ne2 <= 0 {
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }

    let x_bytes = sifive_quantize_q8_1_binding_metadata(kernel_params, grid_dim_z, 0)?.0 as usize;
    let y_bytes = sifive_quantize_q8_1_binding_metadata(kernel_params, grid_dim_z, 1)?.0 as usize;
    if x_bytes == 0
        || y_bytes == 0
        || !sifive_host_or_cuda_alloc_has_bytes(x_addr, x_bytes, false)
        || !sifive_host_or_cuda_alloc_has_bytes(y_addr, y_bytes, true)
    {
        sifive_log_limited(
            &SIFIVE_NAMED_ERROR_LOG_COUNT,
            "HETGPU_SIFIVE_NAMED_ERROR_LOG_LIMIT",
            64,
            || {
                eprintln!(
                    "[SIFIVE Backend] quantize_q8_1 '{}' rejected ranges x=0x{:x}/{} y=0x{:x}/{}",
                    kernel_name, x_addr, x_bytes, y_addr, y_bytes
                );
            },
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }

    let x = x_addr as *const f32;
    let y = y_addr as *mut u8;
    let ne0_u = ne0 as usize;
    let ne00_u = ne00 as usize;
    let ne1_u = ne1 as usize;
    let grid_z_u = grid_z as usize;
    let ne2_u = ne2 as usize;
    let s01_u = s01 as usize;
    let s02_u = s02 as usize;
    let s03_u = s03 as usize;

    for bz in 0..grid_z_u {
        let i3 = bz / ne2_u;
        let i2 = bz - i3 * ne2_u;
        for i1 in 0..ne1_u {
            let mut i0 = 0usize;
            while i0 < ne0_u {
                let block_index = ((bz * ne1_u + i1) * ne0_u + i0) / 32;
                let block = y.add(block_index * 36);
                let mut vals = [0.0f32; 32];
                let mut amax = 0.0f32;
                let mut sum = 0.0f32;
                for lane in 0..32usize {
                    let src_i0 = i0 + lane;
                    let value = if src_i0 < ne00_u {
                        let off = i3
                            .saturating_mul(s03_u)
                            .saturating_add(i2.saturating_mul(s02_u))
                            .saturating_add(i1.saturating_mul(s01_u))
                            .saturating_add(src_i0);
                        x.add(off).read_unaligned()
                    } else {
                        0.0
                    };
                    vals[lane] = value;
                    amax = amax.max(value.abs());
                    sum += value;
                }
                let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
                for lane in 0..32usize {
                    let q = if d == 0.0 {
                        0i8
                    } else {
                        (vals[lane] / d).round().clamp(-128.0, 127.0) as i8
                    };
                    block.add(lane).write_unaligned(q as u8);
                }
                (block.add(32) as *mut u16).write_unaligned(sifive_f32_to_f16(d));
                (block.add(34) as *mut u16).write_unaligned(sifive_f32_to_f16(sum));
                i0 += 32;
            }
        }
    }

    if std::env::var("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!(
            "[SIFIVE Backend] host-fallback quantize_q8_1 '{}' ne0={} ne1={} grid_z={} bytes={}",
            kernel_name, ne0, ne1, grid_z, y_bytes
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_dequantize_block_q8_0_f16_binding_metadata(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<(u64, u32)> {
    let k = read_param_i64(kernel_params, 2)?.max(0) as u64;
    let src_blocks = sifive_div_ceil_u64(k, 32);
    let src_bytes = src_blocks.saturating_mul(34);
    let dst_bytes = k.saturating_mul(2);
    let (bytes, flags) = match index {
        0 => (
            src_bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        1 => (
            dst_bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        ),
        _ => return None,
    };
    let bytes = sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes)?;
    Some((bytes, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_convert_unary_binding_metadata(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<(u64, u32)> {
    let (src_elem, dst_elem) = sifive_parse_convert_unary_element_sizes(kernel_name)?;
    let ne00 = read_param_i64(kernel_params, 2)?.max(0) as u64;
    let ne01 = read_param_i64(kernel_params, 3)?.max(0) as u64;
    let ne0203 = read_param_i64(kernel_params, 4)?.max(0) as u64;
    let ne02 = read_param_uint3_z(kernel_params, 5)?.max(1) as u64;
    let ne03 = sifive_div_ceil_u64(ne0203, ne02).max(1);
    let s01 = read_param_i64(kernel_params, 6)?.max(0) as u64;
    let s02 = read_param_i64(kernel_params, 7)?.max(0) as u64;
    let s03 = read_param_i64(kernel_params, 8)?.max(0) as u64;

    let src_bytes =
        sifive_strided_extent_bytes([ne00, ne01, ne02, ne03], [1, s01, s02, s03], src_elem);
    let dst_bytes = ne00
        .saturating_mul(ne01)
        .saturating_mul(ne0203)
        .saturating_mul(dst_elem);

    let (bytes, flags) = match index {
        0 => (
            src_bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        1 => (
            dst_bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        ),
        _ => return None,
    };
    let bytes = sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes)?;
    Some((bytes, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_ssm_conv_binding_metadata(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    index: usize,
) -> Option<(u64, u32)> {
    let (split_d_inner, d_conv, split_n_t) =
        sifive_parse_ssm_conv_template(kernel_name).unwrap_or((128, 4, 0));
    let src0_nb0 = read_param_i32(kernel_params, 3)?.max(0) as u64;
    let src0_nb1 = read_param_i32(kernel_params, 4)?.max(0) as u64;
    let src0_nb2 = read_param_i32(kernel_params, 5)?.max(0) as u64;
    let src1_nb1 = read_param_i32(kernel_params, 6)?.max(0) as u64;
    let dst_nb0 = read_param_i32(kernel_params, 8)?.max(0) as u64;
    let dst_nb1 = read_param_i32(kernel_params, 9)?.max(0) as u64;
    let dst_nb2 = read_param_i32(kernel_params, 10)?.max(0) as u64;
    let n_t = read_param_i64(kernel_params, 11)?.max(0) as u64;
    let grid_x = (grid_dim_x as u64).max(1);
    let grid_y = (grid_dim_y as u64).max(1);
    let grid_z = (grid_dim_z as u64).max(1);
    let elem = std::mem::size_of::<f32>() as u64;
    let rows = grid_y.saturating_mul(split_d_inner);
    let per_tile_x = if split_n_t == 0 {
        n_t.saturating_add(d_conv.saturating_sub(1))
    } else {
        split_n_t.saturating_add(d_conv.saturating_sub(1))
    };
    let src0_bytes = grid_x
        .saturating_sub(1)
        .saturating_mul(src0_nb2)
        .saturating_add(rows.saturating_sub(1).saturating_mul(src0_nb1))
        .saturating_add(
            grid_z
                .saturating_sub(1)
                .saturating_mul(split_n_t)
                .saturating_mul(src0_nb0),
        )
        .saturating_add(per_tile_x.saturating_sub(1).saturating_mul(src0_nb0))
        .saturating_add(elem);
    let src1_bytes = rows
        .saturating_sub(1)
        .saturating_mul(src1_nb1)
        .saturating_add(d_conv.saturating_mul(elem));
    let dst_steps = if split_n_t == 0 {
        n_t
    } else {
        n_t.min(grid_z.saturating_mul(split_n_t))
    };
    let dst_bytes = grid_x
        .saturating_sub(1)
        .saturating_mul(dst_nb2)
        .saturating_add(rows.saturating_sub(1).saturating_mul(dst_nb0))
        .saturating_add(dst_steps.saturating_sub(1).saturating_mul(dst_nb1))
        .saturating_add(elem);

    let (bytes, flags) = match index {
        0 => (
            src0_bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        1 => (
            src1_bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        7 => (
            dst_bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        ),
        _ => return None,
    };
    let bytes = sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes)?;
    Some((bytes, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_gated_delta_net_binding_metadata(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<(u64, u32)> {
    let (s_v, kda) = sifive_parse_gated_delta_net_template(kernel_name)?;
    let h = read_param_i64(kernel_params, 7)?.max(0) as u64;
    let n_tokens = read_param_i64(kernel_params, 8)?.max(0) as u64;
    let n_seqs = read_param_i64(kernel_params, 9)?.max(0) as u64;
    let sq1 = read_param_i64(kernel_params, 10)?.max(0) as u64;
    let sq2 = read_param_i64(kernel_params, 11)?.max(0) as u64;
    let sq3 = read_param_i64(kernel_params, 12)?.max(0) as u64;
    let sv1 = read_param_i64(kernel_params, 13)?.max(0) as u64;
    let sv2 = read_param_i64(kernel_params, 14)?.max(0) as u64;
    let sv3 = read_param_i64(kernel_params, 15)?.max(0) as u64;
    let sb1 = read_param_i64(kernel_params, 16)?.max(0) as u64;
    let sb2 = read_param_i64(kernel_params, 17)?.max(0) as u64;
    let sb3 = read_param_i64(kernel_params, 18)?.max(0) as u64;
    let elem = std::mem::size_of::<f32>() as u64;

    let qk_elems = n_seqs
        .saturating_sub(1)
        .saturating_mul(sq3)
        .saturating_add(n_tokens.saturating_sub(1).saturating_mul(sq2))
        .saturating_add(h.saturating_sub(1).saturating_mul(sq1))
        .saturating_add(s_v);
    let v_elems = n_seqs
        .saturating_sub(1)
        .saturating_mul(sv3)
        .saturating_add(n_tokens.saturating_sub(1).saturating_mul(sv2))
        .saturating_add(h.saturating_sub(1).saturating_mul(sv1))
        .saturating_add(s_v);
    let gb_base = n_seqs
        .saturating_sub(1)
        .saturating_mul(sb3)
        .saturating_add(n_tokens.saturating_sub(1).saturating_mul(sb2))
        .saturating_add(h.saturating_sub(1).saturating_mul(sb1));
    let g_elems = if kda {
        gb_base.saturating_mul(s_v).saturating_add(s_v)
    } else {
        gb_base.saturating_add(1)
    };
    let beta_elems = gb_base.saturating_add(1);
    let state_elems = n_seqs
        .saturating_mul(h)
        .saturating_mul(s_v)
        .saturating_mul(s_v);
    let dst_elems = s_v
        .saturating_mul(h)
        .saturating_mul(n_tokens)
        .saturating_mul(n_seqs)
        .saturating_add(state_elems);

    let (bytes, flags) = match index {
        0 | 1 => (
            qk_elems.saturating_mul(elem),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        2 => (
            v_elems.saturating_mul(elem),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        3 => (
            g_elems.saturating_mul(elem),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        4 => (
            beta_elems.saturating_mul(elem),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        5 => (
            state_elems.saturating_mul(elem),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        6 => (
            dst_elems.saturating_mul(elem),
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        ),
        _ => return None,
    };
    let bytes = sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes)?;
    Some((bytes, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn read_param_u64(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<u64> {
    if kernel_params.is_null() {
        return None;
    }
    let param = *kernel_params.add(index);
    if param.is_null() || (param as usize) < 0x1_0000 {
        return None;
    }
    Some((param as *const u64).read_unaligned())
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn read_param_i32(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<i32> {
    if kernel_params.is_null() {
        return None;
    }
    let param = *kernel_params.add(index);
    if param.is_null() || (param as usize) < 0x1_0000 {
        return None;
    }
    Some((param as *const i32).read_unaligned())
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn read_param_u32(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<u32> {
    if kernel_params.is_null() {
        return None;
    }
    let param = *kernel_params.add(index);
    if param.is_null() || (param as usize) < 0x1_0000 {
        return None;
    }
    Some((param as *const u32).read_unaligned())
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn read_param_f32(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<f32> {
    if kernel_params.is_null() {
        return None;
    }
    let param = *kernel_params.add(index);
    if param.is_null() || (param as usize) < 0x1_0000 {
        return None;
    }
    Some((param as *const f32).read_unaligned())
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn read_param_i64(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<i64> {
    if kernel_params.is_null() {
        return None;
    }
    let param = *kernel_params.add(index);
    if param.is_null() || (param as usize) < 0x1_0000 {
        return None;
    }
    Some((param as *const i64).read_unaligned())
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn read_param_bool(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<bool> {
    if kernel_params.is_null() {
        return None;
    }
    let param = *kernel_params.add(index);
    if param.is_null() || (param as usize) < 0x1_0000 {
        return None;
    }
    Some((param as *const u8).read_unaligned() != 0)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn read_param_uint3_z(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<u32> {
    if kernel_params.is_null() {
        return None;
    }
    let param = *kernel_params.add(index);
    if param.is_null() || (param as usize) < 0x1_0000 {
        return None;
    }
    Some((param as *const u32).add(2).read_unaligned())
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn rmsnorm_named_offload_template_mode(kernel_name: &str) -> (bool, bool) {
    let name_lower = kernel_name.to_ascii_lowercase();
    if name_lower.contains("elb1elb1") || name_lower.contains(", true, true") {
        (true, true)
    } else if name_lower.contains("elb1elb0") || name_lower.contains(", true, false") {
        (true, false)
    } else if name_lower.contains("elb0elb0") || name_lower.contains(", false, false") {
        (false, false)
    } else {
        (false, false)
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn rmsnorm_named_offload_template_hidden(kernel_name: &str) -> Option<u64> {
    let pos = kernel_name
        .find("rms_norm_f32ILi")
        .or_else(|| kernel_name.find("rmsnorm_f32ILi"))?;
    let start = pos + kernel_name[pos..].find("ILi")? + 3;
    let digits: String = kernel_name[start..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn read_rmsnorm_named_offload_args(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
) -> Option<(
    *const ::core::ffi::c_void,
    *const ::core::ffi::c_void,
    *mut ::core::ffi::c_void,
    u64,
    u64,
    f32,
)> {
    let (do_multiply, do_add) = rmsnorm_named_offload_template_mode(kernel_name);
    if do_add {
        return None;
    }

    let x = read_param_u64(kernel_params, 0)? as *const ::core::ffi::c_void;
    let y = read_param_u64(kernel_params, 1)? as *mut ::core::ffi::c_void;
    let hidden_param = read_param_i32(kernel_params, 2).unwrap_or(0).max(0) as u64;
    let hidden = if hidden_param != 0 {
        hidden_param
    } else {
        rmsnorm_named_offload_template_hidden(kernel_name).unwrap_or(0)
    };
    let stride_row = read_param_i64(kernel_params, 3)
        .unwrap_or(hidden as i64)
        .max(0) as u64;
    let stride_channel = read_param_i64(kernel_params, 4)
        .unwrap_or(hidden as i64)
        .max(0) as u64;
    let stride_sample = read_param_i64(kernel_params, 5)
        .unwrap_or(hidden as i64)
        .max(0) as u64;
    let eps = read_param_f32(kernel_params, 6).unwrap_or(1e-5);

    if hidden == 0 {
        return None;
    }

    let grid_rows = grid_dim_x.max(1) as u64;
    let grid_channels = grid_dim_y.max(1) as u64;
    let grid_samples = grid_dim_z.max(1) as u64;
    let rows = grid_rows
        .saturating_mul(grid_channels)
        .saturating_mul(grid_samples)
        .max(1);

    let expected_stride_row = hidden;
    let expected_stride_channel = hidden.saturating_mul(grid_rows);
    let expected_stride_sample = expected_stride_channel.saturating_mul(grid_channels);
    let strict_strides = std::env::var("HETGPU_SIFIVE_RMSNORM_STRICT_STRIDES")
        .ok()
        .as_deref()
        == Some("1");
    let stride_mismatch = stride_row != expected_stride_row
        || stride_channel != expected_stride_channel
        || stride_sample != expected_stride_sample;
    if strict_strides && stride_mismatch {
        return None;
    }

    let weight = if do_multiply {
        let weight = read_param_u64(kernel_params, 7)? as *const ::core::ffi::c_void;
        let strict_weight_shape = std::env::var("HETGPU_SIFIVE_RMSNORM_STRICT_WEIGHT_SHAPE")
            .ok()
            .as_deref()
            == Some("1");
        if strict_weight_shape {
            let mul_ncols = read_param_uint3_z(kernel_params, 11)? as u64;
            let mul_nrows = read_param_uint3_z(kernel_params, 12)?;
            let mul_nchannels = read_param_uint3_z(kernel_params, 13)?;
            let mul_nsamples = read_param_uint3_z(kernel_params, 14)?;
            if mul_ncols != hidden || mul_nrows != 1 || mul_nchannels != 1 || mul_nsamples != 1 {
                return None;
            }
        }
        weight
    } else {
        ptr::null()
    };

    Some((x, weight, y, rows, hidden, eps))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_rmsnorm_f32_host_fallback(
    kernel_name: &str,
    x: *const ::core::ffi::c_void,
    weight: *const ::core::ffi::c_void,
    y: *mut ::core::ffi::c_void,
    rows: u64,
    hidden: u64,
    eps: f32,
) -> Option<cuda_types::cuda::CUresult> {
    use cuda_types::cuda::*;

    if sifive_rmsnorm_delivery_noop_enabled() {
        sifive_log_limited(
            &SIFIVE_NAMED_FAILOPEN_LOG_COUNT,
            "HETGPU_SIFIVE_NAMED_FAILOPEN_LOG_LIMIT",
            8,
            || {
                eprintln!(
                    "[SIFIVE Backend] delivery-noop RMSNorm '{}' rows={} hidden={} eps={}",
                    kernel_name, rows, hidden, eps
                );
            },
        );
        return Some(Ok(()));
    }

    if std::env::var("HETGPU_SIFIVE_RMSNORM_HOST_FALLBACK")
        .ok()
        .as_deref()
        == Some("0")
    {
        return Some(Err(CUerror::UNKNOWN));
    }

    let rows = usize::try_from(rows).ok()?;
    let hidden = usize::try_from(hidden).ok()?;
    let elem_count = rows.checked_mul(hidden)?;
    let bytes = elem_count.checked_mul(std::mem::size_of::<f32>())?;
    let weight_bytes = hidden.checked_mul(std::mem::size_of::<f32>())?;

    let x_addr = x as usize;
    let y_addr = y as usize;
    let weight_addr = weight as usize;
    if !sifive_host_or_cuda_alloc_has_bytes(x_addr as u64, bytes, false)
        || !sifive_host_or_cuda_alloc_has_bytes(y_addr as u64, bytes, true)
        || (!weight.is_null()
            && !sifive_host_or_cuda_alloc_has_bytes(weight_addr as u64, weight_bytes, false))
    {
        sifive_log_limited(
            &SIFIVE_NAMED_FAILOPEN_LOG_COUNT,
            "HETGPU_SIFIVE_RMSNORM_HOST_FALLBACK_LOG_LIMIT",
            2,
            || {
                eprintln!(
                    "[SIFIVE Backend] host-fallback RMSNorm '{}' rejected inaccessible host range x=0x{:x} w=0x{:x} y=0x{:x} rows={} hidden={}",
                    kernel_name, x_addr, weight_addr, y_addr, rows, hidden
                );
            },
        );
        return Some(Err(CUerror::UNKNOWN));
    }

    let x = x.cast::<f32>();
    let weight = weight.cast::<f32>();
    let y = y.cast::<f32>();
    for row in 0..rows {
        let base = row * hidden;
        let mut sumsq = 0.0f32;
        for col in 0..hidden {
            let v = x.add(base + col).read_unaligned();
            sumsq += v * v;
        }
        let scale = 1.0f32 / (sumsq / hidden as f32 + eps).sqrt();
        for col in 0..hidden {
            let xv = x.add(base + col).read_unaligned();
            let wv = if weight.is_null() {
                1.0
            } else {
                weight.add(col).read_unaligned()
            };
            y.add(base + col).write_unaligned(xv * scale * wv);
        }
    }

    if sifive_env_enabled_default("HETGPU_SIFIVE_RMSNORM_HOST_FALLBACK_LOG", false) {
        sifive_log_limited(
            &SIFIVE_NAMED_FAILOPEN_LOG_COUNT,
            "HETGPU_SIFIVE_RMSNORM_HOST_FALLBACK_LOG_LIMIT",
            2,
            || {
                eprintln!(
                    "[SIFIVE Backend] host-fallback RMSNorm '{}' rows={} hidden={} eps={}",
                    kernel_name, rows, hidden, eps
                );
            },
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[repr(C)]
#[derive(Copy, Clone)]
struct SifiveSoftMaxParams {
    nheads: i64,
    n_head_log2: u32,
    _pad0: u32,
    ncols: i64,
    nrows_x: i64,
    nrows_y: i64,
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb11: i64,
    nb12: i64,
    nb13: i64,
    ne12: i64,
    ne13: i64,
    scale: f32,
    max_bias: f32,
    m0: f32,
    m1: f32,
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn read_softmax_named_args(
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<(
    *const ::core::ffi::c_void,
    *const ::core::ffi::c_void,
    *const ::core::ffi::c_void,
    *mut ::core::ffi::c_void,
    SifiveSoftMaxParams,
)> {
    if kernel_params.is_null() {
        return None;
    }
    let x = read_param_u64(kernel_params, 0)? as *const ::core::ffi::c_void;
    let mask = read_param_u64(kernel_params, 1).unwrap_or(0) as *const ::core::ffi::c_void;
    let sinks = read_param_u64(kernel_params, 2).unwrap_or(0) as *const ::core::ffi::c_void;
    let dst = read_param_u64(kernel_params, 3)? as *mut ::core::ffi::c_void;
    let p_ptr = *kernel_params.add(4) as *const SifiveSoftMaxParams;
    if p_ptr.is_null()
        || !sifive_host_range_has_perms(
            p_ptr as usize,
            std::mem::size_of::<SifiveSoftMaxParams>(),
            false,
        )
    {
        return None;
    }
    Some((x, mask, sinks, dst, p_ptr.read_unaligned()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn read_pytorch_softmax_warp_forward_args(
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<(
    *const ::core::ffi::c_void,
    *mut ::core::ffi::c_void,
    u64,
    u64,
    u64,
    i32,
)> {
    if kernel_params.is_null() {
        return None;
    }

    let dst = read_param_u64(kernel_params, 0)? as *mut ::core::ffi::c_void;
    let src = read_param_u64(kernel_params, 1)? as *const ::core::ffi::c_void;
    let batch_size = read_param_i32(kernel_params, 2)?.max(0) as u64;
    let stride = read_param_i32(kernel_params, 3)?.max(0) as u64;
    let element_count = read_param_i32(kernel_params, 4)?.max(0) as u64;
    if src.is_null() || dst.is_null() || batch_size == 0 || element_count == 0 {
        return None;
    }
    let stride = stride.max(element_count);

    let rows = batch_size;
    let cols = element_count;
    let bytes = rows
        .checked_mul(stride)?
        .checked_mul(std::mem::size_of::<f32>() as u64)?;
    let bytes = usize::try_from(bytes).ok()?;
    if !sifive_host_or_cuda_alloc_has_bytes(src as u64, bytes, false)
        || !sifive_host_or_cuda_alloc_has_bytes(dst as u64, bytes, true)
    {
        eprintln!(
            "[SIFIVE Backend] PyTorch softmax_warp_forward rejected inaccessible range src=0x{:x} dst=0x{:x} rows={} cols={} stride={}",
            src as u64, dst as u64, rows, cols, stride
        );
        return None;
    }

    Some((
        src,
        dst,
        rows,
        cols,
        stride,
        sifive_runtime_sys::SifiveDataType::Float32 as i32,
    ))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
const SIFIVE_PYTORCH_SOFTMAX_ELF_SYMBOL: &str = "sifive_pytorch_softmax_warp_forward";

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
const SIFIVE_PYTORCH_SOFTMAX_ELF_SOURCE: &str = r#"
typedef unsigned long u64;
typedef unsigned int u32;

struct KernelParamCell {
    u64 lo;
    u64 hi;
};

static u64 sifive_cell_lo(u64 cell_addr) {
    volatile const struct KernelParamCell *cell =
        (volatile const struct KernelParamCell *)cell_addr;
    return cell ? cell->lo : 0UL;
}

static float sifive_fast_expf(float x) {
    if (x < -80.0f) return 0.0f;
    if (x > 80.0f) x = 80.0f;
    union {
        u32 i;
        float f;
    } v;
    float y = x * 1.4426950408889634f + 127.0f;
    if (y < 0.0f) y = 0.0f;
    if (y > 255.0f) y = 255.0f;
    v.i = (u32)(y * 8388608.0f);
    return v.f;
}

__attribute__((visibility("default")))
void sifive_pytorch_softmax_warp_forward(u64 dst_cell,
                                       u64 src_cell,
                                       u64 rows_cell,
                                       u64 cols_cell,
                                       u64 stride_cell) {
    float *dst = (float *)(sifive_cell_lo(dst_cell));
    const float *src = (const float *)(sifive_cell_lo(src_cell));
    u64 rows = sifive_cell_lo(rows_cell);
    u64 cols = sifive_cell_lo(cols_cell);
    u64 stride = sifive_cell_lo(stride_cell);
    if (!dst || !src || rows == 0UL || cols == 0UL) return;
    if (stride < cols) stride = cols;

    for (u64 row = 0; row < rows; row++) {
        u64 base = row * stride;
        float max_v = src[base];
        for (u64 col = 1; col < cols; col++) {
            float v = src[base + col];
            if (v > max_v) max_v = v;
        }
        float sum = 0.0f;
        for (u64 col = 0; col < cols; col++) {
            float e = sifive_fast_expf(src[base + col] - max_v);
            dst[base + col] = e;
            sum += e;
        }
        if (sum == 0.0f) sum = 1.0f;
        float inv = 1.0f / sum;
        for (u64 col = 0; col < cols; col++) {
            dst[base + col] *= inv;
        }
    }
}
"#;

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_get_pytorch_softmax_elf_kernel(
    dev_id: i32,
) -> Option<*mut sifive_runtime_sys::sifive_Kernel> {
    let dev_id = if (0..4).contains(&dev_id) {
        dev_id as usize
    } else {
        0
    };
    let cache = SIFIVE_PYTORCH_SOFTMAX_ELF_KERNELS
        .get_or_init(|| std::sync::Mutex::new([None, None, None, None]));
    let mut guard = cache.lock().ok()?;
    if let Some(handles) = guard[dev_id] {
        return Some(handles.kernel as *mut sifive_runtime_sys::sifive_Kernel);
    }

    let device = sifive_runtime_sys::sifive_CreateDevice(dev_id as u32);
    if device.is_null() {
        eprintln!(
            "[SIFIVE Backend] PyTorch softmax ELF failed to create sifive{} device",
            dev_id
        );
        return None;
    }
    let program = sifive_runtime_sys::sifive_CreateProgram();
    if program.is_null() {
        eprintln!("[SIFIVE Backend] PyTorch softmax ELF failed to create program");
        return None;
    }

    let source = SIFIVE_PYTORCH_SOFTMAX_ELF_SOURCE.as_bytes();
    let workdir =
        std::path::Path::new("/home/ubuntu/Documents/hetGPU_sifive/target/sifive_named_kernels");
    if let Err(err) = std::fs::create_dir_all(workdir) {
        eprintln!(
            "[SIFIVE Backend] PyTorch softmax ELF failed to create source dir {}: {}",
            workdir.display(),
            err
        );
        return None;
    }
    let source_path = workdir.join("sifive_pytorch_softmax_warp_forward.c");
    if let Err(err) = std::fs::write(&source_path, source) {
        eprintln!(
            "[SIFIVE Backend] PyTorch softmax ELF failed to write source {}: {}",
            source_path.display(),
            err
        );
        return None;
    }
    let source_name = match std::ffi::CString::new(source_path.to_string_lossy().as_bytes()) {
        Ok(path) => path,
        Err(_) => return None,
    };
    let workdir_c = match std::ffi::CString::new(workdir.to_string_lossy().as_bytes()) {
        Ok(path) => path,
        Err(_) => return None,
    };
    let rc = sifive_runtime_sys::sifive_LoadProgramSource(
        program,
        std::ptr::null(),
        source_name.as_ptr(),
        source.as_ptr(),
        source.len() as u64,
        workdir_c.as_ptr(),
        std::ptr::null(),
        0,
        std::ptr::null(),
        0,
    );
    if rc != sifive_runtime_sys::sifive_Result_Success {
        let compile_error = program
            .as_ref()
            .and_then(|p| p.compile_error.as_deref())
            .map(str::to_owned)
            .unwrap_or_default();
        eprintln!(
            "[SIFIVE Backend] PyTorch softmax ELF compile failed rc={} {}",
            rc, compile_error
        );
        return None;
    }

    let kernel_name = std::ffi::CString::new(SIFIVE_PYTORCH_SOFTMAX_ELF_SYMBOL).ok()?;
    let kernel =
        sifive_runtime_sys::sifive_CreateKernelOnDevice(program, device, kernel_name.as_ptr());
    if kernel.is_null() {
        eprintln!("[SIFIVE Backend] PyTorch softmax ELF failed to create kernel handle");
        return None;
    }
    guard[dev_id] = Some(SifiveCachedKernelHandles {
        device: device as usize,
        program: program as usize,
        kernel: kernel as usize,
    });
    Some(kernel)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_push_softmax_elf_arg(
    kernel: *mut sifive_runtime_sys::sifive_Kernel,
    index: u32,
    kind: u32,
    value: u64,
    binding: Option<(u64, u32)>,
) -> bool {
    let record = sifive_runtime_sys::SifiveKernelArgRecord {
        kind,
        size: 8,
        flags: 0,
        reserved: 0,
        value,
        value_hi: 0,
    };
    if sifive_runtime_sys::sifive_KernelPushArgRecord(kernel, &record)
        != sifive_runtime_sys::sifive_Result_Success
    {
        return false;
    }
    if let Some((size, flags)) = binding {
        let (addr, flags) =
            if let Some(phys) = super::memory::sifive_shared_ddr_physical_addr(value) {
                (
                    phys,
                    flags | sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_DEVICE_PHYS,
                )
            } else {
                (value, flags)
            };
        let binding = sifive_runtime_sys::SifiveKernelBufferBinding {
            arg_index: index,
            flags,
            addr,
            size,
        };
        if sifive_runtime_sys::sifive_KernelAddBufferBinding(kernel, &binding)
            != sifive_runtime_sys::sifive_Result_Success
        {
            return false;
        }
    }
    true
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn launch_pytorch_softmax_warp_forward_elf(
    dev_id: i32,
    src: *const ::core::ffi::c_void,
    dst: *mut ::core::ffi::c_void,
    rows: u64,
    cols: u64,
    stride: u64,
) -> i32 {
    let Some(kernel) = sifive_get_pytorch_softmax_elf_kernel(dev_id) else {
        return -1;
    };
    let bytes = match rows
        .checked_mul(stride.max(cols))
        .and_then(|v| v.checked_mul(std::mem::size_of::<f32>() as u64))
    {
        Some(bytes) => bytes,
        None => return -1,
    };
    if sifive_runtime_sys::sifive_KernelClearLaunchState(kernel)
        != sifive_runtime_sys::sifive_Result_Success
    {
        return -1;
    }
    if !sifive_push_softmax_elf_arg(
        kernel,
        0,
        sifive_runtime_sys::SIFIVE_KERNEL_ARG_KIND_POINTER,
        dst as u64,
        Some((
            bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        )),
    ) || !sifive_push_softmax_elf_arg(
        kernel,
        1,
        sifive_runtime_sys::SIFIVE_KERNEL_ARG_KIND_POINTER,
        src as u64,
        Some((
            bytes,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        )),
    ) || !sifive_push_softmax_elf_arg(
        kernel,
        2,
        sifive_runtime_sys::SIFIVE_KERNEL_ARG_KIND_SCALAR,
        rows,
        None,
    ) || !sifive_push_softmax_elf_arg(
        kernel,
        3,
        sifive_runtime_sys::SIFIVE_KERNEL_ARG_KIND_SCALAR,
        cols,
        None,
    ) || !sifive_push_softmax_elf_arg(
        kernel,
        4,
        sifive_runtime_sys::SIFIVE_KERNEL_ARG_KIND_SCALAR,
        stride,
        None,
    ) {
        return -1;
    }

    let rc = sifive_runtime_sys::sifive_LaunchKernel(
        kernel,
        rows.min(u32::MAX as u64) as u32,
        1,
        1,
        1,
        1,
        1,
    );
    if rc == sifive_runtime_sys::sifive_Result_Success {
        0
    } else {
        rc
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_softmax_binding_metadata(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<(u64, u32)> {
    let (_x, mask, sinks, _dst, p) = read_softmax_named_args(kernel_params)?;
    let ncols = p.ncols.max(1) as u64;
    let ne01 = p.ne01.max(1) as u64;
    let ne02 = p.ne02.max(1) as u64;
    let ne03 = p.ne03.max(1) as u64;
    let rows = ne01.checked_mul(ne02)?.checked_mul(ne03)?;

    let (bytes, flags) = match index {
        0 => (
            rows.checked_mul(ncols)?
                .checked_mul(std::mem::size_of::<f32>() as u64)?,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        1 if !mask.is_null() => {
            let ne12 = p.ne12.max(1) as i128;
            let ne13 = p.ne13.max(1) as i128;
            let max_mask_off = ((ne01.saturating_sub(1)) as i128)
                .checked_mul(p.nb11 as i128)?
                .checked_add(((ne12 - 1) as i128).checked_mul(p.nb12 as i128)?)?
                .checked_add(((ne13 - 1) as i128).checked_mul(p.nb13 as i128)?)?;
            if max_mask_off < 0 {
                return None;
            }
            let mask_elem_size = if kernel_name.contains("6__half") {
                2u64
            } else {
                4u64
            };
            (
                (max_mask_off as u64).checked_add(ncols.checked_mul(mask_elem_size)?)?,
                sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
            )
        }
        2 if !sinks.is_null() => (
            ne02.checked_mul(std::mem::size_of::<f32>() as u64)?,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT,
        ),
        3 => (
            rows.checked_mul(ncols)?
                .checked_mul(std::mem::size_of::<f32>() as u64)?,
            sifive_runtime_sys::SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT,
        ),
        _ => return None,
    };
    let bytes = sifive_binding_bytes_for_host_ptr(kernel_params, index, bytes)?;
    Some((bytes, flags))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_alibi_slope(max_bias: f32, h: u32, n_head_log2: u32, m0: f32, m1: f32) -> f32 {
    if max_bias <= 0.0 {
        return 1.0;
    }
    let (base, exph) = if h < n_head_log2 {
        (m0, h + 1)
    } else {
        (m1, 2 * (h - n_head_log2) + 1)
    };
    base.powi(exph as i32)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut frac_norm = frac;
            let mut exp_norm = -14;
            while (frac_norm & 0x0400) == 0 {
                frac_norm <<= 1;
                exp_norm -= 1;
            }
            frac_norm &= 0x03ff;
            sign | (((exp_norm + 127) as u32) << 23) | (frac_norm << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        sign | (((exp - 15 + 127) as u32) << 23) | (frac << 13)
    };
    f32::from_bits(out)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_softmax_f32_host_fallback(
    kernel_name: &str,
    x: *const ::core::ffi::c_void,
    mask: *const ::core::ffi::c_void,
    sinks: *const ::core::ffi::c_void,
    dst: *mut ::core::ffi::c_void,
    p: SifiveSoftMaxParams,
    mask_is_f16: bool,
) -> Option<cuda_types::cuda::CUresult> {
    use cuda_types::cuda::*;

    if std::env::var("HETGPU_SIFIVE_SOFTMAX_HOST_FALLBACK")
        .ok()
        .as_deref()
        == Some("0")
    {
        return Some(Err(CUerror::UNKNOWN));
    }

    let ncols = usize::try_from(p.ncols).ok()?;
    let ne01 = usize::try_from(p.ne01).ok()?;
    let ne02 = usize::try_from(p.ne02).ok()?;
    let ne03 = usize::try_from(p.ne03).ok()?;
    let rows = ne01.checked_mul(ne02)?.checked_mul(ne03)?;
    if ncols == 0 || rows == 0 {
        return Some(Ok(()));
    }
    let bytes = rows
        .checked_mul(ncols)?
        .checked_mul(std::mem::size_of::<f32>())?;
    let x_addr = x as u64;
    let dst_addr = dst as u64;
    if !sifive_host_or_cuda_alloc_has_bytes(x_addr, bytes, false)
        || !sifive_host_or_cuda_alloc_has_bytes(dst_addr, bytes, true)
    {
        eprintln!(
            "[SIFIVE Backend] host-fallback softmax '{}' rejected inaccessible x/dst range x=0x{:x} dst=0x{:x} rows={} cols={}",
            kernel_name, x_addr, dst_addr, rows, ncols
        );
        return Some(Err(CUerror::UNKNOWN));
    }
    if !sinks.is_null() {
        let sinks_bytes = ne02.checked_mul(std::mem::size_of::<f32>())?;
        if !sifive_host_or_cuda_alloc_has_bytes(sinks as u64, sinks_bytes, false) {
            return Some(Err(CUerror::UNKNOWN));
        }
    }

    let x = x.cast::<f32>();
    let dst = dst.cast::<f32>();
    let sinks = sinks.cast::<f32>();
    let mask_elem_size = if mask_is_f16 { 2usize } else { 4usize };
    if !mask.is_null() {
        let ne12 = usize::try_from(p.ne12).ok()?.max(1);
        let ne13 = usize::try_from(p.ne13).ok()?.max(1);
        let max_mask_off = ((ne01.saturating_sub(1)) as i128)
            .checked_mul(p.nb11 as i128)?
            .checked_add(((ne12.saturating_sub(1)) as i128).checked_mul(p.nb12 as i128)?)?
            .checked_add(((ne13.saturating_sub(1)) as i128).checked_mul(p.nb13 as i128)?)?;
        if max_mask_off < 0 {
            return Some(Err(CUerror::UNKNOWN));
        }
        let mask_bytes = usize::try_from(max_mask_off)
            .ok()?
            .checked_add(ncols.checked_mul(mask_elem_size)?)?;
        if !sifive_host_or_cuda_alloc_has_bytes(mask as u64, mask_bytes, false) {
            return Some(Err(CUerror::UNKNOWN));
        }
    }
    let mut vals = vec![0.0f32; ncols];

    for i03 in 0..ne03 {
        for i02 in 0..ne02 {
            for i01 in 0..ne01 {
                let rowx = i01 + i02 * ne01 + i03 * ne01 * ne02;
                let x_row = x.add(rowx * ncols);
                let dst_row = dst.add(rowx * ncols);
                let mut max_val = if sinks.is_null() {
                    f32::NEG_INFINITY
                } else {
                    sinks.add(i02).read_unaligned()
                };
                let slope = sifive_alibi_slope(p.max_bias, i02 as u32, p.n_head_log2, p.m0, p.m1);
                let mask_row = if mask.is_null() {
                    std::ptr::null::<u8>()
                } else {
                    let ne12 = usize::try_from(p.ne12).ok()?.max(1);
                    let ne13 = usize::try_from(p.ne13).ok()?.max(1);
                    let i12 = i02 % ne12;
                    let i13 = i03 % ne13;
                    let byte_off = (i01 as i64)
                        .checked_mul(p.nb11)?
                        .checked_add((i12 as i64).checked_mul(p.nb12)?)?
                        .checked_add((i13 as i64).checked_mul(p.nb13)?)?;
                    if byte_off < 0 {
                        return Some(Err(CUerror::UNKNOWN));
                    }
                    mask.cast::<u8>().add(byte_off as usize)
                };

                for col in 0..ncols {
                    let mask_v = if mask_row.is_null() {
                        0.0
                    } else if mask_is_f16 {
                        sifive_f16_to_f32((mask_row as *const u16).add(col).read_unaligned())
                    } else {
                        (mask_row as *const f32).add(col).read_unaligned()
                    };
                    let val = x_row.add(col).read_unaligned() * p.scale + slope * mask_v;
                    vals[col] = val;
                    max_val = max_val.max(val);
                }
                let mut sum = if sinks.is_null() {
                    0.0
                } else {
                    (sinks.add(i02).read_unaligned() - max_val).exp()
                };
                for col in 0..ncols {
                    let v = (vals[col] - max_val).exp();
                    vals[col] = v;
                    sum += v;
                }
                let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for col in 0..ncols {
                    dst_row.add(col).write_unaligned(vals[col] * inv_sum);
                }
            }
        }
    }

    eprintln!(
        "[SIFIVE Backend] host-fallback softmax '{}' rows={} cols={} mask={} sinks={}",
        kernel_name,
        rows,
        ncols,
        !mask.is_null(),
        !sinks.is_null()
    );
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_bin_bcast_f32_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    #[derive(Copy, Clone)]
    enum Op {
        Repeat,
        Add,
        Sub,
        Mul,
        Div,
    }

    let op = if kernel_name.contains("op_repeatff") {
        Op::Repeat
    } else if kernel_name.contains("op_addff") {
        Op::Add
    } else if kernel_name.contains("op_subff") {
        Op::Sub
    } else if kernel_name.contains("op_mulff") {
        Op::Mul
    } else if kernel_name.contains("op_divff") {
        Op::Div
    } else {
        return None;
    };

    if kernel_name.contains("k_bin_bcast_unravel") || !kernel_name.contains("EEfffJ") {
        return None;
    }
    let (src0_elem, src1_elem, dst_elem) = sifive_bin_bcast_element_sizes(kernel_name);
    if src0_elem != 4 || src1_elem != 4 || dst_elem != 4 {
        return None;
    }

    let src0_addr = read_param_u64(kernel_params, 0).unwrap_or(0);
    let src1_addr = read_param_u64(kernel_params, 1).unwrap_or(0);
    let dst_addr = read_param_u64(kernel_params, 2)?;
    let src0 = if src0_addr == 0 {
        std::ptr::null()
    } else {
        sifive_host_ptr::<f32>(src0_addr)? as *const f32
    };
    let src1 = if src1_addr == 0 {
        std::ptr::null()
    } else {
        sifive_host_ptr::<f32>(src1_addr)? as *const f32
    };
    let dst = sifive_host_ptr::<f32>(dst_addr)?;

    let ne0 = read_param_i32(kernel_params, 3)?.max(0) as usize;
    let ne1 = read_param_i32(kernel_params, 4)?.max(0) as usize;
    let ne2 = read_param_i32(kernel_params, 5)?.max(0) as usize;
    let ne3 = read_param_uint3_value(kernel_params, 6)?;
    let ne10 = read_param_uint3_value(kernel_params, 7)?;
    let ne11 = read_param_uint3_value(kernel_params, 8)?;
    let ne12 = read_param_uint3_value(kernel_params, 9)?;
    let ne13 = read_param_uint3_value(kernel_params, 10)?;
    let s1 = read_param_i32(kernel_params, 11)?.max(0) as usize;
    let s2 = read_param_i32(kernel_params, 12)?.max(0) as usize;
    let s3 = read_param_i32(kernel_params, 13)?.max(0) as usize;
    let s00 = read_param_i32(kernel_params, 14)?.max(0) as usize;
    let s01 = read_param_i32(kernel_params, 15)?.max(0) as usize;
    let s02 = read_param_i32(kernel_params, 16)?.max(0) as usize;
    let s03 = read_param_i32(kernel_params, 17)?.max(0) as usize;
    let s10 = read_param_i32(kernel_params, 18)?.max(0) as usize;
    let s11 = read_param_i32(kernel_params, 19)?.max(0) as usize;
    let s12 = read_param_i32(kernel_params, 20)?.max(0) as usize;
    let s13 = read_param_i32(kernel_params, 21)?.max(0) as usize;

    let dim = |v: sifive_runtime_sys::HetgpuSifiveUint3| -> usize { v.z.max(1) as usize };
    let ne3z = dim(ne3);
    let ne10z = dim(ne10);
    let ne11z = dim(ne11);
    let ne12z = dim(ne12);
    let ne13z = dim(ne13);
    if ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3z == 0 {
        return Some(Ok(()));
    }

    let fused = sifive_bin_bcast_fuse_count(kernel_name);
    let mut src1s = Vec::with_capacity(fused.max(1));
    if fused != 0 {
        for i in 0..fused {
            let ptr = sifive_host_ptr::<f32>(read_param_u64(kernel_params, 22 + i)?)? as *const f32;
            src1s.push(ptr);
        }
    } else if !src1.is_null() {
        src1s.push(src1);
    } else {
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }

    let checked_extent =
        |a: usize, sa: usize, b: usize, sb: usize, c: usize, sc: usize, d: usize, sd: usize| {
            a.checked_sub(1)
                .and_then(|v| v.checked_mul(sa))
                .and_then(|v| {
                    b.checked_sub(1)
                        .and_then(|w| w.checked_mul(sb))
                        .and_then(|w| v.checked_add(w))
                })
                .and_then(|v| {
                    c.checked_sub(1)
                        .and_then(|w| w.checked_mul(sc))
                        .and_then(|w| v.checked_add(w))
                })
                .and_then(|v| {
                    d.checked_sub(1)
                        .and_then(|w| w.checked_mul(sd))
                        .and_then(|w| v.checked_add(w))
                })
                .and_then(|v| v.checked_add(1))
        };
    let dst_elems = checked_extent(ne3z, s3, ne2, s2, ne1, s1, ne0, 1)?;
    let src0_elems = checked_extent(ne3z, s03, ne2, s02, ne1, s01, ne0, s00)?;
    let src1_i0 = ne0.min(ne10z).max(1);
    let src1_elems = checked_extent(ne13z, s13, ne12z, s12, ne11z, s11, src1_i0, s10)?;
    if !sifive_cuda_alloc_has_elems(dst as *const f32, dst_elems)
        || (!src0.is_null() && !sifive_cuda_alloc_has_elems(src0, src0_elems))
        || src1s
            .iter()
            .any(|&ptr| !sifive_cuda_alloc_has_elems(ptr, src1_elems))
    {
        eprintln!(
            "[SIFIVE Backend] host-fallback bin_bcast '{}' rejected range dst={:p} src0={:p} dst_elems={} src0_elems={} src1_elems={}",
            kernel_name, dst, src0, dst_elems, src0_elems, src1_elems
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }

    let apply = |op: Op, a: f32, b: f32| -> f32 {
        match op {
            Op::Repeat => b,
            Op::Add => a + b,
            Op::Sub => a - b,
            Op::Mul => a * b,
            Op::Div => a / b,
        }
    };
    for row in 0..(ne1 * ne2 * ne3z) {
        let i1 = row % ne1;
        let t = row / ne1;
        let i2 = t % ne2;
        let i3 = t / ne2;
        let i11 = i1 % ne11z;
        let i12 = i2 % ne12z;
        let i13 = i3 % ne13z;
        let src0_base = i3 * s03 + i2 * s02 + i1 * s01;
        let src1_base = i13 * s13 + i12 * s12 + i11 * s11;
        let dst_base = i3 * s3 + i2 * s2 + i1 * s1;
        for i0 in 0..ne0 {
            let i10 = i0 % ne10z;
            let src1_off = src1_base + i10 * s10;
            let mut value = if src0.is_null() {
                0.0
            } else {
                src0.add(src0_base + i0 * s00).read_unaligned()
            };
            for &rhs in &src1s {
                value = apply(op, value, rhs.add(src1_off).read_unaligned());
            }
            dst.add(dst_base + i0).write_unaligned(value);
        }
    }

    if std::env::var("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!(
            "[SIFIVE Backend] host-fallback bin_bcast '{}' elems={} fused={}",
            kernel_name,
            ne0 * ne1 * ne2 * ne3z,
            src1s.len()
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_compute_batched_ptrs_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    let src0 = read_param_u64(kernel_params, 0)?;
    let src1 = read_param_u64(kernel_params, 1)?;
    let dst = read_param_u64(kernel_params, 2)?;
    let ptrs_src = read_param_u64(kernel_params, 3)?;
    let ptrs_dst = read_param_u64(kernel_params, 4)?;
    let ne12 = read_param_i64(kernel_params, 5)?.max(0) as usize;
    let ne13 = read_param_i64(kernel_params, 6)?.max(0) as usize;
    let ne23 = read_param_i64(kernel_params, 7)?.max(0) as usize;
    let nb02 = read_param_u64(kernel_params, 8)? as usize;
    let nb03 = read_param_u64(kernel_params, 9)? as usize;
    let nb12 = read_param_u64(kernel_params, 10)? as usize;
    let nb13 = read_param_u64(kernel_params, 11)? as usize;
    let nbd2 = read_param_u64(kernel_params, 12)? as usize;
    let nbd3 = read_param_u64(kernel_params, 13)? as usize;
    let r2 = read_param_i64(kernel_params, 14)?.max(1) as usize;
    let r3 = read_param_i64(kernel_params, 15)?.max(1) as usize;

    if ne12 == 0 || ne13 == 0 || ne23 == 0 {
        return Some(Ok(()));
    }

    /*
     * This fallback runs on the host and populates CUDA pointer tables that are
     * themselves CUDA allocations.  In SIFIVE shared-DDR mode those allocation
     * pointers are mmap'ed host addresses backed by the shared window.  Do not
     * translate the table address to a physical SIFIVE address here: physical
     * addresses are for jobd/kernel consumption and are not valid AP pointers.
     * Keep the table entries as CUDA pointers too; cublas_shim resolves each
     * per-batch A/B/C pointer when it submits the GEMM tile.
     */
    let ptrs_src_host = ptrs_src as *mut u64;
    let ptrs_dst_host = ptrs_dst as *mut u64;

    if ptrs_src_host.is_null() || ptrs_dst_host.is_null() {
        eprintln!(
            "[SIFIVE Backend] compute_batched_ptrs '{}' could not resolve pointer tables",
            kernel_name
        );
        return sifive_named_assume_success(
            "compute_batched_ptrs pointer tables could not be resolved",
            kernel_name,
        );
    }

    let table_count = ne12.checked_mul(ne13)?;
    let ptrs_src_count = ne23.checked_add(table_count)?;
    if !sifive_cuda_alloc_has_elems(ptrs_src_host as *const u64, ptrs_src_count)
        || !sifive_cuda_alloc_has_elems(ptrs_dst_host as *const u64, table_count)
    {
        eprintln!(
            "[SIFIVE Backend] compute_batched_ptrs '{}' rejected out-of-allocation pointer tables ptrs_src={:p} ptrs_dst={:p} src_count={} dst_count={} ne12={} ne13={} ne23={}",
            kernel_name,
            ptrs_src_host,
            ptrs_dst_host,
            ptrs_src_count,
            table_count,
            ne12,
            ne13,
            ne23
        );
        return sifive_named_assume_success(
            "compute_batched_ptrs pointer table range check failed",
            kernel_name,
        );
    }

    for i13 in 0..ne13 {
        for i12 in 0..ne12 {
            let i03 = i13 / r3;
            let i02 = i12 / r2;
            let index = i12 + i13 * ne12;
            ptrs_src_host
                .add(index)
                .write_unaligned(src0.saturating_add((i02 * nb02 + i03 * nb03) as u64));
            ptrs_src_host
                .add(ne23 + index)
                .write_unaligned(src1.saturating_add((i12 * nb12 + i13 * nb13) as u64));
            ptrs_dst_host
                .add(index)
                .write_unaligned(dst.saturating_add((i12 * nbd2 + i13 * nbd3) as u64));
        }
    }

    if std::env::var("HETGPU_SIFIVE_LOG_COMPUTE_BATCHED_PTRS")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!(
            "[SIFIVE Backend] handled compute_batched_ptrs '{}' ne12={} ne13={} ne23={}",
            kernel_name, ne12, ne13, ne23
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn sifive_host_ptr<T>(addr: u64) -> Option<*mut T> {
    if addr == 0 {
        return None;
    }
    let ptr = addr as usize;
    if !sifive_looks_like_host_param_addr(ptr) {
        return None;
    }
    Some(ptr as *mut T)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_cuda_alloc_has_bytes(addr: u64, bytes: usize) -> bool {
    if bytes == 0 {
        return true;
    }
    super::memory::sifive_allocation_remaining_addr(addr)
        .map(|remaining| remaining >= bytes)
        .unwrap_or(false)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_host_or_cuda_alloc_has_bytes(addr: u64, bytes: usize, need_write: bool) -> bool {
    if sifive_cuda_alloc_has_bytes(addr, bytes) {
        return true;
    }
    let Ok(host_addr) = usize::try_from(addr) else {
        return false;
    };
    sifive_host_range_has_perms(host_addr, bytes, need_write)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_cuda_alloc_has_elems<T>(ptr: *const T, elems: usize) -> bool {
    elems
        .checked_mul(std::mem::size_of::<T>())
        .map(|bytes| sifive_cuda_alloc_has_bytes(ptr as u64, bytes))
        .unwrap_or(false)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_vectorized_gather_host_copy(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    let out_addr = read_param_u64(kernel_params, 0)?;
    let inp_addr = read_param_u64(kernel_params, 1)?;
    let idx_addr = read_param_u64(kernel_params, 2)?;
    let num_ind = read_param_i32(kernel_params, 3)?.max(0) as usize;
    let slice_size = read_param_i64(kernel_params, 4)?.max(0) as usize;
    let ind_dim_size = read_param_i64(kernel_params, 5)?.max(0);
    let inp_stride = read_param_i64(kernel_params, 6)?;
    let out_stride = read_param_i64(kernel_params, 7)?;
    let allow_neg_indices = read_param_bool(kernel_params, 8).unwrap_or(false);

    if num_ind == 0 || slice_size == 0 {
        return Some(Ok(()));
    }
    if inp_stride < 0 || out_stride < 0 || ind_dim_size <= 0 {
        eprintln!(
            "[SIFIVE Backend] vectorized_gather '{}' rejected invalid shape num_ind={} slice={} ind_dim={} inp_stride={} out_stride={}",
            kernel_name, num_ind, slice_size, ind_dim_size, inp_stride, out_stride
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }

    let index_is_i64 = !kernel_name.contains("ILi16EiE");
    let idx_elem_size = if index_is_i64 {
        std::mem::size_of::<i64>()
    } else {
        std::mem::size_of::<i32>()
    };
    let Some(idx_bytes) = num_ind.checked_mul(idx_elem_size) else {
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    };
    let out_stride = out_stride as usize;
    let inp_stride = inp_stride as usize;
    let Some(out_last_off) = num_ind
        .checked_sub(1)
        .and_then(|last| last.checked_mul(out_stride))
    else {
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    };
    let Some(out_bytes) = out_last_off.checked_add(slice_size) else {
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    };

    if !sifive_host_or_cuda_alloc_has_bytes(idx_addr, idx_bytes, false)
        || !sifive_host_or_cuda_alloc_has_bytes(out_addr, out_bytes, true)
    {
        eprintln!(
            "[SIFIVE Backend] vectorized_gather '{}' rejected out/idx allocation out=0x{:x} out_bytes={} idx=0x{:x} idx_bytes={}",
            kernel_name, out_addr, out_bytes, idx_addr, idx_bytes
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }

    let mut max_src_end = 0usize;
    for i in 0..num_ind {
        let mut ind = if index_is_i64 {
            (idx_addr as *const i64).add(i).read_unaligned()
        } else {
            (idx_addr as *const i32).add(i).read_unaligned() as i64
        };
        if allow_neg_indices && ind < 0 {
            ind += ind_dim_size;
        }
        if ind < 0 || ind >= ind_dim_size {
            eprintln!(
                "[SIFIVE Backend] vectorized_gather '{}' index {} out of bounds at {} (dim={})",
                kernel_name, ind, i, ind_dim_size
            );
            return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
        }
        let Some(src_off) = (ind as usize).checked_mul(inp_stride) else {
            return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
        };
        let Some(src_end) = src_off.checked_add(slice_size) else {
            return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
        };
        max_src_end = max_src_end.max(src_end);
    }

    if !sifive_host_or_cuda_alloc_has_bytes(inp_addr, max_src_end, false) {
        eprintln!(
            "[SIFIVE Backend] vectorized_gather '{}' rejected input allocation inp=0x{:x} bytes={}",
            kernel_name, inp_addr, max_src_end
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }

    let inp = inp_addr as *const u8;
    let out = out_addr as *mut u8;
    for i in 0..num_ind {
        let mut ind = if index_is_i64 {
            (idx_addr as *const i64).add(i).read_unaligned()
        } else {
            (idx_addr as *const i32).add(i).read_unaligned() as i64
        };
        if allow_neg_indices && ind < 0 {
            ind += ind_dim_size;
        }
        let src_off = (ind as usize).saturating_mul(inp_stride);
        let dst_off = i.saturating_mul(out_stride);
        std::ptr::copy_nonoverlapping(inp.add(src_off), out.add(dst_off), slice_size);
    }

    if sifive_env_truthy("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS") {
        eprintln!(
            "[SIFIVE Backend] handled vectorized_gather '{}' on host-copy path num_ind={} slice={} ind_dim={} idx={} ",
            kernel_name,
            num_ind,
            slice_size,
            ind_dim_size,
            if index_is_i64 { "i64" } else { "i32" }
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn read_param_data_pair(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<(u64, u64)> {
    if kernel_params.is_null() {
        return None;
    }
    let param = *kernel_params.add(index);
    if param.is_null() || (param as usize) < 0x1_0000 {
        return None;
    }
    if !sifive_host_range_has_perms(param as usize, 2 * std::mem::size_of::<u64>(), false) {
        return None;
    }
    let data = param as *const u64;
    let out = data.read_unaligned();
    let inp = data.add(1).read_unaligned();
    if out < 0x1_0000 || inp < 0x1_0000 {
        return None;
    }
    Some((out, inp))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn read_param_data_single(
    kernel_params: *mut *mut ::core::ffi::c_void,
    index: usize,
) -> Option<u64> {
    if kernel_params.is_null() {
        return None;
    }
    let param = *kernel_params.add(index);
    if param.is_null() || (param as usize) < 0x1_0000 {
        return None;
    }
    if !sifive_host_range_has_perms(param as usize, std::mem::size_of::<u64>(), false) {
        return None;
    }
    let out = (param as *const u64).read_unaligned();
    if out < 0x1_0000 {
        return None;
    }
    Some(out)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn find_tensoriterator_data_pair(
    kernel_params: *mut *mut ::core::ffi::c_void,
    bytes: usize,
) -> Option<(usize, usize, u64, u64)> {
    if kernel_params.is_null() {
        return None;
    }
    for index in 1..6 {
        if let Some((out_addr, inp_addr)) = read_param_data_pair(kernel_params, index) {
            if sifive_cuda_alloc_has_bytes(out_addr, bytes)
                && sifive_cuda_alloc_has_bytes(inp_addr, bytes)
            {
                return Some((index, 0, out_addr, inp_addr));
            }
        }

        let param = *kernel_params.add(index);
        if param.is_null() || (param as usize) < 0x1_0000 {
            continue;
        }
        let base = param as usize;
        for off in (0..4096usize).step_by(std::mem::size_of::<u64>()) {
            if !sifive_host_range_has_perms(
                base.saturating_add(off),
                2 * std::mem::size_of::<u64>(),
                false,
            ) {
                continue;
            }
            let data = (base + off) as *const u64;
            let out_addr = data.read_unaligned();
            let inp_addr = data.add(1).read_unaligned();
            if out_addr < 0x1_0000 || inp_addr < 0x1_0000 {
                continue;
            }
            if sifive_cuda_alloc_has_bytes(out_addr, bytes)
                && sifive_cuda_alloc_has_bytes(inp_addr, bytes)
            {
                return Some((index, off, out_addr, inp_addr));
            }
        }
    }
    None
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn find_tensoriterator_data_single(
    kernel_params: *mut *mut ::core::ffi::c_void,
    bytes: usize,
) -> Option<(usize, usize, u64)> {
    if kernel_params.is_null() {
        return None;
    }
    for index in 1..8 {
        if let Some(out_addr) = read_param_data_single(kernel_params, index) {
            if sifive_cuda_alloc_has_bytes(out_addr, bytes) {
                return Some((index, 0, out_addr));
            }
        }

        let param = *kernel_params.add(index);
        if param.is_null() || (param as usize) < 0x1_0000 {
            continue;
        }
        let base = param as usize;
        for off in (0..4096usize).step_by(std::mem::size_of::<u64>()) {
            if !sifive_host_range_has_perms(
                base.saturating_add(off),
                std::mem::size_of::<u64>(),
                false,
            ) {
                continue;
            }
            let out_addr = ((base + off) as *const u64).read_unaligned();
            if out_addr < 0x1_0000 {
                continue;
            }
            if sifive_cuda_alloc_has_bytes(out_addr, bytes) {
                return Some((index, off, out_addr));
            }
        }
    }
    None
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn log_tensoriterator_triplet_scan_debug(
    kernel_params: *mut *mut ::core::ffi::c_void,
    kernel_name: &str,
    n: usize,
) {
    if !sifive_env_truthy("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS") || kernel_params.is_null() {
        return;
    }
    eprintln!(
        "[SIFIVE Backend] TensorIterator triplet scan debug '{}' n={}",
        kernel_name, n
    );
    for index in 0..6 {
        let param = *kernel_params.add(index);
        eprintln!("[SIFIVE Backend]   param[{}]={:p}", index, param);
        if param.is_null() || (param as usize) < 0x1_0000 {
            continue;
        }
        if !sifive_host_range_has_perms(param as usize, 8 * std::mem::size_of::<u64>(), false) {
            continue;
        }
        let data = param as *const u64;
        let mut words = [0u64; 8];
        for i in 0..8 {
            words[i] = data.add(i).read_unaligned();
        }
        eprintln!(
            "[SIFIVE Backend]   words[{}]= {:x} {:x} {:x} {:x} {:x} {:x} {:x} {:x}",
            index, words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]
        );
        for off in (0..1024usize).step_by(std::mem::size_of::<u64>()) {
            if !sifive_host_range_has_perms(
                (param as usize).saturating_add(off),
                std::mem::size_of::<u64>(),
                false,
            ) {
                continue;
            }
            let value = ((param as usize + off) as *const u64).read_unaligned();
            if let Some(remaining) = super::memory::sifive_allocation_remaining_addr(value) {
                eprintln!(
                    "[SIFIVE Backend]   alloc-candidate param={} off=0x{:x} value=0x{:x} remaining={}",
                    index, off, value, remaining
                );
            }
        }
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn find_tensoriterator_data_triplet(
    kernel_params: *mut *mut ::core::ffi::c_void,
    out_bytes: usize,
) -> Option<(usize, usize, u64, u64, u64)> {
    if kernel_params.is_null() {
        return None;
    }
    for index in 1..6 {
        let param = *kernel_params.add(index);
        if param.is_null() || (param as usize) < 0x1_0000 {
            continue;
        }
        let base = param as usize;
        for off in (0..4096usize).step_by(std::mem::size_of::<u64>()) {
            let p0 = base.saturating_add(off);
            let p1 = p0.saturating_add(std::mem::size_of::<u64>());
            let p2 = p1.saturating_add(std::mem::size_of::<u64>());
            if !sifive_host_range_has_perms(p0, std::mem::size_of::<u64>(), false)
                || !sifive_host_range_has_perms(p1, std::mem::size_of::<u64>(), false)
                || !sifive_host_range_has_perms(p2, std::mem::size_of::<u64>(), false)
            {
                continue;
            }
            let data = (base + off) as *const u64;
            let out_addr = data.read_unaligned();
            let lhs_addr = data.add(1).read_unaligned();
            let rhs_addr = data.add(2).read_unaligned();
            if out_addr < 0x1_0000 || lhs_addr < 0x1_0000 || rhs_addr < 0x1_0000 {
                continue;
            }
            if sifive_cuda_alloc_has_bytes(out_addr, out_bytes)
                && sifive_cuda_alloc_has_bytes(lhs_addr, std::mem::size_of::<f32>())
                && sifive_cuda_alloc_has_bytes(rhs_addr, std::mem::size_of::<f32>())
            {
                return Some((index, off, out_addr, lhs_addr, rhs_addr));
            }
        }
    }
    None
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn read_f32_tensor_prefix(
    addr: u64,
    elems: usize,
) -> Result<Vec<f32>, cuda_types::cuda::CUerror> {
    let bytes = elems.saturating_mul(std::mem::size_of::<f32>());
    let mut values = vec![0f32; elems];
    if elems == 0 {
        return Ok(values);
    }
    if sifive_host_range_has_perms(addr as usize, bytes, false) {
        std::ptr::copy_nonoverlapping(addr as *const f32, values.as_mut_ptr(), elems);
        return Ok(values);
    }
    super::memory::copy_dto_h_v2(
        values.as_mut_ptr() as *mut ::core::ffi::c_void,
        cuda_types::cuda::CUdeviceptr_v2(addr as *mut ::core::ffi::c_void),
        bytes,
    )?;
    Ok(values)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn write_f32_tensor(addr: u64, values: &[f32]) -> Result<(), cuda_types::cuda::CUerror> {
    let bytes = values.len().saturating_mul(std::mem::size_of::<f32>());
    if values.is_empty() {
        return Ok(());
    }
    if sifive_host_range_has_perms(addr as usize, bytes, true) {
        std::ptr::copy_nonoverlapping(values.as_ptr(), addr as *mut f32, values.len());
        return Ok(());
    }
    super::memory::copy_hto_d_v2(
        cuda_types::cuda::CUdeviceptr_v2(addr as *mut ::core::ffi::c_void),
        values.as_ptr() as *const ::core::ffi::c_void,
        bytes,
    )
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn read_bf16_tensor_prefix(
    addr: u64,
    elems: usize,
) -> Result<Vec<u16>, cuda_types::cuda::CUerror> {
    let bytes = elems.saturating_mul(std::mem::size_of::<u16>());
    let mut values = vec![0u16; elems];
    if elems == 0 {
        return Ok(values);
    }
    if sifive_host_range_has_perms(addr as usize, bytes, false) {
        std::ptr::copy_nonoverlapping(addr as *const u16, values.as_mut_ptr(), elems);
        return Ok(values);
    }
    super::memory::copy_dto_h_v2(
        values.as_mut_ptr() as *mut ::core::ffi::c_void,
        cuda_types::cuda::CUdeviceptr_v2(addr as *mut ::core::ffi::c_void),
        bytes,
    )?;
    Ok(values)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn write_bf16_tensor(addr: u64, values: &[u16]) -> Result<(), cuda_types::cuda::CUerror> {
    let bytes = values.len().saturating_mul(std::mem::size_of::<u16>());
    if values.is_empty() {
        return Ok(());
    }
    if sifive_host_range_has_perms(addr as usize, bytes, true) {
        std::ptr::copy_nonoverlapping(values.as_ptr(), addr as *mut u16, values.len());
        return Ok(());
    }
    super::memory::copy_hto_d_v2(
        cuda_types::cuda::CUdeviceptr_v2(addr as *mut ::core::ffi::c_void),
        values.as_ptr() as *const ::core::ffi::c_void,
        bytes,
    )
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_direct_copy_bf16_host(
    kernel_name: &str,
    grid_dim_x: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    let n = read_param_i32(kernel_params, 0)
        .map(|v| v.max(0) as usize)
        .unwrap_or_else(|| {
            (grid_dim_x as usize)
                .saturating_mul(block_dim_x as usize)
                .saturating_mul(4)
        });
    if n == 0 {
        return Some(Ok(()));
    }
    let bytes = n.saturating_mul(std::mem::size_of::<u16>());
    let Some((pair_index, pair_off, out_addr, inp_addr)) =
        find_tensoriterator_data_pair(kernel_params, bytes)
    else {
        eprintln!(
            "[SIFIVE Backend] direct_copy bf16 '{}' could not locate TensorIterator data pair for n={} bytes={}",
            kernel_name, n, bytes
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    };

    if sifive_host_range_has_perms(inp_addr as usize, bytes, false)
        && sifive_host_range_has_perms(out_addr as usize, bytes, true)
    {
        std::ptr::copy(inp_addr as *const u8, out_addr as *mut u8, bytes);
    } else {
        let mut tmp = vec![0u8; bytes];
        if sifive_host_range_has_perms(inp_addr as usize, bytes, false) {
            std::ptr::copy_nonoverlapping(inp_addr as *const u8, tmp.as_mut_ptr(), bytes);
        } else if let Err(err) = super::memory::copy_dto_h_v2(
            tmp.as_mut_ptr() as *mut ::core::ffi::c_void,
            cuda_types::cuda::CUdeviceptr_v2(inp_addr as *mut ::core::ffi::c_void),
            bytes,
        ) {
            eprintln!(
                "[SIFIVE Backend] direct_copy bf16 '{}' failed to read input out=0x{:x} inp=0x{:x} bytes={} err={:?}",
                kernel_name, out_addr, inp_addr, bytes, err
            );
            return Some(Err(err));
        }
        if sifive_host_range_has_perms(out_addr as usize, bytes, true) {
            std::ptr::copy_nonoverlapping(tmp.as_ptr(), out_addr as *mut u8, bytes);
        } else if let Err(err) = super::memory::copy_hto_d_v2(
            cuda_types::cuda::CUdeviceptr_v2(out_addr as *mut ::core::ffi::c_void),
            tmp.as_ptr() as *const ::core::ffi::c_void,
            bytes,
        ) {
            eprintln!(
                "[SIFIVE Backend] direct_copy bf16 '{}' failed to write output out=0x{:x} inp=0x{:x} bytes={} err={:?}",
                kernel_name, out_addr, inp_addr, bytes, err
            );
            return Some(Err(err));
        }
    }

    if sifive_env_truthy("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS") {
        eprintln!(
            "[SIFIVE Backend] handled direct_copy bf16 '{}' on host-copy path n={} bytes={} out=0x{:x} inp=0x{:x} pair_param={} pair_off=0x{:x}",
            kernel_name, n, bytes, out_addr, inp_addr, pair_index, pair_off
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_direct_copy_bool_host_cast(
    kernel_name: &str,
    grid_dim_x: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    let n = read_param_i32(kernel_params, 0)
        .map(|v| v.max(0) as usize)
        .unwrap_or_else(|| {
            (grid_dim_x as usize)
                .saturating_mul(block_dim_x as usize)
                .saturating_mul(4)
        });
    if n == 0 {
        return Some(Ok(()));
    }

    let mut pair = None;
    for index in 1..6 {
        if let Some((out_addr, inp_addr)) = read_param_data_pair(kernel_params, index) {
            if sifive_host_or_cuda_alloc_has_bytes(out_addr, n, true)
                && sifive_host_or_cuda_alloc_has_bytes(inp_addr, n, false)
            {
                pair = Some((index, out_addr, inp_addr));
                break;
            }
        }
    }
    let Some((pair_index, out_addr, inp_addr)) = pair else {
        eprintln!(
            "[SIFIVE Backend] direct_copy '{}' could not locate TensorIterator data pair for n={}",
            kernel_name, n
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    };

    let inp_remaining = super::memory::sifive_allocation_remaining_addr(inp_addr)
        .or_else(|| {
            if sifive_host_range_has_perms(inp_addr as usize, n.saturating_mul(8), false) {
                Some(n.saturating_mul(8))
            } else if sifive_host_range_has_perms(inp_addr as usize, n.saturating_mul(4), false) {
                Some(n.saturating_mul(4))
            } else if sifive_host_range_has_perms(inp_addr as usize, n.saturating_mul(2), false) {
                Some(n.saturating_mul(2))
            } else if sifive_host_range_has_perms(inp_addr as usize, n, false) {
                Some(n)
            } else {
                None
            }
        })
        .unwrap_or(n);
    let src_elem = if inp_remaining >= n.saturating_mul(8) {
        8
    } else if inp_remaining >= n.saturating_mul(4) {
        4
    } else if inp_remaining >= n.saturating_mul(2) {
        2
    } else {
        1
    };

    if !sifive_host_or_cuda_alloc_has_bytes(out_addr, n, true)
        || !sifive_host_or_cuda_alloc_has_bytes(inp_addr, n.saturating_mul(src_elem), false)
    {
        eprintln!(
            "[SIFIVE Backend] direct_copy '{}' rejected allocation range out=0x{:x} inp=0x{:x} n={} src_elem={}",
            kernel_name, out_addr, inp_addr, n, src_elem
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }

    let src_bytes = n.saturating_mul(src_elem);
    let mut src = vec![0u8; src_bytes];
    if sifive_host_range_has_perms(inp_addr as usize, src_bytes, false) {
        std::ptr::copy_nonoverlapping(inp_addr as *const u8, src.as_mut_ptr(), src_bytes);
    } else if let Err(err) = super::memory::copy_dto_h_v2(
        src.as_mut_ptr() as *mut ::core::ffi::c_void,
        cuda_types::cuda::CUdeviceptr_v2(inp_addr as *mut ::core::ffi::c_void),
        src_bytes,
    ) {
        eprintln!(
            "[SIFIVE Backend] direct_copy '{}' failed to read input out=0x{:x} inp=0x{:x} n={} src_elem={} err={:?}",
            kernel_name, out_addr, inp_addr, n, src_elem, err
        );
        return Some(Err(err));
    }

    let mut dst = vec![0u8; n];
    for i in 0..n {
        let nonzero = match src_elem {
            8 => (src.as_ptr().add(i * 8) as *const u64).read_unaligned() != 0,
            4 => (src.as_ptr().add(i * 4) as *const u32).read_unaligned() != 0,
            2 => (src.as_ptr().add(i * 2) as *const u16).read_unaligned() != 0,
            _ => src.as_ptr().add(i).read_unaligned() != 0,
        };
        dst[i] = if nonzero { 1 } else { 0 };
    }

    if sifive_host_range_has_perms(out_addr as usize, n, true) {
        std::ptr::copy_nonoverlapping(dst.as_ptr(), out_addr as *mut u8, n);
    } else if let Err(err) = super::memory::copy_hto_d_v2(
        cuda_types::cuda::CUdeviceptr_v2(out_addr as *mut ::core::ffi::c_void),
        dst.as_ptr() as *const ::core::ffi::c_void,
        n,
    ) {
        eprintln!(
            "[SIFIVE Backend] direct_copy '{}' failed to write output out=0x{:x} inp=0x{:x} n={} src_elem={} err={:?}",
            kernel_name, out_addr, inp_addr, n, src_elem, err
        );
        return Some(Err(err));
    }

    if sifive_env_truthy("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS") {
        eprintln!(
            "[SIFIVE Backend] handled direct_copy '{}' on host bool-cast path n={} src_elem={} pair_param={}",
            kernel_name, n, src_elem, pair_index
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn infer_linear_elem_size(addr: u64, n: usize, writable: bool) -> usize {
    let remaining = super::memory::sifive_allocation_remaining_addr(addr).or_else(|| {
        for elem in [4usize, 2, 1] {
            if sifive_host_range_has_perms(addr as usize, n.saturating_mul(elem), writable) {
                return Some(n.saturating_mul(elem));
            }
        }
        None
    });
    let Some(bytes) = remaining else {
        return 1;
    };
    if bytes >= n.saturating_mul(4) {
        4
    } else if bytes >= n.saturating_mul(2) {
        2
    } else {
        1
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_direct_copy_cast_host(
    kernel_name: &str,
    grid_dim_x: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    let n = read_param_i32(kernel_params, 0)
        .map(|v| v.max(0) as usize)
        .unwrap_or_else(|| {
            (grid_dim_x as usize)
                .saturating_mul(block_dim_x as usize)
                .saturating_mul(4)
        });
    if n == 0 {
        return Some(Ok(()));
    }
    let Some((pair_index, pair_off, out_addr, inp_addr)) =
        find_tensoriterator_data_pair(kernel_params, n)
    else {
        eprintln!(
            "[SIFIVE Backend] direct_copy cast '{}' could not locate TensorIterator data pair for n={}",
            kernel_name, n
        );
        return None;
    };

    let src_elem = infer_linear_elem_size(inp_addr, n, false);
    let dst_elem = infer_linear_elem_size(out_addr, n, true);
    if !sifive_host_or_cuda_alloc_has_bytes(inp_addr, n.saturating_mul(src_elem), false)
        || !sifive_host_or_cuda_alloc_has_bytes(out_addr, n.saturating_mul(dst_elem), true)
    {
        eprintln!(
            "[SIFIVE Backend] direct_copy cast '{}' rejected ranges out=0x{:x}/{} inp=0x{:x}/{} n={}",
            kernel_name, out_addr, dst_elem, inp_addr, src_elem, n
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }

    if src_elem == dst_elem && src_elem != 1 {
        let bytes = n.saturating_mul(src_elem);
        let mut tmp = vec![0u8; bytes];
        if sifive_host_range_has_perms(inp_addr as usize, bytes, false) {
            std::ptr::copy_nonoverlapping(inp_addr as *const u8, tmp.as_mut_ptr(), bytes);
        } else if let Err(err) = super::memory::copy_dto_h_v2(
            tmp.as_mut_ptr() as *mut ::core::ffi::c_void,
            cuda_types::cuda::CUdeviceptr_v2(inp_addr as *mut ::core::ffi::c_void),
            bytes,
        ) {
            return Some(Err(err));
        }
        if sifive_host_range_has_perms(out_addr as usize, bytes, true) {
            std::ptr::copy_nonoverlapping(tmp.as_ptr(), out_addr as *mut u8, bytes);
        } else if let Err(err) = super::memory::copy_hto_d_v2(
            cuda_types::cuda::CUdeviceptr_v2(out_addr as *mut ::core::ffi::c_void),
            tmp.as_ptr() as *const ::core::ffi::c_void,
            bytes,
        ) {
            return Some(Err(err));
        }
    } else {
        let mut values = vec![0f32; n];
        match src_elem {
            4 => {
                values = match read_f32_tensor_prefix(inp_addr, n) {
                    Ok(v) => v,
                    Err(err) => return Some(Err(err)),
                };
            }
            2 => {
                let src = match read_bf16_tensor_prefix(inp_addr, n) {
                    Ok(v) => v,
                    Err(err) => return Some(Err(err)),
                };
                for i in 0..n {
                    values[i] = sifive_bf16_to_f32(src[i]);
                }
            }
            _ => {
                let mut src = vec![0u8; n];
                if sifive_host_range_has_perms(inp_addr as usize, n, false) {
                    std::ptr::copy_nonoverlapping(inp_addr as *const u8, src.as_mut_ptr(), n);
                } else if let Err(err) = super::memory::copy_dto_h_v2(
                    src.as_mut_ptr() as *mut ::core::ffi::c_void,
                    cuda_types::cuda::CUdeviceptr_v2(inp_addr as *mut ::core::ffi::c_void),
                    n,
                ) {
                    return Some(Err(err));
                }
                for i in 0..n {
                    values[i] = if src[i] != 0 { 1.0 } else { 0.0 };
                }
            }
        }

        match dst_elem {
            4 => {
                if let Err(err) = write_f32_tensor(out_addr, &values) {
                    return Some(Err(err));
                }
            }
            2 => {
                let dst: Vec<u16> = values.iter().map(|&v| sifive_f32_to_bf16(v)).collect();
                if let Err(err) = write_bf16_tensor(out_addr, &dst) {
                    return Some(Err(err));
                }
            }
            _ => {
                let dst: Vec<u8> = values
                    .iter()
                    .map(|&v| if v != 0.0 { 1 } else { 0 })
                    .collect();
                if sifive_host_range_has_perms(out_addr as usize, n, true) {
                    std::ptr::copy_nonoverlapping(dst.as_ptr(), out_addr as *mut u8, n);
                } else if let Err(err) = super::memory::copy_hto_d_v2(
                    cuda_types::cuda::CUdeviceptr_v2(out_addr as *mut ::core::ffi::c_void),
                    dst.as_ptr() as *const ::core::ffi::c_void,
                    n,
                ) {
                    return Some(Err(err));
                }
            }
        }
    }

    if sifive_env_truthy("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS") {
        eprintln!(
            "[SIFIVE Backend] handled direct_copy cast '{}' n={} src_elem={} dst_elem={} out=0x{:x} inp=0x{:x} pair_param={} pair_off=0x{:x}",
            kernel_name, n, src_elem, dst_elem, out_addr, inp_addr, pair_index, pair_off
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_unary_bf16_host(
    kernel_name: &str,
    op: &str,
    grid_dim_x: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    let n = read_param_i32(kernel_params, 0)
        .map(|v| v.max(0) as usize)
        .unwrap_or_else(|| {
            (grid_dim_x as usize)
                .saturating_mul(block_dim_x as usize)
                .saturating_mul(4)
        });
    if n == 0 {
        return Some(Ok(()));
    }
    let bytes = n.saturating_mul(std::mem::size_of::<u16>());
    let Some((pair_index, pair_off, out_addr, inp_addr)) =
        find_tensoriterator_data_pair(kernel_params, bytes)
    else {
        eprintln!(
            "[SIFIVE Backend] unary bf16 {} '{}' could not locate TensorIterator data pair for n={} bytes={}",
            op, kernel_name, n, bytes
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    };

    let mut input = vec![0u16; n];
    if sifive_host_range_has_perms(inp_addr as usize, bytes, false) {
        std::ptr::copy_nonoverlapping(inp_addr as *const u16, input.as_mut_ptr(), n);
    } else if let Err(err) = super::memory::copy_dto_h_v2(
        input.as_mut_ptr() as *mut ::core::ffi::c_void,
        cuda_types::cuda::CUdeviceptr_v2(inp_addr as *mut ::core::ffi::c_void),
        bytes,
    ) {
        eprintln!(
            "[SIFIVE Backend] unary bf16 {} '{}' failed to read input out=0x{:x} inp=0x{:x} bytes={} err={:?}",
            op, kernel_name, out_addr, inp_addr, bytes, err
        );
        return Some(Err(err));
    }

    let mut output = vec![0u16; n];
    for i in 0..n {
        let x = sifive_bf16_to_f32(input[i]);
        let y = match op {
            "log" => x.ln(),
            "silu" => sifive_silu(x),
            _ => {
                if x >= 0.0 {
                    let z = (-x).exp();
                    1.0 / (1.0 + z)
                } else {
                    let z = x.exp();
                    z / (1.0 + z)
                }
            }
        };
        output[i] = sifive_f32_to_bf16(y);
    }

    if sifive_host_range_has_perms(out_addr as usize, bytes, true) {
        std::ptr::copy_nonoverlapping(output.as_ptr(), out_addr as *mut u16, n);
    } else if let Err(err) = super::memory::copy_hto_d_v2(
        cuda_types::cuda::CUdeviceptr_v2(out_addr as *mut ::core::ffi::c_void),
        output.as_ptr() as *const ::core::ffi::c_void,
        bytes,
    ) {
        eprintln!(
            "[SIFIVE Backend] unary bf16 {} '{}' failed to write output out=0x{:x} inp=0x{:x} bytes={} err={:?}",
            op, kernel_name, out_addr, inp_addr, bytes, err
        );
        return Some(Err(err));
    }

    if sifive_env_truthy("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS") {
        eprintln!(
            "[SIFIVE Backend] handled unary bf16 {} '{}' on host path n={} bytes={} out=0x{:x} inp=0x{:x} pair_param={} pair_off=0x{:x}",
            op, kernel_name, n, bytes, out_addr, inp_addr, pair_index, pair_off
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_uniform_bf16_host(
    kernel_name: &str,
    grid_dim_x: ::core::ffi::c_uint,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    let n = read_param_i32(kernel_params, 0)
        .map(|v| v.max(0) as usize)
        .unwrap_or_else(|| (grid_dim_x as usize).saturating_mul(4));
    if n == 0 {
        return Some(Ok(()));
    }
    let bytes = n.saturating_mul(std::mem::size_of::<u16>());
    let Some((param_index, param_off, out_addr)) =
        find_tensoriterator_data_single(kernel_params, bytes)
    else {
        eprintln!(
            "[SIFIVE Backend] uniform bf16 '{}' could not locate TensorIterator output for n={} bytes={}",
            kernel_name, n, bytes
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    };

    let mut output = vec![0u16; n];
    for i in 0..n {
        let mut x = (i as u64)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(0xbf58_476d_1ce4_e5b9);
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^= x >> 31;
        let unit = ((x >> 40) as f32) * (1.0 / ((1u32 << 24) as f32));
        output[i] = sifive_f32_to_bf16(unit * 16.0);
    }
    if let Err(err) = write_bf16_tensor(out_addr, &output) {
        eprintln!(
            "[SIFIVE Backend] uniform bf16 '{}' failed to write output out=0x{:x} n={} err={:?}",
            kernel_name, out_addr, n, err
        );
        return Some(Err(err));
    }
    if sifive_env_truthy("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS") {
        eprintln!(
            "[SIFIVE Backend] handled uniform bf16 '{}' on host path n={} out=0x{:x} param={} off=0x{:x}",
            kernel_name, n, out_addr, param_index, param_off
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_unary_f32_host(
    kernel_name: &str,
    op_name: &str,
    grid_dim_x: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    let n = read_param_i32(kernel_params, 0)
        .map(|v| v.max(0) as usize)
        .unwrap_or_else(|| {
            (grid_dim_x as usize)
                .saturating_mul(block_dim_x as usize)
                .saturating_mul(4)
        });
    if n == 0 {
        return Some(Ok(()));
    }
    let bytes = n.saturating_mul(std::mem::size_of::<f32>());
    let Some((pair_index, pair_off, out_addr, inp_addr)) =
        find_tensoriterator_data_pair(kernel_params, bytes)
    else {
        eprintln!(
            "[SIFIVE Backend] unary f32 {} '{}' could not locate TensorIterator data pair for n={} bytes={}",
            op_name, kernel_name, n, bytes
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    };

    let mut input = vec![0f32; n];
    if sifive_host_range_has_perms(inp_addr as usize, bytes, false) {
        std::ptr::copy_nonoverlapping(inp_addr as *const f32, input.as_mut_ptr(), n);
    } else if let Err(err) = super::memory::copy_dto_h_v2(
        input.as_mut_ptr() as *mut ::core::ffi::c_void,
        cuda_types::cuda::CUdeviceptr_v2(inp_addr as *mut ::core::ffi::c_void),
        bytes,
    ) {
        eprintln!(
            "[SIFIVE Backend] unary f32 {} '{}' failed to read input out=0x{:x} inp=0x{:x} bytes={} err={:?}",
            op_name, kernel_name, out_addr, inp_addr, bytes, err
        );
        return Some(Err(err));
    }

    let mut output = vec![0f32; n];
    for i in 0..n {
        let x = input[i];
        output[i] = match op_name {
            "exp" => x.exp(),
            "log" => x.ln(),
            "softplus" => {
                if x > 20.0 {
                    x
                } else if x < -20.0 {
                    x.exp()
                } else {
                    (1.0 + x.exp()).ln()
                }
            }
            "neg" => -x,
            "sigmoid" => {
                if x >= 0.0 {
                    let z = (-x).exp();
                    1.0 / (1.0 + z)
                } else {
                    let z = x.exp();
                    z / (1.0 + z)
                }
            }
            _ => x,
        };
    }

    if sifive_host_range_has_perms(out_addr as usize, bytes, true) {
        std::ptr::copy_nonoverlapping(output.as_ptr(), out_addr as *mut f32, n);
    } else if let Err(err) = super::memory::copy_hto_d_v2(
        cuda_types::cuda::CUdeviceptr_v2(out_addr as *mut ::core::ffi::c_void),
        output.as_ptr() as *const ::core::ffi::c_void,
        bytes,
    ) {
        eprintln!(
            "[SIFIVE Backend] unary f32 {} '{}' failed to write output out=0x{:x} inp=0x{:x} bytes={} err={:?}",
            op_name, kernel_name, out_addr, inp_addr, bytes, err
        );
        return Some(Err(err));
    }

    if sifive_env_truthy("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS") {
        eprintln!(
            "[SIFIVE Backend] handled unary f32 {} '{}' on host path n={} bytes={} out=0x{:x} inp=0x{:x} pair_param={} pair_off=0x{:x}",
            op_name, kernel_name, n, bytes, out_addr, inp_addr, pair_index, pair_off
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_binary_f32_host(
    kernel_name: &str,
    op_name: &str,
    grid_dim_x: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    let n = read_param_i32(kernel_params, 0)
        .map(|v| v.max(0) as usize)
        .unwrap_or_else(|| {
            (grid_dim_x as usize)
                .saturating_mul(block_dim_x as usize)
                .saturating_mul(4)
        });
    if n == 0 {
        return Some(Ok(()));
    }
    let out_bytes = n.saturating_mul(std::mem::size_of::<f32>());
    let Some((triplet_index, triplet_off, out_addr, lhs_addr, rhs_addr)) =
        find_tensoriterator_data_triplet(kernel_params, out_bytes)
    else {
        log_tensoriterator_triplet_scan_debug(kernel_params, kernel_name, n);
        eprintln!(
            "[SIFIVE Backend] binary f32 {} '{}' could not locate TensorIterator data triplet for n={} bytes={}",
            op_name, kernel_name, n, out_bytes
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    };

    let lhs_elems = super::memory::sifive_allocation_remaining_addr(lhs_addr)
        .map(|bytes| (bytes / std::mem::size_of::<f32>()).clamp(1, n))
        .unwrap_or(n);
    let rhs_elems = super::memory::sifive_allocation_remaining_addr(rhs_addr)
        .map(|bytes| (bytes / std::mem::size_of::<f32>()).clamp(1, n))
        .unwrap_or(n);
    let lhs = match read_f32_tensor_prefix(lhs_addr, lhs_elems) {
        Ok(v) => v,
        Err(err) => {
            eprintln!(
                "[SIFIVE Backend] binary f32 {} '{}' failed to read lhs=0x{:x} elems={} err={:?}",
                op_name, kernel_name, lhs_addr, lhs_elems, err
            );
            return Some(Err(err));
        }
    };
    let rhs = match read_f32_tensor_prefix(rhs_addr, rhs_elems) {
        Ok(v) => v,
        Err(err) => {
            eprintln!(
                "[SIFIVE Backend] binary f32 {} '{}' failed to read rhs=0x{:x} elems={} err={:?}",
                op_name, kernel_name, rhs_addr, rhs_elems, err
            );
            return Some(Err(err));
        }
    };

    let mut out = vec![0f32; n];
    for i in 0..n {
        let a = lhs[i % lhs.len()];
        let b = rhs[i % rhs.len()];
        out[i] = match op_name {
            "add" => a + b,
            "mul" => a * b,
            "div" => a / b,
            _ => a,
        };
    }
    if let Err(err) = write_f32_tensor(out_addr, &out) {
        eprintln!(
            "[SIFIVE Backend] binary f32 {} '{}' failed to write out=0x{:x} bytes={} err={:?}",
            op_name, kernel_name, out_addr, out_bytes, err
        );
        return Some(Err(err));
    }

    if sifive_env_truthy("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS") {
        eprintln!(
            "[SIFIVE Backend] handled binary f32 {} '{}' on host path n={} lhs_elems={} rhs_elems={} out=0x{:x} lhs=0x{:x} rhs=0x{:x} triplet_param={} triplet_off=0x{:x}",
            op_name, kernel_name, n, lhs_elems, rhs_elems, out_addr, lhs_addr, rhs_addr, triplet_index, triplet_off
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_binary_bf16_host(
    kernel_name: &str,
    op_name: &str,
    grid_dim_x: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    let n = read_param_i32(kernel_params, 0)
        .map(|v| v.max(0) as usize)
        .unwrap_or_else(|| {
            (grid_dim_x as usize)
                .saturating_mul(block_dim_x as usize)
                .saturating_mul(4)
        });
    if n == 0 {
        return Some(Ok(()));
    }
    let out_bytes = n.saturating_mul(std::mem::size_of::<u16>());
    let Some((triplet_index, triplet_off, out_addr, lhs_addr, rhs_addr)) =
        find_tensoriterator_data_triplet(kernel_params, out_bytes)
    else {
        log_tensoriterator_triplet_scan_debug(kernel_params, kernel_name, n);
        eprintln!(
            "[SIFIVE Backend] binary bf16 {} '{}' could not locate TensorIterator data triplet for n={} bytes={}",
            op_name, kernel_name, n, out_bytes
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    };

    let lhs_elems = super::memory::sifive_allocation_remaining_addr(lhs_addr)
        .map(|bytes| (bytes / std::mem::size_of::<u16>()).clamp(1, n))
        .unwrap_or(n);
    let rhs_elems = super::memory::sifive_allocation_remaining_addr(rhs_addr)
        .map(|bytes| (bytes / std::mem::size_of::<u16>()).clamp(1, n))
        .unwrap_or(n);
    let lhs = match read_bf16_tensor_prefix(lhs_addr, lhs_elems) {
        Ok(v) => v,
        Err(err) => {
            eprintln!(
                "[SIFIVE Backend] binary bf16 {} '{}' failed to read lhs=0x{:x} elems={} err={:?}",
                op_name, kernel_name, lhs_addr, lhs_elems, err
            );
            return Some(Err(err));
        }
    };
    let rhs = match read_bf16_tensor_prefix(rhs_addr, rhs_elems) {
        Ok(v) => v,
        Err(err) => {
            eprintln!(
                "[SIFIVE Backend] binary bf16 {} '{}' failed to read rhs=0x{:x} elems={} err={:?}",
                op_name, kernel_name, rhs_addr, rhs_elems, err
            );
            return Some(Err(err));
        }
    };

    let mut out = vec![0u16; n];
    for i in 0..n {
        let a = sifive_bf16_to_f32(lhs[i % lhs.len()]);
        let b = sifive_bf16_to_f32(rhs[i % rhs.len()]);
        let y = match op_name {
            "add" => a + b,
            "mul" => a * b,
            "div" => a / b,
            _ => a,
        };
        out[i] = sifive_f32_to_bf16(y);
    }
    if let Err(err) = write_bf16_tensor(out_addr, &out) {
        eprintln!(
            "[SIFIVE Backend] binary bf16 {} '{}' failed to write out=0x{:x} bytes={} err={:?}",
            op_name, kernel_name, out_addr, out_bytes, err
        );
        return Some(Err(err));
    }

    if sifive_env_truthy("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS") {
        eprintln!(
            "[SIFIVE Backend] handled binary bf16 {} '{}' on host path n={} lhs_elems={} rhs_elems={} out=0x{:x} lhs=0x{:x} rhs=0x{:x} triplet_param={} triplet_off=0x{:x}",
            op_name, kernel_name, n, lhs_elems, rhs_elems, out_addr, lhs_addr, rhs_addr, triplet_index, triplet_off
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn read_aunary_f32_scalar(kernel_params: *mut *mut ::core::ffi::c_void) -> Option<f32> {
    if kernel_params.is_null() {
        return None;
    }
    for index in 0..3 {
        let param = *kernel_params.add(index);
        if param.is_null() || (param as usize) < 0x1_0000 {
            continue;
        }
        let base = param as usize;
        for off in (0..128usize).step_by(std::mem::size_of::<f32>()) {
            if !sifive_host_range_has_perms(
                base.saturating_add(off),
                std::mem::size_of::<f32>(),
                false,
            ) {
                continue;
            }
            let value = ((base + off) as *const f32).read_unaligned();
            if value.is_finite() && value.abs() >= 1.0e-6 && value.abs() <= 1.0e6 {
                return Some(value);
            }
        }
    }
    None
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_aunary_f32_host(
    kernel_name: &str,
    op_name: &str,
    grid_dim_x: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    let n = read_param_i32(kernel_params, 0)
        .map(|v| v.max(0) as usize)
        .unwrap_or_else(|| {
            (grid_dim_x as usize)
                .saturating_mul(block_dim_x as usize)
                .saturating_mul(4)
        });
    if n == 0 {
        return Some(Ok(()));
    }
    let bytes = n.saturating_mul(std::mem::size_of::<f32>());
    let Some((pair_index, pair_off, out_addr, inp_addr)) =
        find_tensoriterator_data_pair(kernel_params, bytes)
    else {
        eprintln!(
            "[SIFIVE Backend] aunary f32 {} '{}' could not locate TensorIterator data pair for n={} bytes={}",
            op_name, kernel_name, n, bytes
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    };
    let Some(scalar) = read_aunary_f32_scalar(kernel_params) else {
        eprintln!(
            "[SIFIVE Backend] aunary f32 {} '{}' could not locate scalar for n={}",
            op_name, kernel_name, n
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    };
    let input = match read_f32_tensor_prefix(inp_addr, n) {
        Ok(v) => v,
        Err(err) => return Some(Err(err)),
    };
    let mut out = vec![0f32; n];
    for i in 0..n {
        out[i] = match op_name {
            "mul" => input[i] * scalar,
            "add" => input[i] + scalar,
            _ => input[i],
        };
    }
    if let Err(err) = write_f32_tensor(out_addr, &out) {
        return Some(Err(err));
    }
    if sifive_env_truthy("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS") {
        eprintln!(
            "[SIFIVE Backend] handled aunary f32 {} '{}' n={} scalar={} out=0x{:x} inp=0x{:x} pair_param={} pair_off=0x{:x}",
            op_name, kernel_name, n, scalar, out_addr, inp_addr, pair_index, pair_off
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_arange_host_fill(
    kernel_name: &str,
    grid_dim_x: ::core::ffi::c_uint,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    let n = read_param_i32(kernel_params, 0)
        .map(|v| v.max(0) as usize)
        .unwrap_or(grid_dim_x as usize);
    if n == 0 {
        return Some(Ok(()));
    }

    let mut out_addr = None;
    for index in 1..4 {
        let Some(candidate) = read_param_u64(kernel_params, index) else {
            continue;
        };
        if candidate < 0x1_0000 {
            continue;
        }
        if sifive_host_or_cuda_alloc_has_bytes(candidate, n, true) {
            out_addr = Some((index, candidate));
            break;
        }
    }
    let Some((param_index, out_addr)) = out_addr else {
        eprintln!(
            "[SIFIVE Backend] arange '{}' could not locate output allocation for n={}",
            kernel_name, n
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    };

    let remaining = super::memory::sifive_allocation_remaining_addr(out_addr).unwrap_or(n);
    if remaining >= n.saturating_mul(8)
        && sifive_host_or_cuda_alloc_has_bytes(out_addr, n.saturating_mul(8), true)
    {
        let out = out_addr as *mut i64;
        for i in 0..n {
            out.add(i).write(i as i64);
        }
        if sifive_env_truthy("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS") {
            eprintln!(
                "[SIFIVE Backend] handled arange '{}' on host-fill path n={} dtype=i64 param={}",
                kernel_name, n, param_index
            );
        }
        return Some(Ok(()));
    }
    if remaining >= n.saturating_mul(4)
        && sifive_host_or_cuda_alloc_has_bytes(out_addr, n.saturating_mul(4), true)
    {
        let out = out_addr as *mut i32;
        for i in 0..n {
            out.add(i).write(i as i32);
        }
        if sifive_env_truthy("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS") {
            eprintln!(
                "[SIFIVE Backend] handled arange '{}' on host-fill path n={} dtype=i32 param={}",
                kernel_name, n, param_index
            );
        }
        return Some(Ok(()));
    }

    eprintln!(
        "[SIFIVE Backend] arange '{}' rejected output allocation out=0x{:x} n={} remaining={}",
        kernel_name, out_addr, n, remaining
    );
    Some(Err(cuda_types::cuda::CUerror::UNKNOWN))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_vectorized_add_i64_host(
    kernel_name: &str,
    grid_dim_x: ::core::ffi::c_uint,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    let n = read_param_i32(kernel_params, 0)
        .map(|v| v.max(0) as usize)
        .unwrap_or(grid_dim_x as usize);
    if n == 0 {
        return Some(Ok(()));
    }
    let scalar = read_param_i64(kernel_params, 1).unwrap_or(0);
    let Some((out_addr, inp_addr)) = read_param_data_pair(kernel_params, 2) else {
        eprintln!(
            "[SIFIVE Backend] vectorized add '{}' missing data pair n={}",
            kernel_name, n
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    };
    let bytes = n.saturating_mul(std::mem::size_of::<i64>());
    if !sifive_host_or_cuda_alloc_has_bytes(out_addr, bytes, true)
        || !sifive_host_or_cuda_alloc_has_bytes(inp_addr, bytes, false)
    {
        eprintln!(
            "[SIFIVE Backend] vectorized add '{}' rejected allocation out=0x{:x} inp=0x{:x} bytes={}",
            kernel_name, out_addr, inp_addr, bytes
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }

    let out = out_addr as *mut i64;
    let inp = inp_addr as *const i64;
    for i in 0..n {
        out.add(i)
            .write(inp.add(i).read_unaligned().wrapping_add(scalar));
    }
    if sifive_env_truthy("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS") {
        eprintln!(
            "[SIFIVE Backend] handled vectorized add '{}' on host i64 path n={} scalar={}",
            kernel_name, n, scalar
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_fill_bool_host(
    kernel_name: &str,
    grid_dim_x: ::core::ffi::c_uint,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    let n = read_param_i32(kernel_params, 0)
        .map(|v| v.max(0) as usize)
        .unwrap_or(grid_dim_x as usize);
    if n == 0 {
        return Some(Ok(()));
    }
    let value = read_param_bool(kernel_params, 1).unwrap_or(true);
    let Some(out_addr) = read_param_data_single(kernel_params, 2) else {
        eprintln!(
            "[SIFIVE Backend] fill bool '{}' missing output pointer n={}",
            kernel_name, n
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    };
    if !sifive_host_or_cuda_alloc_has_bytes(out_addr, n, true) {
        eprintln!(
            "[SIFIVE Backend] fill bool '{}' rejected output allocation out=0x{:x} n={}",
            kernel_name, out_addr, n
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }
    std::ptr::write_bytes(out_addr as *mut u8, if value { 1 } else { 0 }, n);
    if sifive_env_truthy("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS") {
        eprintln!(
            "[SIFIVE Backend] handled fill bool '{}' on host path n={} value={}",
            kernel_name, n, value
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_fill_bf16_host(
    kernel_name: &str,
    grid_dim_x: ::core::ffi::c_uint,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    let n = read_param_i32(kernel_params, 0)
        .map(|v| v.max(0) as usize)
        .unwrap_or(grid_dim_x as usize);
    if n == 0 {
        return Some(Ok(()));
    }
    let value = if !kernel_params.is_null() {
        let param = *kernel_params.add(1);
        if !param.is_null()
            && (param as usize) >= 0x1_0000
            && sifive_host_range_has_perms(param as usize, std::mem::size_of::<u16>(), false)
        {
            (param as *const u16).read_unaligned()
        } else {
            0
        }
    } else {
        0
    };
    let Some(out_addr) = read_param_data_single(kernel_params, 2) else {
        eprintln!(
            "[SIFIVE Backend] fill bf16 '{}' missing output pointer n={}",
            kernel_name, n
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    };
    let bytes = n.saturating_mul(std::mem::size_of::<u16>());
    if !sifive_host_or_cuda_alloc_has_bytes(out_addr, bytes, true) {
        eprintln!(
            "[SIFIVE Backend] fill bf16 '{}' rejected output allocation out=0x{:x} bytes={}",
            kernel_name, out_addr, bytes
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }
    let out = out_addr as *mut u16;
    for i in 0..n {
        out.add(i).write_unaligned(value);
    }
    if sifive_env_truthy("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS") {
        eprintln!(
            "[SIFIVE Backend] handled fill bf16 '{}' on host path n={} value=0x{:04x}",
            kernel_name, n, value
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_fill_f32_host(
    kernel_name: &str,
    grid_dim_x: ::core::ffi::c_uint,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    let n = read_param_i32(kernel_params, 0)
        .map(|v| v.max(0) as usize)
        .unwrap_or(grid_dim_x as usize);
    if n == 0 {
        return Some(Ok(()));
    }
    let value = read_param_f32(kernel_params, 1).unwrap_or(0.0);
    let Some(out_addr) = read_param_data_single(kernel_params, 2) else {
        eprintln!(
            "[SIFIVE Backend] fill f32 '{}' missing output pointer n={}",
            kernel_name, n
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    };
    let bytes = n.saturating_mul(std::mem::size_of::<f32>());
    if !sifive_host_or_cuda_alloc_has_bytes(out_addr, bytes, true) {
        eprintln!(
            "[SIFIVE Backend] fill f32 '{}' rejected output allocation out=0x{:x} bytes={}",
            kernel_name, out_addr, bytes
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }
    let values = vec![value; n];
    if let Err(err) = write_f32_tensor(out_addr, &values) {
        eprintln!(
            "[SIFIVE Backend] fill f32 '{}' failed to write out=0x{:x} bytes={} err={:?}",
            kernel_name, out_addr, bytes, err
        );
        return Some(Err(err));
    }
    if sifive_env_truthy("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS") {
        eprintln!(
            "[SIFIVE Backend] handled fill f32 '{}' on host path n={} value={}",
            kernel_name, n, value
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_compare_i64_host_debug(
    kernel_name: &str,
    grid_dim_x: ::core::ffi::c_uint,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    let n = read_param_i32(kernel_params, 0)
        .map(|v| v.max(0) as usize)
        .unwrap_or(grid_dim_x as usize);
    eprintln!(
        "[SIFIVE Backend] compare i64 '{}' debug n={} grid_x={}",
        kernel_name, n, grid_dim_x
    );
    if kernel_params.is_null() {
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }
    for index in 0..2 {
        let param = *kernel_params.add(index);
        if param.is_null() {
            eprintln!("[SIFIVE Backend] compare param[{}]=NULL", index);
            continue;
        }
        let addr = param as usize;
        if !sifive_host_range_has_perms(addr, 64, false) {
            eprintln!(
                "[SIFIVE Backend] compare param[{}]={:p} not readable64",
                index, param
            );
            continue;
        }
        let mut rendered = String::new();
        let mut words = [0u64; 8];
        for word_index in 0..words.len() {
            words[word_index] = (param as *const u8)
                .add(word_index * std::mem::size_of::<u64>())
                .cast::<u64>()
                .read_unaligned();
        }
        for (word_index, value) in words.iter().copied().enumerate() {
            let alloc = sifive_host_or_cuda_alloc_has_bytes(value, 1, word_index == 0);
            let _ = std::fmt::Write::write_fmt(
                &mut rendered,
                format_args!(
                    " w{}=0x{:x}{}",
                    word_index,
                    value,
                    if alloc { "*" } else { "" }
                ),
            );
        }
        eprintln!(
            "[SIFIVE Backend] compare param[{}]={:p}{}",
            index, param, rendered
        );
        if index == 0 {
            continue;
        }
        for (word_index, value) in words.iter().copied().enumerate() {
            let Ok(ptr) = usize::try_from(value) else {
                continue;
            };
            if ptr < 0x1_0000 || !sifive_host_range_has_perms(ptr, 128, false) {
                continue;
            }
            let mut nested_rendered = String::new();
            for nested_index in 0..16 {
                let nested_value = (ptr as *const u8)
                    .add(nested_index * std::mem::size_of::<u64>())
                    .cast::<u64>()
                    .read_unaligned();
                let alloc = sifive_host_or_cuda_alloc_has_bytes(nested_value, 1, nested_index == 0);
                let _ = std::fmt::Write::write_fmt(
                    &mut nested_rendered,
                    format_args!(
                        " n{}=0x{:x}{}",
                        nested_index,
                        nested_value,
                        if alloc { "*" } else { "" }
                    ),
                );
            }
            eprintln!(
                "[SIFIVE Backend] compare param[{}].w{} -> 0x{:x}{}",
                index, word_index, ptr, nested_rendered
            );
        }
    }
    Some(Err(cuda_types::cuda::CUerror::UNKNOWN))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_scale_f32_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    let src = sifive_host_ptr::<f32>(read_param_u64(kernel_params, 0)?)?;
    let dst = sifive_host_ptr::<f32>(read_param_u64(kernel_params, 1)?)?;
    let scale = read_param_f32(kernel_params, 2)?;
    let bias = read_param_f32(kernel_params, 3)?;
    let nelements = read_param_i64(kernel_params, 4)?.max(0) as usize;

    if !sifive_cuda_alloc_has_elems(src as *const f32, nelements)
        || !sifive_cuda_alloc_has_elems(dst as *const f32, nelements)
    {
        eprintln!(
            "[SIFIVE Backend] host-fallback scale_f32 '{}' rejected out-of-allocation range src={:p} dst={:p} nelements={}",
            kernel_name, src, dst, nelements
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }

    let src = std::slice::from_raw_parts(src as *const f32, nelements);
    let dst = std::slice::from_raw_parts_mut(dst, nelements);
    for i in 0..nelements {
        dst[i] = scale.mul_add(src[i], bias);
    }

    if sifive_env_truthy("HETGPU_SIFIVE_SCALE_F32_TRACE") {
        eprintln!(
            "[SIFIVE Backend] host-fallback scale_f32 '{}' nelements={}",
            kernel_name, nelements
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_concat_f32_cont_host_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    let x = sifive_host_ptr::<f32>(read_param_u64(kernel_params, 0)?)?;
    let y = sifive_host_ptr::<f32>(read_param_u64(kernel_params, 1)?)?;
    let dst = sifive_host_ptr::<f32>(read_param_u64(kernel_params, 2)?)?;
    let ne00 = read_param_i64(kernel_params, 3)?;
    let ne01 = read_param_i64(kernel_params, 4)?;
    let ne02 = read_param_i64(kernel_params, 5)?;
    let ne0 = read_param_i64(kernel_params, 6)?;
    let ne1 = read_param_i64(kernel_params, 7)?;
    let ne2 = read_param_i64(kernel_params, 8)?;
    if ne00 < 0 || ne01 < 0 || ne02 < 0 || ne0 < 0 || ne1 < 0 || ne2 < 0 {
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }
    let dim = if kernel_name.contains("ILi0") {
        0
    } else if kernel_name.contains("ILi1") {
        1
    } else if kernel_name.contains("ILi2") {
        2
    } else {
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    };
    let ne00 = ne00 as usize;
    let ne01 = ne01 as usize;
    let ne02 = ne02 as usize;
    let ne0 = ne0 as usize;
    let ne1 = ne1 as usize;
    let ne2 = ne2 as usize;
    let n = ne0.checked_mul(ne1)?.checked_mul(ne2)?;
    let (x_elems, y_elems) = match dim {
        0 => (
            ne00.checked_mul(ne1)?.checked_mul(ne2)?,
            ne0.checked_sub(ne00)?.checked_mul(ne1)?.checked_mul(ne2)?,
        ),
        1 => (
            ne0.checked_mul(ne01)?.checked_mul(ne2)?,
            ne0.checked_mul(ne1.checked_sub(ne01)?)?.checked_mul(ne2)?,
        ),
        _ => (
            ne0.checked_mul(ne1)?.checked_mul(ne02)?,
            ne0.checked_mul(ne1)?.checked_mul(ne2.checked_sub(ne02)?)?,
        ),
    };
    if !sifive_cuda_alloc_has_elems(x as *const f32, x_elems)
        || !sifive_cuda_alloc_has_elems(y as *const f32, y_elems)
        || !sifive_cuda_alloc_has_elems(dst as *const f32, n)
    {
        eprintln!(
            "[SIFIVE Backend] host-fallback concat_f32_cont '{}' rejected ranges x={:p}/{} y={:p}/{} dst={:p}/{}",
            kernel_name, x, x_elems, y, y_elems, dst, n
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }
    let x = std::slice::from_raw_parts(x as *const f32, x_elems);
    let y = std::slice::from_raw_parts(y as *const f32, y_elems);
    let dst = std::slice::from_raw_parts_mut(dst, n);
    for i in 0..n {
        match dim {
            0 => {
                let row = i / ne0;
                let i0 = i - row * ne0;
                dst[i] = if i0 < ne00 {
                    x[row * ne00 + i0]
                } else {
                    y[row * (ne0 - ne00) + (i0 - ne00)]
                };
            }
            1 => {
                let dst_plane = ne0 * ne1;
                let src0_plane = ne0 * ne01;
                let src1_plane = dst_plane - src0_plane;
                let i2 = i / dst_plane;
                let i01 = i - i2 * dst_plane;
                dst[i] = if i01 < src0_plane {
                    x[i2 * src0_plane + i01]
                } else {
                    y[i2 * src1_plane + (i01 - src0_plane)]
                };
            }
            _ => {
                let src0_size = ne0 * ne1 * ne02;
                dst[i] = if i < src0_size {
                    x[i]
                } else {
                    y[i - src0_size]
                };
            }
        }
    }
    if sifive_env_truthy("HETGPU_SIFIVE_CONCAT_F32_TRACE") {
        eprintln!(
            "[SIFIVE Backend] host-fallback concat_f32_cont '{}' dim={} n={}",
            kernel_name, dim, n
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_concat_f32_non_cont_host_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    let src0 = sifive_host_ptr::<u8>(read_param_u64(kernel_params, 0)?)?;
    let src1 = sifive_host_ptr::<u8>(read_param_u64(kernel_params, 1)?)?;
    let dst = sifive_host_ptr::<u8>(read_param_u64(kernel_params, 2)?)?;

    let ne00 = read_param_i64(kernel_params, 3)?;
    let ne01 = read_param_i64(kernel_params, 4)?;
    let ne02 = read_param_i64(kernel_params, 5)?;
    let ne03 = read_param_i64(kernel_params, 6)?;
    let nb00 = read_param_u64(kernel_params, 7)? as usize;
    let nb01 = read_param_u64(kernel_params, 8)? as usize;
    let nb02 = read_param_u64(kernel_params, 9)? as usize;
    let nb03 = read_param_u64(kernel_params, 10)? as usize;
    let ne10 = read_param_i64(kernel_params, 11)?;
    let ne11 = read_param_i64(kernel_params, 12)?;
    let ne12 = read_param_i64(kernel_params, 13)?;
    let ne13 = read_param_i64(kernel_params, 14)?;
    let nb10 = read_param_u64(kernel_params, 15)? as usize;
    let nb11 = read_param_u64(kernel_params, 16)? as usize;
    let nb12 = read_param_u64(kernel_params, 17)? as usize;
    let nb13 = read_param_u64(kernel_params, 18)? as usize;
    let ne0 = read_param_i64(kernel_params, 19)?;
    let ne1 = read_param_i64(kernel_params, 20)?;
    let ne2 = read_param_i64(kernel_params, 21)?;
    let ne3 = read_param_i64(kernel_params, 22)?;
    let nb0 = read_param_u64(kernel_params, 23)? as usize;
    let nb1 = read_param_u64(kernel_params, 24)? as usize;
    let nb2 = read_param_u64(kernel_params, 25)? as usize;
    let nb3 = read_param_u64(kernel_params, 26)? as usize;

    let dims = [
        ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3,
    ];
    if dims.iter().any(|&v| v < 0) {
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }
    if ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0 {
        return Some(Ok(()));
    }

    let dim = if kernel_name.contains("ILi0") {
        0
    } else if kernel_name.contains("ILi1") {
        1
    } else if kernel_name.contains("ILi2") {
        2
    } else if kernel_name.contains("ILi3") {
        3
    } else {
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    };

    let ne00 = ne00 as usize;
    let ne01 = ne01 as usize;
    let ne02 = ne02 as usize;
    let ne03 = ne03 as usize;
    let ne10 = ne10 as usize;
    let ne11 = ne11 as usize;
    let ne12 = ne12 as usize;
    let ne13 = ne13 as usize;
    let ne0 = ne0 as usize;
    let ne1 = ne1 as usize;
    let ne2 = ne2 as usize;
    let ne3 = ne3 as usize;

    fn tensor_bytes(
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ne3: usize,
        nb0: usize,
        nb1: usize,
        nb2: usize,
        nb3: usize,
    ) -> Option<usize> {
        if ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0 {
            return Some(0);
        }
        ne3.checked_sub(1)?
            .checked_mul(nb3)?
            .checked_add(ne2.checked_sub(1)?.checked_mul(nb2)?)?
            .checked_add(ne1.checked_sub(1)?.checked_mul(nb1)?)?
            .checked_add(ne0.checked_sub(1)?.checked_mul(nb0)?)?
            .checked_add(std::mem::size_of::<f32>())
    }

    let src0_bytes = tensor_bytes(ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03)?;
    let src1_bytes = tensor_bytes(ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13)?;
    let dst_bytes = tensor_bytes(ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3)?;
    if !sifive_cuda_alloc_has_bytes(src0 as u64, src0_bytes)
        || !sifive_cuda_alloc_has_bytes(src1 as u64, src1_bytes)
        || !sifive_cuda_alloc_has_bytes(dst as u64, dst_bytes)
    {
        eprintln!(
            "[SIFIVE Backend] host-fallback concat_f32_non_cont '{}' rejected ranges src0={:p}/{} src1={:p}/{} dst={:p}/{}",
            kernel_name, src0, src0_bytes, src1, src1_bytes, dst, dst_bytes
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }

    for i3 in 0..ne3 {
        for i2 in 0..ne2 {
            for i1 in 0..ne1 {
                for i0 in 0..ne0 {
                    let src_off = if i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03 {
                        i3.checked_mul(nb03)?
                            .checked_add(i2.checked_mul(nb02)?)?
                            .checked_add(i1.checked_mul(nb01)?)?
                            .checked_add(i0.checked_mul(nb00)?)?
                    } else {
                        match dim {
                            0 => i3
                                .checked_mul(nb13)?
                                .checked_add(i2.checked_mul(nb12)?)?
                                .checked_add(i1.checked_mul(nb11)?)?
                                .checked_add(i0.checked_sub(ne00)?.checked_mul(nb10)?)?,
                            1 => i3
                                .checked_mul(nb13)?
                                .checked_add(i2.checked_mul(nb12)?)?
                                .checked_add(i1.checked_sub(ne01)?.checked_mul(nb11)?)?
                                .checked_add(i0.checked_mul(nb10)?)?,
                            2 => i3
                                .checked_mul(nb13)?
                                .checked_add(i2.checked_sub(ne02)?.checked_mul(nb12)?)?
                                .checked_add(i1.checked_mul(nb11)?)?
                                .checked_add(i0.checked_mul(nb10)?)?,
                            _ => i3
                                .checked_sub(ne03)?
                                .checked_mul(nb13)?
                                .checked_add(i2.checked_mul(nb12)?)?
                                .checked_add(i1.checked_mul(nb11)?)?
                                .checked_add(i0.checked_mul(nb10)?)?,
                        }
                    };
                    let src_base = if i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03 {
                        src0
                    } else {
                        src1
                    };
                    let dst_off = i3
                        .checked_mul(nb3)?
                        .checked_add(i2.checked_mul(nb2)?)?
                        .checked_add(i1.checked_mul(nb1)?)?
                        .checked_add(i0.checked_mul(nb0)?)?;
                    let value = std::ptr::read_unaligned(src_base.add(src_off) as *const f32);
                    std::ptr::write_unaligned(dst.add(dst_off) as *mut f32, value);
                }
            }
        }
    }

    if sifive_env_truthy("HETGPU_SIFIVE_CONCAT_F32_TRACE") {
        eprintln!(
            "[SIFIVE Backend] host-fallback concat_f32_non_cont '{}' dim={} ne=({},{},{},{})",
            kernel_name, dim, ne0, ne1, ne2, ne3
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[derive(Clone, Copy)]
enum SifiveLlamaUnaryF32Op {
    Abs,
    Exp,
    Log,
    Neg,
    Relu,
    Sigmoid,
    Silu,
    Softplus,
    Sqrt,
    Tanh,
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_llama_unary_f32_op(kernel_name: &str) -> Option<SifiveLlamaUnaryF32Op> {
    let name = kernel_name.to_lowercase();
    if name.contains("op_softplus") {
        Some(SifiveLlamaUnaryF32Op::Softplus)
    } else if name.contains("op_exp") {
        Some(SifiveLlamaUnaryF32Op::Exp)
    } else if name.contains("op_neg") {
        Some(SifiveLlamaUnaryF32Op::Neg)
    } else if name.contains("op_sigmoid") {
        Some(SifiveLlamaUnaryF32Op::Sigmoid)
    } else if name.contains("op_silu") {
        Some(SifiveLlamaUnaryF32Op::Silu)
    } else if name.contains("op_log") {
        Some(SifiveLlamaUnaryF32Op::Log)
    } else if name.contains("op_tanh") {
        Some(SifiveLlamaUnaryF32Op::Tanh)
    } else if name.contains("op_relu") {
        Some(SifiveLlamaUnaryF32Op::Relu)
    } else if name.contains("op_sqrt") {
        Some(SifiveLlamaUnaryF32Op::Sqrt)
    } else if name.contains("op_abs") {
        Some(SifiveLlamaUnaryF32Op::Abs)
    } else {
        None
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_llama_unary_f32_apply(op: SifiveLlamaUnaryF32Op, x: f32) -> f32 {
    match op {
        SifiveLlamaUnaryF32Op::Abs => x.abs(),
        SifiveLlamaUnaryF32Op::Exp => x.exp(),
        SifiveLlamaUnaryF32Op::Log => x.ln(),
        SifiveLlamaUnaryF32Op::Neg => -x,
        SifiveLlamaUnaryF32Op::Relu => x.max(0.0),
        SifiveLlamaUnaryF32Op::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        SifiveLlamaUnaryF32Op::Silu => x / (1.0 + (-x).exp()),
        SifiveLlamaUnaryF32Op::Softplus => {
            if x > 20.0 {
                x
            } else {
                (1.0 + x.exp()).ln()
            }
        }
        SifiveLlamaUnaryF32Op::Sqrt => x.sqrt(),
        SifiveLlamaUnaryF32Op::Tanh => x.tanh(),
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_llama_unary_op_f32_host_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    if !kernel_name.contains("unary_op_kernel") || kernel_name.contains("unary_gated_op_kernel") {
        return None;
    }
    let op = sifive_llama_unary_f32_op(kernel_name)?;
    let src = sifive_host_ptr::<f32>(read_param_u64(kernel_params, 0)?)? as *const f32;
    let dst = sifive_host_ptr::<f32>(read_param_u64(kernel_params, 1)?)?;
    let k = read_param_i32(kernel_params, 2)?;
    if k < 0 {
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }
    let elems = k as usize;
    if !sifive_cuda_alloc_has_elems(src, elems)
        || !sifive_cuda_alloc_has_elems(dst as *const f32, elems)
    {
        eprintln!(
            "[SIFIVE Backend] host-fallback unary_op_f32 '{}' rejected ranges src={:p}/{} dst={:p}/{}",
            kernel_name, src, elems, dst, elems
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }
    for i in 0..elems {
        let value = sifive_llama_unary_f32_apply(op, *src.add(i));
        *dst.add(i) = value;
    }
    if sifive_env_truthy("HETGPU_SIFIVE_UNARY_F32_TRACE") {
        eprintln!(
            "[SIFIVE Backend] host-fallback unary_op_f32 '{}' elems={}",
            kernel_name, elems
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_llama_unary_gated_op_f32_host_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    if !kernel_name.contains("unary_gated_op_kernel") {
        return None;
    }
    let op = sifive_llama_unary_f32_op(kernel_name)?;
    let x = sifive_host_ptr::<f32>(read_param_u64(kernel_params, 0)?)? as *const f32;
    let g = sifive_host_ptr::<f32>(read_param_u64(kernel_params, 1)?)? as *const f32;
    let dst = sifive_host_ptr::<f32>(read_param_u64(kernel_params, 2)?)?;
    let k = read_param_i64(kernel_params, 3)?;
    let n = read_param_i64(kernel_params, 4)?;
    let o0 = read_param_i64(kernel_params, 5)?;
    let o1 = read_param_i64(kernel_params, 6)?;
    if k < 0 || n <= 0 || o0 < 0 || o1 < 0 {
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }
    let k = k as usize;
    let n = n as usize;
    let o0 = o0 as usize;
    let o1 = o1 as usize;
    if k == 0 {
        return Some(Ok(()));
    }
    let rows = k.checked_add(n - 1)?.checked_div(n)?;
    let x_elems = rows.checked_sub(1)?.checked_mul(o0)?.checked_add(n)?;
    let g_elems = rows.checked_sub(1)?.checked_mul(o1)?.checked_add(n)?;
    if !sifive_cuda_alloc_has_elems(x, x_elems)
        || !sifive_cuda_alloc_has_elems(g, g_elems)
        || !sifive_cuda_alloc_has_elems(dst as *const f32, k)
    {
        eprintln!(
            "[SIFIVE Backend] host-fallback unary_gated_f32 '{}' rejected ranges x={:p}/{} g={:p}/{} dst={:p}/{}",
            kernel_name, x, x_elems, g, g_elems, dst, k
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }
    for i in 0..k {
        let row = i / n;
        let col = i % n;
        let j0 = row.checked_mul(o0)?.checked_add(col)?;
        let j1 = row.checked_mul(o1)?.checked_add(col)?;
        let value = sifive_llama_unary_f32_apply(op, *x.add(j0)) * *g.add(j1);
        *dst.add(i) = value;
    }
    if sifive_env_truthy("HETGPU_SIFIVE_UNARY_F32_TRACE") {
        eprintln!(
            "[SIFIVE Backend] host-fallback unary_gated_f32 '{}' elems={} n={}",
            kernel_name, k, n
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_l2_norm_f32_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
) -> Option<cuda_types::cuda::CUresult> {
    let src = sifive_host_ptr::<f32>(read_param_u64(kernel_params, 0)?)?;
    let dst = sifive_host_ptr::<f32>(read_param_u64(kernel_params, 1)?)?;
    let ncols = read_param_i32(kernel_params, 2)?.max(0) as usize;
    let stride_row = read_param_i64(kernel_params, 3)?.max(0) as usize;
    let stride_channel = read_param_i64(kernel_params, 4)?.max(0) as usize;
    let stride_sample = read_param_i64(kernel_params, 5)?.max(0) as usize;
    let eps = read_param_f32(kernel_params, 6)?.max(0.0);

    if ncols == 0 {
        return Some(Ok(()));
    }

    let nrows = (grid_dim_x as usize).max(1);
    let nchannels = (grid_dim_y as usize).max(1);
    let nsamples = (grid_dim_z as usize).max(1);
    let src_elems = nsamples
        .checked_sub(1)
        .and_then(|v| v.checked_mul(stride_sample))
        .and_then(|v| {
            nchannels
                .checked_sub(1)
                .and_then(|c| c.checked_mul(stride_channel))
                .and_then(|c| v.checked_add(c))
        })
        .and_then(|v| {
            nrows
                .checked_sub(1)
                .and_then(|r| r.checked_mul(stride_row))
                .and_then(|r| v.checked_add(r))
        })
        .and_then(|v| v.checked_add(ncols))?;
    let dst_elems = nsamples
        .checked_mul(nchannels)?
        .checked_mul(nrows)?
        .checked_mul(ncols)?;
    if !sifive_cuda_alloc_has_elems(src as *const f32, src_elems)
        || !sifive_cuda_alloc_has_elems(dst as *const f32, dst_elems)
    {
        eprintln!(
            "[SIFIVE Backend] host-fallback l2_norm_f32 '{}' rejected out-of-allocation range src={:p} dst={:p} src_elems={} dst_elems={} ncols={} grid={}/{}/{}",
            kernel_name, src, dst, src_elems, dst_elems, ncols, nrows, nchannels, nsamples
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }

    for sample in 0..nsamples {
        for channel in 0..nchannels {
            for row in 0..nrows {
                let src_row =
                    src.add(sample * stride_sample + channel * stride_channel + row * stride_row);
                let dst_row = dst.add(((sample * nchannels + channel) * nrows + row) * ncols);
                let mut sumsq = 0.0f32;
                for col in 0..ncols {
                    let x = *src_row.add(col);
                    sumsq += x * x;
                }
                let scale = 1.0f32 / sumsq.max(eps * eps).sqrt();
                for col in 0..ncols {
                    *dst_row.add(col) = *src_row.add(col) * scale;
                }
            }
        }
    }

    if sifive_env_truthy("HETGPU_SIFIVE_L2_NORM_TRACE") {
        eprintln!(
            "[SIFIVE Backend] host-fallback l2_norm_f32 '{}' ncols={} stride_row={} stride_channel={} stride_sample={}",
            kernel_name, ncols, stride_row, stride_channel, stride_sample
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_get_rows_float_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: ::core::ffi::c_uint,
) -> Option<cuda_types::cuda::CUresult> {
    if sifive_env_enabled_default("HETGPU_SIFIVE_GET_ROWS_FLOAT_NOOP", false) {
        if sifive_env_truthy("HETGPU_SIFIVE_GET_ROWS_FLOAT_TRACE") {
            eprintln!(
                "[SIFIVE Backend] k_get_rows_float '{}' no-op by HETGPU_SIFIVE_GET_ROWS_FLOAT_NOOP",
                kernel_name
            );
        }
        return Some(Ok(()));
    }

    let src0 = sifive_host_ptr::<f32>(read_param_u64(kernel_params, 0)?)? as *const f32;
    let src1 = sifive_host_ptr::<i32>(read_param_u64(kernel_params, 1)?)? as *const i32;
    let dst = sifive_host_ptr::<f32>(read_param_u64(kernel_params, 2)?)?;
    let ne00 = read_param_i64(kernel_params, 3)?.max(0) as usize;
    let ne11 = read_param_i64(kernel_params, 4)?.max(0) as usize;
    let ne12 = read_param_i64(kernel_params, 5)?.max(0) as usize;
    let s1 = read_param_u64(kernel_params, 6)? as usize;
    let s2 = read_param_u64(kernel_params, 7)? as usize;
    let s3 = read_param_u64(kernel_params, 8)? as usize;
    let nb01 = read_param_u64(kernel_params, 9)? as usize;
    let nb02 = read_param_u64(kernel_params, 10)? as usize;
    let nb03 = read_param_u64(kernel_params, 11)? as usize;
    let s10 = read_param_u64(kernel_params, 12)? as usize;
    let s11 = read_param_u64(kernel_params, 13)? as usize;
    let s12 = read_param_u64(kernel_params, 14)? as usize;

    let ne10 = (grid_dim_x as usize).max(1);
    let dst_elems = ne10
        .checked_sub(1)
        .and_then(|v| v.checked_mul(s1))
        .and_then(|v| {
            ne11.checked_sub(1)
                .and_then(|x| x.checked_mul(s2))
                .and_then(|x| v.checked_add(x))
        })
        .and_then(|v| {
            ne12.checked_sub(1)
                .and_then(|x| x.checked_mul(s3))
                .and_then(|x| v.checked_add(x))
        })
        .and_then(|v| v.checked_add(ne00))?;
    let idx_elems = ne10
        .checked_sub(1)
        .and_then(|v| v.checked_mul(s10))
        .and_then(|v| {
            ne11.checked_sub(1)
                .and_then(|x| x.checked_mul(s11))
                .and_then(|x| v.checked_add(x))
        })
        .and_then(|v| {
            ne12.checked_sub(1)
                .and_then(|x| x.checked_mul(s12))
                .and_then(|x| v.checked_add(x))
        })
        .and_then(|v| v.checked_add(1))?;
    if !sifive_cuda_alloc_has_elems(src1 as *const i32, idx_elems)
        || !sifive_cuda_alloc_has_elems(dst as *const f32, dst_elems)
    {
        eprintln!(
            "[SIFIVE Backend] host-fallback k_get_rows_float '{}' rejected out-of-allocation index/dst range src1={:p} dst={:p} idx_elems={} dst_elems={} ne00={} ne10={} ne11={} ne12={}",
            kernel_name, src1, dst, idx_elems, dst_elems, ne00, ne10, ne11, ne12
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }

    for i12 in 0..ne12 {
        for i11 in 0..ne11 {
            for i10 in 0..ne10 {
                let row_index_ptr = src1.add(i10 * s10 + i11 * s11 + i12 * s12);
                let i01 = (*row_index_ptr).max(0) as usize;
                let dst_row = dst.add(i10 * s1 + i11 * s2 + i12 * s3);
                let src0_row =
                    (src0 as *const u8).add(i01 * nb01 + i11 * nb02 + i12 * nb03) as *const f32;
                if !sifive_cuda_alloc_has_elems(src0_row, ne00) {
                    if sifive_env_enabled_default("HETGPU_SIFIVE_GET_ROWS_FLOAT_CLAMP_OOB", true) {
                        let fallback_row =
                            (src0 as *const u8).add(i11 * nb02 + i12 * nb03) as *const f32;
                        if sifive_cuda_alloc_has_elems(fallback_row, ne00) {
                            for i00 in 0..ne00 {
                                *dst_row.add(i00) = *fallback_row.add(i00);
                            }
                        } else {
                            for i00 in 0..ne00 {
                                *dst_row.add(i00) = 0.0;
                            }
                        }
                        continue;
                    }
                    eprintln!(
                        "[SIFIVE Backend] host-fallback k_get_rows_float '{}' rejected source row outside allocation src0={:p} row={:p} idx={} ne00={}",
                        kernel_name, src0, src0_row, i01, ne00
                    );
                    return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
                }
                for i00 in 0..ne00 {
                    *dst_row.add(i00) = *src0_row.add(i00);
                }
            }
        }
    }

    if sifive_env_truthy("HETGPU_SIFIVE_GET_ROWS_FLOAT_TRACE") {
        eprintln!(
            "[SIFIVE Backend] host-fallback k_get_rows_float '{}' ne00={} ne11={} ne12={}",
            kernel_name, ne00, ne11, ne12
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn execute_ssm_conv_f32_host_fallback(
    kernel_name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
) -> Option<cuda_types::cuda::CUresult> {
    let src0_addr = read_param_u64(kernel_params, 0)?;
    let src1_addr = read_param_u64(kernel_params, 1)?;
    let bias_addr = read_param_u64(kernel_params, 2).unwrap_or(0);
    let dst_addr = read_param_u64(kernel_params, 7)?;
    let src0 = sifive_host_ptr::<f32>(src0_addr)? as *const f32;
    let src1 = sifive_host_ptr::<f32>(src1_addr)? as *const f32;
    let bias = if bias_addr == 0 {
        std::ptr::null::<f32>()
    } else {
        sifive_host_ptr::<f32>(bias_addr)? as *const f32
    };
    let dst = sifive_host_ptr::<f32>(dst_addr)?;

    let src0_nb0 = read_param_i32(kernel_params, 3)?;
    let src0_nb1 = read_param_i32(kernel_params, 4)?;
    let src0_nb2 = read_param_i32(kernel_params, 5)?;
    let src1_nb1 = read_param_i32(kernel_params, 6)?;
    let dst_nb0 = read_param_i32(kernel_params, 8)?;
    let dst_nb1 = read_param_i32(kernel_params, 9)?;
    let dst_nb2 = read_param_i32(kernel_params, 10)?;
    let n_t = read_param_i64(kernel_params, 11)?.max(0) as usize;
    if src0_nb0 <= 0
        || src0_nb1 <= 0
        || src0_nb2 <= 0
        || src1_nb1 <= 0
        || dst_nb0 <= 0
        || dst_nb1 <= 0
        || dst_nb2 <= 0
    {
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }
    if n_t == 0 {
        return Some(Ok(()));
    }

    let (split_d_inner, d_conv, split_n_t) =
        sifive_parse_ssm_conv_template(kernel_name).unwrap_or((128, 4, 0));
    let split_d_inner = split_d_inner.max(1) as usize;
    let d_conv = d_conv.max(1) as usize;
    let split_n_t = split_n_t as usize;
    let grid_x = (grid_dim_x as usize).max(1);
    let grid_y = (grid_dim_y as usize).max(1);
    let grid_z = (grid_dim_z as usize).max(1);
    let rows = grid_y.checked_mul(split_d_inner)?;
    let src0_nb0 = src0_nb0 as usize;
    let src0_nb1 = src0_nb1 as usize;
    let src0_nb2 = src0_nb2 as usize;
    let src1_nb1 = src1_nb1 as usize;
    let dst_nb0 = dst_nb0 as usize;
    let dst_nb1 = dst_nb1 as usize;
    let dst_nb2 = dst_nb2 as usize;
    let src0_stride_x = src0_nb1.checked_div(std::mem::size_of::<f32>())?;
    let src1_stride_w = src1_nb1.checked_div(std::mem::size_of::<f32>())?;
    let dst_stride_y = dst_nb1.checked_div(std::mem::size_of::<f32>())?;
    let is_long = split_n_t != 0;
    let tile_n = if is_long { split_n_t.max(1) } else { n_t };
    let per_tile_x = tile_n.checked_add(d_conv)?.checked_sub(1)?;
    let dst_steps = if is_long {
        n_t.min(grid_z.checked_mul(tile_n)?)
    } else {
        n_t
    };

    let src0_bytes = grid_x
        .checked_sub(1)?
        .checked_mul(src0_nb2)?
        .checked_add(rows.checked_sub(1)?.checked_mul(src0_nb1)?)?
        .checked_add(if is_long {
            grid_z
                .checked_sub(1)?
                .checked_mul(tile_n)?
                .checked_mul(src0_nb0)?
        } else {
            0
        })?
        .checked_add(per_tile_x.checked_sub(1)?.checked_mul(src0_nb0)?)?
        .checked_add(std::mem::size_of::<f32>())?;
    let src1_bytes = rows
        .checked_sub(1)?
        .checked_mul(src1_nb1)?
        .checked_add(d_conv.checked_mul(std::mem::size_of::<f32>())?)?;
    let dst_bytes = grid_x
        .checked_sub(1)?
        .checked_mul(dst_nb2)?
        .checked_add(rows.checked_sub(1)?.checked_mul(dst_nb0)?)?
        .checked_add(dst_steps.checked_sub(1)?.checked_mul(dst_nb1)?)?
        .checked_add(std::mem::size_of::<f32>())?;
    let bias_bytes = rows.checked_mul(std::mem::size_of::<f32>())?;
    if !sifive_host_or_cuda_alloc_has_bytes(src0_addr, src0_bytes, false)
        || !sifive_host_or_cuda_alloc_has_bytes(src1_addr, src1_bytes, false)
        || !sifive_host_or_cuda_alloc_has_bytes(dst_addr, dst_bytes, true)
        || (!bias.is_null() && !sifive_host_or_cuda_alloc_has_bytes(bias_addr, bias_bytes, false))
    {
        eprintln!(
            "[SIFIVE Backend] host-fallback ssm_conv '{}' rejected ranges src0=0x{:x}/{} src1=0x{:x}/{} bias=0x{:x}/{} dst=0x{:x}/{} grid={}/{}/{} split={} d_conv={} n_t={}",
            kernel_name,
            src0_addr,
            src0_bytes,
            src1_addr,
            src1_bytes,
            bias_addr,
            if bias.is_null() { 0 } else { bias_bytes },
            dst_addr,
            dst_bytes,
            grid_x,
            grid_y,
            grid_z,
            split_d_inner,
            d_conv,
            n_t
        );
        return Some(Err(cuda_types::cuda::CUerror::UNKNOWN));
    }

    let apply_silu = kernel_name.contains("ssm_conv_f32ILb1")
        || kernel_name.contains("ssm_conv_long_token_f32ILb1");
    for bidx in 0..grid_x {
        for bidy in 0..grid_y {
            for bidz in 0..grid_z {
                let local_n_t = if is_long {
                    let start = bidz.checked_mul(tile_n)?;
                    if start >= n_t {
                        continue;
                    }
                    (n_t - start).min(tile_n)
                } else {
                    n_t
                };
                let x_block = (src0 as *const u8)
                    .add(bidx * src0_nb2 + bidy * split_d_inner * src0_nb1)
                    .add(if is_long { bidz * tile_n * src0_nb0 } else { 0 })
                    as *const f32;
                let w_block =
                    (src1 as *const u8).add(bidy * split_d_inner * src1_nb1) as *const f32;
                let y_block = (dst as *mut u8)
                    .add(bidx * dst_nb2)
                    .add(if is_long { bidz * tile_n * dst_nb1 } else { 0 })
                    .add(bidy * split_d_inner * dst_nb0) as *mut f32;
                for tid in 0..split_d_inner {
                    let bias_value = if bias.is_null() {
                        0.0
                    } else {
                        *bias.add(bidy * split_d_inner + tid)
                    };
                    for i in 0..local_n_t {
                        let mut sumf = bias_value;
                        for j in 0..d_conv {
                            sumf += *x_block.add(tid * src0_stride_x + i + j)
                                * *w_block.add(tid * src1_stride_w + j);
                        }
                        if apply_silu {
                            sumf = sumf / (1.0 + (-sumf).exp());
                        }
                        *y_block.add(i * dst_stride_y + tid) = sumf;
                    }
                }
            }
        }
    }

    if sifive_env_truthy("HETGPU_SIFIVE_SSM_CONV_TRACE") {
        eprintln!(
            "[SIFIVE Backend] host-fallback ssm_conv '{}' grid={}/{}/{} split={} d_conv={} n_t={} silu={}",
            kernel_name, grid_x, grid_y, grid_z, split_d_inner, d_conv, n_t, apply_silu
        );
    }
    Some(Ok(()))
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn current_sifive_device_id_or_zero() -> i32 {
    let _ = super::driver::global_state();
    if let Some(forced) = std::env::var("HETGPU_SIFIVE_FORCE_DEVICE")
        .ok()
        .or_else(|| std::env::var("HETGPU_SIFIVE_DEVICE").ok())
        .and_then(|v| v.parse::<i32>().ok())
    {
        if (0..4).contains(&forced) {
            return forced;
        }
    }
    let device_id = super::context::get_current_sifive()
        .map(|ctx| ctx.device_id)
        .unwrap_or_else(|_| super::driver::sifive_physical_device_for_logical(0));
    if (0..4).contains(&device_id) {
        device_id
    } else {
        0
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_env_enabled_default(name: &str, default_value: bool) -> bool {
    let value = match std::env::var(name) {
        Ok(value) if !value.trim().is_empty() => value,
        _ => return default_value,
    };
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "0" | "false" | "no" | "off"
    )
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_parse_env_u64_default(name: &str, default_value: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| {
            let trimmed = value.trim();
            if let Some(hex) = trimmed
                .strip_prefix("0x")
                .or_else(|| trimmed.strip_prefix("0X"))
            {
                u64::from_str_radix(hex, 16).ok()
            } else {
                trimmed.parse::<u64>().ok()
            }
        })
        .unwrap_or(default_value)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_driver_kernel_noop_enabled() -> bool {
    sifive_env_enabled_default("HETGPU_CUDART_KERNEL_SIFIVE_NOOP", false)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_generic_kernel_fast_success_enabled() -> bool {
    sifive_env_enabled_default("HETGPU_CUDART_GENERIC_KERNEL_FAST_SUCCESS", false)
        || sifive_env_enabled_default("HETGPU_SIFIVE_GENERIC_KERNEL_FAST_SUCCESS", false)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_driver_kernel_noop_every() -> u64 {
    let every = sifive_parse_env_u64_default("HETGPU_CUDART_KERNEL_SIFIVE_NOOP_EVERY", 0);
    let every = if every == 0 {
        sifive_parse_env_u64_default("HETGPU_SIFIVE_KERNEL_NOOP_EVERY", 1)
    } else {
        every
    };
    every.max(1)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_driver_kernel_noop_first() -> u64 {
    sifive_parse_env_u64_default("HETGPU_CUDART_KERNEL_SIFIVE_NOOP_FIRST", 4)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_named_fail_open_enabled() -> bool {
    let value = std::env::var("HETGPU_SIFIVE_NAMED_FAIL_OPEN")
        .or_else(|_| std::env::var("HETGPU_SIFIVE_ASSUME_SUCCESS_ON_WAIT_ERROR"));
    matches!(
        value
            .ok()
            .map(|value| value.trim().to_ascii_lowercase()),
        Some(value)
            if value == "1"
                || value == "true"
                || value == "yes"
                || value == "on"
    )
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_env_truthy(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(false)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_rmsnorm_delivery_noop_enabled() -> bool {
    sifive_env_enabled_default("HETGPU_SIFIVE_RMSNORM_NOOP", false)
        || sifive_env_enabled_default("HETGPU_SIFIVE_DELIVERY_SKIP_RMSNORM", false)
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_log_limited(
    counter: &AtomicU64,
    limit_env: &str,
    default_limit: u64,
    log_line: impl FnOnce(),
) {
    let limit = sifive_parse_env_u64_default(limit_env, default_limit);
    let index = counter.fetch_add(1, Ordering::Relaxed);
    if index < limit {
        log_line();
    } else if index == limit && limit != 0 {
        eprintln!(
            "[SIFIVE Backend] {}={} reached; suppressing further repeated messages",
            limit_env, limit
        );
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
fn sifive_named_assume_success(
    reason: &str,
    kernel_name: &str,
) -> Option<cuda_types::cuda::CUresult> {
    if sifive_named_fail_open_enabled() {
        sifive_log_limited(
            &SIFIVE_NAMED_FAILOPEN_LOG_COUNT,
            "HETGPU_SIFIVE_NAMED_FAILOPEN_LOG_LIMIT",
            64,
            || {
                eprintln!(
                    "[SIFIVE Backend] assuming named-kernel success for '{}' after {}",
                    kernel_name, reason
                );
            },
        );
        Some(Ok(()))
    } else {
        Some(Err(cuda_types::cuda::CUerror::UNKNOWN))
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn try_offload_named_sifive_kernel(
    kernel_name: &str,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Option<cuda_types::cuda::CUresult> {
    use cuda_types::cuda::*;

    let name_lower = kernel_name.to_lowercase();
    let named_sifive_enabled = std::env::var("HETGPU_SIFIVE_OFFLOAD_NAMED_KERNELS")
        .ok()
        .map(|v| v != "0")
        .unwrap_or(true);
    if !named_sifive_enabled {
        return None;
    }
    let allow_named_host_fallback = std::env::var("HETGPU_SIFIVE_ALLOW_NAMED_HOST_FALLBACK")
        .ok()
        .as_deref()
        == Some("1");
    if name_lower.contains("quantize_q8_1")
        && sifive_env_enabled_default("HETGPU_SIFIVE_QUANTIZE_Q8_1_HOST_FALLBACK", true)
    {
        return execute_quantize_q8_1_host_fallback(kernel_name, grid_dim_z, kernel_params);
    }
    if name_lower.contains("scale_f32")
        && sifive_env_enabled_default("HETGPU_SIFIVE_SCALE_F32_HOST_FALLBACK", true)
    {
        return execute_scale_f32_fallback(kernel_name, kernel_params);
    }
    if name_lower.contains("concat_f32_cont")
        && sifive_env_enabled_default("HETGPU_SIFIVE_CONCAT_F32_HOST_FALLBACK", true)
    {
        return execute_concat_f32_cont_host_fallback(kernel_name, kernel_params);
    }
    if name_lower.contains("concat_f32_non_cont")
        && sifive_env_enabled_default("HETGPU_SIFIVE_CONCAT_F32_HOST_FALLBACK", true)
    {
        return execute_concat_f32_non_cont_host_fallback(kernel_name, kernel_params);
    }
    if name_lower.contains("unary_op_kernel")
        && (sifive_env_enabled_default("HETGPU_SIFIVE_LLAMA_UNARY_HOST_FALLBACK", true)
            || allow_named_host_fallback)
    {
        if let Some(result) = execute_llama_unary_op_f32_host_fallback(kernel_name, kernel_params) {
            return Some(result);
        }
        if let Some(result) =
            execute_llama_unary_gated_op_f32_host_fallback(kernel_name, kernel_params)
        {
            return Some(result);
        }
    }
    if (name_lower.contains("ssm_conv_f32") || name_lower.contains("ssm_conv_long_token_f32"))
        && sifive_env_enabled_default("HETGPU_SIFIVE_SSM_CONV_HOST_FALLBACK", true)
    {
        return execute_ssm_conv_f32_host_fallback(
            kernel_name,
            kernel_params,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
        );
    }
    if name_lower.contains("mul_mat_f") && !name_lower.contains("mul_mat_vec_f") {
        if let Some(result) = try_offload_mul_mat_f_named_sifive_kernel(
            kernel_name,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            kernel_params,
        ) {
            return Some(result);
        }
    }
    if name_lower.contains("mul_mat_vec_f") {
        if let Some(result) = try_offload_mmvf_named_sifive_kernel(
            kernel_name,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            kernel_params,
        ) {
            return Some(result);
        }
    }
    if name_lower.contains("mul_mat_vec_q")
        && sifive_env_truthy("HETGPU_SIFIVE_MMVQ_NAMED_FAIL_OPEN")
    {
        return sifive_named_assume_success("MMVQ named fail-open requested", kernel_name);
    }
    if name_lower.contains("softmax_warp_forward") {
        let (src, dst, rows, cols, stride, dtype) =
            match read_pytorch_softmax_warp_forward_args(kernel_params) {
                Some(args) => args,
                None => return None,
            };
        let _ = dtype;
        let dev_id = current_sifive_device_id_or_zero();
        let rc = launch_pytorch_softmax_warp_forward_elf(dev_id, src, dst, rows, cols, stride);
        if rc == 0 {
            eprintln!(
                "[SIFIVE Backend] offloaded PyTorch softmax_warp_forward '{}' via SIFIVE ELF dev={} rows={} cols={} stride={}",
                kernel_name, dev_id, rows, cols, stride
            );
            return Some(Ok(()));
        }
        eprintln!(
            "[SIFIVE Backend] PyTorch softmax_warp_forward '{}' SIFIVE ELF offload failed rc={} rows={} cols={} stride={}",
            kernel_name, rc, rows, cols, stride
        );
        return Some(Err(CUerror::UNKNOWN));
    }
    if name_lower.contains("softmax") || name_lower.contains("soft_max") {
        let (x, mask, sinks, dst, params) = match read_softmax_named_args(kernel_params) {
            Some(args) => args,
            None => return None,
        };
        let named_softmax_enabled = std::env::var("HETGPU_SIFIVE_SOFTMAX_NAMED_OFFLOAD")
            .ok()
            .as_deref()
            == Some("1");
        let allow_host_fallback = allow_named_host_fallback
            || std::env::var("HETGPU_SIFIVE_SOFTMAX_HOST_FALLBACK")
                .ok()
                .as_deref()
                == Some("1");
        let rows = (params.ne01.max(1) as u64)
            .saturating_mul(params.ne02.max(1) as u64)
            .saturating_mul(params.ne03.max(1) as u64);
        let cols = params.ncols.max(1) as u64;
        let can_sifive_simple =
            mask.is_null() && sinks.is_null() && params.scale == 1.0 && params.max_bias == 0.0;
        if named_softmax_enabled && can_sifive_simple {
            let dev_id = current_sifive_device_id_or_zero();
            let rc = sifive_runtime_sys::hetgpu_sifive_submit_softmax_on(
                dev_id,
                x,
                dst,
                rows,
                cols,
                cols,
                sifive_runtime_sys::SifiveDataType::Float32 as i32,
            );
            if rc == 0 {
                eprintln!(
                    "[SIFIVE Backend] offloaded softmax '{}' dev={} rows={} cols={}",
                    kernel_name, dev_id, rows, cols
                );
                return Some(Ok(()));
            }
        }
        if allow_host_fallback {
            return execute_softmax_f32_host_fallback(
                kernel_name,
                x,
                mask,
                sinks,
                dst,
                params,
                name_lower.contains("__half"),
            );
        }
        return None;
    }

    if name_lower.contains("rmsnorm") || name_lower.contains("rms_norm") {
        let allow_normal_fallback = std::env::var("HETGPU_SIFIVE_RMSNORM_ALLOW_NORMAL_FALLBACK")
            .ok()
            .as_deref()
            == Some("1");
        if SIFIVE_RMSNORM_OFFLOAD_DISABLED_AFTER_FAILURE.load(Ordering::Relaxed) {
            if sifive_named_fail_open_enabled() {
                return sifive_named_assume_success(
                    "RMSNorm offload disabled after prior failure",
                    kernel_name,
                );
            }
            if let Some((x, weight, y, rows, hidden, eps)) = read_rmsnorm_named_offload_args(
                kernel_name,
                kernel_params,
                grid_dim_x,
                grid_dim_y,
                grid_dim_z,
            ) {
                return execute_rmsnorm_f32_host_fallback(
                    kernel_name,
                    x,
                    weight,
                    y,
                    rows,
                    hidden,
                    eps,
                );
            };
            return if allow_normal_fallback {
                None
            } else {
                Some(Err(CUerror::UNKNOWN))
            };
        }
        let (x, weight, y, rows, hidden, eps) = match read_rmsnorm_named_offload_args(
            kernel_name,
            kernel_params,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
        ) {
            Some(args) => args,
            None => {
                let hidden = read_param_i32(kernel_params, 2).unwrap_or(0).max(0) as u64;
                if hidden == 0 {
                    sifive_log_limited(
                        &SIFIVE_NAMED_ERROR_LOG_COUNT,
                        "HETGPU_SIFIVE_NAMED_ERROR_LOG_LIMIT",
                        64,
                        || {
                            eprintln!(
                                "[SIFIVE Backend] RMSNorm '{}' missing hidden size",
                                kernel_name
                            );
                        },
                    );
                }
                if sifive_named_fail_open_enabled() {
                    return sifive_named_assume_success(
                        "RMSNorm args could not be parsed",
                        kernel_name,
                    );
                }
                if allow_normal_fallback {
                    return None;
                }
                sifive_log_limited(
                    &SIFIVE_NAMED_ERROR_LOG_COUNT,
                    "HETGPU_SIFIVE_NAMED_ERROR_LOG_LIMIT",
                    64,
                    || {
                        eprintln!(
                            "[SIFIVE Backend] RMSNorm '{}' cannot be parsed for named offload; refusing normal launch to avoid skipped/empty-ELF output",
                            kernel_name
                        );
                    },
                );
                return sifive_named_assume_success(
                    "RMSNorm args could not be parsed",
                    kernel_name,
                );
            }
        };
        if hidden == 0 {
            sifive_log_limited(
                &SIFIVE_NAMED_ERROR_LOG_COUNT,
                "HETGPU_SIFIVE_NAMED_ERROR_LOG_LIMIT",
                64,
                || {
                    eprintln!(
                        "[SIFIVE Backend] RMSNorm '{}' missing hidden size",
                        kernel_name
                    );
                },
            );
            if sifive_named_fail_open_enabled() {
                return sifive_named_assume_success("RMSNorm hidden size is zero", kernel_name);
            }
            return if allow_normal_fallback {
                None
            } else {
                Some(Err(CUerror::UNKNOWN))
            };
        }
        if std::env::var("HETGPU_SIFIVE_RMSNORM_HOST_FALLBACK")
            .ok()
            .map(|v| v != "0")
            .unwrap_or(false)
        {
            return execute_rmsnorm_f32_host_fallback(kernel_name, x, weight, y, rows, hidden, eps);
        }
        let dtype = if name_lower.contains("bf16") || name_lower.contains("bfloat16") {
            sifive_runtime_sys::SifiveDataType::Bfloat16 as i32
        } else {
            sifive_runtime_sys::SifiveDataType::Float32 as i32
        };
        let elem_size = if dtype == sifive_runtime_sys::SifiveDataType::Float32 as i32 {
            std::mem::size_of::<f32>()
        } else {
            std::mem::size_of::<u16>()
        };
        let total_bytes = (rows as usize)
            .checked_mul(hidden as usize)
            .and_then(|v| v.checked_mul(elem_size))
            .unwrap_or(usize::MAX);
        let weight_bytes = (hidden as usize)
            .checked_mul(elem_size)
            .unwrap_or(usize::MAX);
        if total_bytes == usize::MAX
            || !sifive_host_or_cuda_alloc_has_bytes(x as u64, total_bytes, false)
            || !sifive_host_or_cuda_alloc_has_bytes(y as u64, total_bytes, true)
            || (!weight.is_null()
                && !sifive_host_or_cuda_alloc_has_bytes(weight as u64, weight_bytes, false))
        {
            sifive_log_limited(
                &SIFIVE_NAMED_ERROR_LOG_COUNT,
                "HETGPU_SIFIVE_NAMED_ERROR_LOG_LIMIT",
                64,
                || {
                    eprintln!(
                        "[SIFIVE Backend] RMSNorm '{}' rejected out-of-allocation range x={:p} w={:p} y={:p} rows={} hidden={} bytes={}",
                        kernel_name, x, weight, y, rows, hidden, total_bytes
                    );
                },
            );
            return sifive_named_assume_success(
                "RMSNorm allocation range check failed",
                kernel_name,
            );
        }
        let rmsnorm_min_hidden =
            sifive_parse_env_u64_default("HETGPU_SIFIVE_RMSNORM_OFFLOAD_MIN_HIDDEN", 1024);
        if hidden < rmsnorm_min_hidden
            || !sifive_env_enabled_default("HETGPU_SIFIVE_RMSNORM_OFFLOAD", true)
        {
            sifive_log_limited(
                &SIFIVE_NAMED_ERROR_LOG_COUNT,
                "HETGPU_SIFIVE_NAMED_ERROR_LOG_LIMIT",
                64,
                || {
                    eprintln!(
                        "[SIFIVE Backend] RMSNorm '{}' hidden={} uses host path before SIFIVE submit (min_hidden={})",
                        kernel_name, hidden, rmsnorm_min_hidden
                    );
                },
            );
            if let Some(result) =
                execute_rmsnorm_f32_host_fallback(kernel_name, x, weight, y, rows, hidden, eps)
            {
                return Some(result);
            }
            return if allow_normal_fallback {
                None
            } else {
                Some(Err(CUerror::UNKNOWN))
            };
        }
        let dev_id = current_sifive_device_id_or_zero();
        let rc = sifive_runtime_sys::hetgpu_sifive_submit_rmsnorm_on(
            dev_id, x, weight, y, rows, hidden, eps, dtype,
        );
        if rc == 0 {
            if std::env::var("HETGPU_SIFIVE_LOG_NAMED_OFFLOADS")
                .ok()
                .as_deref()
                == Some("1")
            {
                eprintln!(
                    "[SIFIVE Backend] offloaded RMSNorm '{}' dev={} rows={} hidden={} eps={} dtype={} ",
                    kernel_name, dev_id, rows, hidden, eps, dtype
                );
            }
            return Some(Ok(()));
        }
        if !SIFIVE_RMSNORM_OFFLOAD_DISABLED_AFTER_FAILURE.swap(true, Ordering::Relaxed) {
            sifive_log_limited(
                &SIFIVE_NAMED_ERROR_LOG_COUNT,
                "HETGPU_SIFIVE_NAMED_ERROR_LOG_LIMIT",
                64,
                || {
                    eprintln!(
                        "[SIFIVE Backend] RMSNorm '{}' offload failed with rc={}; refusing host fallback unless HETGPU_SIFIVE_ALLOW_NAMED_HOST_FALLBACK=1",
                        kernel_name, rc
                    );
                },
            );
        }
        if sifive_named_fail_open_enabled() {
            return sifive_named_assume_success("RMSNorm SIFIVE offload failed", kernel_name);
        }
        if std::env::var("HETGPU_SIFIVE_ALLOW_NAMED_HOST_FALLBACK")
            .ok()
            .as_deref()
            == Some("1")
        {
            if let Some(result) =
                execute_rmsnorm_f32_host_fallback(kernel_name, x, weight, y, rows, hidden, eps)
            {
                return Some(result);
            }
        }
        return if allow_normal_fallback {
            None
        } else {
            Some(Err(CUerror::UNKNOWN))
        };
    }

    if (name_lower.contains("rope_norm") || name_lower.contains("rope_neox"))
        && (allow_named_host_fallback
            || std::env::var("HETGPU_SIFIVE_ROPE_HOST_FALLBACK")
                .ok()
                .map(|v| v != "0")
                .unwrap_or(false))
    {
        return execute_rope_host_fallback(kernel_name, kernel_params, grid_dim_x);
    }

    if name_lower.contains("compute_batched_ptrs") {
        return execute_compute_batched_ptrs_fallback(kernel_name, kernel_params);
    }

    if name_lower.contains("cpy_scalar")
        && (allow_named_host_fallback
            || sifive_env_enabled_default("HETGPU_SIFIVE_CPY_SCALAR_HOST_FALLBACK", true))
    {
        return execute_cpy_scalar_host_fallback(kernel_name, kernel_params);
    }

    if name_lower.contains("convert_unary") && allow_named_host_fallback {
        return execute_convert_unary_host_fallback(kernel_name, kernel_params);
    }

    if name_lower.contains("k_bin_bcast")
        && (allow_named_host_fallback
            || sifive_env_enabled_default("HETGPU_SIFIVE_BIN_BCAST_HOST_FALLBACK", true))
    {
        if let Some(result) = execute_bin_bcast_f32_fallback(kernel_name, kernel_params) {
            return Some(result);
        }
    }

    if name_lower.contains("scale_f32") && allow_named_host_fallback {
        return execute_scale_f32_fallback(kernel_name, kernel_params);
    }

    if name_lower.contains("k_get_rows_float")
        && (allow_named_host_fallback
            || sifive_env_enabled_default("HETGPU_SIFIVE_GET_ROWS_FLOAT_HOST_FALLBACK", true))
    {
        return execute_get_rows_float_fallback(kernel_name, kernel_params, grid_dim_x);
    }

    if name_lower.contains("k_set_rows")
        && !name_lower.contains("k_set_rows_quant")
        && allow_named_host_fallback
    {
        return execute_set_rows_host_fallback(kernel_name, kernel_params);
    }

    if name_lower.contains("l2_norm_f32")
        && (allow_named_host_fallback
            || sifive_env_enabled_default("HETGPU_SIFIVE_L2_NORM_HOST_FALLBACK", true))
    {
        return execute_l2_norm_f32_fallback(
            kernel_name,
            kernel_params,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
        );
    }

    if name_lower.contains("vectorized_gather_kernel") {
        return execute_vectorized_gather_host_copy(kernel_name, kernel_params);
    }

    if name_lower.contains("direct_copy_kernel_cuda")
        || (name_lower.contains("unrolled_elementwise_kernel")
            && name_lower.contains("loadwithcast")
            && name_lower.contains("storewithcast"))
    {
        return execute_direct_copy_bool_host_cast(kernel_name, grid_dim_x, 1, kernel_params);
    }

    None
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) fn launch_kernel_ex(
    config: &cuda_types::cuda::CUlaunchConfig,
    f: *mut crate::r#impl::module::SifiveKernel,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> cuda_types::cuda::CUresult {
    launch_kernel(
        f,
        config.gridDimX,
        config.gridDimY,
        config.gridDimZ,
        config.blockDimX,
        config.blockDimY,
        config.blockDimZ,
        config.sharedMemBytes,
        config.hStream.0 as *mut _,
        kernel_params,
        extra,
    )
}
