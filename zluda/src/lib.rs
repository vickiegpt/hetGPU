pub(crate) mod r#impl;

/// Production-backed batch planner entrypoint used by external evaluation tests.
///
/// This intentionally exposes only the ordered logical slices, keeping the scheduler's
/// invariant-bearing types private to the runtime.
#[doc(hidden)]
#[cfg(all(
    unix,
    feature = "nvidia",
    feature = "evaluation",
    not(feature = "amd"),
    not(feature = "intel")
))]
pub fn hetgpu_v3_batch_plan_for_evaluation(
    logical_batch: u32,
    live_max_batch: u32,
    configured_limit: Option<&str>,
) -> std::result::Result<Vec<(u32, u32)>, String> {
    let config = crate::r#impl::batch_scheduler::BatchSchedulerConfig::parse(
        configured_limit,
        live_max_batch,
    )?;
    let plan = config.plan(logical_batch)?;
    Ok(plan
        .slices()
        .iter()
        .map(|slice| (slice.first(), slice.count()))
        .collect())
}

#[cfg(all(
    unix,
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_tq1_register_tensor_v1(
    tensor: *const crate::r#impl::tq1_bridge::HetgpuTq1TensorV1,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::r#impl::tq1_bridge::register_tensor(tensor)
    }))
    .unwrap_or(crate::r#impl::tq1_bridge::HETGPU_TQ1_ERROR)
}

#[cfg(all(
    unix,
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_tq1_try_mul_mat_id_v1(
    operation: *const crate::r#impl::tq1_bridge::HetgpuTq1MulMatIdV1,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::r#impl::tq1_bridge::try_mul_mat_id(operation)
    }))
    .unwrap_or(crate::r#impl::tq1_bridge::HETGPU_TQ1_ERROR)
}

#[cfg(all(
    unix,
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_iq1s_register_tensor_v1(
    tensor: *const crate::r#impl::iq1s_weight_registry::HetgpuIq1sTensorV1,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::r#impl::iq1s_weight_registry::register_tensor(tensor)
    }))
    .unwrap_or(crate::r#impl::iq1s_weight_registry::HETGPU_IQ1S_ERROR)
}

#[cfg(all(
    unix,
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_iq1s_bind_device_v1(
    binding: *const crate::r#impl::iq1s_weight_registry::HetgpuIq1sDeviceBindingV1,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::r#impl::iq1s_weight_registry::bind_device(binding)
    }))
    .unwrap_or(crate::r#impl::iq1s_weight_registry::HETGPU_IQ1S_ERROR)
}

#[cfg(all(
    unix,
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_iq1s_layer_begin_v2(
    abi_version: u32,
    layer_id: u32,
    transaction_id: u64,
    batch_count: u32,
    cuda_stream: *mut ::core::ffi::c_void,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::r#impl::iq1s_layer::layer_begin(
            abi_version,
            layer_id,
            transaction_id,
            batch_count,
            cuda_stream,
        )
    }))
    .unwrap_or(crate::r#impl::iq1s_weight_registry::HETGPU_IQ1S_ERROR)
}

#[cfg(all(
    unix,
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_iq1s_layer_set_routes_v2(
    abi_version: u32,
    transaction_id: u64,
    token_ids: *const u32,
    expert_ids: *const i32,
    route_weights: *const f32,
    top_k: u32,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::r#impl::iq1s_layer::layer_set_routes(
            abi_version,
            transaction_id,
            token_ids,
            expert_ids,
            route_weights,
            top_k,
        )
    }))
    .unwrap_or(crate::r#impl::iq1s_weight_registry::HETGPU_IQ1S_ERROR)
}

#[cfg(all(
    unix,
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub extern "C" fn hetgpu_iq1s_layer_phase_commit_v2(
    abi_version: u32,
    transaction_id: u64,
    phase: u32,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::r#impl::iq1s_layer::layer_phase_commit(abi_version, transaction_id, phase)
    }))
    .unwrap_or(crate::r#impl::iq1s_weight_registry::HETGPU_IQ1S_ERROR)
}

#[cfg(all(
    unix,
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub extern "C" fn hetgpu_iq1s_layer_commit_v2(abi_version: u32, transaction_id: u64) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::r#impl::iq1s_layer::layer_commit(abi_version, transaction_id)
    }))
    .unwrap_or(crate::r#impl::iq1s_weight_registry::HETGPU_IQ1S_ERROR)
}

#[cfg(all(
    unix,
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub extern "C" fn hetgpu_iq1s_layer_abort_v2(
    abi_version: u32,
    transaction_id: u64,
    reason: u32,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::r#impl::iq1s_layer::layer_abort(abi_version, transaction_id, reason)
    }))
    .unwrap_or(crate::r#impl::iq1s_weight_registry::HETGPU_IQ1S_ERROR)
}

// The embedded CUDA runtime shim exposes IPC entry points in every build. The
// SIFIVE backend supplies real shared-DDR implementations; NVIDIA-only builds
// must still provide fail-closed providers so the shim has no unresolved DT_NEEDED symbols.
#[cfg(all(
    feature = "nvidia",
    not(feature = "sifive"),
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_sifive_ipc_get_mem_handle(
    _ptr: *const ::core::ffi::c_void,
    _handle: *mut ::core::ffi::c_void,
    _handle_len: usize,
) -> i32 {
    1
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "sifive"),
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_sifive_ipc_open_mem_handle(
    _dev_ptr: *mut *mut ::core::ffi::c_void,
    _handle: *const ::core::ffi::c_void,
    _flags: ::core::ffi::c_uint,
) -> i32 {
    1
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "sifive"),
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_sifive_ipc_close_mem_handle(_ptr: *mut ::core::ffi::c_void) -> i32 {
    1
}

#[cfg(all(
    unix,
    feature = "nvidia",
    feature = "evaluation",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
unsafe fn tq1_evaluate_raw(
    blocks: *const u8,
    blocks_len: usize,
    activations: *const f32,
    activations_len: usize,
    k: u64,
    rows: u64,
    tokens: u64,
    experts: u64,
    output: *mut f32,
    output_len: usize,
) -> std::result::Result<(), String> {
    use crate::r#impl::tq1_tmatmul::{
        ExpertRole, Q8KBlock, TensorRegistry, Tq1TensorRegistration, TQ1_BLOCK_BYTES, TQ1_VALUES,
    };
    use crate::r#impl::tq1_xrt::{execute_mul_mat_id, Tq1MulMatIdOperation};
    use std::io::Write;

    if blocks.is_null() || activations.is_null() || output.is_null() {
        return Err("TQ1_0 evaluation received a null buffer".to_string());
    }
    let k = usize::try_from(k).map_err(|_| "TQ1_0 evaluation K does not fit usize")?;
    let rows = usize::try_from(rows).map_err(|_| "TQ1_0 evaluation rows do not fit usize")?;
    let tokens = usize::try_from(tokens).map_err(|_| "TQ1_0 evaluation tokens do not fit usize")?;
    let experts =
        usize::try_from(experts).map_err(|_| "TQ1_0 evaluation experts do not fit usize")?;
    if k == 0 || !k.is_multiple_of(TQ1_VALUES) {
        return Err("TQ1_0 evaluation K must be a nonzero multiple of 256".to_string());
    }
    if rows == 0 || tokens == 0 || experts == 0 {
        return Err("TQ1_0 evaluation dimensions must be positive".to_string());
    }
    let blocks_per_row = k / TQ1_VALUES;
    let row_bytes = blocks_per_row
        .checked_mul(TQ1_BLOCK_BYTES)
        .ok_or_else(|| "TQ1_0 evaluation row byte count overflow".to_string())?;
    let expected_blocks = experts
        .checked_mul(rows)
        .and_then(|value| value.checked_mul(row_bytes))
        .ok_or_else(|| "TQ1_0 evaluation block byte count overflow".to_string())?;
    let logical_groups = tokens
        .checked_mul(experts)
        .ok_or_else(|| "TQ1_0 evaluation logical group count overflow".to_string())?;
    let expected_activations = logical_groups
        .checked_mul(k)
        .ok_or_else(|| "TQ1_0 evaluation activation count overflow".to_string())?;
    let expected_output = logical_groups
        .checked_mul(rows)
        .ok_or_else(|| "TQ1_0 evaluation output count overflow".to_string())?;
    if blocks_len != expected_blocks {
        return Err(format!(
            "TQ1_0 evaluation has {blocks_len} block bytes, expected {expected_blocks}"
        ));
    }
    if activations_len != expected_activations {
        return Err(format!(
            "TQ1_0 evaluation has {activations_len} activations, expected {expected_activations}"
        ));
    }
    if output_len != expected_output {
        return Err(format!(
            "TQ1_0 evaluation has {output_len} output elements, expected {expected_output}"
        ));
    }

    let blocks = std::slice::from_raw_parts(blocks, blocks_len);
    let activations = std::slice::from_raw_parts(activations, activations_len).to_vec();
    if activations.iter().any(|value| !value.is_finite()) {
        return Err("TQ1_0 evaluation activations must be finite".to_string());
    }
    let q8_dump_path = std::env::var_os("HETGPU_TQ1_Q8_DUMP")
        .ok_or_else(|| "HETGPU_TQ1_Q8_DUMP is required for evaluation".to_string())?;
    let mut q8_blocks = Vec::with_capacity(activations.len() / TQ1_VALUES);
    for block in activations.chunks_exact(TQ1_VALUES) {
        q8_blocks.push(Q8KBlock::quantize(block)?);
    }
    let mut q8_dump = std::fs::File::create(&q8_dump_path)
        .map_err(|error| format!("create TQ1_0 Q8_K dump: {error}"))?;
    q8_dump
        .write_all(b"TQ1Q8K1\0")
        .and_then(|_| q8_dump.write_all(&(q8_blocks.len() as u64).to_le_bytes()))
        .map_err(|error| format!("write TQ1_0 Q8_K dump header: {error}"))?;
    for block in &q8_blocks {
        q8_dump
            .write_all(&block.scale.to_le_bytes())
            .and_then(|_| {
                q8_dump.write_all(unsafe {
                    std::slice::from_raw_parts(block.qs.as_ptr().cast::<u8>(), block.qs.len())
                })
            })
            .map_err(|error| format!("write TQ1_0 Q8_K dump: {error}"))?;
    }
    q8_dump
        .sync_all()
        .map_err(|error| format!("sync TQ1_0 Q8_K dump: {error}"))?;
    let mut fixture = tempfile::NamedTempFile::new()
        .map_err(|error| format!("create TQ1_0 evaluation fixture: {error}"))?;
    fixture
        .write_all(blocks)
        .map_err(|error| format!("write TQ1_0 evaluation fixture: {error}"))?;
    fixture
        .flush()
        .map_err(|error| format!("flush TQ1_0 evaluation fixture: {error}"))?;
    let nbytes = u64::try_from(expected_blocks)
        .map_err(|_| "TQ1_0 evaluation byte count does not fit u64".to_string())?;
    let source = TensorRegistry::default().register(Tq1TensorRegistration {
        path: fixture.path().to_path_buf(),
        file_offset: 0,
        nbytes,
        name: "blk.0.ffn_down_exps.weight".to_string(),
        ne: [k as u64, rows as u64, experts as u64, 1],
        nb: [
            TQ1_BLOCK_BYTES as u64,
            row_bytes as u64,
            (rows * row_bytes) as u64,
            nbytes,
        ],
        role: ExpertRole::Down,
    })?;
    let expert_ids = (0..tokens).flat_map(|_| 0..experts).collect::<Vec<_>>();
    let result = execute_mul_mat_id(&Tq1MulMatIdOperation {
        tensor: source,
        activations,
        expert_ids,
        token_count: tokens,
        expert_slots: experts,
    })?;
    if result.outputs.len() != expected_output {
        return Err("TQ1_0 evaluation adapter returned the wrong output size".to_string());
    }

    let evidence_path = std::env::var_os("HETGPU_TQ1_EVIDENCE_LOG")
        .ok_or_else(|| "HETGPU_TQ1_EVIDENCE_LOG is required for evaluation".to_string())?;
    let mut evidence = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&evidence_path)
        .map_err(|error| format!("open TQ1_0 evaluation evidence: {error}"))?;
    serde_json::to_writer(
        &mut evidence,
        &serde_json::json!({
            "route": "handled",
            "fixture": true,
            "dimensions": { "k": k, "rows": rows, "tokens": tokens, "experts": experts },
            "evidence": result.evidence,
        }),
    )
    .map_err(|error| format!("serialize TQ1_0 evaluation evidence: {error}"))?;
    evidence
        .write_all(b"\n")
        .map_err(|error| format!("write TQ1_0 evaluation evidence: {error}"))?;
    evidence
        .sync_all()
        .map_err(|error| format!("sync TQ1_0 evaluation evidence: {error}"))?;
    std::ptr::copy_nonoverlapping(result.outputs.as_ptr(), output, expected_output);
    Ok(())
}

/// Evaluation-only raw TQ1_0 entry point backed by the production XRT adapter.
#[cfg(all(
    unix,
    feature = "nvidia",
    feature = "evaluation",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_tq1_evaluate_raw_v1(
    blocks: *const u8,
    blocks_len: usize,
    activations: *const f32,
    activations_len: usize,
    k: u64,
    rows: u64,
    tokens: u64,
    experts: u64,
    output: *mut f32,
    output_len: usize,
) -> i32 {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tq1_evaluate_raw(
            blocks,
            blocks_len,
            activations,
            activations_len,
            k,
            rows,
            tokens,
            experts,
            output,
            output_len,
        )
    })) {
        Ok(Ok(())) => crate::r#impl::tq1_bridge::HETGPU_TQ1_HANDLED,
        Ok(Err(error)) => {
            eprintln!("[hetgpu-tq1-evaluation] {error}");
            crate::r#impl::tq1_bridge::HETGPU_TQ1_ERROR
        }
        Err(_) => crate::r#impl::tq1_bridge::HETGPU_TQ1_ERROR,
    }
}

#[cfg(all(test, feature = "evaluation", feature = "nvidia"))]
mod tq1_evaluation_tests {
    use super::*;

    const K: u64 = 1024;
    const ROWS: u64 = 1;
    const TOKENS: u64 = 1;
    const EXPERTS: u64 = 1;
    const BLOCK_BYTES: usize = 4 * 54;

    #[test]
    fn tq1_evaluation_rejects_null_pointers() {
        assert_eq!(
            unsafe {
                hetgpu_tq1_evaluate_raw_v1(
                    std::ptr::null(),
                    BLOCK_BYTES,
                    std::ptr::null(),
                    K as usize,
                    K,
                    ROWS,
                    TOKENS,
                    EXPERTS,
                    std::ptr::null_mut(),
                    1,
                )
            },
            crate::r#impl::tq1_bridge::HETGPU_TQ1_ERROR
        );
    }

    #[test]
    fn tq1_evaluation_rejects_nonmultiple_of_256() {
        let blocks = vec![0u8; BLOCK_BYTES];
        let activations = vec![0.0f32; K as usize];
        let mut output = vec![0.0f32; 1];
        assert_eq!(
            unsafe {
                hetgpu_tq1_evaluate_raw_v1(
                    blocks.as_ptr(),
                    blocks.len(),
                    activations.as_ptr(),
                    activations.len(),
                    1000,
                    ROWS,
                    TOKENS,
                    EXPERTS,
                    output.as_mut_ptr(),
                    output.len(),
                )
            },
            crate::r#impl::tq1_bridge::HETGPU_TQ1_ERROR
        );
    }

    #[test]
    fn tq1_evaluation_rejects_undersized_buffers() {
        let blocks = vec![0u8; BLOCK_BYTES];
        let activations = vec![0.0f32; K as usize];
        let mut output = vec![0.0f32; 1];
        let output_ptr = output.as_mut_ptr();
        let output_len = output.len();
        let invoke = |block_len, activation_len, output_len| unsafe {
            hetgpu_tq1_evaluate_raw_v1(
                blocks.as_ptr(),
                block_len,
                activations.as_ptr(),
                activation_len,
                K,
                ROWS,
                TOKENS,
                EXPERTS,
                output_ptr,
                output_len,
            )
        };
        assert_eq!(invoke(BLOCK_BYTES - 1, activations.len(), output_len), -1);
        assert_eq!(invoke(BLOCK_BYTES, activations.len() - 1, output_len), -1);
        assert_eq!(invoke(BLOCK_BYTES, activations.len(), output_len - 1), -1);
    }
}
// Import necessary for FromCuda
use crate::r#impl::FromCuda;
// Import std::ptr for null_mut
use std::ptr;
// Import Ze types
#[cfg(feature = "intel")]
use ze_runtime_sys::ze_device_handle_t;
// Import CUerror for Result
use cuda_types::cuda::CUerror;
// Define Result type to match FromCuda error return type
type Result<T> = std::result::Result<T, CUerror>;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};

#[cfg(unix)]
use libc::{dlsym, RTLD_DEFAULT};

// Note: The cudart shim symbols are force-linked via --whole-archive in build.rs,
// so we don't need an explicit anchor function here.

// FFI wrapper for LZ4 decompression called from cudart_shim.c
#[no_mangle]
pub extern "C" fn hetgpu_lz4_decompress(
    src: *const c_char,
    dst: *mut c_char,
    compressed_size: c_int,
    dst_capacity: c_int,
) -> c_int {
    unsafe { lz4_sys::LZ4_decompress_safe(src, dst, compressed_size, dst_capacity) }
}

// FFI wrapper for Zstandard PTX payloads in CUDA fatbins. Newer CUDA toolchains
// emit zstd-compressed entries while the legacy shim path only tried LZ4.
#[no_mangle]
pub extern "C" fn hetgpu_zstd_decompress(
    src: *const c_char,
    dst: *mut c_char,
    compressed_size: c_int,
    dst_capacity: c_int,
) -> c_int {
    if src.is_null() || dst.is_null() || compressed_size <= 0 || dst_capacity <= 0 {
        return -1;
    }

    use std::io::Read;

    let input = unsafe { std::slice::from_raw_parts(src as *const u8, compressed_size as usize) };
    let mut cursor = std::io::Cursor::new(input);
    let decoder = match ruzstd::StreamingDecoder::new(&mut cursor) {
        Ok(decoder) => decoder,
        Err(_) => return -2,
    };
    let mut limited = decoder.take(dst_capacity as u64 + 1);
    let mut decoded = Vec::new();
    if limited.read_to_end(&mut decoded).is_err() {
        return -3;
    }
    if decoded.len() > dst_capacity as usize {
        return -4;
    }

    unsafe {
        ptr::copy_nonoverlapping(decoded.as_ptr(), dst as *mut u8, decoded.len());
    }
    decoded.len() as c_int
}

#[cfg(feature = "intel")]
fn hetgpu_log_cuda_calls_enabled() -> bool {
    std::env::var("HETGPU_LOG_CUDA_CALLS")
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
    any(feature = "sifive", feature = "nvidia"),
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_launch_named_kernel(
    kernel_name: *const c_char,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    block_dim_y: ::core::ffi::c_uint,
    block_dim_z: ::core::ffi::c_uint,
    shared_mem_bytes: ::core::ffi::c_uint,
    stream: *mut c_void,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> i32 {
    crate::r#impl::function::launch_named_kernel_c(
        kernel_name,
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

#[cfg(all(
    any(feature = "sifive", feature = "nvidia"),
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_sifive_launch_named_kernel(
    kernel_name: *const c_char,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    block_dim_y: ::core::ffi::c_uint,
    block_dim_z: ::core::ffi::c_uint,
    shared_mem_bytes: ::core::ffi::c_uint,
    stream: *mut c_void,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> i32 {
    hetgpu_launch_named_kernel(
        kernel_name,
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

// Note: cudaDriverGetVersion and cudaGetDeviceCount are now in cudart_shim.c
// to get proper version tags (@@libcudart.so.12). The C implementations
// call through to cuDriverGetVersion/cuDeviceGetCount defined below.

// Note: cuGetProcAddress/cuGetProcAddress_v2/cuGetExportTable are provided
// via cuda_function_declarations! and implemented in r#impl::driver.

// Get device handle by index using the ordinal-to-handle mapping from driver.rs
#[cfg(feature = "intel")]
fn get_device_handle_by_index(index: usize) -> Result<ze_device_handle_t> {
    crate::r#impl::driver::get_ze_handle_by_ordinal(index as i32)
}

// Fix implementation of FromCuda for ze_device_handle_t
#[cfg(feature = "intel")]
impl FromCuda<'_, *mut i32> for *mut ze_device_handle_t {
    fn from_cuda(_: &*mut i32) -> Result<Self> {
        // Simplified implementation - just a placeholder
        Ok(ptr::null_mut())
    }
}

macro_rules! unimplemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                // Always log unimplemented calls to help debug initialization issues
                eprintln!("[hetGPU] Unimplemented CUDA call: {}", stringify!($fn_name));
                crate::r#impl::unimplemented()
            }
        )*
    };
}

#[cfg(feature = "amd")]
macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id).unwrap()),*).unwrap();
                Ok(())
            }
        )*
    };
}
#[cfg(feature = "intel")]
macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                if crate::hetgpu_log_cuda_calls_enabled() {
                    eprintln!("[hetGPU] {} called", stringify!($fn_name));
                }

                // Convert arguments with error handling
                let result = (|| -> std::result::Result<_, CUerror> {
                    let backend_ret = cuda_base::cuda_normalize_fn!( crate::r#impl::$fn_name )($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*);
                    Ok(crate::r#impl::into_cu_result(backend_ret))
                })();

                match result {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("[hetGPU] {} failed with conversion error: {:?}", stringify!($fn_name), e);
                        Err(e)
                    }
                }
            }
        )*
    };
}
#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, i32> for ze_device_handle_t {
    fn from_cuda(cuda_value: &'a i32) -> Result<Self> {
        // Logic to convert i32 to ze_device_handle_t
        if *cuda_value < 0 {
            return Err(CUerror::INVALID_VALUE); // Return an error, not CUresult
        }

        // Get device handle by index
        let device_handle = get_device_handle_by_index(*cuda_value as usize)?;
        Ok(device_handle)
    }
}

#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, cuda_types::cuda::CUdeviceptr_v2> for cuda_types::cuda::CUdeviceptr_v2 {
    fn from_cuda(cuda_value: &'a cuda_types::cuda::CUdeviceptr_v2) -> Result<Self> {
        // Logic to validate CUdeviceptr_v2
        if unsafe { cuda_value.0 as i64 } < 0 {
            return Err(CUerror::INVALID_HANDLE); // Return an error, not CUresult
        }

        Ok(*cuda_value)
    }
}
#[cfg(feature = "amd")]
macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::function::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id).unwrap()),*).unwrap();
                Ok(())
            }
        )*
    };
}

#[cfg(feature = "intel")]
macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                let backend_ret = cuda_base::cuda_normalize_fn!( crate::r#impl::function::$fn_name )($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*);
                crate::r#impl::into_cu_result(backend_ret)
            }
        )*
    };
}

#[cfg(feature = "tenstorrent")]
macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*)
            }
        )*
    };
}

#[cfg(feature = "tenstorrent")]
macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::function::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*)
            }
        )*
    };
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "nvidia")
))]
macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*);
                Ok(())
            }
        )*
    };
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "nvidia")
))]
macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::function::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*);
                Ok(())
            }
        )*
    };
}

// SIFIVE backend macros
#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                let backend_ret = cuda_base::cuda_normalize_fn!( crate::r#impl::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*);
                crate::r#impl::into_cu_result(backend_ret)
            }
        )*
    };
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                let backend_ret = cuda_base::cuda_normalize_fn!( crate::r#impl::function::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*);
                crate::r#impl::into_cu_result(backend_ret)
            }
        )*
    };
}

// NVIDIA backend macros - pass through to real libcuda.so
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*)
            }
        )*
    };
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::function::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*)
            }
        )*
    };
}

cuda_base::cuda_function_declarations!(
    unimplemented,
    implemented
        <= [
            cuCtxCreate_v2,
            cuCtxDestroy_v2,
            cuCtxGetLimit,
            cuCtxSetCurrent,
            cuCtxSetLimit,
            cuCtxSynchronize,
            cuCtxGetCurrent,
            cuCtxGetDevice,
            cuCtxPushCurrent_v2,
            cuCtxPopCurrent_v2,
            cuDeviceComputeCapability,
            cuDeviceGet,
            cuDeviceGetAttribute,
            cuDeviceGetCount,
            cuDeviceGetLuid,
            cuDeviceGetName,
            cuDevicePrimaryCtxRelease,
            cuDevicePrimaryCtxRetain,
            cuDevicePrimaryCtxGetState,
            cuDeviceGetProperties,
            cuDeviceGetUuid,
            cuDeviceGetUuid_v2,
            cuDeviceTotalMem_v2,
            cuDriverGetVersion,
            cuFuncGetAttribute,
            cuInit,
            cuMemAlloc_v2,
            cuMemFree_v2,
            cuMemcpyDtoH_v2,
            cuMemcpyHtoD_v2,
            cuModuleGetFunction,
            cuModuleGetLoadingMode,
            cuModuleLoadData,
            cuModuleUnload,
            cuLibraryLoadData,
            cuLibraryGetKernel,
            cuKernelGetFunction,
            cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
            cuMemAddressFree,
            cuMemAddressReserve,
            cuMemCreate,
            cuMemGetAllocationGranularity,
            cuMemMap,
            cuMemRelease,
            cuMemSetAccess,
            cuMemUnmap,
            cuPointerGetAttribute,
            cuMemGetAddressRange_v2,
            cuMemsetD32_v2,
            cuMemsetD8_v2,
            // Provide custom implementations in r#impl::driver
            cuGetProcAddress,
            cuGetProcAddress_v2,
            cuGetExportTable,
            cuGetErrorString,
            cuGetErrorName
        ],
    implemented_in_function <= [cuLaunchKernel, cuLaunchKernelEx,]
);
// cuGetErrorString/cuGetErrorName are implemented via r#impl::driver
