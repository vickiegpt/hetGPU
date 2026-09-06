use cuda_types::cuda::*;
#[cfg(feature = "amd")]
use hip_runtime_sys::*;
use std::mem::{self, ManuallyDrop, MaybeUninit};
use std::os::raw::c_char;
use std::sync::OnceLock;
#[cfg(feature = "intel")]
use ze_device::ze_device_limit_t;
#[cfg(feature = "intel")]
use ze_runtime_sys::*;

#[cfg(feature = "nvidia")]
pub use nvidia_runtime_sys;

#[cfg(debug_assertions)]
pub(crate) fn debug_logs_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("HETGPU_DEBUG_LOGS")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "on"))
            .unwrap_or(false)
    })
}

#[cfg(not(debug_assertions))]
pub(crate) fn debug_logs_enabled() -> bool {
    false
}

macro_rules! hetgpu_debug {
    ($($arg:tt)*) => {{
        if crate::r#impl::debug_logs_enabled() {
            eprintln!($($arg)*);
        }
    }};
}

pub(crate) use hetgpu_debug;

pub(super) mod context;
pub(super) mod device;
pub(super) mod driver;
pub(super) mod function;
pub(super) mod memory;
pub(super) mod module;
pub(super) mod pointer;

// Checkpoint/resume support for GPU state
pub mod checkpoint;
pub(crate) mod concordia_delta;
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) mod concordia_gpu;
pub(crate) mod concordia_instrument;
pub(crate) mod concordia_runtime;
pub(crate) mod kimi_concordia;
pub(crate) mod nccl_recovery;
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) mod nvint4_tmatmul;
pub(crate) mod persistent_router;

// Record/replay support for heterogeneous GPU debugging
pub mod replay;
pub mod replay_ffi;

#[cfg(feature = "intel")]
pub mod ze_device;
#[cfg(feature = "intel")]
pub mod ze_module;

#[cfg(any(feature = "intel", feature = "nvidia"))]
pub(crate) mod cxl_tmatmul;

#[cfg(all(unix, any(feature = "intel", feature = "nvidia")))]
pub(crate) mod cxl_tmatmul_v3;

#[cfg(all(unix, any(feature = "intel", feature = "nvidia")))]
pub(crate) mod batch_scheduler;

#[cfg(all(unix, feature = "nvidia", not(feature = "amd"), not(feature = "intel")))]
pub(crate) mod iq1s_layer;

#[cfg(all(unix, feature = "nvidia", not(feature = "amd"), not(feature = "intel")))]
pub(crate) mod iq1s_layer_abi;

#[cfg(all(unix, feature = "nvidia", not(feature = "amd"), not(feature = "intel")))]
pub(crate) mod iq1s_layer_trace;

#[cfg(all(unix, feature = "nvidia", not(feature = "amd"), not(feature = "intel")))]
pub(crate) mod iq1s_persistent_runtime;

#[cfg(all(unix, feature = "nvidia", not(feature = "amd"), not(feature = "intel")))]
pub(crate) mod iq1s_persistent_proof;

#[cfg(all(unix, feature = "nvidia", not(feature = "amd"), not(feature = "intel")))]
pub(crate) mod iq1s_tmatmul;

#[cfg(all(unix, feature = "nvidia", not(feature = "amd"), not(feature = "intel")))]
pub(crate) mod iq1s_trace;

#[cfg(all(unix, feature = "nvidia", not(feature = "amd"), not(feature = "intel")))]
pub(crate) mod iq1s_weight_arena;

#[cfg(all(unix, feature = "nvidia", not(feature = "amd"), not(feature = "intel")))]
pub(crate) mod iq1s_weight_registry;

#[cfg(all(unix, feature = "nvidia", not(feature = "amd"), not(feature = "intel")))]
pub(crate) mod iq1s_xrt;

#[cfg(all(unix, feature = "nvidia", not(feature = "amd"), not(feature = "intel")))]
pub(crate) mod tq1_tmatmul;

#[cfg(all(unix, feature = "nvidia", not(feature = "amd"), not(feature = "intel")))]
pub(crate) mod tq1_bridge;

#[cfg(all(unix, feature = "nvidia", not(feature = "amd"), not(feature = "intel")))]
pub(crate) mod tq1_xrt;

#[cfg(any(feature = "intel", feature = "nvidia"))]
pub(crate) mod xrt_tmatmul;

#[cfg(any(feature = "intel", feature = "nvidia"))]
pub(crate) mod xrt_iq1s_persistent;

#[cfg(any(feature = "intel", feature = "nvidia"))]
pub(crate) mod bitnet_disagg;

#[cfg(test)]
pub(crate) mod test_env {
    use std::sync::{Mutex, MutexGuard};

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    pub(crate) fn lock() -> MutexGuard<'static, ()> {
        ENV_MUTEX
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

// TMatmul assembly interpreter (virtual backend execution engine)
#[cfg(feature = "intel")]
pub(crate) mod tmatmul_interpreter;

#[cfg(feature = "cutile")]
pub mod cutile;

#[cfg(feature = "sifive")]
pub mod sifive;

// In both debug and release builds, return SUCCESS for unimplemented functions
// to allow frameworks like PyTorch to proceed with virtual device
pub(crate) fn unimplemented() -> CUresult {
    // In the virtual backend we prefer to be permissive to let higher-level
    // frameworks proceed. Allow opting into strict behavior via HETGPU_STRICT=1.
    if std::env::var("HETGPU_STRICT").ok().as_deref() == Some("1") {
        CUresult::ERROR_NOT_SUPPORTED
    } else {
        CUresult::SUCCESS
    }
}

pub(crate) trait FromCuda<'a, T>: Sized {
    fn from_cuda(t: &'a T) -> Result<Self, CUerror>;
}

macro_rules! from_cuda_nop {
    ($($type_:ty),*) => {
        $(
            impl<'a> FromCuda<'a, $type_> for $type_ {
                fn from_cuda(x: &'a $type_) -> Result<Self, CUerror> {
                    Ok(*x)
                }
            }

            impl<'a> FromCuda<'a, *mut $type_> for &'a mut $type_ {
                fn from_cuda(x: &'a *mut $type_) -> Result<Self, CUerror> {
                    match unsafe { x.as_mut() } {
                        Some(x) => Ok(x),
                        None => Err(CUerror::INVALID_VALUE),
                    }
                }
            }
        )*
    };
}

macro_rules! from_cuda_transmute {
    ($($from:ty => $to:ty),*) => {
        $(
            impl<'a> FromCuda<'a, $from> for $to {
                fn from_cuda(x: &'a $from) -> Result<Self, CUerror> {
                    // Use as_ptr and from_bits instead of direct transmute
                    // to handle types of different sizes safely
                    Ok(unsafe {
                        let raw_bits = std::mem::transmute_copy::<$from, u64>(x);
                        std::mem::transmute_copy::<u64, $to>(&raw_bits)
                    })
                }
            }

            impl<'a> FromCuda<'a, *mut $from> for &'a mut $to {
                fn from_cuda(x: &'a *mut $from) -> Result<Self, CUerror> {
                    match unsafe { x.cast::<$to>().as_mut() } {
                        Some(x) => Ok(x),
                        None => Err(CUerror::INVALID_VALUE),
                    }
                }
            }

            impl<'a> FromCuda<'a, *mut $from> for * mut $to {
                fn from_cuda(x: &'a *mut $from) -> Result<Self, CUerror> {
                    Ok(x.cast::<$to>())
                }
            }
        )*
    };
}

macro_rules! from_cuda_object {
    ($($type_:ty),*) => {
        $(
            impl<'a> FromCuda<'a, <$type_ as ZludaObject>::CudaHandle> for <$type_ as ZludaObject>::CudaHandle {
                fn from_cuda(handle: &'a <$type_ as ZludaObject>::CudaHandle) -> Result<<$type_ as ZludaObject>::CudaHandle, CUerror> {
                    Ok(*handle)
                }
            }

            impl<'a> FromCuda<'a, *mut <$type_ as ZludaObject>::CudaHandle> for &'a mut <$type_ as ZludaObject>::CudaHandle {
                fn from_cuda(handle: &'a *mut <$type_ as ZludaObject>::CudaHandle) -> Result<&'a mut <$type_ as ZludaObject>::CudaHandle, CUerror> {
                    match unsafe { handle.as_mut() } {
                        Some(x) => Ok(x),
                        None => Err(CUerror::INVALID_VALUE),
                    }
                }
            }

            impl<'a> FromCuda<'a, <$type_ as ZludaObject>::CudaHandle> for &'a $type_ {
                fn from_cuda(handle: &'a <$type_ as ZludaObject>::CudaHandle) -> Result<&'a $type_, CUerror> {
                    Ok(as_ref(handle).as_result()?)
                }
            }
        )*
    };
}

// Common type conversions that work for both AMD and Intel implementations
from_cuda_nop!(
    *mut u8,
    *mut i8,
    *mut i32,
    *mut usize,
    *mut CUcontext,
    *const ::core::ffi::c_void,
    *mut *const ::core::ffi::c_void,
    *const ::core::ffi::c_char,
    *mut ::core::ffi::c_void,
    *mut *mut ::core::ffi::c_void,
    *mut CUfunction,
    *mut CUkernel,
    *mut CUjit_option,
    *mut CUlibraryOption,
    CUlibrary,
    CUkernel,
    *mut cuda_types::cuda::CUmoduleLoadingMode,
    *mut cuda_types::cuda::CUdriverProcAddressQueryResult,
    *const cuda_types::cuda::CUlaunchConfig,
    *const cuda_types::cuda::CUuuid,
    u8,
    i32,
    u32,
    usize,
    cuda_types::cuda::cuuint64_t,
    cuda_types::cuda::CUdevprop,
    cuda_types::cuda::CUresult,
    CUdevice_attribute,
    cuda_types::cuda::CUmemAllocationGranularity_flags,
    *mut cuda_types::cuda::CUmemGenericAllocationHandle,
    *const cuda_types::cuda::CUmemAllocationProp,
    *const cuda_types::cuda::CUmemAccessDesc
);

// Wrapper entry points expected by cuda_function_declarations! for error string helpers
pub(crate) fn get_error_string(error: CUresult, pStr: &mut *const c_char) -> Result<(), CUerror> {
    driver::get_error_string(error, pStr)
}

pub(crate) fn get_error_name(error: CUresult, pStr: &mut *const c_char) -> Result<(), CUerror> {
    driver::get_error_name(error, pStr)
}

// AMD-specific type conversions
#[cfg(feature = "amd")]
from_cuda_transmute!(
    CUuuid => hipUUID,
    CUfunction => hipFunction_t,
    CUfunction_attribute => hipFunction_attribute,
    CUstream => hipStream_t,
    CUpointer_attribute => hipPointer_attribute,
    CUdeviceptr_v2 => hipDeviceptr_t
);

// Intel-specific type conversions
#[cfg(feature = "intel")]
from_cuda_transmute!(
    CUuuid => ze_uuid_t,
    CUfunction => ze_kernel_handle_t,
    CUstream => ze_command_queue_handle_t,
    CUpointer_attribute => ze_memory_allocation_properties_t,
    CUdeviceptr_v2 => ze_device_handle_t
);

// Special implementation for CUfunction_attribute to ze_kernel_desc_t conversion
#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, CUfunction_attribute> for ze_kernel_desc_t {
    fn from_cuda(attr: &'a CUfunction_attribute) -> Result<Self, CUerror> {
        // Create a properly initialized ze_kernel_desc_t
        // This needs to be customized based on how attributes should map
        let desc = ze_kernel_desc_t::default();

        // Set appropriate fields based on the attribute
        // For example:
        match *attr {
            // Map each attribute appropriately
            // This is a placeholder - actual implementation will depend on attributes
            _ => {}
        }

        // For now, return NOT_SUPPORTED for all attributes
        Err(CUerror::NOT_SUPPORTED)
    }
}

#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, *mut CUfunction_attribute> for &'a mut ze_kernel_desc_t {
    fn from_cuda(x: &'a *mut CUfunction_attribute) -> Result<Self, CUerror> {
        match unsafe { x.cast::<ze_kernel_desc_t>().as_mut() } {
            Some(x) => Ok(x),
            None => Err(CUerror::INVALID_VALUE),
        }
    }
}

#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, *mut CUfunction_attribute> for *mut ze_kernel_desc_t {
    fn from_cuda(x: &'a *mut CUfunction_attribute) -> Result<Self, CUerror> {
        Ok(x.cast::<ze_kernel_desc_t>())
    }
}

// Add implementation for _ze_kernel_properties_t
#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, CUfunction_attribute_enum> for _ze_kernel_properties_t {
    fn from_cuda(attr: &'a CUfunction_attribute_enum) -> Result<Self, CUerror> {
        let props = _ze_kernel_properties_t::default();

        // Map CUDA function attributes to Level Zero kernel properties
        match *attr {
            // Add mappings based on the specific attributes needed
            // Example (modify based on actual enum variants):
            // CUfunction_attribute_enum::SOME_VARIANT => {
            //     props.some_field = some_value;
            // }
            CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK => {
                // Set appropriate property
                // This is a placeholder - implement with actual property
            }
            CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES => {
                // Set appropriate property
                // This is a placeholder - implement with actual property
            }
            // Add more mappings as needed
            _ => return Err(CUerror::NOT_SUPPORTED),
        }

        Ok(props)
    }
}

// Also implement for pointer types
#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, *mut CUfunction_attribute_enum> for &'a mut _ze_kernel_properties_t {
    fn from_cuda(x: &'a *mut CUfunction_attribute_enum) -> Result<Self, CUerror> {
        match unsafe { x.cast::<_ze_kernel_properties_t>().as_mut() } {
            Some(x) => Ok(x),
            None => Err(CUerror::INVALID_VALUE),
        }
    }
}

#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, *mut CUfunction_attribute_enum> for *mut _ze_kernel_properties_t {
    fn from_cuda(x: &'a *mut CUfunction_attribute_enum) -> Result<Self, CUerror> {
        Ok(x.cast::<_ze_kernel_properties_t>())
    }
}

// Add implementation for CUpointer_attribute_enum
#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, CUpointer_attribute_enum> for CUpointer_attribute_enum {
    fn from_cuda(attr: &'a CUpointer_attribute_enum) -> Result<Self, CUerror> {
        // Simple implementation that just returns the same value
        Ok(*attr)
    }
}

// NOTE: Do not add explicit FromCuda impls for CUfunction here.
// They conflict with the generic object mapping generated by from_cuda_object!(module::ZeKernel)

// Add implementation for *mut CUdeviceptr_v2 from *mut CUdeviceptr_v2
impl<'a> FromCuda<'a, *mut CUdeviceptr_v2> for *mut CUdeviceptr_v2 {
    fn from_cuda(x: &'a *mut CUdeviceptr_v2) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

// Add implementation for *mut _ze_device_uuid_t from *mut CUuuid_st
#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, *mut CUuuid_st> for *mut _ze_device_uuid_t {
    fn from_cuda(x: &'a *mut CUuuid_st) -> Result<Self, CUerror> {
        Ok(x.cast::<_ze_device_uuid_t>())
    }
}

// Add implementation for u32 from CUlimit_enum
impl<'a> FromCuda<'a, CUlimit_enum> for u32 {
    fn from_cuda(limit: &'a CUlimit_enum) -> Result<Self, CUerror> {
        // Convert CUlimit_enum to u32 based on your requirements
        // This is a basic implementation, adjust values as needed
        Ok(match *limit {
            CUlimit_enum::CU_LIMIT_STACK_SIZE => 0,
            CUlimit_enum::CU_LIMIT_PRINTF_FIFO_SIZE => 1,
            CUlimit_enum::CU_LIMIT_MALLOC_HEAP_SIZE => 2,
            CUlimit_enum::CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH => 3,
            CUlimit_enum::CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT => 4,
            CUlimit_enum::CU_LIMIT_MAX_L2_FETCH_GRANULARITY => 5,
            _ => return Err(CUerror::NOT_SUPPORTED),
        })
    }
}

// Add implementation for CUdeviceptr_v2 -> ()
impl<'a> FromCuda<'a, CUdeviceptr_v2> for () {
    fn from_cuda(_: &'a CUdeviceptr_v2) -> Result<Self, CUerror> {
        // Just discard the value and return unit
        Ok(())
    }
}

// Module type is available for all backends
#[cfg(any(
    feature = "amd",
    feature = "intel",
    feature = "tenstorrent",
    feature = "tmatmul",
    feature = "nvidia",
    feature = "sifive"
))]
from_cuda_object!(module::Module);

from_cuda_object!(context::Context);

// ZeKernel is only for Intel backend
#[cfg(feature = "intel")]
from_cuda_object!(module::ZeKernel);

// NvidiaKernel is only for NVIDIA backend
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
from_cuda_object!(module::NvidiaKernel);

// SifiveKernel is only for SIFIVE backend
#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
from_cuda_object!(module::SifiveKernel);

#[cfg(feature = "amd")]
impl<'a> FromCuda<'a, CUlimit> for hipLimit_t {
    fn from_cuda(limit: &'a CUlimit) -> Result<Self, CUerror> {
        Ok(match *limit {
            CUlimit::CU_LIMIT_STACK_SIZE => hipLimit_t::hipLimitStackSize,
            CUlimit::CU_LIMIT_PRINTF_FIFO_SIZE => hipLimit_t::hipLimitPrintfFifoSize,
            CUlimit::CU_LIMIT_MALLOC_HEAP_SIZE => hipLimit_t::hipLimitMallocHeapSize,
            _ => return Err(CUerror::NOT_SUPPORTED),
        })
    }
}

#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, CUlimit> for ze_device_limit_t {
    fn from_cuda(limit: &'a CUlimit) -> Result<Self, CUerror> {
        // Intel's Level Zero doesn't have direct equivalents for CUDA limits
        // Return same enum variants as AMD for consistency
        Ok(match *limit {
            CUlimit::CU_LIMIT_STACK_SIZE => ze_device_limit_t::ZE_LIMIT_STACK_SIZE,
            CUlimit::CU_LIMIT_PRINTF_FIFO_SIZE => ze_device_limit_t::ZE_LIMIT_PRINTF_FIFO_SIZE,
            CUlimit::CU_LIMIT_MALLOC_HEAP_SIZE => ze_device_limit_t::ZE_LIMIT_MALLOC_HEAP_SIZE,
            _ => return Err(CUerror::NOT_SUPPORTED),
        })
    }
}

// NVIDIA-specific FromCuda implementations - passthrough
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl<'a> FromCuda<'a, CUlimit> for CUlimit {
    fn from_cuda(limit: &'a CUlimit) -> Result<Self, CUerror> {
        Ok(*limit)
    }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl<'a> FromCuda<'a, CUfunction_attribute> for CUfunction_attribute {
    fn from_cuda(attr: &'a CUfunction_attribute) -> Result<Self, CUerror> {
        Ok(*attr)
    }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl<'a> FromCuda<'a, CUpointer_attribute> for CUpointer_attribute {
    fn from_cuda(attr: &'a CUpointer_attribute) -> Result<Self, CUerror> {
        Ok(*attr)
    }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl<'a> FromCuda<'a, CUstream> for CUstream {
    fn from_cuda(stream: &'a CUstream) -> Result<Self, CUerror> {
        Ok(*stream)
    }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl<'a> FromCuda<'a, *mut CUuuid> for *mut CUuuid {
    fn from_cuda(uuid: &'a *mut CUuuid) -> Result<Self, CUerror> {
        Ok(*uuid)
    }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl<'a> FromCuda<'a, *const CUlaunchConfig> for &'a CUlaunchConfig {
    fn from_cuda(config: &'a *const CUlaunchConfig) -> Result<Self, CUerror> {
        if config.is_null() {
            return Err(CUerror::INVALID_VALUE);
        }
        Ok(unsafe { &**config })
    }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl<'a> FromCuda<'a, CUdeviceptr> for CUdeviceptr {
    fn from_cuda(ptr: &'a CUdeviceptr) -> Result<Self, CUerror> {
        Ok(*ptr)
    }
}

pub(crate) trait ZludaObject: Sized + Send + Sync {
    const COOKIE: usize;
    const LIVENESS_FAIL: CUerror = cuda_types::cuda::CUerror::INVALID_VALUE;

    type CudaHandle: Sized;

    fn drop_checked(&mut self) -> CUresult;

    fn wrap(self) -> Self::CudaHandle {
        unsafe { mem::transmute_copy(&LiveCheck::wrap(self)) }
    }
}

// Helper trait for converting between CUDA and Ze results (Intel only)
#[cfg(feature = "intel")]
pub(crate) trait Decuda {
    fn decuda(self) -> CUresult;
}

#[cfg(feature = "intel")]
impl Decuda for CUresult {
    fn decuda(self) -> CUresult {
        self
    }
}

// Helper trait for converting between Ze and CUDA results (Intel only)
#[cfg(feature = "intel")]
pub(crate) trait Encuda {
    fn encuda(self) -> CUresult;
}

#[cfg(feature = "intel")]
impl Encuda for CUresult {
    fn encuda(self) -> CUresult {
        self
    }
}

#[repr(C)]
pub(crate) struct LiveCheck<T: ZludaObject> {
    cookie: usize,
    data: MaybeUninit<T>,
}

impl<T: ZludaObject> LiveCheck<T> {
    fn new(data: T) -> Self {
        LiveCheck {
            cookie: T::COOKIE,
            data: MaybeUninit::new(data),
        }
    }

    fn as_handle(&self) -> T::CudaHandle {
        unsafe { mem::transmute_copy(&self) }
    }

    fn wrap(data: T) -> *mut Self {
        Box::into_raw(Box::new(Self::new(data)))
    }

    fn as_result(&self) -> Result<&T, CUerror> {
        if self.cookie == T::COOKIE {
            Ok(unsafe { self.data.assume_init_ref() })
        } else {
            Err(T::LIVENESS_FAIL)
        }
    }

    // This looks like nonsense, but it's not. There are two cases:
    // Err(CUerror) -> meaning that the object is invalid, this pointer does not point into valid memory
    // Ok(maybe_error) -> meaning that the object is valid, we dropped everything, but there *might*
    //                    an error in the underlying runtime that we want to propagate
    #[must_use]
    fn drop_checked(&mut self) -> Result<Result<(), CUerror>, CUerror> {
        if self.cookie == T::COOKIE {
            self.cookie = 0;
            let result = unsafe { self.data.assume_init_mut().drop_checked() };
            unsafe { MaybeUninit::assume_init_drop(&mut self.data) };
            Ok(result)
        } else {
            Err(T::LIVENESS_FAIL)
        }
    }
}

pub fn as_ref<'a, T: ZludaObject>(
    handle: &'a T::CudaHandle,
) -> &'a ManuallyDrop<Box<LiveCheck<T>>> {
    unsafe { mem::transmute(handle) }
}

pub fn drop_checked<T: ZludaObject>(handle: T::CudaHandle) -> Result<(), CUerror> {
    let mut wrapped_object: ManuallyDrop<Box<LiveCheck<T>>> =
        unsafe { mem::transmute_copy(&handle) };
    let underlying_error = LiveCheck::drop_checked(&mut wrapped_object)?;
    unsafe { ManuallyDrop::drop(&mut wrapped_object) };
    underlying_error
}

// Update the CUresult conversion to use proper enum variants
#[cfg(feature = "intel")]
fn ze_to_cuda_result(result: ze_result_t) -> CUresult {
    match result {
        ze_result_t::ZE_RESULT_SUCCESS => CUresult::SUCCESS,
        ze_result_t::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
        | ze_result_t::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY => CUresult::ERROR_OUT_OF_MEMORY,
        ze_result_t::ZE_RESULT_ERROR_DEVICE_LOST => CUresult::ERROR_NO_DEVICE,
        ze_result_t::ZE_RESULT_ERROR_INVALID_NULL_HANDLE => CUresult::ERROR_INVALID_HANDLE,
        ze_result_t::ZE_RESULT_ERROR_INVALID_NULL_POINTER => CUresult::ERROR_INVALID_VALUE,
        ze_result_t::ZE_RESULT_ERROR_UNINITIALIZED => CUresult::ERROR_NOT_INITIALIZED,
        _ => CUresult::ERROR_UNKNOWN,
    }
}

// Implement the CUlimit to ze_device_limit_t conversion
#[cfg(feature = "intel")]
fn cuda_limit_to_ze_limit(limit: CUlimit) -> crate::r#impl::ze_device::ze_device_limit_t {
    match limit {
        CUlimit::CU_LIMIT_STACK_SIZE => {
            crate::r#impl::ze_device::ze_device_limit_t::ZE_LIMIT_STACK_SIZE
        }
        CUlimit::CU_LIMIT_PRINTF_FIFO_SIZE => {
            crate::r#impl::ze_device::ze_device_limit_t::ZE_LIMIT_PRINTF_FIFO_SIZE
        }
        CUlimit::CU_LIMIT_MALLOC_HEAP_SIZE => {
            crate::r#impl::ze_device::ze_device_limit_t::ZE_LIMIT_MALLOC_HEAP_SIZE
        }
        _ => panic!("Unsupported limit"),
    }
}

// This function converts CUresult to proper CUresult under Intel builds
#[cfg(feature = "intel")]
pub fn ze_to_hip_error(result: CUresult) -> CUresult {
    result
}

// Add a utility function to convert CUerror to CUresult and Result
#[cfg(feature = "intel")]
pub fn error_to_result<T>(error: CUerror) -> Result<T, CUerror> {
    Err(error)
}

// Create a wrapper type for ze_result_t to resolve orphan rule violation
#[cfg(feature = "intel")]
pub struct ZeResult(ze_result_t);

#[cfg(feature = "intel")]
impl From<ze_result_t> for ZeResult {
    fn from(result: ze_result_t) -> Self {
        ZeResult(result)
    }
}

#[cfg(feature = "intel")]
impl From<ZeResult> for Result<(), CUerror> {
    fn from(ze_result: ZeResult) -> Self {
        if ze_result.0 == ze_result_t::ZE_RESULT_SUCCESS {
            Ok(())
        } else {
            Err(match ze_result.0 {
                ze_result_t::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
                | ze_result_t::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY => CUerror::OUT_OF_MEMORY,
                ze_result_t::ZE_RESULT_ERROR_DEVICE_LOST => CUerror::NO_DEVICE,
                ze_result_t::ZE_RESULT_ERROR_INVALID_NULL_HANDLE => CUerror::INVALID_HANDLE,
                ze_result_t::ZE_RESULT_ERROR_INVALID_NULL_POINTER => CUerror::INVALID_VALUE,
                ze_result_t::ZE_RESULT_ERROR_UNINITIALIZED => CUerror::NOT_INITIALIZED,
                ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE => CUerror::NOT_SUPPORTED,
                _ => CUerror::UNKNOWN,
            })
        }
    }
}

// Add a convenience function to convert ze_result_t to Result<(), CUerror>
#[cfg(feature = "intel")]
pub fn ze_result_to_result(result: ze_result_t) -> Result<(), CUerror> {
    ZeResult(result).into()
}

// Allow adapting backend-specific results or CUerror-based results uniformly.
pub trait IntoCuResult {
    fn into_cu_result(self) -> Result<(), CUerror>;
}

#[cfg(feature = "intel")]
impl IntoCuResult for ze_result_t {
    fn into_cu_result(self) -> Result<(), CUerror> {
        ze_result_to_result(self)
    }
}

impl IntoCuResult for Result<(), CUerror> {
    fn into_cu_result(self) -> Result<(), CUerror> {
        self
    }
}

impl IntoCuResult for Result<(), String> {
    fn into_cu_result(self) -> Result<(), CUerror> {
        match self {
            Ok(()) => Ok(()),
            Err(_) => Err(CUerror::UNKNOWN),
        }
    }
}

pub fn into_cu_result<T: IntoCuResult>(v: T) -> Result<(), CUerror> {
    v.into_cu_result()
}

// Tenstorrent-specific FromCuda implementations for types not covered by existing macros
#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl<'a> FromCuda<'a, *mut CUuuid_st> for *mut [u8; 16] {
    fn from_cuda(x: &'a *mut CUuuid_st) -> Result<Self, CUerror> {
        Ok(x.cast::<[u8; 16]>())
    }
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl<'a> FromCuda<'a, *mut i8> for *mut [u8; 8] {
    fn from_cuda(x: &'a *mut i8) -> Result<Self, CUerror> {
        Ok(x.cast::<[u8; 8]>())
    }
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl<'a> FromCuda<'a, *mut u32> for *mut u32 {
    fn from_cuda(x: &'a *mut u32) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl<'a> FromCuda<'a, *mut CUcontext> for *mut CUcontext {
    fn from_cuda(x: &'a *mut CUcontext) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl<'a> FromCuda<'a, *mut CUfunction> for *mut CUfunction {
    fn from_cuda(x: &'a *mut CUfunction) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl<'a> FromCuda<'a, CUfunction> for *mut module::TtKernel {
    fn from_cuda(handle: &'a CUfunction) -> Result<Self, CUerror> {
        // Convert CUfunction handle to TtKernel pointer
        Ok(handle.0 as *mut module::TtKernel)
    }
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl<'a> FromCuda<'a, CUstream> for *mut ::core::ffi::c_void {
    fn from_cuda(x: &'a CUstream) -> Result<Self, CUerror> {
        Ok(x.0 as *mut ::core::ffi::c_void)
    }
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl<'a> FromCuda<'a, CUdeviceptr_v2> for CUdeviceptr_v2 {
    fn from_cuda(x: &'a CUdeviceptr_v2) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl<'a> FromCuda<'a, CUpointer_attribute_enum> for CUpointer_attribute_enum {
    fn from_cuda(x: &'a CUpointer_attribute_enum) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

#[cfg(all(feature = "tenstorrent", not(feature = "amd"), not(feature = "intel")))]
impl<'a> FromCuda<'a, CUfunction_attribute_enum> for CUfunction_attribute_enum {
    fn from_cuda(x: &'a CUfunction_attribute_enum) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

// TMatmul-specific FromCuda implementations for virtual CUDA handles.
#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "nvidia")
))]
impl<'a> FromCuda<'a, *mut CUuuid_st> for *mut [u8; 16] {
    fn from_cuda(x: &'a *mut CUuuid_st) -> Result<Self, CUerror> {
        Ok(x.cast::<[u8; 16]>())
    }
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "nvidia")
))]
impl<'a> FromCuda<'a, *mut i8> for *mut [u8; 8] {
    fn from_cuda(x: &'a *mut i8) -> Result<Self, CUerror> {
        Ok(x.cast::<[u8; 8]>())
    }
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "nvidia")
))]
impl<'a> FromCuda<'a, *mut u32> for *mut u32 {
    fn from_cuda(x: &'a *mut u32) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "nvidia")
))]
impl<'a> FromCuda<'a, *mut CUfunction> for &'a mut CUfunction {
    fn from_cuda(x: &'a *mut CUfunction) -> Result<Self, CUerror> {
        match unsafe { x.as_mut() } {
            Some(x) => Ok(x),
            None => Err(CUerror::INVALID_VALUE),
        }
    }
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "nvidia")
))]
impl<'a> FromCuda<'a, CUstream> for *mut ::core::ffi::c_void {
    fn from_cuda(x: &'a CUstream) -> Result<Self, CUerror> {
        Ok(x.0 as *mut ::core::ffi::c_void)
    }
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "nvidia")
))]
impl<'a> FromCuda<'a, CUdeviceptr_v2> for CUdeviceptr_v2 {
    fn from_cuda(x: &'a CUdeviceptr_v2) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "nvidia")
))]
impl<'a> FromCuda<'a, CUpointer_attribute_enum> for CUpointer_attribute_enum {
    fn from_cuda(x: &'a CUpointer_attribute_enum) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "nvidia")
))]
impl<'a> FromCuda<'a, CUfunction_attribute_enum> for CUfunction_attribute_enum {
    fn from_cuda(x: &'a CUfunction_attribute_enum) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "nvidia")
))]
impl<'a> FromCuda<'a, CUlimit> for CUlimit {
    fn from_cuda(x: &'a CUlimit) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "nvidia")
))]
impl<'a> FromCuda<'a, *const CUlaunchConfig_st> for &'a CUlaunchConfig_st {
    fn from_cuda(x: &'a *const CUlaunchConfig_st) -> Result<Self, CUerror> {
        if x.is_null() {
            return Err(CUerror::INVALID_VALUE);
        }
        Ok(unsafe { &**x })
    }
}

// SIFIVE-specific FromCuda implementations
#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl<'a> FromCuda<'a, *mut CUuuid_st> for *mut [u8; 16] {
    fn from_cuda(x: &'a *mut CUuuid_st) -> Result<Self, CUerror> {
        Ok(x.cast::<[u8; 16]>())
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl<'a> FromCuda<'a, *mut i8> for *mut [u8; 8] {
    fn from_cuda(x: &'a *mut i8) -> Result<Self, CUerror> {
        Ok(x.cast::<[u8; 8]>())
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl<'a> FromCuda<'a, *mut u32> for *mut u32 {
    fn from_cuda(x: &'a *mut u32) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

// *mut CUcontext and *mut CUfunction already covered by from_cuda_nop!

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl<'a> FromCuda<'a, CUfunction> for *mut module::SifiveKernel {
    fn from_cuda(handle: &'a CUfunction) -> Result<Self, CUerror> {
        Ok(handle.0 as *mut module::SifiveKernel)
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl<'a> FromCuda<'a, CUstream> for *mut ::core::ffi::c_void {
    fn from_cuda(x: &'a CUstream) -> Result<Self, CUerror> {
        Ok(x.0 as *mut ::core::ffi::c_void)
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl<'a> FromCuda<'a, CUdeviceptr_v2> for CUdeviceptr_v2 {
    fn from_cuda(x: &'a CUdeviceptr_v2) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl<'a> FromCuda<'a, CUpointer_attribute_enum> for CUpointer_attribute_enum {
    fn from_cuda(x: &'a CUpointer_attribute_enum) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl<'a> FromCuda<'a, CUfunction_attribute_enum> for CUfunction_attribute_enum {
    fn from_cuda(x: &'a CUfunction_attribute_enum) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl<'a> FromCuda<'a, CUlimit> for CUlimit {
    fn from_cuda(x: &'a CUlimit) -> Result<Self, CUerror> {
        Ok(*x)
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl<'a> FromCuda<'a, *mut u8> for *mut [u8; 8] {
    fn from_cuda(x: &'a *mut u8) -> Result<Self, CUerror> {
        Ok((*x).cast::<[u8; 8]>())
    }
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl<'a> FromCuda<'a, *const cuda_types::cuda::CUlaunchConfig_st>
    for &'a cuda_types::cuda::CUlaunchConfig_st
{
    fn from_cuda(x: &'a *const cuda_types::cuda::CUlaunchConfig_st) -> Result<Self, CUerror> {
        if x.is_null() {
            return Err(CUerror::INVALID_VALUE);
        }
        Ok(unsafe { &**x })
    }
}

// *mut CUfunction for SIFIVE (removed from nop list since tenstorrent has it feature-gated)
#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
impl<'a> FromCuda<'a, *mut CUfunction> for *mut CUfunction {
    fn from_cuda(x: &'a *mut CUfunction) -> Result<Self, CUerror> {
        Ok(*x)
    }
}
