use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString};
use std::fmt;
use std::fs::OpenOptions;
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

const AXI_INSTANCE_STRIDE: u32 = 0x4000;
const MM2S_DMACR: u32 = 0x0000;
const MM2S_DMASR: u32 = 0x0004;
const MM2S_SA: u32 = 0x0018;
const MM2S_LENGTH: u32 = 0x0028;
const STALL: u32 = 0x1000;
const RESET: u32 = 0x2000;
const DMACR_RESET: u32 = 1 << 2;
const DMASR_HALTED: u32 = 1 << 0;
const DMASR_IDLE: u32 = 1 << 1;
const INSTRUCTION_BYTES: usize = 16;
pub(crate) const XRT_BO_SYNC_TO_DEVICE: i32 = 0;
pub(crate) const XRT_BO_SYNC_FROM_DEVICE: i32 = 1;
const AU250_DIM: usize = 1024;
const AU250_MATRIX_BYTES: usize = AU250_DIM * AU250_DIM / 4;
const AU250_TMATMUL_ASSEMBLY: &str = "ldv v0, PARAM_INPUT\ntmatmul_import v0\ntmatmul_go PARAM_MATRIX\ntmatmul_export v1\nsv v1, PARAM_OUTPUT\nstall\n";

pub(crate) type Handle = *mut libc::c_void;
pub(crate) type Xuid = [u8; 16];

type DeviceOpenFn = unsafe extern "C" fn(u32) -> Handle;
type DeviceCloseFn = unsafe extern "C" fn(Handle) -> i32;
type DeviceLoadXclbinFileFn = unsafe extern "C" fn(Handle, *const libc::c_char) -> i32;
type DeviceGetXclbinUuidFn = unsafe extern "C" fn(Handle, *mut u8) -> i32;
type PlKernelOpenExclusiveFn =
    unsafe extern "C" fn(Handle, *const u8, *const libc::c_char) -> Handle;
type KernelCloseFn = unsafe extern "C" fn(Handle) -> i32;
type KernelArgGroupIdFn = unsafe extern "C" fn(Handle, i32) -> i32;
type KernelReadRegisterFn = unsafe extern "C" fn(Handle, u32, *mut u32) -> i32;
type KernelWriteRegisterFn = unsafe extern "C" fn(Handle, u32, u32) -> i32;
type BoAllocFn = unsafe extern "C" fn(Handle, usize, u64, u32) -> Handle;
type BoFreeFn = unsafe extern "C" fn(Handle) -> i32;
type BoAddressFn = unsafe extern "C" fn(Handle) -> u64;
type BoWriteFn = unsafe extern "C" fn(Handle, *const libc::c_void, usize, usize) -> i32;
type BoReadFn = unsafe extern "C" fn(Handle, *mut libc::c_void, usize, usize) -> i32;
type BoSyncFn = unsafe extern "C" fn(Handle, i32, usize, usize) -> i32;
type XclOpenFn = unsafe extern "C" fn(u32, *const libc::c_char, i32) -> Handle;
type XclCloseFn = unsafe extern "C" fn(Handle);
type XclIpNameToIndexFn = unsafe extern "C" fn(Handle, *const libc::c_char) -> i32;
type XclOpenContextFn = unsafe extern "C" fn(Handle, *const u8, u32, bool) -> i32;
type XclCloseContextFn = unsafe extern "C" fn(Handle, *const u8, u32) -> i32;
type XclRegReadFn = unsafe extern "C" fn(Handle, u32, u32, *mut u32) -> i32;
type XclRegWriteFn = unsafe extern "C" fn(Handle, u32, u32, u32) -> i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InstanceRegisters {
    dma_control: u32,
    dma_source_lo: u32,
    dma_source_hi: u32,
    dma_length: u32,
    stall: u32,
    reset: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct XrtConfig {
    xclbin: PathBuf,
    device_index: u32,
    kernel_name: String,
    ip_name: Option<String>,
    instance: u32,
    memory_arg: i32,
    memory_group: Option<u32>,
    num_vector_registers: u8,
    timeout_ms: u32,
}

impl XrtConfig {
    fn from_env() -> Result<Self, XrtTmatmulError> {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    fn from_lookup(
        mut lookup: impl FnMut(&str) -> Option<String>,
    ) -> Result<Self, XrtTmatmulError> {
        fn parse_u32(
            name: &'static str,
            value: Option<String>,
            default: u32,
        ) -> Result<u32, XrtTmatmulError> {
            value.map_or(Ok(default), |text| {
                text.parse::<u32>()
                    .map_err(|error| XrtTmatmulError::Config(format!("{name}={text:?}: {error}")))
            })
        }

        let xclbin = lookup("HETGPU_XRT_XCLBIN")
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| XrtTmatmulError::Config("HETGPU_XRT_XCLBIN is required".to_string()))?;
        let device_index = parse_u32(
            "HETGPU_XRT_DEVICE_INDEX",
            lookup("HETGPU_XRT_DEVICE_INDEX"),
            0,
        )?;
        let instance = parse_u32("HETGPU_XRT_INSTANCE", lookup("HETGPU_XRT_INSTANCE"), 0)?;
        let timeout_ms = parse_u32(
            "HETGPU_XRT_TIMEOUT_MS",
            lookup("HETGPU_XRT_TIMEOUT_MS"),
            10_000,
        )?;
        if timeout_ms == 0 {
            return Err(XrtTmatmulError::Config(
                "HETGPU_XRT_TIMEOUT_MS must be nonzero".to_string(),
            ));
        }
        let default_memory_arg = 14u32.checked_add(instance).ok_or_else(|| {
            XrtTmatmulError::Config("DDR_PTR argument index overflow".to_string())
        })?;
        let memory_arg_u32 = parse_u32(
            "HETGPU_XRT_MEMORY_ARG",
            lookup("HETGPU_XRT_MEMORY_ARG"),
            default_memory_arg,
        )?;
        let memory_arg = i32::try_from(memory_arg_u32).map_err(|_| {
            XrtTmatmulError::Config(
                "HETGPU_XRT_MEMORY_ARG does not fit an XRT argument index".to_string(),
            )
        })?;
        let kernel_name = lookup("HETGPU_XRT_KERNEL")
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "ternip_ip".to_string());
        let ip_name = lookup("HETGPU_XRT_IP_NAME").filter(|value| !value.trim().is_empty());
        let memory_group = lookup("HETGPU_XRT_MEMORY_GROUP")
            .map(|text| {
                text.parse::<u32>().map_err(|error| {
                    XrtTmatmulError::Config(format!("HETGPU_XRT_MEMORY_GROUP={text:?}: {error}"))
                })
            })
            .transpose()?;
        if ip_name.is_some() && memory_group.is_none() {
            return Err(XrtTmatmulError::Config(
                "HETGPU_XRT_MEMORY_GROUP is required when HETGPU_XRT_IP_NAME is set".to_string(),
            ));
        }
        let num_vector_registers_u32 = parse_u32(
            "HETGPU_XRT_NUM_VECTOR_REGISTERS",
            lookup("HETGPU_XRT_NUM_VECTOR_REGISTERS"),
            8,
        )?;
        let num_vector_registers = u8::try_from(num_vector_registers_u32).map_err(|_| {
            XrtTmatmulError::Config(
                "HETGPU_XRT_NUM_VECTOR_REGISTERS does not fit in u8".to_string(),
            )
        })?;

        Ok(Self {
            xclbin: PathBuf::from(xclbin),
            device_index,
            kernel_name,
            ip_name,
            instance,
            memory_arg,
            memory_group,
            num_vector_registers,
            timeout_ms,
        })
    }
}

pub(crate) trait XrtOps {
    fn device_open(&self, index: u32) -> Handle;
    fn device_close(&self, device: Handle) -> i32;
    fn load_xclbin_file(&self, device: Handle, path: &CStr) -> i32;
    fn get_xclbin_uuid(&self, device: Handle, uuid: &mut Xuid) -> i32;
    fn kernel_open_exclusive(&self, device: Handle, uuid: &Xuid, name: &CStr) -> Handle;
    fn kernel_close(&self, kernel: Handle) -> i32;
    fn kernel_arg_group_id(&self, kernel: Handle, arg: i32) -> i32;
    fn kernel_read_register(&self, kernel: Handle, offset: u32, value: &mut u32) -> i32;
    fn kernel_write_register(&self, kernel: Handle, offset: u32, value: u32) -> i32;
    fn xcl_open(&self, index: u32) -> Handle;
    fn xcl_close(&self, device: Handle);
    fn xcl_ip_name_to_index(&self, device: Handle, name: &CStr) -> i32;
    fn xcl_open_context(&self, device: Handle, uuid: &Xuid, index: u32, shared: bool) -> i32;
    fn xcl_close_context(&self, device: Handle, uuid: &Xuid, index: u32) -> i32;
    fn xcl_reg_read(&self, device: Handle, index: u32, offset: u32, value: &mut u32) -> i32;
    fn xcl_reg_write(&self, device: Handle, index: u32, offset: u32, value: u32) -> i32;
    fn bo_alloc(&self, device: Handle, size: usize, flags: u64, group: u32) -> Handle;
    fn bo_free(&self, bo: Handle) -> i32;
    fn bo_address(&self, bo: Handle) -> u64;
    fn bo_write_range(&self, bo: Handle, bytes: &[u8], offset: usize) -> i32;
    fn bo_read_range(&self, bo: Handle, bytes: &mut [u8], offset: usize) -> i32;
    fn bo_write(&self, bo: Handle, bytes: &[u8]) -> i32 {
        self.bo_write_range(bo, bytes, 0)
    }
    fn bo_read(&self, bo: Handle, bytes: &mut [u8]) -> i32 {
        self.bo_read_range(bo, bytes, 0)
    }
    fn bo_sync(&self, bo: Handle, direction: i32, size: usize, offset: usize) -> i32;
}

struct NativeIpApi {
    library: Handle,
    open: XclOpenFn,
    close: XclCloseFn,
    ip_name_to_index: XclIpNameToIndexFn,
    open_context: XclOpenContextFn,
    close_context: XclCloseContextFn,
    reg_read: XclRegReadFn,
    reg_write: XclRegWriteFn,
}

impl NativeIpApi {
    fn load() -> Result<Self, XrtTmatmulError> {
        let library = open_library(&["libxrt_core.so.2", "libxrt_core.so"])?;
        let result: Result<Self, XrtTmatmulError> = unsafe {
            Ok(Self {
                library,
                open: load_symbol(library, c"xclOpen")?,
                close: load_symbol(library, c"xclClose")?,
                ip_name_to_index: load_symbol(library, c"xclIPName2Index")?,
                open_context: load_symbol(library, c"xclOpenContext")?,
                close_context: load_symbol(library, c"xclCloseContext")?,
                reg_read: load_symbol(library, c"xclRegRead")?,
                reg_write: load_symbol(library, c"xclRegWrite")?,
            })
        };
        if result.is_err() {
            unsafe {
                libc::dlclose(library);
            }
        }
        result.map_err(|error| {
            XrtTmatmulError::DynamicLoad(format!(
                "AP_CTRL_NONE requires the app215-compatible legacy xcl register API: {error}"
            ))
        })
    }
}

impl Drop for NativeIpApi {
    fn drop(&mut self) {
        if !self.library.is_null() {
            unsafe {
                libc::dlclose(self.library);
            }
        }
    }
}

pub(crate) struct RealXrt {
    library: Handle,
    native_ip: Option<NativeIpApi>,
    device_open: DeviceOpenFn,
    device_close: DeviceCloseFn,
    device_load_xclbin_file: DeviceLoadXclbinFileFn,
    device_get_xclbin_uuid: DeviceGetXclbinUuidFn,
    pl_kernel_open_exclusive: PlKernelOpenExclusiveFn,
    kernel_close: KernelCloseFn,
    kernel_arg_group_id: KernelArgGroupIdFn,
    kernel_read_register: KernelReadRegisterFn,
    kernel_write_register: KernelWriteRegisterFn,
    bo_alloc: BoAllocFn,
    bo_free: BoFreeFn,
    bo_address: BoAddressFn,
    bo_write: BoWriteFn,
    bo_read: BoReadFn,
    bo_sync: BoSyncFn,
}

impl RealXrt {
    pub(crate) fn load(needs_native_ip: bool) -> Result<Self, XrtTmatmulError> {
        let library = open_library(&["libxrt_coreutil.so.2", "libxrt_coreutil.so"])?;
        let native_ip = match needs_native_ip.then(NativeIpApi::load).transpose() {
            Ok(api) => api,
            Err(error) => {
                unsafe {
                    libc::dlclose(library);
                }
                return Err(error);
            }
        };

        let result = unsafe {
            Ok(Self {
                library,
                native_ip,
                device_open: load_symbol(library, c"xrtDeviceOpen")?,
                device_close: load_symbol(library, c"xrtDeviceClose")?,
                device_load_xclbin_file: load_symbol(library, c"xrtDeviceLoadXclbinFile")?,
                device_get_xclbin_uuid: load_symbol(library, c"xrtDeviceGetXclbinUUID")?,
                pl_kernel_open_exclusive: load_symbol(library, c"xrtPLKernelOpenExclusive")?,
                kernel_close: load_symbol(library, c"xrtKernelClose")?,
                kernel_arg_group_id: load_symbol(library, c"xrtKernelArgGroupId")?,
                kernel_read_register: load_symbol(library, c"xrtKernelReadRegister")?,
                kernel_write_register: load_symbol(library, c"xrtKernelWriteRegister")?,
                bo_alloc: load_symbol(library, c"xrtBOAlloc")?,
                bo_free: load_symbol(library, c"xrtBOFree")?,
                bo_address: load_symbol(library, c"xrtBOAddress")?,
                bo_write: load_symbol(library, c"xrtBOWrite")?,
                bo_read: load_symbol(library, c"xrtBORead")?,
                bo_sync: load_symbol(library, c"xrtBOSync")?,
            })
        };
        if result.is_err() {
            unsafe {
                libc::dlclose(library);
            }
        }
        result
    }

    fn native_ip(&self) -> &NativeIpApi {
        self.native_ip
            .as_ref()
            .expect("native-IP API must be loaded before native-IP operations")
    }
}

impl Drop for RealXrt {
    fn drop(&mut self) {
        if !self.library.is_null() {
            unsafe {
                libc::dlclose(self.library);
            }
        }
    }
}

impl XrtOps for RealXrt {
    fn device_open(&self, index: u32) -> Handle {
        unsafe { (self.device_open)(index) }
    }

    fn device_close(&self, device: Handle) -> i32 {
        unsafe { (self.device_close)(device) }
    }

    fn load_xclbin_file(&self, device: Handle, path: &CStr) -> i32 {
        unsafe { (self.device_load_xclbin_file)(device, path.as_ptr()) }
    }

    fn get_xclbin_uuid(&self, device: Handle, uuid: &mut Xuid) -> i32 {
        unsafe { (self.device_get_xclbin_uuid)(device, uuid.as_mut_ptr()) }
    }

    fn kernel_open_exclusive(&self, device: Handle, uuid: &Xuid, name: &CStr) -> Handle {
        unsafe { (self.pl_kernel_open_exclusive)(device, uuid.as_ptr(), name.as_ptr()) }
    }

    fn kernel_close(&self, kernel: Handle) -> i32 {
        unsafe { (self.kernel_close)(kernel) }
    }

    fn kernel_arg_group_id(&self, kernel: Handle, arg: i32) -> i32 {
        unsafe { (self.kernel_arg_group_id)(kernel, arg) }
    }

    fn kernel_read_register(&self, kernel: Handle, offset: u32, value: &mut u32) -> i32 {
        unsafe { (self.kernel_read_register)(kernel, offset, value) }
    }

    fn kernel_write_register(&self, kernel: Handle, offset: u32, value: u32) -> i32 {
        unsafe { (self.kernel_write_register)(kernel, offset, value) }
    }

    fn xcl_open(&self, index: u32) -> Handle {
        unsafe { (self.native_ip().open)(index, std::ptr::null(), 0) }
    }

    fn xcl_close(&self, device: Handle) {
        unsafe { (self.native_ip().close)(device) }
    }

    fn xcl_ip_name_to_index(&self, device: Handle, name: &CStr) -> i32 {
        unsafe { (self.native_ip().ip_name_to_index)(device, name.as_ptr()) }
    }

    fn xcl_open_context(&self, device: Handle, uuid: &Xuid, index: u32, shared: bool) -> i32 {
        unsafe { (self.native_ip().open_context)(device, uuid.as_ptr(), index, shared) }
    }

    fn xcl_close_context(&self, device: Handle, uuid: &Xuid, index: u32) -> i32 {
        unsafe { (self.native_ip().close_context)(device, uuid.as_ptr(), index) }
    }

    fn xcl_reg_read(&self, device: Handle, index: u32, offset: u32, value: &mut u32) -> i32 {
        unsafe { (self.native_ip().reg_read)(device, index, offset, value) }
    }

    fn xcl_reg_write(&self, device: Handle, index: u32, offset: u32, value: u32) -> i32 {
        unsafe { (self.native_ip().reg_write)(device, index, offset, value) }
    }

    fn bo_alloc(&self, device: Handle, size: usize, flags: u64, group: u32) -> Handle {
        unsafe { (self.bo_alloc)(device, size, flags, group) }
    }

    fn bo_free(&self, bo: Handle) -> i32 {
        unsafe { (self.bo_free)(bo) }
    }

    fn bo_address(&self, bo: Handle) -> u64 {
        unsafe { (self.bo_address)(bo) }
    }

    fn bo_write_range(&self, bo: Handle, bytes: &[u8], offset: usize) -> i32 {
        unsafe { (self.bo_write)(bo, bytes.as_ptr().cast(), bytes.len(), offset) }
    }

    fn bo_read_range(&self, bo: Handle, bytes: &mut [u8], offset: usize) -> i32 {
        unsafe { (self.bo_read)(bo, bytes.as_mut_ptr().cast(), bytes.len(), offset) }
    }

    fn bo_sync(&self, bo: Handle, direction: i32, size: usize, offset: usize) -> i32 {
        unsafe { (self.bo_sync)(bo, direction, size, offset) }
    }
}

fn open_library(candidates: &[&str]) -> Result<Handle, XrtTmatmulError> {
    let mut failures = Vec::new();
    for candidate in candidates {
        let candidate_c = CString::new(*candidate).expect("XRT library name has no NUL");
        let library =
            unsafe { libc::dlopen(candidate_c.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if !library.is_null() {
            return Ok(library);
        }
        failures.push(format!("{candidate}: {}", dl_error_message()));
    }
    Err(XrtTmatmulError::DynamicLoad(failures.join("; ")))
}

unsafe fn load_symbol<T: Copy>(library: Handle, name: &CStr) -> Result<T, XrtTmatmulError> {
    unsafe {
        libc::dlerror();
    }
    let symbol = unsafe { libc::dlsym(library, name.as_ptr()) };
    if symbol.is_null() {
        return Err(XrtTmatmulError::DynamicLoad(format!(
            "{}: {}",
            name.to_string_lossy(),
            dl_error_message()
        )));
    }
    if std::mem::size_of::<T>() != std::mem::size_of::<Handle>() {
        return Err(XrtTmatmulError::DynamicLoad(format!(
            "{}: incompatible function pointer size",
            name.to_string_lossy()
        )));
    }
    Ok(unsafe { std::mem::transmute_copy(&symbol) })
}

fn dl_error_message() -> String {
    let error = unsafe { libc::dlerror() };
    if error.is_null() {
        "unknown dynamic loader error".to_string()
    } else {
        unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum XrtTmatmulError {
    Config(String),
    DynamicLoad(String),
    Xrt { operation: &'static str, code: i32 },
    NullHandle(&'static str),
    InvalidProgram(String),
    InvalidBuffer(String),
    Assemble(String),
    Timeout { timeout_ms: u32 },
    Quiesce { primary: String, cleanup: String },
}

impl fmt::Display for XrtTmatmulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(message) => write!(f, "XRT tmatmul configuration failed: {message}"),
            Self::DynamicLoad(message) => write!(f, "XRT runtime loading failed: {message}"),
            Self::Xrt { operation, code } => {
                write!(f, "XRT {operation} failed with code {code}")
            }
            Self::NullHandle(operation) => write!(f, "XRT {operation} returned a null handle"),
            Self::InvalidProgram(message) => write!(f, "invalid tmatmul program: {message}"),
            Self::InvalidBuffer(message) => write!(f, "invalid tmatmul buffer: {message}"),
            Self::Assemble(message) => write!(f, "tmatmul assembly failed: {message}"),
            Self::Timeout { timeout_ms } => {
                write!(f, "XRT tmatmul timed out after {timeout_ms} ms")
            }
            Self::Quiesce { primary, cleanup } => write!(
                f,
                "XRT tmatmul failed ({primary}) and device quiescence could not be confirmed ({cleanup}); handles were retained to protect live BOs"
            ),
        }
    }
}

impl std::error::Error for XrtTmatmulError {}

pub(crate) struct XrtTmatmulRequest<'a> {
    pub(crate) assembly: &'a str,
    pub(crate) matrix_label: &'a str,
    pub(crate) input_label: &'a str,
    pub(crate) output_label: &'a str,
    pub(crate) matrix: &'a [u8],
    pub(crate) input: &'a [u8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct XrtTmatmulStatus {
    pub(crate) stall_code: u32,
    pub(crate) program_bytes: usize,
    pub(crate) matrix_address: u64,
    pub(crate) input_address: u64,
    pub(crate) output_address: u64,
    pub(crate) program_address: u64,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct XrtCuTarget {
    pub(crate) ip_name: String,
    pub(crate) memory_group: u32,
    pub(crate) lanes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct XrtWaveJob {
    pub(crate) request_id: u64,
    pub(crate) cu_index: usize,
    pub(crate) matrix_key: [u8; 32],
    pub(crate) matrix_sha256: [u8; 32],
    pub(crate) matrix: Arc<[u8]>,
    pub(crate) input: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct XrtWaveCompletion {
    pub(crate) request_id: u64,
    pub(crate) cu_index: usize,
    pub(crate) stall_code: u32,
    pub(crate) output: Vec<u8>,
    pub(crate) dispatch_to_stall_ns: u64,
    pub(crate) program_bytes: usize,
    pub(crate) matrix_key: [u8; 32],
    pub(crate) matrix_sha256: [u8; 32],
    pub(crate) matrix_address: u64,
    pub(crate) matrix_cache_hit: bool,
    pub(crate) matrix_bytes_transferred: usize,
    pub(crate) program_address: u64,
    pub(crate) program_sha256: [u8; 32],
    pub(crate) program_cache_hit: bool,
    pub(crate) encoded_program: Vec<u8>,
    pub(crate) trace_mode: String,
    pub(crate) model_context_limit: u32,
    pub(crate) trace_semantic_sha256: [u8; 32],
    pub(crate) trace_assembly_sha256: [u8; 32],
    pub(crate) replay_safe_program_sha256: [u8; 32],
    pub(crate) trace_assembly: String,
    pub(crate) trace_instructions: Vec<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct QwenTraceConfig {
    mode: String,
    model_context_limit: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct XrtPoolConfig {
    xclbin: PathBuf,
    device_index: u32,
    targets: Vec<XrtCuTarget>,
    num_vector_registers: u8,
    timeout_ms: u32,
    qwen_trace: Option<QwenTraceConfig>,
    resident_matrix_cache_bytes: usize,
    bar0_resource: Option<PathBuf>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct XrtCuTable {
    version: u32,
    cus: Vec<XrtCuTarget>,
}

struct Bar0Mapping {
    address: *mut u8,
    len: usize,
}

// SAFETY: register access is serialized by the process-global pool mutex, and
// the mapping remains owned by the pool until all CU work has stopped.
unsafe impl Send for Bar0Mapping {}

impl Bar0Mapping {
    fn open(path: &PathBuf) -> Result<Self, XrtTmatmulError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|error| {
                XrtTmatmulError::Config(format!(
                    "cannot open HETGPU_XRT_BAR0_RESOURCE {}: {error}",
                    path.display()
                ))
            })?;
        let len = usize::try_from(
            file.metadata()
                .map_err(|error| {
                    XrtTmatmulError::Config(format!(
                        "cannot stat HETGPU_XRT_BAR0_RESOURCE {}: {error}",
                        path.display()
                    ))
                })?
                .len(),
        )
        .map_err(|_| XrtTmatmulError::Config("BAR0 length does not fit usize".to_string()))?;
        let minimum = 0x0181_0000usize + RESET as usize + 4;
        if len < minimum {
            return Err(XrtTmatmulError::Config(format!(
                "HETGPU_XRT_BAR0_RESOURCE has {len} bytes, expected at least {minimum}"
            )));
        }
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if address == libc::MAP_FAILED {
            return Err(XrtTmatmulError::Config(format!(
                "cannot map HETGPU_XRT_BAR0_RESOURCE {}: {}",
                path.display(),
                std::io::Error::last_os_error()
            )));
        }
        Ok(Self {
            address: address.cast(),
            len,
        })
    }

    fn address(&self, base: u64, offset: u32) -> Result<*mut u32, XrtTmatmulError> {
        let absolute = base
            .checked_add(u64::from(offset))
            .ok_or_else(|| XrtTmatmulError::Config("BAR0 register offset overflow".to_string()))?;
        let end = absolute
            .checked_add(4)
            .ok_or_else(|| XrtTmatmulError::Config("BAR0 register end overflow".to_string()))?;
        if end > self.len as u64 || absolute % 4 != 0 {
            return Err(XrtTmatmulError::Config(format!(
                "BAR0 register range 0x{absolute:x}:0x{end:x} is invalid"
            )));
        }
        Ok(unsafe { self.address.add(absolute as usize).cast() })
    }

    fn read(&self, base: u64, offset: u32) -> Result<u32, XrtTmatmulError> {
        Ok(unsafe { std::ptr::read_volatile(self.address(base, offset)?) })
    }

    fn write(&self, base: u64, offset: u32, value: u32) -> Result<(), XrtTmatmulError> {
        unsafe { std::ptr::write_volatile(self.address(base, offset)?, value) };
        Ok(())
    }
}

impl Drop for Bar0Mapping {
    fn drop(&mut self) {
        if !self.address.is_null() && self.len != 0 {
            unsafe {
                libc::munmap(self.address.cast(), self.len);
            }
        }
    }
}

fn maxcores_register_base(ip_name: &str) -> Option<u64> {
    match ip_name {
        "ternip_big:ternip_big_1" => Some(0x00c1_0000),
        "ternip_big:ternip_big_2" => Some(0x0181_0000),
        "ternip_big:ternip_big_3" => Some(0x0141_0000),
        "ternip_small:ternip_small_1" => Some(0x0101_0000),
        _ => None,
    }
}

impl XrtPoolConfig {
    fn maxcores_targets() -> Vec<XrtCuTarget> {
        vec![
            XrtCuTarget {
                ip_name: "ternip_big:ternip_big_1".to_string(),
                memory_group: 0,
                lanes: 9,
            },
            XrtCuTarget {
                ip_name: "ternip_big:ternip_big_2".to_string(),
                memory_group: 3,
                lanes: 9,
            },
            XrtCuTarget {
                ip_name: "ternip_big:ternip_big_3".to_string(),
                memory_group: 2,
                lanes: 9,
            },
            XrtCuTarget {
                ip_name: "ternip_small:ternip_small_1".to_string(),
                memory_group: 1,
                lanes: 6,
            },
        ]
    }

    fn parse_cu_table(json: &str) -> Result<Vec<XrtCuTarget>, XrtTmatmulError> {
        let table: XrtCuTable = serde_json::from_str(json).map_err(|error| {
            XrtTmatmulError::Config(format!("HETGPU_XRT_CU_CONFIG is not valid JSON: {error}"))
        })?;
        if table.version != 1 {
            return Err(XrtTmatmulError::Config(format!(
                "unsupported HETGPU_XRT_CU_CONFIG version {}, expected 1",
                table.version
            )));
        }
        Self::validate_targets(&table.cus)?;
        Ok(table.cus)
    }

    fn validate_targets(targets: &[XrtCuTarget]) -> Result<(), XrtTmatmulError> {
        if targets.is_empty() || targets.len() > 4 {
            return Err(XrtTmatmulError::Config(format!(
                "XRT CU table must contain between 1 and 4 CUs, got {}",
                targets.len()
            )));
        }
        let mut names = HashSet::new();
        let mut groups = HashSet::new();
        for target in targets {
            if target.ip_name.trim().is_empty() || target.ip_name.as_bytes().contains(&0) {
                return Err(XrtTmatmulError::Config(
                    "XRT CU IP names must not be empty or contain NUL".to_string(),
                ));
            }
            if !names.insert(target.ip_name.as_str()) {
                return Err(XrtTmatmulError::Config(format!(
                    "duplicate XRT CU IP name {:?}",
                    target.ip_name
                )));
            }
            if !groups.insert(target.memory_group) {
                return Err(XrtTmatmulError::Config(format!(
                    "duplicate XRT CU memory group {}",
                    target.memory_group
                )));
            }
            if !matches!(target.lanes, 6 | 9) {
                return Err(XrtTmatmulError::Config(format!(
                    "XRT CU {:?} has {} lanes; expected 6 or 9",
                    target.ip_name, target.lanes
                )));
            }
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), XrtTmatmulError> {
        Self::validate_targets(&self.targets)?;
        if let Some(trace) = &self.qwen_trace {
            if self.targets != Self::maxcores_targets() {
                return Err(XrtTmatmulError::Config(
                    "strict Qwen IQ1_S execution requires the exact four-CU MaxCores topology"
                        .to_string(),
                ));
            }
            if let Some(resource) = &self.bar0_resource {
                let pinned = std::path::Path::new("/sys/bus/pci/devices/0000:64:00.1/resource0");
                if resource != pinned {
                    return Err(XrtTmatmulError::Config(format!(
                        "strict Qwen IQ1_S BAR0 access requires {}, got {}",
                        pinned.display(),
                        resource.display()
                    )));
                }
            }
            super::iq1s_trace::build_handwritten_trace(trace.model_context_limit)
                .map_err(XrtTmatmulError::Config)?;
            if !matches!(trace.mode.as_str(), "handwritten" | "compiler") {
                return Err(XrtTmatmulError::Config(
                    "HETGPU_IQ1S_TRACE_MODE must be exactly handwritten or compiler".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn from_env() -> Result<Self, XrtTmatmulError> {
        fn parse_u32(name: &'static str, default: u32) -> Result<u32, XrtTmatmulError> {
            std::env::var(name).map_or(Ok(default), |text| {
                text.parse::<u32>()
                    .map_err(|error| XrtTmatmulError::Config(format!("{name}={text:?}: {error}")))
            })
        }

        fn parse_u64(name: &'static str, default: u64) -> Result<u64, XrtTmatmulError> {
            std::env::var(name).map_or(Ok(default), |text| {
                text.parse::<u64>()
                    .map_err(|error| XrtTmatmulError::Config(format!("{name}={text:?}: {error}")))
            })
        }

        let xclbin = std::env::var("HETGPU_XRT_XCLBIN")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| XrtTmatmulError::Config("HETGPU_XRT_XCLBIN is required".to_string()))?;
        let device_index = parse_u32("HETGPU_XRT_DEVICE_INDEX", 0)?;
        let timeout_ms = parse_u32("HETGPU_XRT_TIMEOUT_MS", 10_000)?;
        if timeout_ms == 0 {
            return Err(XrtTmatmulError::Config(
                "HETGPU_XRT_TIMEOUT_MS must be nonzero".to_string(),
            ));
        }
        let num_vector_registers = u8::try_from(parse_u32("HETGPU_XRT_NUM_VECTOR_REGISTERS", 4)?)
            .map_err(|_| {
            XrtTmatmulError::Config(
                "HETGPU_XRT_NUM_VECTOR_REGISTERS does not fit in u8".to_string(),
            )
        })?;
        let targets = match std::env::var("HETGPU_XRT_CU_CONFIG") {
            Ok(json) if !json.trim().is_empty() => Self::parse_cu_table(&json)?,
            _ => Self::maxcores_targets(),
        };
        let resident_matrix_cache_bytes = usize::try_from(parse_u64(
            "HETGPU_XRT_RESIDENT_MATRIX_CACHE_BYTES",
            512 * 1024 * 1024,
        )?)
        .map_err(|_| {
            XrtTmatmulError::Config(
                "HETGPU_XRT_RESIDENT_MATRIX_CACHE_BYTES does not fit usize".to_string(),
            )
        })?;
        let bar0_resource = std::env::var("HETGPU_XRT_BAR0_RESOURCE")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from);
        if resident_matrix_cache_bytes < AU250_MATRIX_BYTES {
            return Err(XrtTmatmulError::Config(format!(
                "HETGPU_XRT_RESIDENT_MATRIX_CACHE_BYTES must be at least {AU250_MATRIX_BYTES}"
            )));
        }
        let qwen_trace = if std::env::var("HETGPU_QWEN_IQ1S_STRICT").as_deref() == Ok("1") {
            let mode = std::env::var("HETGPU_IQ1S_TRACE_MODE").map_err(|_| {
                XrtTmatmulError::Config(
                    "HETGPU_IQ1S_TRACE_MODE is required in strict Qwen mode".to_string(),
                )
            })?;
            Some(QwenTraceConfig {
                mode,
                model_context_limit: parse_u32(
                    "HETGPU_QWEN_MODEL_CONTEXT_LIMIT",
                    super::iq1s_trace::QWEN_MODEL_CONTEXT_LIMIT,
                )?,
            })
        } else {
            None
        };
        let config = Self {
            xclbin: PathBuf::from(xclbin),
            device_index,
            targets,
            num_vector_registers,
            timeout_ms,
            qwen_trace,
            resident_matrix_cache_bytes,
            bar0_resource,
        };
        config.validate()?;
        Ok(config)
    }
}

fn expected_vector_bytes(lanes: usize) -> usize {
    AU250_DIM * lanes * 2
}

fn validate_wave_job(target: &XrtCuTarget, job: &XrtWaveJob) -> Result<(), XrtTmatmulError> {
    if job.matrix_key == [0; 32] || job.matrix_sha256 == [0; 32] {
        return Err(XrtTmatmulError::InvalidBuffer(
            "matrix cache key and content SHA-256 must be nonzero".to_string(),
        ));
    }
    if job.matrix.len() != AU250_MATRIX_BYTES {
        return Err(XrtTmatmulError::InvalidBuffer(format!(
            "matrix has {} bytes, expected {AU250_MATRIX_BYTES}",
            job.matrix.len()
        )));
    }
    let expected = expected_vector_bytes(target.lanes);
    if job.input.len() != expected {
        return Err(XrtTmatmulError::InvalidBuffer(format!(
            "input has {} bytes, expected {expected}",
            job.input.len()
        )));
    }
    Ok(())
}

struct ResidentMatrixEntry {
    bo: Handle,
    address: u64,
    content_sha256: [u8; 32],
    last_used: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BoundProgramKey {
    matrix_key: [u8; 32],
    matrix_address: u64,
    input_address: u64,
    output_address: u64,
    num_vector_registers: u8,
    memory_group: u32,
    lanes: usize,
    qwen_trace: Option<QwenTraceConfig>,
}

struct BoundProgramEntry {
    bo: Handle,
    address: u64,
    bytes: usize,
    sha256: [u8; 32],
    program: Vec<u8>,
    trace: ProgramTraceMetadata,
}

#[derive(Debug, Clone)]
struct ProgramTraceMetadata {
    mode: String,
    model_context_limit: u32,
    semantic_sha256: [u8; 32],
    assembly_sha256: [u8; 32],
    replay_safe_program_sha256: [u8; 32],
    assembly: String,
    instructions: Vec<Vec<String>>,
}

struct ReusableCu {
    target: XrtCuTarget,
    ip_device: Handle,
    ip_index: u32,
    register_base: Option<u64>,
    input_bo: Handle,
    output_bo: Handle,
    input_address: u64,
    output_address: u64,
    matrix_cache: HashMap<[u8; 32], ResidentMatrixEntry>,
    matrix_cache_bytes: usize,
    matrix_cache_capacity: usize,
    cache_clock: u64,
    program_cache: HashMap<BoundProgramKey, BoundProgramEntry>,
    release_handles: bool,
}

struct Pool<O: XrtOps> {
    ops: O,
    device: Handle,
    uuid: Xuid,
    cus: Vec<ReusableCu>,
    timeout_ms: u32,
    num_vector_registers: u8,
    qwen_trace: Option<QwenTraceConfig>,
    poisoned: bool,
    release_device: bool,
    bar0: Option<Bar0Mapping>,
}

#[derive(Debug, Clone)]
struct PreparedWaveJob {
    matrix_address: u64,
    matrix_cache_hit: bool,
    matrix_bytes_transferred: usize,
    program_address: u64,
    program_bytes: usize,
    program_sha256: [u8; 32],
    program_cache_hit: bool,
    encoded_program: Vec<u8>,
    trace: ProgramTraceMetadata,
}

pub(crate) struct XrtTmatmulPool {
    inner: Pool<RealXrt>,
}

static PERSISTENT_POOL: OnceLock<Mutex<Result<XrtTmatmulPool, String>>> = OnceLock::new();

// SAFETY: every raw XRT handle is exclusively owned by this pool, all access is
// serialized by the quantized executors' process-global mutex, and destruction is
// performed only by the owning pool. Raw-handle owners deliberately are not Sync.
unsafe impl Send for XrtTmatmulPool {}

fn with_pool_state<P, T>(
    state: &OnceLock<Mutex<Result<P, String>>>,
    initialize: impl FnOnce() -> Result<P, String>,
    operation: impl FnOnce(&mut P) -> Result<T, String>,
) -> Result<T, String> {
    let state = state.get_or_init(|| Mutex::new(initialize()));
    let mut guard = state
        .lock()
        .map_err(|_| "AU250 XRT pool mutex poisoned".to_string())?;
    match &mut *guard {
        Ok(pool) => operation(pool),
        Err(error) => Err(error.clone()),
    }
}

pub(crate) fn with_persistent_pool<T>(
    operation: impl FnOnce(&mut XrtTmatmulPool) -> Result<T, String>,
) -> Result<T, String> {
    with_pool_state(
        &PERSISTENT_POOL,
        || XrtTmatmulPool::open_from_env().map_err(|error| error.to_string()),
        operation,
    )
}

impl XrtTmatmulPool {
    pub(crate) fn open_from_env() -> Result<Self, XrtTmatmulError> {
        let config = XrtPoolConfig::from_env()?;
        let ops = RealXrt::load(config.bar0_resource.is_none())?;
        Ok(Self {
            inner: Pool::open_with_ops(ops, config)?,
        })
    }

    pub(crate) fn lane_capacities(&self) -> Vec<usize> {
        self.inner.cus.iter().map(|cu| cu.target.lanes).collect()
    }

    pub(crate) fn run_wave(
        &mut self,
        jobs: Vec<XrtWaveJob>,
    ) -> Result<Vec<XrtWaveCompletion>, XrtTmatmulError> {
        self.inner.run_wave(jobs)
    }
}

impl<O: XrtOps> Pool<O> {
    fn open_with_ops(ops: O, config: XrtPoolConfig) -> Result<Self, XrtTmatmulError> {
        config.validate()?;
        let bar0 = config
            .bar0_resource
            .as_ref()
            .map(Bar0Mapping::open)
            .transpose()?;
        let device = ops.device_open(config.device_index);
        if device.is_null() {
            return Err(XrtTmatmulError::NullHandle("xrtDeviceOpen"));
        }
        let mut pool = Self {
            ops,
            device,
            uuid: [0; 16],
            cus: Vec::with_capacity(config.targets.len()),
            timeout_ms: config.timeout_ms,
            num_vector_registers: config.num_vector_registers,
            qwen_trace: config.qwen_trace.clone(),
            poisoned: false,
            release_device: true,
            bar0,
        };

        let xclbin = config.xclbin.to_str().ok_or_else(|| {
            XrtTmatmulError::Config("HETGPU_XRT_XCLBIN must be valid UTF-8".to_string())
        })?;
        let xclbin = CString::new(xclbin).map_err(|_| {
            XrtTmatmulError::Config("HETGPU_XRT_XCLBIN contains a NUL byte".to_string())
        })?;
        check_xrt(
            "xrtDeviceLoadXclbinFile",
            pool.ops.load_xclbin_file(pool.device, &xclbin),
        )?;
        check_xrt(
            "xrtDeviceGetXclbinUUID",
            pool.ops.get_xclbin_uuid(pool.device, &mut pool.uuid),
        )?;

        for target in config.targets {
            let (ip_device, ip_index, register_base) = if pool.bar0.is_some() {
                let base = maxcores_register_base(&target.ip_name).ok_or_else(|| {
                    XrtTmatmulError::Config(format!(
                        "BAR0 register access does not recognize CU {:?}",
                        target.ip_name
                    ))
                })?;
                (std::ptr::null_mut(), 0, Some(base))
            } else {
                let ip_device = pool.ops.xcl_open(config.device_index);
                if ip_device.is_null() {
                    return Err(XrtTmatmulError::NullHandle("xclOpen"));
                }
                let ip_name = CString::new(target.ip_name.as_str()).map_err(|_| {
                    XrtTmatmulError::Config("XRT CU IP name contains a NUL byte".to_string())
                })?;
                let raw_index = pool.ops.xcl_ip_name_to_index(ip_device, &ip_name);
                if raw_index < 0 {
                    pool.ops.xcl_close(ip_device);
                    return Err(XrtTmatmulError::Xrt {
                        operation: "xclIPName2Index",
                        code: raw_index,
                    });
                }
                let ip_index = raw_index as u32;
                if let Err(error) = check_xrt(
                    "xclOpenContext",
                    pool.ops
                        .xcl_open_context(ip_device, &pool.uuid, ip_index, false),
                ) {
                    pool.ops.xcl_close(ip_device);
                    return Err(error);
                }
                (ip_device, ip_index, None)
            };

            pool.cus.push(ReusableCu {
                target,
                ip_device,
                ip_index,
                register_base,
                input_bo: std::ptr::null_mut(),
                output_bo: std::ptr::null_mut(),
                input_address: 0,
                output_address: 0,
                matrix_cache: HashMap::new(),
                matrix_cache_bytes: 0,
                matrix_cache_capacity: config.resident_matrix_cache_bytes,
                cache_clock: 0,
                program_cache: HashMap::new(),
                release_handles: true,
            });
            let cu_index = pool.cus.len() - 1;
            let group = pool.cus[cu_index].target.memory_group;
            let vector_bytes = expected_vector_bytes(pool.cus[cu_index].target.lanes);

            // Input/output BOs are stable for the CU lifetime. Matrix and
            // address-bound program BOs are populated lazily in resident caches.
            let input_bo = pool.allocate_bo(vector_bytes, group, "xrtBOAlloc(input)")?;
            pool.cus[cu_index].input_bo = input_bo;
            let output_bo = pool.allocate_bo(vector_bytes, group, "xrtBOAlloc(output)")?;
            pool.cus[cu_index].output_bo = output_bo;
            pool.cus[cu_index].input_address = pool.bo_address(input_bo, "input")?;
            pool.cus[cu_index].output_address = pool.bo_address(output_bo, "output")?;
        }

        Ok(pool)
    }

    fn allocate_bo(
        &self,
        size: usize,
        group: u32,
        operation: &'static str,
    ) -> Result<Handle, XrtTmatmulError> {
        let bo = self.ops.bo_alloc(self.device, size, 0, group);
        if bo.is_null() {
            Err(XrtTmatmulError::NullHandle(operation))
        } else {
            Ok(bo)
        }
    }

    fn bo_address(&self, bo: Handle, name: &'static str) -> Result<u64, XrtTmatmulError> {
        let address = self.ops.bo_address(bo);
        if address == 0 || address == u64::MAX {
            Err(XrtTmatmulError::InvalidBuffer(format!(
                "xrtBOAddress returned 0x{address:016x} for {name}"
            )))
        } else {
            Ok(address)
        }
    }

    fn prepare_wave_job(&mut self, job: &XrtWaveJob) -> Result<PreparedWaveJob, XrtTmatmulError> {
        let actual_sha256: [u8; 32] = Sha256::digest(&job.matrix).into();
        if actual_sha256 != job.matrix_sha256 {
            return Err(XrtTmatmulError::InvalidBuffer(format!(
                "wave job {} matrix content does not match its SHA-256 identity",
                job.request_id
            )));
        }

        let cu_index = job.cu_index;
        self.cus[cu_index].cache_clock = self.cus[cu_index]
            .cache_clock
            .checked_add(1)
            .ok_or_else(|| XrtTmatmulError::Config("resident cache clock overflow".to_string()))?;
        let clock = self.cus[cu_index].cache_clock;
        let existing = self.cus[cu_index]
            .matrix_cache
            .get(&job.matrix_key)
            .map(|entry| (entry.address, entry.content_sha256));
        let (matrix_address, matrix_cache_hit, matrix_bytes_transferred) =
            if let Some((address, content_sha256)) = existing {
                if content_sha256 != job.matrix_sha256 {
                    return Err(XrtTmatmulError::InvalidBuffer(format!(
                        "wave job {} reuses a resident matrix key with changed content",
                        job.request_id
                    )));
                }
                self.cus[cu_index]
                    .matrix_cache
                    .get_mut(&job.matrix_key)
                    .expect("resident entry was just observed")
                    .last_used = clock;
                (address, true, 0)
            } else {
                while self.cus[cu_index]
                    .matrix_cache_bytes
                    .checked_add(job.matrix.len())
                    .ok_or_else(|| {
                        XrtTmatmulError::Config("resident matrix byte count overflow".to_string())
                    })?
                    > self.cus[cu_index].matrix_cache_capacity
                {
                    let evict_key = self.cus[cu_index]
                        .matrix_cache
                        .iter()
                        .min_by_key(|(_, entry)| entry.last_used)
                        .map(|(key, _)| *key)
                        .ok_or_else(|| {
                            XrtTmatmulError::Config(
                                "resident matrix cache cannot fit one AU250 tile".to_string(),
                            )
                        })?;
                    let evicted = self.cus[cu_index]
                        .matrix_cache
                        .remove(&evict_key)
                        .expect("selected resident matrix exists");
                    self.cus[cu_index].matrix_cache_bytes = self.cus[cu_index]
                        .matrix_cache_bytes
                        .checked_sub(AU250_MATRIX_BYTES)
                        .ok_or_else(|| {
                            XrtTmatmulError::Config(
                                "resident matrix cache accounting underflow".to_string(),
                            )
                        })?;
                    let program_keys = self.cus[cu_index]
                        .program_cache
                        .keys()
                        .filter(|key| key.matrix_key == evict_key)
                        .cloned()
                        .collect::<Vec<_>>();
                    for key in program_keys {
                        if let Some(program) = self.cus[cu_index].program_cache.remove(&key) {
                            check_xrt("xrtBOFree(program eviction)", self.ops.bo_free(program.bo))?;
                        }
                    }
                    check_xrt("xrtBOFree(matrix eviction)", self.ops.bo_free(evicted.bo))?;
                }

                let group = self.cus[cu_index].target.memory_group;
                let bo =
                    self.allocate_bo(job.matrix.len(), group, "xrtBOAlloc(resident matrix)")?;
                let address = match self.bo_address(bo, "resident matrix") {
                    Ok(address) => address,
                    Err(error) => {
                        let _ = self.ops.bo_free(bo);
                        return Err(error);
                    }
                };
                if let Err(error) = bo_write_and_sync(&self.ops, bo, &job.matrix, "resident matrix")
                {
                    let _ = self.ops.bo_free(bo);
                    return Err(error);
                }
                self.cus[cu_index].matrix_cache.insert(
                    job.matrix_key,
                    ResidentMatrixEntry {
                        bo,
                        address,
                        content_sha256: job.matrix_sha256,
                        last_used: clock,
                    },
                );
                self.cus[cu_index].matrix_cache_bytes = self.cus[cu_index]
                    .matrix_cache_bytes
                    .checked_add(job.matrix.len())
                    .ok_or_else(|| {
                        XrtTmatmulError::Config("resident matrix byte count overflow".to_string())
                    })?;
                (address, false, job.matrix.len())
            };

        let target = self.cus[cu_index].target.clone();
        let input_address = self.cus[cu_index].input_address;
        let output_address = self.cus[cu_index].output_address;
        let program_key = BoundProgramKey {
            matrix_key: job.matrix_key,
            matrix_address,
            input_address,
            output_address,
            num_vector_registers: self.num_vector_registers,
            memory_group: target.memory_group,
            lanes: target.lanes,
            qwen_trace: self.qwen_trace.clone(),
        };
        if let Some(program) = self.cus[cu_index].program_cache.get(&program_key) {
            return Ok(PreparedWaveJob {
                matrix_address,
                matrix_cache_hit,
                matrix_bytes_transferred,
                program_address: program.address,
                program_bytes: program.bytes,
                program_sha256: program.sha256,
                program_cache_hit: true,
                encoded_program: program.program.clone(),
                trace: program.trace.clone(),
            });
        }

        let labels = bind_labels(
            "PARAM_MATRIX",
            "PARAM_INPUT",
            "PARAM_OUTPUT",
            matrix_address,
            input_address,
            output_address,
        )?;
        let (replay_safe_program, trace_metadata) = if let Some(trace) = &self.qwen_trace {
            let selected = super::iq1s_trace::build_selected_trace(
                &trace.mode,
                trace.model_context_limit,
                &labels,
                self.num_vector_registers,
            )
            .map_err(XrtTmatmulError::Assemble)?;
            let metadata = ProgramTraceMetadata {
                mode: selected.selected_kind.as_str().to_string(),
                model_context_limit: selected.model_context_limit,
                semantic_sha256: selected.semantic_sha256,
                assembly_sha256: selected.assembly_sha256,
                replay_safe_program_sha256: selected.selected_sha256,
                assembly: selected.assembly,
                instructions: selected.instructions,
            };
            (selected.program, metadata)
        } else {
            let program = super::cxl_tmatmul::assemble_tmatmul_program_for_vector_registers(
                AU250_TMATMUL_ASSEMBLY,
                &labels,
                self.num_vector_registers,
            )
            .map_err(|error| XrtTmatmulError::Assemble(error.to_string()))?;
            let metadata = ProgramTraceMetadata {
                mode: "legacy".to_string(),
                model_context_limit: 0,
                semantic_sha256: [0; 32],
                assembly_sha256: Sha256::digest(AU250_TMATMUL_ASSEMBLY.as_bytes()).into(),
                replay_safe_program_sha256: Sha256::digest(&program).into(),
                assembly: AU250_TMATMUL_ASSEMBLY.to_string(),
                instructions: Vec::new(),
            };
            (program, metadata)
        };
        let program = compact_xrt_program(&replay_safe_program)?;
        validate_program(&program)?;
        let program_sha256 = Sha256::digest(&program).into();
        let program_bo =
            self.allocate_bo(program.len(), target.memory_group, "xrtBOAlloc(program)")?;
        let program_address = match self.bo_address(program_bo, "program") {
            Ok(address) => address,
            Err(error) => {
                let _ = self.ops.bo_free(program_bo);
                return Err(error);
            }
        };
        if let Err(error) = bo_write_and_sync(&self.ops, program_bo, &program, "program") {
            let _ = self.ops.bo_free(program_bo);
            return Err(error);
        }
        self.cus[cu_index].program_cache.insert(
            program_key,
            BoundProgramEntry {
                bo: program_bo,
                address: program_address,
                bytes: program.len(),
                sha256: program_sha256,
                program: program.clone(),
                trace: trace_metadata.clone(),
            },
        );
        Ok(PreparedWaveJob {
            matrix_address,
            matrix_cache_hit,
            matrix_bytes_transferred,
            program_address,
            program_bytes: program.len(),
            program_sha256,
            program_cache_hit: false,
            encoded_program: program,
            trace: trace_metadata,
        })
    }

    fn register_read(&self, cu_index: usize, offset: u32) -> Result<u32, XrtTmatmulError> {
        let cu = &self.cus[cu_index];
        if let (Some(bar0), Some(base)) = (&self.bar0, cu.register_base) {
            return bar0.read(base, offset);
        }
        let mut value = 0;
        check_xrt(
            "xclRegRead",
            self.ops
                .xcl_reg_read(cu.ip_device, cu.ip_index, offset, &mut value),
        )?;
        Ok(value)
    }

    fn register_write(
        &self,
        cu_index: usize,
        offset: u32,
        value: u32,
    ) -> Result<(), XrtTmatmulError> {
        let cu = &self.cus[cu_index];
        if let (Some(bar0), Some(base)) = (&self.bar0, cu.register_base) {
            return bar0.write(base, offset, value);
        }
        check_xrt(
            "xclRegWrite",
            self.ops
                .xcl_reg_write(cu.ip_device, cu.ip_index, offset, value),
        )
    }

    fn run_wave(
        &mut self,
        jobs: Vec<XrtWaveJob>,
    ) -> Result<Vec<XrtWaveCompletion>, XrtTmatmulError> {
        if self.poisoned {
            return Err(XrtTmatmulError::Config(
                "persistent XRT pool is poisoned after failed quiescence".to_string(),
            ));
        }
        let mut seen_cus = HashSet::new();
        let mut seen_requests = HashSet::new();
        for job in &jobs {
            let target = self.cus.get(job.cu_index).ok_or_else(|| {
                XrtTmatmulError::Config(format!(
                    "wave job {} selects missing CU {}",
                    job.request_id, job.cu_index
                ))
            })?;
            if !seen_cus.insert(job.cu_index) {
                return Err(XrtTmatmulError::Config(format!(
                    "wave contains more than one job for CU {}",
                    job.cu_index
                )));
            }
            if !seen_requests.insert(job.request_id) {
                return Err(XrtTmatmulError::Config(format!(
                    "wave contains duplicate request id {}",
                    job.request_id
                )));
            }
            validate_wave_job(&target.target, job)?;
        }

        let mut prepared = Vec::with_capacity(jobs.len());
        for job in &jobs {
            let state = match self.prepare_wave_job(job) {
                Ok(state) => state,
                Err(error) => {
                    if self.qwen_trace.is_some() {
                        self.poisoned = true;
                    }
                    return Err(error);
                }
            };
            let cu = &self.cus[job.cu_index];
            if let Err(error) = bo_write_and_sync(&self.ops, cu.input_bo, &job.input, "input") {
                if self.qwen_trace.is_some() {
                    self.poisoned = true;
                }
                return Err(error);
            }
            prepared.push(state);
        }

        let registers = instance_registers(0)?;
        let mut launched = Vec::with_capacity(jobs.len());
        let mut dispatch_starts = Vec::with_capacity(jobs.len());
        for (job, prepared) in jobs.iter().zip(&prepared) {
            dispatch_starts.push(Instant::now());
            launched.push(job.cu_index);
            let start_result = (|| {
                self.register_write(job.cu_index, registers.reset, 0)?;
                self.register_write(job.cu_index, registers.dma_control, 1)?;
                self.register_write(
                    job.cu_index,
                    registers.dma_source_lo,
                    prepared.program_address as u32,
                )?;
                self.register_write(
                    job.cu_index,
                    registers.dma_source_hi,
                    (prepared.program_address >> 32) as u32,
                )?;
                self.register_write(
                    job.cu_index,
                    registers.dma_length,
                    prepared.program_bytes as u32,
                )
            })();
            if let Err(error) = start_result {
                return self.fail_wave(error, &launched);
            }
        }

        let deadline = Instant::now() + Duration::from_millis(u64::from(self.timeout_ms));
        let mut stalls = vec![None; jobs.len()];
        let mut dispatch_to_stall_ns = vec![None; jobs.len()];
        let mut pending = jobs.len();
        while pending != 0 {
            for (job_index, job) in jobs.iter().enumerate() {
                if stalls[job_index].is_some() {
                    continue;
                }
                match self.register_read(job.cu_index, registers.stall) {
                    Ok(0) => {}
                    Ok(value) => {
                        stalls[job_index] = Some(value);
                        dispatch_to_stall_ns[job_index] = Some(
                            u64::try_from(dispatch_starts[job_index].elapsed().as_nanos())
                                .unwrap_or(u64::MAX)
                                .max(1),
                        );
                        pending -= 1;
                    }
                    Err(error) => return self.fail_wave(error, &launched),
                }
            }
            if pending != 0 {
                if Instant::now() >= deadline {
                    return self.fail_wave(
                        XrtTmatmulError::Timeout {
                            timeout_ms: self.timeout_ms,
                        },
                        &launched,
                    );
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }

        let mut completions = Vec::with_capacity(jobs.len());
        for (job_index, (job, prepared)) in jobs.iter().zip(&prepared).enumerate() {
            let cu = &self.cus[job.cu_index];
            let mut output = vec![0u8; expected_vector_bytes(cu.target.lanes)];
            let finish_result = (|| {
                check_xrt(
                    "xrtBOSync(output, from-device)",
                    self.ops
                        .bo_sync(cu.output_bo, XRT_BO_SYNC_FROM_DEVICE, output.len(), 0),
                )?;
                check_xrt(
                    "xrtBORead(output)",
                    self.ops.bo_read(cu.output_bo, &mut output),
                )?;
                self.register_write(job.cu_index, registers.stall, 1)
            })();
            if let Err(error) = finish_result {
                return self.fail_wave(error, &launched);
            }
            completions.push(XrtWaveCompletion {
                request_id: job.request_id,
                cu_index: job.cu_index,
                stall_code: stalls[job_index].expect("all wave jobs completed"),
                output,
                dispatch_to_stall_ns: dispatch_to_stall_ns[job_index]
                    .expect("all wave jobs have dispatch timing"),
                program_bytes: prepared.program_bytes,
                matrix_key: job.matrix_key,
                matrix_sha256: job.matrix_sha256,
                matrix_address: prepared.matrix_address,
                matrix_cache_hit: prepared.matrix_cache_hit,
                matrix_bytes_transferred: prepared.matrix_bytes_transferred,
                program_address: prepared.program_address,
                program_sha256: prepared.program_sha256,
                program_cache_hit: prepared.program_cache_hit,
                encoded_program: prepared.encoded_program.clone(),
                trace_mode: prepared.trace.mode.clone(),
                model_context_limit: prepared.trace.model_context_limit,
                trace_semantic_sha256: prepared.trace.semantic_sha256,
                trace_assembly_sha256: prepared.trace.assembly_sha256,
                replay_safe_program_sha256: prepared.trace.replay_safe_program_sha256,
                trace_assembly: prepared.trace.assembly.clone(),
                trace_instructions: prepared.trace.instructions.clone(),
            });
        }
        Ok(completions)
    }

    fn fail_wave<T>(
        &mut self,
        primary: XrtTmatmulError,
        launched: &[usize],
    ) -> Result<T, XrtTmatmulError> {
        let mut cleanup_errors = Vec::new();
        let mut seen = HashSet::new();
        for &cu_index in launched {
            if seen.insert(cu_index) {
                if let Err(error) = self.quiesce_cu(cu_index) {
                    self.cus[cu_index].release_handles = false;
                    self.release_device = false;
                    cleanup_errors.push(format!("CU {cu_index}: {error}"));
                }
            }
        }
        if cleanup_errors.is_empty() {
            Err(primary)
        } else {
            self.poisoned = true;
            Err(XrtTmatmulError::Quiesce {
                primary: primary.to_string(),
                cleanup: cleanup_errors.join("; "),
            })
        }
    }

    fn quiesce_cu(&self, cu_index: usize) -> Result<(), XrtTmatmulError> {
        let registers = instance_registers(0)?;
        self.register_write(cu_index, registers.reset, 1)?;
        self.register_write(cu_index, registers.dma_control, DMACR_RESET)?;
        let timeout_ms = self.timeout_ms.max(100);
        let deadline = Instant::now() + Duration::from_millis(u64::from(timeout_ms));
        loop {
            let control = self.register_read(cu_index, registers.dma_control)?;
            if control & DMACR_RESET == 0 {
                let status = self.register_read(cu_index, registers.dma_control + MM2S_DMASR)?;
                if status & (DMASR_HALTED | DMASR_IDLE) != 0 {
                    return Ok(());
                }
            }
            if Instant::now() >= deadline {
                return Err(XrtTmatmulError::Timeout { timeout_ms });
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

impl<O: XrtOps> Drop for Pool<O> {
    fn drop(&mut self) {
        for cu in self.cus.iter_mut().rev() {
            if !cu.release_handles {
                continue;
            }
            for program in cu.program_cache.values() {
                let _ = self.ops.bo_free(program.bo);
            }
            for matrix in cu.matrix_cache.values() {
                let _ = self.ops.bo_free(matrix.bo);
            }
            for bo in [cu.output_bo, cu.input_bo] {
                if !bo.is_null() {
                    let _ = self.ops.bo_free(bo);
                }
            }
            if cu.register_base.is_none() {
                let _ = self
                    .ops
                    .xcl_close_context(cu.ip_device, &self.uuid, cu.ip_index);
            }
            if cu.register_base.is_none() && !cu.ip_device.is_null() {
                self.ops.xcl_close(cu.ip_device);
            }
        }
        if self.release_device && !self.device.is_null() {
            let _ = self.ops.device_close(self.device);
        }
    }
}

pub(crate) fn submit_xrt_tmatmul(
    request: XrtTmatmulRequest<'_>,
    output: &mut [u8],
) -> Result<XrtTmatmulStatus, XrtTmatmulError> {
    let config = XrtConfig::from_env()?;
    let xrt = RealXrt::load(config.ip_name.is_some())?;
    submit_with_ops(&xrt, &config, request, output)
}

struct Session<'a, O: XrtOps> {
    ops: &'a O,
    device: Handle,
    kernel: Handle,
    ip_device: Handle,
    ip_index: Option<u32>,
    uuid: Xuid,
    bos: Vec<Handle>,
    release_handles: bool,
}

impl<'a, O: XrtOps> Session<'a, O> {
    fn open(ops: &'a O, config: &XrtConfig) -> Result<Self, XrtTmatmulError> {
        let device = ops.device_open(config.device_index);
        if device.is_null() {
            return Err(XrtTmatmulError::NullHandle("xrtDeviceOpen"));
        }
        let mut session = Self {
            ops,
            device,
            kernel: std::ptr::null_mut(),
            ip_device: std::ptr::null_mut(),
            ip_index: None,
            uuid: [0; 16],
            bos: Vec::new(),
            release_handles: true,
        };

        let xclbin = config.xclbin.to_str().ok_or_else(|| {
            XrtTmatmulError::Config("HETGPU_XRT_XCLBIN must be valid UTF-8".to_string())
        })?;
        let xclbin = CString::new(xclbin).map_err(|_| {
            XrtTmatmulError::Config("HETGPU_XRT_XCLBIN contains a NUL byte".to_string())
        })?;
        check_xrt(
            "xrtDeviceLoadXclbinFile",
            ops.load_xclbin_file(device, &xclbin),
        )?;

        check_xrt(
            "xrtDeviceGetXclbinUUID",
            ops.get_xclbin_uuid(device, &mut session.uuid),
        )?;
        if let Some(ip_name) = config.ip_name.as_deref() {
            session.ip_device = ops.xcl_open(config.device_index);
            if session.ip_device.is_null() {
                return Err(XrtTmatmulError::NullHandle("xclOpen"));
            }
            let ip_name = CString::new(ip_name).map_err(|_| {
                XrtTmatmulError::Config("HETGPU_XRT_IP_NAME contains a NUL byte".to_string())
            })?;
            let ip_index = ops.xcl_ip_name_to_index(session.ip_device, &ip_name);
            if ip_index < 0 {
                return Err(XrtTmatmulError::Xrt {
                    operation: "xclIPName2Index",
                    code: ip_index,
                });
            }
            let ip_index = ip_index as u32;
            check_xrt(
                "xclOpenContext",
                ops.xcl_open_context(session.ip_device, &session.uuid, ip_index, false),
            )?;
            session.ip_index = Some(ip_index);
        } else {
            let kernel_name = CString::new(config.kernel_name.as_str()).map_err(|_| {
                XrtTmatmulError::Config("HETGPU_XRT_KERNEL contains a NUL byte".to_string())
            })?;
            session.kernel = ops.kernel_open_exclusive(device, &session.uuid, &kernel_name);
            if session.kernel.is_null() {
                return Err(XrtTmatmulError::NullHandle("xrtPLKernelOpenExclusive"));
            }
        }
        Ok(session)
    }

    fn allocate_bo(
        &mut self,
        size: usize,
        group: u32,
        name: &'static str,
    ) -> Result<Handle, XrtTmatmulError> {
        let bo = self.ops.bo_alloc(self.device, size, 0, group);
        if bo.is_null() {
            return Err(XrtTmatmulError::NullHandle(name));
        }
        self.bos.push(bo);
        Ok(bo)
    }

    fn bo_address(&self, bo: Handle, name: &'static str) -> Result<u64, XrtTmatmulError> {
        let address = self.ops.bo_address(bo);
        if address == 0 || address == u64::MAX {
            return Err(XrtTmatmulError::InvalidBuffer(format!(
                "xrtBOAddress returned 0x{address:016x} for {name}"
            )));
        }
        Ok(address)
    }

    fn register_read(
        &self,
        offset: u32,
        value: &mut u32,
        kernel_operation: &'static str,
        ip_operation: &'static str,
    ) -> Result<(), XrtTmatmulError> {
        if let Some(index) = self.ip_index {
            check_xrt(
                ip_operation,
                self.ops.xcl_reg_read(self.ip_device, index, offset, value),
            )
        } else {
            check_xrt(
                kernel_operation,
                self.ops.kernel_read_register(self.kernel, offset, value),
            )
        }
    }

    fn register_write(
        &self,
        offset: u32,
        value: u32,
        kernel_operation: &'static str,
        ip_operation: &'static str,
    ) -> Result<(), XrtTmatmulError> {
        if let Some(index) = self.ip_index {
            check_xrt(
                ip_operation,
                self.ops.xcl_reg_write(self.ip_device, index, offset, value),
            )
        } else {
            check_xrt(
                kernel_operation,
                self.ops.kernel_write_register(self.kernel, offset, value),
            )
        }
    }

    fn quiesce_after_launch(
        &mut self,
        registers: InstanceRegisters,
        timeout_ms: u32,
    ) -> Result<(), XrtTmatmulError> {
        // Until reset completion is observed, retaining the BO and device handles is safer
        // than allowing an independently running engine to access released storage.
        self.release_handles = false;
        self.register_write(
            registers.reset,
            1,
            "xrtKernelWriteRegister(RESET quiesce)",
            "xclRegWrite(RESET quiesce)",
        )?;
        self.register_write(
            registers.dma_control,
            DMACR_RESET,
            "xrtKernelWriteRegister(MM2S_DMACR reset)",
            "xclRegWrite(MM2S_DMACR reset)",
        )?;

        let deadline = Instant::now() + Duration::from_millis(u64::from(timeout_ms.max(100)));
        loop {
            let mut control = 0;
            self.register_read(
                registers.dma_control,
                &mut control,
                "xrtKernelReadRegister(MM2S_DMACR reset)",
                "xclRegRead(MM2S_DMACR reset)",
            )?;
            if control & DMACR_RESET == 0 {
                let mut status = 0;
                self.register_read(
                    registers.dma_control + MM2S_DMASR,
                    &mut status,
                    "xrtKernelReadRegister(MM2S_DMASR quiesce)",
                    "xclRegRead(MM2S_DMASR quiesce)",
                )?;
                if status & (DMASR_HALTED | DMASR_IDLE) != 0 {
                    self.release_handles = true;
                    return Ok(());
                }
            }
            if Instant::now() >= deadline {
                return Err(XrtTmatmulError::Timeout {
                    timeout_ms: timeout_ms.max(100),
                });
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

impl<O: XrtOps> Drop for Session<'_, O> {
    fn drop(&mut self) {
        if !self.release_handles {
            // The process/driver owns final recovery. Do not release or reuse addresses that
            // an engine may still be accessing when quiescence could not be confirmed.
            self.bos.clear();
            self.kernel = std::ptr::null_mut();
            self.ip_index = None;
            self.ip_device = std::ptr::null_mut();
            self.device = std::ptr::null_mut();
            return;
        }
        while let Some(bo) = self.bos.pop() {
            let _ = self.ops.bo_free(bo);
        }
        if !self.kernel.is_null() {
            let _ = self.ops.kernel_close(self.kernel);
        }
        if let Some(index) = self.ip_index {
            let _ = self
                .ops
                .xcl_close_context(self.ip_device, &self.uuid, index);
        }
        if !self.ip_device.is_null() {
            self.ops.xcl_close(self.ip_device);
        }
        if !self.device.is_null() {
            let _ = self.ops.device_close(self.device);
        }
    }
}

fn submit_with_ops<O: XrtOps>(
    ops: &O,
    config: &XrtConfig,
    request: XrtTmatmulRequest<'_>,
    output: &mut [u8],
) -> Result<XrtTmatmulStatus, XrtTmatmulError> {
    if request.matrix.is_empty() || request.input.is_empty() || output.is_empty() {
        return Err(XrtTmatmulError::InvalidBuffer(
            "matrix, input, and output BOs must be nonempty".to_string(),
        ));
    }

    let registers = instance_registers(config.instance)?;
    let mut session = Session::open(ops, config)?;
    let group = if let Some(group) = config.memory_group {
        group
    } else {
        let group = ops.kernel_arg_group_id(session.kernel, config.memory_arg);
        if group < 0 {
            return Err(XrtTmatmulError::Xrt {
                operation: "xrtKernelArgGroupId",
                code: group,
            });
        }
        group as u32
    };

    // This ordering is the four-BO ABI: matrix, input, output, then program.
    let matrix_bo = session.allocate_bo(request.matrix.len(), group, "xrtBOAlloc(matrix)")?;
    let input_bo = session.allocate_bo(request.input.len(), group, "xrtBOAlloc(input)")?;
    let output_bo = session.allocate_bo(output.len(), group, "xrtBOAlloc(output)")?;
    let matrix_address = session.bo_address(matrix_bo, "matrix")?;
    let input_address = session.bo_address(input_bo, "input")?;
    let output_address = session.bo_address(output_bo, "output")?;

    let labels = bind_labels(
        request.matrix_label,
        request.input_label,
        request.output_label,
        matrix_address,
        input_address,
        output_address,
    )?;
    let replay_safe_program = super::cxl_tmatmul::assemble_tmatmul_program_for_vector_registers(
        request.assembly,
        &labels,
        config.num_vector_registers,
    )
    .map_err(|error| XrtTmatmulError::Assemble(error.to_string()))?;
    let program = compact_xrt_program(&replay_safe_program)?;
    validate_program(&program)?;
    let program_bo = session.allocate_bo(program.len(), group, "xrtBOAlloc(program)")?;
    let program_address = session.bo_address(program_bo, "program")?;

    bo_write_and_sync(ops, matrix_bo, request.matrix, "matrix")?;
    bo_write_and_sync(ops, input_bo, request.input, "input")?;
    bo_write_and_sync(ops, program_bo, &program, "program")?;

    session.register_write(
        registers.reset,
        0,
        "xrtKernelWriteRegister(RESET)",
        "xclRegWrite(RESET)",
    )?;
    session.register_write(
        registers.dma_control,
        1,
        "xrtKernelWriteRegister(MM2S_DMACR)",
        "xclRegWrite(MM2S_DMACR)",
    )?;
    session.register_write(
        registers.dma_source_lo,
        program_address as u32,
        "xrtKernelWriteRegister(MM2S_SA low)",
        "xclRegWrite(MM2S_SA low)",
    )?;
    session.register_write(
        registers.dma_source_hi,
        (program_address >> 32) as u32,
        "xrtKernelWriteRegister(MM2S_SA high)",
        "xclRegWrite(MM2S_SA high)",
    )?;
    session.register_write(
        registers.dma_length,
        program.len() as u32,
        "xrtKernelWriteRegister(MM2S_LENGTH)",
        "xclRegWrite(MM2S_LENGTH)",
    )?;

    let deadline = Instant::now() + Duration::from_millis(u64::from(config.timeout_ms));
    let wait_result = (|| loop {
        let mut value = 0;
        session.register_read(
            registers.stall,
            &mut value,
            "xrtKernelReadRegister(STALL)",
            "xclRegRead(STALL)",
        )?;
        if value != 0 {
            return Ok(value);
        }
        if Instant::now() >= deadline {
            return Err(XrtTmatmulError::Timeout {
                timeout_ms: config.timeout_ms,
            });
        }
        std::thread::sleep(Duration::from_millis(1));
    })();
    let stall_code = match wait_result {
        Ok(value) => value,
        Err(primary) => {
            if let Err(cleanup) = session.quiesce_after_launch(registers, config.timeout_ms) {
                return Err(XrtTmatmulError::Quiesce {
                    primary: primary.to_string(),
                    cleanup: cleanup.to_string(),
                });
            }
            return Err(primary);
        }
    };

    check_xrt(
        "xrtBOSync(output, from-device)",
        ops.bo_sync(output_bo, XRT_BO_SYNC_FROM_DEVICE, output.len(), 0),
    )?;
    let mut completed_output = vec![0u8; output.len()];
    check_xrt(
        "xrtBORead(output)",
        ops.bo_read(output_bo, &mut completed_output),
    )?;
    session.register_write(
        registers.stall,
        1,
        "xrtKernelWriteRegister(STALL)",
        "xclRegWrite(STALL)",
    )?;
    output.copy_from_slice(&completed_output);

    Ok(XrtTmatmulStatus {
        stall_code,
        program_bytes: program.len(),
        matrix_address,
        input_address,
        output_address,
        program_address,
    })
}

fn bo_write_and_sync<O: XrtOps>(
    ops: &O,
    bo: Handle,
    bytes: &[u8],
    name: &'static str,
) -> Result<(), XrtTmatmulError> {
    check_xrt(
        match name {
            "matrix" => "xrtBOWrite(matrix)",
            "input" => "xrtBOWrite(input)",
            "program" => "xrtBOWrite(program)",
            _ => "xrtBOWrite",
        },
        ops.bo_write(bo, bytes),
    )?;
    check_xrt(
        match name {
            "matrix" => "xrtBOSync(matrix, to-device)",
            "input" => "xrtBOSync(input, to-device)",
            "program" => "xrtBOSync(program, to-device)",
            _ => "xrtBOSync(to-device)",
        },
        ops.bo_sync(bo, XRT_BO_SYNC_TO_DEVICE, bytes.len(), 0),
    )
}

fn check_xrt(operation: &'static str, code: i32) -> Result<(), XrtTmatmulError> {
    if code == 0 {
        Ok(())
    } else {
        Err(XrtTmatmulError::Xrt { operation, code })
    }
}

fn instance_registers(instance: u32) -> Result<InstanceRegisters, XrtTmatmulError> {
    let base = instance.checked_mul(AXI_INSTANCE_STRIDE).ok_or_else(|| {
        XrtTmatmulError::Config(format!(
            "instance {instance} overflows the register aperture"
        ))
    })?;
    let add = |offset: u32| {
        base.checked_add(offset).ok_or_else(|| {
            XrtTmatmulError::Config(format!("instance {instance} register offset overflow"))
        })
    };

    Ok(InstanceRegisters {
        dma_control: add(MM2S_DMACR)?,
        dma_source_lo: add(MM2S_SA)?,
        dma_source_hi: add(MM2S_SA + 4)?,
        dma_length: add(MM2S_LENGTH)?,
        stall: add(STALL)?,
        reset: add(RESET)?,
    })
}

fn validate_program(program: &[u8]) -> Result<(), XrtTmatmulError> {
    if program.is_empty() {
        return Err(XrtTmatmulError::InvalidProgram(
            "program is empty".to_string(),
        ));
    }
    if program.len() % INSTRUCTION_BYTES != 0 {
        return Err(XrtTmatmulError::InvalidProgram(format!(
            "{} bytes is not a multiple of {INSTRUCTION_BYTES}",
            program.len()
        )));
    }
    u32::try_from(program.len()).map_err(|_| {
        XrtTmatmulError::InvalidProgram("program length does not fit MM2S_LENGTH".to_string())
    })?;
    Ok(())
}

fn compact_xrt_program(replay_safe_program: &[u8]) -> Result<Vec<u8>, XrtTmatmulError> {
    validate_program(replay_safe_program)?;
    let terminal_offset = replay_safe_program.len() - INSTRUCTION_BYTES;
    let padding_offset = replay_safe_program[..terminal_offset]
        .chunks_exact(INSTRUCTION_BYTES)
        .position(|instruction| instruction.iter().all(|byte| *byte == 0))
        .map_or(terminal_offset, |slot| slot * INSTRUCTION_BYTES);
    if replay_safe_program[padding_offset..terminal_offset]
        .iter()
        .any(|byte| *byte != 0)
    {
        return Err(XrtTmatmulError::InvalidProgram(
            "replay-safe padding contains a nonzero instruction after its first zero slot"
                .to_string(),
        ));
    }

    let mut compact = Vec::with_capacity(padding_offset + INSTRUCTION_BYTES);
    compact.extend_from_slice(&replay_safe_program[..padding_offset]);
    compact.extend_from_slice(&replay_safe_program[terminal_offset..]);
    validate_program(&compact)?;
    Ok(compact)
}

fn bind_labels(
    matrix_label: &str,
    input_label: &str,
    output_label: &str,
    matrix_address: u64,
    input_address: u64,
    output_address: u64,
) -> Result<HashMap<String, u64>, XrtTmatmulError> {
    if matrix_label.is_empty() || input_label.is_empty() || output_label.is_empty() {
        return Err(XrtTmatmulError::Config(
            "BO labels must not be empty".to_string(),
        ));
    }
    if matrix_label == input_label || matrix_label == output_label || input_label == output_label {
        return Err(XrtTmatmulError::Config(
            "matrix, input, and output labels must be distinct".to_string(),
        ));
    }
    if [matrix_address, input_address, output_address]
        .into_iter()
        .any(|address| address == 0 || address == u64::MAX)
    {
        return Err(XrtTmatmulError::InvalidBuffer(
            "XRT returned an invalid BO address".to_string(),
        ));
    }

    Ok(HashMap::from([
        (matrix_label.to_owned(), matrix_address),
        (input_label.to_owned(), input_address),
        (output_label.to_owned(), output_address),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, OnceLock};

    const DEVICE_HANDLE: usize = 1;
    const KERNEL_HANDLE: usize = 2;
    const IP_DEVICE_HANDLE: usize = 3;
    const FIRST_BO_HANDLE: usize = 100;
    const AU250_BATCH_SIZE: usize = 9;
    const AU250_VECTOR_LENGTH: usize = 1024;
    const AU250_VECTOR_ELEMENTS: usize = AU250_BATCH_SIZE * AU250_VECTOR_LENGTH;
    const TEST_ASSEMBLY: &str = r#"
        ldv v0, PARAM_INPUT
        tmatmul_import v0
        tmatmul_go PARAM_MATRIX
        tmatmul_export v1
        sv v1, PARAM_OUTPUT
        stall
    "#;

    #[test]
    fn persistent_pool_state_initializes_once_and_serializes_mutation() {
        let state = OnceLock::<Mutex<Result<usize, String>>>::new();
        let initializations = AtomicUsize::new(0);

        let first = with_pool_state(
            &state,
            || {
                initializations.fetch_add(1, Ordering::SeqCst);
                Ok(0)
            },
            |value| {
                *value += 1;
                Ok(*value)
            },
        )
        .unwrap();
        let second = with_pool_state(
            &state,
            || {
                initializations.fetch_add(1, Ordering::SeqCst);
                Ok(0)
            },
            |value| {
                *value += 1;
                Ok(*value)
            },
        )
        .unwrap();

        assert_eq!((first, second), (1, 2));
        assert_eq!(initializations.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn persistent_pool_state_caches_initialization_error() {
        let state = OnceLock::<Mutex<Result<usize, String>>>::new();
        let initializations = AtomicUsize::new(0);

        for _ in 0..2 {
            let error = with_pool_state(
                &state,
                || {
                    initializations.fetch_add(1, Ordering::SeqCst);
                    Err("fixture open failed".to_string())
                },
                |_| Ok(()),
            )
            .unwrap_err();
            assert_eq!(error, "fixture open failed");
        }
        assert_eq!(initializations.load(Ordering::SeqCst), 1);
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Event {
        DeviceOpen(u32),
        LoadXclbin(String),
        GetUuid,
        KernelOpen(String),
        GroupId(i32),
        XclOpen(u32),
        IpNameToIndex(String),
        OpenContext {
            index: u32,
            shared: bool,
        },
        BoAlloc {
            bo: usize,
            size: usize,
            group: u32,
        },
        BoWrite {
            bo: usize,
            bytes: Vec<u8>,
        },
        BoSync {
            bo: usize,
            direction: i32,
            offset: usize,
            size: usize,
        },
        RegisterWrite {
            offset: u32,
            value: u32,
        },
        RegisterRead(u32),
        IpRegisterWrite {
            offset: u32,
            value: u32,
        },
        IpRegisterRead(u32),
        BoRead {
            bo: usize,
            size: usize,
        },
        BoFree(usize),
        KernelClose,
        CloseContext(u32),
        XclClose,
        DeviceClose,
    }

    struct FakeState {
        events: Vec<Event>,
        next_bo: usize,
        stall_reads: VecDeque<u32>,
        stall_reads_by_ip: HashMap<u32, VecDeque<u32>>,
        fail_register_write: Option<u32>,
        fail_register_read: Option<u32>,
        output_pattern: u8,
        bo_sizes: HashMap<usize, usize>,
    }

    struct FakeXrt {
        state: RefCell<FakeState>,
    }

    impl FakeXrt {
        fn new(stall_reads: impl IntoIterator<Item = u32>) -> Self {
            Self {
                state: RefCell::new(FakeState {
                    events: Vec::new(),
                    next_bo: FIRST_BO_HANDLE,
                    stall_reads: stall_reads.into_iter().collect(),
                    stall_reads_by_ip: HashMap::new(),
                    fail_register_write: None,
                    fail_register_read: None,
                    output_pattern: 0x5a,
                    bo_sizes: HashMap::new(),
                }),
            }
        }

        fn with_per_ip_stalls(stalls: Vec<Vec<u32>>) -> Self {
            let xrt = Self::new([]);
            xrt.state.borrow_mut().stall_reads_by_ip = stalls
                .into_iter()
                .enumerate()
                .map(|(index, values)| (index as u32, values.into_iter().collect()))
                .collect();
            xrt
        }

        fn fail_register_write(&self, offset: u32) {
            self.state.borrow_mut().fail_register_write = Some(offset);
        }

        fn fail_register_read(&self, offset: u32) {
            self.state.borrow_mut().fail_register_read = Some(offset);
        }

        fn events(&self) -> Vec<Event> {
            self.state.borrow().events.clone()
        }
    }

    impl XrtOps for FakeXrt {
        fn device_open(&self, index: u32) -> Handle {
            self.state
                .borrow_mut()
                .events
                .push(Event::DeviceOpen(index));
            DEVICE_HANDLE as Handle
        }

        fn device_close(&self, _device: Handle) -> i32 {
            self.state.borrow_mut().events.push(Event::DeviceClose);
            0
        }

        fn load_xclbin_file(&self, _device: Handle, path: &CStr) -> i32 {
            self.state
                .borrow_mut()
                .events
                .push(Event::LoadXclbin(path.to_string_lossy().into_owned()));
            0
        }

        fn get_xclbin_uuid(&self, _device: Handle, uuid: &mut Xuid) -> i32 {
            *uuid = [0x2a; 16];
            self.state.borrow_mut().events.push(Event::GetUuid);
            0
        }

        fn kernel_open_exclusive(&self, _device: Handle, _uuid: &Xuid, name: &CStr) -> Handle {
            self.state
                .borrow_mut()
                .events
                .push(Event::KernelOpen(name.to_string_lossy().into_owned()));
            KERNEL_HANDLE as Handle
        }

        fn kernel_close(&self, _kernel: Handle) -> i32 {
            self.state.borrow_mut().events.push(Event::KernelClose);
            0
        }

        fn kernel_arg_group_id(&self, _kernel: Handle, arg: i32) -> i32 {
            self.state.borrow_mut().events.push(Event::GroupId(arg));
            7
        }

        fn kernel_read_register(&self, _kernel: Handle, offset: u32, value: &mut u32) -> i32 {
            let mut state = self.state.borrow_mut();
            state.events.push(Event::RegisterRead(offset));
            if state.fail_register_read == Some(offset) {
                return -5;
            }
            *value = match offset {
                STALL => state.stall_reads.pop_front().unwrap_or(0),
                MM2S_DMACR => 0,
                MM2S_DMASR => 1,
                _ => 0,
            };
            0
        }

        fn kernel_write_register(&self, _kernel: Handle, offset: u32, value: u32) -> i32 {
            let mut state = self.state.borrow_mut();
            state.events.push(Event::RegisterWrite { offset, value });
            if state.fail_register_write == Some(offset) {
                -5
            } else {
                0
            }
        }

        fn xcl_open(&self, index: u32) -> Handle {
            self.state.borrow_mut().events.push(Event::XclOpen(index));
            IP_DEVICE_HANDLE as Handle
        }

        fn xcl_close(&self, _device: Handle) {
            self.state.borrow_mut().events.push(Event::XclClose);
        }

        fn xcl_ip_name_to_index(&self, _device: Handle, name: &CStr) -> i32 {
            let name = name.to_string_lossy().into_owned();
            self.state
                .borrow_mut()
                .events
                .push(Event::IpNameToIndex(name.clone()));
            if name.ends_with("big_1") {
                0
            } else if name.ends_with("big_2") {
                1
            } else if name.ends_with("big_3") {
                2
            } else if name.ends_with("small_1") {
                3
            } else {
                -1
            }
        }

        fn xcl_open_context(&self, _device: Handle, _uuid: &Xuid, index: u32, shared: bool) -> i32 {
            self.state
                .borrow_mut()
                .events
                .push(Event::OpenContext { index, shared });
            0
        }

        fn xcl_close_context(&self, _device: Handle, _uuid: &Xuid, index: u32) -> i32 {
            self.state
                .borrow_mut()
                .events
                .push(Event::CloseContext(index));
            0
        }

        fn xcl_reg_read(&self, _device: Handle, index: u32, offset: u32, value: &mut u32) -> i32 {
            let mut state = self.state.borrow_mut();
            state.events.push(Event::IpRegisterRead(offset));
            if state.fail_register_read == Some(offset) {
                return -5;
            }
            *value = match offset {
                STALL => {
                    if let Some(reads) = state.stall_reads_by_ip.get_mut(&index) {
                        reads.pop_front().unwrap_or(0)
                    } else {
                        state.stall_reads.pop_front().unwrap_or(0)
                    }
                }
                MM2S_DMACR => 0,
                MM2S_DMASR => 1,
                _ => 0,
            };
            0
        }

        fn xcl_reg_write(&self, _device: Handle, _index: u32, offset: u32, value: u32) -> i32 {
            let mut state = self.state.borrow_mut();
            state.events.push(Event::IpRegisterWrite { offset, value });
            if state.fail_register_write == Some(offset) {
                -5
            } else {
                0
            }
        }

        fn bo_alloc(&self, _device: Handle, size: usize, _flags: u64, group: u32) -> Handle {
            let mut state = self.state.borrow_mut();
            let bo = state.next_bo;
            state.next_bo += 1;
            state.bo_sizes.insert(bo, size);
            state.events.push(Event::BoAlloc { bo, size, group });
            bo as Handle
        }

        fn bo_free(&self, bo: Handle) -> i32 {
            self.state
                .borrow_mut()
                .events
                .push(Event::BoFree(bo as usize));
            0
        }

        fn bo_address(&self, bo: Handle) -> u64 {
            0x1000 * (bo as usize - FIRST_BO_HANDLE + 1) as u64
        }

        fn bo_write_range(&self, bo: Handle, bytes: &[u8], offset: usize) -> i32 {
            let mut state = self.state.borrow_mut();
            if offset
                .checked_add(bytes.len())
                .is_none_or(|end| end > state.bo_sizes[&(bo as usize)])
            {
                return -1;
            }
            state.events.push(Event::BoWrite {
                bo: bo as usize,
                bytes: bytes.to_vec(),
            });
            0
        }

        fn bo_read_range(&self, bo: Handle, bytes: &mut [u8], offset: usize) -> i32 {
            let mut state = self.state.borrow_mut();
            if offset
                .checked_add(bytes.len())
                .is_none_or(|end| end > state.bo_sizes[&(bo as usize)])
            {
                return -1;
            }
            bytes.fill(state.output_pattern);
            state.events.push(Event::BoRead {
                bo: bo as usize,
                size: bytes.len(),
            });
            0
        }

        fn bo_sync(&self, bo: Handle, direction: i32, size: usize, offset: usize) -> i32 {
            self.state.borrow_mut().events.push(Event::BoSync {
                bo: bo as usize,
                direction,
                offset,
                size,
            });
            0
        }
    }

    fn test_config(timeout_ms: u32) -> XrtConfig {
        XrtConfig {
            xclbin: PathBuf::from("/tmp/ternary_matmul.xclbin"),
            device_index: 0,
            kernel_name: "ternip_ip".to_string(),
            ip_name: None,
            instance: 0,
            memory_arg: 14,
            memory_group: None,
            num_vector_registers: 8,
            timeout_ms,
        }
    }

    fn test_request() -> XrtTmatmulRequest<'static> {
        XrtTmatmulRequest {
            assembly: TEST_ASSEMBLY,
            matrix_label: "PARAM_MATRIX",
            input_label: "PARAM_INPUT",
            output_label: "PARAM_OUTPUT",
            matrix: &[1, 2, 3, 4],
            input: &[5, 6, 7, 8],
        }
    }

    fn native_ip_config(timeout_ms: u32) -> XrtConfig {
        XrtConfig {
            ip_name: Some("ternip_big:ternip_big_1".to_string()),
            memory_group: Some(0),
            num_vector_registers: 4,
            ..test_config(timeout_ms)
        }
    }

    fn test_wave_jobs() -> Vec<XrtWaveJob> {
        XrtPoolConfig::maxcores_targets()
            .into_iter()
            .enumerate()
            .map(|(cu_index, target)| {
                let matrix: Arc<[u8]> = Arc::from(vec![0_u8; AU250_MATRIX_BYTES]);
                XrtWaveJob {
                    request_id: 10 + cu_index as u64,
                    cu_index,
                    matrix_key: [0x55; 32],
                    matrix_sha256: Sha256::digest(&matrix).into(),
                    matrix,
                    input: vec![0_u8; expected_vector_bytes(target.lanes)],
                }
            })
            .collect()
    }

    fn pool_test_config() -> XrtPoolConfig {
        XrtPoolConfig {
            xclbin: PathBuf::from("/tmp/MaxCores_370M.xclbin"),
            device_index: 0,
            targets: XrtPoolConfig::maxcores_targets(),
            num_vector_registers: 4,
            timeout_ms: 20,
            qwen_trace: None,
            resident_matrix_cache_bytes: 2 * AU250_MATRIX_BYTES,
            bar0_resource: None,
        }
    }

    fn qwen_pool_test_config(mode: &str) -> XrtPoolConfig {
        XrtPoolConfig {
            qwen_trace: Some(QwenTraceConfig {
                mode: mode.to_string(),
                model_context_limit: crate::r#impl::iq1s_trace::QWEN_MODEL_CONTEXT_LIMIT,
            }),
            ..pool_test_config()
        }
    }

    #[test]
    fn strict_qwen_rejects_unpinned_bar0_resource() {
        let mut config = qwen_pool_test_config("compiler");
        config.bar0_resource = Some(PathBuf::from("/tmp/resource0"));
        assert!(matches!(
            config.validate(),
            Err(XrtTmatmulError::Config(message))
                if message.contains("strict Qwen IQ1_S BAR0 access requires")
        ));

        config.bar0_resource = Some(PathBuf::from("/sys/bus/pci/devices/0000:64:00.1/resource0"));
        config.validate().unwrap();
    }

    #[test]
    fn maxcores_targets_match_xclbin_layout() {
        assert_eq!(
            XrtPoolConfig::maxcores_targets(),
            vec![
                XrtCuTarget {
                    ip_name: "ternip_big:ternip_big_1".into(),
                    memory_group: 0,
                    lanes: 9,
                },
                XrtCuTarget {
                    ip_name: "ternip_big:ternip_big_2".into(),
                    memory_group: 3,
                    lanes: 9,
                },
                XrtCuTarget {
                    ip_name: "ternip_big:ternip_big_3".into(),
                    memory_group: 2,
                    lanes: 9,
                },
                XrtCuTarget {
                    ip_name: "ternip_small:ternip_small_1".into(),
                    memory_group: 1,
                    lanes: 6,
                },
            ]
        );
    }

    #[test]
    fn cu_table_override_is_one_versioned_atomic_value() {
        let json = r#"{"version":1,"cus":[{"ip_name":"ternip_big:ternip_big_1","memory_group":0,"lanes":9}]}"#;
        let parsed = XrtPoolConfig::parse_cu_table(json).unwrap();
        assert_eq!(parsed.len(), 1);
        assert!(XrtPoolConfig::parse_cu_table(r#"{"version":2,"cus":[]}"#).is_err());
        assert!(XrtPoolConfig::parse_cu_table(r#"{"version":1,"cus":[]}"#).is_err());
    }

    #[test]
    fn qwen_trace_mode_requires_the_exact_four_cu_maxcores_topology() {
        let exact = qwen_pool_test_config("compiler");
        exact.validate().unwrap();

        let mut missing = exact.clone();
        missing.targets.pop();
        assert!(missing
            .validate()
            .unwrap_err()
            .to_string()
            .contains("exact four-CU"));

        let mut reordered = exact;
        reordered.targets.swap(0, 1);
        assert!(reordered
            .validate()
            .unwrap_err()
            .to_string()
            .contains("exact four-CU"));
    }

    #[test]
    fn qwen_pool_program_bo_is_the_cross_checked_compiler_trace() {
        let xrt = FakeXrt::new([1, 1, 1, 1]);
        let mut pool = Pool::open_with_ops(xrt, qwen_pool_test_config("compiler")).unwrap();
        let job = test_wave_jobs().remove(0);
        pool.run_wave(vec![job]).unwrap();
        let labels = bind_labels(
            "PARAM_MATRIX",
            "PARAM_INPUT",
            "PARAM_OUTPUT",
            0x9000,
            0x1000,
            0x2000,
        )
        .unwrap();
        let expected = crate::r#impl::iq1s_trace::build_selected_trace(
            "compiler",
            crate::r#impl::iq1s_trace::QWEN_MODEL_CONTEXT_LIMIT,
            &labels,
            4,
        )
        .unwrap();
        let expected = compact_xrt_program(&expected.program).unwrap();
        let actual = pool
            .ops
            .events()
            .iter()
            .find_map(|event| match event {
                Event::BoWrite { bo: 109, bytes } => Some(bytes.clone()),
                _ => None,
            })
            .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(actual.len(), 96);
    }

    #[test]
    fn pool_loads_xclbin_once_and_allocates_two_reusable_bos_per_cu() {
        let xrt = FakeXrt::new([1, 1, 1, 1]);
        let pool = Pool::open_with_ops(xrt, pool_test_config()).unwrap();
        assert_eq!(
            pool.ops
                .events()
                .iter()
                .filter(|event| matches!(event, Event::LoadXclbin(_)))
                .count(),
            1
        );
        assert_eq!(
            pool.ops
                .events()
                .iter()
                .filter(|event| matches!(event, Event::BoAlloc { .. }))
                .count(),
            8
        );
    }

    #[test]
    fn qwen_resident_matrix_and_bound_program_are_reused_in_the_same_bank() {
        let xrt = FakeXrt::new([1, 1]);
        let mut pool = Pool::open_with_ops(xrt, qwen_pool_test_config("compiler")).unwrap();
        let job = test_wave_jobs().remove(0);

        pool.run_wave(vec![job.clone()]).unwrap();
        pool.run_wave(vec![job]).unwrap();

        let matrix_writes = pool
            .ops
            .events()
            .iter()
            .filter(|event| matches!(event, Event::BoWrite { bytes, .. } if bytes.len() == AU250_MATRIX_BYTES))
            .count();
        let program_writes = pool
            .ops
            .events()
            .iter()
            .filter(|event| matches!(event, Event::BoWrite { bytes, .. } if bytes.len() == 96))
            .count();

        assert_eq!(matrix_writes, 1);
        assert_eq!(program_writes, 1);
    }

    #[test]
    fn resident_matrix_is_bank_local_even_for_the_same_key() {
        let xrt = FakeXrt::new([1, 1]);
        let mut pool = Pool::open_with_ops(xrt, qwen_pool_test_config("handwritten")).unwrap();
        let mut jobs = test_wave_jobs();
        let first = jobs.remove(0);
        let mut second = jobs.remove(0);
        second.matrix_key = first.matrix_key;
        second.matrix_sha256 = first.matrix_sha256;
        second.matrix = first.matrix.clone();

        pool.run_wave(vec![first]).unwrap();
        pool.run_wave(vec![second]).unwrap();

        let matrix_writes = pool
            .ops
            .events()
            .iter()
            .filter(|event| matches!(event, Event::BoWrite { bytes, .. } if bytes.len() == AU250_MATRIX_BYTES))
            .count();
        assert_eq!(matrix_writes, 2);
        assert_eq!(pool.cus[0].matrix_cache.len(), 1);
        assert_eq!(pool.cus[1].matrix_cache.len(), 1);
    }

    #[test]
    fn handwritten_and_compiler_programs_share_one_resident_weight_cache() {
        let xrt = FakeXrt::new([1, 1]);
        let mut pool = Pool::open_with_ops(xrt, qwen_pool_test_config("compiler")).unwrap();
        let job = test_wave_jobs().remove(0);
        let compiler = pool.run_wave(vec![job.clone()]).unwrap().remove(0);
        pool.qwen_trace.as_mut().unwrap().mode = "handwritten".to_string();
        let handwritten = pool.run_wave(vec![job]).unwrap().remove(0);

        assert!(!compiler.matrix_cache_hit);
        assert!(!compiler.program_cache_hit);
        assert!(handwritten.matrix_cache_hit);
        assert!(!handwritten.program_cache_hit);
        assert_eq!(compiler.matrix_address, handwritten.matrix_address);
        assert_eq!(compiler.program_sha256, handwritten.program_sha256);
        assert_ne!(compiler.program_address, handwritten.program_address);
        assert_eq!(pool.cus[0].matrix_cache.len(), 1);
        assert_eq!(pool.cus[0].program_cache.len(), 2);
    }

    #[test]
    fn resident_matrix_lru_evicts_bound_program_with_the_matrix() {
        let xrt = FakeXrt::new([1, 1, 1]);
        let mut config = qwen_pool_test_config("compiler");
        config.resident_matrix_cache_bytes = AU250_MATRIX_BYTES;
        let mut pool = Pool::open_with_ops(xrt, config).unwrap();
        let first = test_wave_jobs().remove(0);
        let mut second = first.clone();
        second.request_id += 100;
        second.matrix_key = [0x66; 32];

        pool.run_wave(vec![first.clone()]).unwrap();
        pool.run_wave(vec![second]).unwrap();
        pool.run_wave(vec![first]).unwrap();

        let events = pool.ops.events();
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::BoWrite { bytes, .. } if bytes.len() == AU250_MATRIX_BYTES))
                .count(),
            3
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::BoWrite { bytes, .. } if bytes.len() == 96))
                .count(),
            3
        );
        assert!(
            events
                .iter()
                .filter(|event| matches!(event, Event::BoFree(_)))
                .count()
                >= 4
        );
    }

    #[test]
    fn changed_content_under_a_resident_key_poisons_strict_qwen() {
        let xrt = FakeXrt::new([1]);
        let mut pool = Pool::open_with_ops(xrt, qwen_pool_test_config("compiler")).unwrap();
        let first = test_wave_jobs().remove(0);
        pool.run_wave(vec![first.clone()]).unwrap();

        let mut changed = first;
        changed.request_id += 100;
        let mut bytes = changed.matrix.to_vec();
        bytes[0] = 1;
        changed.matrix = Arc::from(bytes);
        changed.matrix_sha256 = Sha256::digest(&changed.matrix).into();
        let error = pool.run_wave(vec![changed]).unwrap_err();
        assert!(error.to_string().contains("changed content"), "{error}");
        assert!(pool.poisoned);
    }

    #[test]
    fn wave_returns_request_order_after_out_of_order_stalls() {
        let xrt =
            FakeXrt::with_per_ip_stalls(vec![vec![0, 0, 1], vec![1], vec![0, 0, 0, 1], vec![0, 1]]);
        let mut pool = Pool::open_with_ops(xrt, pool_test_config()).unwrap();
        let completions = pool.run_wave(test_wave_jobs()).unwrap();
        assert_eq!(
            completions
                .iter()
                .map(|completion| completion.request_id)
                .collect::<Vec<_>>(),
            vec![10, 11, 12, 13]
        );
        assert!(completions
            .iter()
            .all(|completion| completion.dispatch_to_stall_ns > 0));
        assert!(completions
            .iter()
            .all(|completion| completion.program_bytes == 96));
    }

    #[test]
    fn wave_rejects_wrong_payload_sizes_before_register_writes() {
        let xrt = FakeXrt::new([1]);
        let mut pool = Pool::open_with_ops(xrt, pool_test_config()).unwrap();
        let mut jobs = test_wave_jobs();
        jobs.truncate(1);
        jobs[0].input.pop();

        let error = pool.run_wave(jobs).unwrap_err();
        assert!(error.to_string().contains("input has"), "{error}");
        assert!(!pool.ops.events().iter().any(|event| matches!(
            event,
            Event::IpRegisterWrite { .. } | Event::IpRegisterRead(_)
        )));
    }

    #[test]
    fn bar0_pool_uses_exact_cu_base_without_legacy_contexts() {
        use std::os::unix::fs::FileExt;

        let resource = tempfile::NamedTempFile::new().unwrap();
        resource.as_file().set_len(32 * 1024 * 1024).unwrap();
        let xrt = FakeXrt::new([1]);
        let mut config = pool_test_config();
        config.bar0_resource = Some(resource.path().to_path_buf());
        let pool = Pool::open_with_ops(xrt, config).unwrap();

        assert!(!pool.ops.events().iter().any(|event| matches!(
            event,
            Event::XclOpen(_) | Event::OpenContext { .. } | Event::IpRegisterWrite { .. }
        )));
        for base in [0x00c1_0000, 0x0181_0000, 0x0141_0000, 0x0101_0000] {
            pool.bar0.as_ref().unwrap().write(base, STALL, 1).unwrap();
            assert_eq!(pool.bar0.as_ref().unwrap().read(base, STALL).unwrap(), 1);
            let mut stall = [0u8; 4];
            resource
                .as_file()
                .read_exact_at(&mut stall, base + u64::from(STALL))
                .unwrap();
            assert_eq!(u32::from_ne_bytes(stall), 1);
        }
    }

    #[test]
    fn failed_quiescence_poisons_pool_and_retains_uncertain_cu() {
        let xrt = FakeXrt::new([1]);
        let mut pool = Pool::open_with_ops(xrt, pool_test_config()).unwrap();
        pool.ops.fail_register_write(RESET);
        let mut jobs = test_wave_jobs();
        jobs.truncate(1);

        let error = pool.run_wave(jobs.clone()).unwrap_err();
        assert!(matches!(error, XrtTmatmulError::Quiesce { .. }));
        let retry = pool.run_wave(jobs).unwrap_err();
        assert!(retry.to_string().contains("poisoned"), "{retry}");
        assert!(!pool.cus[0].release_handles);
        assert!(!pool.release_device);
    }

    fn au250_vector_add_fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut vector_a = Vec::with_capacity(AU250_VECTOR_ELEMENTS * 2);
        let mut vector_b = Vec::with_capacity(AU250_VECTOR_ELEMENTS * 2);
        let mut expected = Vec::with_capacity(AU250_VECTOR_ELEMENTS * 2);

        for index in 0..AU250_VECTOR_ELEMENTS {
            let a_raw = ((index % 8) as i16) * 32;
            let b_raw = 48i16;
            vector_a.extend_from_slice(&a_raw.to_le_bytes());
            vector_b.extend_from_slice(&b_raw.to_le_bytes());
            expected.extend_from_slice(&(a_raw + b_raw).to_le_bytes());
        }
        (vector_a, vector_b, expected)
    }

    fn au250_tmatmul_fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut matrix = vec![0u8; AU250_VECTOR_LENGTH * AU250_VECTOR_LENGTH / 4];
        matrix[..2 * AU250_VECTOR_LENGTH / 4].fill(0x55);

        let mut input = vec![0u8; AU250_VECTOR_ELEMENTS * 2];
        let mut expected = vec![0u8; AU250_VECTOR_ELEMENTS * 2];
        for lane in 0..AU250_BATCH_SIZE {
            for column in [lane, lane + 1] {
                let offset = (column * AU250_BATCH_SIZE + lane) * 2;
                input[offset..offset + 2].copy_from_slice(&32i16.to_le_bytes());
            }
            for row in 0..2 {
                let offset = (row * AU250_BATCH_SIZE + lane) * 2;
                expected[offset..offset + 2].copy_from_slice(&64i16.to_le_bytes());
            }
        }
        (matrix, input, expected)
    }

    #[test]
    fn instance_registers_match_ternary_matmul_contract() {
        assert_eq!(
            instance_registers(0).unwrap(),
            InstanceRegisters {
                dma_control: 0x0000,
                dma_source_lo: 0x0018,
                dma_source_hi: 0x001c,
                dma_length: 0x0028,
                stall: 0x1000,
                reset: 0x2000,
            }
        );
        assert_eq!(instance_registers(2).unwrap().stall, 0x9000);
        assert_eq!(instance_registers(2).unwrap().reset, 0xa000);
    }

    #[test]
    fn program_requires_complete_128_bit_words() {
        assert!(validate_program(&[0; 16]).is_ok());
        assert!(matches!(
            validate_program(&[]),
            Err(XrtTmatmulError::InvalidProgram(_))
        ));
        assert!(matches!(
            validate_program(&[0; 17]),
            Err(XrtTmatmulError::InvalidProgram(_))
        ));
    }

    #[test]
    fn labels_bind_to_real_bo_addresses() {
        let labels = bind_labels("PARAM_0", "PARAM_1", "PARAM_2", 0x1000, 0x2000, 0x3000).unwrap();
        assert_eq!(labels["PARAM_0"], 0x1000);
        assert_eq!(labels["PARAM_1"], 0x2000);
        assert_eq!(labels["PARAM_2"], 0x3000);
    }

    #[test]
    fn duplicate_labels_are_rejected() {
        let error = bind_labels("PARAM_0", "PARAM_0", "PARAM_2", 1, 2, 3).unwrap_err();
        assert!(error.to_string().contains("distinct"));
    }

    #[test]
    fn config_uses_repo_compatible_defaults() {
        let config = XrtConfig::from_lookup(|name| match name {
            "HETGPU_XRT_XCLBIN" => Some("/tmp/kernel.xclbin".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(config.xclbin, PathBuf::from("/tmp/kernel.xclbin"));
        assert_eq!(config.device_index, 0);
        assert_eq!(config.kernel_name, "ternip_ip");
        assert_eq!(config.ip_name, None);
        assert_eq!(config.instance, 0);
        assert_eq!(config.memory_arg, 14);
        assert_eq!(config.memory_group, None);
        assert_eq!(config.num_vector_registers, 8);
        assert_eq!(config.timeout_ms, 10_000);
    }

    #[test]
    fn config_selects_instance_memory_pointer_arg() {
        let config = XrtConfig::from_lookup(|name| match name {
            "HETGPU_XRT_XCLBIN" => Some("kernel.xclbin".into()),
            "HETGPU_XRT_INSTANCE" => Some("2".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(config.memory_arg, 16);
    }

    #[test]
    fn config_selects_native_ip_and_explicit_memory_group() {
        let config = XrtConfig::from_lookup(|name| match name {
            "HETGPU_XRT_XCLBIN" => Some("kernel.xclbin".into()),
            "HETGPU_XRT_IP_NAME" => Some("ternip_big:ternip_big_1".into()),
            "HETGPU_XRT_MEMORY_GROUP" => Some("0".into()),
            "HETGPU_XRT_NUM_VECTOR_REGISTERS" => Some("4".into()),
            _ => None,
        })
        .unwrap();

        assert_eq!(config.ip_name.as_deref(), Some("ternip_big:ternip_big_1"));
        assert_eq!(config.memory_group, Some(0));
        assert_eq!(config.num_vector_registers, 4);
    }

    #[test]
    fn native_ip_requires_explicit_memory_group() {
        let error = XrtConfig::from_lookup(|name| match name {
            "HETGPU_XRT_XCLBIN" => Some("kernel.xclbin".into()),
            "HETGPU_XRT_IP_NAME" => Some("ternip_big:ternip_big_1".into()),
            _ => None,
        })
        .unwrap_err();

        assert!(error.to_string().contains("HETGPU_XRT_MEMORY_GROUP"));
    }

    #[test]
    fn xclbin_is_required() {
        let error = XrtConfig::from_lookup(|_| None).unwrap_err();
        assert!(error.to_string().contains("HETGPU_XRT_XCLBIN"));
    }

    #[test]
    fn au250_vector_add_fixture_matches_known_good_example() {
        let (vector_a, vector_b, expected) = au250_vector_add_fixture();

        assert_eq!(vector_a.len(), 9 * 1024 * 2);
        assert_eq!(vector_b.len(), vector_a.len());
        assert_eq!(expected.len(), vector_a.len());

        let first_a: Vec<i16> = vector_a[..16]
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        let first_b: Vec<i16> = vector_b[..16]
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        let first_expected: Vec<i16> = expected[..16]
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes(bytes.try_into().unwrap()))
            .collect();

        assert_eq!(first_a, vec![0, 32, 64, 96, 128, 160, 192, 224]);
        assert_eq!(first_b, vec![48; 8]);
        assert_eq!(first_expected, vec![48, 80, 112, 144, 176, 208, 240, 272]);
    }

    #[test]
    fn xrt_compaction_matches_known_good_five_instruction_image() {
        let labels = HashMap::from([
            ("PARAM_MATRIX".to_string(), 0x1000),
            ("PARAM_INPUT".to_string(), 0x2000),
            ("PARAM_OUTPUT".to_string(), 0x3000),
        ]);
        let replay_safe = super::super::cxl_tmatmul::assemble_tmatmul_program_for_vector_registers(
            r#"
                ldv v0, PARAM_MATRIX
                ldv v1, PARAM_INPUT
                add v2, v0, v1
                sv v2, PARAM_OUTPUT
                stall
            "#,
            &labels,
            4,
        )
        .unwrap();

        let compact = compact_xrt_program(&replay_safe).unwrap();
        let expected = [
            [
                0x00, 0x10, 0, 0, 0, 0, 0, 0, 0x20, 0x00, 0x02, 0, 0, 0, 0, 0,
            ],
            [
                0x00, 0x20, 0, 0, 0, 0, 0, 0, 0xa0, 0x0a, 0x02, 0, 0, 0, 0, 0,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0x00, 0x23, 0x04, 0, 0, 0, 0, 0],
            [
                0x00, 0x30, 0, 0, 0, 0, 0, 0, 0x40, 0x15, 0x02, 0, 0, 0, 0, 0,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0x00, 0x00, 0x0a, 0, 0, 0, 0, 0],
        ]
        .concat();

        assert_eq!(compact, expected);
        assert_eq!(compact.len(), 5 * INSTRUCTION_BYTES);
    }

    #[test]
    fn au250_tmatmul_fixture_matches_repository_canonical_case() {
        let (matrix, input, expected) = au250_tmatmul_fixture();

        assert_eq!(matrix.len(), 1024 * 1024 / 4);
        assert!(matrix[..512].iter().all(|byte| *byte == 0x55));
        assert!(matrix[512..].iter().all(|byte| *byte == 0));
        let input_raw: Vec<i16> = input
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        let expected_raw: Vec<i16> = expected
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(input_raw.iter().filter(|value| **value == 32).count(), 18);
        for lane in 0..9 {
            assert_eq!(input_raw[lane * AU250_BATCH_SIZE + lane], 32);
            assert_eq!(input_raw[(lane + 1) * AU250_BATCH_SIZE + lane], 32);
            assert_eq!(expected_raw[lane], 64);
            assert_eq!(expected_raw[AU250_BATCH_SIZE + lane], 64);
        }
        assert!(expected_raw[2 * AU250_BATCH_SIZE..]
            .iter()
            .all(|value| *value == 0));
    }

    #[test]
    fn tmatmul_image_matches_existing_xcu250_assembler() {
        let labels = HashMap::from([
            ("PARAM_MATRIX".to_string(), 0x1000),
            ("PARAM_INPUT".to_string(), 0x2000),
            ("PARAM_OUTPUT".to_string(), 0x3000),
        ]);
        let replay_safe = super::super::cxl_tmatmul::assemble_tmatmul_program_for_vector_registers(
            TEST_ASSEMBLY,
            &labels,
            4,
        )
        .unwrap();
        let compact = compact_xrt_program(&replay_safe).unwrap();
        let expected = [
            [
                0x00, 0x20, 0, 0, 0, 0, 0, 0, 0x20, 0x00, 0x02, 0, 0, 0, 0, 0,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0x08, 0x00, 0x06, 0, 0, 0, 0, 0],
            [
                0x00, 0x10, 0, 0, 0, 0, 0, 0, 0x10, 0x00, 0x06, 0, 0, 0, 0, 0,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0x98, 0x0a, 0x06, 0, 0, 0, 0, 0],
            [
                0x00, 0x30, 0, 0, 0, 0, 0, 0, 0xc0, 0x0a, 0x02, 0, 0, 0, 0, 0,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0x00, 0x00, 0x0a, 0, 0, 0, 0, 0],
        ]
        .concat();

        assert_eq!(compact, expected);
        assert_eq!(compact.len(), 6 * INSTRUCTION_BYTES);
    }

    #[test]
    #[ignore = "requires the app215 XRT legacy xcl register API used by AU250"]
    fn installed_xrt_symbols_resolve() {
        let api = RealXrt::load(true).unwrap();
        assert!(!api.library.is_null());
        assert!(!api.native_ip.as_ref().unwrap().library.is_null());
    }

    #[test]
    fn submit_uses_four_bo_abi_and_embeds_real_addresses() {
        let xrt = FakeXrt::new([0, 1]);
        let mut output = [0u8; 8];

        let status = submit_with_ops(&xrt, &test_config(20), test_request(), &mut output).unwrap();

        assert_eq!(output, [0x5a; 8]);
        assert_eq!(status.program_bytes, 96);
        assert_eq!(status.matrix_address, 0x1000);
        assert_eq!(status.input_address, 0x2000);
        assert_eq!(status.output_address, 0x3000);
        assert_eq!(status.program_address, 0x4000);
        assert_eq!(status.stall_code, 1);

        let events = xrt.events();
        let allocations: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Event::BoAlloc { bo, size, group } => Some((*bo, *size, *group)),
                _ => None,
            })
            .collect();
        assert_eq!(
            allocations,
            vec![(100, 4, 7), (101, 4, 7), (102, 8, 7), (103, 96, 7)]
        );

        let program = events
            .iter()
            .find_map(|event| match event {
                Event::BoWrite { bo: 103, bytes } => Some(bytes),
                _ => None,
            })
            .unwrap();
        assert_eq!(program.len(), 96);
        assert_eq!(
            u64::from_le_bytes(program[0..8].try_into().unwrap()),
            0x2000
        );
        assert_eq!(
            u64::from_le_bytes(program[32..40].try_into().unwrap()),
            0x1000
        );
        assert_eq!(
            u64::from_le_bytes(program[64..72].try_into().unwrap()),
            0x3000
        );

        let syncs: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Event::BoSync {
                    bo,
                    direction,
                    offset: _,
                    size,
                } => Some((*bo, *direction, *size)),
                _ => None,
            })
            .collect();
        assert_eq!(
            syncs,
            vec![
                (100, XRT_BO_SYNC_TO_DEVICE, 4),
                (101, XRT_BO_SYNC_TO_DEVICE, 4),
                (103, XRT_BO_SYNC_TO_DEVICE, 96),
                (102, XRT_BO_SYNC_FROM_DEVICE, 8),
            ]
        );

        let writes: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Event::RegisterWrite { offset, value } => Some((*offset, *value)),
                _ => None,
            })
            .collect();
        assert_eq!(
            writes,
            vec![
                (0x2000, 0),
                (0x0000, 1),
                (0x0018, 0x4000),
                (0x001c, 0),
                (0x0028, 96),
                (0x1000, 1)
            ]
        );
        assert_eq!(
            &events[events.len() - 6..],
            &[
                Event::BoFree(103),
                Event::BoFree(102),
                Event::BoFree(101),
                Event::BoFree(100),
                Event::KernelClose,
                Event::DeviceClose,
            ]
        );
    }

    #[test]
    fn native_ip_submit_uses_xcl_registers_and_explicit_four_bo_group() {
        let xrt = FakeXrt::new([0, 1]);
        let mut output = [0u8; 8];

        let status =
            submit_with_ops(&xrt, &native_ip_config(20), test_request(), &mut output).unwrap();

        assert_eq!(status.stall_code, 1);
        assert_eq!(output, [0x5a; 8]);
        let events = xrt.events();
        assert!(events.contains(&Event::XclOpen(0)));
        assert!(events.contains(&Event::IpNameToIndex("ternip_big:ternip_big_1".to_string())));
        assert!(events.contains(&Event::OpenContext {
            index: 0,
            shared: false,
        }));
        assert!(!events.iter().any(|event| matches!(
            event,
            Event::KernelOpen(_) | Event::GroupId(_) | Event::KernelClose
        )));

        let allocations: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Event::BoAlloc { bo, size, group } => Some((*bo, *size, *group)),
                _ => None,
            })
            .collect();
        assert_eq!(
            allocations,
            vec![(100, 4, 0), (101, 4, 0), (102, 8, 0), (103, 96, 0)]
        );

        let writes: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Event::IpRegisterWrite { offset, value } => Some((*offset, *value)),
                _ => None,
            })
            .collect();
        assert_eq!(
            writes,
            vec![
                (0x2000, 0),
                (0x0000, 1),
                (0x0018, 0x4000),
                (0x001c, 0),
                (0x0028, 96),
                (0x1000, 1),
            ]
        );
        assert!(events.contains(&Event::IpRegisterRead(0x1000)));
        assert_eq!(
            &events[events.len() - 7..],
            &[
                Event::BoFree(103),
                Event::BoFree(102),
                Event::BoFree(101),
                Event::BoFree(100),
                Event::CloseContext(0),
                Event::XclClose,
                Event::DeviceClose,
            ]
        );
    }

    #[test]
    #[ignore = "requires HETGPU_XRT_AU250_TEST=1 and the live AU250"]
    fn au250_vector_add_runs_when_requested() {
        assert_eq!(
            std::env::var("HETGPU_XRT_AU250_TEST").as_deref(),
            Ok("1"),
            "set HETGPU_XRT_AU250_TEST=1 only inside au250-run"
        );

        let (vector_a, vector_b, expected) = au250_vector_add_fixture();
        let mut output = vec![0u8; expected.len()];
        let request = XrtTmatmulRequest {
            assembly: r#"
                ldv v0, PARAM_MATRIX
                ldv v1, PARAM_INPUT
                add v2, v0, v1
                sv v2, PARAM_OUTPUT
                stall
            "#,
            matrix_label: "PARAM_MATRIX",
            input_label: "PARAM_INPUT",
            output_label: "PARAM_OUTPUT",
            matrix: &vector_a,
            input: &vector_b,
        };

        let status = submit_xrt_tmatmul(request, &mut output).unwrap();
        if output != expected {
            let byte = output
                .iter()
                .zip(&expected)
                .position(|(actual, wanted)| actual != wanted)
                .unwrap();
            let element_offset = byte & !1;
            panic!(
                "AU250 output mismatch at element {}: got raw {}, expected raw {}",
                byte / 2,
                i16::from_le_bytes(
                    output[element_offset..element_offset + 2]
                        .try_into()
                        .unwrap()
                ),
                i16::from_le_bytes(
                    expected[element_offset..element_offset + 2]
                        .try_into()
                        .unwrap()
                )
            );
        }
        eprintln!(
            "AU250 XRT vector add PASS: {} elements, program={} bytes, stall=0x{:08x}",
            AU250_VECTOR_ELEMENTS, status.program_bytes, status.stall_code
        );
    }

    #[test]
    #[ignore = "requires HETGPU_XRT_AU250_TEST=1 and the live AU250"]
    fn au250_tmatmul_runs_when_requested() {
        assert_eq!(
            std::env::var("HETGPU_XRT_AU250_TEST").as_deref(),
            Ok("1"),
            "set HETGPU_XRT_AU250_TEST=1 only inside au250-run"
        );

        let (matrix, input, expected) = au250_tmatmul_fixture();
        let mut output = vec![0u8; expected.len()];
        let request = XrtTmatmulRequest {
            assembly: TEST_ASSEMBLY,
            matrix_label: "PARAM_MATRIX",
            input_label: "PARAM_INPUT",
            output_label: "PARAM_OUTPUT",
            matrix: &matrix,
            input: &input,
        };

        let status = submit_xrt_tmatmul(request, &mut output).unwrap();
        if output != expected {
            let byte = output
                .iter()
                .zip(&expected)
                .position(|(actual, wanted)| actual != wanted)
                .unwrap();
            let element_offset = byte & !1;
            panic!(
                "AU250 tmatmul mismatch at lane {}, element {}: got raw {}, expected raw {}",
                byte / 2 % AU250_BATCH_SIZE,
                byte / 2 / AU250_BATCH_SIZE,
                i16::from_le_bytes(
                    output[element_offset..element_offset + 2]
                        .try_into()
                        .unwrap()
                ),
                i16::from_le_bytes(
                    expected[element_offset..element_offset + 2]
                        .try_into()
                        .unwrap()
                )
            );
        }
        eprintln!(
            "AU250 XRT tmatmul PASS: {}x{} ternary matrix, {} lanes, program={} bytes, stall=0x{:08x}",
            AU250_VECTOR_LENGTH,
            AU250_VECTOR_LENGTH,
            AU250_BATCH_SIZE,
            status.program_bytes,
            status.stall_code
        );
    }

    #[test]
    fn timeout_quiesces_native_ip_before_releasing_all_handles() {
        let xrt = FakeXrt::new([]);
        let mut output = [0xa5; 8];

        let error =
            submit_with_ops(&xrt, &native_ip_config(1), test_request(), &mut output).unwrap_err();

        assert_eq!(error, XrtTmatmulError::Timeout { timeout_ms: 1 });
        assert_eq!(output, [0xa5; 8]);
        let events = xrt.events();
        assert!(!events
            .iter()
            .any(|event| matches!(event, Event::BoRead { .. })));
        assert_eq!(
            &events[events.len() - 11..],
            &[
                Event::IpRegisterWrite {
                    offset: RESET,
                    value: 1,
                },
                Event::IpRegisterWrite {
                    offset: MM2S_DMACR,
                    value: 1 << 2,
                },
                Event::IpRegisterRead(MM2S_DMACR),
                Event::IpRegisterRead(MM2S_DMASR),
                Event::BoFree(103),
                Event::BoFree(102),
                Event::BoFree(101),
                Event::BoFree(100),
                Event::CloseContext(0),
                Event::XclClose,
                Event::DeviceClose,
            ]
        );
    }

    #[test]
    fn register_failure_releases_bos_in_reverse_order() {
        let xrt = FakeXrt::new([1]);
        xrt.fail_register_write(MM2S_LENGTH);
        let mut output = [0u8; 8];

        let error =
            submit_with_ops(&xrt, &test_config(20), test_request(), &mut output).unwrap_err();

        assert_eq!(
            error,
            XrtTmatmulError::Xrt {
                operation: "xrtKernelWriteRegister(MM2S_LENGTH)",
                code: -5,
            }
        );
        let events = xrt.events();
        assert_eq!(
            &events[events.len() - 6..],
            &[
                Event::BoFree(103),
                Event::BoFree(102),
                Event::BoFree(101),
                Event::BoFree(100),
                Event::KernelClose,
                Event::DeviceClose,
            ]
        );
    }

    #[test]
    fn unconfirmed_quiescence_retains_live_bos_and_device_handles() {
        let xrt = FakeXrt::new([]);
        xrt.fail_register_read(MM2S_DMACR);
        let mut output = [0xa5; 8];

        let error =
            submit_with_ops(&xrt, &native_ip_config(1), test_request(), &mut output).unwrap_err();

        assert!(matches!(error, XrtTmatmulError::Quiesce { .. }));
        assert_eq!(output, [0xa5; 8]);
        let events = xrt.events();
        assert!(events.contains(&Event::IpRegisterWrite {
            offset: RESET,
            value: 1,
        }));
        assert!(events.contains(&Event::IpRegisterWrite {
            offset: MM2S_DMACR,
            value: DMACR_RESET,
        }));
        assert!(!events.iter().any(|event| matches!(
            event,
            Event::BoFree(_) | Event::CloseContext(_) | Event::XclClose | Event::DeviceClose
        )));
    }
}
