mod command_wrapper;

use std::ffi::{CStr, c_char, c_int, c_uint};
use std::num::NonZeroU32;
use std::os::raw;

pub const SIFIVE_COMGR_INTERFACE_VERSION_MAJOR: u32 = 1;
pub const SIFIVE_COMGR_INTERFACE_VERSION_MINOR: u32 = 0;

// Status types
#[derive(Debug, Clone, Copy)]
pub struct sifive_comgr_status_s(pub NonZeroU32);
pub type sifive_comgr_status_t = Result<(), self::sifive_comgr_status_s>;

impl sifive_comgr_status_s {
    pub const SIFIVE_COMGR_STATUS_SUCCESS: Result<(), sifive_comgr_status_s> = Ok(());

    pub const SIFIVE_COMGR_STATUS_ERROR: sifive_comgr_status_s =
        sifive_comgr_status_s(unsafe { NonZeroU32::new_unchecked(1) });

    pub const SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT: sifive_comgr_status_s =
        sifive_comgr_status_s(unsafe { NonZeroU32::new_unchecked(2) });

    pub const SIFIVE_COMGR_STATUS_ERROR_OUT_OF_RESOURCES: sifive_comgr_status_s =
        sifive_comgr_status_s(unsafe { NonZeroU32::new_unchecked(3) });
}

// Language types
#[derive(Default, Clone, Copy)]
pub struct sifive_comgr_language_s(pub c_uint);
pub type sifive_comgr_language_t = sifive_comgr_language_s;

impl sifive_comgr_language_s {
    pub const SIFIVE_COMGR_LANGUAGE_NONE: sifive_comgr_language_s = sifive_comgr_language_s(0);
    pub const SIFIVE_COMGR_LANGUAGE_OPENCL_1_2: sifive_comgr_language_s =
        sifive_comgr_language_s(1);
    pub const SIFIVE_COMGR_LANGUAGE_OPENCL_2_0: sifive_comgr_language_s =
        sifive_comgr_language_s(2);
    pub const SIFIVE_COMGR_LANGUAGE_SYCL: sifive_comgr_language_s = sifive_comgr_language_s(3);
    pub const SIFIVE_COMGR_LANGUAGE_LLVM_IR: sifive_comgr_language_s = sifive_comgr_language_s(4);
    pub const SIFIVE_COMGR_LANGUAGE_LAST: sifive_comgr_language_s = sifive_comgr_language_s(4);
}

// Data kinds
#[derive(Default, Clone, Copy)]
pub struct sifive_comgr_data_kind_s(pub c_uint);
pub type sifive_comgr_data_kind_t = sifive_comgr_data_kind_s;

impl sifive_comgr_data_kind_s {
    pub const SIFIVE_COMGR_DATA_KIND_UNDEF: sifive_comgr_data_kind_s = sifive_comgr_data_kind_s(0);
    pub const SIFIVE_COMGR_DATA_KIND_SOURCE: sifive_comgr_data_kind_s = sifive_comgr_data_kind_s(1);
    pub const SIFIVE_COMGR_DATA_KIND_INCLUDE: sifive_comgr_data_kind_s =
        sifive_comgr_data_kind_s(2);
    pub const SIFIVE_COMGR_DATA_KIND_PRECOMPILED_HEADER: sifive_comgr_data_kind_s =
        sifive_comgr_data_kind_s(3);
    pub const SIFIVE_COMGR_DATA_KIND_DIAGNOSTIC: sifive_comgr_data_kind_s =
        sifive_comgr_data_kind_s(4);
    pub const SIFIVE_COMGR_DATA_KIND_LOG: sifive_comgr_data_kind_s = sifive_comgr_data_kind_s(5);
    pub const SIFIVE_COMGR_DATA_KIND_BC: sifive_comgr_data_kind_s = sifive_comgr_data_kind_s(6);
    pub const SIFIVE_COMGR_DATA_KIND_RELOCATABLE: sifive_comgr_data_kind_s =
        sifive_comgr_data_kind_s(7);
    pub const SIFIVE_COMGR_DATA_KIND_EXECUTABLE: sifive_comgr_data_kind_s =
        sifive_comgr_data_kind_s(8);
    pub const SIFIVE_COMGR_DATA_KIND_BYTES: sifive_comgr_data_kind_s = sifive_comgr_data_kind_s(9);
    pub const SIFIVE_COMGR_DATA_KIND_FATBIN: sifive_comgr_data_kind_s =
        sifive_comgr_data_kind_s(16);
    pub const SIFIVE_COMGR_DATA_KIND_LAST: sifive_comgr_data_kind_s = sifive_comgr_data_kind_s(16);
}

// Data structures
#[derive(Default, Clone, Copy)]
pub struct sifive_comgr_data_s {
    pub handle: u64,
}
pub type sifive_comgr_data_t = sifive_comgr_data_s;

#[derive(Default, Clone, Copy)]
pub struct sifive_comgr_data_set_s {
    pub handle: u64,
}
pub type sifive_comgr_data_set_t = sifive_comgr_data_set_s;

#[derive(Default, Clone, Copy)]
pub struct sifive_comgr_action_info_s {
    pub handle: u64,
}
pub type sifive_comgr_action_info_t = sifive_comgr_action_info_s;

pub struct sifive_comgr_metadata_node_s {
    pub handle: u64,
}
pub type sifive_comgr_metadata_node_t = sifive_comgr_metadata_node_s;

pub struct sifive_comgr_symbol_s {
    pub handle: u64,
}
pub type sifive_comgr_symbol_t = sifive_comgr_symbol_s;

// Metadata kind constants
pub struct sifive_comgr_metadata_kind_s(pub c_uint);
pub type sifive_comgr_metadata_kind_t = sifive_comgr_metadata_kind_s;

impl sifive_comgr_metadata_kind_s {
    pub const SIFIVE_COMGR_METADATA_KIND_NULL: sifive_comgr_metadata_kind_s =
        sifive_comgr_metadata_kind_s(0);
    pub const SIFIVE_COMGR_METADATA_KIND_STRING: sifive_comgr_metadata_kind_s =
        sifive_comgr_metadata_kind_s(1);
    pub const SIFIVE_COMGR_METADATA_KIND_MAP: sifive_comgr_metadata_kind_s =
        sifive_comgr_metadata_kind_s(2);
    pub const SIFIVE_COMGR_METADATA_KIND_LIST: sifive_comgr_metadata_kind_s =
        sifive_comgr_metadata_kind_s(3);
    pub const SIFIVE_COMGR_METADATA_KIND_LAST: sifive_comgr_metadata_kind_s =
        sifive_comgr_metadata_kind_s(3);
}

// Action kinds
#[derive(Default, Clone, Copy)]
pub struct sifive_comgr_action_kind_s(pub c_uint);
pub type sifive_comgr_action_kind_t = sifive_comgr_action_kind_s;

impl sifive_comgr_action_kind_s {
    pub const SIFIVE_COMGR_ACTION_SOURCE_TO_PREPROCESSOR: sifive_comgr_action_kind_s =
        sifive_comgr_action_kind_s(0);
    pub const SIFIVE_COMGR_ACTION_ADD_PRECOMPILED_HEADERS: sifive_comgr_action_kind_s =
        sifive_comgr_action_kind_s(1);
    pub const SIFIVE_COMGR_ACTION_COMPILE_SOURCE_TO_BC: sifive_comgr_action_kind_s =
        sifive_comgr_action_kind_s(2);
    pub const SIFIVE_COMGR_ACTION_ADD_DEVICE_LIBRARIES: sifive_comgr_action_kind_s =
        sifive_comgr_action_kind_s(3);
    pub const SIFIVE_COMGR_ACTION_LINK_BC_TO_BC: sifive_comgr_action_kind_s =
        sifive_comgr_action_kind_s(4);
    pub const SIFIVE_COMGR_ACTION_OPTIMIZE_BC_TO_BC: sifive_comgr_action_kind_s =
        sifive_comgr_action_kind_s(5);
    pub const SIFIVE_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE: sifive_comgr_action_kind_s =
        sifive_comgr_action_kind_s(6);
    pub const SIFIVE_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY: sifive_comgr_action_kind_s =
        sifive_comgr_action_kind_s(7);
    pub const SIFIVE_COMGR_ACTION_COMPILE_SOURCE_TO_FATBIN: sifive_comgr_action_kind_s =
        sifive_comgr_action_kind_s(8);
    pub const SIFIVE_COMGR_ACTION_LAST: sifive_comgr_action_kind_s = sifive_comgr_action_kind_s(8);
}

// Symbol types
pub struct sifive_comgr_symbol_type_s(pub c_int);
pub type sifive_comgr_symbol_type_t = sifive_comgr_symbol_type_s;

impl sifive_comgr_symbol_type_s {
    pub const SIFIVE_COMGR_SYMBOL_TYPE_UNKNOWN: sifive_comgr_symbol_type_s =
        sifive_comgr_symbol_type_s(-1);
    pub const SIFIVE_COMGR_SYMBOL_TYPE_NOTYPE: sifive_comgr_symbol_type_s =
        sifive_comgr_symbol_type_s(0);
    pub const SIFIVE_COMGR_SYMBOL_TYPE_OBJECT: sifive_comgr_symbol_type_s =
        sifive_comgr_symbol_type_s(1);
    pub const SIFIVE_COMGR_SYMBOL_TYPE_FUNC: sifive_comgr_symbol_type_s =
        sifive_comgr_symbol_type_s(2);
    pub const SIFIVE_COMGR_SYMBOL_TYPE_SECTION: sifive_comgr_symbol_type_s =
        sifive_comgr_symbol_type_s(3);
    pub const SIFIVE_COMGR_SYMBOL_TYPE_FILE: sifive_comgr_symbol_type_s =
        sifive_comgr_symbol_type_s(4);
    pub const SIFIVE_COMGR_SYMBOL_TYPE_COMMON: sifive_comgr_symbol_type_s =
        sifive_comgr_symbol_type_s(5);
}

pub struct sifive_comgr_symbol_info_s(pub c_uint);
pub type sifive_comgr_symbol_info_t = sifive_comgr_symbol_info_s;

impl sifive_comgr_symbol_info_s {
    pub const SIFIVE_COMGR_SYMBOL_INFO_NAME_LENGTH: sifive_comgr_symbol_info_s =
        sifive_comgr_symbol_info_s(0);
    pub const SIFIVE_COMGR_SYMBOL_INFO_NAME: sifive_comgr_symbol_info_s =
        sifive_comgr_symbol_info_s(1);
    pub const SIFIVE_COMGR_SYMBOL_INFO_TYPE: sifive_comgr_symbol_info_s =
        sifive_comgr_symbol_info_s(2);
    pub const SIFIVE_COMGR_SYMBOL_INFO_SIZE: sifive_comgr_symbol_info_s =
        sifive_comgr_symbol_info_s(3);
    pub const SIFIVE_COMGR_SYMBOL_INFO_IS_UNDEFINED: sifive_comgr_symbol_info_s =
        sifive_comgr_symbol_info_s(4);
    pub const SIFIVE_COMGR_SYMBOL_INFO_VALUE: sifive_comgr_symbol_info_s =
        sifive_comgr_symbol_info_s(5);
    pub const SIFIVE_COMGR_SYMBOL_INFO_LAST: sifive_comgr_symbol_info_s =
        sifive_comgr_symbol_info_s(5);
}

// Code object info
pub struct sifive_comgr_code_object_info_s {
    pub isa: *const c_char,
    pub size: usize,
    pub offset: u64,
}
pub type sifive_comgr_code_object_info_t = sifive_comgr_code_object_info_s;

// --- SIFIVE-specific configuration ---

/// SIFIVE codegen mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SifiveCodegenMode {
    /// VCIX coprocessor interface (for real SiFive XM hardware)
    /// Uses sf.vc.v.vvv/vvw intrinsics → CUSTOM-2 opcode space
    Vcix,
    /// Native Zvbdot block dot product instructions (for SiFive spike simulation)
    /// Uses vqbdotu/vfwbdot/vfqbdot → standard V encoding
    Zvbdot,
}

/// SIFIVE hardware configuration for SiFive Intelligence X390 / RISC-V IME
#[derive(Debug, Clone)]
pub struct SifiveConfig {
    /// VLEN in bits (128, 256, 512, 1024, 2048, 4096)
    pub vlen: usize,
    /// Default SEW for matrix operations (4, 8, 16)
    pub default_sew: usize,
    /// Target triple
    pub target_triple: String,
    /// March string for LLVM codegen
    pub march: String,
    /// LLVM -mattr string for codegen
    pub mattr: String,
    /// Codegen mode: VCIX (hardware) or Zvbdot (spike simulation)
    pub codegen_mode: SifiveCodegenMode,
    /// ISA string for spike simulator
    pub spike_isa: String,
}

impl Default for SifiveConfig {
    fn default() -> Self {
        Self::x390_sim()
    }
}

impl SifiveConfig {
    /// SiFive X390 configuration for spike simulation (Zvbdot path)
    pub fn x390_sim() -> Self {
        Self {
            vlen: 1024,
            default_sew: 8,
            target_triple: "riscv64-unknown-elf".to_string(),
            march: "rv64gcv_zbb_zfh_zvfh_zfbfmin_zvfbfmin_zvfbfwma_zvl1024b".to_string(),
            mattr: "+v,+d,+f,+zbb,+zfh,+zvfh,+zfbfmin,+zvfbfmin,+zvfbfwma,+zvl1024b"
                .to_string(),
            codegen_mode: SifiveCodegenMode::Zvbdot,
            spike_isa: "rv64gcv_zbb_zfh_zvfh_zfbfmin_zvfbfmin_zvfbfwma_zvfbfa_zvqbdot8i_zvqbdot16i_zvfqbdot8f_zvfwbdot16bf_zvfbdot32f_zvl1024b".to_string(),
        }
    }

    /// SiFive XM hardware configuration (VCIX path)
    pub fn xm_hardware() -> Self {
        Self {
            vlen: 1024,
            default_sew: 8,
            target_triple: "riscv64-unknown-elf".to_string(),
            march: "rv64gcv_zbb_zfh_zvfh_zfbfmin_zvfbfmin_zvfbfwma_zvl1024b_xsfvcp_xsfvfnrclipxfqf_xsfvqmaccqoq"
                .to_string(),
            mattr: "+v,+d,+f,+zbb,+zfh,+zvfh,+zfbfmin,+zvfbfmin,+zvfbfwma,+zvl1024b,+xsfvcp,+xsfvfnrclipxfqf,+xsfvqmaccqoq"
                .to_string(),
            codegen_mode: SifiveCodegenMode::Vcix,
            spike_isa: String::new(),
        }
    }

    /// Linux RISC-V RVV+BF16 configuration for source-level operator offload
    /// compatibility with llama.cpp CPU kernels such as ggml-cpu/vec.cpp.
    pub fn rvv_linux_bf16() -> Self {
        Self {
            vlen: 1024,
            default_sew: 8,
            target_triple: "riscv64-linux-gnu".to_string(),
            march: "rv64gcv_zbb_zfh_zvfh_zfbfmin_zvfbfmin_zvfbfwma_zvl1024b".to_string(),
            mattr: "+v,+d,+f,+zbb,+zfh,+zvfh,+zfbfmin,+zvfbfmin,+zvfbfwma,+zvl1024b".to_string(),
            codegen_mode: SifiveCodegenMode::Zvbdot,
            spike_isa: String::new(),
        }
    }

    /// Compute matrix tile dimensions (M, N, K) for given SEW
    pub fn tile_dims(&self, sew: usize) -> (usize, usize, usize) {
        let total_bits = self.vlen;
        let m = (total_bits as f64 / 64.0).sqrt() as usize;
        let n = m;
        let k = total_bits / (m * sew);
        (m, n, k)
    }

    /// Whether the hardware uses Copies=2 mode
    pub fn uses_copies2(&self) -> bool {
        let sqrt_val = (self.vlen as f64 / 64.0).sqrt();
        (sqrt_val - sqrt_val.floor()).abs() > 1e-9
    }

    /// Number of elements per vector register for given SEW
    pub fn vl(&self, sew: usize) -> usize {
        self.vlen / sew
    }
}

// --- API functions ---

pub fn sifive_comgr_create_data(
    kind: sifive_comgr_data_kind_t,
    data: *mut sifive_comgr_data_t,
) -> sifive_comgr_status_t {
    let mut store = command_wrapper::DATA_STORE.lock().unwrap();
    let handle = command_wrapper::get_next_handle();
    let data_obj = sifive_comgr_data_t { handle };
    store.insert(
        handle,
        command_wrapper::DataContent {
            kind,
            content: Vec::new(),
            name: None,
        },
    );
    unsafe {
        *data = data_obj;
    }
    Ok(())
}

pub fn sifive_comgr_release_data(_data: sifive_comgr_data_t) -> sifive_comgr_status_t {
    // Keep backing storage alive for the duration of this process. The current
    // SIFIVE COMGR shim stores bare handles inside data sets, so eagerly dropping
    // data here makes multi-stage pipelines (SOURCE -> BC -> OPT -> CODEGEN)
    // lose their payloads between actions.
    Ok(())
}

pub fn sifive_comgr_data_set_bytes(
    data: sifive_comgr_data_t,
    bytes: *const raw::c_void,
    size: usize,
) -> sifive_comgr_status_t {
    if bytes.is_null() && size > 0 {
        return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let content = if size > 0 {
        let slice = unsafe { std::slice::from_raw_parts(bytes as *const u8, size) };
        slice.to_vec()
    } else {
        Vec::new()
    };

    let mut data_store = command_wrapper::DATA_STORE.lock().unwrap();
    if let Some(data_content) = data_store.get_mut(&data.handle) {
        data_content.content = content;
        Ok(())
    } else {
        Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn sifive_comgr_data_set_name(
    data: sifive_comgr_data_t,
    name: *const c_char,
) -> sifive_comgr_status_t {
    if name.is_null() {
        return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let name_str = unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .to_string();

    let mut data_store = command_wrapper::DATA_STORE.lock().unwrap();
    if let Some(data_content) = data_store.get_mut(&data.handle) {
        data_content.name = Some(name_str);
        Ok(())
    } else {
        Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn sifive_comgr_create_data_set(
    data_set: *mut sifive_comgr_data_set_t,
) -> sifive_comgr_status_t {
    if data_set.is_null() {
        return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let handle = command_wrapper::get_next_handle();
    {
        let mut data_set_store = command_wrapper::DATA_SET_STORE.lock().unwrap();
        data_set_store.insert(handle, Vec::new());
    }
    unsafe {
        *data_set = sifive_comgr_data_set_t { handle };
    }
    Ok(())
}

pub fn sifive_comgr_release_data_set(data_set: sifive_comgr_data_set_t) -> sifive_comgr_status_t {
    let mut data_set_store = command_wrapper::DATA_SET_STORE.lock().unwrap();
    data_set_store.remove(&data_set.handle);
    Ok(())
}

pub fn sifive_comgr_data_set_add(
    data_set: sifive_comgr_data_set_t,
    data: sifive_comgr_data_t,
) -> sifive_comgr_status_t {
    let mut data_set_store = command_wrapper::DATA_SET_STORE.lock().unwrap();
    if let Some(set_handles) = data_set_store.get_mut(&data_set.handle) {
        set_handles.push(data.handle);
        Ok(())
    } else {
        Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn sifive_comgr_create_action_info(
    action_info: *mut sifive_comgr_action_info_t,
) -> sifive_comgr_status_t {
    if action_info.is_null() {
        return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let handle = command_wrapper::get_next_handle();
    {
        let mut action_info_store = command_wrapper::ACTION_INFO_STORE.lock().unwrap();
        action_info_store.insert(handle, command_wrapper::ActionInfo::default());
    }
    unsafe {
        *action_info = sifive_comgr_action_info_t { handle };
    }
    Ok(())
}

pub fn sifive_comgr_release_action_info(
    action_info: sifive_comgr_action_info_t,
) -> sifive_comgr_status_t {
    let mut action_info_store = command_wrapper::ACTION_INFO_STORE.lock().unwrap();
    action_info_store.remove(&action_info.handle);
    Ok(())
}

pub fn sifive_comgr_action_info_set_language(
    action_info: sifive_comgr_action_info_t,
    language: sifive_comgr_language_t,
) -> sifive_comgr_status_t {
    let mut action_info_store = command_wrapper::ACTION_INFO_STORE.lock().unwrap();
    if let Some(info) = action_info_store.get_mut(&action_info.handle) {
        info.language = Some(language);
        Ok(())
    } else {
        Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn sifive_comgr_action_info_set_option_list(
    action_info: sifive_comgr_action_info_t,
    options: *const *const c_char,
    count: usize,
) -> sifive_comgr_status_t {
    if options.is_null() && count > 0 {
        return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let mut option_strings = Vec::new();
    for i in 0..count {
        let opt_ptr = unsafe { *options.add(i) };
        if opt_ptr.is_null() {
            continue;
        }

        let opt_str = unsafe { CStr::from_ptr(opt_ptr) }
            .to_string_lossy()
            .to_string();
        option_strings.push(opt_str);
    }

    let mut action_info_store = command_wrapper::ACTION_INFO_STORE.lock().unwrap();
    if let Some(info) = action_info_store.get_mut(&action_info.handle) {
        info.options = option_strings;
        Ok(())
    } else {
        Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn sifive_comgr_action_info_set_working_directory(
    action_info: sifive_comgr_action_info_t,
    directory: *const c_char,
) -> sifive_comgr_status_t {
    if directory.is_null() {
        return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let dir_str = unsafe { CStr::from_ptr(directory) }
        .to_string_lossy()
        .to_string();

    let mut action_info_store = command_wrapper::ACTION_INFO_STORE.lock().unwrap();
    if let Some(info) = action_info_store.get_mut(&action_info.handle) {
        info.working_directory = Some(dir_str);
        Ok(())
    } else {
        Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn sifive_comgr_action_info_set_target(
    action_info: sifive_comgr_action_info_t,
    target: *const c_char,
) -> sifive_comgr_status_t {
    if target.is_null() {
        return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let target_str = unsafe { CStr::from_ptr(target) }
        .to_string_lossy()
        .to_string();

    let mut action_info_store = command_wrapper::ACTION_INFO_STORE.lock().unwrap();
    if let Some(info) = action_info_store.get_mut(&action_info.handle) {
        info.target = Some(target_str);
        Ok(())
    } else {
        Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn sifive_comgr_do_action(
    action_kind: sifive_comgr_action_kind_t,
    action_info: sifive_comgr_action_info_t,
    input_set: sifive_comgr_data_set_t,
    output_set: sifive_comgr_data_set_t,
) -> sifive_comgr_status_t {
    command_wrapper::perform_action(action_kind, action_info, input_set, output_set)
}

pub fn sifive_comgr_get_data_count(
    data_set: sifive_comgr_data_set_t,
    count: *mut usize,
) -> sifive_comgr_status_t {
    if count.is_null() {
        return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let data_set_store = command_wrapper::DATA_SET_STORE.lock().unwrap();
    if let Some(set_handles) = data_set_store.get(&data_set.handle) {
        unsafe {
            *count = set_handles.len();
        }
        Ok(())
    } else {
        Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn sifive_comgr_get_data(
    data_set: sifive_comgr_data_set_t,
    index: usize,
    data: *mut sifive_comgr_data_t,
) -> sifive_comgr_status_t {
    if data.is_null() {
        return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let data_set_store = command_wrapper::DATA_SET_STORE.lock().unwrap();
    if let Some(set_handles) = data_set_store.get(&data_set.handle) {
        if index >= set_handles.len() {
            return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
        }
        unsafe {
            *data = sifive_comgr_data_t {
                handle: set_handles[index],
            };
        }
        Ok(())
    } else {
        Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn sifive_comgr_get_data_kind(
    data: sifive_comgr_data_t,
    kind: *mut sifive_comgr_data_kind_t,
) -> sifive_comgr_status_t {
    if kind.is_null() {
        return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let data_store = command_wrapper::DATA_STORE.lock().unwrap();
    if let Some(data_content) = data_store.get(&data.handle) {
        unsafe {
            *kind = data_content.kind;
        }
        Ok(())
    } else {
        Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

pub fn sifive_comgr_data_get_bytes(
    data: sifive_comgr_data_t,
    bytes: *mut raw::c_void,
    size: *mut usize,
) -> sifive_comgr_status_t {
    if size.is_null() {
        return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let data_store = command_wrapper::DATA_STORE.lock().unwrap();
    if let Some(data_content) = data_store.get(&data.handle) {
        unsafe {
            *size = data_content.content.len();
        }

        if !bytes.is_null() && !data_content.content.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data_content.content.as_ptr(),
                    bytes as *mut u8,
                    data_content.content.len(),
                );
            }
        }

        Ok(())
    } else {
        Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_release_data() {
        let mut data = sifive_comgr_data_t { handle: 0 };
        let result = sifive_comgr_create_data(
            sifive_comgr_data_kind_s::SIFIVE_COMGR_DATA_KIND_SOURCE,
            &mut data,
        );
        assert!(result.is_ok());

        let result = sifive_comgr_release_data(data);
        assert!(result.is_ok());
    }

    #[test]
    fn sifive_tile_dims_x390() {
        let config = SifiveConfig::default(); // X390: VLEN=512, SEW=8
        let (m, n, k) = config.tile_dims(8);
        // sqrt(512/64) = sqrt(8) ≈ 2.83 → floor = 2
        assert_eq!(m, 2);
        assert_eq!(n, 2);
        assert_eq!(k, 32); // 512 / (2 * 8) = 32
        assert_eq!(config.vlen, 512);
        assert_eq!(config.codegen_mode, SifiveCodegenMode::Zvbdot);
    }
}
