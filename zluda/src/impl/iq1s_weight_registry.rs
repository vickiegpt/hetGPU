use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::ffi::{c_char, c_void, CStr, OsString};
use std::fs::{File, OpenOptions};
use std::os::unix::ffi::OsStringExt;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::UNIX_EPOCH;

const IQ1S_GGML_TYPE: u32 = 19;
const IQ1S_BLOCK_VALUES: u64 = 256;
const IQ1S_BLOCK_BYTES: u64 = 50;

pub(crate) const HETGPU_IQ1S_ABI_VERSION: u32 = 1;
pub(crate) const HETGPU_IQ1S_HANDLED: i32 = 1;
pub(crate) const HETGPU_IQ1S_ERROR: i32 = -1;
pub(crate) const HETGPU_IQ1S_ROLE_GATE_EXPS: u32 = 1;
pub(crate) const HETGPU_IQ1S_ROLE_UP_EXPS: u32 = 2;
pub(crate) const HETGPU_IQ1S_ROLE_DOWN_EXPS: u32 = 3;
pub(crate) const HETGPU_IQ1S_ROLE_GATE_UP_EXPS: u32 = 4;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HetgpuIq1sTensorV1 {
    pub abi_version: u32,
    pub ggml_type: u32,
    pub role: u32,
    pub file_index: u32,
    pub name: *const c_char,
    pub path: *const c_char,
    pub file_offset: u64,
    pub nbytes: u64,
    pub ne: [i64; 4],
    pub nb: [u64; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HetgpuIq1sDeviceBindingV1 {
    pub abi_version: u32,
    pub reserved: u32,
    pub name: *const c_char,
    pub device_base: *const c_void,
    pub allocation_bytes: u64,
    pub allocation_generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Iq1sExpertRole {
    Gate,
    Up,
    Down,
    GateUp,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Iq1sTensorRegistration {
    pub(crate) path: PathBuf,
    pub(crate) file_offset: u64,
    pub(crate) nbytes: u64,
    pub(crate) name: String,
    pub(crate) ne: [u64; 4],
    pub(crate) nb: [u64; 4],
    pub(crate) role: Iq1sExpertRole,
    pub(crate) ggml_type: u32,
    pub(crate) model_sha256: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct Iq1sTensorIdentity {
    pub(crate) canonical_path: PathBuf,
    pub(crate) file_offset: u64,
    pub(crate) nbytes: u64,
    pub(crate) name: String,
    pub(crate) layer: u32,
    pub(crate) ne: [u64; 4],
    pub(crate) nb: [u64; 4],
    pub(crate) role: Iq1sExpertRole,
    pub(crate) model_sha256: [u8; 32],
    pub(crate) content_sha256: [u8; 32],
    pub(crate) device: u64,
    pub(crate) inode: u64,
    pub(crate) modified_ns: u128,
}

#[derive(Debug, Clone)]
pub(crate) struct Iq1sTensorSource {
    pub(crate) identity: Iq1sTensorIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Iq1sDeviceBinding {
    pub(crate) name: String,
    pub(crate) base_ptr: usize,
    pub(crate) allocation_bytes: u64,
    pub(crate) allocation_generation: u64,
}

#[derive(Debug, Clone)]
struct BoundIq1sTensor {
    binding: Iq1sDeviceBinding,
    source: Arc<Iq1sTensorSource>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedIq1sWeight {
    pub(crate) identity: Iq1sTensorIdentity,
    pub(crate) expert: u64,
    pub(crate) allocation_generation: u64,
    pub(crate) content_sha256: [u8; 32],
}

#[derive(Debug, Default)]
pub(crate) struct Iq1sWeightRegistry {
    sources: RwLock<HashMap<String, Arc<Iq1sTensorSource>>>,
    bindings: RwLock<Vec<BoundIq1sTensor>>,
}

static GLOBAL_REGISTRY: OnceLock<Iq1sWeightRegistry> = OnceLock::new();

pub(crate) fn global_registry() -> &'static Iq1sWeightRegistry {
    GLOBAL_REGISTRY.get_or_init(Iq1sWeightRegistry::default)
}

fn parse_model_sha256(value: &str) -> Result<[u8; 32], String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(
            "HETGPU_QWEN_MODEL_SHA256 must contain exactly 64 hexadecimal digits".to_string(),
        );
    }
    let mut digest = [0u8; 32];
    for (index, output) in digest.iter_mut().enumerate() {
        *output = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| "HETGPU_QWEN_MODEL_SHA256 contains invalid hexadecimal".to_string())?;
    }
    if digest == [0; 32] {
        return Err("HETGPU_QWEN_MODEL_SHA256 must be nonzero".to_string());
    }
    Ok(digest)
}

fn role_from_ffi(value: u32) -> Result<Iq1sExpertRole, String> {
    match value {
        HETGPU_IQ1S_ROLE_GATE_EXPS => Ok(Iq1sExpertRole::Gate),
        HETGPU_IQ1S_ROLE_UP_EXPS => Ok(Iq1sExpertRole::Up),
        HETGPU_IQ1S_ROLE_DOWN_EXPS => Ok(Iq1sExpertRole::Down),
        HETGPU_IQ1S_ROLE_GATE_UP_EXPS => Ok(Iq1sExpertRole::GateUp),
        _ => Err("IQ1_S registration role is invalid".to_string()),
    }
}

unsafe fn registration_from_ffi(
    tensor: *const HetgpuIq1sTensorV1,
    model_sha256: [u8; 32],
) -> Result<Iq1sTensorRegistration, String> {
    let tensor = tensor
        .as_ref()
        .ok_or_else(|| "IQ1_S registration pointer is null".to_string())?;
    if tensor.abi_version != HETGPU_IQ1S_ABI_VERSION {
        return Err(format!(
            "IQ1_S registration ABI version {} is unsupported",
            tensor.abi_version
        ));
    }
    if tensor.ggml_type != IQ1S_GGML_TYPE {
        return Err(format!(
            "IQ1_S registration requires GGML type 19, got {}",
            tensor.ggml_type
        ));
    }
    if tensor.name.is_null() || tensor.path.is_null() {
        return Err("IQ1_S registration name or path is null".to_string());
    }
    let name = CStr::from_ptr(tensor.name)
        .to_str()
        .map_err(|_| "IQ1_S registration name is not UTF-8".to_string())?
        .to_string();
    let path = PathBuf::from(OsString::from_vec(
        CStr::from_ptr(tensor.path).to_bytes().to_vec(),
    ));
    let mut ne = [0u64; 4];
    for (index, value) in tensor.ne.into_iter().enumerate() {
        ne[index] = u64::try_from(value)
            .map_err(|_| format!("IQ1_S registration ne[{index}] must be nonnegative"))?;
    }
    Ok(Iq1sTensorRegistration {
        path,
        file_offset: tensor.file_offset,
        nbytes: tensor.nbytes,
        name,
        ne,
        nb: tensor.nb,
        role: role_from_ffi(tensor.role)?,
        ggml_type: tensor.ggml_type,
        model_sha256,
    })
}

unsafe fn binding_from_ffi(
    binding: *const HetgpuIq1sDeviceBindingV1,
) -> Result<Iq1sDeviceBinding, String> {
    let binding = binding
        .as_ref()
        .ok_or_else(|| "IQ1_S device binding pointer is null".to_string())?;
    if binding.abi_version != HETGPU_IQ1S_ABI_VERSION {
        return Err(format!(
            "IQ1_S device binding ABI version {} is unsupported",
            binding.abi_version
        ));
    }
    if binding.reserved != 0 {
        return Err("IQ1_S device binding reserved bits must be zero".to_string());
    }
    if binding.name.is_null() || binding.device_base.is_null() {
        return Err("IQ1_S device binding name or device pointer is null".to_string());
    }
    let name = CStr::from_ptr(binding.name)
        .to_str()
        .map_err(|_| "IQ1_S device binding name is not UTF-8".to_string())?
        .to_string();
    Ok(Iq1sDeviceBinding {
        name,
        base_ptr: binding.device_base as usize,
        allocation_bytes: binding.allocation_bytes,
        allocation_generation: binding.allocation_generation,
    })
}

pub(crate) unsafe fn register_tensor(tensor: *const HetgpuIq1sTensorV1) -> i32 {
    let result = (|| -> Result<(), String> {
        let model_sha256 = std::env::var("HETGPU_QWEN_MODEL_SHA256")
            .map_err(|error| format!("read HETGPU_QWEN_MODEL_SHA256: {error}"))
            .and_then(|value| parse_model_sha256(&value))?;
        global_registry().register(registration_from_ffi(tensor, model_sha256)?)?;
        Ok(())
    })();
    match result {
        Ok(()) => HETGPU_IQ1S_HANDLED,
        Err(error) => {
            eprintln!("[hetgpu-iq1s] registration failed: {error}");
            HETGPU_IQ1S_ERROR
        }
    }
}

pub(crate) unsafe fn bind_device(binding: *const HetgpuIq1sDeviceBindingV1) -> i32 {
    match binding_from_ffi(binding).and_then(|binding| global_registry().bind(binding)) {
        Ok(()) => HETGPU_IQ1S_HANDLED,
        Err(error) => {
            eprintln!("[hetgpu-iq1s] device binding failed: {error}");
            HETGPU_IQ1S_ERROR
        }
    }
}

pub(crate) fn classify_tensor_name(name: &str) -> Result<(u32, Iq1sExpertRole), String> {
    let remainder = name
        .strip_prefix("blk.")
        .ok_or_else(|| "invalid IQ1_S expert tensor name".to_string())?;
    let (layer, projection) = remainder
        .split_once(".ffn_")
        .ok_or_else(|| "invalid IQ1_S expert tensor name".to_string())?;
    if layer.is_empty() || !layer.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err("invalid IQ1_S expert tensor layer in name".to_string());
    }
    let layer = layer
        .parse::<u32>()
        .map_err(|_| "IQ1_S expert tensor layer does not fit u32".to_string())?;
    let role = match projection {
        "gate_exps.weight" => Iq1sExpertRole::Gate,
        "up_exps.weight" => Iq1sExpertRole::Up,
        "down_exps.weight" => Iq1sExpertRole::Down,
        "gate_up_exps.weight" => Iq1sExpertRole::GateUp,
        _ => return Err("invalid IQ1_S expert tensor projection in name".to_string()),
    };
    Ok((layer, role))
}

fn required_span(registration: &Iq1sTensorRegistration) -> Result<u64, String> {
    if registration.ggml_type != IQ1S_GGML_TYPE {
        return Err(format!(
            "IQ1_S registration requires GGML type 19, got {}",
            registration.ggml_type
        ));
    }
    let (layer, expected_role) = classify_tensor_name(&registration.name)?;
    let _ = layer;
    if expected_role != registration.role {
        return Err("IQ1_S expert tensor role does not agree with its name".to_string());
    }
    if registration.model_sha256 == [0; 32] {
        return Err("IQ1_S registration model SHA-256 must be nonzero".to_string());
    }
    if registration.ne[0] == 0 || !registration.ne[0].is_multiple_of(IQ1S_BLOCK_VALUES) {
        return Err("IQ1_S K must be a nonzero multiple of 256".to_string());
    }
    if registration.ne[1] == 0 || registration.ne[2] == 0 || registration.ne[3] != 1 {
        return Err(
            "IQ1_S row/expert dimensions must be positive and ne[3] must equal 1".to_string(),
        );
    }
    if registration.nb[0] != IQ1S_BLOCK_BYTES {
        return Err("IQ1_S nb[0] must equal 50".to_string());
    }
    let blocks = registration.ne[0] / IQ1S_BLOCK_VALUES;
    let packed_row = blocks
        .checked_mul(IQ1S_BLOCK_BYTES)
        .ok_or_else(|| "IQ1_S row span overflow".to_string())?;
    if registration.nb[1] < packed_row {
        return Err("IQ1_S row stride is smaller than the packed row".to_string());
    }
    let expert_span = registration.ne[1]
        .checked_mul(registration.nb[1])
        .ok_or_else(|| "IQ1_S expert span overflow".to_string())?;
    if registration.nb[2] < expert_span {
        return Err("IQ1_S expert stride is smaller than its rows".to_string());
    }
    let outer_span = registration.ne[2]
        .checked_mul(registration.nb[2])
        .ok_or_else(|| "IQ1_S outer span overflow".to_string())?;
    if registration.nb[3] < outer_span {
        return Err("IQ1_S outer stride is smaller than its experts".to_string());
    }
    registration.ne[2]
        .checked_sub(1)
        .and_then(|expert| expert.checked_mul(registration.nb[2]))
        .and_then(|prefix| {
            registration.ne[1]
                .checked_sub(1)
                .and_then(|row| row.checked_mul(registration.nb[1]))
                .and_then(|row_offset| prefix.checked_add(row_offset))
        })
        .and_then(|prefix| {
            blocks
                .checked_sub(1)
                .and_then(|block| block.checked_mul(IQ1S_BLOCK_BYTES))
                .and_then(|block_offset| prefix.checked_add(block_offset))
        })
        .and_then(|prefix| prefix.checked_add(IQ1S_BLOCK_BYTES))
        .ok_or_else(|| "IQ1_S tensor span overflow".to_string())
}

fn hash_file_range(file: &File, offset: u64, nbytes: u64) -> Result<[u8; 32], String> {
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut consumed = 0u64;
    while consumed < nbytes {
        let remaining = nbytes - consumed;
        let count = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| "IQ1_S hash chunk size does not fit usize")?;
        let absolute = offset
            .checked_add(consumed)
            .ok_or("IQ1_S hash offset overflow")?;
        file.read_exact_at(&mut buffer[..count], absolute)
            .map_err(|error| format!("read IQ1_S tensor bytes: {error}"))?;
        digest.update(&buffer[..count]);
        consumed = consumed
            .checked_add(count as u64)
            .ok_or("IQ1_S hash byte count overflow")?;
    }
    Ok(digest.finalize().into())
}

impl Iq1sWeightRegistry {
    pub(crate) fn registered_sources(&self) -> Result<Vec<Arc<Iq1sTensorSource>>, String> {
        let sources = self
            .sources
            .read()
            .map_err(|_| "IQ1_S tensor registry lock poisoned".to_string())?;
        let mut result = sources.values().cloned().collect::<Vec<_>>();
        result.sort_by(|left, right| {
            left.identity
                .layer
                .cmp(&right.identity.layer)
                .then_with(|| left.identity.role.cmp(&right.identity.role))
                .then_with(|| left.identity.name.cmp(&right.identity.name))
        });
        Ok(result)
    }

    pub(crate) fn expected_roles_for_layer(
        &self,
        layer: u32,
    ) -> Result<BTreeSet<Iq1sExpertRole>, String> {
        let sources = self
            .sources
            .read()
            .map_err(|_| "IQ1_S tensor registry lock poisoned".to_string())?;
        Ok(sources
            .values()
            .filter(|source| source.identity.layer == layer)
            .map(|source| source.identity.role)
            .collect())
    }

    pub(crate) fn register(
        &self,
        registration: Iq1sTensorRegistration,
    ) -> Result<Arc<Iq1sTensorSource>, String> {
        let required = required_span(&registration)?;
        if required > registration.nbytes {
            return Err("IQ1_S tensor span is smaller than its strided layout".to_string());
        }
        if registration.path.is_symlink() {
            return Err("IQ1_S tensor source must not be a symlink".to_string());
        }
        let canonical_path = registration
            .path
            .canonicalize()
            .map_err(|error| format!("canonicalize IQ1_S tensor file: {error}"))?;
        let file = OpenOptions::new()
            .read(true)
            .open(&canonical_path)
            .map_err(|error| format!("open IQ1_S tensor file read-only: {error}"))?;
        let metadata = file
            .metadata()
            .map_err(|error| format!("inspect IQ1_S tensor file: {error}"))?;
        if !metadata.is_file() {
            return Err("IQ1_S tensor source is not a regular file".to_string());
        }
        let file_end = registration
            .file_offset
            .checked_add(registration.nbytes)
            .ok_or("IQ1_S file range overflow")?;
        if file_end > metadata.len() {
            return Err("IQ1_S tensor range exceeds file size".to_string());
        }
        let modified_ns = metadata
            .modified()
            .map_err(|error| format!("read IQ1_S modification time: {error}"))?
            .duration_since(UNIX_EPOCH)
            .map_err(|_| "IQ1_S modification time predates Unix epoch")?
            .as_nanos();
        let content_sha256 = hash_file_range(&file, registration.file_offset, registration.nbytes)?;
        if content_sha256 == [0; 32] {
            return Err("IQ1_S tensor content SHA-256 must be nonzero".to_string());
        }
        let (layer, _) = classify_tensor_name(&registration.name)?;
        let identity = Iq1sTensorIdentity {
            canonical_path,
            file_offset: registration.file_offset,
            nbytes: registration.nbytes,
            name: registration.name,
            layer,
            ne: registration.ne,
            nb: registration.nb,
            role: registration.role,
            model_sha256: registration.model_sha256,
            content_sha256,
            device: metadata.dev(),
            inode: metadata.ino(),
            modified_ns,
        };
        let mut sources = self
            .sources
            .write()
            .map_err(|_| "IQ1_S tensor registry lock poisoned".to_string())?;
        if let Some(existing) = sources.get(&identity.name) {
            if existing.identity == identity {
                return Ok(existing.clone());
            }
            return Err(format!(
                "conflicting IQ1_S registration for {}",
                identity.name
            ));
        }
        let source = Arc::new(Iq1sTensorSource { identity });
        sources.insert(source.identity.name.clone(), source.clone());
        Ok(source)
    }

    pub(crate) fn bind(&self, binding: Iq1sDeviceBinding) -> Result<(), String> {
        if binding.base_ptr < 0x1000 || binding.allocation_bytes == 0 {
            return Err("IQ1_S device binding has an invalid pointer or empty span".to_string());
        }
        if binding.allocation_generation == 0 {
            return Err("IQ1_S allocation generation must be nonzero".to_string());
        }
        let source = self
            .sources
            .read()
            .map_err(|_| "IQ1_S tensor registry lock poisoned".to_string())?
            .get(&binding.name)
            .cloned()
            .ok_or_else(|| {
                format!(
                    "IQ1_S device binding references unregistered tensor {}",
                    binding.name
                )
            })?;
        if binding.allocation_bytes != source.identity.nbytes {
            return Err(format!(
                "IQ1_S device binding span {} does not match registered tensor span {}",
                binding.allocation_bytes, source.identity.nbytes
            ));
        }
        let end = (binding.base_ptr as u64)
            .checked_add(binding.allocation_bytes)
            .ok_or("IQ1_S device binding range overflow")?;
        let mut bindings = self
            .bindings
            .write()
            .map_err(|_| "IQ1_S device binding lock poisoned".to_string())?;
        if let Some(existing) = bindings
            .iter()
            .find(|item| item.binding.name == binding.name)
        {
            if existing.binding == binding {
                return Ok(());
            }
            if binding.allocation_generation <= existing.binding.allocation_generation {
                return Err("IQ1_S device binding has a stale allocation generation".to_string());
            }
        }
        for existing in bindings
            .iter()
            .filter(|item| item.binding.name != binding.name)
        {
            let existing_start = existing.binding.base_ptr as u64;
            let existing_end = existing_start
                .checked_add(existing.binding.allocation_bytes)
                .ok_or("registered IQ1_S device binding range overflow")?;
            if (binding.base_ptr as u64) < existing_end && existing_start < end {
                return Err(format!(
                    "IQ1_S device binding for {} overlaps {}",
                    binding.name, existing.binding.name
                ));
            }
        }
        bindings.retain(|item| item.binding.name != binding.name);
        bindings.push(BoundIq1sTensor { binding, source });
        Ok(())
    }

    pub(crate) fn lookup(
        &self,
        matrix_ptr: usize,
        matrix_bytes: u64,
    ) -> Result<ResolvedIq1sWeight, String> {
        if matrix_ptr < 0x1000 || matrix_bytes == 0 {
            return Err("IQ1_S lookup has an invalid pointer or empty range".to_string());
        }
        let requested_end = (matrix_ptr as u64)
            .checked_add(matrix_bytes)
            .ok_or("IQ1_S lookup range overflow")?;
        let bindings = self
            .bindings
            .read()
            .map_err(|_| "IQ1_S device binding lock poisoned".to_string())?;
        let bound = bindings
            .iter()
            .find(|item| {
                let start = item.binding.base_ptr as u64;
                let end = start.saturating_add(item.binding.allocation_bytes);
                (matrix_ptr as u64) >= start && requested_end <= end
            })
            .ok_or_else(|| {
                "IQ1_S matrix range is not inside a registered device binding".to_string()
            })?;
        let relative = (matrix_ptr as u64) - bound.binding.base_ptr as u64;
        let expert_stride = bound.source.identity.nb[2];
        if relative % expert_stride != 0 || matrix_bytes > expert_stride {
            return Err("IQ1_S matrix range is not aligned to one expert span".to_string());
        }
        let expert = relative / expert_stride;
        if expert >= bound.source.identity.ne[2] {
            return Err("IQ1_S expert coordinate is outside the registered tensor".to_string());
        }
        Ok(ResolvedIq1sWeight {
            identity: bound.source.identity.clone(),
            expert,
            allocation_generation: bound.binding.allocation_generation,
            content_sha256: bound.source.identity.content_sha256,
        })
    }

    pub(crate) fn resolve_launch(
        &self,
        matrix_ptr: usize,
        matrix_bytes: u64,
        ncols: u64,
        nrows: u64,
        row_stride_bytes: u64,
    ) -> Result<ResolvedIq1sWeight, String> {
        let resolved = self.lookup(matrix_ptr, matrix_bytes)?;
        if matrix_bytes != resolved.identity.nb[2] {
            return Err(format!(
                "IQ1_S launch matrix bytes {matrix_bytes} do not equal registered expert span {}",
                resolved.identity.nb[2]
            ));
        }
        if ncols != resolved.identity.ne[0]
            || nrows != resolved.identity.ne[1]
            || row_stride_bytes != resolved.identity.nb[1]
        {
            return Err(format!(
                "IQ1_S launch layout ({ncols}, {nrows}, {row_stride_bytes}) does not match registered layout ({}, {}, {})",
                resolved.identity.ne[0], resolved.identity.ne[1], resolved.identity.nb[1]
            ));
        }
        Ok(resolved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{c_void, CString};
    use std::io::{Seek, SeekFrom, Write};

    fn registration(
        file: &tempfile::NamedTempFile,
        name: &str,
        role: Iq1sExpertRole,
    ) -> Iq1sTensorRegistration {
        Iq1sTensorRegistration {
            path: file.path().to_path_buf(),
            file_offset: 0,
            nbytes: 200,
            name: name.to_string(),
            ne: [256, 2, 2, 1],
            nb: [50, 50, 100, 200],
            role,
            ggml_type: 19,
            model_sha256: [0x11; 32],
        }
    }

    fn tensor_file(fill: u8) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&vec![fill; 200]).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn registers_iq1s_tensor_and_resolves_expert_from_bound_device_range() {
        let file = tensor_file(0x5a);
        let registry = Iq1sWeightRegistry::default();
        let source = registry
            .register(registration(
                &file,
                "blk.7.ffn_down_exps.weight",
                Iq1sExpertRole::Down,
            ))
            .unwrap();
        assert_ne!(source.identity.content_sha256, [0; 32]);

        registry
            .bind(Iq1sDeviceBinding {
                name: source.identity.name.clone(),
                base_ptr: 0x10_000,
                allocation_bytes: 200,
                allocation_generation: 7,
            })
            .unwrap();
        let resolved = registry.lookup(0x10_000 + 100, 100).unwrap();

        assert_eq!(resolved.identity.name, "blk.7.ffn_down_exps.weight");
        assert_eq!(resolved.identity.layer, 7);
        assert_eq!(resolved.identity.role, Iq1sExpertRole::Down);
        assert_eq!(resolved.expert, 1);
        assert_eq!(resolved.allocation_generation, 7);
        assert_eq!(resolved.content_sha256, source.identity.content_sha256);
    }

    #[test]
    fn reports_exact_registered_iq1s_roles_per_layer() {
        let gate_file = tensor_file(0x31);
        let down_file = tensor_file(0x32);
        let registry = Iq1sWeightRegistry::default();
        registry
            .register(registration(
                &gate_file,
                "blk.7.ffn_gate_exps.weight",
                Iq1sExpertRole::Gate,
            ))
            .unwrap();
        registry
            .register(registration(
                &down_file,
                "blk.7.ffn_down_exps.weight",
                Iq1sExpertRole::Down,
            ))
            .unwrap();

        assert_eq!(
            registry.expected_roles_for_layer(7).unwrap(),
            [Iq1sExpertRole::Gate, Iq1sExpertRole::Down]
                .into_iter()
                .collect()
        );
        assert!(registry.expected_roles_for_layer(8).unwrap().is_empty());
    }

    #[test]
    fn registered_sources_are_immutable_and_sorted_by_layer_role_and_name() {
        let files = [tensor_file(0x31), tensor_file(0x32), tensor_file(0x33)];
        let registry = Iq1sWeightRegistry::default();
        for (file, name, role) in [
            (
                &files[0],
                "blk.9.ffn_down_exps.weight",
                Iq1sExpertRole::Down,
            ),
            (&files[1], "blk.2.ffn_up_exps.weight", Iq1sExpertRole::Up),
            (
                &files[2],
                "blk.2.ffn_gate_exps.weight",
                Iq1sExpertRole::Gate,
            ),
        ] {
            registry.register(registration(file, name, role)).unwrap();
        }

        let sources = registry.registered_sources().unwrap();
        let names = sources
            .iter()
            .map(|source| source.identity.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                "blk.2.ffn_gate_exps.weight",
                "blk.2.ffn_up_exps.weight",
                "blk.9.ffn_down_exps.weight",
            ]
        );
        assert!(Arc::strong_count(&sources[0]) >= 2);
    }

    #[test]
    fn registration_is_idempotent_but_rejects_changed_content_and_identity() {
        let mut file = tensor_file(0x31);
        let registry = Iq1sWeightRegistry::default();
        let first = registry
            .register(registration(
                &file,
                "blk.2.ffn_gate_exps.weight",
                Iq1sExpertRole::Gate,
            ))
            .unwrap();
        let same = registry
            .register(registration(
                &file,
                "blk.2.ffn_gate_exps.weight",
                Iq1sExpertRole::Gate,
            ))
            .unwrap();
        assert_eq!(first.identity, same.identity);

        file.as_file_mut().seek(SeekFrom::Start(0)).unwrap();
        file.as_file_mut().write_all(&[0x99]).unwrap();
        file.as_file_mut().flush().unwrap();
        let error = registry
            .register(registration(
                &file,
                "blk.2.ffn_gate_exps.weight",
                Iq1sExpertRole::Gate,
            ))
            .unwrap_err();
        assert!(error.contains("conflicting IQ1_S registration"));
    }

    #[test]
    fn rejects_wrong_type_layout_role_name_range_and_model_hash() {
        let file = tensor_file(0x42);
        let registry = Iq1sWeightRegistry::default();
        let base = registration(&file, "blk.1.ffn_up_exps.weight", Iq1sExpertRole::Up);

        let mut wrong_name = base.clone();
        wrong_name.name = "token_embd.weight".to_string();
        assert!(registry.register(wrong_name).unwrap_err().contains("name"));

        let mut wrong_type = base.clone();
        wrong_type.ggml_type = 34;
        assert!(registry
            .register(wrong_type)
            .unwrap_err()
            .contains("type 19"));

        let mut wrong_role = base.clone();
        wrong_role.role = Iq1sExpertRole::Down;
        assert!(registry.register(wrong_role).unwrap_err().contains("role"));

        let mut wrong_block = base.clone();
        wrong_block.nb[0] = 54;
        assert!(registry.register(wrong_block).unwrap_err().contains("50"));

        let mut short = base.clone();
        short.nbytes = 199;
        assert!(registry.register(short).unwrap_err().contains("span"));

        let mut overflow = base.clone();
        overflow.ne[2] = u64::MAX;
        assert!(registry
            .register(overflow)
            .unwrap_err()
            .contains("overflow"));

        let mut zero_model = base;
        zero_model.model_sha256 = [0; 32];
        assert!(registry
            .register(zero_model)
            .unwrap_err()
            .contains("model SHA-256"));
    }

    #[test]
    fn rejects_overlapping_binding_stale_generation_and_out_of_range_lookup() {
        let first_file = tensor_file(0x18);
        let second_file = tensor_file(0x29);
        let registry = Iq1sWeightRegistry::default();
        registry
            .register(registration(
                &first_file,
                "blk.3.ffn_gate_exps.weight",
                Iq1sExpertRole::Gate,
            ))
            .unwrap();
        registry
            .register(registration(
                &second_file,
                "blk.3.ffn_up_exps.weight",
                Iq1sExpertRole::Up,
            ))
            .unwrap();
        registry
            .bind(Iq1sDeviceBinding {
                name: "blk.3.ffn_gate_exps.weight".to_string(),
                base_ptr: 0x20_000,
                allocation_bytes: 200,
                allocation_generation: 9,
            })
            .unwrap();

        let overlap = registry
            .bind(Iq1sDeviceBinding {
                name: "blk.3.ffn_up_exps.weight".to_string(),
                base_ptr: 0x20_080,
                allocation_bytes: 200,
                allocation_generation: 10,
            })
            .unwrap_err();
        assert!(overlap.contains("overlaps"));

        let stale = registry
            .bind(Iq1sDeviceBinding {
                name: "blk.3.ffn_gate_exps.weight".to_string(),
                base_ptr: 0x20_000,
                allocation_bytes: 200,
                allocation_generation: 8,
            })
            .unwrap_err();
        assert!(stale.contains("stale"));
        assert!(registry
            .lookup(0x20_000 + 150, 100)
            .unwrap_err()
            .contains("range"));
        assert!(registry
            .lookup(0x30_000, 50)
            .unwrap_err()
            .contains("registered"));
    }

    #[test]
    fn classifies_all_supported_qwen_expert_projection_names() {
        assert_eq!(
            classify_tensor_name("blk.0.ffn_gate_exps.weight").unwrap(),
            (0, Iq1sExpertRole::Gate)
        );
        assert_eq!(
            classify_tensor_name("blk.59.ffn_up_exps.weight").unwrap(),
            (59, Iq1sExpertRole::Up)
        );
        assert_eq!(
            classify_tensor_name("blk.12.ffn_down_exps.weight").unwrap(),
            (12, Iq1sExpertRole::Down)
        );
        assert_eq!(
            classify_tensor_name("blk.4.ffn_gate_up_exps.weight").unwrap(),
            (4, Iq1sExpertRole::GateUp)
        );
        assert!(classify_tensor_name("blk.60.attn_q.weight").is_err());
    }

    #[test]
    fn launch_resolution_requires_exact_registered_expert_layout() {
        let file = tensor_file(0x27);
        let registry = Iq1sWeightRegistry::default();
        registry
            .register(registration(
                &file,
                "blk.9.ffn_gate_exps.weight",
                Iq1sExpertRole::Gate,
            ))
            .unwrap();
        registry
            .bind(Iq1sDeviceBinding {
                name: "blk.9.ffn_gate_exps.weight".to_string(),
                base_ptr: 0x40_000,
                allocation_bytes: 200,
                allocation_generation: 12,
            })
            .unwrap();

        let resolved = registry
            .resolve_launch(0x40_000 + 100, 100, 256, 2, 50)
            .unwrap();
        assert_eq!(resolved.expert, 1);
        assert_eq!(resolved.allocation_generation, 12);

        for (ncols, nrows, row_stride) in [(512, 2, 50), (256, 3, 50), (256, 2, 100)] {
            assert!(registry
                .resolve_launch(0x40_000 + 100, 100, ncols, nrows, row_stride)
                .unwrap_err()
                .contains("layout"));
        }
        assert!(registry
            .resolve_launch(0x40_000 + 100, 50, 256, 2, 50)
            .unwrap_err()
            .contains("expert span"));
    }

    #[test]
    fn iq1s_ffi_converts_versioned_tensor_and_binding_records() {
        let file = tensor_file(0x6b);
        let name = CString::new("blk.7.ffn_down_exps.weight").unwrap();
        let path = CString::new(file.path().as_os_str().as_encoded_bytes()).unwrap();
        let tensor = HetgpuIq1sTensorV1 {
            abi_version: HETGPU_IQ1S_ABI_VERSION,
            ggml_type: IQ1S_GGML_TYPE,
            role: HETGPU_IQ1S_ROLE_DOWN_EXPS,
            file_index: 3,
            name: name.as_ptr(),
            path: path.as_ptr(),
            file_offset: 0,
            nbytes: 200,
            ne: [256, 2, 2, 1],
            nb: [50, 50, 100, 200],
        };
        let converted = unsafe { registration_from_ffi(&tensor, [0x44; 32]) }.unwrap();
        assert_eq!(converted.name, "blk.7.ffn_down_exps.weight");
        assert_eq!(converted.path, file.path());
        assert_eq!(converted.role, Iq1sExpertRole::Down);
        assert_eq!(converted.ne, [256, 2, 2, 1]);

        let binding = HetgpuIq1sDeviceBindingV1 {
            abi_version: HETGPU_IQ1S_ABI_VERSION,
            reserved: 0,
            name: name.as_ptr(),
            device_base: 0x10_000usize as *const c_void,
            allocation_bytes: 200,
            allocation_generation: 9,
        };
        let converted = unsafe { binding_from_ffi(&binding) }.unwrap();
        assert_eq!(converted.name, "blk.7.ffn_down_exps.weight");
        assert_eq!(converted.base_ptr, 0x10_000);
        assert_eq!(converted.allocation_generation, 9);
    }

    #[test]
    fn iq1s_ffi_rejects_null_bad_version_negative_shape_and_reserved_bits() {
        assert!(
            unsafe { registration_from_ffi(std::ptr::null(), [0x44; 32]) }
                .unwrap_err()
                .contains("null")
        );
        assert!(unsafe { binding_from_ffi(std::ptr::null()) }
            .unwrap_err()
            .contains("null"));

        let file = tensor_file(0x73);
        let name = CString::new("blk.1.ffn_up_exps.weight").unwrap();
        let path = CString::new(file.path().as_os_str().as_encoded_bytes()).unwrap();
        let mut tensor = HetgpuIq1sTensorV1 {
            abi_version: HETGPU_IQ1S_ABI_VERSION + 1,
            ggml_type: IQ1S_GGML_TYPE,
            role: HETGPU_IQ1S_ROLE_UP_EXPS,
            file_index: 0,
            name: name.as_ptr(),
            path: path.as_ptr(),
            file_offset: 0,
            nbytes: 200,
            ne: [256, 2, 2, 1],
            nb: [50, 50, 100, 200],
        };
        assert!(unsafe { registration_from_ffi(&tensor, [0x44; 32]) }
            .unwrap_err()
            .contains("ABI version"));
        tensor.abi_version = HETGPU_IQ1S_ABI_VERSION;
        tensor.ne[0] = -1;
        assert!(unsafe { registration_from_ffi(&tensor, [0x44; 32]) }
            .unwrap_err()
            .contains("negative"));

        let binding = HetgpuIq1sDeviceBindingV1 {
            abi_version: HETGPU_IQ1S_ABI_VERSION,
            reserved: 1,
            name: name.as_ptr(),
            device_base: 0x10_000usize as *const c_void,
            allocation_bytes: 200,
            allocation_generation: 9,
        };
        assert!(unsafe { binding_from_ffi(&binding) }
            .unwrap_err()
            .contains("reserved"));
    }

    #[test]
    fn model_sha256_parser_requires_exact_nonzero_hex_digest() {
        assert_eq!(parse_model_sha256(&"11".repeat(32)).unwrap(), [0x11; 32]);
        assert!(parse_model_sha256("").is_err());
        assert!(parse_model_sha256(&"1".repeat(63)).is_err());
        assert!(parse_model_sha256(&"gg".repeat(32)).is_err());
        assert!(parse_model_sha256(&"00".repeat(32)).is_err());
    }
}
