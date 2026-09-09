use super::iq1s_layer_abi::{
    iq1s_command_crc32, Iq1sCommand, Iq1sCompletion, IQ1S_ABI_VERSION, IQ1S_COMMAND_BYTES,
    IQ1S_COMPLETION_BYTES, IQ1S_COMPLETION_MAGIC, IQ1S_COMPLETION_STATUS_OK, IQ1S_FAULT_CODE_NONE,
    IQ1S_REGISTER_MAGIC, IQ1S_REG_ABI_MAGIC_OFFSET, IQ1S_REG_ABI_VERSION_OFFSET,
    IQ1S_REG_ACTIVATION_BASE_HI_OFFSET, IQ1S_REG_ACTIVATION_BASE_LO_OFFSET,
    IQ1S_REG_ACTIVATION_BYTES_OFFSET, IQ1S_REG_ARENA_MANIFEST_BASE_HI_OFFSET,
    IQ1S_REG_ARENA_MANIFEST_BASE_LO_OFFSET, IQ1S_REG_ARENA_MANIFEST_BYTES_OFFSET,
    IQ1S_REG_COMMAND_BASE_HI_OFFSET, IQ1S_REG_COMMAND_BASE_LO_OFFSET,
    IQ1S_REG_COMMAND_CAPACITY_OFFSET, IQ1S_REG_COMMAND_CONSUMER_OFFSET,
    IQ1S_REG_COMMAND_PRODUCER_OFFSET, IQ1S_REG_COMPLETION_BASE_HI_OFFSET,
    IQ1S_REG_COMPLETION_BASE_LO_OFFSET, IQ1S_REG_COMPLETION_CAPACITY_OFFSET,
    IQ1S_REG_COMPLETION_CONSUMER_OFFSET, IQ1S_REG_COMPLETION_PRODUCER_OFFSET,
    IQ1S_REG_CONTROL_OFFSET, IQ1S_REG_CU_ID_OFFSET, IQ1S_REG_DOORBELL_OFFSET,
    IQ1S_REG_FAULT_CODE_OFFSET, IQ1S_REG_FAULT_DETAIL_HI_OFFSET,
    IQ1S_REG_FAULT_DETAIL_LO_OFFSET, IQ1S_REG_MODEL_TAG_HI_OFFSET,
    IQ1S_REG_MODEL_TAG_LO_OFFSET, IQ1S_REG_PROGRAM_BASE_HI_OFFSET,
    IQ1S_REG_PROGRAM_BASE_LO_OFFSET,
    IQ1S_REG_PROGRAM_BYTES_OFFSET, IQ1S_REG_QUIESCENT_OFFSET, IQ1S_REG_RESULT_BASE_HI_OFFSET,
    IQ1S_REG_RESULT_BASE_LO_OFFSET, IQ1S_REG_RESULT_BYTES_OFFSET,
    IQ1S_REG_SESSION_GENERATION_HI_OFFSET, IQ1S_REG_SESSION_GENERATION_LO_OFFSET,
    IQ1S_REG_TOKEN_MAP_BASE_HI_OFFSET, IQ1S_REG_TOKEN_MAP_BASE_LO_OFFSET,
    IQ1S_REG_TOKEN_MAP_BYTES_OFFSET, IQ1S_ROLE_DOWN, IQ1S_ROLE_GATE, IQ1S_ROLE_UP,
};
use super::iq1s_layer_trace::{
    validate_compiled_layer_phase, CompiledLayerPhase, ExpandedIq1sCounts,
};
use super::iq1s_trace::QWEN_MODEL_CONTEXT_LIMIT;
#[cfg(test)]
use super::iq1s_weight_arena::ARENA_SUPERBLOCK_BYTES;
use super::iq1s_weight_arena::{ARENA_ALIGNMENT, ARENA_BANK_COUNT, ARENA_STAGING_CHUNK_BYTES};
use super::xrt_tmatmul::{
    Handle, XrtOps, Xuid, XRT_BO_FLAGS_DEVICE_ONLY, XRT_BO_SYNC_FROM_DEVICE, XRT_BO_SYNC_TO_DEVICE,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::ffi::CString;
use std::fmt;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const CONTROL_START: u32 = 1;
const CONTROL_SHUTDOWN: u32 = 2;
const CONTROL_FAULT_RESET: u32 = 4;
const COMMAND_RING_DEFAULT_CAPACITY: u32 = 512;
const PROGRAM_BYTES: usize = 4 * 1024 * 1024;
const ARENA_MANIFEST_BYTES: usize = 8 * 1024 * 1024;
const ARENA_MANIFEST_MAGIC: u32 = 0x4d41_5149;
const ARENA_MANIFEST_RECORD_BYTES: usize = 64;
#[cfg(not(test))]
const ACTIVATION_BYTES: usize = 256 * 1024 * 1024;
#[cfg(test)]
const ACTIVATION_BYTES: usize = 8 * 1024 * 1024;
#[cfg(not(test))]
const OUTPUT_BYTES: usize = 256 * 1024 * 1024;
#[cfg(test)]
const OUTPUT_BYTES: usize = 8 * 1024 * 1024;
#[cfg(not(test))]
const TOKEN_MAP_BYTES: usize = 16 * 1024 * 1024;
#[cfg(test)]
const TOKEN_MAP_BYTES: usize = 2 * 1024 * 1024;
const TICKET_SLOT_COUNT: usize = 2;
const ACTIVATION_SLOT_BYTES: usize = ACTIVATION_BYTES / TICKET_SLOT_COUNT;
const OUTPUT_SLOT_BYTES: usize = OUTPUT_BYTES / TICKET_SLOT_COUNT;
const TOKEN_MAP_SLOT_BYTES: usize = TOKEN_MAP_BYTES / TICKET_SLOT_COUNT;
const MAX_BACKOFF_US: u64 = 1_000;
const MEMORY_GROUPS: [u32; ARENA_BANK_COUNT] = [0, 3, 2, 1];
const IP_NAMES: [&str; ARENA_BANK_COUNT] = [
    "iq1s_layer_big:iq1s_layer_big_1",
    "iq1s_layer_big:iq1s_layer_big_2",
    "iq1s_layer_big:iq1s_layer_big_3",
    "iq1s_layer_small:iq1s_layer_small_1",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PersistentIq1sConfig {
    pub(crate) xclbin: PathBuf,
    pub(crate) device_index: u32,
    pub(crate) command_capacity: u32,
    pub(crate) timeout_ms: u32,
}

impl PersistentIq1sConfig {
    pub(crate) fn checked(
        xclbin: PathBuf,
        device_index: u32,
        command_capacity: Option<u32>,
        timeout_ms: u32,
    ) -> Result<Self, PersistentError> {
        let command_capacity = command_capacity.unwrap_or(COMMAND_RING_DEFAULT_CAPACITY);
        if xclbin.as_os_str().is_empty()
            || command_capacity == 0
            || !command_capacity.is_power_of_two()
            || timeout_ms == 0
        {
            return Err(PersistentError::Config(
                "xclbin, power-of-two ring capacity, and timeout must be valid".to_string(),
            ));
        }
        Ok(Self {
            xclbin,
            device_index,
            command_capacity,
            timeout_ms,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ArenaShardSpec {
    pub(crate) bank: u8,
    pub(crate) logical_offset: u64,
    pub(crate) bytes: usize,
    pub(crate) layer_id: u32,
    pub(crate) role: u16,
    pub(crate) expert_id: u16,
    pub(crate) row_start: u32,
    pub(crate) row_count: u32,
    pub(crate) sha256: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ArenaChunkSpec {
    pub(crate) bank: u8,
    pub(crate) logical_offset: u64,
    pub(crate) bytes: usize,
    pub(crate) sha256: [u8; 32],
    pub(crate) shards: Vec<ArenaShardSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PersistentFault {
    pub(crate) cu: Option<usize>,
    pub(crate) operation: &'static str,
    pub(crate) detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PersistentError {
    Config(String),
    Xrt {
        operation: &'static str,
        code: i32,
    },
    NullHandle(&'static str),
    InvalidPhase(String),
    RingFull {
        cu: usize,
        capacity: u32,
    },
    TicketSlotsFull,
    Timeout {
        operation: &'static str,
        timeout_ms: u32,
    },
    Fault(PersistentFault),
    Poisoned(PersistentFault),
    Shutdown(String),
}

impl fmt::Display for PersistentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(message) => write!(formatter, "persistent IQ1_S configuration: {message}"),
            Self::Xrt { operation, code } => {
                write!(
                    formatter,
                    "persistent IQ1_S XRT {operation} failed with code {code}"
                )
            }
            Self::NullHandle(operation) => {
                write!(formatter, "persistent IQ1_S XRT {operation} returned null")
            }
            Self::InvalidPhase(message) => write!(formatter, "invalid IQ1_S phase: {message}"),
            Self::RingFull { cu, capacity } => {
                write!(
                    formatter,
                    "IQ1_S CU {cu} command ring capacity {capacity} is full"
                )
            }
            Self::TicketSlotsFull => {
                write!(formatter, "both persistent IQ1_S ticket slots are busy")
            }
            Self::Timeout {
                operation,
                timeout_ms,
            } => write!(
                formatter,
                "persistent IQ1_S {operation} timed out after {timeout_ms} ms"
            ),
            Self::Fault(fault) => write!(
                formatter,
                "persistent IQ1_S fault during {} on {:?}: {}",
                fault.operation, fault.cu, fault.detail
            ),
            Self::Poisoned(fault) => write!(
                formatter,
                "persistent IQ1_S pool is poisoned by {} on {:?}: {}",
                fault.operation, fault.cu, fault.detail
            ),
            Self::Shutdown(message) => write!(formatter, "persistent IQ1_S shutdown: {message}"),
        }
    }
}

impl std::error::Error for PersistentError {}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PersistentDmaCounters {
    pub(crate) command_ranges: u64,
    pub(crate) activation_ranges: u64,
    pub(crate) result_ranges: u64,
    pub(crate) program_ranges: u64,
    pub(crate) weight_ranges: u64,
    pub(crate) weight_bytes: u64,
}

impl PersistentDmaCounters {
    fn checked_delta(self, baseline: Self) -> Result<Self, PersistentError> {
        let subtract = |value: u64, before: u64| {
            value.checked_sub(before).ok_or_else(|| {
                PersistentError::Config("persistent DMA counters regressed".to_string())
            })
        };
        Ok(Self {
            command_ranges: subtract(self.command_ranges, baseline.command_ranges)?,
            activation_ranges: subtract(self.activation_ranges, baseline.activation_ranges)?,
            result_ranges: subtract(self.result_ranges, baseline.result_ranges)?,
            program_ranges: subtract(self.program_ranges, baseline.program_ranges)?,
            weight_ranges: subtract(self.weight_ranges, baseline.weight_ranges)?,
            weight_bytes: subtract(self.weight_bytes, baseline.weight_bytes)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HostRange {
    pub(crate) offset: u64,
    pub(crate) bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PhaseBuffers {
    pub(crate) activations: Vec<HostRange>,
    pub(crate) token_maps: Vec<HostRange>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PersistentPhaseTimings {
    pub(crate) activation_sync_us: u64,
    pub(crate) ring_publish_us: u64,
    pub(crate) doorbell_us: u64,
    pub(crate) device_wait_us: u64,
    pub(crate) completion_sync_us: u64,
    pub(crate) result_copy_us: u64,
    pub(crate) phase_wall_us: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompletedLayerPhase {
    pub(crate) transaction_id: u64,
    pub(crate) semantic_sha256: [u8; 32],
    pub(crate) completions: [Vec<Iq1sCompletion>; ARENA_BANK_COUNT],
    pub(crate) results: [Vec<HostRange>; ARENA_BANK_COUNT],
    pub(crate) expanded: ExpandedIq1sCounts,
    pub(crate) dma: PersistentDmaCounters,
    pub(crate) timings: PersistentPhaseTimings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Free,
    Prepared,
    Published,
    Complete,
    Poisoned,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SubmissionTicket {
    slot: u8,
    slot_generation: u64,
    transaction_id: u64,
    command_end: [u32; ARENA_BANK_COUNT],
    expected_completion: [u32; ARENA_BANK_COUNT],
    deadline: Instant,
}

impl SubmissionTicket {
    pub(crate) fn slot(&self) -> u8 {
        self.slot
    }

    pub(crate) fn slot_generation(&self) -> u64 {
        self.slot_generation
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TicketPoll {
    Pending,
    Complete,
}

#[derive(Debug)]
struct PreparedTicket {
    phase: CompiledLayerPhase,
    expected: [Vec<Iq1sCommand>; ARENA_BANK_COUNT],
    result_ranges: [Vec<(usize, usize, usize)>; ARENA_BANK_COUNT],
    dma: PersistentDmaCounters,
    timings: PersistentPhaseTimings,
    phase_wall_start: Instant,
}

#[derive(Debug)]
struct TicketSlot {
    generation: u64,
    state: SlotState,
    prepared: Option<PreparedTicket>,
}

impl Default for TicketSlot {
    fn default() -> Self {
        Self {
            generation: 0,
            state: SlotState::Free,
            prepared: None,
        }
    }
}

#[derive(Debug)]
struct ArenaChunk {
    logical_offset: u64,
    bytes: usize,
    bo: Handle,
    address: u64,
    sha256: [u8; 32],
    shards: Vec<ResidentArenaShard>,
}

#[derive(Debug)]
struct ResidentArenaShard {
    address: u64,
    bytes: usize,
    layer_id: u32,
    role: u16,
    expert_id: u16,
    row_start: u32,
    row_count: u32,
    sha256: [u8; 32],
}

#[derive(Debug)]
struct PersistentCu {
    ip_index: u32,
    command_bo: Handle,
    completion_bo: Handle,
    program_bo: Handle,
    arena_manifest_bo: Handle,
    activation_bo: Handle,
    output_bo: Handle,
    token_map_bo: Handle,
    arena_staging_bo: Handle,
    command_address: u64,
    completion_address: u64,
    program_address: u64,
    arena_manifest_address: u64,
    activation_address: u64,
    output_address: u64,
    token_map_address: u64,
    arena: Vec<ArenaChunk>,
    command_shadow: Vec<u8>,
    completion_shadow: Vec<u8>,
    program_shadow: Vec<u8>,
    command_producer: u32,
    command_published: u32,
    command_consumer: u32,
    completion_consumer: u32,
    completion_reserved: u32,
    cached_program_id: Option<u64>,
}

impl PersistentCu {
    fn runtime_bos(&self) -> [Handle; 8] {
        [
            self.command_bo,
            self.completion_bo,
            self.program_bo,
            self.arena_manifest_bo,
            self.activation_bo,
            self.output_bo,
            self.token_map_bo,
            self.arena_staging_bo,
        ]
    }
}

pub(crate) struct PersistentIq1sPool<O: XrtOps> {
    ops: O,
    device: Handle,
    native_device: Handle,
    xclbin_uuid: Xuid,
    generation: u64,
    model_tag: u64,
    config: PersistentIq1sConfig,
    cus: [PersistentCu; ARENA_BANK_COUNT],
    poisoned: Option<PersistentFault>,
    measured: bool,
    measurement_baseline: PersistentDmaCounters,
    dma: PersistentDmaCounters,
    ticket_slots: [TicketSlot; TICKET_SLOT_COUNT],
    next_poll_cu: usize,
    closed: bool,
}

fn checked_code(operation: &'static str, code: i32) -> Result<(), PersistentError> {
    if code == 0 {
        Ok(())
    } else {
        Err(PersistentError::Xrt { operation, code })
    }
}

fn checked_handle(operation: &'static str, handle: Handle) -> Result<Handle, PersistentError> {
    if handle.is_null() {
        Err(PersistentError::NullHandle(operation))
    } else {
        Ok(handle)
    }
}

fn split_u64(value: u64) -> (u32, u32) {
    (value as u32, (value >> 32) as u32)
}

fn struct_bytes<T>(value: &T) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts((value as *const T).cast::<u8>(), std::mem::size_of::<T>())
    }
}

fn struct_from_bytes<T: Copy>(bytes: &[u8]) -> Result<T, PersistentError> {
    if bytes.len() != std::mem::size_of::<T>() {
        return Err(PersistentError::InvalidPhase(
            "record byte length does not match ABI".to_string(),
        ));
    }
    let mut value = std::mem::MaybeUninit::<T>::uninit();
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), value.as_mut_ptr().cast::<u8>(), bytes.len());
        Ok(value.assume_init())
    }
}

fn command_crc(command: &Iq1sCommand) -> u32 {
    let mut bytes = [0u8; IQ1S_COMMAND_BYTES];
    bytes.copy_from_slice(struct_bytes(command));
    iq1s_command_crc32(&bytes)
}

fn merge_adjacent_ranges(
    ranges: impl IntoIterator<Item = (usize, usize)>,
    name: &str,
) -> Result<Vec<(usize, usize)>, PersistentError> {
    let mut ranges = ranges
        .into_iter()
        .filter(|(_, bytes)| *bytes != 0)
        .collect::<Vec<_>>();
    ranges.sort_unstable_by_key(|(offset, _)| *offset);
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (offset, bytes) in ranges {
        let end = offset.checked_add(bytes).ok_or_else(|| {
            PersistentError::InvalidPhase(format!("{name} range overflows usize"))
        })?;
        if let Some((last_offset, last_bytes)) = merged.last_mut() {
            let last_end = last_offset.checked_add(*last_bytes).ok_or_else(|| {
                PersistentError::InvalidPhase(format!("{name} range overflows usize"))
            })?;
            if offset < last_end {
                return Err(PersistentError::InvalidPhase(format!(
                    "{name} ranges overlap"
                )));
            }
            if offset == last_end {
                *last_bytes = end - *last_offset;
                continue;
            }
        }
        merged.push((offset, bytes));
    }
    Ok(merged)
}

fn validate_host_ranges(
    ranges: &[HostRange],
    capacity: usize,
    name: &str,
) -> Result<Vec<(usize, usize)>, PersistentError> {
    if ranges.is_empty() || ranges.iter().any(|range| range.bytes.is_empty()) {
        return Err(PersistentError::InvalidPhase(format!(
            "{name} host ranges are empty"
        )));
    }
    let checked = ranges
        .iter()
        .map(|range| {
            let offset = usize::try_from(range.offset).map_err(|_| {
                PersistentError::InvalidPhase(format!("{name} offset does not fit usize"))
            })?;
            let end = offset.checked_add(range.bytes.len()).ok_or_else(|| {
                PersistentError::InvalidPhase(format!("{name} host range overflows"))
            })?;
            if end > capacity {
                return Err(PersistentError::InvalidPhase(format!(
                    "{name} host range exceeds its slab"
                )));
            }
            Ok((offset, range.bytes.len()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    merge_adjacent_ranges(checked, name)
}

fn coalesce_host_ranges(
    ranges: &[HostRange],
    capacity: usize,
    name: &str,
) -> Result<Vec<HostRange>, PersistentError> {
    let merged = validate_host_ranges(ranges, capacity, name)?;
    let mut coalesced = Vec::with_capacity(merged.len());
    for (offset, bytes) in merged {
        let mut packed = vec![0u8; bytes];
        for range in ranges {
            let source_offset = usize::try_from(range.offset).map_err(|_| {
                PersistentError::InvalidPhase(format!("{name} offset does not fit usize"))
            })?;
            if source_offset >= offset && source_offset < offset + bytes {
                let relative = source_offset - offset;
                packed[relative..relative + range.bytes.len()].copy_from_slice(&range.bytes);
            }
        }
        coalesced.push(HostRange {
            offset: offset as u64,
            bytes: packed,
        });
    }
    Ok(coalesced)
}

fn range_is_covered(ranges: &[(usize, usize)], offset: usize, bytes: usize) -> bool {
    offset.checked_add(bytes).is_some_and(|end| {
        ranges.iter().any(|(range_offset, range_bytes)| {
            range_offset
                .checked_add(*range_bytes)
                .is_some_and(|range_end| offset >= *range_offset && end <= range_end)
        })
    })
}

fn elapsed_us(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX)
}

fn resident_model_tag(chunks: &[ArenaChunkSpec]) -> u64 {
    let mut ordered = chunks.to_vec();
    ordered.sort_by_key(|chunk| (chunk.bank, chunk.logical_offset));
    let mut hash = Sha256::new();
    hash.update(b"hetgpu-qwen-iq1s-resident-model-v1\0");
    for chunk in ordered {
        hash.update([chunk.bank]);
        hash.update(chunk.logical_offset.to_le_bytes());
        hash.update((chunk.bytes as u64).to_le_bytes());
        hash.update(chunk.sha256);
        let mut shards = chunk.shards;
        shards.sort_by_key(|shard| {
            (
                shard.logical_offset,
                shard.layer_id,
                shard.role,
                shard.expert_id,
                shard.row_start,
            )
        });
        for shard in shards {
            hash.update(shard.logical_offset.to_le_bytes());
            hash.update((shard.bytes as u64).to_le_bytes());
            hash.update(shard.layer_id.to_le_bytes());
            hash.update(shard.role.to_le_bytes());
            hash.update(shard.expert_id.to_le_bytes());
            hash.update(shard.row_start.to_le_bytes());
            hash.update(shard.row_count.to_le_bytes());
            hash.update(shard.sha256);
        }
    }
    u64::from_le_bytes(
        hash.finalize()[..8]
            .try_into()
            .expect("eight-byte model tag"),
    )
}

fn arena_manifest(model_tag: u64, arena: &[ArenaChunk]) -> Result<Vec<u8>, PersistentError> {
    let record_count = arena
        .iter()
        .try_fold(0usize, |count, chunk| count.checked_add(chunk.shards.len()))
        .ok_or_else(|| {
            PersistentError::Config("arena manifest record count overflow".to_string())
        })?;
    let used = 64usize
        .checked_add(
            record_count
                .checked_mul(ARENA_MANIFEST_RECORD_BYTES)
                .ok_or_else(|| {
                    PersistentError::Config("arena manifest length overflow".to_string())
                })?,
        )
        .ok_or_else(|| PersistentError::Config("arena manifest length overflow".to_string()))?;
    if used > ARENA_MANIFEST_BYTES {
        return Err(PersistentError::Config(
            "arena manifest exceeds its bank-local BO".to_string(),
        ));
    }
    let mut bytes = vec![0u8; ARENA_MANIFEST_BYTES];
    bytes[0..4].copy_from_slice(&ARENA_MANIFEST_MAGIC.to_le_bytes());
    bytes[4..8].copy_from_slice(&IQ1S_ABI_VERSION.to_le_bytes());
    let record_count = u32::try_from(record_count).map_err(|_| {
        PersistentError::Config("arena manifest record count exceeds u32".to_string())
    })?;
    bytes[8..12].copy_from_slice(&record_count.to_le_bytes());
    bytes[12..16].copy_from_slice(&(ARENA_MANIFEST_RECORD_BYTES as u32).to_le_bytes());
    bytes[16..24].copy_from_slice(&model_tag.to_le_bytes());
    for (index, shard) in arena
        .iter()
        .flat_map(|chunk| chunk.shards.iter())
        .enumerate()
    {
        let offset = 64 + index * ARENA_MANIFEST_RECORD_BYTES;
        bytes[offset..offset + 8].copy_from_slice(&shard.address.to_le_bytes());
        bytes[offset + 8..offset + 16].copy_from_slice(&(shard.bytes as u64).to_le_bytes());
        bytes[offset + 16..offset + 20].copy_from_slice(&shard.layer_id.to_le_bytes());
        bytes[offset + 20..offset + 22].copy_from_slice(&shard.role.to_le_bytes());
        bytes[offset + 22..offset + 24].copy_from_slice(&shard.expert_id.to_le_bytes());
        bytes[offset + 24..offset + 28].copy_from_slice(&shard.row_start.to_le_bytes());
        bytes[offset + 28..offset + 32].copy_from_slice(&shard.row_count.to_le_bytes());
        bytes[offset + 32..offset + 64].copy_from_slice(&shard.sha256);
    }
    Ok(bytes)
}

impl<O: XrtOps> PersistentIq1sPool<O> {
    pub(crate) fn open(
        ops: O,
        config: PersistentIq1sConfig,
        generation: u64,
        chunks: &[ArenaChunkSpec],
        mut read_chunk: impl FnMut(&ArenaChunkSpec) -> Result<Vec<u8>, String>,
        mut chunk_resident: impl FnMut(&ArenaChunkSpec) -> Result<(), String>,
    ) -> Result<Self, PersistentError> {
        if generation == 0 {
            return Err(PersistentError::Config(
                "session generation must be nonzero".to_string(),
            ));
        }
        let model_tag = resident_model_tag(chunks);
        if model_tag == 0 {
            return Err(PersistentError::Config(
                "resident model identity tag must be nonzero".to_string(),
            ));
        }
        let mut by_bank: [Vec<ArenaChunkSpec>; ARENA_BANK_COUNT] =
            std::array::from_fn(|_| Vec::new());
        let mut shard_identities = BTreeSet::new();
        for chunk in chunks {
            let bank = usize::from(chunk.bank);
            if bank >= ARENA_BANK_COUNT
                || chunk.bytes == 0
                || chunk.bytes as u64 > ARENA_STAGING_CHUNK_BYTES
                || chunk.logical_offset % ARENA_ALIGNMENT != 0
                || chunk.sha256 == [0; 32]
                || chunk.shards.is_empty()
            {
                return Err(PersistentError::Config(
                    "arena chunk violates bank, size, alignment, hash, or shard contract"
                        .to_string(),
                ));
            }
            let chunk_end = chunk
                .logical_offset
                .checked_add(chunk.bytes as u64)
                .ok_or_else(|| PersistentError::Config("arena chunk overflow".to_string()))?;
            let mut shard_ranges = Vec::new();
            for shard in &chunk.shards {
                let shard_end = shard
                    .logical_offset
                    .checked_add(shard.bytes as u64)
                    .ok_or_else(|| PersistentError::Config("arena shard overflow".to_string()))?;
                let (expected_rows, expected_bytes_per_row) = match u32::from(shard.role) {
                    IQ1S_ROLE_GATE | IQ1S_ROLE_UP => (256u32, 800usize),
                    IQ1S_ROLE_DOWN => (1024u32, 200usize),
                    _ => {
                        return Err(PersistentError::Config(
                            "arena shard has an unsupported role".to_string(),
                        ));
                    }
                };
                let expected_bytes = usize::try_from(expected_rows)
                    .ok()
                    .and_then(|rows| rows.checked_mul(expected_bytes_per_row));
                if shard.bank != chunk.bank
                    || shard.logical_offset % ARENA_ALIGNMENT != 0
                    || shard.logical_offset < chunk.logical_offset
                    || shard_end > chunk_end
                    || shard.bytes == 0
                    || Some(shard.bytes) != expected_bytes
                    || shard.expert_id >= 512
                    || shard.row_count != expected_rows
                    || shard.row_start != u32::from(shard.bank) * expected_rows
                    || shard.sha256 == [0; 32]
                    || !shard_identities.insert((
                        shard.layer_id,
                        shard.role,
                        shard.expert_id,
                        shard.row_start,
                        shard.row_count,
                    ))
                {
                    return Err(PersistentError::Config(
                        "arena shard violates exact identity, shape, bounds, alignment, or hash contract"
                            .to_string(),
                    ));
                }
                shard_ranges.push((shard.logical_offset, shard_end));
            }
            shard_ranges.sort_unstable();
            if shard_ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
                return Err(PersistentError::Config(
                    "arena shards overlap within a transfer chunk".to_string(),
                ));
            }
            by_bank[bank].push(chunk.clone());
        }
        if by_bank.iter().any(Vec::is_empty) {
            return Err(PersistentError::Config(
                "all four banks require at least one arena chunk".to_string(),
            ));
        }
        for bank in &mut by_bank {
            bank.sort_by_key(|chunk| chunk.logical_offset);
            for pair in bank.windows(2) {
                let end = pair[0]
                    .logical_offset
                    .checked_add(pair[0].bytes as u64)
                    .ok_or_else(|| PersistentError::Config("arena chunk overflow".to_string()))?;
                if end > pair[1].logical_offset {
                    return Err(PersistentError::Config(
                        "arena chunks overlap within a bank".to_string(),
                    ));
                }
            }
        }

        let xclbin = CString::new(config.xclbin.to_string_lossy().as_bytes()).map_err(|_| {
            PersistentError::Config("xclbin path contains an interior NUL".to_string())
        })?;
        let device = checked_handle("device_open", ops.device_open(config.device_index))?;
        checked_code("load_xclbin_file", ops.load_xclbin_file(device, &xclbin))?;
        let mut xclbin_uuid = [0u8; 16];
        checked_code(
            "get_xclbin_uuid",
            ops.get_xclbin_uuid(device, &mut xclbin_uuid),
        )?;
        if xclbin_uuid == [0; 16] {
            return Err(PersistentError::Config(
                "loaded xclbin returned a zero UUID".to_string(),
            ));
        }
        let native_device = checked_handle("xcl_open", ops.xcl_open(config.device_index))?;

        let mut ip_indices = [0u32; ARENA_BANK_COUNT];
        for (cu, name) in IP_NAMES.iter().enumerate() {
            let name = CString::new(*name).map_err(|_| {
                PersistentError::Config("persistent CU name contains NUL".to_string())
            })?;
            let index = ops.xcl_ip_name_to_index(native_device, &name);
            if index < 0 {
                return Err(PersistentError::Xrt {
                    operation: "xcl_ip_name_to_index",
                    code: index,
                });
            }
            ip_indices[cu] = index as u32;
            checked_code(
                "xcl_open_context",
                ops.xcl_open_context(native_device, &xclbin_uuid, index as u32, false),
            )?;
            let mut magic = 0u32;
            let mut version = 0u32;
            checked_code(
                "read ABI magic",
                ops.xcl_reg_read(
                    native_device,
                    index as u32,
                    IQ1S_REG_ABI_MAGIC_OFFSET as u32,
                    &mut magic,
                ),
            )?;
            checked_code(
                "read ABI version",
                ops.xcl_reg_read(
                    native_device,
                    index as u32,
                    IQ1S_REG_ABI_VERSION_OFFSET as u32,
                    &mut version,
                ),
            )?;
            if magic != IQ1S_REGISTER_MAGIC || version != IQ1S_ABI_VERSION {
                return Err(PersistentError::Config(format!(
                    "CU {cu} ABI is magic=0x{magic:08x} version={version}, expected 0x{IQ1S_REGISTER_MAGIC:08x}/{IQ1S_ABI_VERSION}"
                )));
            }
        }

        // The xclbin can remain programmed after the owning process exits, so
        // the read-only ring counters are not necessarily zero when a new host
        // runtime opens the CUs.  Stop every CU and rebind the writable side of
        // each ring to the hardware's monotonic counters before installing new
        // BO addresses.  Starting from zero against a persisted non-zero
        // consumer makes the unsigned occupancy wrap and immediately raises
        // IQ1S_FAULT_CODE_RING_OVERFLOW.
        let mut command_baselines = [0u32; ARENA_BANK_COUNT];
        let mut completion_baselines = [0u32; ARENA_BANK_COUNT];
        for cu in 0..ARENA_BANK_COUNT {
            checked_code(
                "stop persistent CU before ring rebind",
                ops.xcl_reg_write(
                    native_device,
                    ip_indices[cu],
                    IQ1S_REG_CONTROL_OFFSET as u32,
                    CONTROL_SHUTDOWN,
                ),
            )?;
        }
        let rebind_deadline =
            Instant::now() + Duration::from_millis(u64::from(config.timeout_ms));
        for cu in 0..ARENA_BANK_COUNT {
            loop {
                let mut command_consumer = 0u32;
                let mut completion_producer = 0u32;
                checked_code(
                    "read persisted command consumer",
                    ops.xcl_reg_read(
                        native_device,
                        ip_indices[cu],
                        IQ1S_REG_COMMAND_CONSUMER_OFFSET as u32,
                        &mut command_consumer,
                    ),
                )?;
                checked_code(
                    "read persisted completion producer",
                    ops.xcl_reg_read(
                        native_device,
                        ip_indices[cu],
                        IQ1S_REG_COMPLETION_PRODUCER_OFFSET as u32,
                        &mut completion_producer,
                    ),
                )?;
                checked_code(
                    "rebind command producer",
                    ops.xcl_reg_write(
                        native_device,
                        ip_indices[cu],
                        IQ1S_REG_COMMAND_PRODUCER_OFFSET as u32,
                        command_consumer,
                    ),
                )?;
                checked_code(
                    "rebind completion consumer",
                    ops.xcl_reg_write(
                        native_device,
                        ip_indices[cu],
                        IQ1S_REG_COMPLETION_CONSUMER_OFFSET as u32,
                        completion_producer,
                    ),
                )?;
                let mut quiescent = 0u32;
                checked_code(
                    "read quiescent during ring rebind",
                    ops.xcl_reg_read(
                        native_device,
                        ip_indices[cu],
                        IQ1S_REG_QUIESCENT_OFFSET as u32,
                        &mut quiescent,
                    ),
                )?;
                if quiescent == 1 {
                    command_baselines[cu] = command_consumer;
                    completion_baselines[cu] = completion_producer;
                    break;
                }
                if Instant::now() >= rebind_deadline {
                    return Err(PersistentError::Shutdown(format!(
                        "CU {cu} did not become quiescent during ring rebind"
                    )));
                }
                std::thread::yield_now();
            }

            let mut fault_code = 0u32;
            checked_code(
                "read fault before ring rebind reset",
                ops.xcl_reg_read(
                    native_device,
                    ip_indices[cu],
                    IQ1S_REG_FAULT_CODE_OFFSET as u32,
                    &mut fault_code,
                ),
            )?;
            if fault_code != IQ1S_FAULT_CODE_NONE {
                checked_code(
                    "reset fault after ring rebind",
                    ops.xcl_reg_write(
                        native_device,
                        ip_indices[cu],
                        IQ1S_REG_CONTROL_OFFSET as u32,
                        CONTROL_FAULT_RESET,
                    ),
                )?;
                loop {
                    checked_code(
                        "verify fault reset after ring rebind",
                        ops.xcl_reg_read(
                            native_device,
                            ip_indices[cu],
                            IQ1S_REG_FAULT_CODE_OFFSET as u32,
                            &mut fault_code,
                        ),
                    )?;
                    if fault_code == IQ1S_FAULT_CODE_NONE {
                        break;
                    }
                    if Instant::now() >= rebind_deadline {
                        return Err(PersistentError::Config(format!(
                            "CU {cu} fault {fault_code} did not clear after ring rebind"
                        )));
                    }
                    std::thread::yield_now();
                }
            }
        }

        let ring_bytes = usize::try_from(config.command_capacity)
            .ok()
            .and_then(|capacity| capacity.checked_mul(IQ1S_COMMAND_BYTES))
            .ok_or_else(|| PersistentError::Config("command ring size overflow".to_string()))?;
        let mut cus = Vec::with_capacity(ARENA_BANK_COUNT);
        let mut dma = PersistentDmaCounters::default();
        for cu in 0..ARENA_BANK_COUNT {
            let group = MEMORY_GROUPS[cu];
            let allocate = |bytes: usize, flags: u64| -> Result<Handle, PersistentError> {
                checked_handle("bo_alloc", ops.bo_alloc(device, bytes, flags, group))
            };
            let command_bo = allocate(ring_bytes, 0)?;
            let completion_bo = allocate(ring_bytes, 0)?;
            let program_bo = allocate(PROGRAM_BYTES, 0)?;
            let arena_manifest_bo = allocate(ARENA_MANIFEST_BYTES, 0)?;
            let activation_bo = allocate(ACTIVATION_BYTES, 0)?;
            let output_bo = allocate(OUTPUT_BYTES, 0)?;
            let token_map_bo = allocate(TOKEN_MAP_BYTES, 0)?;
            let arena_staging_bytes = by_bank[cu]
                .iter()
                .map(|chunk| chunk.bytes)
                .max()
                .ok_or_else(|| {
                    PersistentError::Config(format!("arena bank {cu} has no transfer chunks"))
                })?;
            let arena_staging_bo = allocate(arena_staging_bytes, 0)?;
            let mut arena = Vec::new();
            for spec in &by_bank[cu] {
                let bo = allocate(spec.bytes, XRT_BO_FLAGS_DEVICE_ONLY)?;
                let bytes = read_chunk(spec).map_err(PersistentError::Config)?;
                if bytes.len() != spec.bytes
                    || <[u8; 32]>::from(Sha256::digest(&bytes)) != spec.sha256
                {
                    return Err(PersistentError::Config(format!(
                        "arena chunk bank {} offset {} failed length/hash verification",
                        spec.bank, spec.logical_offset
                    )));
                }
                checked_code(
                    "write arena staging chunk",
                    ops.bo_write(arena_staging_bo, &bytes),
                )?;
                checked_code(
                    "sync arena staging chunk",
                    ops.bo_sync(arena_staging_bo, XRT_BO_SYNC_TO_DEVICE, bytes.len(), 0),
                )?;
                checked_code(
                    "copy arena chunk to device-only BO",
                    ops.bo_copy(bo, arena_staging_bo, bytes.len(), 0, 0),
                )?;
                dma.weight_ranges += 1;
                dma.weight_bytes = dma.weight_bytes.saturating_add(bytes.len() as u64);
                let address = ops.bo_address(bo);
                if address == 0 || address % ARENA_ALIGNMENT != 0 {
                    return Err(PersistentError::Config(
                        "arena BO device address is zero or unaligned".to_string(),
                    ));
                }
                let mut resident_shards = Vec::with_capacity(spec.shards.len());
                for shard in &spec.shards {
                    let relative = usize::try_from(shard.logical_offset - spec.logical_offset)
                        .map_err(|_| {
                            PersistentError::Config(
                                "arena shard relative offset exceeds usize".to_string(),
                            )
                        })?;
                    let end = relative.checked_add(shard.bytes).ok_or_else(|| {
                        PersistentError::Config("arena shard byte range overflow".to_string())
                    })?;
                    if end > bytes.len()
                        || <[u8; 32]>::from(Sha256::digest(&bytes[relative..end])) != shard.sha256
                    {
                        return Err(PersistentError::Config(format!(
                            "arena shard bank {} layer {} role {} expert {} failed length/hash verification",
                            shard.bank, shard.layer_id, shard.role, shard.expert_id
                        )));
                    }
                    let shard_address = address.checked_add(relative as u64).ok_or_else(|| {
                        PersistentError::Config("arena shard physical address overflow".to_string())
                    })?;
                    resident_shards.push(ResidentArenaShard {
                        address: shard_address,
                        bytes: shard.bytes,
                        layer_id: shard.layer_id,
                        role: shard.role,
                        expert_id: shard.expert_id,
                        row_start: shard.row_start,
                        row_count: shard.row_count,
                        sha256: shard.sha256,
                    });
                }
                arena.push(ArenaChunk {
                    logical_offset: spec.logical_offset,
                    bytes: spec.bytes,
                    bo,
                    address,
                    sha256: spec.sha256,
                    shards: resident_shards,
                });
                chunk_resident(spec).map_err(PersistentError::Config)?;
            }
            let manifest = arena_manifest(model_tag, &arena)?;
            checked_code(
                "write arena manifest",
                ops.bo_write(arena_manifest_bo, &manifest),
            )?;
            checked_code(
                "sync arena manifest",
                ops.bo_sync(arena_manifest_bo, XRT_BO_SYNC_TO_DEVICE, manifest.len(), 0),
            )?;
            let command_address = ops.bo_address(command_bo);
            let completion_address = ops.bo_address(completion_bo);
            let program_address = ops.bo_address(program_bo);
            let arena_manifest_address = ops.bo_address(arena_manifest_bo);
            let activation_address = ops.bo_address(activation_bo);
            let output_address = ops.bo_address(output_bo);
            let token_map_address = ops.bo_address(token_map_bo);
            if [
                command_address,
                completion_address,
                program_address,
                arena_manifest_address,
                activation_address,
                output_address,
                token_map_address,
            ]
            .contains(&0)
            {
                return Err(PersistentError::Config(
                    "runtime BO has a zero device address".to_string(),
                ));
            }
            cus.push(PersistentCu {
                ip_index: ip_indices[cu],
                command_bo,
                completion_bo,
                program_bo,
                arena_manifest_bo,
                activation_bo,
                output_bo,
                token_map_bo,
                arena_staging_bo,
                command_address,
                completion_address,
                program_address,
                arena_manifest_address,
                activation_address,
                output_address,
                token_map_address,
                arena,
                command_shadow: vec![0; ring_bytes],
                completion_shadow: vec![0; ring_bytes],
                program_shadow: vec![0; PROGRAM_BYTES],
                command_producer: command_baselines[cu],
                command_published: command_baselines[cu],
                command_consumer: command_baselines[cu],
                completion_consumer: completion_baselines[cu],
                completion_reserved: completion_baselines[cu],
                cached_program_id: None,
            });
        }
        let cus: [PersistentCu; ARENA_BANK_COUNT] = cus.try_into().map_err(|_| {
            PersistentError::Config("did not construct exactly four CUs".to_string())
        })?;
        let mut pool = Self {
            ops,
            device,
            native_device,
            xclbin_uuid,
            generation,
            model_tag,
            config,
            cus,
            poisoned: None,
            measured: false,
            measurement_baseline: PersistentDmaCounters::default(),
            dma,
            ticket_slots: std::array::from_fn(|_| TicketSlot::default()),
            next_poll_cu: 0,
            closed: false,
        };
        for cu in 0..ARENA_BANK_COUNT {
            pool.configure_and_start(cu)?;
        }
        Ok(pool)
    }

    fn reg_write(&self, cu: usize, offset: usize, value: u32) -> Result<(), PersistentError> {
        checked_code(
            "register write",
            self.ops.xcl_reg_write(
                self.native_device,
                self.cus[cu].ip_index,
                offset as u32,
                value,
            ),
        )
    }

    fn reg_read(&self, cu: usize, offset: usize) -> Result<u32, PersistentError> {
        let mut value = 0;
        checked_code(
            "register read",
            self.ops.xcl_reg_read(
                self.native_device,
                self.cus[cu].ip_index,
                offset as u32,
                &mut value,
            ),
        )?;
        Ok(value)
    }

    fn configure_and_start(&mut self, cu: usize) -> Result<(), PersistentError> {
        let command_address = split_u64(self.cus[cu].command_address);
        let completion_address = split_u64(self.cus[cu].completion_address);
        let program_address = split_u64(self.cus[cu].program_address);
        let arena_manifest_address = split_u64(self.cus[cu].arena_manifest_address);
        let activation_address = split_u64(self.cus[cu].activation_address);
        let result_address = split_u64(self.cus[cu].output_address);
        let token_map_address = split_u64(self.cus[cu].token_map_address);
        let model_tag = split_u64(self.model_tag);
        let generation = split_u64(self.generation);
        for (offset, value) in [
            (IQ1S_REG_SESSION_GENERATION_LO_OFFSET, generation.0),
            (IQ1S_REG_SESSION_GENERATION_HI_OFFSET, generation.1),
            (IQ1S_REG_COMMAND_BASE_LO_OFFSET, command_address.0),
            (IQ1S_REG_COMMAND_BASE_HI_OFFSET, command_address.1),
            (
                IQ1S_REG_COMMAND_CAPACITY_OFFSET,
                self.config.command_capacity,
            ),
            (
                IQ1S_REG_COMMAND_PRODUCER_OFFSET,
                self.cus[cu].command_producer,
            ),
            (IQ1S_REG_COMPLETION_BASE_LO_OFFSET, completion_address.0),
            (IQ1S_REG_COMPLETION_BASE_HI_OFFSET, completion_address.1),
            (
                IQ1S_REG_COMPLETION_CAPACITY_OFFSET,
                self.config.command_capacity,
            ),
            (
                IQ1S_REG_COMPLETION_CONSUMER_OFFSET,
                self.cus[cu].completion_consumer,
            ),
            (IQ1S_REG_PROGRAM_BASE_LO_OFFSET, program_address.0),
            (IQ1S_REG_PROGRAM_BASE_HI_OFFSET, program_address.1),
            (
                IQ1S_REG_ARENA_MANIFEST_BASE_LO_OFFSET,
                arena_manifest_address.0,
            ),
            (
                IQ1S_REG_ARENA_MANIFEST_BASE_HI_OFFSET,
                arena_manifest_address.1,
            ),
            (IQ1S_REG_ACTIVATION_BASE_LO_OFFSET, activation_address.0),
            (IQ1S_REG_ACTIVATION_BASE_HI_OFFSET, activation_address.1),
            (IQ1S_REG_RESULT_BASE_LO_OFFSET, result_address.0),
            (IQ1S_REG_RESULT_BASE_HI_OFFSET, result_address.1),
            (IQ1S_REG_TOKEN_MAP_BASE_LO_OFFSET, token_map_address.0),
            (IQ1S_REG_TOKEN_MAP_BASE_HI_OFFSET, token_map_address.1),
            (IQ1S_REG_MODEL_TAG_LO_OFFSET, model_tag.0),
            (IQ1S_REG_MODEL_TAG_HI_OFFSET, model_tag.1),
            (IQ1S_REG_ACTIVATION_BYTES_OFFSET, ACTIVATION_BYTES as u32),
            (IQ1S_REG_RESULT_BYTES_OFFSET, OUTPUT_BYTES as u32),
            (IQ1S_REG_TOKEN_MAP_BYTES_OFFSET, TOKEN_MAP_BYTES as u32),
            (IQ1S_REG_PROGRAM_BYTES_OFFSET, PROGRAM_BYTES as u32),
            (
                IQ1S_REG_ARENA_MANIFEST_BYTES_OFFSET,
                ARENA_MANIFEST_BYTES as u32,
            ),
            (IQ1S_REG_CU_ID_OFFSET, cu as u32),
            (IQ1S_REG_CONTROL_OFFSET, CONTROL_START),
        ] {
            self.reg_write(cu, offset, value)?;
        }
        Ok(())
    }

    fn hardware_fault_snapshot(
        &self,
        cu: usize,
        fault_code: u32,
    ) -> Result<PersistentFault, PersistentError> {
        let detail_lo = self.reg_read(cu, IQ1S_REG_FAULT_DETAIL_LO_OFFSET)?;
        let detail_hi = self.reg_read(cu, IQ1S_REG_FAULT_DETAIL_HI_OFFSET)?;
        let command_producer = self.reg_read(cu, IQ1S_REG_COMMAND_PRODUCER_OFFSET)?;
        let command_consumer = self.reg_read(cu, IQ1S_REG_COMMAND_CONSUMER_OFFSET)?;
        let command_capacity = self.reg_read(cu, IQ1S_REG_COMMAND_CAPACITY_OFFSET)?;
        let completion_producer = self.reg_read(cu, IQ1S_REG_COMPLETION_PRODUCER_OFFSET)?;
        let completion_consumer = self.reg_read(cu, IQ1S_REG_COMPLETION_CONSUMER_OFFSET)?;
        let completion_capacity = self.reg_read(cu, IQ1S_REG_COMPLETION_CAPACITY_OFFSET)?;
        Ok(PersistentFault {
            cu: Some(cu),
            operation: "hardware fault register",
            detail: format!(
                "fault_code={fault_code} fault_detail=0x{detail_hi:08x}{detail_lo:08x} command={command_producer}/{command_consumer}/{command_capacity} completion={completion_producer}/{completion_consumer}/{completion_capacity}"
            ),
        })
    }

    pub(crate) fn measurement_begin(&mut self) -> Result<(), PersistentError> {
        if let Some(fault) = &self.poisoned {
            return Err(PersistentError::Poisoned(fault.clone()));
        }
        if self.measured {
            return Err(PersistentError::Config(
                "measurement is already active".to_string(),
            ));
        }
        self.measured = true;
        self.measurement_baseline = self.dma;
        Ok(())
    }

    pub(crate) fn measurement_end(&mut self) -> Result<PersistentDmaCounters, PersistentError> {
        if !self.measured {
            return Err(PersistentError::Config(
                "measurement is not active".to_string(),
            ));
        }
        self.measured = false;
        if self.dma.weight_ranges != self.measurement_baseline.weight_ranges
            || self.dma.weight_bytes != self.measurement_baseline.weight_bytes
        {
            return self.poison(PersistentFault {
                cu: None,
                operation: "weight residency",
                detail: "weight DMA occurred inside the measured window".to_string(),
            });
        }
        self.dma.checked_delta(self.measurement_baseline)
    }

    fn poison<T>(&mut self, fault: PersistentFault) -> Result<T, PersistentError> {
        if self.poisoned.is_none() {
            self.poisoned = Some(fault.clone());
        }
        for slot in &mut self.ticket_slots {
            if slot.state != SlotState::Free {
                slot.state = SlotState::Poisoned;
            }
        }
        Err(PersistentError::Fault(
            self.poisoned.clone().unwrap_or(fault),
        ))
    }

    fn resolve_arena(cu: &PersistentCu, logical: u64, bytes: u64) -> Result<u64, PersistentError> {
        let requested_end = logical.checked_add(bytes).ok_or_else(|| {
            PersistentError::InvalidPhase("arena descriptor range overflow".to_string())
        })?;
        for chunk in &cu.arena {
            let end = chunk
                .logical_offset
                .checked_add(chunk.bytes as u64)
                .ok_or_else(|| PersistentError::InvalidPhase("arena range overflow".to_string()))?;
            if logical >= chunk.logical_offset && requested_end <= end {
                return chunk
                    .address
                    .checked_add(logical - chunk.logical_offset)
                    .ok_or_else(|| {
                        PersistentError::InvalidPhase("arena relocation overflow".to_string())
                    });
            }
        }
        Err(PersistentError::InvalidPhase(format!(
            "arena offset {logical} is not resident"
        )))
    }

    fn descriptor_ranges(start: u32, count: usize, capacity: u32) -> Vec<(usize, usize)> {
        let slot = (start & (capacity - 1)) as usize;
        let capacity = capacity as usize;
        let first = count.min(capacity - slot);
        let mut ranges = vec![(slot * IQ1S_COMMAND_BYTES, first * IQ1S_COMMAND_BYTES)];
        if first < count {
            ranges.push((0, (count - first) * IQ1S_COMMAND_BYTES));
        }
        ranges
    }

    fn write_program_if_needed(
        &mut self,
        cu: usize,
        phase: &CompiledLayerPhase,
    ) -> Result<(), PersistentError> {
        let program_id = phase.commands[cu][0].program_id;
        if self.cus[cu].cached_program_id == Some(program_id) {
            return Ok(());
        }
        let encoded = &phase.programs[cu].encoded;
        if encoded.is_empty() || encoded.len() > PROGRAM_BYTES {
            return Err(PersistentError::InvalidPhase(
                "encoded program is empty or exceeds program BO".to_string(),
            ));
        }
        self.cus[cu].program_shadow.fill(0);
        self.cus[cu].program_shadow[..encoded.len()].copy_from_slice(encoded);
        checked_code(
            "write program BO",
            self.ops
                .bo_write(self.cus[cu].program_bo, &self.cus[cu].program_shadow),
        )?;
        checked_code(
            "sync program BO",
            self.ops.bo_sync(
                self.cus[cu].program_bo,
                XRT_BO_SYNC_TO_DEVICE,
                encoded.len(),
                0,
            ),
        )?;
        self.dma.program_ranges += 1;
        self.cus[cu].cached_program_id = Some(program_id);
        Ok(())
    }

    pub(crate) fn submit_phase(
        &mut self,
        phase: &CompiledLayerPhase,
        buffers: &PhaseBuffers,
    ) -> Result<CompletedLayerPhase, PersistentError> {
        let result = (|| {
            let ticket = self.prepare_ticket_inner(phase, buffers)?;
            self.publish_ticket_inner(&ticket)?;
            let mut backoff = 1u64;
            while self.poll_ticket_inner(&ticket)? == TicketPoll::Pending {
                std::thread::sleep(Duration::from_micros(backoff));
                backoff = (backoff * 2).min(MAX_BACKOFF_US);
            }
            self.collect_ticket_inner(&ticket)
        })();
        self.record_ticket_error(result)
    }

    fn record_ticket_error<T>(
        &mut self,
        result: Result<T, PersistentError>,
    ) -> Result<T, PersistentError> {
        if let Err(error) = &result {
            if self.poisoned.is_none()
                && !matches!(
                    error,
                    PersistentError::TicketSlotsFull
                        | PersistentError::Poisoned(_)
                        | PersistentError::Shutdown(_)
                )
            {
                let fault = PersistentFault {
                    cu: None,
                    operation: "ticket lifecycle",
                    detail: error.to_string(),
                };
                self.poisoned = Some(fault);
                for slot in &mut self.ticket_slots {
                    if slot.state != SlotState::Free {
                        slot.state = SlotState::Poisoned;
                    }
                }
            }
        }
        result
    }

    pub(crate) fn has_free_ticket_slot(&self) -> bool {
        self.poisoned.is_none()
            && !self.closed
            && self
                .ticket_slots
                .iter()
                .any(|slot| slot.state == SlotState::Free)
    }

    pub(crate) fn prepare_ticket(
        &mut self,
        phase: &CompiledLayerPhase,
        buffers: &PhaseBuffers,
    ) -> Result<SubmissionTicket, PersistentError> {
        let result = self.prepare_ticket_inner(phase, buffers);
        self.record_ticket_error(result)
    }

    fn prepare_ticket_inner(
        &mut self,
        phase: &CompiledLayerPhase,
        buffers: &PhaseBuffers,
    ) -> Result<SubmissionTicket, PersistentError> {
        if let Some(fault) = &self.poisoned {
            return Err(PersistentError::Poisoned(fault.clone()));
        }
        if self.closed {
            return Err(PersistentError::Shutdown("pool is closed".to_string()));
        }
        let slot_index = self
            .ticket_slots
            .iter()
            .position(|slot| slot.state == SlotState::Free)
            .ok_or(PersistentError::TicketSlotsFull)?;
        validate_compiled_layer_phase(phase, QWEN_MODEL_CONTEXT_LIMIT)
            .map_err(PersistentError::InvalidPhase)?;
        let activations =
            coalesce_host_ranges(&buffers.activations, ACTIVATION_SLOT_BYTES, "activation")?;
        let token_maps =
            coalesce_host_ranges(&buffers.token_maps, TOKEN_MAP_SLOT_BYTES, "token-map")?;
        let activation_ranges = activations
            .iter()
            .map(|range| (range.offset as usize, range.bytes.len()))
            .collect::<Vec<_>>();
        let token_map_ranges = token_maps
            .iter()
            .map(|range| (range.offset as usize, range.bytes.len()))
            .collect::<Vec<_>>();
        let activation_manifest = merge_adjacent_ranges(
            phase
                .activations
                .iter()
                .map(|range| {
                    let offset = usize::try_from(range.slab_offset).map_err(|_| {
                        PersistentError::InvalidPhase(
                            "activation manifest offset does not fit usize".to_string(),
                        )
                    })?;
                    let bytes = usize::try_from(range.bytes).map_err(|_| {
                        PersistentError::InvalidPhase(
                            "activation manifest bytes do not fit usize".to_string(),
                        )
                    })?;
                    Ok((offset, bytes))
                })
                .collect::<Result<Vec<_>, PersistentError>>()?,
            "activation manifest",
        )?;
        let supplied_activations = activations
            .iter()
            .map(|range| (range.offset as usize, range.bytes.len()))
            .collect::<Vec<_>>();
        if supplied_activations != activation_manifest {
            return Err(PersistentError::InvalidPhase(
                "activation manifest differs from compiled phase".to_string(),
            ));
        }

        let activation_slot_base = slot_index * ACTIVATION_SLOT_BYTES;
        let output_slot_base = slot_index * OUTPUT_SLOT_BYTES;
        let token_map_slot_base = slot_index * TOKEN_MAP_SLOT_BYTES;
        let mut timings = PersistentPhaseTimings::default();
        let mut ticket_dma = PersistentDmaCounters::default();
        let mut result_ranges: [Vec<(usize, usize, usize)>; ARENA_BANK_COUNT] =
            std::array::from_fn(|_| Vec::new());
        let mut expected: [Vec<Iq1sCommand>; ARENA_BANK_COUNT] =
            std::array::from_fn(|_| Vec::new());
        let mut command_end = [0u32; ARENA_BANK_COUNT];
        let mut expected_completion = [0u32; ARENA_BANK_COUNT];
        let phase_wall_start = Instant::now();

        for cu in 0..ARENA_BANK_COUNT {
            let mut outputs = Vec::with_capacity(phase.commands[cu].len());
            for command in &phase.commands[cu] {
                let input_offset = usize::try_from(command.input_offset).map_err(|_| {
                    PersistentError::InvalidPhase(
                        "descriptor activation offset does not fit usize".to_string(),
                    )
                })?;
                let output_offset = usize::try_from(command.output_offset).map_err(|_| {
                    PersistentError::InvalidPhase(
                        "descriptor output offset does not fit usize".to_string(),
                    )
                })?;
                let token_offset = usize::try_from(command.token_map_offset).map_err(|_| {
                    PersistentError::InvalidPhase(
                        "descriptor token-map offset does not fit usize".to_string(),
                    )
                })?;
                let input_bytes = command.input_bytes as usize;
                let output_bytes = command.output_bytes as usize;
                let token_bytes =
                    usize::from(command.lane_count)
                        .checked_mul(4)
                        .ok_or_else(|| {
                            PersistentError::InvalidPhase(
                                "descriptor token-map byte count overflow".to_string(),
                            )
                        })?;
                if input_offset
                    .checked_add(input_bytes)
                    .is_none_or(|end| end > ACTIVATION_SLOT_BYTES)
                    || output_offset
                        .checked_add(output_bytes)
                        .is_none_or(|end| end > OUTPUT_SLOT_BYTES)
                    || token_offset
                        .checked_add(token_bytes)
                        .is_none_or(|end| end > TOKEN_MAP_SLOT_BYTES)
                    || !range_is_covered(&activation_ranges, input_offset, input_bytes)
                    || !range_is_covered(&token_map_ranges, token_offset, token_bytes)
                {
                    return Err(PersistentError::InvalidPhase(format!(
                        "CU {cu} descriptor is outside its ticket activation, output, or token-map slot"
                    )));
                }
                outputs.push((output_offset, output_bytes));
            }
            result_ranges[cu] = merge_adjacent_ranges(outputs, "result")?
                .into_iter()
                .map(|(logical, bytes)| (logical, output_slot_base + logical, bytes))
                .collect();

            let hardware_consumer = self.reg_read(cu, IQ1S_REG_COMMAND_CONSUMER_OFFSET)?;
            self.cus[cu].command_consumer = hardware_consumer;
            let count = u32::try_from(phase.commands[cu].len()).map_err(|_| {
                PersistentError::InvalidPhase("command count does not fit u32".to_string())
            })?;
            let used = self.cus[cu]
                .command_producer
                .wrapping_sub(self.cus[cu].command_consumer);
            if used > self.config.command_capacity
                || count > self.config.command_capacity.saturating_sub(used)
            {
                return Err(PersistentError::RingFull {
                    cu,
                    capacity: self.config.command_capacity,
                });
            }
            let producer = self.cus[cu].command_producer;
            for (index, source) in phase.commands[cu].iter().enumerate() {
                let mut command = *source;
                command.session_generation = self.generation;
                let input_columns = match u32::from(command.role) {
                    IQ1S_ROLE_GATE | IQ1S_ROLE_UP => 4096u64,
                    IQ1S_ROLE_DOWN => 1024u64,
                    _ => {
                        return Err(PersistentError::InvalidPhase(
                            "descriptor has an unsupported role".to_string(),
                        ));
                    }
                };
                let weight_bytes = u64::from(command.row_count)
                    .checked_mul(input_columns / 256)
                    .and_then(|blocks| blocks.checked_mul(50))
                    .ok_or_else(|| {
                        PersistentError::InvalidPhase(
                            "descriptor weight byte count overflow".to_string(),
                        )
                    })?;
                command.arena_offset =
                    Self::resolve_arena(&self.cus[cu], source.arena_offset, weight_bytes)?;
                command.input_offset = self.cus[cu]
                    .activation_address
                    .checked_add((activation_slot_base as u64) + source.input_offset)
                    .ok_or_else(|| {
                        PersistentError::InvalidPhase("input relocation overflow".to_string())
                    })?;
                command.output_offset = self.cus[cu]
                    .output_address
                    .checked_add((output_slot_base as u64) + source.output_offset)
                    .ok_or_else(|| {
                        PersistentError::InvalidPhase("output relocation overflow".to_string())
                    })?;
                command.token_map_offset = self.cus[cu]
                    .token_map_address
                    .checked_add((token_map_slot_base as u64) + source.token_map_offset)
                    .ok_or_else(|| {
                        PersistentError::InvalidPhase("token-map relocation overflow".to_string())
                    })?;
                command.crc32 = 0;
                command.crc32 = command_crc(&command);
                let ring_slot =
                    producer.wrapping_add(index as u32) & (self.config.command_capacity - 1);
                let offset = ring_slot as usize * IQ1S_COMMAND_BYTES;
                self.cus[cu].command_shadow[offset..offset + IQ1S_COMMAND_BYTES]
                    .copy_from_slice(struct_bytes(&command));
                expected[cu].push(command);
            }

            let ring_publish_start = Instant::now();
            for (offset, bytes) in Self::descriptor_ranges(
                producer,
                phase.commands[cu].len(),
                self.config.command_capacity,
            ) {
                let end = offset.checked_add(bytes).ok_or_else(|| {
                    PersistentError::InvalidPhase("command ring range overflow".to_string())
                })?;
                checked_code(
                    "write command ring",
                    self.ops.bo_write_range(
                        self.cus[cu].command_bo,
                        &self.cus[cu].command_shadow[offset..end],
                        offset,
                    ),
                )?;
                checked_code(
                    "sync command ring",
                    self.ops.bo_sync(
                        self.cus[cu].command_bo,
                        XRT_BO_SYNC_TO_DEVICE,
                        bytes,
                        offset,
                    ),
                )?;
                self.dma.command_ranges += 1;
                ticket_dma.command_ranges += 1;
            }
            timings.ring_publish_us = timings
                .ring_publish_us
                .saturating_add(elapsed_us(ring_publish_start));

            let activation_sync_start = Instant::now();
            for range in &activations {
                let logical = range.offset as usize;
                let physical = activation_slot_base + logical;
                checked_code(
                    "write activation range",
                    self.ops
                        .bo_write_range(self.cus[cu].activation_bo, &range.bytes, physical),
                )?;
                checked_code(
                    "sync activation range",
                    self.ops.bo_sync(
                        self.cus[cu].activation_bo,
                        XRT_BO_SYNC_TO_DEVICE,
                        range.bytes.len(),
                        physical,
                    ),
                )?;
                self.dma.activation_ranges += 1;
                ticket_dma.activation_ranges += 1;
            }
            for range in &token_maps {
                let logical = range.offset as usize;
                let physical = token_map_slot_base + logical;
                checked_code(
                    "write token-map range",
                    self.ops
                        .bo_write_range(self.cus[cu].token_map_bo, &range.bytes, physical),
                )?;
                checked_code(
                    "sync token-map range",
                    self.ops.bo_sync(
                        self.cus[cu].token_map_bo,
                        XRT_BO_SYNC_TO_DEVICE,
                        range.bytes.len(),
                        physical,
                    ),
                )?;
            }
            timings.activation_sync_us = timings
                .activation_sync_us
                .saturating_add(elapsed_us(activation_sync_start));
            let next_producer = producer.wrapping_add(count);
            self.cus[cu].command_producer = next_producer;
            command_end[cu] = next_producer;
            let next_completion = self.cus[cu].completion_reserved.wrapping_add(count);
            self.cus[cu].completion_reserved = next_completion;
            expected_completion[cu] = next_completion;
        }

        let slot_generation = self.ticket_slots[slot_index].generation.wrapping_add(1);
        if slot_generation == 0 {
            return Err(PersistentError::InvalidPhase(
                "ticket slot generation overflow".to_string(),
            ));
        }
        let ticket = SubmissionTicket {
            slot: slot_index as u8,
            slot_generation,
            transaction_id: phase.transaction_id,
            command_end,
            expected_completion,
            deadline: Instant::now() + Duration::from_millis(u64::from(self.config.timeout_ms)),
        };
        self.ticket_slots[slot_index] = TicketSlot {
            generation: slot_generation,
            state: SlotState::Prepared,
            prepared: Some(PreparedTicket {
                phase: phase.clone(),
                expected,
                result_ranges,
                dma: ticket_dma,
                timings,
                phase_wall_start,
            }),
        };
        Ok(ticket)
    }

    fn ticket_index(
        &self,
        ticket: &SubmissionTicket,
        allowed: &[SlotState],
    ) -> Result<usize, PersistentError> {
        let index = usize::from(ticket.slot);
        let Some(slot) = self.ticket_slots.get(index) else {
            return Err(PersistentError::InvalidPhase(
                "ticket slot is invalid".to_string(),
            ));
        };
        if slot.generation != ticket.slot_generation
            || slot
                .prepared
                .as_ref()
                .is_none_or(|prepared| prepared.phase.transaction_id != ticket.transaction_id)
        {
            return Err(PersistentError::InvalidPhase(
                "ticket slot generation or transaction is stale".to_string(),
            ));
        }
        if !allowed.contains(&slot.state) {
            return Err(PersistentError::InvalidPhase(format!(
                "ticket slot is in state {:?}",
                slot.state
            )));
        }
        Ok(index)
    }

    pub(crate) fn publish_ticket(
        &mut self,
        ticket: &SubmissionTicket,
    ) -> Result<(), PersistentError> {
        let result = self.publish_ticket_inner(ticket);
        self.record_ticket_error(result)
    }

    fn publish_ticket_inner(&mut self, ticket: &SubmissionTicket) -> Result<(), PersistentError> {
        if let Some(fault) = &self.poisoned {
            return Err(PersistentError::Poisoned(fault.clone()));
        }
        let index = self.ticket_index(ticket, &[SlotState::Prepared])?;
        let phase = self.ticket_slots[index]
            .prepared
            .as_ref()
            .expect("validated ticket has prepared data")
            .phase
            .clone();
        for cu in 0..ARENA_BANK_COUNT {
            let count = phase.commands[cu].len() as u32;
            let start = ticket.command_end[cu].wrapping_sub(count);
            if self.cus[cu].command_published != start {
                return Err(PersistentError::InvalidPhase(
                    "tickets must be published in reservation order".to_string(),
                ));
            }
            self.write_program_if_needed(cu, &phase)?;
        }
        let publish_start = Instant::now();
        for cu in 0..ARENA_BANK_COUNT {
            self.reg_write(
                cu,
                IQ1S_REG_COMMAND_PRODUCER_OFFSET,
                ticket.command_end[cu],
            )?;
            self.reg_write(cu, IQ1S_REG_DOORBELL_OFFSET, 1)?;
            self.cus[cu].command_published = ticket.command_end[cu];
        }
        let elapsed = elapsed_us(publish_start);
        if let Some(prepared) = self.ticket_slots[index].prepared.as_mut() {
            prepared.timings.doorbell_us = prepared.timings.doorbell_us.saturating_add(elapsed);
            for cu in 0..ARENA_BANK_COUNT {
                if self.cus[cu].cached_program_id == Some(phase.commands[cu][0].program_id) {
                    // Global counters remain authoritative; this ticket owns no weight DMA.
                }
            }
        }
        self.ticket_slots[index].state = SlotState::Published;
        Ok(())
    }

    pub(crate) fn poll_ticket(
        &mut self,
        ticket: &SubmissionTicket,
    ) -> Result<TicketPoll, PersistentError> {
        let result = self.poll_ticket_inner(ticket);
        self.record_ticket_error(result)
    }

    fn poll_ticket_inner(
        &mut self,
        ticket: &SubmissionTicket,
    ) -> Result<TicketPoll, PersistentError> {
        if let Some(fault) = &self.poisoned {
            return Err(PersistentError::Poisoned(fault.clone()));
        }
        let index = self.ticket_index(ticket, &[SlotState::Published, SlotState::Complete])?;
        if self.ticket_slots[index].state == SlotState::Complete {
            return Ok(TicketPoll::Complete);
        }
        let poll_start = Instant::now();
        let start_cu = self.next_poll_cu;
        let mut all_complete = true;
        for step in 0..ARENA_BANK_COUNT {
            let cu = (start_cu + step) % ARENA_BANK_COUNT;
            let fault_code = self.reg_read(cu, IQ1S_REG_FAULT_CODE_OFFSET)?;
            if fault_code != IQ1S_FAULT_CODE_NONE {
                let fault = self.hardware_fault_snapshot(cu, fault_code)?;
                return self.poison(fault);
            }
            let producer = self.reg_read(cu, IQ1S_REG_COMPLETION_PRODUCER_OFFSET)?;
            if producer.wrapping_sub(ticket.expected_completion[cu]) >= (1u32 << 31) {
                all_complete = false;
            }
        }
        self.next_poll_cu = (start_cu + 1) % ARENA_BANK_COUNT;
        if let Some(prepared) = self.ticket_slots[index].prepared.as_mut() {
            prepared.timings.device_wait_us = prepared
                .timings
                .device_wait_us
                .saturating_add(elapsed_us(poll_start));
        }
        if all_complete {
            self.ticket_slots[index].state = SlotState::Complete;
            return Ok(TicketPoll::Complete);
        }
        if Instant::now() >= ticket.deadline {
            return self.poison(PersistentFault {
                cu: None,
                operation: "completion poll",
                detail: format!("timeout after {} ms", self.config.timeout_ms),
            });
        }
        Ok(TicketPoll::Pending)
    }

    pub(crate) fn collect_ticket(
        &mut self,
        ticket: &SubmissionTicket,
    ) -> Result<CompletedLayerPhase, PersistentError> {
        let result = self.collect_ticket_inner(ticket);
        self.record_ticket_error(result)
    }

    fn collect_ticket_inner(
        &mut self,
        ticket: &SubmissionTicket,
    ) -> Result<CompletedLayerPhase, PersistentError> {
        if let Some(fault) = &self.poisoned {
            return Err(PersistentError::Poisoned(fault.clone()));
        }
        let index = self.ticket_index(ticket, &[SlotState::Complete])?;
        let mut prepared = self.ticket_slots[index]
            .prepared
            .take()
            .expect("validated ticket has prepared data");
        let mut completed: [Vec<Iq1sCompletion>; ARENA_BANK_COUNT] =
            std::array::from_fn(|_| Vec::new());
        let mut results: [Vec<HostRange>; ARENA_BANK_COUNT] = std::array::from_fn(|_| Vec::new());

        for cu in 0..ARENA_BANK_COUNT {
            let count = prepared.expected[cu].len();
            let start_counter = ticket.expected_completion[cu].wrapping_sub(count as u32);
            if self.cus[cu].completion_consumer != start_counter {
                self.ticket_slots[index].prepared = Some(prepared);
                return Err(PersistentError::InvalidPhase(
                    "completed tickets must be collected in reservation order".to_string(),
                ));
            }
            let completion_sync_start = Instant::now();
            for (offset, bytes) in
                Self::descriptor_ranges(start_counter, count, self.config.command_capacity)
            {
                checked_code(
                    "sync completion ring",
                    self.ops.bo_sync(
                        self.cus[cu].completion_bo,
                        XRT_BO_SYNC_FROM_DEVICE,
                        bytes,
                        offset,
                    ),
                )?;
                let end = offset.checked_add(bytes).ok_or_else(|| {
                    PersistentError::InvalidPhase("completion ring range overflow".to_string())
                })?;
                checked_code(
                    "read completion ring",
                    self.ops.bo_read_range(
                        self.cus[cu].completion_bo,
                        &mut self.cus[cu].completion_shadow[offset..end],
                        offset,
                    ),
                )?;
            }
            prepared.timings.completion_sync_us = prepared
                .timings
                .completion_sync_us
                .saturating_add(elapsed_us(completion_sync_start));
            for (command_index, command) in prepared.expected[cu].iter().enumerate() {
                let counter = start_counter.wrapping_add(command_index as u32);
                let command_counter = ticket.command_end[cu]
                    .wrapping_sub(count as u32)
                    .wrapping_add(command_index as u32);
                let ring_slot = counter & (self.config.command_capacity - 1);
                let offset = ring_slot as usize * IQ1S_COMPLETION_BYTES;
                let completion = struct_from_bytes::<Iq1sCompletion>(
                    &self.cus[cu].completion_shadow[offset..offset + IQ1S_COMPLETION_BYTES],
                )?;
                let mismatch = completion.magic != IQ1S_COMPLETION_MAGIC
                    || completion.abi_version != IQ1S_ABI_VERSION as u16
                    || completion.completion_bytes != IQ1S_COMPLETION_BYTES as u16
                    || completion.status != IQ1S_COMPLETION_STATUS_OK
                    || completion.fault_code != IQ1S_FAULT_CODE_NONE
                    || completion.session_generation != self.generation
                    || completion.transaction_id != command.transaction_id
                    || completion.program_id != command.program_id
                    || completion.trace_id != command.trace_id
                    || completion.layer_id != command.layer_id
                    || completion.phase != command.phase
                    || usize::from(completion.cu_id) != cu
                    || completion.expert_id != command.expert_id
                    || completion.lane_mask != command.lane_mask
                    || completion.rows_completed != command.row_count as u16
                    || completion.descriptor_crc32 != command.crc32
                    || completion.command_index != command_counter
                    || completion.result_fence == 0;
                if mismatch {
                    self.ticket_slots[index].prepared = Some(prepared);
                    return self.poison(PersistentFault {
                        cu: Some(cu),
                        operation: "completion validation",
                        detail: format!("completion {counter} does not match its descriptor"),
                    });
                }
                completed[cu].push(completion);
            }
            self.cus[cu].completion_consumer = ticket.expected_completion[cu];
            self.cus[cu].command_consumer = ticket.command_end[cu];
            self.reg_write(
                cu,
                IQ1S_REG_COMPLETION_CONSUMER_OFFSET,
                ticket.expected_completion[cu],
            )?;
            let result_copy_start = Instant::now();
            for (logical_offset, physical_offset, bytes) in &prepared.result_ranges[cu] {
                checked_code(
                    "sync result range",
                    self.ops.bo_sync(
                        self.cus[cu].output_bo,
                        XRT_BO_SYNC_FROM_DEVICE,
                        *bytes,
                        *physical_offset,
                    ),
                )?;
                let mut result = vec![0u8; *bytes];
                checked_code(
                    "read result range",
                    self.ops
                        .bo_read_range(self.cus[cu].output_bo, &mut result, *physical_offset),
                )?;
                results[cu].push(HostRange {
                    offset: *logical_offset as u64,
                    bytes: result,
                });
                self.dma.result_ranges += 1;
                prepared.dma.result_ranges += 1;
            }
            prepared.timings.result_copy_us = prepared
                .timings
                .result_copy_us
                .saturating_add(elapsed_us(result_copy_start));
        }
        if self.measured
            && (self.dma.weight_ranges != self.measurement_baseline.weight_ranges
                || self.dma.weight_bytes != self.measurement_baseline.weight_bytes)
        {
            self.ticket_slots[index].prepared = Some(prepared);
            return self.poison(PersistentFault {
                cu: None,
                operation: "weight residency",
                detail: "weight DMA occurred during submit".to_string(),
            });
        }
        let expanded = prepared.phase.programs.iter().fold(
            ExpandedIq1sCounts::default(),
            |mut total, program| {
                total.blocks = total.blocks.saturating_add(program.expanded.blocks);
                total.grid_passes = total
                    .grid_passes
                    .saturating_add(program.expanded.grid_passes);
                total.delta_passes = total
                    .delta_passes
                    .saturating_add(program.expanded.delta_passes);
                total
            },
        );
        prepared.timings.phase_wall_us = elapsed_us(prepared.phase_wall_start);
        let completed_phase = CompletedLayerPhase {
            transaction_id: prepared.phase.transaction_id,
            semantic_sha256: prepared.phase.semantic_sha256,
            completions: completed,
            results,
            expanded,
            dma: prepared.dma,
            timings: prepared.timings,
        };
        self.ticket_slots[index].state = SlotState::Free;
        Ok(completed_phase)
    }

    #[allow(dead_code)]
    fn legacy_submit_phase_inner(
        &mut self,
        phase: &CompiledLayerPhase,
        buffers: &PhaseBuffers,
    ) -> Result<CompletedLayerPhase, PersistentError> {
        let phase_wall_start = Instant::now();
        let mut timings = PersistentPhaseTimings::default();
        if let Some(fault) = &self.poisoned {
            return Err(PersistentError::Poisoned(fault.clone()));
        }
        if self.closed {
            return Err(PersistentError::Shutdown("pool is closed".to_string()));
        }
        validate_compiled_layer_phase(phase, QWEN_MODEL_CONTEXT_LIMIT)
            .map_err(PersistentError::InvalidPhase)?;
        let activation_ranges =
            validate_host_ranges(&buffers.activations, ACTIVATION_BYTES, "activation")?;
        let token_map_ranges =
            validate_host_ranges(&buffers.token_maps, TOKEN_MAP_BYTES, "token-map")?;
        let mut activation_manifest = phase
            .activations
            .iter()
            .map(|range| {
                (
                    range.slab_offset,
                    usize::try_from(range.bytes).unwrap_or(usize::MAX),
                )
            })
            .collect::<Vec<_>>();
        let mut supplied_activations = buffers
            .activations
            .iter()
            .map(|range| (range.offset, range.bytes.len()))
            .collect::<Vec<_>>();
        activation_manifest.sort_unstable();
        supplied_activations.sort_unstable();
        if supplied_activations != activation_manifest {
            return Err(PersistentError::InvalidPhase(
                "activation manifest differs from compiled phase".to_string(),
            ));
        }
        let dma_before = self.dma;
        let mut result_ranges: [Vec<(usize, usize)>; ARENA_BANK_COUNT] =
            std::array::from_fn(|_| Vec::new());
        for (cu, commands) in phase.commands.iter().enumerate() {
            let mut outputs = Vec::with_capacity(commands.len());
            for command in commands {
                let input_offset = usize::try_from(command.input_offset).map_err(|_| {
                    PersistentError::InvalidPhase(
                        "descriptor activation offset does not fit usize".to_string(),
                    )
                })?;
                let input_bytes = command.input_bytes as usize;
                let output_offset = usize::try_from(command.output_offset).map_err(|_| {
                    PersistentError::InvalidPhase(
                        "descriptor output offset does not fit usize".to_string(),
                    )
                })?;
                let output_bytes = command.output_bytes as usize;
                let token_offset = usize::try_from(command.token_map_offset).map_err(|_| {
                    PersistentError::InvalidPhase(
                        "descriptor token-map offset does not fit usize".to_string(),
                    )
                })?;
                let token_bytes =
                    usize::from(command.lane_count)
                        .checked_mul(4)
                        .ok_or_else(|| {
                            PersistentError::InvalidPhase(
                                "descriptor token-map byte count overflow".to_string(),
                            )
                        })?;
                if input_offset
                    .checked_add(input_bytes)
                    .is_none_or(|end| end > ACTIVATION_BYTES)
                    || output_offset
                        .checked_add(output_bytes)
                        .is_none_or(|end| end > OUTPUT_BYTES)
                    || token_offset
                        .checked_add(token_bytes)
                        .is_none_or(|end| end > TOKEN_MAP_BYTES)
                    || !range_is_covered(&activation_ranges, input_offset, input_bytes)
                    || !range_is_covered(&token_map_ranges, token_offset, token_bytes)
                {
                    return Err(PersistentError::InvalidPhase(format!(
                        "CU {cu} descriptor is outside supplied activation, output, or token-map ranges"
                    )));
                }
                outputs.push((output_offset, output_bytes));
            }
            result_ranges[cu] = merge_adjacent_ranges(outputs, "result")?;
        }

        let mut expected: [Vec<Iq1sCommand>; ARENA_BANK_COUNT] =
            std::array::from_fn(|_| Vec::new());
        for cu in 0..ARENA_BANK_COUNT {
            let hardware_consumer = self.reg_read(cu, IQ1S_REG_COMMAND_CONSUMER_OFFSET)?;
            self.cus[cu].command_consumer = hardware_consumer;
            let count = u32::try_from(phase.commands[cu].len()).map_err(|_| {
                PersistentError::InvalidPhase("command count does not fit u32".to_string())
            })?;
            let used = self.cus[cu]
                .command_producer
                .wrapping_sub(self.cus[cu].command_consumer);
            if used > self.config.command_capacity
                || count > self.config.command_capacity.saturating_sub(used)
            {
                let fault = PersistentFault {
                    cu: Some(cu),
                    operation: "command ring",
                    detail: format!(
                        "producer={} consumer={} count={} capacity={}",
                        self.cus[cu].command_producer,
                        self.cus[cu].command_consumer,
                        count,
                        self.config.command_capacity
                    ),
                };
                self.poisoned = Some(fault);
                return Err(PersistentError::RingFull {
                    cu,
                    capacity: self.config.command_capacity,
                });
            }
            self.write_program_if_needed(cu, phase)?;
            let producer = self.cus[cu].command_producer;
            for (index, source) in phase.commands[cu].iter().enumerate() {
                let mut command = *source;
                command.session_generation = self.generation;
                let input_columns = match u32::from(command.role) {
                    IQ1S_ROLE_GATE | IQ1S_ROLE_UP => 4096u64,
                    IQ1S_ROLE_DOWN => 1024u64,
                    _ => {
                        return Err(PersistentError::InvalidPhase(
                            "descriptor has an unsupported role".to_string(),
                        ));
                    }
                };
                let weight_bytes = u64::from(command.row_count)
                    .checked_mul(input_columns / 256)
                    .and_then(|blocks| blocks.checked_mul(50))
                    .ok_or_else(|| {
                        PersistentError::InvalidPhase(
                            "descriptor weight byte count overflow".to_string(),
                        )
                    })?;
                command.arena_offset =
                    Self::resolve_arena(&self.cus[cu], source.arena_offset, weight_bytes)?;
                if source
                    .input_offset
                    .checked_add(u64::from(source.input_bytes))
                    .is_none_or(|end| end > ACTIVATION_BYTES as u64)
                    || source
                        .output_offset
                        .checked_add(u64::from(source.output_bytes))
                        .is_none_or(|end| end > OUTPUT_BYTES as u64)
                    || source
                        .token_map_offset
                        .checked_add(u64::from(source.lane_count) * 4)
                        .is_none_or(|end| end > TOKEN_MAP_BYTES as u64)
                {
                    return Err(PersistentError::InvalidPhase(
                        "descriptor activation, output, or token-map range exceeds its slab"
                            .to_string(),
                    ));
                }
                command.input_offset = self.cus[cu]
                    .activation_address
                    .checked_add(source.input_offset)
                    .ok_or_else(|| {
                        PersistentError::InvalidPhase("input relocation overflow".to_string())
                    })?;
                command.output_offset = self.cus[cu]
                    .output_address
                    .checked_add(source.output_offset)
                    .ok_or_else(|| {
                        PersistentError::InvalidPhase("output relocation overflow".to_string())
                    })?;
                command.token_map_offset = self.cus[cu]
                    .token_map_address
                    .checked_add(source.token_map_offset)
                    .ok_or_else(|| {
                        PersistentError::InvalidPhase("token-map relocation overflow".to_string())
                    })?;
                command.crc32 = 0;
                command.crc32 = command_crc(&command);
                let slot = producer.wrapping_add(index as u32) & (self.config.command_capacity - 1);
                let offset = slot as usize * IQ1S_COMMAND_BYTES;
                self.cus[cu].command_shadow[offset..offset + IQ1S_COMMAND_BYTES]
                    .copy_from_slice(struct_bytes(&command));
                expected[cu].push(command);
            }
            let ring_publish_start = Instant::now();
            let command_ranges = Self::descriptor_ranges(
                producer,
                phase.commands[cu].len(),
                self.config.command_capacity,
            );
            for (offset, bytes) in command_ranges {
                let end = offset.checked_add(bytes).ok_or_else(|| {
                    PersistentError::InvalidPhase("command ring range overflow".to_string())
                })?;
                checked_code(
                    "write command ring",
                    self.ops.bo_write_range(
                        self.cus[cu].command_bo,
                        &self.cus[cu].command_shadow[offset..end],
                        offset,
                    ),
                )?;
                checked_code(
                    "sync command ring",
                    self.ops.bo_sync(
                        self.cus[cu].command_bo,
                        XRT_BO_SYNC_TO_DEVICE,
                        bytes,
                        offset,
                    ),
                )?;
                self.dma.command_ranges += 1;
            }
            let next_producer = producer.wrapping_add(count);
            timings.ring_publish_us = timings
                .ring_publish_us
                .saturating_add(elapsed_us(ring_publish_start));

            let activation_sync_start = Instant::now();
            for range in &buffers.activations {
                let offset = usize::try_from(range.offset).map_err(|_| {
                    PersistentError::InvalidPhase(
                        "activation offset does not fit usize".to_string(),
                    )
                })?;
                checked_code(
                    "write activation range",
                    self.ops
                        .bo_write_range(self.cus[cu].activation_bo, &range.bytes, offset),
                )?;
                checked_code(
                    "sync activation range",
                    self.ops.bo_sync(
                        self.cus[cu].activation_bo,
                        XRT_BO_SYNC_TO_DEVICE,
                        range.bytes.len(),
                        offset,
                    ),
                )?;
                self.dma.activation_ranges += 1;
            }
            for range in &buffers.token_maps {
                let offset = usize::try_from(range.offset).map_err(|_| {
                    PersistentError::InvalidPhase("token-map offset does not fit usize".to_string())
                })?;
                checked_code(
                    "write token-map range",
                    self.ops
                        .bo_write_range(self.cus[cu].token_map_bo, &range.bytes, offset),
                )?;
                checked_code(
                    "sync token-map range",
                    self.ops.bo_sync(
                        self.cus[cu].token_map_bo,
                        XRT_BO_SYNC_TO_DEVICE,
                        range.bytes.len(),
                        offset,
                    ),
                )?;
            }
            timings.activation_sync_us = timings
                .activation_sync_us
                .saturating_add(elapsed_us(activation_sync_start));
            let producer_publish_start = Instant::now();
            self.reg_write(cu, IQ1S_REG_COMMAND_PRODUCER_OFFSET, next_producer)?;
            timings.ring_publish_us = timings
                .ring_publish_us
                .saturating_add(elapsed_us(producer_publish_start));
            let doorbell_start = Instant::now();
            self.reg_write(cu, IQ1S_REG_DOORBELL_OFFSET, 1)?;
            timings.doorbell_us = timings
                .doorbell_us
                .saturating_add(elapsed_us(doorbell_start));
            self.cus[cu].command_producer = next_producer;
        }

        let deadline = Instant::now() + Duration::from_millis(u64::from(self.config.timeout_ms));
        let mut completed: [Vec<Iq1sCompletion>; ARENA_BANK_COUNT] =
            std::array::from_fn(|_| Vec::new());
        let mut results: [Vec<HostRange>; ARENA_BANK_COUNT] = std::array::from_fn(|_| Vec::new());
        for cu in 0..ARENA_BANK_COUNT {
            let expected_end = self.cus[cu]
                .completion_consumer
                .wrapping_add(expected[cu].len() as u32);
            let mut backoff = 1u64;
            let device_wait_start = Instant::now();
            let observed_producer = loop {
                let fault_code = self.reg_read(cu, IQ1S_REG_FAULT_CODE_OFFSET)?;
                if fault_code != IQ1S_FAULT_CODE_NONE {
                    let fault = self.hardware_fault_snapshot(cu, fault_code)?;
                    return self.poison(fault);
                }
                let producer = self.reg_read(cu, IQ1S_REG_COMPLETION_PRODUCER_OFFSET)?;
                if producer.wrapping_sub(expected_end) < (1u32 << 31) {
                    break producer;
                }
                if Instant::now() >= deadline {
                    return self.poison(PersistentFault {
                        cu: Some(cu),
                        operation: "completion poll",
                        detail: format!("timeout after {} ms", self.config.timeout_ms),
                    });
                }
                std::thread::sleep(Duration::from_micros(backoff));
                backoff = (backoff * 2).min(MAX_BACKOFF_US);
            };
            timings.device_wait_us = timings
                .device_wait_us
                .saturating_add(elapsed_us(device_wait_start));
            if observed_producer != expected_end {
                return self.poison(PersistentFault {
                    cu: Some(cu),
                    operation: "completion producer",
                    detail: format!(
                        "producer={observed_producer} expected={expected_end}; no other submission may be in flight"
                    ),
                });
            }
            let completion_sync_start = Instant::now();
            for (offset, bytes) in Self::descriptor_ranges(
                self.cus[cu].completion_consumer,
                expected[cu].len(),
                self.config.command_capacity,
            ) {
                checked_code(
                    "sync completion ring",
                    self.ops.bo_sync(
                        self.cus[cu].completion_bo,
                        XRT_BO_SYNC_FROM_DEVICE,
                        bytes,
                        offset,
                    ),
                )?;
                let end = offset.checked_add(bytes).ok_or_else(|| {
                    PersistentError::InvalidPhase("completion ring range overflow".to_string())
                })?;
                checked_code(
                    "read completion ring",
                    self.ops.bo_read_range(
                        self.cus[cu].completion_bo,
                        &mut self.cus[cu].completion_shadow[offset..end],
                        offset,
                    ),
                )?;
            }
            timings.completion_sync_us = timings
                .completion_sync_us
                .saturating_add(elapsed_us(completion_sync_start));
            for (index, command) in expected[cu].iter().enumerate() {
                let counter = self.cus[cu].completion_consumer.wrapping_add(index as u32);
                let command_counter = self.cus[cu]
                    .command_producer
                    .wrapping_sub(expected[cu].len() as u32)
                    .wrapping_add(index as u32);
                let slot = counter & (self.config.command_capacity - 1);
                let offset = slot as usize * IQ1S_COMPLETION_BYTES;
                let completion = struct_from_bytes::<Iq1sCompletion>(
                    &self.cus[cu].completion_shadow[offset..offset + IQ1S_COMPLETION_BYTES],
                )?;
                let mismatch = completion.magic != IQ1S_COMPLETION_MAGIC
                    || completion.abi_version != IQ1S_ABI_VERSION as u16
                    || completion.completion_bytes != IQ1S_COMPLETION_BYTES as u16
                    || completion.status != IQ1S_COMPLETION_STATUS_OK
                    || completion.fault_code != IQ1S_FAULT_CODE_NONE
                    || completion.session_generation != self.generation
                    || completion.transaction_id != command.transaction_id
                    || completion.program_id != command.program_id
                    || completion.trace_id != command.trace_id
                    || completion.layer_id != command.layer_id
                    || completion.phase != command.phase
                    || usize::from(completion.cu_id) != cu
                    || completion.expert_id != command.expert_id
                    || completion.lane_mask != command.lane_mask
                    || completion.rows_completed != command.row_count as u16
                    || completion.descriptor_crc32 != command.crc32
                    || completion.command_index != command_counter
                    || completion.result_fence == 0;
                if mismatch {
                    return self.poison(PersistentFault {
                        cu: Some(cu),
                        operation: "completion validation",
                        detail: format!("completion {counter} does not match its descriptor"),
                    });
                }
                completed[cu].push(completion);
            }
            self.cus[cu].completion_consumer = expected_end;
            self.cus[cu].command_consumer = self.cus[cu].command_producer;
            self.reg_write(cu, IQ1S_REG_COMPLETION_CONSUMER_OFFSET, expected_end)?;
            let result_copy_start = Instant::now();
            for (offset, bytes) in &result_ranges[cu] {
                checked_code(
                    "sync result range",
                    self.ops.bo_sync(
                        self.cus[cu].output_bo,
                        XRT_BO_SYNC_FROM_DEVICE,
                        *bytes,
                        *offset,
                    ),
                )?;
                let mut result = vec![0u8; *bytes];
                checked_code(
                    "read result range",
                    self.ops
                        .bo_read_range(self.cus[cu].output_bo, &mut result, *offset),
                )?;
                results[cu].push(HostRange {
                    offset: *offset as u64,
                    bytes: result,
                });
                self.dma.result_ranges += 1;
            }
            timings.result_copy_us = timings
                .result_copy_us
                .saturating_add(elapsed_us(result_copy_start));
        }
        if self.measured
            && (self.dma.weight_ranges != self.measurement_baseline.weight_ranges
                || self.dma.weight_bytes != self.measurement_baseline.weight_bytes)
        {
            return self.poison(PersistentFault {
                cu: None,
                operation: "weight residency",
                detail: "weight DMA occurred during submit".to_string(),
            });
        }
        let expanded =
            phase
                .programs
                .iter()
                .fold(ExpandedIq1sCounts::default(), |mut total, program| {
                    total.blocks = total.blocks.saturating_add(program.expanded.blocks);
                    total.grid_passes = total
                        .grid_passes
                        .saturating_add(program.expanded.grid_passes);
                    total.delta_passes = total
                        .delta_passes
                        .saturating_add(program.expanded.delta_passes);
                    total
                });
        timings.phase_wall_us = elapsed_us(phase_wall_start);
        Ok(CompletedLayerPhase {
            transaction_id: phase.transaction_id,
            semantic_sha256: phase.semantic_sha256,
            completions: completed,
            results,
            expanded,
            dma: self.dma.checked_delta(dma_before)?,
            timings,
        })
    }

    pub(crate) fn shutdown(&mut self) -> Result<(), PersistentError> {
        if self.closed {
            return Ok(());
        }
        for cu in 0..ARENA_BANK_COUNT {
            self.reg_write(cu, IQ1S_REG_CONTROL_OFFSET, CONTROL_SHUTDOWN)?;
        }
        let deadline = Instant::now() + Duration::from_millis(u64::from(self.config.timeout_ms));
        for cu in 0..ARENA_BANK_COUNT {
            loop {
                if self.reg_read(cu, IQ1S_REG_QUIESCENT_OFFSET)? == 1 {
                    break;
                }
                if Instant::now() >= deadline {
                    return Err(PersistentError::Shutdown(format!(
                        "CU {cu} did not become quiescent"
                    )));
                }
                std::thread::yield_now();
            }
        }
        for cu in &self.cus {
            for chunk in &cu.arena {
                checked_code("free arena BO", self.ops.bo_free(chunk.bo))?;
            }
            for bo in cu.runtime_bos() {
                checked_code("free runtime BO", self.ops.bo_free(bo))?;
            }
            checked_code(
                "xcl_close_context",
                self.ops
                    .xcl_close_context(self.native_device, &self.xclbin_uuid, cu.ip_index),
            )?;
        }
        self.ops.xcl_close(self.native_device);
        checked_code("device_close", self.ops.device_close(self.device))?;
        self.closed = true;
        Ok(())
    }
}

impl<O: XrtOps> Drop for PersistentIq1sPool<O> {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r#impl::iq1s_layer_trace::{
        compile_layer_phase, ActivationRange, LayerPhase, LayerPhasePlan, SemanticIq1sCommand,
    };
    use crate::r#impl::iq1s_tmatmul::{
        raw_component_dots, reconstruct_from_raw, validated_grid, Iq1sBlock, Q8_1Block,
    };
    use crate::r#impl::iq1s_weight_arena::ArenaShard;
    use crate::r#impl::iq1s_weight_registry::{Iq1sExpertRole, Iq1sTensorIdentity};
    use crate::r#impl::xrt_tmatmul::RealXrt;
    use std::cell::RefCell;
    use std::collections::{BTreeMap, HashMap};
    use std::ffi::CStr;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::Arc;

    const DEVICE: usize = 1;
    const NATIVE_DEVICE: usize = 2;
    const FIRST_BO: usize = 100;
    const MAX_SMOKE_MISMATCH_SAMPLES: usize = 16;

    #[derive(Debug, serde::Serialize)]
    struct SmokeResultMismatch {
        cu: usize,
        generation: usize,
        row: usize,
        expected_bits: u32,
        actual_bits: u32,
    }

    #[derive(Debug)]
    struct SmokeResultComparison {
        rows_checked: u64,
        mismatch_count: u64,
        actual_bits_histogram: BTreeMap<u32, u64>,
        first_mismatches: Vec<SmokeResultMismatch>,
    }

    fn compare_smoke_result_bytes(
        cu: usize,
        generation: usize,
        expected_bits: u32,
        bytes: &[u8],
    ) -> SmokeResultComparison {
        assert_eq!(bytes.len() % 4, 0, "smoke result must contain whole f32 rows");
        let mut comparison = SmokeResultComparison {
            rows_checked: 0,
            mismatch_count: 0,
            actual_bits_histogram: BTreeMap::new(),
            first_mismatches: Vec::new(),
        };
        for (row, actual) in bytes.chunks_exact(4).enumerate() {
            let actual_bits = u32::from_le_bytes(actual.try_into().expect("four-byte f32"));
            comparison.rows_checked += 1;
            *comparison
                .actual_bits_histogram
                .entry(actual_bits)
                .or_default() += 1;
            if actual_bits != expected_bits {
                comparison.mismatch_count += 1;
                if comparison.first_mismatches.len() < MAX_SMOKE_MISMATCH_SAMPLES {
                    comparison.first_mismatches.push(SmokeResultMismatch {
                        cu,
                        generation,
                        row,
                        expected_bits,
                        actual_bits,
                    });
                }
            }
        }
        comparison
    }

    fn classify_smoke_signature(
        rows_checked: u64,
        mismatch_count: u64,
        actual_bits_histogram: &BTreeMap<u32, u64>,
        zero_grid_expected_bits: u32,
    ) -> &'static str {
        if mismatch_count == 0 {
            "matches_reference"
        } else if rows_checked != 0
            && mismatch_count == rows_checked
            && actual_bits_histogram.len() == 1
            && actual_bits_histogram.get(&zero_grid_expected_bits) == Some(&rows_checked)
        {
            "matches_zero_grid_oracle"
        } else {
            "unclassified_numerical_mismatch"
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Event {
        DeviceOpen,
        LoadXclbin,
        XclOpen,
        OpenContext(u32),
        BoAlloc {
            bo: usize,
            bytes: usize,
            flags: u64,
            group: u32,
        },
        BoCopy {
            destination: usize,
            source: usize,
            bytes: usize,
            destination_offset: usize,
            source_offset: usize,
        },
        BoSync {
            bo: usize,
            direction: i32,
            offset: usize,
            bytes: usize,
        },
        BoWriteRange {
            bo: usize,
            offset: usize,
            bytes: Vec<u8>,
        },
        BoReadRange {
            bo: usize,
            offset: usize,
            bytes: usize,
        },
        RegisterRead {
            cu: u32,
            offset: u32,
        },
        RegisterWrite {
            cu: u32,
            offset: u32,
            value: u32,
        },
        CloseContext(u32),
        DeviceClose,
    }

    #[derive(Debug, Clone, Copy)]
    enum CompletionMutation {
        Transaction,
        Program,
        Trace,
        Generation,
        Cu,
        Crc,
    }

    #[derive(Default)]
    struct FakeState {
        events: Vec<Event>,
        next_bo: usize,
        memories: HashMap<usize, Vec<u8>>,
        addresses: HashMap<usize, u64>,
        registers: HashMap<(u32, u32), u32>,
        doorbell_consumer: HashMap<u32, u32>,
        completion_mutation: Option<CompletionMutation>,
        hold_completions: bool,
        fail_read_bo: Option<usize>,
    }

    struct FakeXrt {
        state: RefCell<FakeState>,
    }

    impl FakeXrt {
        fn new() -> Self {
            let mut state = FakeState {
                next_bo: FIRST_BO,
                ..Default::default()
            };
            for cu in 0..4 {
                state
                    .registers
                    .insert((cu, IQ1S_REG_ABI_MAGIC_OFFSET as u32), IQ1S_REGISTER_MAGIC);
                state
                    .registers
                    .insert((cu, IQ1S_REG_ABI_VERSION_OFFSET as u32), IQ1S_ABI_VERSION);
                state
                    .registers
                    .insert((cu, IQ1S_REG_QUIESCENT_OFFSET as u32), 1);
                state
                    .registers
                    .insert((cu, IQ1S_REG_FAULT_CODE_OFFSET as u32), 0);
                state
                    .registers
                    .insert((cu, IQ1S_REG_COMMAND_CONSUMER_OFFSET as u32), 0);
                state
                    .registers
                    .insert((cu, IQ1S_REG_COMPLETION_PRODUCER_OFFSET as u32), 0);
            }
            Self {
                state: RefCell::new(state),
            }
        }

        fn events(&self) -> Vec<Event> {
            self.state.borrow().events.clone()
        }

        fn set_completion_mutation(&self, mutation: CompletionMutation) {
            self.state.borrow_mut().completion_mutation = Some(mutation);
        }

        fn fail_read(&self, bo: Handle) {
            self.state.borrow_mut().fail_read_bo = Some(bo as usize);
        }

        fn set_hold_completions(&self, hold: bool) {
            self.state.borrow_mut().hold_completions = hold;
        }

        fn release_completions(&self) {
            let mut state = self.state.borrow_mut();
            state.hold_completions = false;
            for cu in 0..ARENA_BANK_COUNT as u32 {
                Self::ring_doorbell(&mut state, cu);
            }
        }

        fn set_fault(&self, cu: usize, fault_code: u32) {
            let mut state = self.state.borrow_mut();
            state
                .registers
                .insert((cu as u32, IQ1S_REG_FAULT_CODE_OFFSET as u32), fault_code);
            state.registers.insert(
                (cu as u32, IQ1S_REG_FAULT_DETAIL_LO_OFFSET as u32),
                0x5566_7788,
            );
            state.registers.insert(
                (cu as u32, IQ1S_REG_FAULT_DETAIL_HI_OFFSET as u32),
                0x1122_3344,
            );
        }

        fn set_persisted_ring_state(
            &self,
            cu: usize,
            command_consumer: u32,
            completion_producer: u32,
            fault_code: u32,
        ) {
            let mut state = self.state.borrow_mut();
            state.registers.insert(
                (cu as u32, IQ1S_REG_COMMAND_CONSUMER_OFFSET as u32),
                command_consumer,
            );
            state.registers.insert(
                (cu as u32, IQ1S_REG_COMPLETION_PRODUCER_OFFSET as u32),
                completion_producer,
            );
            state.registers.insert(
                (cu as u32, IQ1S_REG_FAULT_CODE_OFFSET as u32),
                fault_code,
            );
            state
                .doorbell_consumer
                .insert(cu as u32, command_consumer);
        }

        fn ring_doorbell(state: &mut FakeState, cu: u32) {
            if state.hold_completions {
                return;
            }
            let command_producer = state.registers[&(cu, IQ1S_REG_COMMAND_PRODUCER_OFFSET as u32)];
            let start = *state.doorbell_consumer.get(&cu).unwrap_or(&0);
            let completion_start =
                state.registers[&(cu, IQ1S_REG_COMPLETION_PRODUCER_OFFSET as u32)];
            let capacity = state.registers[&(cu, IQ1S_REG_COMMAND_CAPACITY_OFFSET as u32)];
            let command_address =
                u64::from(state.registers[&(cu, IQ1S_REG_COMMAND_BASE_LO_OFFSET as u32)])
                    | (u64::from(state.registers[&(cu, IQ1S_REG_COMMAND_BASE_HI_OFFSET as u32)])
                        << 32);
            let completion_address =
                u64::from(state.registers[&(cu, IQ1S_REG_COMPLETION_BASE_LO_OFFSET as u32)])
                    | (u64::from(
                        state.registers[&(cu, IQ1S_REG_COMPLETION_BASE_HI_OFFSET as u32)],
                    ) << 32);
            let output_address =
                u64::from(state.registers[&(cu, IQ1S_REG_RESULT_BASE_LO_OFFSET as u32)])
                    | (u64::from(state.registers[&(cu, IQ1S_REG_RESULT_BASE_HI_OFFSET as u32)])
                        << 32);
            let command_bo = *state
                .addresses
                .iter()
                .find(|(_, address)| **address == command_address)
                .unwrap()
                .0;
            let completion_bo = *state
                .addresses
                .iter()
                .find(|(_, address)| **address == completion_address)
                .unwrap()
                .0;
            let output_bo = *state
                .addresses
                .iter()
                .find(|(_, address)| **address == output_address)
                .unwrap()
                .0;
            let commands = state.memories[&command_bo].clone();
            for (completion_index, counter) in (start..command_producer).enumerate() {
                let command_slot = counter & (capacity - 1);
                let command_offset = command_slot as usize * IQ1S_COMMAND_BYTES;
                let command = struct_from_bytes::<Iq1sCommand>(
                    &commands[command_offset..command_offset + IQ1S_COMMAND_BYTES],
                )
                .unwrap();
                let output_offset =
                    usize::try_from(command.output_offset - output_address).unwrap();
                let output_end = output_offset + command.output_bytes as usize;
                state.memories.get_mut(&output_bo).unwrap()[output_offset..output_end]
                    .fill(0x40 + cu as u8);
                let mut completion = Iq1sCompletion {
                    magic: IQ1S_COMPLETION_MAGIC,
                    abi_version: IQ1S_ABI_VERSION as u16,
                    completion_bytes: IQ1S_COMPLETION_BYTES as u16,
                    status: IQ1S_COMPLETION_STATUS_OK,
                    fault_code: 0,
                    session_generation: command.session_generation,
                    transaction_id: command.transaction_id,
                    program_id: command.program_id,
                    trace_id: command.trace_id,
                    layer_id: command.layer_id,
                    phase: command.phase,
                    role: command.role,
                    cu_id: cu as u16,
                    expert_id: command.expert_id,
                    lane_mask: command.lane_mask,
                    rows_completed: command.row_count as u16,
                    descriptor_crc32: command.crc32,
                    command_index: counter,
                    cycles: 100,
                    ddr_read_bytes: 1000,
                    ddr_write_bytes: 100,
                    iq1s_blocks: 1,
                    grid_passes: 8,
                    delta_passes: 8,
                    result_fence: completion_start
                        .wrapping_add(completion_index as u32) as u64
                        + 1,
                    fault_detail: 0,
                };
                if let Some(mutation) = state.completion_mutation.take() {
                    match mutation {
                        CompletionMutation::Transaction => completion.transaction_id ^= 1,
                        CompletionMutation::Program => completion.program_id ^= 1,
                        CompletionMutation::Trace => completion.trace_id ^= 1,
                        CompletionMutation::Generation => completion.session_generation ^= 1,
                        CompletionMutation::Cu => completion.cu_id ^= 1,
                        CompletionMutation::Crc => completion.descriptor_crc32 ^= 1,
                    }
                }
                let completion_counter =
                    completion_start.wrapping_add(completion_index as u32);
                let completion_slot = completion_counter & (capacity - 1);
                let completion_offset = completion_slot as usize * IQ1S_COMPLETION_BYTES;
                state.memories.get_mut(&completion_bo).unwrap()
                    [completion_offset..completion_offset + IQ1S_COMPLETION_BYTES]
                    .copy_from_slice(struct_bytes(&completion));
            }
            state.doorbell_consumer.insert(cu, command_producer);
            state.registers.insert(
                (cu, IQ1S_REG_COMMAND_CONSUMER_OFFSET as u32),
                command_producer,
            );
            state.registers.insert(
                (cu, IQ1S_REG_COMPLETION_PRODUCER_OFFSET as u32),
                completion_start.wrapping_add(command_producer.wrapping_sub(start)),
            );
        }
    }

    impl XrtOps for FakeXrt {
        fn device_open(&self, _index: u32) -> Handle {
            self.state.borrow_mut().events.push(Event::DeviceOpen);
            DEVICE as Handle
        }
        fn device_close(&self, _device: Handle) -> i32 {
            self.state.borrow_mut().events.push(Event::DeviceClose);
            0
        }
        fn load_xclbin_file(&self, _device: Handle, _path: &CStr) -> i32 {
            self.state.borrow_mut().events.push(Event::LoadXclbin);
            0
        }
        fn get_xclbin_uuid(&self, _device: Handle, uuid: &mut Xuid) -> i32 {
            *uuid = [7; 16];
            0
        }
        fn kernel_open_exclusive(&self, _: Handle, _: &Xuid, _: &CStr) -> Handle {
            3 as Handle
        }
        fn kernel_close(&self, _: Handle) -> i32 {
            0
        }
        fn kernel_arg_group_id(&self, _: Handle, _: i32) -> i32 {
            0
        }
        fn kernel_read_register(&self, _: Handle, _: u32, _: &mut u32) -> i32 {
            -1
        }
        fn kernel_write_register(&self, _: Handle, _: u32, _: u32) -> i32 {
            -1
        }
        fn xcl_open(&self, _index: u32) -> Handle {
            self.state.borrow_mut().events.push(Event::XclOpen);
            NATIVE_DEVICE as Handle
        }
        fn xcl_close(&self, _device: Handle) {}
        fn xcl_ip_name_to_index(&self, _: Handle, name: &CStr) -> i32 {
            let name = name.to_string_lossy();
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
        fn xcl_open_context(&self, _: Handle, _: &Xuid, index: u32, _: bool) -> i32 {
            self.state
                .borrow_mut()
                .events
                .push(Event::OpenContext(index));
            0
        }
        fn xcl_close_context(&self, _: Handle, _: &Xuid, index: u32) -> i32 {
            self.state
                .borrow_mut()
                .events
                .push(Event::CloseContext(index));
            0
        }
        fn xcl_reg_read(&self, _: Handle, index: u32, offset: u32, value: &mut u32) -> i32 {
            let mut state = self.state.borrow_mut();
            state.events.push(Event::RegisterRead { cu: index, offset });
            *value = *state.registers.get(&(index, offset)).unwrap_or(&0);
            0
        }
        fn xcl_reg_write(&self, _: Handle, index: u32, offset: u32, value: u32) -> i32 {
            let mut state = self.state.borrow_mut();
            state.events.push(Event::RegisterWrite {
                cu: index,
                offset,
                value,
            });
            state.registers.insert((index, offset), value);
            if offset == IQ1S_REG_CONTROL_OFFSET as u32 && value == 4 {
                state
                    .registers
                    .insert((index, IQ1S_REG_FAULT_CODE_OFFSET as u32), 0);
            }
            if offset == IQ1S_REG_DOORBELL_OFFSET as u32 {
                Self::ring_doorbell(&mut state, index);
            }
            0
        }
        fn bo_alloc(&self, _: Handle, size: usize, flags: u64, group: u32) -> Handle {
            let mut state = self.state.borrow_mut();
            let bo = state.next_bo;
            state.next_bo += 1;
            state
                .memories
                .insert(bo, vec![0; size.min(8 * 1024 * 1024)]);
            state.addresses.insert(bo, (bo as u64) << 20);
            state.events.push(Event::BoAlloc {
                bo,
                bytes: size,
                flags,
                group,
            });
            bo as Handle
        }
        fn bo_free(&self, _: Handle) -> i32 {
            0
        }
        fn bo_address(&self, bo: Handle) -> u64 {
            self.state.borrow().addresses[&(bo as usize)]
        }
        fn bo_write_range(&self, bo: Handle, bytes: &[u8], offset: usize) -> i32 {
            let mut state = self.state.borrow_mut();
            let memory = state.memories.get_mut(&(bo as usize)).unwrap();
            let Some(end) = offset.checked_add(bytes.len()) else {
                return -1;
            };
            if end > memory.len() {
                return -1;
            }
            memory[offset..end].copy_from_slice(bytes);
            state.events.push(Event::BoWriteRange {
                bo: bo as usize,
                offset,
                bytes: bytes.to_vec(),
            });
            0
        }

        fn bo_read_range(&self, bo: Handle, bytes: &mut [u8], offset: usize) -> i32 {
            let mut state = self.state.borrow_mut();
            if state.fail_read_bo == Some(bo as usize) {
                return -1;
            }
            let memory = &state.memories[&(bo as usize)];
            let Some(end) = offset.checked_add(bytes.len()) else {
                return -1;
            };
            if end > memory.len() {
                return -1;
            }
            bytes.copy_from_slice(&memory[offset..end]);
            state.events.push(Event::BoReadRange {
                bo: bo as usize,
                offset,
                bytes: bytes.len(),
            });
            0
        }
        fn bo_copy(
            &self,
            destination: Handle,
            source: Handle,
            size: usize,
            destination_offset: usize,
            source_offset: usize,
        ) -> i32 {
            let mut state = self.state.borrow_mut();
            let source_bytes = {
                let Some(source_memory) = state.memories.get(&(source as usize)) else {
                    return -1;
                };
                let Some(source_end) = source_offset.checked_add(size) else {
                    return -1;
                };
                if source_end > source_memory.len() {
                    return -1;
                }
                source_memory[source_offset..source_end].to_vec()
            };
            let Some(destination_memory) = state.memories.get_mut(&(destination as usize)) else {
                return -1;
            };
            let Some(destination_end) = destination_offset.checked_add(size) else {
                return -1;
            };
            if destination_end > destination_memory.len() {
                return -1;
            }
            destination_memory[destination_offset..destination_end].copy_from_slice(&source_bytes);
            state.events.push(Event::BoCopy {
                destination: destination as usize,
                source: source as usize,
                bytes: size,
                destination_offset,
                source_offset,
            });
            0
        }
        fn bo_sync(&self, bo: Handle, direction: i32, size: usize, offset: usize) -> i32 {
            self.state.borrow_mut().events.push(Event::BoSync {
                bo: bo as usize,
                direction,
                offset,
                bytes: size,
            });
            0
        }
    }

    fn fixture_phase(transaction_id: u64, distinct: bool) -> CompiledLayerPhase {
        let role = Iq1sExpertRole::Gate;
        let tensor = Arc::new(Iq1sTensorIdentity {
            canonical_path: PathBuf::from("/tmp/qwen.gguf"),
            file_offset: 0,
            nbytes: 419_430_400,
            name: "blk.7.ffn_gate_exps.weight".to_string(),
            layer: 7,
            ne: [4096, 1024, 512, 1],
            nb: [50, 800, 819_200, 419_430_400],
            role,
            model_sha256: [1; 32],
            content_sha256: [2; 32],
            device: 1,
            inode: 2,
            modified_ns: 3,
        });
        let token_groups = if distinct {
            vec![vec![0], vec![1]]
        } else {
            vec![vec![0, 1]]
        };
        let mut commands = Vec::new();
        for tokens in token_groups {
            let expert = if distinct { tokens[0] as u16 } else { 0 };
            let mask = tokens.iter().fold(0u16, |mask, token| mask | (1 << token));
            for bank in 0..4 {
                commands.push(SemanticIq1sCommand {
                    layer_id: 7,
                    phase: LayerPhase::PhaseA,
                    role,
                    expert_id: expert,
                    lane_mask: mask,
                    token_ids: tokens.clone(),
                    input_offset: 0x2000 + u64::from(expert) * 8192,
                    output_offset: 0x8000 + u64::from(expert) * 8192,
                    token_map_offset: 0x3000 + u64::from(expert) * 64,
                    row_shard: ArenaShard {
                        tensor: tensor.clone(),
                        expert,
                        bank,
                        row_start: u32::from(bank) * 256,
                        row_count: 256,
                        superblock: 0,
                        offset: u64::from(expert) * 1024 * 1024,
                        bytes: 204_800,
                        sha256: [3; 32],
                    },
                });
            }
        }
        compile_layer_phase(
            &LayerPhasePlan {
                transaction_id,
                phase: LayerPhase::PhaseA,
                commands,
                activations: vec![ActivationRange {
                    cuda_ptr: 0x10000,
                    slab_offset: 0x2000,
                    bytes: if distinct { 32768 } else { 16384 },
                    stream: 1,
                    source_identity_sha256: [0x51; 32],
                }],
            },
            "compiler",
            QWEN_MODEL_CONTEXT_LIMIT,
        )
        .unwrap()
    }

    fn fixture_buffers(distinct: bool) -> PhaseBuffers {
        let activation_bytes = if distinct { 32768 } else { 16384 };
        PhaseBuffers {
            activations: vec![HostRange {
                offset: 0x2000,
                bytes: vec![0x11; activation_bytes],
            }],
            token_maps: if distinct {
                vec![
                    HostRange {
                        offset: 0x3000,
                        bytes: 0u32.to_le_bytes().to_vec(),
                    },
                    HostRange {
                        offset: 0x3040,
                        bytes: 1u32.to_le_bytes().to_vec(),
                    },
                ]
            } else {
                vec![HostRange {
                    offset: 0x3000,
                    bytes: [0u32, 1u32]
                        .into_iter()
                        .flat_map(u32::to_le_bytes)
                        .collect(),
                }]
            },
        }
    }

    fn pool_with_ops(capacity: u32, ops: FakeXrt) -> PersistentIq1sPool<FakeXrt> {
        let bytes = vec![0x5a; 2 * 1024 * 1024];
        let hash: [u8; 32] = Sha256::digest(&bytes).into();
        let shard_hash: [u8; 32] = Sha256::digest(vec![0x5a; 204_800]).into();
        let chunks = (0..4)
            .map(|bank| ArenaChunkSpec {
                bank,
                logical_offset: 0,
                bytes: bytes.len(),
                sha256: hash,
                shards: (0..2)
                    .map(|expert| ArenaShardSpec {
                        bank,
                        logical_offset: u64::from(expert) * 1024 * 1024,
                        bytes: 204_800,
                        layer_id: 7,
                        role: IQ1S_ROLE_GATE as u16,
                        expert_id: expert,
                        row_start: u32::from(bank) * 256,
                        row_count: 256,
                        sha256: shard_hash,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        PersistentIq1sPool::open(
            ops,
            PersistentIq1sConfig::checked(
                PathBuf::from("/tmp/qwen.xclbin"),
                0,
                Some(capacity),
                5_000,
            )
            .unwrap(),
            9,
            &chunks,
            |_| Ok(bytes.clone()),
            |_| Ok(()),
        )
        .unwrap()
    }

    fn pool(capacity: u32) -> PersistentIq1sPool<FakeXrt> {
        pool_with_ops(capacity, FakeXrt::new())
    }

    #[test]
    fn xrt_iq1s_persistent_two_ticket_slots_overlap_and_reject_a_third() {
        let mut pool = pool(8);
        pool.ops.set_hold_completions(true);
        let phase_a = fixture_phase(17, false);
        let phase_b = fixture_phase(18, false);
        let phase_c = fixture_phase(19, false);
        let buffers = fixture_buffers(false);

        let first = pool.prepare_ticket(&phase_a, &buffers).unwrap();
        let second = pool.prepare_ticket(&phase_b, &buffers).unwrap();
        assert_ne!(first.slot(), second.slot());
        assert!(matches!(
            pool.prepare_ticket(&phase_c, &buffers),
            Err(PersistentError::TicketSlotsFull)
        ));

        pool.publish_ticket(&first).unwrap();
        pool.publish_ticket(&second).unwrap();
        assert_eq!(pool.poll_ticket(&first).unwrap(), TicketPoll::Pending);
        pool.ops.release_completions();
        assert_eq!(pool.poll_ticket(&second).unwrap(), TicketPoll::Complete);
        assert_eq!(pool.poll_ticket(&first).unwrap(), TicketPoll::Complete);
        assert_eq!(pool.collect_ticket(&first).unwrap().transaction_id, 17);
        assert_eq!(pool.collect_ticket(&second).unwrap().transaction_id, 18);
    }

    #[test]
    fn xrt_iq1s_persistent_ticket_slots_use_disjoint_physical_slabs() {
        let mut pool = pool(8);
        let phase_a = fixture_phase(20, false);
        let phase_b = fixture_phase(21, false);
        let buffers = fixture_buffers(false);
        let activation_bo = pool.cus[0].activation_bo as usize;

        let first = pool.prepare_ticket(&phase_a, &buffers).unwrap();
        let second = pool.prepare_ticket(&phase_b, &buffers).unwrap();
        assert_eq!(first.slot(), 0);
        assert_eq!(second.slot(), 1);
        let offsets = pool
            .ops
            .events()
            .into_iter()
            .filter_map(|event| match event {
                Event::BoWriteRange { bo, offset, .. } if bo == activation_bo => Some(offset),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(offsets.contains(&0x2000));
        assert!(offsets.contains(&(ACTIVATION_SLOT_BYTES + 0x2000)));
    }

    #[test]
    fn xrt_iq1s_persistent_poll_rotates_across_all_four_cus() {
        let mut pool = pool(8);
        pool.ops.set_hold_completions(true);
        let ticket = pool
            .prepare_ticket(&fixture_phase(22, false), &fixture_buffers(false))
            .unwrap();
        pool.publish_ticket(&ticket).unwrap();
        let baseline = pool.ops.events().len();
        assert_eq!(pool.poll_ticket(&ticket).unwrap(), TicketPoll::Pending);
        assert_eq!(pool.poll_ticket(&ticket).unwrap(), TicketPoll::Pending);
        let order = pool.ops.events()[baseline..]
            .iter()
            .filter_map(|event| match event {
                Event::RegisterRead { cu, offset }
                    if *offset == IQ1S_REG_COMPLETION_PRODUCER_OFFSET as u32 =>
                {
                    Some(*cu)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(order, vec![0, 1, 2, 3, 1, 2, 3, 0]);
    }

    #[test]
    fn xrt_iq1s_persistent_stale_ticket_generation_poisons_live_slot() {
        let mut pool = pool(8);
        let first = pool
            .prepare_ticket(&fixture_phase(23, false), &fixture_buffers(false))
            .unwrap();
        pool.publish_ticket(&first).unwrap();
        assert_eq!(pool.poll_ticket(&first).unwrap(), TicketPoll::Complete);
        pool.collect_ticket(&first).unwrap();
        let second = pool
            .prepare_ticket(&fixture_phase(24, false), &fixture_buffers(false))
            .unwrap();
        assert_eq!(first.slot(), second.slot());
        assert!(matches!(
            pool.poll_ticket(&first),
            Err(PersistentError::InvalidPhase(_))
        ));
        assert!(matches!(
            pool.publish_ticket(&second),
            Err(PersistentError::Poisoned(_))
        ));
    }

    #[test]
    fn xrt_iq1s_persistent_fault_poisons_both_ticket_slots() {
        let mut pool = pool(8);
        pool.ops.set_hold_completions(true);
        let first = pool
            .prepare_ticket(&fixture_phase(25, false), &fixture_buffers(false))
            .unwrap();
        let second = pool
            .prepare_ticket(&fixture_phase(26, false), &fixture_buffers(false))
            .unwrap();
        pool.publish_ticket(&first).unwrap();
        pool.publish_ticket(&second).unwrap();
        pool.ops.set_fault(2, 7);
        let error = pool.poll_ticket(&first).unwrap_err().to_string();
        assert!(error.contains("fault_code=7"), "{error}");
        assert!(
            error.contains("fault_detail=0x1122334455667788"),
            "{error}"
        );
        assert!(error.contains("command="), "{error}");
        assert!(error.contains("completion="), "{error}");
        assert!(matches!(
            pool.poll_ticket(&second),
            Err(PersistentError::Poisoned(_))
        ));
    }

    #[test]
    fn xrt_iq1s_persistent_merges_adjacent_activation_dma_ranges() {
        let mut pool = pool(8);
        let phase = fixture_phase(27, false);
        let mut buffers = fixture_buffers(false);
        let bytes = buffers.activations.remove(0).bytes;
        buffers.activations = vec![
            HostRange {
                offset: 0x2000,
                bytes: bytes[..8192].to_vec(),
            },
            HostRange {
                offset: 0x4000,
                bytes: bytes[8192..].to_vec(),
            },
        ];
        let activation_bos = pool
            .cus
            .iter()
            .map(|cu| cu.activation_bo as usize)
            .collect::<Vec<_>>();
        pool.prepare_ticket(&phase, &buffers).unwrap();
        let syncs = pool
            .ops
            .events()
            .iter()
            .filter(|event| {
                matches!(event,
                Event::BoSync { bo, direction: XRT_BO_SYNC_TO_DEVICE, offset: 0x2000, bytes: 16384 }
                    if activation_bos.contains(bo))
            })
            .count();
        assert_eq!(syncs, ARENA_BANK_COUNT);
    }

    #[test]
    fn arena_manifest_records_exact_physical_shards() {
        let hash = [0x5au8; 32];
        let manifest = arena_manifest(
            7,
            &[ArenaChunk {
                logical_offset: 0,
                bytes: 800,
                bo: core::ptr::null_mut(),
                address: 0x1234_0000,
                sha256: hash,
                shards: vec![ResidentArenaShard {
                    address: 0x1234_0000,
                    bytes: 800,
                    layer_id: 7,
                    role: IQ1S_ROLE_GATE as u16,
                    expert_id: 17,
                    row_start: 0,
                    row_count: 1,
                    sha256: hash,
                }],
            }],
        )
        .unwrap();
        assert_eq!(&manifest[64..72], &0x1234_0000u64.to_le_bytes());
        assert_eq!(&manifest[72..80], &800u64.to_le_bytes());
        assert_eq!(&manifest[80..84], &7u32.to_le_bytes());
        assert_eq!(&manifest[84..86], &(IQ1S_ROLE_GATE as u16).to_le_bytes());
        assert_eq!(&manifest[86..88], &17u16.to_le_bytes());
        assert_eq!(&manifest[88..92], &0u32.to_le_bytes());
        assert_eq!(&manifest[92..96], &1u32.to_le_bytes());
        assert_eq!(&manifest[96..128], &hash);
    }

    #[test]
    fn xrt_iq1s_persistent_opens_once_starts_four_and_bounds_arena_chunks() {
        let mut pool = pool(4);
        let events = pool.ops.events();
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::DeviceOpen))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::LoadXclbin))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::OpenContext(_)))
                .count(),
            4
        );
        assert_eq!(events.iter().filter(|event| matches!(event, Event::RegisterWrite { offset, value: CONTROL_START, .. } if *offset == IQ1S_REG_CONTROL_OFFSET as u32)).count(), 4);
        for offset in [
            IQ1S_REG_PROGRAM_BASE_LO_OFFSET,
            IQ1S_REG_PROGRAM_BASE_HI_OFFSET,
            IQ1S_REG_ARENA_MANIFEST_BASE_LO_OFFSET,
            IQ1S_REG_ARENA_MANIFEST_BASE_HI_OFFSET,
            IQ1S_REG_ACTIVATION_BASE_LO_OFFSET,
            IQ1S_REG_ACTIVATION_BASE_HI_OFFSET,
            IQ1S_REG_RESULT_BASE_LO_OFFSET,
            IQ1S_REG_RESULT_BASE_HI_OFFSET,
            IQ1S_REG_TOKEN_MAP_BASE_LO_OFFSET,
            IQ1S_REG_TOKEN_MAP_BASE_HI_OFFSET,
            IQ1S_REG_MODEL_TAG_LO_OFFSET,
            IQ1S_REG_MODEL_TAG_HI_OFFSET,
            IQ1S_REG_ACTIVATION_BYTES_OFFSET,
            IQ1S_REG_RESULT_BYTES_OFFSET,
            IQ1S_REG_TOKEN_MAP_BYTES_OFFSET,
            IQ1S_REG_PROGRAM_BYTES_OFFSET,
            IQ1S_REG_ARENA_MANIFEST_BYTES_OFFSET,
            IQ1S_REG_CU_ID_OFFSET,
        ] {
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, Event::RegisterWrite { offset: actual, .. } if *actual == offset as u32))
                    .count(),
                4,
                "register 0x{offset:x} must be configured on all four CUs"
            );
        }
        for cu in 0..4u32 {
            assert!(events.iter().any(|event| matches!(event,
                Event::RegisterWrite { cu: actual_cu, offset, value }
                    if *actual_cu == cu && *offset == IQ1S_REG_CU_ID_OFFSET as u32 && *value == cu)));
        }
        assert!(events
            .iter()
            .filter_map(|event| match event {
                Event::BoAlloc { bytes, .. } => Some(*bytes),
                _ => None,
            })
            .all(|bytes| bytes <= ARENA_SUPERBLOCK_BYTES as usize));
        pool.shutdown().unwrap();
    }

    #[test]
    fn xrt_iq1s_persistent_rebinds_persisted_ring_counters_before_start() {
        let ops = FakeXrt::new();
        for cu in 0..ARENA_BANK_COUNT {
            ops.set_persisted_ring_state(cu, 37 + cu as u32, 29 + cu as u32, 11);
        }

        let mut pool = pool_with_ops(8, ops);
        let events = pool.ops.events();
        for cu in 0..ARENA_BANK_COUNT {
            let command_baseline = 37 + cu as u32;
            let completion_baseline = 29 + cu as u32;
            assert_eq!(pool.cus[cu].command_producer, command_baseline);
            assert_eq!(pool.cus[cu].command_published, command_baseline);
            assert_eq!(pool.cus[cu].command_consumer, command_baseline);
            assert_eq!(pool.cus[cu].completion_consumer, completion_baseline);

            let start = events
                .iter()
                .position(|event| matches!(event,
                    Event::RegisterWrite { cu: actual, offset, value: CONTROL_START }
                        if *actual == cu as u32 && *offset == IQ1S_REG_CONTROL_OFFSET as u32))
                .unwrap();
            assert!(events[..start].iter().any(|event| matches!(event,
                Event::RegisterWrite { cu: actual, offset, value: CONTROL_SHUTDOWN }
                    if *actual == cu as u32 && *offset == IQ1S_REG_CONTROL_OFFSET as u32)));
            assert!(events[..start].iter().any(|event| matches!(event,
                Event::RegisterWrite { cu: actual, offset, value }
                    if *actual == cu as u32
                        && *offset == IQ1S_REG_COMMAND_PRODUCER_OFFSET as u32
                        && *value == command_baseline)));
            assert!(events[..start].iter().any(|event| matches!(event,
                Event::RegisterWrite { cu: actual, offset, value }
                    if *actual == cu as u32
                        && *offset == IQ1S_REG_COMPLETION_CONSUMER_OFFSET as u32
                        && *value == completion_baseline)));
            assert!(events[..start].iter().any(|event| matches!(event,
                Event::RegisterWrite { cu: actual, offset, value: 4 }
                    if *actual == cu as u32 && *offset == IQ1S_REG_CONTROL_OFFSET as u32)));
            assert_eq!(
                pool.reg_read(cu, IQ1S_REG_FAULT_CODE_OFFSET).unwrap(),
                IQ1S_FAULT_CODE_NONE
            );
        }

        let phase = fixture_phase(31, false);
        let buffers = fixture_buffers(false);
        let ticket = pool.prepare_ticket(&phase, &buffers).unwrap();
        pool.publish_ticket(&ticket).unwrap();
        assert_eq!(pool.poll_ticket(&ticket).unwrap(), TicketPoll::Complete);
        pool.collect_ticket(&ticket).unwrap();
        for cu in 0..ARENA_BANK_COUNT {
            assert_eq!(pool.cus[cu].command_consumer, 38 + cu as u32);
            assert_eq!(pool.cus[cu].completion_consumer, 30 + cu as u32);
        }
        pool.shutdown().unwrap();
    }

    #[test]
    fn xrt_iq1s_persistent_weights_use_device_only_bos_and_bounded_staging() {
        let mut pool = pool(4);
        let resident = pool
            .cus
            .iter()
            .flat_map(|cu| cu.arena.iter().map(|chunk| chunk.bo as usize))
            .collect::<Vec<_>>();
        let staging = pool
            .cus
            .iter()
            .map(|cu| cu.arena_staging_bo as usize)
            .collect::<Vec<_>>();
        let events = pool.ops.events();

        assert_eq!(resident.len(), 4);
        assert_eq!(staging.len(), 4);
        for bo in &resident {
            assert!(events.iter().any(|event| matches!(event,
                Event::BoAlloc { bo: actual, flags: XRT_BO_FLAGS_DEVICE_ONLY, .. }
                    if actual == bo)));
        }
        for bo in &staging {
            assert!(events.iter().any(|event| matches!(event,
                Event::BoAlloc { bo: actual, flags: 0, bytes, .. }
                    if actual == bo && *bytes == 2 * 1024 * 1024)));
        }
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event,
                    Event::BoCopy {
                        destination,
                        source,
                        bytes: 2_097_152,
                        destination_offset: 0,
                        source_offset: 0,
                    } if resident.contains(destination) && staging.contains(source)))
                .count(),
            4
        );
        pool.shutdown().unwrap();
    }

    #[test]
    fn xrt_iq1s_persistent_range_io_writes_inputs_and_returns_output_ranges() {
        let mut pool = pool(4);
        let phase = fixture_phase(31, false);
        let buffers = fixture_buffers(false);
        let handles = pool
            .cus
            .iter()
            .map(|cu| {
                (
                    cu.activation_bo as usize,
                    cu.token_map_bo as usize,
                    cu.output_bo as usize,
                )
            })
            .collect::<Vec<_>>();

        let completed = pool.submit_phase(&phase, &buffers).unwrap();
        let events = pool.ops.events();
        for (cu, (activation_bo, token_map_bo, output_bo)) in handles.into_iter().enumerate() {
            assert!(events.iter().any(|event| matches!(event,
                Event::BoWriteRange { bo, offset: 0x2000, bytes }
                    if *bo == activation_bo && bytes == &buffers.activations[0].bytes)));
            assert!(events.iter().any(|event| matches!(event,
                Event::BoWriteRange { bo, offset: 0x3000, bytes }
                    if *bo == token_map_bo && bytes == &buffers.token_maps[0].bytes)));
            assert!(events.iter().any(|event| matches!(event,
                Event::BoReadRange { bo, offset: 0x8000, bytes: 2048 }
                    if *bo == output_bo)));
            let last_input_sync = events
                .iter()
                .rposition(|event| {
                    matches!(event,
                    Event::BoSync { bo, direction: XRT_BO_SYNC_TO_DEVICE, .. }
                        if *bo == activation_bo || *bo == token_map_bo)
                })
                .unwrap();
            let producer_publish = events
                .iter()
                .position(|event| {
                    matches!(event,
                    Event::RegisterWrite { cu: actual, offset, value: 1 }
                        if *actual == cu as u32
                            && *offset == IQ1S_REG_COMMAND_PRODUCER_OFFSET as u32)
                })
                .unwrap();
            let doorbell = events
                .iter()
                .position(|event| {
                    matches!(event,
                    Event::RegisterWrite { cu: actual, offset, value: 1 }
                        if *actual == cu as u32
                            && *offset == IQ1S_REG_DOORBELL_OFFSET as u32)
                })
                .unwrap();
            assert!(last_input_sync < producer_publish);
            assert!(producer_publish < doorbell);
            assert_eq!(completed.results[cu].len(), 1);
            assert_eq!(completed.results[cu][0].offset, 0x8000);
            assert_eq!(completed.results[cu][0].bytes, vec![0x40 + cu as u8; 2048]);
        }
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event,
                    Event::RegisterWrite { offset, value: 1, .. }
                        if *offset == IQ1S_REG_DOORBELL_OFFSET as u32))
                .count(),
            4
        );
        assert_eq!(completed.dma.weight_bytes, 0);
        pool.shutdown().unwrap();
    }

    fn assert_invalid_phase_poisons(
        pool: &mut PersistentIq1sPool<FakeXrt>,
        phase: &CompiledLayerPhase,
        buffers: &PhaseBuffers,
    ) {
        assert!(matches!(
            pool.submit_phase(phase, buffers),
            Err(PersistentError::InvalidPhase(_)) | Err(PersistentError::Xrt { .. })
        ));
        assert!(matches!(
            pool.submit_phase(phase, buffers),
            Err(PersistentError::Poisoned(_))
        ));
    }

    #[test]
    fn xrt_iq1s_persistent_range_io_rejects_bad_host_and_device_ranges() {
        {
            let mut pool = pool(4);
            let phase = fixture_phase(32, false);
            let mut buffers = fixture_buffers(false);
            buffers.activations = vec![HostRange {
                offset: ACTIVATION_BYTES as u64 - 8,
                bytes: vec![0; 16],
            }];
            assert_invalid_phase_poisons(&mut pool, &phase, &buffers);
        }
        {
            let mut pool = pool(4);
            let mut phase = fixture_phase(33, false);
            for commands in &mut phase.commands {
                commands[0].output_offset = OUTPUT_BYTES as u64 - 8;
                commands[0].crc32 = 0;
                commands[0].crc32 = command_crc(&commands[0]);
            }
            assert_invalid_phase_poisons(&mut pool, &phase, &fixture_buffers(false));
        }
        {
            let mut pool = pool(4);
            let phase = fixture_phase(34, false);
            let mut buffers = fixture_buffers(false);
            buffers.activations.push(HostRange {
                offset: 0x3000,
                bytes: vec![0; 0x1000],
            });
            assert_invalid_phase_poisons(&mut pool, &phase, &buffers);
        }
        {
            let mut pool = pool(4);
            let phase = fixture_phase(35, false);
            let mut buffers = fixture_buffers(false);
            buffers.token_maps.clear();
            assert_invalid_phase_poisons(&mut pool, &phase, &buffers);
        }
    }

    #[test]
    fn xrt_iq1s_persistent_range_io_short_read_poisons_before_results() {
        let mut pool = pool(4);
        let phase = fixture_phase(36, false);
        pool.ops.fail_read(pool.cus[0].output_bo);
        assert_invalid_phase_poisons(&mut pool, &phase, &fixture_buffers(false));
    }

    #[test]
    fn xrt_iq1s_persistent_coalesces_dma_and_keeps_weights_resident() {
        let mut pool = pool(4);
        pool.measurement_begin().unwrap();
        let phase = fixture_phase(41, false);
        let completed = pool.submit_phase(&phase, &fixture_buffers(false)).unwrap();
        assert_eq!(completed.completions.iter().map(Vec::len).sum::<usize>(), 4);
        let measured = pool.measurement_end().unwrap();
        assert_eq!(measured.weight_ranges, 0);
        assert_eq!(measured.weight_bytes, 0);
        assert_eq!(measured.activation_ranges, 4);
        assert_eq!(measured.result_ranges, 4);
        assert_eq!(measured.command_ranges, 4);
        pool.shutdown().unwrap();
    }

    #[test]
    fn xrt_iq1s_persistent_wraps_without_overwrite() {
        let mut pool = pool(2);
        for transaction in 1..=3 {
            let phase = fixture_phase(transaction, true);
            pool.submit_phase(&phase, &fixture_buffers(true)).unwrap();
        }
        assert_eq!(pool.cus[0].command_producer, 6);
        assert_eq!(pool.cus[0].completion_consumer, 6);
        pool.shutdown().unwrap();
    }

    #[test]
    fn xrt_iq1s_persistent_first_fault_poisons_later_submissions() {
        for mutation in [
            CompletionMutation::Transaction,
            CompletionMutation::Program,
            CompletionMutation::Trace,
            CompletionMutation::Generation,
            CompletionMutation::Cu,
            CompletionMutation::Crc,
        ] {
            let mut pool = pool(4);
            pool.ops.set_completion_mutation(mutation);
            let phase = fixture_phase(51, false);
            assert!(matches!(
                pool.submit_phase(&phase, &fixture_buffers(false)),
                Err(PersistentError::Fault(_))
            ));
            assert!(matches!(
                pool.submit_phase(&phase, &fixture_buffers(false)),
                Err(PersistentError::Poisoned(_))
            ));
            pool.shutdown().unwrap();
        }
    }

    #[test]
    fn xrt_iq1s_persistent_rejects_unconsumed_ring_overwrite() {
        let mut pool = pool(4);
        pool.cus[0].command_producer = 4;
        let phase = fixture_phase(61, false);
        assert!(matches!(
            pool.submit_phase(&phase, &fixture_buffers(false)),
            Err(PersistentError::RingFull { cu: 0, capacity: 4 })
        ));
        assert!(matches!(
            pool.submit_phase(&phase, &fixture_buffers(false)),
            Err(PersistentError::Poisoned(_))
        ));
        pool.shutdown().unwrap();
    }

    #[test]
    fn xrt_iq1s_persistent_shutdown_closes_only_after_quiescent() {
        let mut pool = pool(4);
        pool.shutdown().unwrap();
        let events = pool.ops.events();
        let last_shutdown = events.iter().rposition(|event| matches!(event, Event::RegisterWrite { offset, value: CONTROL_SHUTDOWN, .. } if *offset == IQ1S_REG_CONTROL_OFFSET as u32)).unwrap();
        let first_close = events
            .iter()
            .position(|event| matches!(event, Event::CloseContext(_)))
            .unwrap();
        let last_quiescent_read = events
            .iter()
            .rposition(|event| matches!(event, Event::RegisterRead { offset, .. } if *offset == IQ1S_REG_QUIESCENT_OFFSET as u32))
            .unwrap();
        assert!(last_shutdown < first_close);
        assert!(last_quiescent_read < first_close);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::CloseContext(_)))
                .count(),
            4
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::DeviceClose))
                .count(),
            1
        );
    }

    fn persistent_smoke_fixture_for_grid(zero_grid: bool) -> (Vec<u8>, Vec<u8>, f32) {
        const D_VALUES: [f32; 4] = [0.5, -0.25, 1.5, 0.0625];
        const D_HALF: [u16; 4] = [0x3800, 0xb400, 0x3e00, 0x2c00];
        const S_VALUES: [f32; 4] = [-2.0, 0.0, 0.75, 4.0];
        const S_HALF: [u16; 4] = [0xc000, 0x0000, 0x3a00, 0x4400];

        let grid = validated_grid(None).expect("recover the verified IQ1_S grid");
        let mut row = Vec::with_capacity(800);
        let mut activations = vec![0u8; 32 * 144];
        let mut expected = 0.0f32;
        for vector_index in 0..16usize {
            let mut packed = [0u8; 50];
            let iq1s_d_half = if vector_index % 2 == 0 {
                0x3555u16
            } else {
                0xb400u16
            };
            packed[..2].copy_from_slice(&iq1s_d_half.to_le_bytes());
            let mut q8_groups = Vec::with_capacity(8);
            for group_index in 0..8usize {
                let odd_scale = 1 + 2 * ((vector_index + group_index) % 8) as u8;
                let negative = (vector_index + group_index) % 2 != 0;
                let indices = [
                    (vector_index * 137 + group_index * 29) & 0x7ff,
                    0x7ff - ((vector_index * 73 + group_index * 11) & 0x7ff),
                    ((vector_index << 8) | (group_index * 31)) & 0x7ff,
                    (((7 - group_index) << 8) | ((255 - vector_index * 13) & 0xff)) & 0x7ff,
                ];
                let mut qh =
                    (u16::from((odd_scale - 1) / 2) << 12) | if negative { 0x8000 } else { 0 };
                for (position, index) in indices.iter().enumerate() {
                    packed[2 + group_index * 4 + position] = *index as u8;
                    qh |= (((index >> 8) & 7) as u16) << (3 * position);
                }
                let qh_offset = 34 + group_index * 2;
                packed[qh_offset..qh_offset + 2].copy_from_slice(&qh.to_le_bytes());

                let mut qs = [0i8; 32];
                for (position, quant) in qs.iter_mut().enumerate() {
                    *quant = match vector_index {
                        0 => 0,
                        1 if position % 2 == 0 => i8::MIN,
                        1 => i8::MAX,
                        _ => {
                            (((position * 37 + group_index * 19 + vector_index * 11) % 255) as i16
                                - 127) as i8
                        }
                    };
                }
                let d_index = (vector_index + group_index) % D_VALUES.len();
                let s_index = (vector_index * 3 + group_index) % S_VALUES.len();
                let q8 = Q8_1Block {
                    d: D_VALUES[d_index],
                    s: S_VALUES[s_index],
                    qs,
                };
                let global_group = vector_index * 8 + group_index;
                let record_offset = (global_group / 4) * 144;
                let subblock = global_group % 4;
                activations[record_offset + subblock * 4..record_offset + subblock * 4 + 2]
                    .copy_from_slice(&D_HALF[d_index].to_le_bytes());
                activations[record_offset + subblock * 4 + 2..record_offset + subblock * 4 + 4]
                    .copy_from_slice(&S_HALF[s_index].to_le_bytes());
                for (destination, value) in activations
                    [record_offset + 16 + subblock * 32..record_offset + 16 + (subblock + 1) * 32]
                    .iter_mut()
                    .zip(qs)
                {
                    *destination = value as u8;
                }
                q8_groups.push(q8);
            }
            let parsed = Iq1sBlock::parse(&packed, &grid).expect("parse smoke IQ1_S block");
            for (group, q8) in parsed.groups.iter().zip(q8_groups.iter()) {
                let mut oracle_group = *group;
                if zero_grid {
                    oracle_group.grid_values.fill(0);
                }
                let (grid_dot, delta_dot) = raw_component_dots(&oracle_group, q8);
                let contribution = reconstruct_from_raw(
                    &oracle_group,
                    parsed.d,
                    q8,
                    grid_dot << 8,
                    delta_dot << 8,
                )
                .expect("reconstruct smoke IQ1_S contribution");
                expected = (expected + contribution) as f32;
            }
            row.extend_from_slice(&packed);
        }
        assert_eq!(row.len(), 800);
        assert_eq!(activations.len(), 4608);
        assert!(expected.is_finite() && expected != 0.0);
        (row, activations, expected)
    }

    fn persistent_smoke_fixture() -> (Vec<u8>, Vec<u8>, f32) {
        persistent_smoke_fixture_for_grid(false)
    }

    fn persistent_smoke_zero_grid_expected() -> f32 {
        persistent_smoke_fixture_for_grid(true).2
    }

    #[test]
    fn smoke_result_comparison_records_the_first_numerical_boundary() {
        let bytes = [1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat();
        let comparison = compare_smoke_result_bytes(2, 1, 1.0f32.to_bits(), &bytes);

        assert_eq!(comparison.rows_checked, 2);
        assert_eq!(comparison.mismatch_count, 1);
        assert_eq!(comparison.actual_bits_histogram.get(&1.0f32.to_bits()), Some(&1));
        assert_eq!(comparison.actual_bits_histogram.get(&2.0f32.to_bits()), Some(&1));
        assert_eq!(comparison.first_mismatches.len(), 1);
        assert_eq!(comparison.first_mismatches[0].cu, 2);
        assert_eq!(comparison.first_mismatches[0].generation, 1);
        assert_eq!(comparison.first_mismatches[0].row, 1);
        assert_eq!(comparison.first_mismatches[0].expected_bits, 1.0f32.to_bits());
        assert_eq!(comparison.first_mismatches[0].actual_bits, 2.0f32.to_bits());
    }

    #[test]
    fn smoke_zero_grid_oracle_matches_the_observed_hardware_signature() {
        assert_eq!(persistent_smoke_zero_grid_expected().to_bits(), 0x454c_4e9f);
    }

    #[test]
    fn smoke_signature_classifies_an_all_row_zero_grid_match() {
        let histogram = BTreeMap::from([(0x454c_4e9f, 2048)]);
        assert_eq!(
            classify_smoke_signature(2048, 2048, &histogram, 0x454c_4e9f),
            "matches_zero_grid_oracle"
        );
    }

    struct CapturedQwenReplayFixture {
        chunks: Vec<ArenaChunkSpec>,
        phase: CompiledLayerPhase,
        buffers: PhaseBuffers,
    }

    fn captured_qwen_replay_fixture(
        matrix: &[u8],
        activation: &[u8],
        mode: &str,
        transaction_id: u64,
    ) -> Result<CapturedQwenReplayFixture, String> {
        const MATRIX_BYTES: usize = 819_200;
        const SHARD_BYTES: usize = MATRIX_BYTES / ARENA_BANK_COUNT;
        const ACTIVATION_BYTES: usize = 4_608;
        if matrix.len() != MATRIX_BYTES || activation.len() != ACTIVATION_BYTES {
            return Err(format!(
                "captured Qwen replay requires {MATRIX_BYTES} matrix bytes and {ACTIVATION_BYTES} activation bytes"
            ));
        }
        let matrix_sha256: [u8; 32] = Sha256::digest(matrix).into();
        let activation_sha256: [u8; 32] = Sha256::digest(activation).into();
        let tensor = Arc::new(Iq1sTensorIdentity {
            canonical_path: PathBuf::from("/qwen397b-replay/blk.0.ffn_gate_exps.weight"),
            file_offset: 0,
            nbytes: MATRIX_BYTES as u64,
            name: "blk.0.ffn_gate_exps.weight".to_string(),
            layer: 0,
            ne: [4096, 1024, 512, 1],
            nb: [50, 800, 819_200, 419_430_400],
            role: Iq1sExpertRole::Gate,
            model_sha256: [0x39; 32],
            content_sha256: matrix_sha256,
            device: 1,
            inode: 2,
            modified_ns: 3,
        });
        let mut chunks = Vec::with_capacity(ARENA_BANK_COUNT);
        let mut commands = Vec::with_capacity(ARENA_BANK_COUNT);
        for bank in 0..ARENA_BANK_COUNT {
            let shard_bytes = &matrix[bank * SHARD_BYTES..(bank + 1) * SHARD_BYTES];
            let shard_sha256: [u8; 32] = Sha256::digest(shard_bytes).into();
            chunks.push(ArenaChunkSpec {
                bank: bank as u8,
                logical_offset: 0,
                bytes: SHARD_BYTES,
                sha256: shard_sha256,
                shards: vec![ArenaShardSpec {
                    bank: bank as u8,
                    logical_offset: 0,
                    bytes: SHARD_BYTES,
                    layer_id: 0,
                    role: IQ1S_ROLE_GATE as u16,
                    expert_id: 7,
                    row_start: bank as u32 * 256,
                    row_count: 256,
                    sha256: shard_sha256,
                }],
            });
            commands.push(SemanticIq1sCommand {
                layer_id: 0,
                phase: LayerPhase::PhaseA,
                role: Iq1sExpertRole::Gate,
                expert_id: 7,
                lane_mask: 1,
                token_ids: vec![0],
                input_offset: 0,
                output_offset: 0,
                token_map_offset: 0,
                row_shard: ArenaShard {
                    tensor: tensor.clone(),
                    expert: 7,
                    bank: bank as u8,
                    row_start: bank as u32 * 256,
                    row_count: 256,
                    superblock: 0,
                    offset: 0,
                    bytes: SHARD_BYTES as u64,
                    sha256: shard_sha256,
                },
            });
        }
        let phase = compile_layer_phase(
            &LayerPhasePlan {
                transaction_id,
                phase: LayerPhase::PhaseA,
                commands,
                activations: vec![ActivationRange {
                    cuda_ptr: 0x10000,
                    slab_offset: 0,
                    bytes: ACTIVATION_BYTES as u32,
                    stream: 1,
                    source_identity_sha256: activation_sha256,
                }],
            },
            mode,
            QWEN_MODEL_CONTEXT_LIMIT,
        )?;
        Ok(CapturedQwenReplayFixture {
            chunks,
            phase,
            buffers: PhaseBuffers {
                activations: vec![HostRange {
                    offset: 0,
                    bytes: activation.to_vec(),
                }],
                token_maps: vec![HostRange {
                    offset: 0,
                    bytes: 0u32.to_le_bytes().to_vec(),
                }],
            },
        })
    }

    #[test]
    fn captured_qwen_replay_fixture_maps_one_expert_across_four_banks() {
        let matrix = vec![0x5a; 819_200];
        let activation = vec![0xa5; 4_608];
        let fixture = captured_qwen_replay_fixture(&matrix, &activation, "handwritten", 17)
            .expect("build captured Qwen replay fixture");

        assert_eq!(fixture.chunks.len(), ARENA_BANK_COUNT);
        assert_eq!(fixture.phase.transaction_id, 17);
        assert_eq!(fixture.phase.commands.iter().map(Vec::len).sum::<usize>(), 4);
        assert_eq!(fixture.buffers.activations[0].bytes, activation);
        for bank in 0..ARENA_BANK_COUNT {
            assert_eq!(fixture.chunks[bank].bank, bank as u8);
            assert_eq!(fixture.chunks[bank].bytes, 204_800);
            assert_eq!(fixture.phase.commands[bank].len(), 1);
            assert_eq!(fixture.phase.commands[bank][0].row_start, bank as u32 * 256);
            assert_eq!(fixture.phase.commands[bank][0].lane_count, 1);
        }
    }

    fn persistent_smoke_chunks(bytes: &[u8], sha256: [u8; 32]) -> Vec<ArenaChunkSpec> {
        (0..ARENA_BANK_COUNT)
            .map(|bank| ArenaChunkSpec {
                bank: bank as u8,
                logical_offset: 0,
                bytes: bytes.len(),
                sha256,
                shards: vec![ArenaShardSpec {
                    bank: bank as u8,
                    logical_offset: 0,
                    bytes: bytes.len(),
                    layer_id: 7,
                    role: IQ1S_ROLE_GATE as u16,
                    expert_id: 17,
                    row_start: bank as u32 * 256,
                    row_count: 256,
                    sha256,
                }],
            })
            .collect()
    }

    fn persistent_smoke_phase(transaction_id: u64, shard_sha256: [u8; 32]) -> CompiledLayerPhase {
        let tensor = Arc::new(Iq1sTensorIdentity {
            canonical_path: PathBuf::from("/qwen397b-smoke/blk.7.ffn_gate_exps.weight"),
            file_offset: 0,
            nbytes: 819_200,
            name: "blk.7.ffn_gate_exps.weight".to_string(),
            layer: 7,
            ne: [4096, 1024, 512, 1],
            nb: [50, 800, 819_200, 419_430_400],
            role: Iq1sExpertRole::Gate,
            model_sha256: [0x51; 32],
            content_sha256: shard_sha256,
            device: 1,
            inode: 2,
            modified_ns: 3,
        });
        let commands = (0..ARENA_BANK_COUNT)
            .map(|bank| SemanticIq1sCommand {
                layer_id: 7,
                phase: LayerPhase::PhaseA,
                role: Iq1sExpertRole::Gate,
                expert_id: 17,
                lane_mask: 1,
                token_ids: vec![0],
                input_offset: 0x2000,
                output_offset: 0x8000,
                token_map_offset: 0x3000,
                row_shard: ArenaShard {
                    tensor: tensor.clone(),
                    expert: 17,
                    bank: bank as u8,
                    row_start: bank as u32 * 256,
                    row_count: 256,
                    superblock: 0,
                    offset: 0,
                    bytes: 204_800,
                    sha256: shard_sha256,
                },
            })
            .collect();
        compile_layer_phase(
            &LayerPhasePlan {
                transaction_id,
                phase: LayerPhase::PhaseA,
                commands,
                activations: vec![ActivationRange {
                    cuda_ptr: 0x10000,
                    slab_offset: 0x2000,
                    bytes: 4608,
                    stream: 1,
                    source_identity_sha256: [0x51; 32],
                }],
            },
            "compiler",
            QWEN_MODEL_CONTEXT_LIMIT,
        )
        .expect("compile persistent smoke phase")
    }

    fn format_xuid(uuid: Xuid) -> String {
        let hex = uuid
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        format!(
            "{}-{}-{}-{}-{}",
            &hex[0..8],
            &hex[8..12],
            &hex[12..16],
            &hex[16..20],
            &hex[20..32]
        )
    }

    #[test]
    #[ignore = "opens and programs the real AU250 only under the explicit persistent smoke guard"]
    fn au250_iq1s_persistent_four_cu_smoke() {
        if std::env::var("HETGPU_XRT_AU250_IQ1S_PERSISTENT_TEST").as_deref() != Ok("1") {
            return;
        }
        let xclbin = PathBuf::from(
            std::env::var_os("HETGPU_XRT_XCLBIN")
                .expect("HETGPU_XRT_XCLBIN must name the qualified persistent image"),
        );
        let expected_uuid = std::env::var("HETGPU_XRT_EXPECTED_UUID")
            .expect("HETGPU_XRT_EXPECTED_UUID is required");
        let summary_path = PathBuf::from(
            std::env::var_os("HETGPU_XRT_PERSISTENT_SUMMARY")
                .expect("HETGPU_XRT_PERSISTENT_SUMMARY is required"),
        );
        let timeout_ms = std::env::var("HETGPU_XRT_TIMEOUT_MS")
            .unwrap_or_else(|_| "10000".to_string())
            .parse::<u32>()
            .expect("HETGPU_XRT_TIMEOUT_MS must be u32");

        let (row, activations, expected) = persistent_smoke_fixture();
        let zero_grid_expected = persistent_smoke_zero_grid_expected();
        let arena_bytes = row.repeat(256);
        assert_eq!(arena_bytes.len(), 204_800);
        let arena_sha256: [u8; 32] = Sha256::digest(&arena_bytes).into();
        let chunks = persistent_smoke_chunks(&arena_bytes, arena_sha256);
        let ops = RealXrt::load(true).expect("load XRT with native-IP API");
        let mut pool = PersistentIq1sPool::open(
            ops,
            PersistentIq1sConfig::checked(xclbin, 0, Some(4), timeout_ms)
                .expect("validate persistent smoke config"),
            1,
            &chunks,
            |_| Ok(arena_bytes.clone()),
            |_| Ok(()),
        )
        .expect("open four-CU persistent IQ1_S pool");
        let actual_uuid = format_xuid(pool.xclbin_uuid);
        assert_eq!(actual_uuid, expected_uuid);
        let command_baselines =
            std::array::from_fn::<u32, ARENA_BANK_COUNT, _>(|cu| pool.cus[cu].command_producer);

        let buffers = PhaseBuffers {
            activations: vec![HostRange {
                offset: 0x2000,
                bytes: activations,
            }],
            token_maps: vec![HostRange {
                offset: 0x3000,
                bytes: 0u32.to_le_bytes().to_vec(),
            }],
        };
        let mut ring_generations: [Vec<u32>; ARENA_BANK_COUNT] =
            std::array::from_fn(|_| Vec::new());
        let mut completion_counts = [0u64; ARENA_BANK_COUNT];
        let mut result_rows_checked = 0u64;
        let mut result_mismatch_count = 0u64;
        let mut actual_bits_histogram = BTreeMap::<u32, u64>::new();
        let mut first_mismatches = Vec::<SmokeResultMismatch>::new();
        pool.measurement_begin()
            .expect("begin persistent DMA window");
        for (generation, transaction_id) in [101u64, 102].into_iter().enumerate() {
            let completed = pool
                .submit_phase(
                    &persistent_smoke_phase(transaction_id, arena_sha256),
                    &buffers,
                )
                .expect("submit persistent smoke descriptor generation");
            assert_eq!(completed.dma.weight_ranges, 0);
            assert_eq!(completed.dma.weight_bytes, 0);
            for cu in 0..ARENA_BANK_COUNT {
                assert_eq!(completed.completions[cu].len(), 1);
                let completion = completed.completions[cu][0];
                assert_eq!(
                    completion.command_index,
                    command_baselines[cu].wrapping_add(generation as u32)
                );
                assert_eq!(completion.fault_code, IQ1S_FAULT_CODE_NONE);
                assert!(completion.result_fence != 0 && completion.cycles != 0);
                ring_generations[cu].push(completion.command_index);
                completion_counts[cu] += 1;
                assert_eq!(completed.results[cu].len(), 1);
                let result = &completed.results[cu][0];
                assert_eq!(result.offset, 0x8000);
                assert_eq!(result.bytes.len(), 256 * 4);
                let comparison = compare_smoke_result_bytes(
                    cu,
                    generation,
                    expected.to_bits(),
                    &result.bytes,
                );
                result_rows_checked += comparison.rows_checked;
                result_mismatch_count += comparison.mismatch_count;
                for (actual_bits, count) in comparison.actual_bits_histogram {
                    *actual_bits_histogram.entry(actual_bits).or_default() += count;
                }
                for mismatch in comparison.first_mismatches {
                    if first_mismatches.len() < MAX_SMOKE_MISMATCH_SAMPLES {
                        first_mismatches.push(mismatch);
                    }
                }
            }
        }
        let measured = pool.measurement_end().expect("end persistent DMA window");
        assert_eq!(measured.weight_ranges, 0);
        assert_eq!(measured.weight_bytes, 0);
        assert_eq!(measured.command_ranges, 8);
        assert_eq!(measured.activation_ranges, 8);
        assert_eq!(measured.result_ranges, 8);
        assert_eq!(measured.program_ranges, 4);
        let mut sticky_fault_codes = [0u32; ARENA_BANK_COUNT];
        let mut quiescent = [0u32; ARENA_BANK_COUNT];
        for cu in 0..ARENA_BANK_COUNT {
            sticky_fault_codes[cu] = pool
                .reg_read(cu, IQ1S_REG_FAULT_CODE_OFFSET)
                .expect("read sticky fault code");
            quiescent[cu] = pool
                .reg_read(cu, IQ1S_REG_QUIESCENT_OFFSET)
                .expect("read quiescent state");
        }
        assert_eq!(sticky_fault_codes, [0; ARENA_BANK_COUNT]);
        assert_eq!(quiescent, [1; ARENA_BANK_COUNT]);
        pool.shutdown()
            .expect("gracefully shut down persistent CUs");

        let numerical_signature = classify_smoke_signature(
            result_rows_checked,
            result_mismatch_count,
            &actual_bits_histogram,
            zero_grid_expected.to_bits(),
        );

        let summary = serde_json::json!({
            "schema_version": 1,
            "status": if result_mismatch_count == 0 { "pass" } else { "numerical_mismatch" },
            "xclbin_uuid": actual_uuid,
            "persistent_starts_per_cu": [1, 1, 1, 1],
            "command_baselines_per_cu": command_baselines,
            "ring_generations_per_cu": ring_generations,
            "per_cu_completions": completion_counts,
            "sticky_fault_codes": sticky_fault_codes,
            "quiescent_before_shutdown": quiescent,
            "result_rows_checked": result_rows_checked,
            "result_mismatch_count": result_mismatch_count,
            "actual_f32_bits_histogram": actual_bits_histogram,
            "first_mismatches": first_mismatches,
            "expected_f32_bits": expected.to_bits(),
            "zero_grid_expected_f32_bits": zero_grid_expected.to_bits(),
            "numerical_signature": numerical_signature,
            "measured_dma": {
                "command_ranges": measured.command_ranges,
                "activation_ranges": measured.activation_ranges,
                "result_ranges": measured.result_ranges,
                "program_ranges": measured.program_ranges,
                "weight_ranges": measured.weight_ranges,
                "weight_bytes": measured.weight_bytes,
            },
        });
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&summary_path)
            .expect("create persistent smoke summary without overwriting evidence");
        serde_json::to_writer_pretty(&mut output, &summary)
            .expect("write persistent smoke summary");
        writeln!(output).expect("terminate persistent smoke summary");
        output.sync_all().expect("sync persistent smoke summary");
        assert_eq!(
            result_mismatch_count,
            0,
            "persistent hardware smoke numerical mismatch; evidence: {}",
            summary_path.display()
        );
    }
}
