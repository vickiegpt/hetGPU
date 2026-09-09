use super::iq1s_layer_abi::{IQ1S_ROLE_DOWN, IQ1S_ROLE_GATE, IQ1S_ROLE_UP};
use super::iq1s_weight_registry::{
    classify_tensor_name, Iq1sExpertRole, Iq1sTensorIdentity, Iq1sTensorSource, Iq1sWeightRegistry,
};
use super::xrt_iq1s_persistent::{ArenaChunkSpec, ArenaShardSpec};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::{FileExt, MetadataExt};
use std::sync::{Arc, Mutex};
use std::time::UNIX_EPOCH;

pub(crate) const ARENA_BANK_COUNT: usize = 4;
pub(crate) const ARENA_ALIGNMENT: u64 = 4 * 1024;
pub(crate) const ARENA_SUPERBLOCK_BYTES: u64 = 512 * 1024 * 1024;
pub(crate) const ARENA_STAGING_CHUNK_BYTES: u64 = 64 * 1024 * 1024;
pub(crate) const ARENA_BANK_CAPACITY_BYTES: u64 = 16 * 1024 * 1024 * 1024;
pub(crate) const ARENA_EXPECTED_TENSORS: usize = 141;
pub(crate) const ARENA_EXPECTED_EXPERTS: u64 = 512;
pub(crate) const ARENA_EXPECTED_RAW_BYTES: u64 = 59_139_686_400;

const IQ1S_BLOCK_VALUES: u64 = 256;
const IQ1S_BLOCK_BYTES: u64 = 50;
const COMMAND_RING_BYTES: u64 = 64 * 1024 * 1024;
const COMPLETION_RING_BYTES: u64 = 64 * 1024 * 1024;
const ACTIVATION_SLAB_BYTES: u64 = 256 * 1024 * 1024;
const OUTPUT_SLAB_BYTES: u64 = 256 * 1024 * 1024;
const TOKEN_MAP_SLAB_BYTES: u64 = 16 * 1024 * 1024;
const FAIL_CLOSED_RESERVE_BYTES: u64 = 128 * 1024 * 1024;
const BANK_RUNTIME_BYTES: u64 = COMMAND_RING_BYTES
    + COMPLETION_RING_BYTES
    + ACTIVATION_SLAB_BYTES
    + OUTPUT_SLAB_BYTES
    + TOKEN_MAP_SLAB_BYTES
    + FAIL_CLOSED_RESERVE_BYTES;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ArenaShard {
    pub(crate) tensor: Arc<Iq1sTensorIdentity>,
    pub(crate) expert: u16,
    pub(crate) bank: u8,
    pub(crate) row_start: u32,
    pub(crate) row_count: u32,
    pub(crate) superblock: u16,
    pub(crate) offset: u64,
    pub(crate) bytes: u64,
    pub(crate) sha256: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ArenaPlan {
    pub(crate) generation: u64,
    pub(crate) model_sha256: [u8; 32],
    pub(crate) bank_bytes: [u64; ARENA_BANK_COUNT],
    pub(crate) weight_bytes: [u64; ARENA_BANK_COUNT],
    pub(crate) shards: Vec<ArenaShard>,
    pub(crate) hashes_verified: bool,
}

pub(crate) trait ArenaBackend {
    type Handle;

    fn allocate(&mut self, bank: u8, bytes: usize) -> Result<Self::Handle, String>;
    fn write_range(
        &mut self,
        handle: &Self::Handle,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), String>;
    fn sync_to_device(
        &mut self,
        handle: &Self::Handle,
        offset: usize,
        bytes: usize,
    ) -> Result<(), String>;
    fn device_address(&self, handle: &Self::Handle) -> Result<u64, String>;
}

#[derive(Debug)]
pub(crate) struct ResidentArena<H> {
    pub(crate) plan: ArenaPlan,
    pub(crate) handles: Vec<H>,
    pub(crate) device_addresses: [u64; ARENA_BANK_COUNT],
}

#[derive(Debug, Default)]
struct GenerationState {
    installed: Option<u64>,
    in_flight: BTreeMap<u64, usize>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ArenaGenerationTracker {
    state: Arc<Mutex<GenerationState>>,
}

#[derive(Debug)]
pub(crate) struct ArenaGenerationLease {
    generation: u64,
    state: Arc<Mutex<GenerationState>>,
}

impl ArenaGenerationTracker {
    fn require_quiescent(&self) -> Result<(), String> {
        let state = self
            .state
            .lock()
            .map_err(|_| "IQ1_S arena generation lock poisoned".to_string())?;
        if state.in_flight.values().copied().sum::<usize>() != 0 {
            return Err(
                "cannot replace an IQ1_S arena while a generation is in flight".to_string(),
            );
        }
        Ok(())
    }

    pub(crate) fn install(&self, generation: u64) -> Result<(), String> {
        if generation == 0 {
            return Err("IQ1_S arena generation must be nonzero".to_string());
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| "IQ1_S arena generation lock poisoned".to_string())?;
        if state.installed == Some(generation) {
            return Ok(());
        }
        if state.in_flight.values().copied().sum::<usize>() != 0 {
            return Err(
                "cannot replace an IQ1_S arena while a generation is in flight".to_string(),
            );
        }
        state.installed = Some(generation);
        Ok(())
    }

    pub(crate) fn acquire(&self, generation: u64) -> Result<ArenaGenerationLease, String> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "IQ1_S arena generation lock poisoned".to_string())?;
        if state.installed != Some(generation) {
            return Err(format!(
                "IQ1_S arena generation {generation} is not the installed generation"
            ));
        }
        *state.in_flight.entry(generation).or_default() += 1;
        Ok(ArenaGenerationLease {
            generation,
            state: self.state.clone(),
        })
    }
}

impl Drop for ArenaGenerationLease {
    fn drop(&mut self) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        let Some(count) = state.in_flight.get_mut(&self.generation) else {
            return;
        };
        *count = count.saturating_sub(1);
        if *count == 0 {
            state.in_flight.remove(&self.generation);
        }
    }
}

pub(crate) struct SharedWeightArena<B: ArenaBackend> {
    backend: B,
    tracker: ArenaGenerationTracker,
    resident: Option<ResidentArena<B::Handle>>,
}

impl<B: ArenaBackend> SharedWeightArena<B> {
    pub(crate) fn new(backend: B) -> Self {
        Self {
            backend,
            tracker: ArenaGenerationTracker::default(),
            resident: None,
        }
    }

    pub(crate) fn install(
        &mut self,
        plan: ArenaPlan,
        sources: &[Arc<Iq1sTensorSource>],
    ) -> Result<(), String> {
        if let Some(resident) = &self.resident {
            if resident.plan == plan {
                return Ok(());
            }
            if resident.plan.generation == plan.generation {
                return Err(
                    "cannot replace IQ1_S arena contents without a new generation".to_string(),
                );
            }
        }
        self.tracker.require_quiescent()?;
        let generation = plan.generation;
        let resident = load_arena(&mut self.backend, plan, sources)?;
        self.tracker.install(generation)?;
        self.resident = Some(resident);
        Ok(())
    }

    pub(crate) fn acquire(&self, generation: u64) -> Result<ArenaGenerationLease, String> {
        if self
            .resident
            .as_ref()
            .map(|resident| resident.plan.generation)
            != Some(generation)
        {
            return Err("requested IQ1_S arena is not resident".to_string());
        }
        self.tracker.acquire(generation)
    }

    pub(crate) fn resident(&self) -> Option<&ResidentArena<B::Handle>> {
        self.resident.as_ref()
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64, String> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err("IQ1_S arena alignment must be a nonzero power of two".to_string());
    }
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .ok_or_else(|| "IQ1_S arena alignment overflow".to_string())
}

fn source_modified_ns(file: &File) -> Result<u128, String> {
    file.metadata()
        .map_err(|error| format!("inspect IQ1_S arena source: {error}"))?
        .modified()
        .map_err(|error| format!("read IQ1_S arena source modification time: {error}"))?
        .duration_since(UNIX_EPOCH)
        .map_err(|_| "IQ1_S arena source modification time predates Unix epoch".to_string())
        .map(|duration| duration.as_nanos())
}

fn open_verified_source(identity: &Iq1sTensorIdentity) -> Result<File, String> {
    let file = OpenOptions::new()
        .read(true)
        .open(&identity.canonical_path)
        .map_err(|error| format!("open IQ1_S arena source read-only: {error}"))?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("inspect IQ1_S arena source: {error}"))?;
    if metadata.dev() != identity.device
        || metadata.ino() != identity.inode
        || source_modified_ns(&file)? != identity.modified_ns
    {
        return Err(format!(
            "IQ1_S arena source identity changed for {}",
            identity.name
        ));
    }
    let end = identity
        .file_offset
        .checked_add(identity.nbytes)
        .ok_or("IQ1_S arena source range overflow")?;
    if end > metadata.len() {
        return Err(format!(
            "IQ1_S arena source range exceeds file for {}",
            identity.name
        ));
    }
    Ok(file)
}

fn checked_packed_row_bytes(identity: &Iq1sTensorIdentity) -> Result<u64, String> {
    if identity.ne[0] == 0 || !identity.ne[0].is_multiple_of(IQ1S_BLOCK_VALUES) {
        return Err(format!(
            "IQ1_S arena tensor {} has invalid input columns",
            identity.name
        ));
    }
    let row_bytes = (identity.ne[0] / IQ1S_BLOCK_VALUES)
        .checked_mul(IQ1S_BLOCK_BYTES)
        .ok_or("IQ1_S arena row byte count overflow")?;
    if identity.nb[0] != IQ1S_BLOCK_BYTES || identity.nb[1] < row_bytes {
        return Err(format!(
            "IQ1_S arena tensor {} has an invalid packed row stride",
            identity.name
        ));
    }
    Ok(row_bytes)
}

fn checked_row_range(rows: u64, bank: usize) -> Result<(u32, u32), String> {
    let base = rows / ARENA_BANK_COUNT as u64;
    let remainder = rows % ARENA_BANK_COUNT as u64;
    let bank_u64 = bank as u64;
    let row_start = bank_u64
        .checked_mul(base)
        .and_then(|value| value.checked_add(bank_u64.min(remainder)))
        .ok_or("IQ1_S arena row range overflow")?;
    let row_count = base + u64::from(bank_u64 < remainder);
    Ok((
        u32::try_from(row_start).map_err(|_| "IQ1_S row start does not fit u32")?,
        u32::try_from(row_count).map_err(|_| "IQ1_S row count does not fit u32")?,
    ))
}

fn reserve_shard_offset(cursor: &mut u64, bytes: u64) -> Result<(u64, u16), String> {
    let mut start = align_up(*cursor, ARENA_ALIGNMENT)?;
    let within_superblock = start % ARENA_SUPERBLOCK_BYTES;
    if within_superblock
        .checked_add(bytes)
        .ok_or("IQ1_S arena superblock range overflow")?
        > ARENA_SUPERBLOCK_BYTES
    {
        start = align_up(start, ARENA_SUPERBLOCK_BYTES)?;
    }
    let end = start
        .checked_add(bytes)
        .ok_or("IQ1_S arena shard range overflow")?;
    if end > ARENA_BANK_CAPACITY_BYTES {
        return Err("IQ1_S weight shards exceed a 16 GiB bank".to_string());
    }
    let superblock = u16::try_from(start / ARENA_SUPERBLOCK_BYTES)
        .map_err(|_| "IQ1_S arena superblock index does not fit u16")?;
    *cursor = end;
    Ok((start, superblock))
}

fn read_shard_bytes(file: &File, shard: &ArenaShard) -> Result<Vec<u8>, String> {
    let row_bytes = checked_packed_row_bytes(&shard.tensor)?;
    let expected = u64::from(shard.row_count)
        .checked_mul(row_bytes)
        .ok_or("IQ1_S arena shard byte count overflow")?;
    if expected != shard.bytes {
        return Err("IQ1_S arena shard metadata byte count mismatch".to_string());
    }
    let output_len =
        usize::try_from(expected).map_err(|_| "IQ1_S arena shard byte count does not fit usize")?;
    let mut output = vec![0u8; output_len];
    if shard.tensor.nb[1] == row_bytes {
        let source_offset = shard
            .tensor
            .file_offset
            .checked_add(u64::from(shard.expert) * shard.tensor.nb[2])
            .and_then(|value| value.checked_add(u64::from(shard.row_start) * row_bytes))
            .ok_or("IQ1_S arena source offset overflow")?;
        file.read_exact_at(&mut output, source_offset)
            .map_err(|error| format!("read IQ1_S arena shard: {error}"))?;
        return Ok(output);
    }
    let row_len =
        usize::try_from(row_bytes).map_err(|_| "IQ1_S arena packed row does not fit usize")?;
    for row in 0..u64::from(shard.row_count) {
        let source_offset = shard
            .tensor
            .file_offset
            .checked_add(u64::from(shard.expert) * shard.tensor.nb[2])
            .and_then(|value| {
                value.checked_add((u64::from(shard.row_start) + row) * shard.tensor.nb[1])
            })
            .ok_or("IQ1_S arena strided source offset overflow")?;
        let destination = usize::try_from(row)
            .ok()
            .and_then(|value| value.checked_mul(row_len))
            .ok_or("IQ1_S arena destination offset overflow")?;
        file.read_exact_at(
            &mut output[destination..destination + row_len],
            source_offset,
        )
        .map_err(|error| format!("read strided IQ1_S arena shard: {error}"))?;
    }
    Ok(output)
}

fn shard_sha256(file: &File, shard: &ArenaShard) -> Result<[u8; 32], String> {
    let bytes = read_shard_bytes(file, shard)?;
    Ok(Sha256::digest(bytes).into())
}

fn validate_qwen_sources(sources: &[Arc<Iq1sTensorSource>]) -> Result<[u8; 32], String> {
    if sources.len() != ARENA_EXPECTED_TENSORS {
        return Err(format!(
            "Qwen IQ1_S arena requires exactly {ARENA_EXPECTED_TENSORS} tensors, got {}",
            sources.len()
        ));
    }
    let mut roles = BTreeMap::<Iq1sExpertRole, usize>::new();
    let mut names = BTreeSet::new();
    let mut model_sha256 = None;
    let mut raw_bytes = 0u64;
    for source in sources {
        let identity = &source.identity;
        if !names.insert(identity.name.clone()) {
            return Err(format!("duplicate IQ1_S arena tensor {}", identity.name));
        }
        let (layer, role) = classify_tensor_name(&identity.name)?;
        if layer != identity.layer || role != identity.role || role == Iq1sExpertRole::GateUp {
            return Err(format!(
                "invalid Qwen IQ1_S arena identity {}",
                identity.name
            ));
        }
        if identity.model_sha256 == [0; 32] {
            return Err("Qwen IQ1_S arena model hash must be nonzero".to_string());
        }
        match model_sha256 {
            None => model_sha256 = Some(identity.model_sha256),
            Some(hash) if hash == identity.model_sha256 => {}
            Some(_) => return Err("Qwen IQ1_S arena sources span multiple models".to_string()),
        }
        let expected_ne = match role {
            Iq1sExpertRole::Gate | Iq1sExpertRole::Up => [4096, 1024, 512, 1],
            Iq1sExpertRole::Down => [1024, 4096, 512, 1],
            Iq1sExpertRole::GateUp => unreachable!(),
        };
        if identity.ne != expected_ne || identity.ne[2] != ARENA_EXPECTED_EXPERTS {
            return Err(format!("unexpected Qwen IQ1_S shape for {}", identity.name));
        }
        let row_bytes = checked_packed_row_bytes(identity)?;
        let expert_bytes = identity.ne[1]
            .checked_mul(row_bytes)
            .ok_or("Qwen IQ1_S expert byte count overflow")?;
        let tensor_bytes = identity.ne[2]
            .checked_mul(expert_bytes)
            .ok_or("Qwen IQ1_S tensor byte count overflow")?;
        if identity.nb[1] != row_bytes
            || identity.nb[2] != expert_bytes
            || identity.nb[3] != tensor_bytes
            || identity.nbytes != tensor_bytes
        {
            return Err(format!(
                "unexpected Qwen IQ1_S strides for {}",
                identity.name
            ));
        }
        raw_bytes = raw_bytes
            .checked_add(identity.nbytes)
            .ok_or("Qwen IQ1_S raw byte count overflow")?;
        *roles.entry(role).or_default() += 1;
    }
    if roles.get(&Iq1sExpertRole::Gate) != Some(&50)
        || roles.get(&Iq1sExpertRole::Up) != Some(&50)
        || roles.get(&Iq1sExpertRole::Down) != Some(&41)
        || roles.len() != 3
    {
        return Err(format!(
            "Qwen IQ1_S role counts are {roles:?}, expected 50/50/41"
        ));
    }
    if raw_bytes != ARENA_EXPECTED_RAW_BYTES {
        return Err(format!(
            "Qwen IQ1_S raw bytes are {raw_bytes}, expected {ARENA_EXPECTED_RAW_BYTES}"
        ));
    }
    model_sha256.ok_or_else(|| "Qwen IQ1_S arena has no model hash".to_string())
}

fn build_plan(
    sources: &[Arc<Iq1sTensorSource>],
    generation: u64,
    require_qwen_contract: bool,
    hash_shards: bool,
) -> Result<ArenaPlan, String> {
    if generation == 0 {
        return Err("IQ1_S arena generation must be nonzero".to_string());
    }
    if sources.is_empty() {
        return Err("IQ1_S arena source set is empty".to_string());
    }
    let model_sha256 = if require_qwen_contract {
        validate_qwen_sources(sources)?
    } else {
        let hash = sources[0].identity.model_sha256;
        if hash == [0; 32]
            || sources
                .iter()
                .any(|source| source.identity.model_sha256 != hash)
        {
            return Err("IQ1_S arena test sources have an invalid model hash".to_string());
        }
        hash
    };
    let mut sorted = sources.to_vec();
    sorted.sort_by(|left, right| {
        left.identity
            .layer
            .cmp(&right.identity.layer)
            .then_with(|| left.identity.role.cmp(&right.identity.role))
            .then_with(|| left.identity.name.cmp(&right.identity.name))
    });
    let mut cursors = [0u64; ARENA_BANK_COUNT];
    let mut shards = Vec::new();
    for source in sorted {
        let identity = Arc::new(source.identity.clone());
        let row_bytes = checked_packed_row_bytes(&identity)?;
        if identity.ne[1] == 0 || identity.ne[2] == 0 || identity.ne[2] > u64::from(u16::MAX) {
            return Err(format!(
                "IQ1_S arena dimensions are invalid for {}",
                identity.name
            ));
        }
        let file = if hash_shards {
            Some(open_verified_source(&identity)?)
        } else {
            None
        };
        for expert in 0..identity.ne[2] {
            for bank in 0..ARENA_BANK_COUNT {
                let (row_start, row_count) = checked_row_range(identity.ne[1], bank)?;
                if row_count == 0 {
                    return Err("IQ1_S arena row sharding produced an empty bank".to_string());
                }
                let bytes = u64::from(row_count)
                    .checked_mul(row_bytes)
                    .ok_or("IQ1_S arena shard byte count overflow")?;
                let (offset, superblock) = reserve_shard_offset(&mut cursors[bank], bytes)?;
                let mut shard = ArenaShard {
                    tensor: identity.clone(),
                    expert: expert as u16,
                    bank: bank as u8,
                    row_start,
                    row_count,
                    superblock,
                    offset,
                    bytes,
                    sha256: [0; 32],
                };
                if let Some(file) = &file {
                    shard.sha256 = shard_sha256(file, &shard)?;
                    if shard.sha256 == [0; 32] {
                        return Err("IQ1_S arena shard hash must be nonzero".to_string());
                    }
                }
                shards.push(shard);
            }
        }
    }
    let mut bank_bytes = [0u64; ARENA_BANK_COUNT];
    let mut weight_bytes = [0u64; ARENA_BANK_COUNT];
    for bank in 0..ARENA_BANK_COUNT {
        weight_bytes[bank] = align_up(cursors[bank], ARENA_ALIGNMENT)?;
        bank_bytes[bank] = align_up(
            weight_bytes[bank]
                .checked_add(BANK_RUNTIME_BYTES)
                .ok_or("IQ1_S arena bank byte count overflow")?,
            ARENA_ALIGNMENT,
        )?;
        if bank_bytes[bank] > ARENA_BANK_CAPACITY_BYTES {
            return Err(format!(
                "IQ1_S arena bank {bank} requires {} bytes, above 16 GiB",
                bank_bytes[bank]
            ));
        }
    }
    Ok(ArenaPlan {
        generation,
        model_sha256,
        bank_bytes,
        weight_bytes,
        shards,
        hashes_verified: hash_shards,
    })
}

pub(crate) fn plan_registered_arena(
    registry: &Iq1sWeightRegistry,
    generation: u64,
) -> Result<ArenaPlan, String> {
    let sources = registry.registered_sources()?;
    build_plan(&sources, generation, true, true)
}

pub(crate) fn plan_registered_arena_metadata(
    registry: &Iq1sWeightRegistry,
    generation: u64,
) -> Result<ArenaPlan, String> {
    let sources = registry.registered_sources()?;
    build_plan(&sources, generation, true, false)
}

fn validate_arena_plan(plan: &ArenaPlan, sources: &[Arc<Iq1sTensorSource>]) -> Result<(), String> {
    if plan.generation == 0 || plan.model_sha256 == [0; 32] {
        return Err("IQ1_S arena plan has an invalid generation or model hash".to_string());
    }
    for bank in 0..ARENA_BANK_COUNT {
        let expected_bank_bytes = align_up(
            plan.weight_bytes[bank]
                .checked_add(BANK_RUNTIME_BYTES)
                .ok_or("IQ1_S arena plan bank byte count overflow")?,
            ARENA_ALIGNMENT,
        )?;
        if plan.weight_bytes[bank] % ARENA_ALIGNMENT != 0
            || plan.bank_bytes[bank] != expected_bank_bytes
            || plan.bank_bytes[bank] > ARENA_BANK_CAPACITY_BYTES
        {
            return Err(format!("IQ1_S arena plan bank {bank} bounds are invalid"));
        }
    }
    let mut source_map = HashMap::<String, Arc<Iq1sTensorSource>>::new();
    for source in sources {
        if source.identity.model_sha256 != plan.model_sha256 {
            return Err("IQ1_S arena plan and source model hashes differ".to_string());
        }
        if source_map
            .insert(source.identity.name.clone(), source.clone())
            .is_some()
        {
            return Err(format!(
                "duplicate IQ1_S arena source {}",
                source.identity.name
            ));
        }
    }
    let mut coordinates = BTreeSet::new();
    let mut coverage = BTreeMap::<(String, u16), Vec<(u8, u32, u32)>>::new();
    let mut bank_ranges = [(); ARENA_BANK_COUNT].map(|_| Vec::<(u64, u64)>::new());
    for shard in &plan.shards {
        let bank = usize::from(shard.bank);
        if bank >= ARENA_BANK_COUNT {
            return Err(format!("IQ1_S arena shard bank {} is invalid", shard.bank));
        }
        let source = source_map
            .get(&shard.tensor.name)
            .ok_or_else(|| format!("missing IQ1_S arena source {}", shard.tensor.name))?;
        if source.identity != *shard.tensor {
            return Err(format!(
                "IQ1_S arena source identity mismatch for {}",
                shard.tensor.name
            ));
        }
        if u64::from(shard.expert) >= shard.tensor.ne[2]
            || shard.row_count == 0
            || u64::from(shard.row_start) + u64::from(shard.row_count) > shard.tensor.ne[1]
        {
            return Err("IQ1_S arena shard tensor coordinates are out of bounds".to_string());
        }
        let expected_bytes = u64::from(shard.row_count)
            .checked_mul(checked_packed_row_bytes(&shard.tensor)?)
            .ok_or("IQ1_S arena shard byte count overflow")?;
        let end = shard
            .offset
            .checked_add(shard.bytes)
            .ok_or("IQ1_S arena shard end overflow")?;
        if shard.bytes != expected_bytes
            || shard.offset % ARENA_ALIGNMENT != 0
            || u64::from(shard.superblock) != shard.offset / ARENA_SUPERBLOCK_BYTES
            || shard.offset % ARENA_SUPERBLOCK_BYTES + shard.bytes > ARENA_SUPERBLOCK_BYTES
            || end > plan.weight_bytes[bank]
            || (plan.hashes_verified && shard.sha256 == [0; 32])
        {
            return Err("IQ1_S arena shard layout or hash metadata is invalid".to_string());
        }
        if !coordinates.insert((shard.tensor.name.clone(), shard.expert, shard.bank)) {
            return Err("IQ1_S arena plan contains a duplicate shard coordinate".to_string());
        }
        coverage
            .entry((shard.tensor.name.clone(), shard.expert))
            .or_default()
            .push((shard.bank, shard.row_start, shard.row_count));
        bank_ranges[bank].push((shard.offset, end));
    }
    for source in sources {
        for expert in 0..source.identity.ne[2] {
            let key = (
                source.identity.name.clone(),
                u16::try_from(expert).map_err(|_| "IQ1_S expert ID does not fit u16")?,
            );
            let mut ranges = coverage
                .remove(&key)
                .ok_or("IQ1_S arena plan omitted an expert row range")?;
            ranges.sort_by_key(|(_, row_start, _)| *row_start);
            if ranges.len() != ARENA_BANK_COUNT {
                return Err("IQ1_S arena expert does not have exactly four row shards".to_string());
            }
            let mut row_cursor = 0u64;
            let mut banks = BTreeSet::new();
            for (bank, row_start, row_count) in ranges {
                if !banks.insert(bank) || u64::from(row_start) != row_cursor {
                    return Err("IQ1_S arena expert row coverage overlaps or has a gap".to_string());
                }
                row_cursor = row_cursor
                    .checked_add(u64::from(row_count))
                    .ok_or("IQ1_S arena row coverage overflow")?;
            }
            if row_cursor != source.identity.ne[1] {
                return Err("IQ1_S arena expert row coverage is incomplete".to_string());
            }
        }
    }
    if !coverage.is_empty() {
        return Err("IQ1_S arena plan contains shards for unknown experts".to_string());
    }
    for ranges in &mut bank_ranges {
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err("IQ1_S arena plan contains overlapping bank ranges".to_string());
            }
        }
    }
    Ok(())
}

pub(crate) fn load_arena<B: ArenaBackend>(
    backend: &mut B,
    plan: ArenaPlan,
    sources: &[Arc<Iq1sTensorSource>],
) -> Result<ResidentArena<B::Handle>, String> {
    if !plan.hashes_verified || plan.shards.iter().any(|shard| shard.sha256 == [0; 32]) {
        return Err("IQ1_S arena load requires verified per-shard hashes".to_string());
    }
    validate_arena_plan(&plan, sources)?;
    let source_map = sources
        .iter()
        .map(|source| (source.identity.name.clone(), source.clone()))
        .collect::<HashMap<_, _>>();
    let mut handles = Vec::with_capacity(ARENA_BANK_COUNT);
    let mut device_addresses = [0u64; ARENA_BANK_COUNT];
    for bank in 0..ARENA_BANK_COUNT {
        let bytes = usize::try_from(plan.bank_bytes[bank])
            .map_err(|_| "IQ1_S arena bank bytes do not fit usize")?;
        let handle = backend.allocate(bank as u8, bytes)?;
        device_addresses[bank] = backend.device_address(&handle)?;
        if device_addresses[bank] == 0 || device_addresses[bank] % ARENA_ALIGNMENT != 0 {
            return Err(format!(
                "IQ1_S arena bank {bank} has an invalid device address"
            ));
        }
        handles.push(handle);
    }

    let mut open_files = HashMap::<String, File>::new();
    for bank in 0..ARENA_BANK_COUNT {
        let mut by_superblock = BTreeMap::<u16, Vec<&ArenaShard>>::new();
        for shard in plan
            .shards
            .iter()
            .filter(|shard| usize::from(shard.bank) == bank)
        {
            by_superblock
                .entry(shard.superblock)
                .or_default()
                .push(shard);
        }
        for (superblock, mut group) in by_superblock {
            group.sort_by_key(|shard| shard.offset);
            let superblock_start = u64::from(superblock) * ARENA_SUPERBLOCK_BYTES;
            let span_end = group.iter().try_fold(0u64, |maximum, shard| {
                shard
                    .offset
                    .checked_add(shard.bytes)
                    .map(|end| maximum.max(end))
                    .ok_or("IQ1_S arena shard end overflow")
            })?;
            let span_bytes = span_end
                .checked_sub(superblock_start)
                .ok_or("IQ1_S arena superblock start exceeds its shards")?;
            if span_bytes > ARENA_SUPERBLOCK_BYTES {
                return Err("IQ1_S arena shard crosses its device superblock".to_string());
            }
            let mut staging = vec![
                0u8;
                usize::try_from(span_bytes)
                    .map_err(|_| "IQ1_S arena staging bytes do not fit usize")?
            ];
            for shard in group {
                let source = source_map
                    .get(&shard.tensor.name)
                    .ok_or_else(|| format!("missing IQ1_S arena source {}", shard.tensor.name))?;
                if source.identity != *shard.tensor {
                    return Err(format!(
                        "IQ1_S arena source identity mismatch for {}",
                        shard.tensor.name
                    ));
                }
                if !open_files.contains_key(&shard.tensor.name) {
                    open_files.insert(
                        shard.tensor.name.clone(),
                        open_verified_source(&shard.tensor)?,
                    );
                }
                let file = open_files
                    .get(&shard.tensor.name)
                    .ok_or("IQ1_S arena source cache insertion failed")?;
                let bytes = read_shard_bytes(file, shard)?;
                let actual_hash: [u8; 32] = Sha256::digest(&bytes).into();
                if actual_hash != shard.sha256 {
                    return Err(format!(
                        "IQ1_S arena shard hash mismatch for {} expert {} bank {}",
                        shard.tensor.name, shard.expert, shard.bank
                    ));
                }
                let destination = usize::try_from(shard.offset - superblock_start)
                    .map_err(|_| "IQ1_S arena staging offset does not fit usize")?;
                let end = destination
                    .checked_add(bytes.len())
                    .ok_or("IQ1_S arena staging range overflow")?;
                staging
                    .get_mut(destination..end)
                    .ok_or("IQ1_S arena staging range is out of bounds")?
                    .copy_from_slice(&bytes);
            }
            let offset = usize::try_from(superblock_start)
                .map_err(|_| "IQ1_S arena backend offset does not fit usize")?;
            backend.write_range(&handles[bank], offset, &staging)?;
            backend.sync_to_device(&handles[bank], offset, staging.len())?;
        }
    }
    Ok(ResidentArena {
        plan,
        handles,
        device_addresses,
    })
}

fn validate_persistent_plan(plan: &ArenaPlan) -> Result<(), String> {
    if !plan.hashes_verified || plan.generation == 0 || plan.model_sha256 == [0; 32] {
        return Err("persistent IQ1_S chunks require a verified arena plan".to_string());
    }
    Ok(())
}

fn persistent_chunk_blueprint(
    bank: u8,
    logical_offset: u64,
    shards: &mut [&ArenaShard],
) -> Result<ArenaChunkSpec, String> {
    if shards.is_empty() {
        return Err("persistent IQ1_S chunk has no arena shards".to_string());
    }
    if logical_offset % ARENA_ALIGNMENT != 0 {
        return Err("persistent IQ1_S staging chunk offset is not aligned".to_string());
    }
    shards.sort_by_key(|shard| shard.offset);
    let superblock = u16::try_from(logical_offset / ARENA_SUPERBLOCK_BYTES)
        .map_err(|_| "persistent IQ1_S chunk superblock does not fit u16")?;
    let superblock_start = u64::from(superblock)
        .checked_mul(ARENA_SUPERBLOCK_BYTES)
        .ok_or("persistent IQ1_S chunk start overflow")?;
    let superblock_limit = superblock_start
        .checked_add(ARENA_SUPERBLOCK_BYTES)
        .ok_or("persistent IQ1_S chunk limit overflow")?;
    let staging_limit = logical_offset
        .checked_add(ARENA_STAGING_CHUNK_BYTES)
        .ok_or("persistent IQ1_S staging chunk limit overflow")?;
    let mut cursor = logical_offset;
    let mut specs = Vec::with_capacity(shards.len());
    for shard in shards {
        if shard.bank != bank || shard.superblock != superblock {
            return Err("persistent IQ1_S chunk mixes bank or superblock identities".to_string());
        }
        let expected = align_up(cursor, ARENA_ALIGNMENT)?;
        if shard.offset != expected {
            return Err(format!(
                "persistent IQ1_S chunk gap before {} expert {} is not alignment padding",
                shard.tensor.name, shard.expert
            ));
        }
        cursor = shard
            .offset
            .checked_add(shard.bytes)
            .ok_or("persistent IQ1_S shard end overflow")?;
        if cursor > superblock_limit {
            return Err("persistent IQ1_S chunk crosses its device superblock".to_string());
        }
        if cursor > staging_limit {
            return Err("persistent IQ1_S chunk exceeds the 64 MiB staging bound".to_string());
        }
        let role = match shard.tensor.role {
            Iq1sExpertRole::Gate => IQ1S_ROLE_GATE as u16,
            Iq1sExpertRole::Up => IQ1S_ROLE_UP as u16,
            Iq1sExpertRole::Down => IQ1S_ROLE_DOWN as u16,
            Iq1sExpertRole::GateUp => {
                return Err("fused gate_up cannot enter the persistent arena".to_string())
            }
        };
        specs.push(ArenaShardSpec {
            bank,
            logical_offset: shard.offset,
            bytes: usize::try_from(shard.bytes)
                .map_err(|_| "persistent IQ1_S shard bytes do not fit usize")?,
            layer_id: shard.tensor.layer,
            role,
            expert_id: shard.expert,
            row_start: shard.row_start,
            row_count: shard.row_count,
            sha256: shard.sha256,
        });
    }
    let bytes = usize::try_from(cursor - logical_offset)
        .map_err(|_| "persistent IQ1_S chunk bytes do not fit usize")?;
    if bytes == 0 || bytes as u64 > ARENA_STAGING_CHUNK_BYTES {
        return Err("persistent IQ1_S chunk has an invalid byte count".to_string());
    }
    Ok(ArenaChunkSpec {
        bank,
        logical_offset,
        bytes,
        sha256: [0; 32],
        shards: specs,
    })
}

fn persistent_chunk_blueprints(plan: &ArenaPlan) -> Result<Vec<ArenaChunkSpec>, String> {
    validate_persistent_plan(plan)?;
    let mut grouped = BTreeMap::<(u8, u16), Vec<&ArenaShard>>::new();
    for shard in &plan.shards {
        if usize::from(shard.bank) >= ARENA_BANK_COUNT || shard.sha256 == [0; 32] {
            return Err(
                "persistent IQ1_S chunk contains an invalid bank or shard hash".to_string(),
            );
        }
        grouped
            .entry((shard.bank, shard.superblock))
            .or_default()
            .push(shard);
    }
    if (0..ARENA_BANK_COUNT).any(|bank| {
        !grouped
            .keys()
            .any(|(actual, _)| usize::from(*actual) == bank)
    }) {
        return Err("persistent IQ1_S chunks must populate all four banks".to_string());
    }

    let mut chunks = Vec::with_capacity(grouped.len());
    for ((bank, _superblock), mut shards) in grouped {
        shards.sort_by_key(|shard| shard.offset);
        let mut begin = 0usize;
        while begin < shards.len() {
            let logical_offset = shards[begin].offset;
            let mut end = begin + 1;
            while end < shards.len() {
                let candidate_end = shards[end]
                    .offset
                    .checked_add(shards[end].bytes)
                    .ok_or("persistent IQ1_S staging candidate overflow")?;
                let candidate_bytes = candidate_end
                    .checked_sub(logical_offset)
                    .ok_or("persistent IQ1_S staging candidate precedes its chunk")?;
                if candidate_bytes > ARENA_STAGING_CHUNK_BYTES {
                    break;
                }
                end += 1;
            }
            chunks.push(persistent_chunk_blueprint(
                bank,
                logical_offset,
                &mut shards[begin..end],
            )?);
            begin = end;
        }
    }
    chunks.sort_by_key(|chunk| (chunk.bank, chunk.logical_offset));
    Ok(chunks)
}

fn assemble_persistent_chunk(
    plan: &ArenaPlan,
    sources: &[Arc<Iq1sTensorSource>],
    chunk: &ArenaChunkSpec,
    verify_chunk_hash: bool,
) -> Result<Vec<u8>, String> {
    if chunk.bytes == 0 || chunk.bytes as u64 > ARENA_STAGING_CHUNK_BYTES {
        return Err("persistent IQ1_S chunk exceeds the staging chunk bound".to_string());
    }
    validate_persistent_plan(plan)?;
    if chunk.logical_offset % ARENA_ALIGNMENT != 0 {
        return Err("persistent IQ1_S staging chunk offset is not aligned".to_string());
    }
    let chunk_end = chunk
        .logical_offset
        .checked_add(chunk.bytes as u64)
        .ok_or("persistent IQ1_S chunk range overflow")?;
    let mut plan_shards = plan
        .shards
        .iter()
        .filter(|shard| {
            shard.bank == chunk.bank
                && shard.offset >= chunk.logical_offset
                && shard
                    .offset
                    .checked_add(shard.bytes)
                    .is_some_and(|end| end <= chunk_end)
        })
        .collect::<Vec<_>>();
    let blueprint = persistent_chunk_blueprint(chunk.bank, chunk.logical_offset, &mut plan_shards)?;
    if blueprint.bytes != chunk.bytes || blueprint.shards != chunk.shards {
        return Err("persistent IQ1_S chunk metadata differs from the arena plan".to_string());
    }
    let source_map = sources
        .iter()
        .map(|source| (source.identity.name.clone(), source.clone()))
        .collect::<HashMap<_, _>>();
    let mut files = HashMap::<String, File>::new();
    let mut output = vec![0u8; chunk.bytes];
    for (shard_spec, shard) in chunk.shards.iter().zip(plan_shards) {
        let source = source_map
            .get(&shard.tensor.name)
            .ok_or_else(|| format!("missing IQ1_S arena source {}", shard.tensor.name))?;
        if source.identity != *shard.tensor {
            return Err(format!(
                "IQ1_S arena source identity mismatch for {}",
                shard.tensor.name
            ));
        }
        if !files.contains_key(&shard.tensor.name) {
            files.insert(
                shard.tensor.name.clone(),
                open_verified_source(&shard.tensor)?,
            );
        }
        let bytes = read_shard_bytes(&files[&shard.tensor.name], shard)?;
        if <[u8; 32]>::from(Sha256::digest(&bytes)) != shard_spec.sha256 {
            return Err(format!(
                "persistent IQ1_S shard hash mismatch for {} expert {} bank {}",
                shard.tensor.name, shard.expert, shard.bank
            ));
        }
        let relative = usize::try_from(shard.offset - chunk.logical_offset)
            .map_err(|_| "persistent IQ1_S chunk offset does not fit usize")?;
        let end = relative
            .checked_add(bytes.len())
            .ok_or("persistent IQ1_S chunk copy overflow")?;
        output
            .get_mut(relative..end)
            .ok_or("persistent IQ1_S shard lies outside its chunk")?
            .copy_from_slice(&bytes);
    }
    let digest: [u8; 32] = Sha256::digest(&output).into();
    if verify_chunk_hash && digest != chunk.sha256 {
        return Err("persistent IQ1_S chunk hash mismatch".to_string());
    }
    Ok(output)
}

pub(crate) fn persistent_chunk_specs(plan: &ArenaPlan) -> Result<Vec<ArenaChunkSpec>, String> {
    let sources = plan
        .shards
        .iter()
        .map(|shard| shard.tensor.clone())
        .map(|identity| {
            (
                identity.name.clone(),
                Arc::new(Iq1sTensorSource {
                    identity: (*identity).clone(),
                }),
            )
        })
        .collect::<BTreeMap<_, _>>()
        .into_values()
        .collect::<Vec<_>>();
    let mut chunks = persistent_chunk_blueprints(plan)?;
    for chunk in &mut chunks {
        let bytes = assemble_persistent_chunk(plan, &sources, chunk, false)?;
        chunk.sha256 = Sha256::digest(bytes).into();
    }
    Ok(chunks)
}

pub(crate) fn read_persistent_chunk(
    plan: &ArenaPlan,
    sources: &[Arc<Iq1sTensorSource>],
    chunk: &ArenaChunkSpec,
) -> Result<Vec<u8>, String> {
    assemble_persistent_chunk(plan, sources, chunk, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom, Write};
    use std::path::{Path, PathBuf};
    use std::process::Command;

    fn identity(
        path: PathBuf,
        layer: u32,
        role: Iq1sExpertRole,
        ne: [u64; 4],
        nb: [u64; 4],
        nbytes: u64,
        model_sha256: [u8; 32],
        file_metadata: Option<&std::fs::Metadata>,
    ) -> Arc<Iq1sTensorSource> {
        let role_name = match role {
            Iq1sExpertRole::Gate => "gate",
            Iq1sExpertRole::Up => "up",
            Iq1sExpertRole::Down => "down",
            Iq1sExpertRole::GateUp => "gate_up",
        };
        let (device, inode, modified_ns) = if let Some(metadata) = file_metadata {
            (
                metadata.dev(),
                metadata.ino(),
                metadata
                    .modified()
                    .unwrap()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
            )
        } else {
            (1, 2 + u64::from(layer), 3)
        };
        Arc::new(Iq1sTensorSource {
            identity: Iq1sTensorIdentity {
                canonical_path: path,
                file_offset: 0,
                nbytes,
                name: format!("blk.{layer}.ffn_{role_name}_exps.weight"),
                layer,
                ne,
                nb,
                role,
                model_sha256,
                content_sha256: [0x44; 32],
                device,
                inode,
                modified_ns,
            },
        })
    }

    fn canonical_sources(path: &Path, model_sha256: [u8; 32]) -> Vec<Arc<Iq1sTensorSource>> {
        let mut result = Vec::new();
        for layer in 0..50 {
            for role in [Iq1sExpertRole::Gate, Iq1sExpertRole::Up] {
                result.push(identity(
                    path.to_path_buf(),
                    layer,
                    role,
                    [4096, 1024, 512, 1],
                    [50, 800, 819_200, 419_430_400],
                    419_430_400,
                    model_sha256,
                    None,
                ));
            }
            if layer < 41 {
                result.push(identity(
                    path.to_path_buf(),
                    layer,
                    Iq1sExpertRole::Down,
                    [1024, 4096, 512, 1],
                    [50, 200, 819_200, 419_430_400],
                    419_430_400,
                    model_sha256,
                    None,
                ));
            }
        }
        result
    }

    #[test]
    fn iq1s_weight_arena_has_exact_qwen_counts_capacity_and_row_coverage() {
        let sources = canonical_sources(Path::new("/tmp/qwen-metadata.gguf"), [0x11; 32]);
        let plan = build_plan(&sources, 7, true, false).unwrap();
        assert_eq!(sources.len(), ARENA_EXPECTED_TENSORS);
        assert_eq!(
            sources
                .iter()
                .map(|source| source.identity.nbytes)
                .sum::<u64>(),
            ARENA_EXPECTED_RAW_BYTES
        );
        assert_eq!(
            plan.shards.len(),
            ARENA_EXPECTED_TENSORS * 512 * ARENA_BANK_COUNT
        );
        assert!(plan
            .bank_bytes
            .iter()
            .all(|bytes| *bytes <= ARENA_BANK_CAPACITY_BYTES));
        assert!(plan
            .bank_bytes
            .iter()
            .all(|bytes| bytes % ARENA_ALIGNMENT == 0));

        let mut coverage = HashMap::<(String, u16), Vec<(u32, u32)>>::new();
        for shard in &plan.shards {
            assert_eq!(shard.offset % ARENA_ALIGNMENT, 0);
            assert_eq!(
                u64::from(shard.superblock),
                shard.offset / ARENA_SUPERBLOCK_BYTES
            );
            assert!(shard.offset % ARENA_SUPERBLOCK_BYTES + shard.bytes <= ARENA_SUPERBLOCK_BYTES);
            coverage
                .entry((shard.tensor.name.clone(), shard.expert))
                .or_default()
                .push((shard.row_start, shard.row_count));
        }
        for ((name, _), mut ranges) in coverage {
            ranges.sort_unstable();
            let rows = if name.contains(".ffn_down_exps.") {
                4096
            } else {
                1024
            };
            let mut cursor = 0u32;
            for (start, count) in ranges {
                assert_eq!(start, cursor);
                cursor += count;
            }
            assert_eq!(cursor, rows);
        }
        assert!(plan.shards.iter().any(|shard| shard.row_count == 256));
        assert!(plan.shards.iter().any(|shard| shard.row_count == 1024));
    }

    #[derive(Default)]
    struct FakeBackend {
        allocations: Vec<(u8, usize)>,
        writes: usize,
        syncs: usize,
        fail_writes: bool,
    }

    impl ArenaBackend for FakeBackend {
        type Handle = u8;

        fn allocate(&mut self, bank: u8, bytes: usize) -> Result<Self::Handle, String> {
            self.allocations.push((bank, bytes));
            Ok(bank)
        }

        fn write_range(
            &mut self,
            _handle: &Self::Handle,
            _offset: usize,
            _bytes: &[u8],
        ) -> Result<(), String> {
            if self.fail_writes {
                return Err("injected arena write failure".to_string());
            }
            self.writes += 1;
            Ok(())
        }

        fn sync_to_device(
            &mut self,
            _handle: &Self::Handle,
            _offset: usize,
            _bytes: usize,
        ) -> Result<(), String> {
            self.syncs += 1;
            Ok(())
        }

        fn device_address(&self, handle: &Self::Handle) -> Result<u64, String> {
            Ok(0x1_0000 + u64::from(*handle) * 0x1_0000)
        }
    }

    #[test]
    fn iq1s_weight_arena_rejects_changed_source_shard() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&vec![0x5a; 800]).unwrap();
        file.flush().unwrap();
        let metadata = file.as_file().metadata().unwrap();
        let source = identity(
            file.path().canonicalize().unwrap(),
            0,
            Iq1sExpertRole::Gate,
            [256, 8, 2, 1],
            [50, 50, 400, 800],
            800,
            [0x22; 32],
            Some(&metadata),
        );
        let sources = vec![source];
        let plan = build_plan(&sources, 9, false, true).unwrap();

        file.as_file_mut().seek(SeekFrom::Start(0)).unwrap();
        file.as_file_mut().write_all(&[0x6b]).unwrap();
        file.as_file_mut().flush().unwrap();
        let modified = file.as_file().metadata().unwrap().modified().unwrap();
        let original_modified = UNIX_EPOCH
            + std::time::Duration::from_nanos(
                u64::try_from(sources[0].identity.modified_ns).unwrap(),
            );
        if modified != original_modified {
            file.as_file().set_modified(original_modified).unwrap();
        }

        let error = load_arena(&mut FakeBackend::default(), plan, &sources).unwrap_err();
        assert!(error.contains("shard hash mismatch"), "{error}");
    }

    #[test]
    fn iq1s_weight_arena_batches_one_dma_per_populated_superblock() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&vec![0x3c; 800]).unwrap();
        file.flush().unwrap();
        let metadata = file.as_file().metadata().unwrap();
        let sources = vec![identity(
            file.path().canonicalize().unwrap(),
            0,
            Iq1sExpertRole::Gate,
            [256, 8, 2, 1],
            [50, 50, 400, 800],
            800,
            [0x23; 32],
            Some(&metadata),
        )];
        let plan = build_plan(&sources, 12, false, true).unwrap();
        let mut backend = FakeBackend::default();
        let resident = load_arena(&mut backend, plan, &sources).unwrap();

        assert_eq!(backend.allocations.len(), ARENA_BANK_COUNT);
        assert_eq!(backend.writes, ARENA_BANK_COUNT);
        assert_eq!(backend.syncs, ARENA_BANK_COUNT);
        assert_eq!(
            resident.device_addresses,
            [0x1_0000, 0x2_0000, 0x3_0000, 0x4_0000]
        );
    }

    #[test]
    fn iq1s_persistent_runtime_exposes_verified_bounded_chunks() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&vec![0x3c; 800]).unwrap();
        file.flush().unwrap();
        let metadata = file.as_file().metadata().unwrap();
        let sources = vec![identity(
            file.path().canonicalize().unwrap(),
            0,
            Iq1sExpertRole::Gate,
            [256, 8, 2, 1],
            [50, 50, 400, 800],
            800,
            [0x27; 32],
            Some(&metadata),
        )];
        let plan = build_plan(&sources, 14, false, true).unwrap();
        let chunks = persistent_chunk_specs(&plan).unwrap();

        assert_eq!(chunks.len(), ARENA_BANK_COUNT);
        for chunk in &chunks {
            assert!(chunk.bytes as u64 <= ARENA_SUPERBLOCK_BYTES);
            let bytes = read_persistent_chunk(&plan, &sources, chunk).unwrap();
            assert_eq!(bytes.len(), chunk.bytes);
            assert_eq!(<[u8; 32]>::from(Sha256::digest(bytes)), chunk.sha256);
        }

        let mut misaligned = chunks[0].clone();
        misaligned.logical_offset += 1;
        assert!(read_persistent_chunk(&plan, &sources, &misaligned)
            .unwrap_err()
            .contains("staging chunk offset is not aligned"));

        let mut gapped = plan;
        gapped.shards[4].offset += ARENA_ALIGNMENT;
        assert!(persistent_chunk_specs(&gapped)
            .unwrap_err()
            .contains("alignment padding"));
    }

    #[test]
    fn qwen_persistent_weight_staging_is_bounded_to_64_mib() {
        let sources = canonical_sources(Path::new("/tmp/qwen-metadata.gguf"), [0x31; 32]);
        let mut plan = build_plan(&sources, 15, true, false).unwrap();
        plan.hashes_verified = true;
        for shard in &mut plan.shards {
            shard.sha256 = [0x5a; 32];
        }
        let chunks = persistent_chunk_blueprints(&plan).unwrap();

        assert!(chunks.len() > ARENA_BANK_COUNT);
        assert!(chunks.iter().all(|chunk| chunk.bytes <= 64 * 1024 * 1024));
    }

    #[test]
    fn iq1s_weight_arena_rejects_tampered_layout_before_allocation() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&vec![0x4d; 800]).unwrap();
        file.flush().unwrap();
        let metadata = file.as_file().metadata().unwrap();
        let sources = vec![identity(
            file.path().canonicalize().unwrap(),
            0,
            Iq1sExpertRole::Gate,
            [256, 8, 2, 1],
            [50, 50, 400, 800],
            800,
            [0x24; 32],
            Some(&metadata),
        )];
        let mut plan = build_plan(&sources, 13, false, true).unwrap();
        plan.shards[4].offset = plan.shards[0].offset;
        plan.shards[4].superblock = plan.shards[0].superblock;
        let mut backend = FakeBackend::default();

        let error = load_arena(&mut backend, plan, &sources).unwrap_err();
        assert!(error.contains("overlapping"), "{error}");
        assert!(backend.allocations.is_empty());
    }

    #[test]
    fn iq1s_weight_arena_generation_cannot_be_replaced_in_flight() {
        let tracker = ArenaGenerationTracker::default();
        tracker.install(10).unwrap();
        let lease = tracker.acquire(10).unwrap();
        assert!(tracker.install(11).unwrap_err().contains("in flight"));
        drop(lease);
        tracker.install(11).unwrap();
        assert!(tracker.acquire(10).is_err());
        tracker.acquire(11).unwrap();
    }

    #[test]
    fn iq1s_shared_weight_arena_preserves_resident_generation_on_failure() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&vec![0x2d; 800]).unwrap();
        file.flush().unwrap();
        let metadata = file.as_file().metadata().unwrap();
        let sources = vec![identity(
            file.path().canonicalize().unwrap(),
            0,
            Iq1sExpertRole::Gate,
            [256, 8, 2, 1],
            [50, 50, 400, 800],
            800,
            [0x25; 32],
            Some(&metadata),
        )];
        let plan_20 = build_plan(&sources, 20, false, true).unwrap();
        let mut plan_21 = plan_20.clone();
        plan_21.generation = 21;
        let mut arena = SharedWeightArena::new(FakeBackend::default());
        arena.install(plan_20, &sources).unwrap();

        let lease = arena.acquire(20).unwrap();
        assert!(arena
            .install(plan_21.clone(), &sources)
            .unwrap_err()
            .contains("in flight"));
        drop(lease);

        arena.backend.fail_writes = true;
        assert!(arena
            .install(plan_21.clone(), &sources)
            .unwrap_err()
            .contains("injected"));
        arena.acquire(20).unwrap();
        assert_eq!(arena.resident().unwrap().plan.generation, 20);

        arena.backend.fail_writes = false;
        arena.install(plan_21, &sources).unwrap();
        arena.acquire(21).unwrap();
        assert_eq!(arena.resident().unwrap().plan.generation, 21);
    }

    #[test]
    fn iq1s_weight_arena_rejects_incomplete_or_mixed_model_sources() {
        let mut sources = canonical_sources(Path::new("/tmp/qwen-metadata.gguf"), [0x11; 32]);
        sources.pop();
        assert!(build_plan(&sources, 1, true, false)
            .unwrap_err()
            .contains("141"));
        let mut sources = canonical_sources(Path::new("/tmp/qwen-metadata.gguf"), [0x11; 32]);
        Arc::make_mut(&mut sources[0]).identity.model_sha256 = [0x12; 32];
        assert!(build_plan(&sources, 1, true, false)
            .unwrap_err()
            .contains("multiple models"));
    }

    #[test]
    #[ignore = "requires the pinned 88 GiB Qwen model"]
    fn qwen_real_model_arena_plan() {
        let expected_hash = std::env::var("HETGPU_QWEN_MODEL_SHA256").unwrap();
        assert_eq!(
            expected_hash,
            "0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568"
        );
        let model = Path::new("/root/models/qwen35-tq1/Qwen3.5-397B-A17B-UD-TQ1_0.gguf");
        assert_eq!(model.metadata().unwrap().len(), 94_155_830_880);
        let tools = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("tools");
        let script = r#"import sys
sys.path.insert(0, sys.argv[1])
import qwen35_gguf_audit as q
arch, tensors = q.parse_header(sys.argv[2])
print(arch)
for t in tensors:
    if t.type_id == 19:
        print(t.name + '|' + ','.join(str(x) for x in t.dimensions))"#;
        let output = Command::new("python3")
            .arg("-c")
            .arg(script)
            .arg(&tools)
            .arg(model)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        let mut lines = stdout.lines();
        assert_eq!(lines.next(), Some("qwen35moe"));
        let mut sources = Vec::new();
        let mut model_hash = [0u8; 32];
        for (index, byte) in model_hash.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&expected_hash[index * 2..index * 2 + 2], 16).unwrap();
        }
        for line in lines {
            let (name, dimensions) = line.split_once('|').unwrap();
            let (layer, role) = classify_tensor_name(name).unwrap();
            let values = dimensions
                .split(',')
                .map(|value| value.parse::<u64>().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(values.len(), 3);
            let ne = [values[0], values[1], values[2], 1];
            let row_bytes = ne[0] / 256 * 50;
            let expert_bytes = ne[1] * row_bytes;
            let tensor_bytes = ne[2] * expert_bytes;
            let mut source = identity(
                model.to_path_buf(),
                layer,
                role,
                ne,
                [50, row_bytes, expert_bytes, tensor_bytes],
                tensor_bytes,
                model_hash,
                None,
            );
            Arc::make_mut(&mut source).identity.name = name.to_string();
            sources.push(source);
        }
        let plan = build_plan(&sources, 1, true, false).unwrap();
        assert_eq!(sources.len(), 141);
        assert_eq!(plan.shards.len(), 141 * 512 * 4);
        assert!(plan
            .bank_bytes
            .iter()
            .all(|bytes| *bytes <= ARENA_BANK_CAPACITY_BYTES));
        assert!(!plan.hashes_verified);
    }
}
