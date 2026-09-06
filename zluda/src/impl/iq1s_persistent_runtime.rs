use super::cxl_tmatmul::copy_host_to_cuda;
use super::iq1s_layer::{CapturedProjection, LayerKey, RouteAssignment};
use super::iq1s_layer_trace::{
    compile_layer_phase, ActivationRange, CompiledLayerPhase, LayerPhase, LayerPhasePlan,
    SemanticIq1sCommand,
};
use super::iq1s_tmatmul::Q8_1_MMQ_BYTES;
use super::iq1s_trace::QWEN_MODEL_CONTEXT_LIMIT;
use super::iq1s_weight_arena::{
    persistent_chunk_specs, plan_registered_arena, read_persistent_chunk, ArenaPlan, ArenaShard,
    ARENA_ALIGNMENT, ARENA_EXPECTED_RAW_BYTES, ARENA_EXPECTED_TENSORS,
};
use super::iq1s_weight_registry::{global_registry, Iq1sExpertRole, Iq1sTensorSource};
use super::xrt_iq1s_persistent::{
    CompletedLayerPhase, HostRange, PersistentIq1sConfig, PersistentIq1sPool, PhaseBuffers,
};
use super::xrt_tmatmul::RealXrt;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

const QWEN_MODEL_SHA256: [u8; 32] = [
    0x0a, 0x32, 0xc2, 0x70, 0x2f, 0xbb, 0x61, 0x93, 0x49, 0x60, 0xcf, 0xee, 0xf3, 0x45, 0x24, 0xb8,
    0x1e, 0xc6, 0xd9, 0x26, 0x71, 0x58, 0xf2, 0x46, 0xd4, 0x5f, 0xc8, 0x6f, 0x5a, 0xaa, 0x75, 0x68,
];
const PERSISTENT_XCLBIN_SHA256: [u8; 32] = [
    0x9c, 0x83, 0xdc, 0xae, 0x07, 0xb4, 0xc7, 0xbf, 0x1d, 0x2e, 0x1c, 0xeb, 0xf4, 0x6c, 0xcf, 0x0f,
    0xf1, 0xeb, 0xf8, 0x84, 0x8a, 0x43, 0x70, 0x35, 0xfe, 0xf1, 0x45, 0x1d, 0xee, 0x77, 0x70, 0xa3,
];
const PERSISTENT_XCLBIN_NAME: &str = "qwen397b_iq1s_layer_persistent_9c83dcae.xclbin";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PersistentRuntimeIdentity {
    pub(crate) model_sha256: [u8; 32],
    pub(crate) xclbin_sha256: [u8; 32],
    pub(crate) device_index: u32,
    pub(crate) session_generation: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct PhaseSnapshot {
    pub(crate) key: LayerKey,
    pub(crate) batch_count: u16,
    pub(crate) routes: Vec<RouteAssignment>,
    pub(crate) projections: Vec<CapturedProjection>,
    pub(crate) phase: LayerPhase,
}

#[derive(Debug, Clone)]
pub(crate) struct OutputBinding {
    pub(crate) cuda_ptr: usize,
    pub(crate) expert_id: u16,
    pub(crate) token_id: u32,
    pub(crate) row_count: u32,
    role: Iq1sExpertRole,
    lane_index: usize,
    output_offset: u64,
    shards: Vec<ArenaShard>,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedPhase {
    pub(crate) arena: Arc<ArenaPlan>,
    pub(crate) compiled: CompiledLayerPhase,
    pub(crate) buffers: PhaseBuffers,
    pub(crate) output_bindings: Vec<OutputBinding>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PublishedOutput {
    pub(crate) cuda_ptr: usize,
    pub(crate) bytes: Vec<u8>,
}

#[derive(Debug)]
pub(crate) struct PhaseOutcome {
    pub(crate) prepared: PreparedPhase,
    pub(crate) completed: CompletedLayerPhase,
    pub(crate) outputs: Vec<PublishedOutput>,
}

pub(crate) struct PersistentRuntime {
    identity: PersistentRuntimeIdentity,
    arena: Arc<ArenaPlan>,
    sources: Vec<Arc<Iq1sTensorSource>>,
    pool: PersistentIq1sPool<RealXrt>,
    poisoned: Option<String>,
}

// XRT opaque handles and the dlopen handle are process-wide C handles.  The
// runtime never exposes them and every operation is serialized by RUNTIME's
// mutex, so moving this owner between CUDA-calling host threads cannot create
// concurrent access or outlive the owning pool.
unsafe impl Send for PersistentRuntime {}

type RuntimeBinding = Option<(PersistentRuntimeIdentity, Arc<ArenaPlan>)>;

fn bind_runtime_identity(
    binding: &mut RuntimeBinding,
    identity: PersistentRuntimeIdentity,
    arena: Arc<ArenaPlan>,
) -> Result<Arc<ArenaPlan>, String> {
    if let Some((installed, resident)) = binding {
        if installed != &identity || resident.as_ref() != arena.as_ref() {
            return Err(
                "persistent IQ1_S runtime identity cannot change in one process".to_string(),
            );
        }
        return Ok(resident.clone());
    }
    *binding = Some((identity, arena.clone()));
    Ok(arena)
}

pub(crate) fn pack_q8_lanes(lanes: &[Vec<u8>], records: usize) -> Result<Vec<u8>, String> {
    if lanes.is_empty() || lanes.len() > 16 || records == 0 {
        return Err("persistent IQ1_S Q8 pack requires 1..=16 lanes and positive K records".into());
    }
    let lane_bytes = records
        .checked_mul(Q8_1_MMQ_BYTES)
        .ok_or("persistent IQ1_S Q8 lane byte count overflow")?;
    if lanes.iter().any(|lane| lane.len() != lane_bytes) {
        return Err("persistent IQ1_S Q8 lane length does not match K records".to_string());
    }
    let total = lane_bytes
        .checked_mul(lanes.len())
        .ok_or("persistent IQ1_S packed activation size overflow")?;
    let mut packed = Vec::with_capacity(total);
    for record in 0..records {
        let start = record * Q8_1_MMQ_BYTES;
        let end = start + Q8_1_MMQ_BYTES;
        for lane in lanes {
            packed.extend_from_slice(&lane[start..end]);
        }
    }
    Ok(packed)
}

fn align_up(value: u64, alignment: u64) -> Result<u64, String> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .ok_or("persistent IQ1_S slab alignment overflow".to_string())
}

fn role_matches_phase(role: Iq1sExpertRole, phase: LayerPhase) -> bool {
    matches!(
        (role, phase),
        (
            Iq1sExpertRole::Gate | Iq1sExpertRole::Up,
            LayerPhase::PhaseA
        ) | (Iq1sExpertRole::Down, LayerPhase::PhaseB)
    )
}

pub(crate) fn prepare_phase(
    arena: &Arc<ArenaPlan>,
    snapshot: PhaseSnapshot,
    trace_mode: &str,
) -> Result<PreparedPhase, String> {
    if !matches!(trace_mode, "handwritten" | "compiler") {
        return Err("persistent IQ1_S trace mode must be handwritten or compiler".to_string());
    }
    if snapshot.key.session_generation != arena.generation
        || snapshot.key.transaction_id == 0
        || snapshot.batch_count == 0
        || snapshot.batch_count > 16
        || snapshot.routes.is_empty()
        || snapshot.projections.is_empty()
    {
        return Err("persistent IQ1_S phase snapshot metadata is invalid".to_string());
    }
    let token_ids = snapshot
        .routes
        .iter()
        .map(|route| route.token_id)
        .collect::<BTreeSet<_>>();
    if token_ids.len() != usize::from(snapshot.batch_count)
        || token_ids
            .iter()
            .copied()
            .ne(0..u32::from(snapshot.batch_count))
    {
        return Err("persistent IQ1_S phase token IDs must exactly cover the active batch".into());
    }

    let mut groups = BTreeMap::<
        (Iq1sExpertRole, u16),
        Vec<(
            RouteAssignment,
            super::iq1s_tmatmul::CapturedActivationLaunch,
        )>,
    >::new();
    for projection in &snapshot.projections {
        if !role_matches_phase(projection.role, snapshot.phase)
            || projection.weight.identity.layer != snapshot.key.layer_id
            || projection.weight.identity.role != projection.role
            || projection.launches.len() != snapshot.routes.len()
        {
            return Err("persistent IQ1_S projection does not match its phase snapshot".into());
        }
        for (route, launch) in snapshot
            .routes
            .iter()
            .copied()
            .zip(projection.launches.iter().cloned())
        {
            if !route.route_weight.is_finite()
                || route.expert_id >= 512
                || launch.launch.allocation_generation != arena.generation
                || launch.launch.content_hash != projection.weight.content_sha256
                || launch.launch.signature.ne00 != projection.weight.identity.ne[0]
                || launch.launch.signature.ne01 != projection.weight.identity.ne[1]
            {
                return Err("persistent IQ1_S route or activation identity is invalid".into());
            }
            groups
                .entry((projection.role, route.expert_id))
                .or_default()
                .push((route, launch));
        }
    }

    let mut activation_cursor = 0u64;
    let mut token_cursor = 0u64;
    let mut output_cursor = 0u64;
    let mut commands = Vec::new();
    let mut activation_manifest = Vec::new();
    let mut activation_buffers = Vec::new();
    let mut token_maps = Vec::new();
    let mut output_bindings = Vec::new();
    for ((role, expert_id), mut lanes) in groups {
        lanes.sort_by_key(|(route, _)| route.token_id);
        if lanes
            .windows(2)
            .any(|pair| pair[0].0.token_id == pair[1].0.token_id)
        {
            return Err("persistent IQ1_S expert group repeats a token lane".to_string());
        }
        let identity = snapshot
            .projections
            .iter()
            .find(|projection| projection.role == role)
            .map(|projection| &projection.weight.identity)
            .ok_or("persistent IQ1_S group lost its projection identity")?;
        let k_records = usize::try_from(identity.ne[0] / 128)
            .map_err(|_| "persistent IQ1_S K record count does not fit usize")?;
        let lane_bytes = lanes
            .iter()
            .map(|(_, launch)| launch.packed_activations().to_vec())
            .collect::<Vec<_>>();
        let packed = pack_q8_lanes(&lane_bytes, k_records)?;
        activation_cursor = align_up(activation_cursor, ARENA_ALIGNMENT)?;
        token_cursor = align_up(token_cursor, ARENA_ALIGNMENT)?;
        output_cursor = align_up(output_cursor, ARENA_ALIGNMENT)?;
        let activation_offset = activation_cursor;
        let token_offset = token_cursor;
        let output_offset = output_cursor;
        let token_bytes = lanes
            .iter()
            .flat_map(|(route, _)| route.token_id.to_le_bytes())
            .collect::<Vec<_>>();
        let shards = (0..4u8)
            .map(|bank| {
                arena
                    .shards
                    .iter()
                    .find(|shard| {
                        shard.tensor.name == identity.name
                            && shard.expert == expert_id
                            && shard.bank == bank
                    })
                    .cloned()
                    .ok_or_else(|| {
                        format!(
                            "persistent IQ1_S arena omitted role {role:?} expert {expert_id} bank {bank}"
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let lane_count = u64::try_from(lanes.len())
            .map_err(|_| "persistent IQ1_S lane count does not fit u64")?;
        let max_result_bytes = shards
            .iter()
            .map(|shard| u64::from(shard.row_count) * 4 * lane_count)
            .max()
            .ok_or("persistent IQ1_S group has no arena shards")?;
        let token_ids = lanes
            .iter()
            .map(|(route, _)| route.token_id)
            .collect::<Vec<_>>();
        let lane_mask = token_ids
            .iter()
            .fold(0u16, |mask, token| mask | (1u16 << *token));
        for shard in &shards {
            commands.push(SemanticIq1sCommand {
                layer_id: snapshot.key.layer_id,
                phase: snapshot.phase,
                role,
                expert_id,
                lane_mask,
                token_ids: token_ids.clone(),
                input_offset: activation_offset,
                output_offset,
                token_map_offset: token_offset,
                row_shard: shard.clone(),
            });
        }
        for (lane_index, (route, launch)) in lanes.iter().enumerate() {
            output_bindings.push(OutputBinding {
                cuda_ptr: launch.launch.output_ptr,
                expert_id,
                token_id: route.token_id,
                row_count: u32::try_from(identity.ne[1])
                    .map_err(|_| "persistent IQ1_S output rows do not fit u32")?,
                role,
                lane_index,
                output_offset,
                shards: shards.clone(),
            });
        }
        activation_manifest.push(ActivationRange {
            cuda_ptr: lanes[0].1.launch.activation_ptr,
            slab_offset: activation_offset,
            bytes: u32::try_from(packed.len())
                .map_err(|_| "persistent IQ1_S activation bytes do not fit u32")?,
            stream: snapshot.key.stream,
        });
        activation_buffers.push(HostRange {
            offset: activation_offset,
            bytes: packed,
        });
        token_maps.push(HostRange {
            offset: token_offset,
            bytes: token_bytes,
        });
        activation_cursor = activation_offset
            .checked_add(activation_buffers.last().unwrap().bytes.len() as u64)
            .ok_or("persistent IQ1_S activation slab overflow")?;
        token_cursor = token_offset
            .checked_add(token_maps.last().unwrap().bytes.len() as u64)
            .ok_or("persistent IQ1_S token-map slab overflow")?;
        output_cursor = output_offset
            .checked_add(max_result_bytes)
            .ok_or("persistent IQ1_S result slab overflow")?;
    }
    let compiled = compile_layer_phase(
        &LayerPhasePlan {
            transaction_id: snapshot.key.transaction_id,
            phase: snapshot.phase,
            commands,
            activations: activation_manifest,
        },
        trace_mode,
        QWEN_MODEL_CONTEXT_LIMIT,
    )?;
    Ok(PreparedPhase {
        arena: arena.clone(),
        compiled,
        buffers: PhaseBuffers {
            activations: activation_buffers,
            token_maps,
        },
        output_bindings,
    })
}

fn result_slice<'a>(
    completed: &'a CompletedLayerPhase,
    bank: usize,
    offset: u64,
    bytes: usize,
) -> Result<&'a [u8], String> {
    let end = offset
        .checked_add(bytes as u64)
        .ok_or("persistent IQ1_S result lookup overflow")?;
    for range in &completed.results[bank] {
        let range_end = range
            .offset
            .checked_add(range.bytes.len() as u64)
            .ok_or("persistent IQ1_S completed result range overflow")?;
        if offset >= range.offset && end <= range_end {
            let start = usize::try_from(offset - range.offset)
                .map_err(|_| "persistent IQ1_S result relative offset does not fit usize")?;
            return Ok(&range.bytes[start..start + bytes]);
        }
    }
    Err(format!(
        "persistent IQ1_S result bank {bank} does not cover offset {offset:#x}"
    ))
}

pub(crate) fn reconstruct_full_rows(
    prepared: &PreparedPhase,
    completed: &CompletedLayerPhase,
) -> Result<Vec<PublishedOutput>, String> {
    if completed.transaction_id != prepared.compiled.transaction_id
        || completed.semantic_sha256 != prepared.compiled.semantic_sha256
    {
        return Err("persistent IQ1_S completion identity differs from prepared phase".into());
    }
    let mut outputs = Vec::with_capacity(prepared.output_bindings.len());
    for binding in &prepared.output_bindings {
        let bytes = usize::try_from(binding.row_count)
            .ok()
            .and_then(|rows| rows.checked_mul(4))
            .ok_or("persistent IQ1_S full output byte count overflow")?;
        let mut output = vec![0u8; bytes];
        let mut covered = vec![false; binding.row_count as usize];
        for shard in &binding.shards {
            let bank = usize::from(shard.bank);
            let shard_bytes = usize::try_from(shard.row_count)
                .ok()
                .and_then(|rows| rows.checked_mul(4))
                .ok_or("persistent IQ1_S row-shard output size overflow")?;
            let lane_offset = binding
                .output_offset
                .checked_add((binding.lane_index * shard_bytes) as u64)
                .ok_or("persistent IQ1_S lane result offset overflow")?;
            let source = result_slice(completed, bank, lane_offset, shard_bytes)?;
            let destination = shard.row_start as usize * 4;
            let destination_end = destination + shard_bytes;
            output
                .get_mut(destination..destination_end)
                .ok_or("persistent IQ1_S row shard exceeds full output")?
                .copy_from_slice(source);
            for row in shard.row_start..shard.row_start + shard.row_count {
                let covered_row = covered
                    .get_mut(row as usize)
                    .ok_or("persistent IQ1_S row coverage exceeds output")?;
                if *covered_row {
                    return Err("persistent IQ1_S row coverage overlaps".to_string());
                }
                *covered_row = true;
            }
        }
        if covered.iter().any(|covered| !covered) {
            return Err("persistent IQ1_S row coverage has a gap".to_string());
        }
        outputs.push(PublishedOutput {
            cuda_ptr: binding.cuda_ptr,
            bytes: output,
        });
    }
    Ok(outputs)
}

fn validate_all_finite(outputs: &[PublishedOutput]) -> Result<(), String> {
    for output in outputs {
        if output.bytes.len() % 4 != 0
            || output.bytes.chunks_exact(4).any(|bytes| {
                !f32::from_le_bytes(bytes.try_into().expect("four-byte float")).is_finite()
            })
        {
            return Err("persistent IQ1_S output contains a nonfinite or partial f32".to_string());
        }
    }
    Ok(())
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "on" | "ON"))
        .unwrap_or(false)
}

fn env_u64(name: &str, default: u64) -> Result<u64, String> {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<u64>()
            .map_err(|_| format!("{name} must be an unsigned integer")),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(format!("read {name}: {error}")),
    }
}

fn sha256_file(path: &Path) -> Result<[u8; 32], String> {
    let mut file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hash.finalize().into())
}

fn initialize_runtime_from_env() -> Result<PersistentRuntime, String> {
    if !env_truthy("HETGPU_QWEN_IQ1S_STRICT") || !env_truthy("HETGPU_QWEN_IQ1S_PERSISTENT") {
        return Err("persistent IQ1_S runtime requires strict persistent mode".to_string());
    }
    let generation = env_u64("HETGPU_QWEN_IQ1S_SESSION_GENERATION", 1)?;
    if generation == 0 {
        return Err("persistent IQ1_S session generation must be nonzero".to_string());
    }
    let sources = global_registry().registered_sources()?;
    if sources.len() != ARENA_EXPECTED_TENSORS
        || sources
            .iter()
            .map(|source| source.identity.nbytes)
            .sum::<u64>()
            != ARENA_EXPECTED_RAW_BYTES
    {
        return Err("persistent IQ1_S registry does not contain the exact Qwen tensor set".into());
    }
    let arena = Arc::new(plan_registered_arena(global_registry(), generation)?);
    if arena.model_sha256 != QWEN_MODEL_SHA256 || !arena.hashes_verified {
        return Err("persistent IQ1_S arena model identity or hashes are invalid".to_string());
    }
    let xclbin = PathBuf::from(
        std::env::var("HETGPU_XRT_XCLBIN")
            .map_err(|_| "HETGPU_XRT_XCLBIN is required for persistent IQ1_S".to_string())?,
    );
    if xclbin.file_name().and_then(|name| name.to_str()) != Some(PERSISTENT_XCLBIN_NAME)
        || sha256_file(&xclbin)? != PERSISTENT_XCLBIN_SHA256
    {
        return Err("persistent IQ1_S xclbin name or SHA-256 is not qualified".to_string());
    }
    let device_index = u32::try_from(env_u64("HETGPU_XRT_DEVICE_INDEX", 0)?)
        .map_err(|_| "HETGPU_XRT_DEVICE_INDEX does not fit u32")?;
    let ring_capacity = u32::try_from(env_u64("HETGPU_QWEN_IQ1S_RING_CAPACITY", 512)?)
        .map_err(|_| "HETGPU_QWEN_IQ1S_RING_CAPACITY does not fit u32")?;
    let timeout_ms = u32::try_from(env_u64("HETGPU_XRT_TIMEOUT_MS", 10_000)?)
        .map_err(|_| "HETGPU_XRT_TIMEOUT_MS does not fit u32")?;
    let chunks = persistent_chunk_specs(&arena)?;
    let ops = RealXrt::load(true).map_err(|error| error.to_string())?;
    let config =
        PersistentIq1sConfig::checked(xclbin, device_index, Some(ring_capacity), timeout_ms)
            .map_err(|error| error.to_string())?;
    let pool = PersistentIq1sPool::open(ops, config, generation, &chunks, |chunk| {
        read_persistent_chunk(&arena, &sources, chunk)
    })
    .map_err(|error| error.to_string())?;
    Ok(PersistentRuntime {
        identity: PersistentRuntimeIdentity {
            model_sha256: QWEN_MODEL_SHA256,
            xclbin_sha256: PERSISTENT_XCLBIN_SHA256,
            device_index,
            session_generation: generation,
        },
        arena,
        sources,
        pool,
        poisoned: None,
    })
}

enum GlobalRuntimeState {
    Uninitialized,
    Ready(PersistentRuntime),
    Failed(String),
    Poisoned {
        error: String,
        runtime: Option<PersistentRuntime>,
    },
}

static RUNTIME: OnceLock<Mutex<GlobalRuntimeState>> = OnceLock::new();

fn global_runtime_state() -> &'static Mutex<GlobalRuntimeState> {
    RUNTIME.get_or_init(|| Mutex::new(GlobalRuntimeState::Uninitialized))
}

pub(crate) fn poison_global_persistent_runtime(error: &str) {
    let Ok(mut state) = global_runtime_state().lock() else {
        return;
    };
    let previous = std::mem::replace(&mut *state, GlobalRuntimeState::Uninitialized);
    *state = match previous {
        GlobalRuntimeState::Poisoned { error, runtime } => {
            GlobalRuntimeState::Poisoned { error, runtime }
        }
        GlobalRuntimeState::Ready(mut runtime) => {
            runtime.poisoned = Some(error.to_string());
            GlobalRuntimeState::Poisoned {
                error: error.to_string(),
                runtime: Some(runtime),
            }
        }
        GlobalRuntimeState::Failed(first_error) => GlobalRuntimeState::Poisoned {
            error: first_error,
            runtime: None,
        },
        GlobalRuntimeState::Uninitialized => GlobalRuntimeState::Poisoned {
            error: error.to_string(),
            runtime: None,
        },
    };
}

pub(crate) fn with_global_persistent_runtime<T>(
    operation: impl FnOnce(&mut PersistentRuntime) -> Result<T, String>,
) -> Result<T, String> {
    let mut state = global_runtime_state()
        .lock()
        .map_err(|_| "persistent IQ1_S runtime lock poisoned".to_string())?;
    if matches!(*state, GlobalRuntimeState::Uninitialized) {
        *state = match initialize_runtime_from_env() {
            Ok(runtime) => GlobalRuntimeState::Ready(runtime),
            Err(error) => GlobalRuntimeState::Failed(error),
        };
    }
    let result = match &mut *state {
        GlobalRuntimeState::Ready(runtime) => operation(runtime),
        GlobalRuntimeState::Failed(error) => Err(error.clone()),
        GlobalRuntimeState::Poisoned { error, .. } => {
            Err(format!("persistent IQ1_S runtime is poisoned: {error}"))
        }
        GlobalRuntimeState::Uninitialized => {
            Err("persistent IQ1_S runtime remained uninitialized".to_string())
        }
    };
    match result {
        Ok(value) => Ok(value),
        Err(error) => {
            let previous = std::mem::replace(&mut *state, GlobalRuntimeState::Uninitialized);
            *state = match previous {
                GlobalRuntimeState::Ready(mut runtime) => {
                    runtime.poisoned = Some(error.clone());
                    GlobalRuntimeState::Poisoned {
                        error: error.clone(),
                        runtime: Some(runtime),
                    }
                }
                GlobalRuntimeState::Poisoned {
                    error: first_error,
                    runtime,
                } => GlobalRuntimeState::Poisoned {
                    error: first_error,
                    runtime,
                },
                GlobalRuntimeState::Failed(first_error) => GlobalRuntimeState::Poisoned {
                    error: first_error,
                    runtime: None,
                },
                GlobalRuntimeState::Uninitialized => GlobalRuntimeState::Poisoned {
                    error: error.clone(),
                    runtime: None,
                },
            };
            Err(error)
        }
    }
}

impl PersistentRuntime {
    pub(crate) fn execute_phase(
        &mut self,
        snapshot: PhaseSnapshot,
    ) -> Result<PhaseOutcome, String> {
        if snapshot.key.session_generation != self.identity.session_generation
            || self.sources.len() != ARENA_EXPECTED_TENSORS
        {
            return Err("persistent IQ1_S runtime/session identity mismatch".to_string());
        }
        let trace_mode = std::env::var("HETGPU_IQ1S_TRACE_MODE")
            .map_err(|_| "HETGPU_IQ1S_TRACE_MODE is required".to_string())?;
        let prepared = prepare_phase(&self.arena, snapshot, &trace_mode)?;
        let completed = self
            .pool
            .submit_phase(&prepared.compiled, &prepared.buffers)
            .map_err(|error| error.to_string())?;
        let outputs = reconstruct_full_rows(&prepared, &completed)?;
        validate_all_finite(&outputs)?;
        for output in &outputs {
            unsafe { copy_host_to_cuda(output.cuda_ptr, &output.bytes) }
                .map_err(|error| error.to_string())?;
        }
        Ok(PhaseOutcome {
            prepared,
            completed,
            outputs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r#impl::iq1s_layer::{CapturedProjection, LayerKey, RouteAssignment};
    use crate::r#impl::iq1s_tmatmul::{
        capture_activation_from_host, GgmlType19Signature, LogicalLaunch, Q8_1_MMQ_BYTES,
    };
    use crate::r#impl::iq1s_weight_arena::{ArenaPlan, ArenaShard, ARENA_BANK_COUNT};
    use crate::r#impl::iq1s_weight_registry::{
        Iq1sExpertRole, Iq1sTensorIdentity, ResolvedIq1sWeight,
    };
    use std::path::PathBuf;
    use std::sync::Arc;

    fn identity(role: Iq1sExpertRole) -> Iq1sTensorIdentity {
        let (name, ne, nb) = match role {
            Iq1sExpertRole::Gate => (
                "gate",
                [4096, 1024, 512, 1],
                [50, 800, 819_200, 419_430_400],
            ),
            Iq1sExpertRole::Up => ("up", [4096, 1024, 512, 1], [50, 800, 819_200, 419_430_400]),
            Iq1sExpertRole::Down => (
                "down",
                [1024, 4096, 512, 1],
                [50, 200, 819_200, 419_430_400],
            ),
            Iq1sExpertRole::GateUp => unreachable!(),
        };
        Iq1sTensorIdentity {
            canonical_path: PathBuf::from("/tmp/qwen-runtime-test.gguf"),
            file_offset: 0,
            nbytes: 419_430_400,
            name: format!("blk.7.ffn_{name}_exps.weight"),
            layer: 7,
            ne,
            nb,
            role,
            model_sha256: [0x11; 32],
            content_sha256: [0x22 + role as u8; 32],
            device: 1,
            inode: 2,
            modified_ns: 3,
        }
    }

    fn arena(role: Iq1sExpertRole, experts: u16) -> Arc<ArenaPlan> {
        let identity = Arc::new(identity(role));
        let (row_count, row_bytes): (u32, u64) = match role {
            Iq1sExpertRole::Gate | Iq1sExpertRole::Up => (256, 800),
            Iq1sExpertRole::Down => (1024, 200),
            Iq1sExpertRole::GateUp => unreachable!(),
        };
        let shards = (0..experts)
            .flat_map(|expert| {
                let identity = identity.clone();
                (0..ARENA_BANK_COUNT).map(move |bank| ArenaShard {
                    tensor: identity.clone(),
                    expert,
                    bank: bank as u8,
                    row_start: bank as u32 * row_count,
                    row_count,
                    superblock: 0,
                    offset: u64::from(expert) * 1024 * 1024,
                    bytes: u64::from(row_count) * row_bytes,
                    sha256: [0x40 + expert as u8; 32],
                })
            })
            .collect();
        Arc::new(ArenaPlan {
            generation: 9,
            model_sha256: [0x11; 32],
            bank_bytes: [16 * 1024 * 1024; ARENA_BANK_COUNT],
            weight_bytes: [8 * 1024 * 1024; ARENA_BANK_COUNT],
            shards,
            hashes_verified: true,
        })
    }

    fn snapshot(batch: u16, distinct: bool) -> PhaseSnapshot {
        let role = Iq1sExpertRole::Gate;
        let tensor = identity(role);
        let routes = (0..u32::from(batch))
            .map(|token_id| RouteAssignment {
                token_id,
                expert_id: if distinct { (token_id % 3) as u16 } else { 0 },
                route_weight: 1.0,
            })
            .collect::<Vec<_>>();
        let signature = GgmlType19Signature {
            kernel: "mul_mat_q".to_string(),
            ne00: 4096,
            ne01: 1024,
            stride01: 16,
            ne10: 4096,
            ne11: 1,
            stride11: 1,
            ne0: 1024,
        };
        let lane = vec![0u8; 32 * Q8_1_MMQ_BYTES];
        let launches = routes
            .iter()
            .enumerate()
            .map(|(index, route)| {
                capture_activation_from_host(
                    LogicalLaunch {
                        matrix_ptr: 0x10_0000
                            + usize::from(route.expert_id) * tensor.nb[2] as usize,
                        activation_ptr: 0x20_0000 + index * 0x2000,
                        output_ptr: 0x30_0000 + index * 0x2000,
                        allocation_generation: 9,
                        content_hash: tensor.content_sha256,
                        signature: signature.clone(),
                    },
                    &lane,
                )
                .unwrap()
            })
            .collect();
        PhaseSnapshot {
            key: LayerKey {
                session_generation: 9,
                transaction_id: 71,
                layer_id: 7,
                stream: 0xabc0,
            },
            batch_count: batch,
            routes,
            projections: vec![CapturedProjection {
                role,
                weight: ResolvedIq1sWeight {
                    identity: tensor.clone(),
                    expert: 0,
                    allocation_generation: 9,
                    content_sha256: tensor.content_sha256,
                },
                launches,
            }],
            phase: LayerPhase::PhaseA,
        }
    }

    #[test]
    fn iq1s_persistent_runtime_packs_q8_k_major_lane_minor() {
        let lane0 = (0..32 * Q8_1_MMQ_BYTES)
            .map(|value| value as u8)
            .collect::<Vec<_>>();
        let lane1 = (0..32 * Q8_1_MMQ_BYTES)
            .map(|value| (value as u8).wrapping_add(73))
            .collect::<Vec<_>>();
        let packed = pack_q8_lanes(&[lane0.clone(), lane1.clone()], 32).unwrap();
        for record in 0..32 {
            assert_eq!(
                &packed[(record * 2) * 144..(record * 2 + 1) * 144],
                &lane0[record * 144..(record + 1) * 144]
            );
            assert_eq!(
                &packed[(record * 2 + 1) * 144..(record * 2 + 2) * 144],
                &lane1[record * 144..(record + 1) * 144]
            );
        }
    }

    #[test]
    fn iq1s_persistent_runtime_groups_experts_for_all_active_batches_and_modes() {
        let arena = arena(Iq1sExpertRole::Gate, 3);
        for batch in [1, 6, 9, 16] {
            for distinct in [false, true] {
                for mode in ["handwritten", "compiler"] {
                    let prepared = prepare_phase(&arena, snapshot(batch, distinct), mode).unwrap();
                    let groups = if distinct {
                        usize::from(batch.min(3))
                    } else {
                        1
                    };
                    assert_eq!(
                        prepared
                            .compiled
                            .commands
                            .iter()
                            .map(Vec::len)
                            .collect::<Vec<_>>(),
                        vec![groups; 4]
                    );
                    assert_eq!(prepared.buffers.activations.len(), groups);
                    assert_eq!(prepared.buffers.token_maps.len(), groups);
                    assert!(Arc::ptr_eq(&prepared.arena, &arena));
                    for commands in &prepared.compiled.commands {
                        assert!(commands
                            .windows(2)
                            .all(|pair| pair[0].expert_id < pair[1].expert_id));
                    }
                }
            }
        }
    }

    #[test]
    fn iq1s_persistent_runtime_identity_is_immutable_and_reuses_arena() {
        let arena = arena(Iq1sExpertRole::Gate, 3);
        let identity = PersistentRuntimeIdentity {
            model_sha256: [0x11; 32],
            xclbin_sha256: [0x33; 32],
            device_index: 0,
            session_generation: 9,
        };
        let mut binding = None;
        let handwritten =
            bind_runtime_identity(&mut binding, identity.clone(), arena.clone()).unwrap();
        let compiler =
            bind_runtime_identity(&mut binding, identity.clone(), arena.clone()).unwrap();
        assert!(Arc::ptr_eq(&handwritten, &compiler));
        for changed in [
            PersistentRuntimeIdentity {
                model_sha256: [0x12; 32],
                ..identity.clone()
            },
            PersistentRuntimeIdentity {
                xclbin_sha256: [0x34; 32],
                ..identity.clone()
            },
            PersistentRuntimeIdentity {
                session_generation: 10,
                ..identity.clone()
            },
        ] {
            assert!(bind_runtime_identity(&mut binding, changed, arena.clone()).is_err());
        }
    }

    #[test]
    fn iq1s_persistent_runtime_reconstructs_four_complete_row_shards() {
        let arena = arena(Iq1sExpertRole::Gate, 1);
        let prepared = prepare_phase(&arena, snapshot(1, false), "compiler").unwrap();
        let results = std::array::from_fn(|bank| {
            let command = prepared.compiled.commands[bank][0];
            let mut bytes = Vec::with_capacity(command.output_bytes as usize);
            for _ in 0..command.row_count {
                bytes.extend_from_slice(&(bank as f32 + 1.0).to_le_bytes());
            }
            vec![HostRange {
                offset: command.output_offset,
                bytes,
            }]
        });
        let completed = CompletedLayerPhase {
            transaction_id: prepared.compiled.transaction_id,
            semantic_sha256: prepared.compiled.semantic_sha256,
            completions: std::array::from_fn(|_| Vec::new()),
            results,
            expanded: Default::default(),
            dma: Default::default(),
            timings: Default::default(),
        };

        let outputs = reconstruct_full_rows(&prepared, &completed).unwrap();
        assert_eq!(outputs.len(), 1);
        let values = outputs[0]
            .bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values.len(), 1024);
        for bank in 0..4 {
            assert!(values[bank * 256..(bank + 1) * 256]
                .iter()
                .all(|value| *value == bank as f32 + 1.0));
        }
        validate_all_finite(&outputs).unwrap();
    }
}
