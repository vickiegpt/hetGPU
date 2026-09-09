use super::iq1s_layer::{
    enqueue_cuda_output_batch, CapturedProjection, LayerKey, PendingCudaOutputBatch,
    RouteAssignment,
};
use super::iq1s_layer_trace::{
    compile_layer_phase, ActivationRange, CompiledLayerPhase, LayerPhase, LayerPhasePlan,
    SemanticIq1sCommand,
};
use super::iq1s_persistent_proof::{
    checked_proof_path_from_env, hex_sha256, PersistentPhaseRecord, PersistentProofLedger,
    PhaseComparison, PhaseTimingsUs, CUDA_MMQ_ABSOLUTE_TOLERANCE, CUDA_MMQ_REFERENCE_BACKEND,
    CUDA_MMQ_RELATIVE_TOLERANCE,
};
use super::iq1s_tmatmul::{
    cuda_mmq_iq1s_reference_outputs, CapturedActivationLaunch, GgmlType19Signature, Q8_1_MMQ_BYTES,
};
use super::iq1s_trace::QWEN_MODEL_CONTEXT_LIMIT;
use super::iq1s_weight_arena::{
    persistent_chunk_specs, plan_registered_arena, read_persistent_chunk, ArenaPlan, ArenaShard,
    ARENA_ALIGNMENT, ARENA_BANK_COUNT, ARENA_EXPECTED_RAW_BYTES, ARENA_EXPECTED_TENSORS,
};
use super::iq1s_weight_registry::{global_registry, Iq1sExpertRole, Iq1sTensorSource};
use super::xrt_iq1s_persistent::{
    CompletedLayerPhase, HostRange, PersistentIq1sConfig, PersistentIq1sPool, PhaseBuffers,
    SubmissionTicket, TicketPoll,
};
use super::xrt_tmatmul::RealXrt;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant, UNIX_EPOCH};

const QWEN_MODEL_SHA256: [u8; 32] = [
    0x0a, 0x32, 0xc2, 0x70, 0x2f, 0xbb, 0x61, 0x93, 0x49, 0x60, 0xcf, 0xee, 0xf3, 0x45, 0x24, 0xb8,
    0x1e, 0xc6, 0xd9, 0x26, 0x71, 0x58, 0xf2, 0x46, 0xd4, 0x5f, 0xc8, 0x6f, 0x5a, 0xaa, 0x75, 0x68,
];
const PERSISTENT_XCLBIN_SHA256: [u8; 32] = [
    0x9c, 0x83, 0xdc, 0xae, 0x07, 0xb4, 0xc7, 0xbf, 0x1d, 0x2e, 0x1c, 0xeb, 0xf4, 0x6c, 0xcf, 0x0f,
    0xf1, 0xeb, 0xf8, 0x84, 0x8a, 0x43, 0x70, 0x35, 0xfe, 0xf1, 0x45, 0x1d, 0xee, 0x77, 0x70, 0xa3,
];
const PERSISTENT_XCLBIN_NAME: &str = "qwen397b_iq1s_layer_persistent_9c83dcae.xclbin";
const GRIDROM_PERSISTENT_XCLBIN_SHA256: [u8; 32] = [
    0xd7, 0x2f, 0xf7, 0x33, 0x6c, 0xb4, 0xdd, 0x49, 0x86, 0x7c, 0x5f, 0x22, 0x08, 0xa3, 0x43, 0x92,
    0x81, 0xea, 0xab, 0x1b, 0x0a, 0x9e, 0x59, 0x4b, 0x45, 0x8a, 0x4f, 0xec, 0xc6, 0xb7, 0x99, 0x48,
];
const GRIDROM_PERSISTENT_XCLBIN_NAME: &str =
    "qwen397b_iq1s_layer_persistent_d72ff7336cb4dd49.xclbin";

fn qualified_persistent_xclbin_identity(name: &str, sha256: [u8; 32]) -> bool {
    matches!(
        (name, sha256),
        (PERSISTENT_XCLBIN_NAME, PERSISTENT_XCLBIN_SHA256)
            | (
                GRIDROM_PERSISTENT_XCLBIN_NAME,
                GRIDROM_PERSISTENT_XCLBIN_SHA256
            )
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PersistentRuntimeIdentity {
    pub(crate) model_sha256: [u8; 32],
    pub(crate) xclbin_sha256: [u8; 32],
    pub(crate) device_index: u32,
    pub(crate) session_generation: u64,
}

#[derive(Debug, Serialize)]
struct ProgressRecord<'a> {
    schema_version: u32,
    stage: &'a str,
    generation: u64,
    model_sha256: String,
    xclbin_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    bank: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logical_offset: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    transaction: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    layer: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    slot_generation: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,
}

struct ProgressSink {
    file: File,
    generation: u64,
    model_sha256: String,
    xclbin_sha256: String,
}

impl ProgressSink {
    fn create(ledger_path: &Path, identity: &PersistentRuntimeIdentity) -> Result<Self, String> {
        let path = PathBuf::from(
            std::env::var("HETGPU_QWEN_IQ1S_PROGRESS_LOG")
                .map_err(|_| "HETGPU_QWEN_IQ1S_PROGRESS_LOG is required".to_string())?,
        );
        Self::create_at(&path, ledger_path, identity)
    }

    fn create_at(
        path: &Path,
        ledger_path: &Path,
        identity: &PersistentRuntimeIdentity,
    ) -> Result<Self, String> {
        if path.parent() != ledger_path.parent() {
            return Err("persistent progress log must be beside the proof ledger".to_string());
        }
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| format!("create persistent progress log: {error}"))?;
        Ok(Self {
            file,
            generation: identity.session_generation,
            model_sha256: hex_sha256(&identity.model_sha256),
            xclbin_sha256: hex_sha256(&identity.xclbin_sha256),
        })
    }

    fn append(
        &mut self,
        stage: &str,
        bank: Option<u8>,
        logical_offset: Option<u64>,
        bytes: Option<usize>,
        transaction: Option<u64>,
        layer: Option<u32>,
        phase: Option<&str>,
        slot_generation: Option<u64>,
        error: Option<&str>,
    ) -> Result<(), String> {
        let record = ProgressRecord {
            schema_version: 1,
            stage,
            generation: self.generation,
            model_sha256: self.model_sha256.clone(),
            xclbin_sha256: self.xclbin_sha256.clone(),
            bank,
            logical_offset,
            bytes,
            transaction,
            layer,
            phase,
            slot_generation,
            error,
        };
        serde_json::to_writer(&mut self.file, &record)
            .map_err(|error| format!("serialize persistent progress record: {error}"))?;
        self.file
            .write_all(b"\n")
            .and_then(|_| self.file.flush())
            .map_err(|error| format!("flush persistent progress record: {error}"))
    }
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
    lane_count: usize,
    input_offset: u64,
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

struct ActivePhase {
    phase_wall_start: Instant,
    layer_id: u32,
    phase_name: &'static str,
    snapshot_stream: usize,
    prepare_us: u64,
    trace_mode: String,
    prepared: PreparedPhase,
    ticket: SubmissionTicket,
}

pub(crate) struct PersistentRuntime {
    identity: PersistentRuntimeIdentity,
    arena: Arc<ArenaPlan>,
    sources: Vec<Arc<Iq1sTensorSource>>,
    pool: PersistentIq1sPool<RealXrt>,
    ledger: PersistentProofLedger,
    progress: ProgressSink,
    failure_capture_dir: PathBuf,
    sampled_comparison_complete: bool,
    output_publisher: NativeResultPublisher,
    poisoned: Option<String>,
}

trait ResultPublisher {
    fn publish_batch(&mut self, stream: usize, outputs: &[PublishedOutput]) -> Result<(), String>;
}

#[derive(Default)]
struct NativeResultPublisher {
    pending: VecDeque<PendingCudaOutputBatch>,
}

impl ResultPublisher for NativeResultPublisher {
    fn publish_batch(&mut self, stream: usize, outputs: &[PublishedOutput]) -> Result<(), String> {
        // Two retained staging buffers match the two XRT ticket slots. Reap
        // the oldest only when its pinned storage would otherwise be reused.
        while self.pending.len() >= 2 {
            self.pending.pop_front().expect("length checked").finish()?;
        }
        let copies = outputs
            .iter()
            .map(|output| (output.cuda_ptr, output.bytes.as_slice()))
            .collect::<Vec<_>>();
        let pending = unsafe { enqueue_cuda_output_batch(stream, &copies) }?;
        self.pending.push_back(pending);
        Ok(())
    }
}

fn publish_outputs_with(
    publisher: &mut impl ResultPublisher,
    stream: usize,
    outputs: &[PublishedOutput],
) -> Result<(), String> {
    publisher.publish_batch(stream, outputs)
}

// XRT opaque handles and the dlopen handle are process-wide C handles. The
// runtime never exposes them, and each short ticket operation is serialized by
// the phase-pipeline mutex. Pending callers release that mutex before sleeping,
// so two hardware tickets can remain in flight without concurrent XRT access.
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
    if lanes.is_empty() || lanes.len() > 32 || records == 0 {
        return Err("persistent IQ1_S Q8 pack requires 1..=32 lanes and positive K records".into());
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

fn packed_activation_identity(
    key: &LayerKey,
    lanes: &[(RouteAssignment, CapturedActivationLaunch)],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"hetgpu-qwen-iq1s-packed-activation-v1\0");
    hash.update(key.session_generation.to_le_bytes());
    hash.update(key.transaction_id.to_le_bytes());
    hash.update(key.layer_id.to_le_bytes());
    hash.update((key.stream as u64).to_le_bytes());
    hash.update((lanes.len() as u64).to_le_bytes());
    for (route, launch) in lanes {
        hash.update(route.token_id.to_le_bytes());
        hash.update((launch.launch.activation_ptr as u64).to_le_bytes());
        hash.update((launch.packed_activations().len() as u64).to_le_bytes());
        hash.update(Sha256::digest(launch.packed_activations()));
    }
    hash.finalize().into()
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
        || snapshot.batch_count > 32
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
                || launch.launch.allocation_generation != projection.weight.allocation_generation
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
    let mut activation_manifest: Vec<ActivationRange> = Vec::new();
    let mut activation_buffers: Vec<HostRange> = Vec::new();
    let mut activation_cache = BTreeMap::<[u8; 32], usize>::new();
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
        // The ABI lane mask is 16 bits. A 32-active server batch therefore
        // remains one layer transaction but is encoded as at most two
        // commands per expert, with global token IDs carried by token_map.
        for lanes in lanes.chunks(16) {
            let lane_bytes = lanes
                .iter()
                .map(|(_, launch)| launch.packed_activations().to_vec())
                .collect::<Vec<_>>();
            let packed = pack_q8_lanes(&lane_bytes, k_records)?;
            let source_identity_sha256 = packed_activation_identity(&snapshot.key, lanes);
            let cached_activation = activation_cache.get(&source_identity_sha256).copied();
            let (activation_offset, activation_is_new) = if let Some(buffer_index) =
                cached_activation
            {
                let cached = activation_buffers
                    .get(buffer_index)
                    .ok_or("persistent IQ1_S activation cache index is invalid")?;
                let cached_manifest = activation_manifest
                    .get(buffer_index)
                    .ok_or("persistent IQ1_S activation manifest cache index is invalid")?;
                if cached.bytes != packed
                    || cached_manifest.cuda_ptr != lanes[0].1.launch.activation_ptr
                {
                    return Err("persistent IQ1_S packed activation identity collision".to_string());
                }
                (cached.offset, false)
            } else {
                activation_cursor = align_up(activation_cursor, ARENA_ALIGNMENT)?;
                (activation_cursor, true)
            };
            token_cursor = align_up(token_cursor, ARENA_ALIGNMENT)?;
            output_cursor = align_up(output_cursor, ARENA_ALIGNMENT)?;
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
            let token_ids = (0..lanes.len() as u32).collect::<Vec<_>>();
            let lane_mask = if lanes.len() == 16 {
                u16::MAX
            } else {
                (1u16 << lanes.len()) - 1
            };
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
                    lane_count: lanes.len(),
                    input_offset: activation_offset,
                    output_offset,
                    shards: shards.clone(),
                });
            }
            if activation_is_new {
                activation_manifest.push(ActivationRange {
                    cuda_ptr: lanes[0].1.launch.activation_ptr,
                    slab_offset: activation_offset,
                    bytes: u32::try_from(packed.len())
                        .map_err(|_| "persistent IQ1_S activation bytes do not fit u32")?,
                    stream: snapshot.key.stream,
                    source_identity_sha256,
                });
                activation_buffers.push(HostRange {
                    offset: activation_offset,
                    bytes: packed,
                });
                activation_cache.insert(source_identity_sha256, activation_buffers.len() - 1);
                activation_cursor = activation_offset
                    .checked_add(activation_buffers.last().unwrap().bytes.len() as u64)
                    .ok_or("persistent IQ1_S activation slab overflow")?;
            }
            token_maps.push(HostRange {
                offset: token_offset,
                bytes: token_bytes,
            });
            token_cursor = token_offset
                .checked_add(token_maps.last().unwrap().bytes.len() as u64)
                .ok_or("persistent IQ1_S token-map slab overflow")?;
            output_cursor = output_offset
                .checked_add(max_result_bytes)
                .ok_or("persistent IQ1_S result slab overflow")?;
        }
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

fn output_values(output: &PublishedOutput) -> Result<Vec<f32>, String> {
    if output.bytes.len() % 4 != 0 {
        return Err("persistent IQ1_S output contains a partial f32".to_string());
    }
    Ok(output
        .bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
        .collect())
}

fn read_sample_matrix(binding: &OutputBinding) -> Result<Vec<u8>, String> {
    let mut shards = binding.shards.iter().collect::<Vec<_>>();
    shards.sort_by_key(|shard| shard.row_start);
    let first = shards
        .first()
        .ok_or("persistent IQ1_S sampled binding has no matrix shards")?;
    let identity = &first.tensor;
    if shards.iter().any(|shard| {
        shard.expert != binding.expert_id
            || shard.tensor.as_ref() != identity.as_ref()
            || shard.sha256 == [0; 32]
    }) {
        return Err("persistent IQ1_S sampled matrix shard identity mismatch".into());
    }
    let file = File::open(&identity.canonical_path)
        .map_err(|error| format!("open sampled IQ1_S source: {error}"))?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("inspect sampled IQ1_S source: {error}"))?;
    let modified_ns = metadata
        .modified()
        .map_err(|error| format!("sampled IQ1_S source modification time: {error}"))?
        .duration_since(UNIX_EPOCH)
        .map_err(|_| "sampled IQ1_S source modification time predates epoch")?
        .as_nanos();
    if metadata.dev() != identity.device
        || metadata.ino() != identity.inode
        || modified_ns != identity.modified_ns
    {
        return Err("sampled IQ1_S source identity changed after registration".into());
    }
    let tensor_end = identity
        .file_offset
        .checked_add(identity.nbytes)
        .ok_or("sampled IQ1_S tensor source range overflow")?;
    if tensor_end > metadata.len() {
        return Err("sampled IQ1_S tensor source exceeds its registered file".into());
    }
    let row_bytes = identity.ne[0]
        .checked_div(256)
        .and_then(|blocks| blocks.checked_mul(50))
        .ok_or("sampled IQ1_S row byte count overflow")?;
    if identity.nb[1] != row_bytes {
        return Err("sampled IQ1_S reference requires tightly packed rows".into());
    }
    let expected_rows =
        u32::try_from(identity.ne[1]).map_err(|_| "sampled IQ1_S row count does not fit u32")?;
    let mut next_row = 0u32;
    let mut matrix = Vec::new();
    for shard in shards {
        if shard.row_start != next_row {
            return Err("sampled IQ1_S matrix shards have a gap or overlap".into());
        }
        let byte_count = u64::from(shard.row_count)
            .checked_mul(row_bytes)
            .ok_or("sampled IQ1_S shard byte count overflow")?;
        if byte_count != shard.bytes {
            return Err("sampled IQ1_S shard byte count mismatch".into());
        }
        let length = usize::try_from(byte_count)
            .map_err(|_| "sampled IQ1_S shard does not fit host memory")?;
        let mut bytes = vec![0u8; length];
        let source_offset = identity
            .file_offset
            .checked_add(u64::from(binding.expert_id) * identity.nb[2])
            .and_then(|offset| offset.checked_add(u64::from(shard.row_start) * row_bytes))
            .ok_or("sampled IQ1_S source offset overflow")?;
        file.read_exact_at(&mut bytes, source_offset)
            .map_err(|error| format!("read sampled IQ1_S shard: {error}"))?;
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        if digest != shard.sha256 {
            return Err("sampled IQ1_S shard SHA-256 changed after arena load".into());
        }
        matrix.extend_from_slice(&bytes);
        next_row = next_row
            .checked_add(shard.row_count)
            .ok_or("sampled IQ1_S row coverage overflow")?;
    }
    if next_row != expected_rows {
        return Err("sampled IQ1_S matrix shards do not cover every row".into());
    }
    Ok(matrix)
}

fn sample_activation(prepared: &PreparedPhase, binding: &OutputBinding) -> Result<Vec<u8>, String> {
    if binding.lane_count == 0 || binding.lane_index >= binding.lane_count {
        return Err("persistent IQ1_S sampled activation lane is invalid".into());
    }
    let source = prepared
        .buffers
        .activations
        .iter()
        .find(|range| range.offset == binding.input_offset)
        .ok_or("persistent IQ1_S sampled activation range is absent")?;
    let records = usize::try_from(binding.shards[0].tensor.ne[0] / 128)
        .map_err(|_| "persistent IQ1_S activation record count does not fit usize")?;
    let expected = records
        .checked_mul(binding.lane_count)
        .and_then(|count| count.checked_mul(Q8_1_MMQ_BYTES))
        .ok_or("persistent IQ1_S sampled activation extent overflow")?;
    if source.bytes.len() != expected {
        return Err("persistent IQ1_S sampled activation range has the wrong extent".into());
    }
    let mut activation = Vec::with_capacity(records * Q8_1_MMQ_BYTES);
    for record in 0..records {
        let offset = (record * binding.lane_count + binding.lane_index) * Q8_1_MMQ_BYTES;
        activation.extend_from_slice(&source.bytes[offset..offset + Q8_1_MMQ_BYTES]);
    }
    Ok(activation)
}

#[derive(Debug, Clone, Copy)]
struct NumericalMismatch {
    index: usize,
    actual: f32,
    reference: f32,
    absolute_error: f32,
    limit: f32,
}

#[derive(Debug, Serialize)]
struct FailureCaptureFile {
    name: String,
    bytes: usize,
    sha256: String,
}

fn write_capture_file(root: &Path, name: &str, bytes: &[u8]) -> Result<FailureCaptureFile, String> {
    let path = root.join(name);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|error| {
            format!(
                "create numerical failure capture {}: {error}",
                path.display()
            )
        })?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| {
            format!(
                "write numerical failure capture {}: {error}",
                path.display()
            )
        })?;
    let digest: [u8; 32] = Sha256::digest(bytes).into();
    Ok(FailureCaptureFile {
        name: name.to_string(),
        bytes: bytes.len(),
        sha256: hex_sha256(&digest),
    })
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn command_json(command: &super::iq1s_layer_abi::Iq1sCommand) -> serde_json::Value {
    serde_json::json!({
        "magic": command.magic,
        "abi_version": command.abi_version,
        "descriptor_bytes": command.descriptor_bytes,
        "crc32": command.crc32,
        "flags": command.flags,
        "session_generation": command.session_generation,
        "transaction_id": command.transaction_id,
        "program_id": command.program_id,
        "trace_id": command.trace_id,
        "layer_id": command.layer_id,
        "phase": command.phase,
        "role": command.role,
        "expert_id": command.expert_id,
        "lane_mask": command.lane_mask,
        "lane_count": command.lane_count,
        "weight_format": command.weight_format,
        "arena_offset": command.arena_offset,
        "input_offset": command.input_offset,
        "output_offset": command.output_offset,
        "row_start": command.row_start,
        "row_count": command.row_count,
        "input_bytes": command.input_bytes,
        "output_bytes": command.output_bytes,
        "token_map_offset": command.token_map_offset,
        "dependency_fence": command.dependency_fence,
        "completion_slot": command.completion_slot,
        "reserved": command.reserved,
    })
}

#[allow(clippy::too_many_arguments)]
fn write_numerical_failure_capture_at(
    root: &Path,
    identity: &PersistentRuntimeIdentity,
    trace_mode: &str,
    prepared: &PreparedPhase,
    binding: &OutputBinding,
    matrix: &[u8],
    activation: &[u8],
    reference: &[f32],
    actual: &[f32],
    mismatch: NumericalMismatch,
) -> Result<(), String> {
    std::fs::create_dir(root).map_err(|error| {
        format!(
            "create numerical failure capture directory {}: {error}",
            root.display()
        )
    })?;
    let mut files = Vec::new();
    files.push(write_capture_file(root, "matrix.iq1s.bin", matrix)?);
    files.push(write_capture_file(root, "activation.q8_1.bin", activation)?);
    files.push(write_capture_file(
        root,
        "reference.f32.bin",
        &f32_bytes(reference),
    )?);
    files.push(write_capture_file(
        root,
        "actual.f32.bin",
        &f32_bytes(actual),
    )?);
    for bank in 0..ARENA_BANK_COUNT {
        let commands = prepared.compiled.commands[bank]
            .iter()
            .map(command_json)
            .collect::<Vec<_>>();
        let command_bytes = serde_json::to_vec_pretty(&commands)
            .map_err(|error| format!("serialize bank {bank} failure commands: {error}"))?;
        files.push(write_capture_file(
            root,
            &format!("bank-{bank}.commands.json"),
            &command_bytes,
        )?);
        files.push(write_capture_file(
            root,
            &format!("bank-{bank}.program.bin"),
            &prepared.compiled.programs[bank].encoded,
        )?);
        files.push(write_capture_file(
            root,
            &format!("bank-{bank}.program.asm"),
            prepared.compiled.programs[bank].assembly.as_bytes(),
        )?);
    }
    let tensor = &binding.shards[0].tensor;
    let command_counts_per_bank: [usize; ARENA_BANK_COUNT] =
        std::array::from_fn(|bank| prepared.compiled.commands[bank].len());
    let manifest = serde_json::json!({
        "schema_version": 1,
        "status": "numerical_mismatch_nonproof",
        "trace_mode": trace_mode,
        "model_sha256": hex_sha256(&identity.model_sha256),
        "xclbin_sha256": hex_sha256(&identity.xclbin_sha256),
        "device_index": identity.device_index,
        "session_generation": identity.session_generation,
        "transaction_id": prepared.compiled.transaction_id,
        "layer_id": tensor.layer,
        "phase": match prepared.compiled.phase {
            LayerPhase::PhaseA => "A",
            LayerPhase::PhaseB => "B",
        },
        "semantic_sha256": hex_sha256(&prepared.compiled.semantic_sha256),
        "mismatch": {
            "index": mismatch.index,
            "actual": mismatch.actual,
            "reference": mismatch.reference,
            "absolute_error": mismatch.absolute_error,
            "limit": mismatch.limit,
        },
        "binding": {
            "tensor_name": tensor.name,
            "tensor_content_sha256": hex_sha256(&tensor.content_sha256),
            "expert_id": binding.expert_id,
            "token_id": binding.token_id,
            "role": format!("{:?}", binding.role).to_ascii_lowercase(),
            "lane_index": binding.lane_index,
            "lane_count": binding.lane_count,
            "row_count": binding.row_count,
            "input_offset": binding.input_offset,
            "output_offset": binding.output_offset,
            "ne": tensor.ne,
            "nb": tensor.nb,
            "shards": binding.shards.iter().map(|shard| serde_json::json!({
                "bank": shard.bank,
                "row_start": shard.row_start,
                "row_count": shard.row_count,
                "arena_offset": shard.offset,
                "bytes": shard.bytes,
                "sha256": hex_sha256(&shard.sha256),
            })).collect::<Vec<_>>(),
        },
        "command_counts_per_bank": command_counts_per_bank,
        "files": files,
    });
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("serialize numerical failure capture manifest: {error}"))?;
    write_capture_file(root, "manifest.json", &manifest_bytes)?;
    Ok(())
}

fn compare_sampled_output(
    prepared: &PreparedPhase,
    outputs: &[PublishedOutput],
    identity: &PersistentRuntimeIdentity,
    trace_mode: &str,
    failure_capture_dir: &Path,
) -> Result<PhaseComparison, String> {
    let binding = prepared
        .output_bindings
        .first()
        .ok_or("persistent IQ1_S phase has no sampled output binding")?;
    let actual = outputs
        .first()
        .ok_or("persistent IQ1_S phase has no sampled output")?;
    if actual.cuda_ptr != binding.cuda_ptr {
        return Err("persistent IQ1_S sampled output binding order changed".into());
    }
    let matrix = read_sample_matrix(binding)?;
    let activation = sample_activation(prepared, binding)?;
    let tensor = &binding.shards[0].tensor;
    let signature = GgmlType19Signature {
        kernel: "mul_mat_q".to_string(),
        ne00: tensor.ne[0],
        ne01: tensor.ne[1],
        stride01: tensor.nb[1] / 50,
        ne10: tensor.ne[0],
        ne11: 1,
        stride11: 1,
        ne0: tensor.ne[1],
    };
    let reference = cuda_mmq_iq1s_reference_outputs(&signature, &matrix, &activation, None)?;
    let actual = output_values(actual)?;
    if reference.len() != actual.len() {
        return Err("persistent IQ1_S sampled reference/output length mismatch".into());
    }
    let mut max_abs_error = 0.0_f32;
    let mut max_rel_error = 0.0_f32;
    let mut max_tolerance_ratio = 0.0_f32;
    for (index, (&reference_value, &actual_value)) in reference.iter().zip(&actual).enumerate() {
        if !reference_value.is_finite() || !actual_value.is_finite() {
            return Err(format!(
                "persistent IQ1_S sampled output {index} is nonfinite"
            ));
        }
        let abs = (actual_value - reference_value).abs();
        let rel = if reference_value == 0.0 {
            abs
        } else {
            abs / reference_value.abs()
        };
        let limit =
            CUDA_MMQ_ABSOLUTE_TOLERANCE + CUDA_MMQ_RELATIVE_TOLERANCE * reference_value.abs();
        let tolerance_ratio = abs / limit;
        max_abs_error = max_abs_error.max(abs);
        max_rel_error = max_rel_error.max(rel);
        max_tolerance_ratio = max_tolerance_ratio.max(tolerance_ratio);
        if abs > limit {
            let mismatch = NumericalMismatch {
                index,
                actual: actual_value,
                reference: reference_value,
                absolute_error: abs,
                limit,
            };
            let original = format!(
                "persistent IQ1_S CUDA-MMQ sample {index} is outside tolerance: actual={actual_value}, reference={reference_value}, absolute_error={abs}, limit={limit}"
            );
            write_numerical_failure_capture_at(
                failure_capture_dir,
                identity,
                trace_mode,
                prepared,
                binding,
                &matrix,
                &activation,
                &reference,
                &actual,
                mismatch,
            )
            .map_err(|capture_error| {
                format!("{original}; failure capture failed: {capture_error}")
            })?;
            return Err(format!(
                "{original}; failure_capture={}",
                failure_capture_dir.display()
            ));
        }
    }
    PhaseComparison::sampled_pass(
        CUDA_MMQ_REFERENCE_BACKEND,
        actual.len(),
        max_abs_error,
        max_rel_error,
        max_tolerance_ratio,
    )
}

fn finite_only_comparison(outputs: &[PublishedOutput]) -> Result<PhaseComparison, String> {
    validate_all_finite(outputs)?;
    let elements = outputs.iter().try_fold(0usize, |total, output| {
        total
            .checked_add(output.bytes.len() / 4)
            .ok_or("persistent IQ1_S finite element count overflow")
    })?;
    PhaseComparison::finite_only(elements)
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "on" | "ON"))
        .unwrap_or(false)
}

fn initial_sampled_comparison_complete(diagnostic_finite_only: bool) -> bool {
    diagnostic_finite_only
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
    let xclbin_sha256 = sha256_file(&xclbin)?;
    if !qualified_persistent_xclbin_identity(
        xclbin
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(""),
        xclbin_sha256,
    ) {
        return Err("persistent IQ1_S xclbin name or SHA-256 is not qualified".to_string());
    }
    let device_index = u32::try_from(env_u64("HETGPU_XRT_DEVICE_INDEX", 0)?)
        .map_err(|_| "HETGPU_XRT_DEVICE_INDEX does not fit u32")?;
    let ring_capacity = u32::try_from(env_u64("HETGPU_QWEN_IQ1S_RING_CAPACITY", 512)?)
        .map_err(|_| "HETGPU_QWEN_IQ1S_RING_CAPACITY does not fit u32")?;
    let timeout_ms = u32::try_from(env_u64("HETGPU_XRT_TIMEOUT_MS", 10_000)?)
        .map_err(|_| "HETGPU_XRT_TIMEOUT_MS does not fit u32")?;
    let identity = PersistentRuntimeIdentity {
        model_sha256: QWEN_MODEL_SHA256,
        xclbin_sha256,
        device_index,
        session_generation: generation,
    };
    let ledger_path = checked_proof_path_from_env()?;
    let failure_capture_dir = ledger_path
        .parent()
        .ok_or("persistent IQ1_S proof ledger has no parent")?
        .join("failure-capture");
    let mut progress = ProgressSink::create(&ledger_path, &identity)?;
    let diagnostic_finite_only = env_truthy("HETGPU_QWEN_IQ1S_DIAGNOSTIC_FINITE_ONLY");
    if diagnostic_finite_only {
        progress.append(
            "diagnostic_finite_only_enabled",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;
    }
    progress.append(
        "arena_plan",
        None,
        None,
        Some(
            usize::try_from(ARENA_EXPECTED_RAW_BYTES)
                .map_err(|_| "persistent IQ1_S arena byte count does not fit usize".to_string())?,
        ),
        None,
        None,
        None,
        None,
        None,
    )?;
    let ledger = PersistentProofLedger::create(&ledger_path)?;
    let chunks = persistent_chunk_specs(&arena)?;
    let ops = RealXrt::load(true).map_err(|error| error.to_string())?;
    let config =
        PersistentIq1sConfig::checked(xclbin, device_index, Some(ring_capacity), timeout_ms)
            .map_err(|error| error.to_string())?;
    let pool = PersistentIq1sPool::open(
        ops,
        config,
        generation,
        &chunks,
        |chunk| read_persistent_chunk(&arena, &sources, chunk),
        |chunk| {
            progress.append(
                "arena_chunk_resident",
                Some(chunk.bank),
                Some(chunk.logical_offset),
                Some(chunk.bytes),
                None,
                None,
                None,
                None,
                None,
            )
        },
    )
    .map_err(|error| error.to_string())?;
    progress.append("pool_ready", None, None, None, None, None, None, None, None)?;
    Ok(PersistentRuntime {
        identity,
        arena,
        sources,
        pool,
        ledger,
        progress,
        failure_capture_dir,
        // This opt-in exists only to obtain diagnostic hardware timing from an
        // otherwise finite run. The strict proof validator still requires a
        // sampled libggml comparison and therefore rejects this mode.
        sampled_comparison_complete: initial_sampled_comparison_complete(diagnostic_finite_only),
        output_publisher: NativeResultPublisher::default(),
        poisoned: None,
    })
}

trait PhasePipelineEngine: Send {
    type Input;
    type Active;
    type Output;

    fn has_capacity(&self) -> bool;
    fn start(&mut self, input: Self::Input) -> Result<Self::Active, String>;
    fn poll(&mut self, active: &Self::Active) -> Result<TicketPoll, String>;
    fn finish(&mut self, active: Self::Active) -> Result<Self::Output, String>;
    fn poison(&mut self, error: &str);
}

enum PhasePipelineState<E> {
    Uninitialized,
    Ready(E),
    Failed(String),
    Poisoned { error: String, engine: Option<E> },
}

struct PhasePipeline<E: PhasePipelineEngine> {
    state: Mutex<PhasePipelineState<E>>,
    capacity: Condvar,
    initialize: Option<fn() -> Result<E, String>>,
}

impl<E: PhasePipelineEngine> PhasePipeline<E> {
    fn with_engine(engine: E) -> Self {
        Self {
            state: Mutex::new(PhasePipelineState::Ready(engine)),
            capacity: Condvar::new(),
            initialize: None,
        }
    }

    fn lazy(initialize: fn() -> Result<E, String>) -> Self {
        Self {
            state: Mutex::new(PhasePipelineState::Uninitialized),
            capacity: Condvar::new(),
            initialize: Some(initialize),
        }
    }

    fn poison_locked(state: &mut PhasePipelineState<E>, error: String) {
        let previous = std::mem::replace(state, PhasePipelineState::Uninitialized);
        *state = match previous {
            PhasePipelineState::Ready(mut engine) => {
                engine.poison(&error);
                PhasePipelineState::Poisoned {
                    error,
                    engine: Some(engine),
                }
            }
            PhasePipelineState::Poisoned {
                error: first_error,
                engine,
            } => PhasePipelineState::Poisoned {
                error: first_error,
                engine,
            },
            PhasePipelineState::Failed(first_error) => PhasePipelineState::Poisoned {
                error: first_error,
                engine: None,
            },
            PhasePipelineState::Uninitialized => PhasePipelineState::Poisoned {
                error,
                engine: None,
            },
        };
    }

    fn initialize_locked(&self, state: &mut PhasePipelineState<E>) {
        if !matches!(state, PhasePipelineState::Uninitialized) {
            return;
        }
        *state = match self.initialize {
            Some(initialize) => match initialize() {
                Ok(engine) => PhasePipelineState::Ready(engine),
                Err(error) => PhasePipelineState::Failed(error),
            },
            None => PhasePipelineState::Failed(
                "persistent IQ1_S phase pipeline has no initializer".to_string(),
            ),
        };
    }

    fn execute(&self, input: E::Input) -> Result<E::Output, String> {
        let mut input = Some(input);
        let active = loop {
            let mut state = self
                .state
                .lock()
                .map_err(|_| "persistent IQ1_S pipeline lock poisoned".to_string())?;
            self.initialize_locked(&mut state);
            match &mut *state {
                PhasePipelineState::Ready(engine) if engine.has_capacity() => {
                    let input = input.take().expect("pipeline input consumed once");
                    match engine.start(input) {
                        Ok(active) => break active,
                        Err(error) => {
                            Self::poison_locked(&mut state, error.clone());
                            self.capacity.notify_all();
                            return Err(error);
                        }
                    }
                }
                PhasePipelineState::Ready(_) => {
                    state = self
                        .capacity
                        .wait(state)
                        .map_err(|_| "persistent IQ1_S pipeline lock poisoned".to_string())?;
                    drop(state);
                }
                PhasePipelineState::Failed(error) => return Err(error.clone()),
                PhasePipelineState::Poisoned { error, .. } => {
                    return Err(format!("persistent IQ1_S runtime is poisoned: {error}"));
                }
                PhasePipelineState::Uninitialized => {
                    return Err("persistent IQ1_S runtime remained uninitialized".to_string());
                }
            }
        };

        let mut backoff_us = 1u64;
        loop {
            let mut state = self
                .state
                .lock()
                .map_err(|_| "persistent IQ1_S pipeline lock poisoned".to_string())?;
            match &mut *state {
                PhasePipelineState::Ready(engine) => match engine.poll(&active) {
                    Ok(TicketPoll::Pending) => {
                        drop(state);
                        std::thread::sleep(Duration::from_micros(backoff_us));
                        backoff_us = (backoff_us * 2).min(1_000);
                    }
                    Ok(TicketPoll::Complete) => match engine.finish(active) {
                        Ok(output) => {
                            self.capacity.notify_one();
                            return Ok(output);
                        }
                        Err(error) => {
                            Self::poison_locked(&mut state, error.clone());
                            self.capacity.notify_all();
                            return Err(error);
                        }
                    },
                    Err(error) => {
                        Self::poison_locked(&mut state, error.clone());
                        self.capacity.notify_all();
                        return Err(error);
                    }
                },
                PhasePipelineState::Failed(error) => return Err(error.clone()),
                PhasePipelineState::Poisoned { error, .. } => {
                    return Err(format!("persistent IQ1_S runtime is poisoned: {error}"));
                }
                PhasePipelineState::Uninitialized => {
                    return Err("persistent IQ1_S runtime remained uninitialized".to_string());
                }
            }
        }
    }

    fn poison(&self, error: &str) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        Self::poison_locked(&mut state, error.to_string());
        self.capacity.notify_all();
    }
}

fn global_phase_pipeline() -> &'static PhasePipeline<PersistentRuntime> {
    static PIPELINE: OnceLock<PhasePipeline<PersistentRuntime>> = OnceLock::new();
    PIPELINE.get_or_init(|| PhasePipeline::lazy(initialize_runtime_from_env))
}

pub(crate) fn execute_global_persistent_phase(snapshot: PhaseSnapshot) -> Result<(), String> {
    global_phase_pipeline().execute(snapshot).map(|_| ())
}

pub(crate) fn poison_global_persistent_runtime(error: &str) {
    global_phase_pipeline().poison(error);
}

impl PersistentRuntime {
    fn start_phase(&mut self, snapshot: PhaseSnapshot) -> Result<ActivePhase, String> {
        let phase_wall_start = Instant::now();
        if snapshot.key.session_generation != self.identity.session_generation
            || self.sources.len() != ARENA_EXPECTED_TENSORS
        {
            return Err("persistent IQ1_S runtime/session identity mismatch".to_string());
        }
        let layer_id = snapshot.key.layer_id;
        let snapshot_stream = snapshot.key.stream;
        let phase_name = match snapshot.phase {
            LayerPhase::PhaseA => "A",
            LayerPhase::PhaseB => "B",
        };
        let trace_mode = std::env::var("HETGPU_IQ1S_TRACE_MODE")
            .map_err(|_| "HETGPU_IQ1S_TRACE_MODE is required".to_string())?;
        let prepare_start = Instant::now();
        let prepared = prepare_phase(&self.arena, snapshot, &trace_mode)?;
        let prepare_us = u64::try_from(prepare_start.elapsed().as_micros()).unwrap_or(u64::MAX);
        let transaction_id = prepared.compiled.transaction_id;
        let ticket = match self
            .pool
            .prepare_ticket(&prepared.compiled, &prepared.buffers)
        {
            Ok(ticket) => ticket,
            Err(error) => {
                let error = error.to_string();
                self.progress.append(
                    "phase_error",
                    None,
                    None,
                    None,
                    Some(transaction_id),
                    Some(layer_id),
                    Some(phase_name),
                    None,
                    Some(&error),
                )?;
                return Err(error);
            }
        };
        self.progress.append(
            "phase_prepared",
            None,
            None,
            None,
            Some(transaction_id),
            Some(layer_id),
            Some(phase_name),
            Some(ticket.slot_generation()),
            None,
        )?;
        if let Err(error) = self.pool.publish_ticket(&ticket) {
            let error = error.to_string();
            self.progress.append(
                "phase_error",
                None,
                None,
                None,
                Some(transaction_id),
                Some(layer_id),
                Some(phase_name),
                Some(ticket.slot_generation()),
                Some(&error),
            )?;
            return Err(error);
        }
        self.progress.append(
            "phase_published",
            None,
            None,
            None,
            Some(transaction_id),
            Some(layer_id),
            Some(phase_name),
            Some(ticket.slot_generation()),
            None,
        )?;
        Ok(ActivePhase {
            phase_wall_start,
            layer_id,
            phase_name,
            snapshot_stream,
            prepare_us,
            trace_mode,
            prepared,
            ticket,
        })
    }

    fn poll_phase(&mut self, active: &ActivePhase) -> Result<TicketPoll, String> {
        match self.pool.poll_ticket(&active.ticket) {
            Ok(status) => Ok(status),
            Err(error) => {
                let error = error.to_string();
                self.progress.append(
                    "phase_error",
                    None,
                    None,
                    None,
                    Some(active.prepared.compiled.transaction_id),
                    Some(active.layer_id),
                    Some(active.phase_name),
                    Some(active.ticket.slot_generation()),
                    Some(&error),
                )?;
                Err(error)
            }
        }
    }

    fn finish_phase(&mut self, active: ActivePhase) -> Result<PhaseOutcome, String> {
        let ActivePhase {
            phase_wall_start,
            layer_id,
            phase_name,
            snapshot_stream,
            prepare_us,
            trace_mode,
            prepared,
            ticket,
        } = active;
        let transaction_id = prepared.compiled.transaction_id;
        let completed = match self.pool.collect_ticket(&ticket) {
            Ok(completed) => completed,
            Err(error) => {
                let error = error.to_string();
                self.progress.append(
                    "phase_error",
                    None,
                    None,
                    None,
                    Some(transaction_id),
                    Some(layer_id),
                    Some(phase_name),
                    Some(ticket.slot_generation()),
                    Some(&error),
                )?;
                return Err(error);
            }
        };
        self.progress.append(
            "phase_collected",
            None,
            None,
            None,
            Some(transaction_id),
            Some(layer_id),
            Some(phase_name),
            Some(ticket.slot_generation()),
            None,
        )?;
        let reconstruct_start = Instant::now();
        let outputs = reconstruct_full_rows(&prepared, &completed)?;
        let reconstruct_us =
            u64::try_from(reconstruct_start.elapsed().as_micros()).unwrap_or(u64::MAX);
        let compare_start = Instant::now();
        let comparison = if self.sampled_comparison_complete {
            finite_only_comparison(&outputs)?
        } else {
            validate_all_finite(&outputs)?;
            compare_sampled_output(
                &prepared,
                &outputs,
                &self.identity,
                &trace_mode,
                &self.failure_capture_dir,
            )?
        };
        let compare_us = u64::try_from(compare_start.elapsed().as_micros()).unwrap_or(u64::MAX);
        let commands_per_cu = std::array::from_fn(|cu| prepared.compiled.commands[cu].len());
        let completions_per_cu = std::array::from_fn(|cu| completed.completions[cu].len());
        let program_sha256 = std::array::from_fn(|cu| {
            let digest: [u8; 32] = Sha256::digest(&prepared.compiled.programs[cu].encoded).into();
            hex_sha256(&digest)
        });
        let device_timings = completed.timings;
        let phase_wall_us =
            u64::try_from(phase_wall_start.elapsed().as_micros()).unwrap_or(u64::MAX);
        let proof = PersistentPhaseRecord::new(
            prepared.compiled.transaction_id,
            layer_id,
            phase_name,
            trace_mode,
            self.identity.session_generation,
            program_sha256,
            hex_sha256(&prepared.compiled.semantic_sha256),
            commands_per_cu,
            completions_per_cu,
            completed.dma.weight_bytes,
            comparison,
            PhaseTimingsUs {
                capture: 0,
                route_dma: 0,
                trace_build_or_cache: prepare_us,
                activation_pack: 0,
                activation_sync: device_timings.activation_sync_us,
                ring_publish: device_timings.ring_publish_us,
                doorbell: device_timings.doorbell_us,
                device_wait: device_timings.device_wait_us,
                completion_sync: device_timings.completion_sync_us,
                result_copy: device_timings.result_copy_us,
                reconstruct: reconstruct_us,
                compare: compare_us,
                log: 0,
                phase_wall: phase_wall_us.max(device_timings.phase_wall_us),
            },
        )?;
        // Proof storage is a publication precondition: no CUDA destination is
        // updated if the compact phase record cannot be written.
        self.ledger.append_phase(proof)?;
        self.sampled_comparison_complete = true;
        if env_truthy("HETGPU_QWEN_IQ1S_PROOF_SYNC_PHASE") {
            self.ledger.sync_boundary()?;
        }
        publish_outputs_with(&mut self.output_publisher, snapshot_stream, &outputs)?;
        Ok(PhaseOutcome {
            prepared,
            completed,
            outputs,
        })
    }

    #[cfg(test)]
    fn execute_phase(&mut self, snapshot: PhaseSnapshot) -> Result<PhaseOutcome, String> {
        let active = self.start_phase(snapshot)?;
        let mut backoff_us = 1u64;
        loop {
            match self.poll_phase(&active)? {
                TicketPoll::Complete => break,
                TicketPoll::Pending => {
                    std::thread::sleep(Duration::from_micros(backoff_us));
                    backoff_us = (backoff_us * 2).min(1_000);
                }
            }
        }
        self.finish_phase(active)
    }
}

impl PhasePipelineEngine for PersistentRuntime {
    type Input = PhaseSnapshot;
    type Active = ActivePhase;
    type Output = PhaseOutcome;

    fn has_capacity(&self) -> bool {
        self.pool.has_free_ticket_slot()
    }

    fn start(&mut self, input: Self::Input) -> Result<Self::Active, String> {
        self.start_phase(input)
    }

    fn poll(&mut self, active: &Self::Active) -> Result<TicketPoll, String> {
        self.poll_phase(active)
    }

    fn finish(&mut self, active: Self::Active) -> Result<Self::Output, String> {
        self.finish_phase(active)
    }

    fn poison(&mut self, error: &str) {
        if self.poisoned.is_none() {
            self.poisoned = Some(error.to_string());
        }
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use tempfile::tempdir;

    #[derive(Default)]
    struct FakeResultPublisher {
        batches: usize,
        copies: usize,
        streams: Vec<usize>,
        context_synchronizations: usize,
    }

    impl ResultPublisher for FakeResultPublisher {
        fn publish_batch(
            &mut self,
            stream: usize,
            outputs: &[PublishedOutput],
        ) -> Result<(), String> {
            self.batches += 1;
            self.copies += outputs.len();
            self.streams.push(stream);
            Ok(())
        }
    }

    struct FakePipelinedEngine {
        active: BTreeSet<u64>,
        released: bool,
        peak_active: Arc<AtomicUsize>,
    }

    impl PhasePipelineEngine for FakePipelinedEngine {
        type Input = u64;
        type Active = u64;
        type Output = u64;

        fn has_capacity(&self) -> bool {
            self.active.len() < 2
        }

        fn start(&mut self, input: Self::Input) -> Result<Self::Active, String> {
            if !self.active.insert(input) {
                return Err("duplicate fake pipeline input".to_string());
            }
            self.peak_active
                .fetch_max(self.active.len(), Ordering::SeqCst);
            if self.active.len() == 2 {
                self.released = true;
            }
            Ok(input)
        }

        fn poll(&mut self, _active: &Self::Active) -> Result<TicketPoll, String> {
            Ok(if self.released {
                TicketPoll::Complete
            } else {
                TicketPoll::Pending
            })
        }

        fn finish(&mut self, active: Self::Active) -> Result<Self::Output, String> {
            if !self.active.remove(&active) {
                return Err("unknown fake pipeline ticket".to_string());
            }
            Ok(active)
        }

        fn poison(&mut self, _error: &str) {}
    }

    #[test]
    fn persistent_pipeline_releases_the_runtime_lock_while_tickets_are_pending() {
        let peak_active = Arc::new(AtomicUsize::new(0));
        let pipeline = Arc::new(PhasePipeline::with_engine(FakePipelinedEngine {
            active: BTreeSet::new(),
            released: false,
            peak_active: peak_active.clone(),
        }));
        let start = Arc::new(Barrier::new(3));
        let workers = [17, 18].map(|transaction| {
            let pipeline = pipeline.clone();
            let start = start.clone();
            std::thread::spawn(move || {
                start.wait();
                pipeline.execute(transaction)
            })
        });
        start.wait();
        let mut completed = workers
            .into_iter()
            .map(|worker| worker.join().unwrap().unwrap())
            .collect::<Vec<_>>();
        completed.sort_unstable();
        assert_eq!(completed, vec![17, 18]);
        assert_eq!(peak_active.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn persistent_runtime_publishes_all_phase_outputs_in_one_stream_batch() {
        let outputs = (0..4)
            .map(|index| PublishedOutput {
                cuda_ptr: 0x1000 + index * 0x100,
                bytes: vec![index as u8; 64],
            })
            .collect::<Vec<_>>();
        let mut publisher = FakeResultPublisher::default();
        publish_outputs_with(&mut publisher, 0x55, &outputs).unwrap();
        assert_eq!(publisher.batches, 1);
        assert_eq!(publisher.copies, 4);
        assert_eq!(publisher.streams, vec![0x55]);
        assert_eq!(publisher.context_synchronizations, 0);
    }

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

    fn shared_gate_up_phase(distinct_up_activation: bool) -> (Arc<ArenaPlan>, PhaseSnapshot) {
        let gate_arena = arena(Iq1sExpertRole::Gate, 1);
        let up_arena = arena(Iq1sExpertRole::Up, 1);
        let mut shards = gate_arena.shards.clone();
        shards.extend(up_arena.shards.iter().cloned().map(|mut shard| {
            shard.offset += 8 * 1024 * 1024;
            shard
        }));
        let combined_arena = Arc::new(ArenaPlan {
            generation: gate_arena.generation,
            model_sha256: gate_arena.model_sha256,
            bank_bytes: gate_arena.bank_bytes,
            weight_bytes: gate_arena.weight_bytes,
            shards,
            hashes_verified: true,
        });

        let mut phase = snapshot(2, false);
        let up_identity = identity(Iq1sExpertRole::Up);
        let up_launches = phase.projections[0]
            .launches
            .iter()
            .enumerate()
            .map(|(index, gate_launch)| {
                let mut packed = gate_launch.packed_activations().to_vec();
                if distinct_up_activation {
                    packed[0] = packed[0].wrapping_add(1);
                }
                capture_activation_from_host(
                    LogicalLaunch {
                        matrix_ptr: 0x50_0000,
                        activation_ptr: gate_launch.launch.activation_ptr,
                        output_ptr: 0x60_0000 + index * 0x2000,
                        allocation_generation: 9,
                        content_hash: up_identity.content_sha256,
                        signature: gate_launch.launch.signature.clone(),
                    },
                    &packed,
                )
                .unwrap()
            })
            .collect();
        phase.projections.push(CapturedProjection {
            role: Iq1sExpertRole::Up,
            weight: ResolvedIq1sWeight {
                identity: up_identity.clone(),
                expert: 0,
                allocation_generation: 9,
                content_sha256: up_identity.content_sha256,
            },
            launches: up_launches,
        });
        (combined_arena, phase)
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
        for batch in [1, 6, 9, 16, 32] {
            for distinct in [false, true] {
                for mode in ["handwritten", "compiler"] {
                    let prepared = prepare_phase(&arena, snapshot(batch, distinct), mode).unwrap();
                    let groups = if distinct {
                        usize::from(batch.min(3))
                    } else {
                        usize::from(batch).div_ceil(16)
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
                        assert!(commands.windows(2).all(|pair| (
                            pair[0].expert_id,
                            pair[0].token_map_offset
                        ) < (
                            pair[1].expert_id,
                            pair[1].token_map_offset
                        )));
                    }
                }
            }
        }
    }

    #[test]
    fn iq1s_persistent_runtime_reuses_exact_gate_up_activation_slab() {
        let (arena, phase) = shared_gate_up_phase(false);
        let prepared = prepare_phase(&arena, phase, "compiler").unwrap();
        assert_eq!(prepared.buffers.activations.len(), 1);
        assert_eq!(prepared.compiled.activations.len(), 1);
        assert_ne!(
            prepared.compiled.activations[0].source_identity_sha256,
            [0; 32]
        );
        for commands in &prepared.compiled.commands {
            assert_eq!(commands.len(), 2);
            assert_eq!(commands[0].input_offset, commands[1].input_offset);
        }

        let (arena, phase) = shared_gate_up_phase(true);
        let distinct = prepare_phase(&arena, phase, "compiler").unwrap();
        assert_eq!(distinct.buffers.activations.len(), 2);
        assert_eq!(distinct.compiled.activations.len(), 2);
        assert_ne!(
            distinct.compiled.activations[0].source_identity_sha256,
            distinct.compiled.activations[1].source_identity_sha256
        );
    }

    #[test]
    fn iq1s_persistent_runtime_activation_identity_binds_lane_order() {
        let phase = snapshot(2, false);
        let mut lanes = phase
            .routes
            .iter()
            .copied()
            .zip(phase.projections[0].launches.iter().cloned())
            .collect::<Vec<_>>();
        let forward = packed_activation_identity(&phase.key, &lanes);
        lanes.reverse();
        let reversed = packed_activation_identity(&phase.key, &lanes);
        assert_ne!(forward, reversed);
    }

    #[test]
    fn iq1s_persistent_runtime_keeps_allocation_and_session_generations_independent() {
        let arena = arena(Iq1sExpertRole::Gate, 1);
        let mut phase = snapshot(1, false);
        phase.projections[0].weight.allocation_generation = 27;
        phase.projections[0].launches[0]
            .launch
            .allocation_generation = 27;

        let prepared = prepare_phase(&arena, phase, "handwritten").unwrap();
        assert_eq!(prepared.arena.generation, 9);
    }

    #[test]
    fn iq1s_persistent_runtime_diagnostic_comparison_policy_is_explicit_opt_in() {
        assert!(!initial_sampled_comparison_complete(false));
        assert!(initial_sampled_comparison_complete(true));
    }

    #[test]
    fn iq1s_persistent_runtime_qualifies_complete_name_and_digest_pairs() {
        let original = [
            0x9c, 0x83, 0xdc, 0xae, 0x07, 0xb4, 0xc7, 0xbf, 0x1d, 0x2e, 0x1c, 0xeb, 0xf4, 0x6c,
            0xcf, 0x0f, 0xf1, 0xeb, 0xf8, 0x84, 0x8a, 0x43, 0x70, 0x35, 0xfe, 0xf1, 0x45, 0x1d,
            0xee, 0x77, 0x70, 0xa3,
        ];
        let gridrom = [
            0xd7, 0x2f, 0xf7, 0x33, 0x6c, 0xb4, 0xdd, 0x49, 0x86, 0x7c, 0x5f, 0x22, 0x08, 0xa3,
            0x43, 0x92, 0x81, 0xea, 0xab, 0x1b, 0x0a, 0x9e, 0x59, 0x4b, 0x45, 0x8a, 0x4f, 0xec,
            0xc6, 0xb7, 0x99, 0x48,
        ];

        assert!(qualified_persistent_xclbin_identity(
            "qwen397b_iq1s_layer_persistent_9c83dcae.xclbin",
            original,
        ));
        assert!(qualified_persistent_xclbin_identity(
            "qwen397b_iq1s_layer_persistent_d72ff7336cb4dd49.xclbin",
            gridrom,
        ));
        assert!(!qualified_persistent_xclbin_identity(
            "qwen397b_iq1s_layer_persistent_9c83dcae.xclbin",
            gridrom,
        ));
        assert!(!qualified_persistent_xclbin_identity(
            "qwen397b_iq1s_layer_persistent_d72ff7336cb4dd49.xclbin",
            original,
        ));
        assert!(!qualified_persistent_xclbin_identity(
            "qwen397b_iq1s_layer_persistent_unreviewed.xclbin",
            [0x5a; 32],
        ));
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
    fn iq1s_persistent_progress_is_separate_fail_closed_nonproof_jsonl() {
        let directory = tempdir().unwrap();
        let ledger_path = directory.path().join("phase-ledger.jsonl");
        let progress_path = directory.path().join("progress.jsonl");
        let identity = PersistentRuntimeIdentity {
            model_sha256: [0x11; 32],
            xclbin_sha256: [0x33; 32],
            device_index: 0,
            session_generation: 9,
        };
        let mut sink = ProgressSink::create_at(&progress_path, &ledger_path, &identity).unwrap();
        sink.append(
            "phase_prepared",
            None,
            None,
            None,
            Some(71),
            Some(7),
            Some("A"),
            Some(3),
            None,
        )
        .unwrap();
        drop(sink);

        let line = std::fs::read_to_string(&progress_path).unwrap();
        let record: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(record["schema_version"], 1);
        assert_eq!(record["stage"], "phase_prepared");
        assert_eq!(record["transaction"], 71);
        assert_eq!(record["slot_generation"], 3);
        for proof_only in ["comparison", "timings_us", "commands_per_cu", "status"] {
            assert!(record.get(proof_only).is_none());
        }
        assert!(ProgressSink::create_at(&progress_path, &ledger_path, &identity).is_err());
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

    #[test]
    fn iq1s_persistent_numerical_failure_capture_is_complete_nonproof_and_immutable() {
        let directory = tempdir().unwrap();
        let capture = directory.path().join("failure-capture");
        let prepared = prepare_phase(
            &arena(Iq1sExpertRole::Gate, 1),
            snapshot(1, false),
            "handwritten",
        )
        .unwrap();
        let binding = prepared.output_bindings[0].clone();
        let identity = PersistentRuntimeIdentity {
            model_sha256: [0x11; 32],
            xclbin_sha256: [0x33; 32],
            device_index: 0,
            session_generation: 9,
        };
        let matrix = vec![0x41; 819_200];
        let activation = vec![0x52; 32 * Q8_1_MMQ_BYTES];
        let reference = vec![0.119_955_09_f32, -0.5];
        let actual = vec![0.111_208_32_f32, -0.5];
        let mismatch = NumericalMismatch {
            index: 0,
            actual: actual[0],
            reference: reference[0],
            absolute_error: (actual[0] - reference[0]).abs(),
            limit: 1.0e-4 + 1.0e-3 * reference[0].abs(),
        };

        write_numerical_failure_capture_at(
            &capture,
            &identity,
            "handwritten",
            &prepared,
            &binding,
            &matrix,
            &activation,
            &reference,
            &actual,
            mismatch,
        )
        .unwrap();

        assert_eq!(
            std::fs::read(capture.join("matrix.iq1s.bin")).unwrap(),
            matrix
        );
        assert_eq!(
            std::fs::read(capture.join("activation.q8_1.bin")).unwrap(),
            activation
        );
        assert_eq!(
            std::fs::read(capture.join("reference.f32.bin")).unwrap(),
            reference
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            std::fs::read(capture.join("actual.f32.bin")).unwrap(),
            actual
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        );
        for bank in 0..ARENA_BANK_COUNT {
            assert!(capture.join(format!("bank-{bank}.commands.json")).is_file());
            assert!(capture.join(format!("bank-{bank}.program.bin")).is_file());
            assert!(capture.join(format!("bank-{bank}.program.asm")).is_file());
        }
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(capture.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["schema_version"], 1);
        assert_eq!(manifest["status"], "numerical_mismatch_nonproof");
        assert_eq!(manifest["trace_mode"], "handwritten");
        assert_eq!(manifest["transaction_id"], 71);
        assert_eq!(manifest["layer_id"], 7);
        assert_eq!(manifest["mismatch"]["index"], 0);
        assert_eq!(
            manifest["binding"]["tensor_name"],
            "blk.7.ffn_gate_exps.weight"
        );
        assert_eq!(manifest["files"].as_array().unwrap().len(), 16);
        assert!(manifest.get("proof_status").is_none());

        let error = write_numerical_failure_capture_at(
            &capture,
            &identity,
            "handwritten",
            &prepared,
            &binding,
            &matrix,
            &activation,
            &reference,
            &actual,
            mismatch,
        )
        .unwrap_err();
        assert!(error.contains("create numerical failure capture directory"));
    }
}
