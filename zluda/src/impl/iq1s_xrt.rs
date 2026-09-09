//! AU250 D=1024 planning and execution for captured IQ1_S launches.

use super::iq1s_tmatmul::{
    checked_output_element_count, raw_component_dots, reconstruct_from_raw, CapturedLaunch,
    ComponentKind, GgmlType19Signature, MatrixCacheIdentity, MatrixSource, Q8_1Block,
};
use super::xrt_tmatmul::{XrtTmatmulPool, XrtWaveCompletion, XrtWaveJob};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

pub(crate) const AU250_DIM: usize = 1024;
pub(crate) const AU250_GROUP_VALUES: usize = 32;
pub(crate) const AU250_GROUPS_PER_K_TILE: usize = 32;
pub(crate) const AU250_MATRIX_BYTES: usize = AU250_DIM * AU250_DIM / 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Au250Tile {
    pub(crate) row_tile: usize,
    pub(crate) k_tile: usize,
    pub(crate) valid_out: usize,
    pub(crate) valid_in: usize,
    pub(crate) group_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Au250MatrixKey {
    pub(crate) row_tile: usize,
    pub(crate) k_tile: usize,
    pub(crate) kind: ComponentKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LaneAssignment {
    pub(crate) lane: usize,
    pub(crate) batch_index: usize,
    pub(crate) global_group: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlannedAu250Job {
    pub(crate) request_id: u64,
    pub(crate) cu_index: usize,
    pub(crate) matrix_key: Au250MatrixKey,
    pub(crate) assignments: Vec<LaneAssignment>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct XrtIq1sEvidence {
    pub(crate) backend: &'static str,
    pub(crate) logical_batch: usize,
    pub(crate) row_tiles: usize,
    pub(crate) k_tiles: usize,
    pub(crate) submission_count: u64,
    pub(crate) completion_count: u64,
    pub(crate) per_cu_submissions: Vec<u64>,
    pub(crate) per_cu_completions: Vec<u64>,
    pub(crate) request_ids: Vec<u64>,
    pub(crate) stall_codes: Vec<u32>,
    pub(crate) raw_min: i16,
    pub(crate) raw_max: i16,
    pub(crate) reference_checked_components: u64,
    pub(crate) comparison_status: &'static str,
    pub(crate) resident_matrix_hits: u64,
    pub(crate) resident_matrix_misses: u64,
    pub(crate) resident_matrix_bytes_transferred: u64,
    pub(crate) program_cache_hits: u64,
    pub(crate) program_cache_misses: u64,
    pub(crate) physical_completions: Vec<XrtPhysicalCompletionEvidence>,
    pub(crate) host_pack_hits: u64,
    pub(crate) host_pack_misses: u64,
    pub(crate) host_pack_bytes_built: u64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct XrtPhysicalCompletionEvidence {
    pub(crate) request_id: u64,
    pub(crate) cu_index: usize,
    pub(crate) stall_code: u32,
    pub(crate) dispatch_to_stall_ns: u64,
    pub(crate) matrix_key_sha256: String,
    pub(crate) matrix_content_sha256: String,
    pub(crate) matrix_address: u64,
    pub(crate) matrix_cache_hit: bool,
    pub(crate) matrix_bytes_transferred: usize,
    pub(crate) trace_mode: String,
    pub(crate) model_context_limit: u32,
    pub(crate) trace_semantic_sha256: String,
    pub(crate) trace_assembly_sha256: String,
    pub(crate) replay_safe_program_sha256: String,
    pub(crate) trace_assembly: String,
    pub(crate) trace_instructions: Vec<Vec<String>>,
    pub(crate) encoded_program_sha256: String,
    pub(crate) encoded_program_hex: String,
    pub(crate) program_address: u64,
    pub(crate) program_bytes: usize,
    pub(crate) program_cache_hit: bool,
}

fn sha256_hex(value: [u8; 32]) -> String {
    value.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn bytes_hex(value: &[u8]) -> String {
    value.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[derive(Debug)]
pub(crate) struct XrtIq1sResult {
    pub(crate) outputs: Vec<f32>,
    pub(crate) evidence: XrtIq1sEvidence,
}

pub(crate) trait CapturedLaunchExecutor {
    type Output;

    fn execute(&mut self, captured: &CapturedLaunch) -> Result<Self::Output, String>;
}

pub(crate) struct DirectCapturedLaunchExecutor;

#[derive(Debug, Clone, Serialize)]
pub(crate) struct SampledOutputComparison {
    pub(crate) status: &'static str,
    pub(crate) reference_backend: &'static str,
    pub(crate) checked_elements: usize,
    pub(crate) atol: f32,
    pub(crate) rtol: f32,
    pub(crate) max_absolute_error: f32,
    pub(crate) max_relative_error: f32,
    pub(crate) reference_outputs: Vec<f32>,
    pub(crate) actual_outputs: Vec<f32>,
}

fn scalar_reference_outputs(captured: &CapturedLaunch) -> Result<Vec<f32>, String> {
    let signature = &captured.launch.signature;
    let batch = usize::try_from(signature.ne11).map_err(|_| "batch overflow")?;
    let rows = usize::try_from(signature.ne0).map_err(|_| "row overflow")?;
    let groups = usize::try_from(signature.ne00 / 32).map_err(|_| "group overflow")?;
    let mut outputs = vec![0_f32; batch.checked_mul(rows).ok_or("output overflow")?];
    for batch_index in 0..batch {
        for row in 0..rows {
            for global_group in 0..groups {
                let q8 = captured.q8_group(batch_index, global_group)?;
                let (d, group) = captured.matrix.group(row, global_group)?;
                let (grid, delta) = raw_component_dots(&group, &q8);
                let contribution = reconstruct_from_raw(&group, d, &q8, grid << 8, delta << 8)?;
                let output = &mut outputs[batch_index * rows + row];
                *output = (*output + contribution) as f32;
            }
        }
    }
    Ok(outputs)
}

pub(crate) fn compare_outputs_with_reference(
    captured: &CapturedLaunch,
    actual_outputs: &[f32],
    atol: f32,
    rtol: f32,
) -> Result<SampledOutputComparison, String> {
    if !atol.is_finite() || atol < 0.0 || !rtol.is_finite() || rtol < 0.0 {
        return Err("IQ1_S comparison tolerances must be finite and nonnegative".into());
    }
    let reference_outputs = scalar_reference_outputs(captured)?;
    if reference_outputs.len() != actual_outputs.len() {
        return Err(format!(
            "IQ1_S sampled comparison length mismatch: reference {}, actual {}",
            reference_outputs.len(),
            actual_outputs.len()
        ));
    }
    let mut max_absolute_error = 0.0_f32;
    let mut max_relative_error = 0.0_f32;
    for (index, (&reference, &actual)) in reference_outputs.iter().zip(actual_outputs).enumerate() {
        if !reference.is_finite() || !actual.is_finite() {
            return Err(format!("IQ1_S sampled output {index} is non-finite"));
        }
        let absolute_error = (actual - reference).abs();
        let relative_error = if reference == 0.0 {
            absolute_error
        } else {
            absolute_error / reference.abs()
        };
        max_absolute_error = max_absolute_error.max(absolute_error);
        max_relative_error = max_relative_error.max(relative_error);
        let limit = atol + rtol * reference.abs();
        if absolute_error > limit {
            return Err(format!(
                "IQ1_S sampled output {index} is outside tolerance: actual={actual}, reference={reference}, absolute_error={absolute_error}, limit={limit}"
            ));
        }
    }
    Ok(SampledOutputComparison {
        status: "pass",
        reference_backend: "scalar_iq1s",
        checked_elements: actual_outputs.len(),
        atol,
        rtol,
        max_absolute_error,
        max_relative_error,
        reference_outputs,
        actual_outputs: actual_outputs.to_vec(),
    })
}

pub(crate) trait Au250WaveExecutor {
    fn lane_capacities(&self) -> Vec<usize>;
    fn run_wave(&mut self, jobs: Vec<XrtWaveJob>) -> Result<Vec<XrtWaveCompletion>, String>;
}

impl Au250WaveExecutor for XrtTmatmulPool {
    fn lane_capacities(&self) -> Vec<usize> {
        XrtTmatmulPool::lane_capacities(self)
    }

    fn run_wave(&mut self, jobs: Vec<XrtWaveJob>) -> Result<Vec<XrtWaveCompletion>, String> {
        XrtTmatmulPool::run_wave(self, jobs).map_err(|error| error.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PackedMatrixKey {
    identity: MatrixCacheIdentity,
    tile: Au250MatrixKey,
}

struct PackedMatrixEntry {
    value: Arc<[u8]>,
    last_used: u64,
}

struct PackedMatrixCache {
    capacity_bytes: usize,
    resident_bytes: usize,
    clock: u64,
    entries: HashMap<PackedMatrixKey, PackedMatrixEntry>,
}

impl PackedMatrixCache {
    fn from_env() -> Result<Self, String> {
        let configured = match std::env::var("HETGPU_XRT_MATRIX_CACHE_BYTES") {
            Ok(text) => Some(text),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => return Err(format!("read HETGPU_XRT_MATRIX_CACHE_BYTES: {error}")),
        };
        let capacity_bytes = parse_matrix_cache_capacity(configured.as_deref())?;
        Ok(Self {
            capacity_bytes,
            resident_bytes: 0,
            clock: 0,
            entries: HashMap::new(),
        })
    }

    fn get_or_insert(
        &mut self,
        key: PackedMatrixKey,
        build: impl FnOnce() -> Result<Vec<u8>, String>,
    ) -> Result<Arc<[u8]>, String> {
        self.get_or_insert_with_status(key, build)
            .map(|(value, _)| value)
    }

    fn get_or_insert_with_status(
        &mut self,
        key: PackedMatrixKey,
        build: impl FnOnce() -> Result<Vec<u8>, String>,
    ) -> Result<(Arc<[u8]>, bool), String> {
        self.clock = self
            .clock
            .checked_add(1)
            .ok_or("AU250 matrix cache clock overflow")?;
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_used = self.clock;
            return Ok((entry.value.clone(), true));
        }

        let built = build()?;
        if built.len() > self.capacity_bytes {
            return Err(format!(
                "packed AU250 matrix requires {} bytes, exceeding cache capacity {}",
                built.len(),
                self.capacity_bytes
            ));
        }
        while self
            .resident_bytes
            .checked_add(built.len())
            .ok_or("AU250 matrix cache byte count overflow")?
            > self.capacity_bytes
        {
            let evict_key = self
                .entries
                .iter()
                .filter(|(_, entry)| Arc::strong_count(&entry.value) == 1)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone())
                .ok_or("AU250 matrix cache is full of in-flight values")?;
            let evicted = self
                .entries
                .remove(&evict_key)
                .expect("selected cache entry exists");
            self.resident_bytes = self
                .resident_bytes
                .checked_sub(evicted.value.len())
                .ok_or("AU250 matrix cache accounting underflow")?;
        }
        let value: Arc<[u8]> = Arc::from(built);
        self.resident_bytes = self
            .resident_bytes
            .checked_add(value.len())
            .ok_or("AU250 matrix cache byte count overflow")?;
        self.entries.insert(
            key,
            PackedMatrixEntry {
                value: value.clone(),
                last_used: self.clock,
            },
        );
        Ok((value, false))
    }
}

fn parse_matrix_cache_capacity(configured: Option<&str>) -> Result<usize, String> {
    const DEFAULT_BYTES: u64 = 512 * 1024 * 1024;
    let capacity_u64 = match configured {
        Some(text) => text
            .parse::<u64>()
            .map_err(|error| format!("HETGPU_XRT_MATRIX_CACHE_BYTES={text:?}: {error}"))?,
        None => DEFAULT_BYTES,
    };
    if capacity_u64 == 0 {
        return Err("HETGPU_XRT_MATRIX_CACHE_BYTES must be nonzero".to_string());
    }
    usize::try_from(capacity_u64)
        .map_err(|_| "HETGPU_XRT_MATRIX_CACHE_BYTES does not fit usize".to_string())
}

fn packed_matrix_for(
    captured: &CapturedLaunch,
    tile: Au250Tile,
    key: Au250MatrixKey,
) -> Result<(Arc<[u8]>, bool), String> {
    static CACHE: OnceLock<Mutex<Result<PackedMatrixCache, String>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(PackedMatrixCache::from_env()));
    let mut guard = cache
        .lock()
        .map_err(|_| "AU250 matrix cache mutex poisoned".to_string())?;
    let cache = match &mut *guard {
        Ok(cache) => cache,
        Err(error) => return Err(error.clone()),
    };
    let identity = MatrixCacheIdentity {
        matrix_ptr: captured.launch.matrix_ptr,
        signature: captured.launch.signature.clone(),
        allocation_generation: captured.launch.allocation_generation,
        content_hash: captured.launch.content_hash,
    };
    cache.get_or_insert_with_status(
        PackedMatrixKey {
            identity,
            tile: key,
        },
        || pack_component_matrix(&captured.matrix, tile, key.kind),
    )
}

fn resident_matrix_key(captured: &CapturedLaunch, key: Au250MatrixKey) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"hetgpu-iq1s-resident-matrix-v1");
    hash.update(captured.launch.content_hash);
    hash.update(captured.launch.allocation_generation.to_le_bytes());
    hash.update(captured.launch.matrix_ptr.to_le_bytes());
    let signature = &captured.launch.signature;
    hash.update(signature.kernel.as_bytes());
    for value in [
        signature.ne00,
        signature.ne01,
        signature.stride01,
        signature.ne10,
        signature.ne11,
        signature.stride11,
        signature.ne0,
    ] {
        hash.update(value.to_le_bytes());
    }
    hash.update(key.row_tile.to_le_bytes());
    hash.update(key.k_tile.to_le_bytes());
    hash.update([match key.kind {
        ComponentKind::Grid => 0,
        ComponentKind::Delta => 1,
    }]);
    hash.finalize().into()
}

fn trace_tokens(assembly: &str) -> Vec<Vec<String>> {
    assembly
        .lines()
        .filter_map(|raw| {
            let line = raw.split(';').next().unwrap_or("").trim();
            (!line.is_empty()).then(|| {
                line.split(|byte: char| byte.is_whitespace() || byte == ',')
                    .filter(|token| !token.is_empty())
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
        })
        .collect()
}

fn validate_completion_binding(
    completion: &XrtWaveCompletion,
    expected_matrix_key: [u8; 32],
    expected_matrix_sha256: [u8; 32],
    strict_qwen: bool,
    expected_trace_mode: Option<&str>,
) -> Result<(), String> {
    if completion.matrix_key != expected_matrix_key
        || completion.matrix_sha256 != expected_matrix_sha256
    {
        return Err(format!(
            "AU250 completion {} matrix identity does not match its submitted job",
            completion.request_id
        ));
    }
    if completion.matrix_address == 0 || completion.program_address == 0 {
        return Err(format!(
            "AU250 completion {} has a zero resident matrix or program address",
            completion.request_id
        ));
    }
    let expected_transfer = if completion.matrix_cache_hit {
        0
    } else {
        AU250_MATRIX_BYTES
    };
    if completion.matrix_bytes_transferred != expected_transfer {
        return Err(format!(
            "AU250 completion {} resident transfer bytes {} do not match cache status",
            completion.request_id, completion.matrix_bytes_transferred
        ));
    }
    if completion.program_bytes == 0
        || completion.program_bytes % 16 != 0
        || completion.encoded_program.len() != completion.program_bytes
        || Sha256::digest(&completion.encoded_program).as_slice() != completion.program_sha256
    {
        return Err(format!(
            "AU250 completion {} encoded program body/hash/length mismatch",
            completion.request_id
        ));
    }
    if !strict_qwen {
        return Ok(());
    }
    let expected_trace_mode = expected_trace_mode.ok_or_else(|| {
        "strict Qwen completion validation requires HETGPU_IQ1S_TRACE_MODE".to_string()
    })?;
    if completion.trace_mode != expected_trace_mode
        || !matches!(completion.trace_mode.as_str(), "handwritten" | "compiler")
    {
        return Err(format!(
            "AU250 completion {} trace mode {:?} does not match {:?}",
            completion.request_id, completion.trace_mode, expected_trace_mode
        ));
    }
    if completion.model_context_limit != super::iq1s_trace::QWEN_MODEL_CONTEXT_LIMIT {
        return Err(format!(
            "AU250 completion {} model context limit {} is not {}",
            completion.request_id,
            completion.model_context_limit,
            super::iq1s_trace::QWEN_MODEL_CONTEXT_LIMIT
        ));
    }
    if trace_tokens(&completion.trace_assembly) != completion.trace_instructions
        || completion.trace_instructions.last() != Some(&vec!["stall".to_string()])
    {
        return Err(format!(
            "AU250 completion {} trace body is not the recorded terminal-STALL instruction sequence",
            completion.request_id
        ));
    }
    if Sha256::digest(completion.trace_assembly.as_bytes()).as_slice()
        != completion.trace_assembly_sha256
        || super::iq1s_trace::semantic_sha256(&completion.trace_instructions)
            != completion.trace_semantic_sha256
        || completion.replay_safe_program_sha256 == [0; 32]
        || completion.program_sha256 == [0; 32]
    {
        return Err(format!(
            "AU250 completion {} trace or program hash mismatch",
            completion.request_id
        ));
    }
    Ok(())
}

pub(crate) fn plan_au250_tiles(signature: &GgmlType19Signature) -> Result<Vec<Au250Tile>, String> {
    signature.validate()?;
    let rows = usize::try_from(signature.ne0).map_err(|_| "row count does not fit usize")?;
    let columns = usize::try_from(signature.ne00).map_err(|_| "column count does not fit usize")?;
    let mut result = Vec::new();
    for row_start in (0..rows).step_by(AU250_DIM) {
        for k_start in (0..columns).step_by(AU250_DIM) {
            let valid_in = (columns - k_start).min(AU250_DIM);
            result.push(Au250Tile {
                row_tile: row_start / AU250_DIM,
                k_tile: k_start / AU250_DIM,
                valid_out: (rows - row_start).min(AU250_DIM),
                valid_in,
                group_count: valid_in.div_ceil(AU250_GROUP_VALUES),
            });
        }
    }
    Ok(result)
}

pub(crate) fn pack_component_matrix(
    source: &MatrixSource,
    tile: Au250Tile,
    kind: ComponentKind,
) -> Result<Vec<u8>, String> {
    if tile.valid_out == 0
        || tile.valid_out > AU250_DIM
        || tile.valid_in == 0
        || tile.valid_in > AU250_DIM
        || tile.group_count != tile.valid_in.div_ceil(AU250_GROUP_VALUES)
        || tile.group_count > AU250_GROUPS_PER_K_TILE
    {
        return Err("invalid AU250 tile geometry".to_string());
    }
    let row_start = tile
        .row_tile
        .checked_mul(AU250_DIM)
        .ok_or("AU250 row tile overflow")?;
    let group_start = tile
        .k_tile
        .checked_mul(AU250_GROUPS_PER_K_TILE)
        .ok_or("AU250 K tile overflow")?;
    let mut packed = vec![0u8; AU250_MATRIX_BYTES];
    for local_row in 0..tile.valid_out {
        let global_row = row_start
            .checked_add(local_row)
            .ok_or("AU250 row coordinate overflow")?;
        for local_group in 0..tile.group_count {
            let global_group = group_start
                .checked_add(local_group)
                .ok_or("AU250 group coordinate overflow")?;
            let (_, group) = source.group(global_row, global_group)?;
            for position in 0..AU250_GROUP_VALUES {
                let value = match kind {
                    ComponentKind::Grid => group.grid_values[position],
                    ComponentKind::Delta => group.delta_sign,
                };
                let code = match value {
                    -1 => 3u8,
                    0 => 0u8,
                    1 => 1u8,
                    _ => return Err(format!("component value {value} is not ternary")),
                };
                let column = local_group * AU250_GROUP_VALUES + position;
                if column >= tile.valid_in {
                    continue;
                }
                let element = local_row * AU250_DIM + column;
                packed[element / 4] |= code << (2 * (element % 4));
            }
        }
    }
    Ok(packed)
}

pub(crate) fn pack_lane_input(
    lanes: usize,
    assignments: &[(usize, usize, Q8_1Block)],
) -> Result<Vec<u8>, String> {
    if lanes == 0 {
        return Err("AU250 lane count must be positive".to_string());
    }
    let elements = AU250_DIM
        .checked_mul(lanes)
        .ok_or("AU250 lane input element count overflow")?;
    let byte_count = elements
        .checked_mul(std::mem::size_of::<i16>())
        .ok_or("AU250 lane input byte count overflow")?;
    let mut bytes = vec![0u8; byte_count];
    let mut used_lanes = HashSet::new();
    for &(lane, group_slot, q8) in assignments {
        if lane >= lanes {
            return Err(format!("AU250 lane {lane} is outside {lanes} lanes"));
        }
        if !used_lanes.insert(lane) {
            return Err(format!("duplicate AU250 lane assignment {lane}"));
        }
        if group_slot >= AU250_GROUPS_PER_K_TILE {
            return Err(format!(
                "AU250 group slot {group_slot} is outside one K tile"
            ));
        }
        if !q8.d.is_finite() || !q8.s.is_finite() {
            return Err("Q8 lane assignment has non-finite factors".to_string());
        }
        for (position, quant) in q8.qs.into_iter().enumerate() {
            let dimension = group_slot * AU250_GROUP_VALUES + position;
            let element = dimension * lanes + lane;
            let offset = element * std::mem::size_of::<i16>();
            bytes[offset..offset + 2].copy_from_slice(&i16::from(quant).to_le_bytes());
        }
    }
    Ok(bytes)
}

pub(crate) fn raw_dot_bounds() -> (i16, i16) {
    (-4096, 4096)
}

pub(crate) fn plan_au250_jobs(
    signature: &GgmlType19Signature,
    lane_capacities: &[usize],
) -> Result<Vec<Vec<PlannedAu250Job>>, String> {
    if lane_capacities.is_empty() || lane_capacities.iter().any(|capacity| *capacity == 0) {
        return Err("AU250 CU lane capacities must be nonempty and positive".to_string());
    }
    let batch_count =
        usize::try_from(signature.ne11).map_err(|_| "batch count does not fit usize")?;
    let mut waves = Vec::new();
    let mut next_request_id = 0u64;
    for tile in plan_au250_tiles(signature)? {
        let group_start = tile
            .k_tile
            .checked_mul(AU250_GROUPS_PER_K_TILE)
            .ok_or("AU250 group coordinate overflow")?;
        let assignment_count = batch_count
            .checked_mul(tile.group_count)
            .ok_or("AU250 assignment count overflow")?;
        for kind in [ComponentKind::Grid, ComponentKind::Delta] {
            let matrix_key = Au250MatrixKey {
                row_tile: tile.row_tile,
                k_tile: tile.k_tile,
                kind,
            };
            let mut cursor = 0usize;
            while cursor < assignment_count {
                let mut wave = Vec::new();
                for (cu_index, &capacity) in lane_capacities.iter().enumerate() {
                    if cursor == assignment_count {
                        break;
                    }
                    let take = capacity.min(assignment_count - cursor);
                    let mut assignments = Vec::with_capacity(take);
                    for lane in 0..take {
                        let ordinal = cursor
                            .checked_add(lane)
                            .ok_or("AU250 assignment ordinal overflow")?;
                        let batch_index = ordinal / tile.group_count;
                        let local_group = ordinal % tile.group_count;
                        let global_group = group_start
                            .checked_add(local_group)
                            .ok_or("AU250 global group overflow")?;
                        assignments.push(LaneAssignment {
                            lane,
                            batch_index,
                            global_group,
                        });
                    }
                    let request_id = next_request_id;
                    next_request_id = next_request_id
                        .checked_add(1)
                        .ok_or("AU250 request ID overflow")?;
                    wave.push(PlannedAu250Job {
                        request_id,
                        cu_index,
                        matrix_key,
                        assignments,
                    });
                    cursor += take;
                }
                if wave.is_empty() {
                    return Err("AU250 planner made no progress".to_string());
                }
                validate_planned_wave(&wave, lane_capacities, tile, batch_count)?;
                waves.push(wave);
            }
        }
    }
    Ok(waves)
}

fn validate_planned_wave(
    wave: &[PlannedAu250Job],
    lane_capacities: &[usize],
    tile: Au250Tile,
    batch_count: usize,
) -> Result<(), String> {
    let mut cu_indices = HashSet::new();
    for job in wave {
        if !cu_indices.insert(job.cu_index) {
            return Err(format!("duplicate CU {} in AU250 wave", job.cu_index));
        }
        let capacity = *lane_capacities
            .get(job.cu_index)
            .ok_or("AU250 planned job selects an unknown CU")?;
        let mut lanes = HashSet::new();
        for assignment in &job.assignments {
            if assignment.lane >= capacity || !lanes.insert(assignment.lane) {
                return Err(format!(
                    "invalid or duplicate lane {} for CU {}",
                    assignment.lane, job.cu_index
                ));
            }
            let local_group = assignment
                .global_group
                .checked_sub(tile.k_tile * AU250_GROUPS_PER_K_TILE)
                .ok_or("AU250 assignment group precedes its K tile")?;
            if local_group >= tile.group_count || assignment.batch_index >= batch_count {
                return Err("AU250 assignment is outside its tile or batch".to_string());
            }
        }
    }
    Ok(())
}

pub(crate) fn execute_captured_with(
    captured: &CapturedLaunch,
    backend: &mut impl Au250WaveExecutor,
) -> Result<XrtIq1sResult, String> {
    captured.launch.signature.validate()?;
    let signature = &captured.launch.signature;
    let batch = usize::try_from(signature.ne11).map_err(|_| "batch count does not fit usize")?;
    let rows = usize::try_from(signature.ne0).map_err(|_| "row count does not fit usize")?;
    let columns = usize::try_from(signature.ne00).map_err(|_| "column count does not fit usize")?;
    let groups = columns / AU250_GROUP_VALUES;
    let output_elements = checked_output_element_count(signature)?;
    let lane_capacities = backend.lane_capacities();
    if lane_capacities.is_empty() || lane_capacities.iter().any(|lanes| *lanes == 0) {
        return Err("AU250 backend returned invalid lane capacities".to_string());
    }

    let tiles = plan_au250_tiles(signature)?;
    let tile_by_coordinate = tiles
        .iter()
        .copied()
        .map(|tile| ((tile.row_tile, tile.k_tile), tile))
        .collect::<HashMap<_, _>>();
    let planned_waves = plan_au250_jobs(signature, &lane_capacities)?;
    let slot_count = 2usize
        .checked_mul(batch)
        .and_then(|value| value.checked_mul(rows))
        .and_then(|value| value.checked_mul(groups))
        .ok_or("AU250 raw component slot count overflow")?;
    let mut raw_slots = vec![None::<i16>; slot_count];
    let mut submission_count = 0u64;
    let mut completion_count = 0u64;
    let mut per_cu_submissions = vec![0u64; lane_capacities.len()];
    let mut per_cu_completions = vec![0u64; lane_capacities.len()];
    let mut request_ids = Vec::new();
    let mut all_request_ids = HashSet::new();
    let mut stall_codes = Vec::new();
    let mut raw_min = None::<i16>;
    let mut raw_max = None::<i16>;
    let mut resident_matrix_hits = 0u64;
    let mut resident_matrix_misses = 0u64;
    let mut resident_matrix_bytes_transferred = 0u64;
    let mut program_cache_hits = 0u64;
    let mut program_cache_misses = 0u64;
    let mut physical_completions = Vec::new();
    let mut host_pack_hits = 0u64;
    let mut host_pack_misses = 0u64;
    let mut host_pack_bytes_built = 0u64;
    let strict_qwen = std::env::var("HETGPU_QWEN_IQ1S_STRICT").as_deref() == Ok("1");
    let expected_trace_mode = std::env::var("HETGPU_IQ1S_TRACE_MODE").ok();

    for planned_wave in planned_waves {
        let mut xrt_jobs = Vec::with_capacity(planned_wave.len());
        for planned in &planned_wave {
            let tile = *tile_by_coordinate
                .get(&(planned.matrix_key.row_tile, planned.matrix_key.k_tile))
                .ok_or("planned AU250 job refers to unknown tile")?;
            let (matrix, host_cache_hit) = packed_matrix_for(captured, tile, planned.matrix_key)?;
            if host_cache_hit {
                host_pack_hits = host_pack_hits
                    .checked_add(1)
                    .ok_or("AU250 host pack hit count overflow")?;
            } else {
                host_pack_misses = host_pack_misses
                    .checked_add(1)
                    .ok_or("AU250 host pack miss count overflow")?;
                host_pack_bytes_built = host_pack_bytes_built
                    .checked_add(
                        u64::try_from(matrix.len())
                            .map_err(|_| "AU250 host pack bytes do not fit u64")?,
                    )
                    .ok_or("AU250 host pack byte count overflow")?;
            }
            let lanes = *lane_capacities
                .get(planned.cu_index)
                .ok_or("planned AU250 job refers to unknown CU")?;
            let group_start = tile
                .k_tile
                .checked_mul(AU250_GROUPS_PER_K_TILE)
                .ok_or("AU250 group start overflow")?;
            let mut q8_assignments = Vec::with_capacity(planned.assignments.len());
            for assignment in &planned.assignments {
                let group_slot = assignment
                    .global_group
                    .checked_sub(group_start)
                    .ok_or("AU250 assignment group precedes its K tile")?;
                let q8 = captured.q8_group(assignment.batch_index, assignment.global_group)?;
                q8_assignments.push((assignment.lane, group_slot, q8));
            }
            xrt_jobs.push(XrtWaveJob {
                request_id: planned.request_id,
                cu_index: planned.cu_index,
                matrix_key: resident_matrix_key(captured, planned.matrix_key),
                matrix_sha256: Sha256::digest(&matrix).into(),
                matrix,
                input: pack_lane_input(lanes, &q8_assignments)?,
            });
        }

        submission_count = submission_count
            .checked_add(
                u64::try_from(planned_wave.len())
                    .map_err(|_| "AU250 wave size does not fit u64")?,
            )
            .ok_or("AU250 submission count overflow")?;
        for planned in &planned_wave {
            per_cu_submissions[planned.cu_index] = per_cu_submissions[planned.cu_index]
                .checked_add(1)
                .ok_or("AU250 per-CU submission count overflow")?;
        }
        let expected_matrix_by_request = xrt_jobs
            .iter()
            .map(|job| (job.request_id, (job.matrix_key, job.matrix_sha256)))
            .collect::<HashMap<_, _>>();
        let completions = backend.run_wave(xrt_jobs)?;
        if completions.len() != planned_wave.len() {
            return Err(format!(
                "AU250 completion count {} does not match planned count {}",
                completions.len(),
                planned_wave.len()
            ));
        }
        let planned_by_request = planned_wave
            .iter()
            .enumerate()
            .map(|(index, job)| (job.request_id, index))
            .collect::<HashMap<_, _>>();
        let mut completion_indices = Vec::with_capacity(completions.len());
        let mut seen = vec![false; planned_wave.len()];
        for completion in &completions {
            let planned_index =
                *planned_by_request
                    .get(&completion.request_id)
                    .ok_or_else(|| {
                        format!(
                            "AU250 completion has unknown request id {}",
                            completion.request_id
                        )
                    })?;
            if std::mem::replace(&mut seen[planned_index], true) {
                return Err(format!(
                    "AU250 completion duplicated request id {}",
                    completion.request_id
                ));
            }
            let planned = &planned_wave[planned_index];
            if completion.cu_index != planned.cu_index {
                return Err(format!(
                    "AU250 completion request {} returned CU {}, expected {}",
                    completion.request_id, completion.cu_index, planned.cu_index
                ));
            }
            if completion.stall_code == 0 {
                return Err(format!(
                    "AU250 completion request {} has zero STALL code",
                    completion.request_id
                ));
            }
            let (expected_matrix_key, expected_matrix_sha256) = expected_matrix_by_request
                .get(&completion.request_id)
                .copied()
                .ok_or_else(|| {
                    format!(
                        "AU250 completion {} has no submitted matrix identity",
                        completion.request_id
                    )
                })?;
            validate_completion_binding(
                completion,
                expected_matrix_key,
                expected_matrix_sha256,
                strict_qwen,
                expected_trace_mode.as_deref(),
            )?;
            if completion.matrix_cache_hit {
                resident_matrix_hits = resident_matrix_hits
                    .checked_add(1)
                    .ok_or("AU250 resident matrix hit count overflow")?;
            } else {
                resident_matrix_misses = resident_matrix_misses
                    .checked_add(1)
                    .ok_or("AU250 resident matrix miss count overflow")?;
            }
            resident_matrix_bytes_transferred = resident_matrix_bytes_transferred
                .checked_add(
                    u64::try_from(completion.matrix_bytes_transferred)
                        .map_err(|_| "AU250 resident transfer bytes do not fit u64")?,
                )
                .ok_or("AU250 resident transfer byte count overflow")?;
            if completion.program_cache_hit {
                program_cache_hits = program_cache_hits
                    .checked_add(1)
                    .ok_or("AU250 program cache hit count overflow")?;
            } else {
                program_cache_misses = program_cache_misses
                    .checked_add(1)
                    .ok_or("AU250 program cache miss count overflow")?;
            }
            physical_completions.push(XrtPhysicalCompletionEvidence {
                request_id: completion.request_id,
                cu_index: completion.cu_index,
                stall_code: completion.stall_code,
                dispatch_to_stall_ns: completion.dispatch_to_stall_ns,
                matrix_key_sha256: sha256_hex(completion.matrix_key),
                matrix_content_sha256: sha256_hex(completion.matrix_sha256),
                matrix_address: completion.matrix_address,
                matrix_cache_hit: completion.matrix_cache_hit,
                matrix_bytes_transferred: completion.matrix_bytes_transferred,
                trace_mode: completion.trace_mode.clone(),
                model_context_limit: completion.model_context_limit,
                trace_semantic_sha256: sha256_hex(completion.trace_semantic_sha256),
                trace_assembly_sha256: sha256_hex(completion.trace_assembly_sha256),
                replay_safe_program_sha256: sha256_hex(completion.replay_safe_program_sha256),
                trace_assembly: completion.trace_assembly.clone(),
                trace_instructions: completion.trace_instructions.clone(),
                encoded_program_sha256: sha256_hex(completion.program_sha256),
                encoded_program_hex: bytes_hex(&completion.encoded_program),
                program_address: completion.program_address,
                program_bytes: completion.program_bytes,
                program_cache_hit: completion.program_cache_hit,
            });
            stall_codes.push(completion.stall_code);
            let expected_bytes = AU250_DIM
                .checked_mul(lane_capacities[completion.cu_index])
                .and_then(|value| value.checked_mul(2))
                .ok_or("AU250 completion size overflow")?;
            if completion.output.len() != expected_bytes {
                return Err(format!(
                    "AU250 completion request {} has {} output bytes, expected {}",
                    completion.request_id,
                    completion.output.len(),
                    expected_bytes
                ));
            }
            if !all_request_ids.insert(completion.request_id) {
                return Err(format!(
                    "AU250 completion duplicated request id {}",
                    completion.request_id
                ));
            }
            completion_count = completion_count
                .checked_add(1)
                .ok_or("AU250 completion count overflow")?;
            per_cu_completions[completion.cu_index] = per_cu_completions[completion.cu_index]
                .checked_add(1)
                .ok_or("AU250 per-CU completion count overflow")?;
            request_ids.push(completion.request_id);
            completion_indices.push(planned_index);
        }
        if let Some(missing) = seen.iter().position(|present| !present) {
            return Err(format!(
                "AU250 completion missing request id {}",
                planned_wave[missing].request_id
            ));
        }

        for (completion, planned_index) in completions.iter().zip(completion_indices) {
            let planned = &planned_wave[planned_index];
            let tile =
                tile_by_coordinate[&(planned.matrix_key.row_tile, planned.matrix_key.k_tile)];
            let lanes = lane_capacities[planned.cu_index];
            let assignment_by_lane = planned
                .assignments
                .iter()
                .map(|assignment| (assignment.lane, assignment))
                .collect::<HashMap<_, _>>();
            for local_row in 0..AU250_DIM {
                for lane in 0..lanes {
                    let offset = (local_row * lanes + lane) * 2;
                    let raw = i16::from_le_bytes(
                        completion.output[offset..offset + 2]
                            .try_into()
                            .expect("validated output chunk"),
                    );
                    let Some(assignment) = assignment_by_lane.get(&lane).copied() else {
                        if raw != 0 {
                            return Err(format!(
                                "AU250 completion request {} has nonzero lane padding at row {local_row}, lane {lane}",
                                completion.request_id
                            ));
                        }
                        continue;
                    };
                    if local_row >= tile.valid_out {
                        if raw != 0 {
                            return Err(format!(
                                "AU250 completion request {} has nonzero row padding at row {local_row}, lane {lane}",
                                completion.request_id
                            ));
                        }
                        continue;
                    }
                    let (minimum, maximum) = raw_dot_bounds();
                    if raw < minimum || raw > maximum {
                        return Err(format!(
                            "AU250 raw component {raw} is outside [{minimum}, {maximum}]"
                        ));
                    }
                    let global_row = tile
                        .row_tile
                        .checked_mul(AU250_DIM)
                        .and_then(|base| base.checked_add(local_row))
                        .ok_or("AU250 global row overflow")?;
                    let component = match planned.matrix_key.kind {
                        ComponentKind::Grid => 0,
                        ComponentKind::Delta => 1,
                    };
                    let slot = raw_slot_index(
                        component,
                        assignment.batch_index,
                        global_row,
                        assignment.global_group,
                        batch,
                        rows,
                        groups,
                    )?;
                    if raw_slots[slot].replace(raw).is_some() {
                        return Err(format!(
                            "duplicate AU250 raw component for batch {}, row {}, group {}",
                            assignment.batch_index, global_row, assignment.global_group
                        ));
                    }
                    raw_min = Some(raw_min.map_or(raw, |current| current.min(raw)));
                    raw_max = Some(raw_max.map_or(raw, |current| current.max(raw)));
                }
            }
        }
    }

    if submission_count != completion_count {
        return Err(format!(
            "AU250 validated completion count {completion_count} does not match submission count {submission_count}"
        ));
    }
    if per_cu_submissions != per_cu_completions {
        return Err(format!(
            "AU250 per-CU completions {per_cu_completions:?} do not match submissions {per_cu_submissions:?}"
        ));
    }

    if let Some(missing) = raw_slots.iter().position(Option::is_none) {
        return Err(format!("missing AU250 raw component slot {missing}"));
    }
    let mut outputs = vec![0f32; output_elements];
    let mut reference_checked_components = 0u64;
    for batch_index in 0..batch {
        for row in 0..rows {
            for global_group in 0..groups {
                let grid_slot =
                    raw_slot_index(0, batch_index, row, global_group, batch, rows, groups)?;
                let delta_slot =
                    raw_slot_index(1, batch_index, row, global_group, batch, rows, groups)?;
                let grid_raw = raw_slots[grid_slot].expect("all raw slots checked");
                let delta_raw = raw_slots[delta_slot].expect("all raw slots checked");
                let q8 = captured.q8_group(batch_index, global_group)?;
                let (iq1s_d, group) = captured.matrix.group(row, global_group)?;
                let contribution = reconstruct_from_raw(
                    &group,
                    iq1s_d,
                    &q8,
                    i64::from(grid_raw) << 8,
                    i64::from(delta_raw) << 8,
                )?;
                let output = &mut outputs[batch_index * rows + row];
                *output = (*output + contribution) as f32;
                reference_checked_components = reference_checked_components
                    .checked_add(2)
                    .ok_or("AU250 reference-check count overflow")?;
            }
        }
    }
    if usize::try_from(reference_checked_components)
        .map_err(|_| "AU250 reference-check count does not fit usize")?
        != slot_count
    {
        return Err("not every AU250 component was reference checked".to_string());
    }
    if outputs.iter().any(|value| !value.is_finite()) {
        return Err("AU250 reconstructed output contains a non-finite value".to_string());
    }

    Ok(XrtIq1sResult {
        outputs,
        evidence: XrtIq1sEvidence {
            backend: "xrt",
            logical_batch: batch,
            row_tiles: rows.div_ceil(AU250_DIM),
            k_tiles: columns.div_ceil(AU250_DIM),
            submission_count,
            completion_count,
            per_cu_submissions,
            per_cu_completions,
            request_ids,
            stall_codes,
            raw_min: raw_min.ok_or("AU250 execution produced no raw components")?,
            raw_max: raw_max.ok_or("AU250 execution produced no raw components")?,
            reference_checked_components,
            comparison_status: "pass",
            resident_matrix_hits,
            resident_matrix_misses,
            resident_matrix_bytes_transferred,
            program_cache_hits,
            program_cache_misses,
            physical_completions,
            host_pack_hits,
            host_pack_misses,
            host_pack_bytes_built,
        },
    })
}

fn raw_slot_index(
    component: usize,
    batch_index: usize,
    row: usize,
    global_group: usize,
    batch: usize,
    rows: usize,
    groups: usize,
) -> Result<usize, String> {
    if component >= 2 || batch_index >= batch || row >= rows || global_group >= groups {
        return Err("AU250 raw component coordinate is out of range".to_string());
    }
    component
        .checked_mul(batch)
        .and_then(|value| value.checked_add(batch_index))
        .and_then(|value| value.checked_mul(rows))
        .and_then(|value| value.checked_add(row))
        .and_then(|value| value.checked_mul(groups))
        .and_then(|value| value.checked_add(global_group))
        .ok_or("AU250 raw component index overflow".to_string())
}

pub(crate) fn execute_captured(captured: &CapturedLaunch) -> Result<XrtIq1sResult, String> {
    let result =
        super::xrt_tmatmul::with_persistent_pool(|pool| execute_captured_with(captured, pool))?;
    append_execution_log_from_env(&result.evidence)?;
    Ok(result)
}

impl CapturedLaunchExecutor for DirectCapturedLaunchExecutor {
    type Output = XrtIq1sResult;

    fn execute(&mut self, captured: &CapturedLaunch) -> Result<Self::Output, String> {
        let result = execute_captured(captured)?;
        unsafe {
            super::iq1s_tmatmul::copy_outputs_to_cuda(captured, &result.outputs)?;
        }
        Ok(result)
    }
}

fn append_execution_log_from_env(evidence: &XrtIq1sEvidence) -> Result<(), String> {
    let Ok(path) = std::env::var("HETGPU_XRT_EXECUTION_LOG") else {
        return Ok(());
    };
    if path.trim().is_empty() {
        return Ok(());
    }
    append_execution_log(Path::new(path.trim()), evidence)
}

fn append_execution_log(path: &Path, evidence: &XrtIq1sEvidence) -> Result<(), String> {
    let record = serde_json::json!({
        "event": "au250_xrt_iq1s_completed",
        "evidence": evidence,
    });
    let mut line = serde_json::to_string(&record)
        .map_err(|error| format!("serialize XRT execution log {}: {error}", path.display()))?;
    line.push('\n');
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| format!("open XRT execution log {}: {error}", path.display()))?;
    file.write_all(line.as_bytes())
        .map_err(|error| format!("write XRT execution log {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::super::iq1s_tmatmul::{
        capture_from_host, raw_component_dots, reconstruct_from_raw, ComponentKind, GridTable,
        LogicalLaunch, MatrixSource, GRID_ENTRIES, IQ1S_BLOCK_BYTES, Q8_1_MMQ_BYTES,
    };
    use super::super::xrt_tmatmul::{XrtWaveCompletion, XrtWaveJob};
    use super::*;
    use std::sync::Arc;

    fn valid_strict_completion() -> XrtWaveCompletion {
        let labels = HashMap::from([
            ("PARAM_MATRIX".to_string(), 0x1000),
            ("PARAM_INPUT".to_string(), 0x2000),
            ("PARAM_OUTPUT".to_string(), 0x3000),
        ]);
        let selected = crate::r#impl::iq1s_trace::build_selected_trace(
            "compiler",
            crate::r#impl::iq1s_trace::QWEN_MODEL_CONTEXT_LIMIT,
            &labels,
            4,
        )
        .unwrap();
        let program_sha256 = Sha256::digest(&selected.program).into();
        XrtWaveCompletion {
            request_id: 7,
            cu_index: 0,
            stall_code: 1,
            output: vec![0; AU250_DIM * 9 * 2],
            dispatch_to_stall_ns: 10,
            program_bytes: selected.program.len(),
            matrix_key: [0x11; 32],
            matrix_sha256: [0x22; 32],
            matrix_address: 0x1000,
            matrix_cache_hit: true,
            matrix_bytes_transferred: 0,
            program_address: 0x4000,
            program_sha256,
            program_cache_hit: true,
            encoded_program: selected.program,
            trace_mode: selected.selected_kind.as_str().to_string(),
            model_context_limit: selected.model_context_limit,
            trace_semantic_sha256: selected.semantic_sha256,
            trace_assembly_sha256: selected.assembly_sha256,
            replay_safe_program_sha256: selected.selected_sha256,
            trace_assembly: selected.assembly,
            trace_instructions: selected.instructions,
        }
    }

    #[test]
    fn strict_completion_binding_rejects_logged_only_or_mutated_trace_evidence() {
        let valid = valid_strict_completion();
        validate_completion_binding(
            &valid,
            valid.matrix_key,
            valid.matrix_sha256,
            true,
            Some("compiler"),
        )
        .unwrap();

        let mut changed_program = valid.clone();
        changed_program.encoded_program[0] ^= 1;
        assert!(validate_completion_binding(
            &changed_program,
            valid.matrix_key,
            valid.matrix_sha256,
            true,
            Some("compiler"),
        )
        .unwrap_err()
        .contains("program body/hash/length"));

        let mut logged_only = valid.clone();
        logged_only.trace_assembly.push_str("nop\n");
        assert!(validate_completion_binding(
            &logged_only,
            valid.matrix_key,
            valid.matrix_sha256,
            true,
            Some("compiler"),
        )
        .unwrap_err()
        .contains("trace body"));

        let mut wrong_mode = valid.clone();
        wrong_mode.trace_mode = "handwritten".to_string();
        assert!(validate_completion_binding(
            &wrong_mode,
            valid.matrix_key,
            valid.matrix_sha256,
            true,
            Some("compiler"),
        )
        .unwrap_err()
        .contains("trace mode"));

        assert!(validate_completion_binding(
            &valid,
            [0x99; 32],
            valid.matrix_sha256,
            true,
            Some("compiler"),
        )
        .unwrap_err()
        .contains("matrix identity"));
    }

    fn kimi_signature() -> GgmlType19Signature {
        GgmlType19Signature {
            kernel: "mul_mat_q".into(),
            ne00: 7168,
            ne01: 2048,
            stride01: 28,
            ne10: 7168,
            ne11: 1,
            stride11: 8064,
            ne0: 2048,
        }
    }

    #[test]
    fn kimi_7168_by_2048_uses_seven_k_tiles_and_two_row_tiles() {
        let geometry = plan_au250_tiles(&kimi_signature()).unwrap();
        assert_eq!(geometry.iter().map(|tile| tile.k_tile).max(), Some(6));
        assert_eq!(geometry.iter().map(|tile| tile.row_tile).max(), Some(1));
        assert_eq!(geometry.last().unwrap().valid_in, 1024);
    }

    #[test]
    fn lane_input_is_dimension_major_and_group_sparse() {
        let q8 = Q8_1Block {
            d: 1.0,
            s: 0.0,
            qs: [7; 32],
        };
        let bytes = pack_lane_input(9, &[(3, 5, q8)]).unwrap();
        let raw = bytes
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(raw[5 * 32 * 9 + 3], 7);
        assert_eq!(raw.iter().filter(|value| **value == 7).count(), 32);
    }

    #[test]
    fn direct_q8_raw_values_keep_every_group_dot_in_i16() {
        assert_eq!(raw_dot_bounds(), (-4096, 4096));
    }

    #[test]
    fn component_matrix_packs_grid_and_delta_trits_with_zero_padding() {
        let signature = GgmlType19Signature {
            kernel: "mul_mat_q".into(),
            ne00: 1024,
            ne01: 1,
            stride01: 4,
            ne10: 1024,
            ne11: 1,
            stride11: 1,
            ne0: 1,
        };
        let mut grid: GridTable = [[0; 8]; 2048];
        grid[0] = [-1, 0, 1, -1, 0, 1, -1, 1];
        let source = MatrixSource::new(
            signature.clone(),
            Arc::from(vec![0_u8; signature.matrix_storage_bytes().unwrap()]),
            Arc::new(grid),
        )
        .unwrap();
        let tile = plan_au250_tiles(&signature).unwrap()[0];

        let packed_grid = pack_component_matrix(&source, tile, ComponentKind::Grid).unwrap();
        let first_group = (0..32)
            .map(|index| decode_trit(&packed_grid, index))
            .collect::<Vec<_>>();
        assert_eq!(first_group, [-1, 0, 1, -1, 0, 1, -1, 1].repeat(4));
        assert!(packed_grid[AU250_DIM / 4..].iter().all(|byte| *byte == 0));

        let packed_delta = pack_component_matrix(&source, tile, ComponentKind::Delta).unwrap();
        assert!((0..32).all(|index| decode_trit(&packed_delta, index) == 1));
    }

    #[test]
    fn jobs_are_batch_major_and_fill_each_cu_once_per_wave() {
        let mut signature = kimi_signature();
        signature.ne00 = 1024;
        signature.ne10 = 1024;
        signature.ne01 = 1024;
        signature.ne0 = 1024;
        signature.stride01 = 4;
        signature.ne11 = 2;
        signature.stride11 = 2;
        let waves = plan_au250_jobs(&signature, &[9, 9, 9, 6]).unwrap();
        let first = &waves[0];
        assert_eq!(first.len(), 4);
        assert_eq!(
            first[0]
                .assignments
                .iter()
                .map(|assignment| (
                    assignment.lane,
                    assignment.batch_index,
                    assignment.global_group,
                ))
                .collect::<Vec<_>>(),
            (0..9).map(|group| (group, 0, group)).collect::<Vec<_>>()
        );
        assert_eq!(first[3].assignments.len(), 6);
        assert_eq!(first[3].cu_index, 3);
        assert!(waves.iter().all(|wave| wave
            .iter()
            .map(|job| job.cu_index)
            .collect::<std::collections::HashSet<_>>()
            .len()
            == wave.len()));
    }

    fn decode_trit(packed: &[u8], element: usize) -> i8 {
        match (packed[element / 4] >> (2 * (element % 4))) & 3 {
            0 => 0,
            1 => 1,
            3 => -1,
            code => panic!("invalid ternary code {code}"),
        }
    }

    #[test]
    fn executor_demultiplexes_grid_delta_and_matches_reference_bits() {
        let captured = two_k_tile_two_row_tile_fixture();
        let expected = software_reference(&captured).unwrap();
        let mut backend = CpuDotWaveExecutor::new(vec![9, 9, 9, 6]);
        let result = execute_captured_with(&captured, &mut backend).unwrap();
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert!(result.evidence.submission_count > 4);
        assert_eq!(result.evidence.backend, "xrt");
        assert_eq!(result.evidence.comparison_status, "pass");
        assert_eq!(
            result.evidence.reference_checked_components,
            2 * 1030 * (2048 / 32)
        );
    }

    #[test]
    fn sampled_output_comparison_retains_vectors_and_rejects_drift() {
        let captured = small_fixture();
        let reference = software_reference(&captured).unwrap();
        let comparison =
            compare_outputs_with_reference(&captured, &reference, 1.0e-4, 1.0e-3).unwrap();
        assert_eq!(comparison.status, "pass");
        assert_eq!(comparison.reference_outputs, reference);
        assert_eq!(comparison.actual_outputs, reference);
        assert_eq!(comparison.checked_elements, comparison.actual_outputs.len());

        let mut drifted = comparison.actual_outputs.clone();
        drifted[0] += 1.0;
        let error =
            compare_outputs_with_reference(&captured, &drifted, 1.0e-4, 1.0e-3).unwrap_err();
        assert!(error.contains("outside tolerance"), "{error}");
    }

    #[test]
    #[ignore = "requires HETGPU_XRT_AU250_IQ1S_TEST=1 and live persistent IQ1_S AU250"]
    fn au250_iq1s_two_by_two_tiles_match_reference() {
        assert_eq!(
            std::env::var("HETGPU_XRT_AU250_IQ1S_TEST").as_deref(),
            Ok("1"),
            "set HETGPU_XRT_AU250_IQ1S_TEST=1 only inside the guarded AU250 wrapper"
        );
        let captured = two_k_tile_two_row_tile_fixture();
        let expected = software_reference(&captured).unwrap();
        let actual = execute_captured(&captured).unwrap();
        assert_eq!(
            actual
                .outputs
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert!(
            actual
                .evidence
                .per_cu_submissions
                .iter()
                .filter(|submissions| **submissions > 0)
                .count()
                == 4,
            "tiled proof must exercise all four physical CUs: {:?}",
            actual.evidence.per_cu_submissions
        );
        eprintln!(
            "AU250 XRT IQ1_S PASS: rows={} columns={} row_tiles={} k_tiles={} submissions={} per_cu={:?}",
            captured.launch.signature.ne0,
            captured.launch.signature.ne00,
            actual.evidence.row_tiles,
            actual.evidence.k_tiles,
            actual.evidence.submission_count,
            actual.evidence.per_cu_submissions,
        );
    }

    #[test]
    fn executor_preserves_batch_two_output_order() {
        let captured = captured_fixture(1024, 3, 2);
        let expected = software_reference(&captured).unwrap();
        let mut backend = CpuDotWaveExecutor::new(vec![9, 9, 9, 6]);
        let result = execute_captured_with(&captured, &mut backend).unwrap();
        assert_eq!(result.outputs.len(), 6);
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(result.evidence.logical_batch, 2);
    }

    #[test]
    fn executor_evidence_accounts_for_reordered_physical_completions() {
        let captured = captured_fixture(1024, 1, 1);
        let mut backend = CpuDotWaveExecutor::new(vec![9, 9, 9, 6]);
        let result = execute_captured_with(&captured, &mut backend).unwrap();
        assert_eq!(result.evidence.submission_count, 8);
        assert_eq!(result.evidence.completion_count, 8);
        assert_eq!(result.evidence.per_cu_submissions, vec![2, 2, 2, 2]);
        assert_eq!(result.evidence.per_cu_completions, vec![2, 2, 2, 2]);
        assert_eq!(result.evidence.request_ids.len(), 8);
        assert_eq!(
            result
                .evidence
                .request_ids
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len(),
            8
        );
        assert_eq!(
            result.evidence.stall_codes.len(),
            result.evidence.completion_count as usize
        );
    }

    #[test]
    fn missing_or_duplicate_completion_fails_before_output_copy() {
        let captured = small_fixture();
        let mut missing = FaultingWaveExecutor::new(CompletionFault::Missing);
        assert!(execute_captured_with(&captured, &mut missing)
            .unwrap_err()
            .contains("completion"));

        let captured = captured_fixture(1024, 1, 1);
        let mut duplicate = FaultingWaveExecutor::new(CompletionFault::Duplicate);
        assert!(execute_captured_with(&captured, &mut duplicate)
            .unwrap_err()
            .contains("duplicated request id"));
    }

    #[test]
    fn wrong_cu_completion_fails_before_output_copy() {
        let captured = small_fixture();
        let mut backend = FaultingWaveExecutor::new(CompletionFault::WrongCu);
        assert!(execute_captured_with(&captured, &mut backend)
            .unwrap_err()
            .contains("returned CU"));
    }

    #[test]
    fn nonzero_padding_is_rejected() {
        let captured = small_fixture();
        let mut backend = PaddingWaveExecutor::new();
        assert!(execute_captured_with(&captured, &mut backend)
            .unwrap_err()
            .contains("padding"));
    }

    #[test]
    fn backend_error_is_propagated_without_a_result() {
        let captured = small_fixture();
        let mut backend = ErrorWaveExecutor;
        let error = execute_captured_with(&captured, &mut backend).unwrap_err();
        assert!(error.contains("stall timeout"), "{error}");
    }

    #[test]
    fn matrix_cache_does_not_evict_an_inflight_arc() {
        let captured = small_fixture();
        let identity = MatrixCacheIdentity {
            matrix_ptr: captured.launch.matrix_ptr,
            signature: captured.launch.signature.clone(),
            allocation_generation: captured.launch.allocation_generation,
            content_hash: captured.launch.content_hash,
        };
        let first_key = PackedMatrixKey {
            identity: identity.clone(),
            tile: Au250MatrixKey {
                row_tile: 0,
                k_tile: 0,
                kind: ComponentKind::Grid,
            },
        };
        let second_key = PackedMatrixKey {
            identity,
            tile: Au250MatrixKey {
                row_tile: 0,
                k_tile: 0,
                kind: ComponentKind::Delta,
            },
        };
        let mut cache = PackedMatrixCache {
            capacity_bytes: 8,
            resident_bytes: 0,
            clock: 0,
            entries: HashMap::new(),
        };
        let in_flight = cache
            .get_or_insert(first_key.clone(), || Ok(vec![1_u8; 8]))
            .unwrap();
        assert!(cache
            .get_or_insert(second_key.clone(), || Ok(vec![2_u8; 8]))
            .unwrap_err()
            .contains("in-flight"));
        drop(in_flight);
        let second = cache
            .get_or_insert(second_key.clone(), || Ok(vec![2_u8; 8]))
            .unwrap();
        assert_eq!(&*second, &[2_u8; 8]);
        assert!(!cache.entries.contains_key(&first_key));
        assert!(cache.entries.contains_key(&second_key));
        assert_eq!(cache.resident_bytes, 8);
    }

    #[test]
    fn matrix_cache_capacity_is_bounded_and_rejects_zero() {
        assert_eq!(
            parse_matrix_cache_capacity(None).unwrap(),
            512 * 1024 * 1024
        );
        assert_eq!(parse_matrix_cache_capacity(Some("262144")).unwrap(), 262144);
        assert!(parse_matrix_cache_capacity(Some("0")).is_err());
        assert!(parse_matrix_cache_capacity(Some("not-a-size")).is_err());
    }

    #[test]
    fn execution_log_is_jsonl_and_open_failures_are_strict() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let evidence = XrtIq1sEvidence {
            backend: "xrt",
            logical_batch: 1,
            row_tiles: 1,
            k_tiles: 1,
            submission_count: 2,
            completion_count: 2,
            per_cu_submissions: vec![1, 1],
            per_cu_completions: vec![1, 1],
            request_ids: vec![0, 1],
            stall_codes: vec![1, 1],
            raw_min: -3,
            raw_max: 4,
            reference_checked_components: 16,
            comparison_status: "pass",
            resident_matrix_hits: 1,
            resident_matrix_misses: 1,
            resident_matrix_bytes_transferred: AU250_MATRIX_BYTES as u64,
            program_cache_hits: 1,
            program_cache_misses: 1,
            physical_completions: Vec::new(),
            host_pack_hits: 1,
            host_pack_misses: 1,
            host_pack_bytes_built: AU250_MATRIX_BYTES as u64,
        };
        append_execution_log(file.path(), &evidence).unwrap();
        let text = std::fs::read_to_string(file.path()).unwrap();
        let value: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(value["event"], "au250_xrt_iq1s_completed");
        assert_eq!(value["evidence"]["backend"], "xrt");
        assert_eq!(value["evidence"]["stall_codes"], serde_json::json!([1, 1]));
        let directory = tempfile::tempdir().unwrap();
        assert!(append_execution_log(directory.path(), &evidence).is_err());
    }

    fn captured_fixture(ne00: u64, ne01: u64, batch: u64) -> CapturedLaunch {
        assert!(ne00.is_multiple_of(256));
        let blocks_per_row = usize::try_from(ne00 / 256).unwrap();
        let signature = GgmlType19Signature {
            kernel: "mul_mat_q".into(),
            ne00,
            ne01,
            stride01: blocks_per_row as u64,
            ne10: ne00,
            ne11: batch,
            stride11: batch,
            ne0: ne01,
        };
        let mut grid = [[0_i8; 8]; GRID_ENTRIES];
        for (index, values) in grid.iter_mut().enumerate() {
            for (column, value) in values.iter_mut().enumerate() {
                *value = [-1, 0, 1][(index + column) % 3];
            }
        }
        let mut one_block = [0_u8; IQ1S_BLOCK_BYTES];
        one_block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        let mut matrix = Vec::with_capacity(ne01 as usize * blocks_per_row * IQ1S_BLOCK_BYTES);
        for row in 0..ne01 as usize {
            for block in 0..blocks_per_row {
                one_block[2] = ((row + block) & 0xff) as u8;
                matrix.extend_from_slice(&one_block);
            }
        }
        let records = usize::try_from((ne00 / 128 - 1) * batch + batch).unwrap();
        let mut activations = vec![0_u8; records * Q8_1_MMQ_BYTES];
        for (record_index, record) in activations.chunks_exact_mut(Q8_1_MMQ_BYTES).enumerate() {
            for pair in 0..4 {
                record[pair * 4..pair * 4 + 2].copy_from_slice(&0x3c00_u16.to_le_bytes());
                record[pair * 4 + 2..pair * 4 + 4].copy_from_slice(&0x3c00_u16.to_le_bytes());
                for (index, value) in record[16 + pair * 32..16 + (pair + 1) * 32]
                    .iter_mut()
                    .enumerate()
                {
                    *value = record_index
                        .wrapping_mul(131)
                        .wrapping_add(pair * 37)
                        .wrapping_add(index * 17) as u8;
                }
            }
        }
        capture_from_host(
            LogicalLaunch {
                matrix_ptr: 0x1000,
                activation_ptr: 0x2000,
                output_ptr: 0x3000,
                allocation_generation: 1,
                content_hash: [0x5a; 32],
                signature,
            },
            &matrix,
            &activations,
            &grid,
        )
        .unwrap()
    }

    fn small_fixture() -> CapturedLaunch {
        captured_fixture(256, 1, 1)
    }

    fn two_k_tile_two_row_tile_fixture() -> CapturedLaunch {
        captured_fixture(2048, 1030, 1)
    }

    fn software_reference(captured: &CapturedLaunch) -> Result<Vec<f32>, String> {
        let batch =
            usize::try_from(captured.launch.signature.ne11).map_err(|_| "batch overflow")?;
        let rows = usize::try_from(captured.launch.signature.ne0).map_err(|_| "row overflow")?;
        let groups =
            usize::try_from(captured.launch.signature.ne00 / 32).map_err(|_| "group overflow")?;
        let mut outputs = vec![0_f32; batch.checked_mul(rows).ok_or("output overflow")?];
        for batch_index in 0..batch {
            for row in 0..rows {
                for global_group in 0..groups {
                    let q8 = captured.q8_group(batch_index, global_group)?;
                    let (d, group) = captured.matrix.group(row, global_group)?;
                    let (grid, delta) = raw_component_dots(&group, &q8);
                    let contribution = reconstruct_from_raw(&group, d, &q8, grid << 8, delta << 8)?;
                    let output = &mut outputs[batch_index * rows + row];
                    *output = (*output + contribution) as f32;
                }
            }
        }
        Ok(outputs)
    }

    struct CpuDotWaveExecutor {
        capacities: Vec<usize>,
    }

    impl CpuDotWaveExecutor {
        fn new(capacities: Vec<usize>) -> Self {
            Self { capacities }
        }
    }

    impl Au250WaveExecutor for CpuDotWaveExecutor {
        fn lane_capacities(&self) -> Vec<usize> {
            self.capacities.clone()
        }

        fn run_wave(&mut self, jobs: Vec<XrtWaveJob>) -> Result<Vec<XrtWaveCompletion>, String> {
            let mut completions = Vec::with_capacity(jobs.len());
            for job in jobs {
                let lanes = *self
                    .capacities
                    .get(job.cu_index)
                    .ok_or("mock job selects unknown CU")?;
                if job.matrix.len() != AU250_MATRIX_BYTES
                    || job.input.len() != AU250_DIM * lanes * 2
                {
                    return Err("mock job has invalid payload length".to_string());
                }
                let mut sparse = vec![Vec::<(usize, i16)>::new(); lanes];
                for dimension in 0..AU250_DIM {
                    for lane in 0..lanes {
                        let offset = (dimension * lanes + lane) * 2;
                        let quant =
                            i16::from_le_bytes(job.input[offset..offset + 2].try_into().unwrap());
                        if quant != 0 {
                            sparse[lane].push((dimension, quant));
                        }
                    }
                }
                let mut output = vec![0_u8; AU250_DIM * lanes * 2];
                for row in 0..AU250_DIM {
                    for (lane, nonzero) in sparse.iter().enumerate() {
                        let mut dot = 0i32;
                        for &(dimension, quant) in nonzero {
                            dot += i32::from(decode_trit(&job.matrix, row * AU250_DIM + dimension))
                                * i32::from(quant);
                        }
                        let raw = i16::try_from(dot).map_err(|_| "mock raw dot overflow")?;
                        let offset = (row * lanes + lane) * 2;
                        output[offset..offset + 2].copy_from_slice(&raw.to_le_bytes());
                    }
                }
                completions.push(XrtWaveCompletion {
                    request_id: job.request_id,
                    cu_index: job.cu_index,
                    stall_code: 1,
                    output,
                    dispatch_to_stall_ns: 1,
                    program_bytes: 96,
                    matrix_key: job.matrix_key,
                    matrix_sha256: job.matrix_sha256,
                    matrix_address: 0x1000,
                    matrix_cache_hit: false,
                    matrix_bytes_transferred: job.matrix.len(),
                    program_address: 0x2000,
                    program_sha256: Sha256::digest(vec![0x77; 96]).into(),
                    program_cache_hit: false,
                    encoded_program: vec![0x77; 96],
                    trace_mode: "compiler".to_string(),
                    model_context_limit: 262_144,
                    trace_semantic_sha256: [0x44; 32],
                    trace_assembly_sha256: [0x55; 32],
                    replay_safe_program_sha256: [0x66; 32],
                    trace_assembly: "fixture".to_string(),
                    trace_instructions: vec![vec!["stall".to_string()]],
                });
            }
            completions.reverse();
            Ok(completions)
        }
    }

    #[derive(Clone, Copy)]
    enum CompletionFault {
        Missing,
        Duplicate,
        WrongCu,
    }

    struct FaultingWaveExecutor {
        inner: CpuDotWaveExecutor,
        mode: CompletionFault,
    }

    impl FaultingWaveExecutor {
        fn new(mode: CompletionFault) -> Self {
            Self {
                inner: CpuDotWaveExecutor::new(vec![9, 9, 9, 6]),
                mode,
            }
        }
    }

    impl Au250WaveExecutor for FaultingWaveExecutor {
        fn lane_capacities(&self) -> Vec<usize> {
            self.inner.lane_capacities()
        }

        fn run_wave(&mut self, jobs: Vec<XrtWaveJob>) -> Result<Vec<XrtWaveCompletion>, String> {
            let mut completions = self.inner.run_wave(jobs)?;
            match self.mode {
                CompletionFault::Missing => {
                    completions.pop();
                }
                CompletionFault::Duplicate => {
                    let duplicate = completions
                        .first()
                        .cloned()
                        .ok_or("no completion to duplicate")?;
                    let second = completions
                        .get_mut(1)
                        .ok_or("duplicate fault requires two completions")?;
                    *second = duplicate;
                }
                CompletionFault::WrongCu => {
                    let completion = completions
                        .first_mut()
                        .ok_or("no completion for CU fault")?;
                    completion.cu_index = (completion.cu_index + 1) % 4;
                }
            }
            Ok(completions)
        }
    }

    struct PaddingWaveExecutor {
        inner: CpuDotWaveExecutor,
    }

    impl PaddingWaveExecutor {
        fn new() -> Self {
            Self {
                inner: CpuDotWaveExecutor::new(vec![9, 9, 9, 6]),
            }
        }
    }

    impl Au250WaveExecutor for PaddingWaveExecutor {
        fn lane_capacities(&self) -> Vec<usize> {
            self.inner.lane_capacities()
        }

        fn run_wave(&mut self, jobs: Vec<XrtWaveJob>) -> Result<Vec<XrtWaveCompletion>, String> {
            let mut completions = self.inner.run_wave(jobs)?;
            let output = &mut completions
                .first_mut()
                .ok_or("no completion for padding fault")?
                .output;
            let last = output.len() - 2;
            output[last..].copy_from_slice(&1_i16.to_le_bytes());
            Ok(completions)
        }
    }

    struct ErrorWaveExecutor;

    impl Au250WaveExecutor for ErrorWaveExecutor {
        fn lane_capacities(&self) -> Vec<usize> {
            vec![9, 9, 9, 6]
        }

        fn run_wave(&mut self, _jobs: Vec<XrtWaveJob>) -> Result<Vec<XrtWaveCompletion>, String> {
            Err("stall timeout from poisoned backend".to_string())
        }
    }
}
