use super::iq1s_layer_abi::{
    iq1s_command_crc32, Iq1sCommand, IQ1S_ABI_VERSION, IQ1S_COMMAND_ARENA_OFFSET_OFFSET,
    IQ1S_COMMAND_BYTES, IQ1S_COMMAND_INPUT_OFFSET_OFFSET, IQ1S_COMMAND_MAGIC,
    IQ1S_COMMAND_OUTPUT_OFFSET_OFFSET, IQ1S_COMMAND_TOKEN_MAP_OFFSET_OFFSET, IQ1S_PHASE_A,
    IQ1S_PHASE_B, IQ1S_ROLE_DOWN, IQ1S_ROLE_GATE, IQ1S_ROLE_UP, IQ1S_WEIGHT_FORMAT_IQ1_S,
};
use super::iq1s_trace::{build_selected_trace_for_shape, TraceKind, QWEN_MODEL_CONTEXT_LIMIT};
use super::iq1s_weight_arena::{ArenaShard, ARENA_ALIGNMENT, ARENA_BANK_COUNT};
use super::iq1s_weight_registry::Iq1sExpertRole;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};

const MAX_BATCH: usize = 16;
const Q8_1_MMQ_VALUES: u64 = 128;
const Q8_1_MMQ_BYTES: u64 = 144;
const F32_BYTES: u64 = 4;
const OUTPUT_SLAB_BYTES: u64 = 256 * 1024 * 1024;
const TOKEN_MAP_SLAB_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum LayerPhase {
    PhaseA,
    PhaseB,
}

impl LayerPhase {
    fn abi(self) -> u16 {
        match self {
            Self::PhaseA => IQ1S_PHASE_A as u16,
            Self::PhaseB => IQ1S_PHASE_B as u16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum RelocationSource {
    ArenaOffset,
    InputOffset,
    OutputOffset,
    TokenMapOffset,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct Relocation {
    pub(crate) command_index: u32,
    pub(crate) field_offset: u16,
    pub(crate) source: RelocationSource,
    pub(crate) addend: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ActivationRange {
    pub(crate) cuda_ptr: usize,
    pub(crate) slab_offset: u64,
    pub(crate) bytes: u32,
    pub(crate) stream: usize,
    pub(crate) source_identity_sha256: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SemanticIq1sCommand {
    pub(crate) layer_id: u32,
    pub(crate) phase: LayerPhase,
    pub(crate) role: Iq1sExpertRole,
    pub(crate) expert_id: u16,
    pub(crate) lane_mask: u16,
    pub(crate) token_ids: Vec<u32>,
    pub(crate) input_offset: u64,
    pub(crate) output_offset: u64,
    pub(crate) token_map_offset: u64,
    pub(crate) row_shard: ArenaShard,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub(crate) struct ExpandedIq1sCounts {
    pub(crate) blocks: u64,
    pub(crate) grid_passes: u64,
    pub(crate) delta_passes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LayerProgram {
    pub(crate) kind: TraceKind,
    pub(crate) assembly: String,
    pub(crate) encoded: Vec<u8>,
    pub(crate) relocations: Vec<Relocation>,
    pub(crate) semantic_sha256: [u8; 32],
    pub(crate) assembly_sha256: [u8; 32],
    pub(crate) expanded: ExpandedIq1sCounts,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LayerPhasePlan {
    pub(crate) transaction_id: u64,
    pub(crate) phase: LayerPhase,
    pub(crate) commands: Vec<SemanticIq1sCommand>,
    pub(crate) activations: Vec<ActivationRange>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompiledLayerPhase {
    pub(crate) transaction_id: u64,
    pub(crate) phase: LayerPhase,
    pub(crate) programs: [LayerProgram; ARENA_BANK_COUNT],
    pub(crate) commands: [Vec<Iq1sCommand>; ARENA_BANK_COUNT],
    pub(crate) activations: Vec<ActivationRange>,
    pub(crate) semantic_sha256: [u8; 32],
}

fn role_abi(role: Iq1sExpertRole) -> Result<u16, String> {
    match role {
        Iq1sExpertRole::Gate => Ok(IQ1S_ROLE_GATE as u16),
        Iq1sExpertRole::Up => Ok(IQ1S_ROLE_UP as u16),
        Iq1sExpertRole::Down => Ok(IQ1S_ROLE_DOWN as u16),
        Iq1sExpertRole::GateUp => {
            Err("fused gate_up is not an offloaded Qwen IQ1_S role".to_string())
        }
    }
}

fn role_input_columns(role: Iq1sExpertRole) -> Result<u32, String> {
    match role {
        Iq1sExpertRole::Gate | Iq1sExpertRole::Up => Ok(4096),
        Iq1sExpertRole::Down => Ok(1024),
        Iq1sExpertRole::GateUp => {
            Err("fused gate_up is not an offloaded Qwen IQ1_S role".to_string())
        }
    }
}

fn role_from_abi(role: u16) -> Result<Iq1sExpertRole, String> {
    match u32::from(role) {
        IQ1S_ROLE_GATE => Ok(Iq1sExpertRole::Gate),
        IQ1S_ROLE_UP => Ok(Iq1sExpertRole::Up),
        IQ1S_ROLE_DOWN => Ok(Iq1sExpertRole::Down),
        _ => Err(format!("invalid Qwen IQ1_S command role {role}")),
    }
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

fn lane_mask(token_ids: &[u32]) -> Result<u16, String> {
    if token_ids.is_empty() || token_ids.len() > MAX_BATCH {
        return Err("Qwen IQ1_S command must contain 1..16 token lanes".to_string());
    }
    let mut mask = 0u16;
    for token_id in token_ids {
        let lane = u16::try_from(*token_id)
            .ok()
            .filter(|lane| usize::from(*lane) < MAX_BATCH)
            .ok_or("Qwen IQ1_S token lane is outside 0..15")?;
        let bit = 1u16 << lane;
        if mask & bit != 0 {
            return Err("Qwen IQ1_S command repeats a token lane".to_string());
        }
        mask |= bit;
    }
    Ok(mask)
}

fn validate_activation_ranges(ranges: &[ActivationRange]) -> Result<(), String> {
    if ranges.is_empty() {
        return Err("Qwen IQ1_S layer phase has no activation ranges".to_string());
    }
    let mut source_bindings = HashMap::new();
    let mut slab_ranges = BTreeSet::new();
    for range in ranges {
        if range.cuda_ptr < 0x1000
            || range.stream == 0
            || range.bytes == 0
            || range.source_identity_sha256 == [0; 32]
            || range.slab_offset % ARENA_ALIGNMENT != 0
        {
            return Err("Qwen IQ1_S activation range is invalid or unaligned".to_string());
        }
        let binding = (range.slab_offset, range.bytes, range.stream);
        if let Some(previous) = source_bindings.insert(range.source_identity_sha256, binding) {
            if previous != binding {
                return Err(
                    "Qwen IQ1_S activation identity maps to inconsistent ranges".to_string(),
                );
            }
        }
        let end = range
            .slab_offset
            .checked_add(u64::from(range.bytes))
            .ok_or("Qwen IQ1_S activation slab range overflow")?;
        slab_ranges.insert((range.slab_offset, end));
    }
    let slab_ranges = slab_ranges.into_iter().collect::<Vec<_>>();
    if slab_ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err("Qwen IQ1_S activation slab ranges overlap".to_string());
    }
    Ok(())
}

fn validate_semantic_command(
    command: &SemanticIq1sCommand,
    phase: LayerPhase,
) -> Result<u16, String> {
    if command.phase != phase || !role_matches_phase(command.role, phase) {
        return Err("Qwen IQ1_S command role does not match its layer phase".to_string());
    }
    if command.row_shard.tensor.layer != command.layer_id
        || command.row_shard.tensor.role != command.role
        || command.row_shard.expert != command.expert_id
    {
        return Err("Qwen IQ1_S command does not match its registered arena shard".to_string());
    }
    if usize::from(command.row_shard.bank) >= ARENA_BANK_COUNT
        || command.row_shard.sha256 == [0; 32]
        || command.row_shard.row_count == 0
        || command.row_shard.offset % ARENA_ALIGNMENT != 0
    {
        return Err("Qwen IQ1_S command arena shard is invalid".to_string());
    }
    let input_columns = role_input_columns(command.role)?;
    if command.row_shard.tensor.ne[0] != u64::from(input_columns)
        || command.row_shard.tensor.ne[2] != 512
        || !matches!(
            (command.role, command.row_shard.row_count),
            (Iq1sExpertRole::Gate | Iq1sExpertRole::Up, 256) | (Iq1sExpertRole::Down, 1024)
        )
    {
        return Err("Qwen IQ1_S command has an unsupported matrix shape".to_string());
    }
    let derived_mask = lane_mask(&command.token_ids)?;
    if command.lane_mask != derived_mask {
        return Err("Qwen IQ1_S command lane mask does not match its token IDs".to_string());
    }
    Ok(derived_mask)
}

fn ranges_overlap(left: (u64, u64), right: (u64, u64)) -> bool {
    left.0 < right.1 && right.0 < left.1
}

fn semantic_buffer_ranges(
    command: &SemanticIq1sCommand,
) -> Result<((u64, u64), (u64, u64), (u64, u64)), String> {
    let lane_count = u64::try_from(command.token_ids.len())
        .map_err(|_| "Qwen IQ1_S lane count does not fit u64")?;
    let input_bytes = (u64::from(role_input_columns(command.role)?) / Q8_1_MMQ_VALUES)
        .checked_mul(Q8_1_MMQ_BYTES)
        .and_then(|value| value.checked_mul(lane_count))
        .ok_or("Qwen IQ1_S activation byte count overflow")?;
    let output_bytes = u64::from(command.row_shard.row_count)
        .checked_mul(F32_BYTES)
        .and_then(|value| value.checked_mul(lane_count))
        .ok_or("Qwen IQ1_S output byte count overflow")?;
    let token_bytes = lane_count
        .checked_mul(4)
        .ok_or("Qwen IQ1_S token-map byte count overflow")?;
    let input_end = command
        .input_offset
        .checked_add(input_bytes)
        .ok_or("Qwen IQ1_S activation slab range overflow")?;
    let output_end = command
        .output_offset
        .checked_add(output_bytes)
        .ok_or("Qwen IQ1_S output slab range overflow")?;
    let token_end = command
        .token_map_offset
        .checked_add(token_bytes)
        .ok_or("Qwen IQ1_S token-map slab range overflow")?;
    if command.input_offset % 16 != 0
        || command.output_offset % F32_BYTES != 0
        || command.token_map_offset % 4 != 0
        || output_end > OUTPUT_SLAB_BYTES
        || token_end > TOKEN_MAP_SLAB_BYTES
    {
        return Err("Qwen IQ1_S command buffer range is unaligned or exceeds its slab".to_string());
    }
    Ok((
        (command.input_offset, input_end),
        (command.output_offset, output_end),
        (command.token_map_offset, token_end),
    ))
}

fn validate_semantic_buffer_bindings(plan: &LayerPhasePlan) -> Result<(), String> {
    let activation_ranges = plan
        .activations
        .iter()
        .map(|range| {
            range
                .slab_offset
                .checked_add(u64::from(range.bytes))
                .map(|end| (range.slab_offset, end))
                .ok_or("Qwen IQ1_S activation slab range overflow")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut by_bank: [Vec<(&SemanticIq1sCommand, (u64, u64), (u64, u64), (u64, u64))>;
        ARENA_BANK_COUNT] = std::array::from_fn(|_| Vec::new());
    for command in &plan.commands {
        if usize::from(command.row_shard.bank) >= ARENA_BANK_COUNT {
            return Err("Qwen IQ1_S command references an invalid U250 bank".to_string());
        }
        let (input, output, token) = semantic_buffer_ranges(command)?;
        if !activation_ranges
            .iter()
            .any(|range| input.0 >= range.0 && input.1 <= range.1)
        {
            return Err(
                "Qwen IQ1_S command input is not contained in the activation manifest".to_string(),
            );
        }
        by_bank[usize::from(command.row_shard.bank)].push((command, input, output, token));
    }
    for commands in by_bank {
        for left in 0..commands.len() {
            for right in left + 1..commands.len() {
                let (left_command, left_input, left_output, left_token) = commands[left];
                let (right_command, right_input, right_output, right_token) = commands[right];
                if ranges_overlap(left_output, right_output) {
                    return Err("Qwen IQ1_S output ranges overlap within one U250 bank".to_string());
                }
                if ranges_overlap(left_input, right_input)
                    && left_command.token_ids != right_command.token_ids
                {
                    return Err(
                        "Qwen IQ1_S activation ranges overlap for different token lanes"
                            .to_string(),
                    );
                }
                if ranges_overlap(left_token, right_token)
                    && (left_token != right_token
                        || left_command.token_ids != right_command.token_ids)
                {
                    return Err(
                        "Qwen IQ1_S token-map ranges overlap for different token lanes".to_string(),
                    );
                }
            }
        }
    }
    Ok(())
}

fn expanded_counts(
    role: Iq1sExpertRole,
    row_count: u32,
    lane_count: u16,
) -> Result<ExpandedIq1sCounts, String> {
    let input_columns = u64::from(role_input_columns(role)?);
    let blocks = u64::from(row_count)
        .checked_mul(input_columns / 256)
        .and_then(|value| value.checked_mul(u64::from(lane_count)))
        .ok_or("Qwen IQ1_S expanded block count overflow")?;
    Ok(ExpandedIq1sCounts {
        blocks,
        grid_passes: blocks
            .checked_mul(8)
            .ok_or("Qwen IQ1_S grid pass count overflow")?,
        delta_passes: blocks
            .checked_mul(8)
            .ok_or("Qwen IQ1_S delta pass count overflow")?,
    })
}

fn add_counts(
    left: ExpandedIq1sCounts,
    right: ExpandedIq1sCounts,
) -> Result<ExpandedIq1sCounts, String> {
    Ok(ExpandedIq1sCounts {
        blocks: left
            .blocks
            .checked_add(right.blocks)
            .ok_or("Qwen IQ1_S total block count overflow")?,
        grid_passes: left
            .grid_passes
            .checked_add(right.grid_passes)
            .ok_or("Qwen IQ1_S total grid pass count overflow")?,
        delta_passes: left
            .delta_passes
            .checked_add(right.delta_passes)
            .ok_or("Qwen IQ1_S total delta pass count overflow")?,
    })
}

fn command_bytes(command: &Iq1sCommand) -> [u8; IQ1S_COMMAND_BYTES] {
    let mut bytes = [0u8; IQ1S_COMMAND_BYTES];
    unsafe {
        std::ptr::copy_nonoverlapping(
            command as *const Iq1sCommand as *const u8,
            bytes.as_mut_ptr(),
            bytes.len(),
        );
    }
    bytes
}

fn refresh_crc(command: &mut Iq1sCommand) {
    command.crc32 = 0;
    command.crc32 = iq1s_command_crc32(&command_bytes(command));
}

fn first_u64(hash: &[u8; 32]) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&hash[..8]);
    u64::from_le_bytes(bytes)
}

fn hash_relocation(hash: &mut Sha256, relocation: &Relocation) {
    hash.update(relocation.command_index.to_le_bytes());
    hash.update(relocation.field_offset.to_le_bytes());
    hash.update([match relocation.source {
        RelocationSource::ArenaOffset => 1,
        RelocationSource::InputOffset => 2,
        RelocationSource::OutputOffset => 3,
        RelocationSource::TokenMapOffset => 4,
    }]);
    hash.update(relocation.addend.to_le_bytes());
}

fn hash_command_semantics(hash: &mut Sha256, command: &Iq1sCommand) {
    hash.update(command.layer_id.to_le_bytes());
    hash.update(command.phase.to_le_bytes());
    hash.update(command.role.to_le_bytes());
    hash.update(command.expert_id.to_le_bytes());
    hash.update(command.lane_mask.to_le_bytes());
    hash.update(command.lane_count.to_le_bytes());
    hash.update(command.weight_format.to_le_bytes());
    hash.update(command.arena_offset.to_le_bytes());
    hash.update(command.row_start.to_le_bytes());
    hash.update(command.row_count.to_le_bytes());
    hash.update(command.input_bytes.to_le_bytes());
    hash.update(command.output_bytes.to_le_bytes());
}

fn program_semantic_sha256(
    bank: usize,
    phase: LayerPhase,
    commands: &[Iq1sCommand],
    relocations: &[Relocation],
    expanded: ExpandedIq1sCounts,
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"hetgpu-qwen-iq1s-layer-program-v1\0");
    hash.update((bank as u32).to_le_bytes());
    hash.update(phase.abi().to_le_bytes());
    hash.update((commands.len() as u64).to_le_bytes());
    for command in commands {
        hash_command_semantics(&mut hash, command);
    }
    hash.update((relocations.len() as u64).to_le_bytes());
    for relocation in relocations {
        hash_relocation(&mut hash, relocation);
    }
    hash.update(expanded.blocks.to_le_bytes());
    hash.update(expanded.grid_passes.to_le_bytes());
    hash.update(expanded.delta_passes.to_le_bytes());
    hash.finalize().into()
}

fn phase_semantic_sha256(
    transaction_id: u64,
    phase: LayerPhase,
    programs: &[LayerProgram; ARENA_BANK_COUNT],
    activations: &[ActivationRange],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"hetgpu-qwen-iq1s-layer-phase-v2\0");
    hash.update(transaction_id.to_le_bytes());
    hash.update(phase.abi().to_le_bytes());
    for program in programs {
        hash.update(program.semantic_sha256);
    }
    for activation in activations {
        hash.update((activation.cuda_ptr as u64).to_le_bytes());
        hash.update(activation.source_identity_sha256);
        hash.update(activation.slab_offset.to_le_bytes());
        hash.update(activation.bytes.to_le_bytes());
        hash.update((activation.stream as u64).to_le_bytes());
    }
    hash.finalize().into()
}

fn command_relocations(command_index: u32, arena_offset: u64) -> [Relocation; 4] {
    [
        Relocation {
            command_index,
            field_offset: IQ1S_COMMAND_ARENA_OFFSET_OFFSET as u16,
            source: RelocationSource::ArenaOffset,
            addend: arena_offset,
        },
        Relocation {
            command_index,
            field_offset: IQ1S_COMMAND_INPUT_OFFSET_OFFSET as u16,
            source: RelocationSource::InputOffset,
            addend: 0,
        },
        Relocation {
            command_index,
            field_offset: IQ1S_COMMAND_OUTPUT_OFFSET_OFFSET as u16,
            source: RelocationSource::OutputOffset,
            addend: 0,
        },
        Relocation {
            command_index,
            field_offset: IQ1S_COMMAND_TOKEN_MAP_OFFSET_OFFSET as u16,
            source: RelocationSource::TokenMapOffset,
            addend: 0,
        },
    ]
}

fn build_command(
    semantic: &SemanticIq1sCommand,
    transaction_id: u64,
    completion_slot: u32,
) -> Result<Iq1sCommand, String> {
    let lane_mask = validate_semantic_command(semantic, semantic.phase)?;
    let lane_count = u16::try_from(semantic.token_ids.len())
        .map_err(|_| "Qwen IQ1_S lane count does not fit u16")?;
    let input_columns = u64::from(role_input_columns(semantic.role)?);
    let input_bytes = (input_columns / Q8_1_MMQ_VALUES)
        .checked_mul(Q8_1_MMQ_BYTES)
        .and_then(|value| value.checked_mul(u64::from(lane_count)))
        .ok_or("Qwen IQ1_S activation byte count overflow")?;
    let output_bytes = u64::from(semantic.row_shard.row_count)
        .checked_mul(F32_BYTES)
        .and_then(|value| value.checked_mul(u64::from(lane_count)))
        .ok_or("Qwen IQ1_S output byte count overflow")?;
    let mut command = Iq1sCommand {
        magic: IQ1S_COMMAND_MAGIC,
        abi_version: IQ1S_ABI_VERSION as u16,
        descriptor_bytes: IQ1S_COMMAND_BYTES as u16,
        crc32: 0,
        flags: 0,
        session_generation: 1,
        transaction_id,
        program_id: 0,
        trace_id: 0,
        layer_id: semantic.layer_id,
        phase: semantic.phase.abi(),
        role: role_abi(semantic.role)?,
        expert_id: semantic.expert_id,
        lane_mask,
        lane_count,
        weight_format: IQ1S_WEIGHT_FORMAT_IQ1_S as u16,
        arena_offset: semantic.row_shard.offset,
        input_offset: semantic.input_offset,
        output_offset: semantic.output_offset,
        row_start: semantic.row_shard.row_start,
        row_count: semantic.row_shard.row_count,
        input_bytes: u32::try_from(input_bytes)
            .map_err(|_| "Qwen IQ1_S input byte count does not fit u32")?,
        output_bytes: u32::try_from(output_bytes)
            .map_err(|_| "Qwen IQ1_S output byte count does not fit u32")?,
        token_map_offset: semantic.token_map_offset,
        dependency_fence: 0,
        completion_slot,
        reserved: 0,
    };
    refresh_crc(&mut command);
    Ok(command)
}

pub(crate) fn compile_layer_phase(
    plan: &LayerPhasePlan,
    mode: &str,
    model_context_limit: u32,
) -> Result<CompiledLayerPhase, String> {
    if plan.transaction_id == 0 {
        return Err("Qwen IQ1_S layer transaction ID must be nonzero".to_string());
    }
    if model_context_limit == 0 || model_context_limit > QWEN_MODEL_CONTEXT_LIMIT {
        return Err(format!(
            "Qwen model context limit {model_context_limit} exceeds supported limit {QWEN_MODEL_CONTEXT_LIMIT}"
        ));
    }
    if plan.commands.is_empty() {
        return Err("Qwen IQ1_S layer phase has no commands".to_string());
    }
    validate_activation_ranges(&plan.activations)?;
    validate_semantic_buffer_bindings(plan)?;
    let mut semantic_by_bank: [Vec<SemanticIq1sCommand>; ARENA_BANK_COUNT] =
        std::array::from_fn(|_| Vec::new());
    let mut coordinates = BTreeSet::new();
    for command in &plan.commands {
        validate_semantic_command(command, plan.phase)?;
        let key = (
            command.row_shard.bank,
            command.role,
            command.expert_id,
            command.lane_mask,
            command.token_map_offset,
        );
        if !coordinates.insert(key) {
            return Err("Qwen IQ1_S layer phase repeats a semantic command".to_string());
        }
        semantic_by_bank[usize::from(command.row_shard.bank)].push(command.clone());
    }
    for commands in &mut semantic_by_bank {
        commands.sort_by(|left, right| {
            left.role
                .cmp(&right.role)
                .then_with(|| left.expert_id.cmp(&right.expert_id))
                .then_with(|| left.token_ids.cmp(&right.token_ids))
        });
    }
    if semantic_by_bank.iter().any(Vec::is_empty) {
        return Err("Qwen IQ1_S layer phase must cover all four U250 banks".to_string());
    }

    let labels = HashMap::from([
        ("PARAM_MATRIX".to_string(), 0x1000),
        ("PARAM_INPUT".to_string(), 0x2000),
        ("PARAM_OUTPUT".to_string(), 0x3000),
    ]);
    let mut descriptor_banks: [Vec<Iq1sCommand>; ARENA_BANK_COUNT] =
        std::array::from_fn(|_| Vec::new());
    let mut program_slots: [Option<LayerProgram>; ARENA_BANK_COUNT] = std::array::from_fn(|_| None);
    let mut completion_slot = 0u32;
    for bank in 0..ARENA_BANK_COUNT {
        let first = &semantic_by_bank[bank][0];
        let input_columns = role_input_columns(first.role)?;
        let output_rows = first.row_shard.row_count;
        if semantic_by_bank[bank].iter().any(|command| {
            role_input_columns(command.role) != Ok(input_columns)
                || command.row_shard.row_count != output_rows
        }) {
            return Err("one U250 bank program contains incompatible Qwen shapes".to_string());
        }
        let trace = build_selected_trace_for_shape(
            mode,
            model_context_limit,
            input_columns,
            output_rows,
            &labels,
            4,
        )?;
        let mut expanded = ExpandedIq1sCounts::default();
        let mut relocations = Vec::new();
        for semantic in &semantic_by_bank[bank] {
            let command_index = u32::try_from(descriptor_banks[bank].len())
                .map_err(|_| "Qwen IQ1_S command index does not fit u32")?;
            let command = build_command(semantic, plan.transaction_id, completion_slot)?;
            completion_slot = completion_slot
                .checked_add(1)
                .ok_or("Qwen IQ1_S completion slot overflow")?;
            expanded = add_counts(
                expanded,
                expanded_counts(
                    role_from_abi(command.role)?,
                    command.row_count,
                    command.lane_count,
                )?,
            )?;
            relocations.extend(command_relocations(command_index, command.arena_offset));
            descriptor_banks[bank].push(command);
        }
        let semantic_sha256 = program_semantic_sha256(
            bank,
            plan.phase,
            &descriptor_banks[bank],
            &relocations,
            expanded,
        );
        let program_id = first_u64(&trace.selected_sha256);
        let trace_id = first_u64(&semantic_sha256);
        for command in &mut descriptor_banks[bank] {
            command.program_id = program_id;
            command.trace_id = trace_id;
            refresh_crc(command);
        }
        program_slots[bank] = Some(LayerProgram {
            kind: trace.selected_kind,
            assembly: trace.assembly,
            encoded: trace.program,
            relocations,
            semantic_sha256,
            assembly_sha256: trace.assembly_sha256,
            expanded,
        });
    }
    let [Some(program0), Some(program1), Some(program2), Some(program3)] = program_slots else {
        return Err("Qwen IQ1_S layer phase did not compile all four bank programs".to_string());
    };
    let programs = [program0, program1, program2, program3];
    let semantic_sha256 = phase_semantic_sha256(
        plan.transaction_id,
        plan.phase,
        &programs,
        &plan.activations,
    );
    let compiled = CompiledLayerPhase {
        transaction_id: plan.transaction_id,
        phase: plan.phase,
        programs,
        commands: descriptor_banks,
        activations: plan.activations.clone(),
        semantic_sha256,
    };
    validate_compiled_layer_phase(&compiled, model_context_limit)?;
    Ok(compiled)
}

pub(crate) fn validate_compiled_layer_phase(
    compiled: &CompiledLayerPhase,
    model_context_limit: u32,
) -> Result<(), String> {
    if compiled.transaction_id == 0
        || model_context_limit == 0
        || model_context_limit > QWEN_MODEL_CONTEXT_LIMIT
    {
        return Err("compiled Qwen IQ1_S layer phase has invalid global metadata".to_string());
    }
    validate_activation_ranges(&compiled.activations)?;
    let labels = HashMap::from([
        ("PARAM_MATRIX".to_string(), 0x1000),
        ("PARAM_INPUT".to_string(), 0x2000),
        ("PARAM_OUTPUT".to_string(), 0x3000),
    ]);
    for bank in 0..ARENA_BANK_COUNT {
        let program = &compiled.programs[bank];
        let commands = &compiled.commands[bank];
        if commands.is_empty() {
            return Err("compiled Qwen IQ1_S bank contains no commands".to_string());
        }
        let mut recomputed_expanded = ExpandedIq1sCounts::default();
        for command in commands {
            let role = role_from_abi(command.role)?;
            recomputed_expanded = add_counts(
                recomputed_expanded,
                expanded_counts(role, command.row_count, command.lane_count)?,
            )?;
        }
        let recomputed_semantic = program_semantic_sha256(
            bank,
            compiled.phase,
            commands,
            &program.relocations,
            recomputed_expanded,
        );
        if recomputed_semantic != program.semantic_sha256 || recomputed_expanded != program.expanded
        {
            return Err(format!("Qwen IQ1_S bank {bank} semantic hash mismatch"));
        }
        let first_role = role_from_abi(commands[0].role)?;
        let reference = build_selected_trace_for_shape(
            program.kind.as_str(),
            model_context_limit,
            role_input_columns(first_role)?,
            commands[0].row_count,
            &labels,
            4,
        )?;
        let assembly_sha256: [u8; 32] = Sha256::digest(program.assembly.as_bytes()).into();
        if reference.program != program.encoded
            || reference.assembly != program.assembly
            || assembly_sha256 != program.assembly_sha256
        {
            return Err(format!("Qwen IQ1_S bank {bank} assembly proof mismatch"));
        }
        let expected_program_id = first_u64(&reference.selected_sha256);
        let expected_trace_id = first_u64(&program.semantic_sha256);
        for command in commands {
            if command.magic != IQ1S_COMMAND_MAGIC
                || command.abi_version != IQ1S_ABI_VERSION as u16
                || command.descriptor_bytes != IQ1S_COMMAND_BYTES as u16
                || command.phase != compiled.phase.abi()
                || command.weight_format != IQ1S_WEIGHT_FORMAT_IQ1_S as u16
                || command.program_id != expected_program_id
                || command.trace_id != expected_trace_id
            {
                return Err("compiled Qwen IQ1_S descriptor metadata mismatch".to_string());
            }
            let mut bytes = command_bytes(command);
            bytes[8..12].fill(0);
            if command.crc32 != iq1s_command_crc32(&bytes) {
                return Err("compiled Qwen IQ1_S descriptor CRC mismatch".to_string());
            }
        }
    }
    let phase_hash = phase_semantic_sha256(
        compiled.transaction_id,
        compiled.phase,
        &compiled.programs,
        &compiled.activations,
    );
    if phase_hash != compiled.semantic_sha256 {
        return Err("Qwen IQ1_S layer phase semantic hash mismatch".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r#impl::iq1s_weight_registry::Iq1sTensorIdentity;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn tensor(role: Iq1sExpertRole) -> Arc<Iq1sTensorIdentity> {
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
        Arc::new(Iq1sTensorIdentity {
            canonical_path: PathBuf::from("/tmp/qwen-trace.gguf"),
            file_offset: 0,
            nbytes: 419_430_400,
            name: format!("blk.7.ffn_{name}_exps.weight"),
            layer: 7,
            ne,
            nb,
            role,
            model_sha256: [0x11; 32],
            content_sha256: [0x22; 32],
            device: 1,
            inode: 2,
            modified_ns: 3,
        })
    }

    fn shard(role: Iq1sExpertRole, expert: u16, bank: u8) -> ArenaShard {
        let tensor = tensor(role);
        let row_count = match role {
            Iq1sExpertRole::Gate | Iq1sExpertRole::Up => 256,
            Iq1sExpertRole::Down => 1024,
            Iq1sExpertRole::GateUp => unreachable!(),
        };
        let row_bytes = tensor.ne[0] / 256 * 50;
        ArenaShard {
            tensor,
            expert,
            bank,
            row_start: u32::from(bank) * row_count,
            row_count,
            superblock: 0,
            offset: u64::from(expert) * 1024 * 1024,
            bytes: u64::from(row_count) * row_bytes,
            sha256: [0x33; 32],
        }
    }

    fn phase_for_role(role: Iq1sExpertRole) -> LayerPhase {
        match role {
            Iq1sExpertRole::Gate | Iq1sExpertRole::Up => LayerPhase::PhaseA,
            Iq1sExpertRole::Down => LayerPhase::PhaseB,
            Iq1sExpertRole::GateUp => unreachable!(),
        }
    }

    fn plan(role: Iq1sExpertRole, batch: u32, repeated: bool) -> LayerPhasePlan {
        let phase = phase_for_role(role);
        let mut commands = Vec::new();
        if repeated {
            let token_ids = (0..batch).collect::<Vec<_>>();
            let mask = lane_mask(&token_ids).unwrap();
            for bank in 0..4 {
                commands.push(SemanticIq1sCommand {
                    layer_id: 7,
                    phase,
                    role,
                    expert_id: 17,
                    lane_mask: mask,
                    token_ids: token_ids.clone(),
                    input_offset: 0,
                    output_offset: 0,
                    token_map_offset: 0,
                    row_shard: shard(role, 17, bank),
                });
            }
        } else {
            for token_id in 0..batch {
                for bank in 0..4 {
                    commands.push(SemanticIq1sCommand {
                        layer_id: 7,
                        phase,
                        role,
                        expert_id: token_id as u16,
                        lane_mask: 1u16 << token_id,
                        token_ids: vec![token_id],
                        input_offset: u64::from(token_id) * 64 * 1024,
                        output_offset: u64::from(token_id) * 64 * 1024,
                        token_map_offset: u64::from(token_id) * 64,
                        row_shard: shard(role, token_id as u16, bank),
                    });
                }
            }
        }
        LayerPhasePlan {
            transaction_id: 44,
            phase,
            commands,
            activations: vec![
                ActivationRange {
                    cuda_ptr: 0x10_0000,
                    slab_offset: 0,
                    bytes: 1024 * 1024,
                    stream: 0xabc0,
                    source_identity_sha256: [0x51; 32],
                },
                ActivationRange {
                    cuda_ptr: 0x20_0000,
                    slab_offset: 1024 * 1024,
                    bytes: 1024 * 1024,
                    stream: 0xabc0,
                    source_identity_sha256: [0x52; 32],
                },
            ],
        }
    }

    #[test]
    fn iq1s_layer_trace_accepts_shared_cuda_pointer_with_distinct_source_identities() {
        let mut shared = plan(Iq1sExpertRole::Gate, 1, true);
        shared.activations[1].cuda_ptr = shared.activations[0].cuda_ptr;
        compile_layer_phase(&shared, "compiler", QWEN_MODEL_CONTEXT_LIMIT).unwrap();
    }

    #[test]
    fn iq1s_layer_trace_rejects_one_source_identity_mapped_to_two_slab_ranges() {
        let mut ambiguous = plan(Iq1sExpertRole::Gate, 1, true);
        ambiguous.activations[1].source_identity_sha256 =
            ambiguous.activations[0].source_identity_sha256;
        assert!(compile_layer_phase(&ambiguous, "compiler", QWEN_MODEL_CONTEXT_LIMIT)
            .unwrap_err()
            .contains("activation identity maps to inconsistent ranges"));
    }

    #[test]
    fn iq1s_layer_trace_matches_handwritten_and_compiler_for_all_shapes_and_batches() {
        for role in [
            Iq1sExpertRole::Gate,
            Iq1sExpertRole::Up,
            Iq1sExpertRole::Down,
        ] {
            for batch in [1, 6, 9, 16] {
                for repeated in [false, true] {
                    let plan = plan(role, batch, repeated);
                    let handwritten = compile_layer_phase(&plan, "handwritten", 262_144).unwrap();
                    let compiler = compile_layer_phase(&plan, "compiler", 262_144).unwrap();
                    assert_eq!(handwritten.semantic_sha256, compiler.semantic_sha256);
                    let row_count = plan.commands[0].row_shard.row_count;
                    let input_columns = role_input_columns(role).unwrap();
                    let expected_blocks =
                        u64::from(row_count) * (u64::from(input_columns) / 256) * u64::from(batch);
                    for bank in 0..4 {
                        assert_eq!(handwritten.programs[bank].expanded.blocks, expected_blocks);
                        assert_eq!(compiler.programs[bank].expanded.blocks, expected_blocks);
                        assert_eq!(
                            handwritten.programs[bank].expanded.grid_passes,
                            expected_blocks * 8
                        );
                        assert_eq!(
                            handwritten.programs[bank].expanded.delta_passes,
                            expected_blocks * 8
                        );
                        assert_eq!(
                            handwritten.programs[bank].encoded,
                            compiler.programs[bank].encoded
                        );
                        assert_eq!(
                            handwritten.commands[bank].len(),
                            if repeated { 1 } else { batch as usize }
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn iq1s_layer_trace_assigns_disjoint_multi_matrix_outputs() {
        let compiled = compile_layer_phase(
            &plan(Iq1sExpertRole::Gate, 2, false),
            "compiler",
            QWEN_MODEL_CONTEXT_LIMIT,
        )
        .unwrap();
        for bank in 0..ARENA_BANK_COUNT {
            assert_ne!(
                compiled.commands[bank][0].output_offset,
                compiled.commands[bank][1].output_offset
            );
            assert_ne!(
                compiled.commands[bank][0].token_map_offset,
                compiled.commands[bank][1].token_map_offset
            );
        }
    }

    #[test]
    fn iq1s_layer_trace_rejects_ambiguous_or_out_of_range_slab_bindings() {
        let mut overlap = plan(Iq1sExpertRole::Gate, 2, false);
        overlap.commands[4].output_offset = overlap.commands[0].output_offset;
        assert!(
            compile_layer_phase(&overlap, "compiler", QWEN_MODEL_CONTEXT_LIMIT)
                .unwrap_err()
                .contains("output ranges overlap")
        );

        let mut outside = plan(Iq1sExpertRole::Gate, 1, true);
        outside.commands[0].input_offset = 8 * 1024 * 1024;
        assert!(
            compile_layer_phase(&outside, "compiler", QWEN_MODEL_CONTEXT_LIMIT)
                .unwrap_err()
                .contains("activation manifest")
        );

        let mut token_overlap = plan(Iq1sExpertRole::Gate, 2, false);
        token_overlap.commands[4].token_map_offset = token_overlap.commands[0].token_map_offset;
        assert!(
            compile_layer_phase(&token_overlap, "compiler", QWEN_MODEL_CONTEXT_LIMIT)
                .unwrap_err()
                .contains("token-map ranges overlap")
        );
    }

    #[test]
    fn iq1s_layer_trace_accepts_model_limit_and_rejects_above_limit() {
        let plan = plan(Iq1sExpertRole::Gate, 1, true);
        compile_layer_phase(&plan, "compiler", 262_144).unwrap();
        assert!(compile_layer_phase(&plan, "compiler", 262_145)
            .unwrap_err()
            .contains("262144"));
    }

    #[test]
    fn iq1s_layer_trace_mutations_fail_semantic_hash_validation() {
        let plan = plan(Iq1sExpertRole::Down, 6, false);
        let original = compile_layer_phase(&plan, "compiler", 262_144).unwrap();

        let mut relocation = original.clone();
        relocation.programs[0].relocations[0].source = RelocationSource::OutputOffset;
        assert!(validate_compiled_layer_phase(&relocation, 262_144)
            .unwrap_err()
            .contains("semantic hash"));

        let mut row = original.clone();
        row.commands[0][0].row_start += 1;
        assert!(validate_compiled_layer_phase(&row, 262_144)
            .unwrap_err()
            .contains("semantic hash"));

        let mut expert = original;
        expert.commands[0][0].expert_id += 1;
        assert!(validate_compiled_layer_phase(&expert, 262_144)
            .unwrap_err()
            .contains("semantic hash"));
    }
}
