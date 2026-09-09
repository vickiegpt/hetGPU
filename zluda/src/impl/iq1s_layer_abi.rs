// Generated from tools/qwen35-iq1s-layer-abi.json; do not edit.
// Canonical schema SHA-256: 0e88b8acb51968aaf847b17cb34fc7e5c703a4834ebcf764b0a84443247ee619

pub const IQ1S_SCHEMA_SHA256: &str =
    "0e88b8acb51968aaf847b17cb34fc7e5c703a4834ebcf764b0a84443247ee619";
pub const IQ1S_ABI_VERSION: u32 = 2;
pub const IQ1S_REGISTER_MAGIC: u32 = 0x324c5149;
pub const IQ1S_COMMAND_MAGIC: u32 = 0x32435149;
pub const IQ1S_COMPLETION_MAGIC: u32 = 0x32445149;
pub const IQ1S_PHASE_A: u32 = 1;
pub const IQ1S_PHASE_B: u32 = 2;
pub const IQ1S_ROLE_GATE: u32 = 1;
pub const IQ1S_ROLE_UP: u32 = 2;
pub const IQ1S_ROLE_DOWN: u32 = 3;
pub const IQ1S_WEIGHT_FORMAT_IQ1_S: u32 = 19;
pub const IQ1S_COMPLETION_STATUS_OK: u32 = 0;
pub const IQ1S_COMPLETION_STATUS_FAULT: u32 = 1;
pub const IQ1S_COMPLETION_STATUS_ABORTED: u32 = 2;
pub const IQ1S_FAULT_CODE_NONE: u32 = 0;
pub const IQ1S_FAULT_CODE_BAD_MAGIC: u32 = 1;
pub const IQ1S_FAULT_CODE_BAD_VERSION: u32 = 2;
pub const IQ1S_FAULT_CODE_BAD_LENGTH: u32 = 3;
pub const IQ1S_FAULT_CODE_BAD_GENERATION: u32 = 4;
pub const IQ1S_FAULT_CODE_BAD_CRC: u32 = 5;
pub const IQ1S_FAULT_CODE_BAD_BOUNDS: u32 = 6;
pub const IQ1S_FAULT_CODE_BAD_RELOCATION: u32 = 7;
pub const IQ1S_FAULT_CODE_AXI: u32 = 8;
pub const IQ1S_FAULT_CODE_TIMEOUT: u32 = 9;
pub const IQ1S_FAULT_CODE_NONFINITE: u32 = 10;
pub const IQ1S_FAULT_CODE_RING_OVERFLOW: u32 = 11;

pub const IQ1S_REG_ABI_MAGIC_OFFSET: usize = 0;
pub const IQ1S_REG_ABI_VERSION_OFFSET: usize = 4;
pub const IQ1S_REG_CONTROL_OFFSET: usize = 8;
pub const IQ1S_REG_STATUS_OFFSET: usize = 12;
pub const IQ1S_REG_SESSION_GENERATION_LO_OFFSET: usize = 16;
pub const IQ1S_REG_SESSION_GENERATION_HI_OFFSET: usize = 20;
pub const IQ1S_REG_COMMAND_BASE_LO_OFFSET: usize = 24;
pub const IQ1S_REG_COMMAND_BASE_HI_OFFSET: usize = 28;
pub const IQ1S_REG_COMMAND_CAPACITY_OFFSET: usize = 32;
pub const IQ1S_REG_COMMAND_PRODUCER_OFFSET: usize = 36;
pub const IQ1S_REG_COMMAND_CONSUMER_OFFSET: usize = 40;
pub const IQ1S_REG_DOORBELL_OFFSET: usize = 44;
pub const IQ1S_REG_COMPLETION_BASE_LO_OFFSET: usize = 48;
pub const IQ1S_REG_COMPLETION_BASE_HI_OFFSET: usize = 52;
pub const IQ1S_REG_COMPLETION_CAPACITY_OFFSET: usize = 56;
pub const IQ1S_REG_COMPLETION_PRODUCER_OFFSET: usize = 60;
pub const IQ1S_REG_COMPLETION_CONSUMER_OFFSET: usize = 64;
pub const IQ1S_REG_FAULT_CODE_OFFSET: usize = 68;
pub const IQ1S_REG_FAULT_DETAIL_LO_OFFSET: usize = 72;
pub const IQ1S_REG_FAULT_DETAIL_HI_OFFSET: usize = 76;
pub const IQ1S_REG_QUIESCENT_OFFSET: usize = 80;
pub const IQ1S_REG_PROGRAM_BASE_LO_OFFSET: usize = 88;
pub const IQ1S_REG_PROGRAM_BASE_HI_OFFSET: usize = 92;
pub const IQ1S_REG_ARENA_MANIFEST_BASE_LO_OFFSET: usize = 96;
pub const IQ1S_REG_ARENA_MANIFEST_BASE_HI_OFFSET: usize = 100;
pub const IQ1S_REG_ACTIVATION_BASE_LO_OFFSET: usize = 104;
pub const IQ1S_REG_ACTIVATION_BASE_HI_OFFSET: usize = 108;
pub const IQ1S_REG_RESULT_BASE_LO_OFFSET: usize = 112;
pub const IQ1S_REG_RESULT_BASE_HI_OFFSET: usize = 116;
pub const IQ1S_REG_TOKEN_MAP_BASE_LO_OFFSET: usize = 120;
pub const IQ1S_REG_TOKEN_MAP_BASE_HI_OFFSET: usize = 124;
pub const IQ1S_REG_MODEL_TAG_LO_OFFSET: usize = 128;
pub const IQ1S_REG_MODEL_TAG_HI_OFFSET: usize = 132;
pub const IQ1S_REG_ACTIVATION_BYTES_OFFSET: usize = 136;
pub const IQ1S_REG_RESULT_BYTES_OFFSET: usize = 140;
pub const IQ1S_REG_TOKEN_MAP_BYTES_OFFSET: usize = 144;
pub const IQ1S_REG_PROGRAM_BYTES_OFFSET: usize = 148;
pub const IQ1S_REG_ARENA_MANIFEST_BYTES_OFFSET: usize = 152;
pub const IQ1S_REG_CU_ID_OFFSET: usize = 156;

pub const IQ1S_COMMAND_BYTES: usize = 128;
pub const IQ1S_COMMAND_MAGIC_OFFSET: usize = 0;
pub const IQ1S_COMMAND_ABI_VERSION_OFFSET: usize = 4;
pub const IQ1S_COMMAND_DESCRIPTOR_BYTES_OFFSET: usize = 6;
pub const IQ1S_COMMAND_CRC32_OFFSET: usize = 8;
pub const IQ1S_COMMAND_FLAGS_OFFSET: usize = 12;
pub const IQ1S_COMMAND_SESSION_GENERATION_OFFSET: usize = 16;
pub const IQ1S_COMMAND_TRANSACTION_ID_OFFSET: usize = 24;
pub const IQ1S_COMMAND_PROGRAM_ID_OFFSET: usize = 32;
pub const IQ1S_COMMAND_TRACE_ID_OFFSET: usize = 40;
pub const IQ1S_COMMAND_LAYER_ID_OFFSET: usize = 48;
pub const IQ1S_COMMAND_PHASE_OFFSET: usize = 52;
pub const IQ1S_COMMAND_ROLE_OFFSET: usize = 54;
pub const IQ1S_COMMAND_EXPERT_ID_OFFSET: usize = 56;
pub const IQ1S_COMMAND_LANE_MASK_OFFSET: usize = 58;
pub const IQ1S_COMMAND_LANE_COUNT_OFFSET: usize = 60;
pub const IQ1S_COMMAND_WEIGHT_FORMAT_OFFSET: usize = 62;
pub const IQ1S_COMMAND_ARENA_OFFSET_OFFSET: usize = 64;
pub const IQ1S_COMMAND_INPUT_OFFSET_OFFSET: usize = 72;
pub const IQ1S_COMMAND_OUTPUT_OFFSET_OFFSET: usize = 80;
pub const IQ1S_COMMAND_ROW_START_OFFSET: usize = 88;
pub const IQ1S_COMMAND_ROW_COUNT_OFFSET: usize = 92;
pub const IQ1S_COMMAND_INPUT_BYTES_OFFSET: usize = 96;
pub const IQ1S_COMMAND_OUTPUT_BYTES_OFFSET: usize = 100;
pub const IQ1S_COMMAND_TOKEN_MAP_OFFSET_OFFSET: usize = 104;
pub const IQ1S_COMMAND_DEPENDENCY_FENCE_OFFSET: usize = 112;
pub const IQ1S_COMMAND_COMPLETION_SLOT_OFFSET: usize = 120;
pub const IQ1S_COMMAND_RESERVED_OFFSET: usize = 124;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Iq1sCommand {
    pub magic: u32,
    pub abi_version: u16,
    pub descriptor_bytes: u16,
    pub crc32: u32,
    pub flags: u32,
    pub session_generation: u64,
    pub transaction_id: u64,
    pub program_id: u64,
    pub trace_id: u64,
    pub layer_id: u32,
    pub phase: u16,
    pub role: u16,
    pub expert_id: u16,
    pub lane_mask: u16,
    pub lane_count: u16,
    pub weight_format: u16,
    pub arena_offset: u64,
    pub input_offset: u64,
    pub output_offset: u64,
    pub row_start: u32,
    pub row_count: u32,
    pub input_bytes: u32,
    pub output_bytes: u32,
    pub token_map_offset: u64,
    pub dependency_fence: u64,
    pub completion_slot: u32,
    pub reserved: u32,
}

const _: [(); IQ1S_COMMAND_BYTES] = [(); core::mem::size_of::<Iq1sCommand>()];
const _: [(); IQ1S_COMMAND_MAGIC_OFFSET] = [(); core::mem::offset_of!(Iq1sCommand, magic)];
const _: [(); IQ1S_COMMAND_ABI_VERSION_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCommand, abi_version)];
const _: [(); IQ1S_COMMAND_DESCRIPTOR_BYTES_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCommand, descriptor_bytes)];
const _: [(); IQ1S_COMMAND_CRC32_OFFSET] = [(); core::mem::offset_of!(Iq1sCommand, crc32)];
const _: [(); IQ1S_COMMAND_FLAGS_OFFSET] = [(); core::mem::offset_of!(Iq1sCommand, flags)];
const _: [(); IQ1S_COMMAND_SESSION_GENERATION_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCommand, session_generation)];
const _: [(); IQ1S_COMMAND_TRANSACTION_ID_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCommand, transaction_id)];
const _: [(); IQ1S_COMMAND_PROGRAM_ID_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCommand, program_id)];
const _: [(); IQ1S_COMMAND_TRACE_ID_OFFSET] = [(); core::mem::offset_of!(Iq1sCommand, trace_id)];
const _: [(); IQ1S_COMMAND_LAYER_ID_OFFSET] = [(); core::mem::offset_of!(Iq1sCommand, layer_id)];
const _: [(); IQ1S_COMMAND_PHASE_OFFSET] = [(); core::mem::offset_of!(Iq1sCommand, phase)];
const _: [(); IQ1S_COMMAND_ROLE_OFFSET] = [(); core::mem::offset_of!(Iq1sCommand, role)];
const _: [(); IQ1S_COMMAND_EXPERT_ID_OFFSET] = [(); core::mem::offset_of!(Iq1sCommand, expert_id)];
const _: [(); IQ1S_COMMAND_LANE_MASK_OFFSET] = [(); core::mem::offset_of!(Iq1sCommand, lane_mask)];
const _: [(); IQ1S_COMMAND_LANE_COUNT_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCommand, lane_count)];
const _: [(); IQ1S_COMMAND_WEIGHT_FORMAT_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCommand, weight_format)];
const _: [(); IQ1S_COMMAND_ARENA_OFFSET_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCommand, arena_offset)];
const _: [(); IQ1S_COMMAND_INPUT_OFFSET_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCommand, input_offset)];
const _: [(); IQ1S_COMMAND_OUTPUT_OFFSET_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCommand, output_offset)];
const _: [(); IQ1S_COMMAND_ROW_START_OFFSET] = [(); core::mem::offset_of!(Iq1sCommand, row_start)];
const _: [(); IQ1S_COMMAND_ROW_COUNT_OFFSET] = [(); core::mem::offset_of!(Iq1sCommand, row_count)];
const _: [(); IQ1S_COMMAND_INPUT_BYTES_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCommand, input_bytes)];
const _: [(); IQ1S_COMMAND_OUTPUT_BYTES_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCommand, output_bytes)];
const _: [(); IQ1S_COMMAND_TOKEN_MAP_OFFSET_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCommand, token_map_offset)];
const _: [(); IQ1S_COMMAND_DEPENDENCY_FENCE_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCommand, dependency_fence)];
const _: [(); IQ1S_COMMAND_COMPLETION_SLOT_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCommand, completion_slot)];
const _: [(); IQ1S_COMMAND_RESERVED_OFFSET] = [(); core::mem::offset_of!(Iq1sCommand, reserved)];

pub const IQ1S_COMPLETION_BYTES: usize = 128;
pub const IQ1S_COMPLETION_MAGIC_OFFSET: usize = 0;
pub const IQ1S_COMPLETION_ABI_VERSION_OFFSET: usize = 4;
pub const IQ1S_COMPLETION_COMPLETION_BYTES_OFFSET: usize = 6;
pub const IQ1S_COMPLETION_STATUS_OFFSET: usize = 8;
pub const IQ1S_COMPLETION_FAULT_CODE_OFFSET: usize = 12;
pub const IQ1S_COMPLETION_SESSION_GENERATION_OFFSET: usize = 16;
pub const IQ1S_COMPLETION_TRANSACTION_ID_OFFSET: usize = 24;
pub const IQ1S_COMPLETION_PROGRAM_ID_OFFSET: usize = 32;
pub const IQ1S_COMPLETION_TRACE_ID_OFFSET: usize = 40;
pub const IQ1S_COMPLETION_LAYER_ID_OFFSET: usize = 48;
pub const IQ1S_COMPLETION_PHASE_OFFSET: usize = 52;
pub const IQ1S_COMPLETION_ROLE_OFFSET: usize = 54;
pub const IQ1S_COMPLETION_CU_ID_OFFSET: usize = 56;
pub const IQ1S_COMPLETION_EXPERT_ID_OFFSET: usize = 58;
pub const IQ1S_COMPLETION_LANE_MASK_OFFSET: usize = 60;
pub const IQ1S_COMPLETION_ROWS_COMPLETED_OFFSET: usize = 62;
pub const IQ1S_COMPLETION_DESCRIPTOR_CRC32_OFFSET: usize = 64;
pub const IQ1S_COMPLETION_COMMAND_INDEX_OFFSET: usize = 68;
pub const IQ1S_COMPLETION_CYCLES_OFFSET: usize = 72;
pub const IQ1S_COMPLETION_DDR_READ_BYTES_OFFSET: usize = 80;
pub const IQ1S_COMPLETION_DDR_WRITE_BYTES_OFFSET: usize = 88;
pub const IQ1S_COMPLETION_IQ1S_BLOCKS_OFFSET: usize = 96;
pub const IQ1S_COMPLETION_GRID_PASSES_OFFSET: usize = 104;
pub const IQ1S_COMPLETION_DELTA_PASSES_OFFSET: usize = 108;
pub const IQ1S_COMPLETION_RESULT_FENCE_OFFSET: usize = 112;
pub const IQ1S_COMPLETION_FAULT_DETAIL_OFFSET: usize = 120;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Iq1sCompletion {
    pub magic: u32,
    pub abi_version: u16,
    pub completion_bytes: u16,
    pub status: u32,
    pub fault_code: u32,
    pub session_generation: u64,
    pub transaction_id: u64,
    pub program_id: u64,
    pub trace_id: u64,
    pub layer_id: u32,
    pub phase: u16,
    pub role: u16,
    pub cu_id: u16,
    pub expert_id: u16,
    pub lane_mask: u16,
    pub rows_completed: u16,
    pub descriptor_crc32: u32,
    pub command_index: u32,
    pub cycles: u64,
    pub ddr_read_bytes: u64,
    pub ddr_write_bytes: u64,
    pub iq1s_blocks: u64,
    pub grid_passes: u32,
    pub delta_passes: u32,
    pub result_fence: u64,
    pub fault_detail: u64,
}

const _: [(); IQ1S_COMPLETION_BYTES] = [(); core::mem::size_of::<Iq1sCompletion>()];
const _: [(); IQ1S_COMPLETION_MAGIC_OFFSET] = [(); core::mem::offset_of!(Iq1sCompletion, magic)];
const _: [(); IQ1S_COMPLETION_ABI_VERSION_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCompletion, abi_version)];
const _: [(); IQ1S_COMPLETION_COMPLETION_BYTES_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCompletion, completion_bytes)];
const _: [(); IQ1S_COMPLETION_STATUS_OFFSET] = [(); core::mem::offset_of!(Iq1sCompletion, status)];
const _: [(); IQ1S_COMPLETION_FAULT_CODE_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCompletion, fault_code)];
const _: [(); IQ1S_COMPLETION_SESSION_GENERATION_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCompletion, session_generation)];
const _: [(); IQ1S_COMPLETION_TRANSACTION_ID_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCompletion, transaction_id)];
const _: [(); IQ1S_COMPLETION_PROGRAM_ID_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCompletion, program_id)];
const _: [(); IQ1S_COMPLETION_TRACE_ID_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCompletion, trace_id)];
const _: [(); IQ1S_COMPLETION_LAYER_ID_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCompletion, layer_id)];
const _: [(); IQ1S_COMPLETION_PHASE_OFFSET] = [(); core::mem::offset_of!(Iq1sCompletion, phase)];
const _: [(); IQ1S_COMPLETION_ROLE_OFFSET] = [(); core::mem::offset_of!(Iq1sCompletion, role)];
const _: [(); IQ1S_COMPLETION_CU_ID_OFFSET] = [(); core::mem::offset_of!(Iq1sCompletion, cu_id)];
const _: [(); IQ1S_COMPLETION_EXPERT_ID_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCompletion, expert_id)];
const _: [(); IQ1S_COMPLETION_LANE_MASK_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCompletion, lane_mask)];
const _: [(); IQ1S_COMPLETION_ROWS_COMPLETED_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCompletion, rows_completed)];
const _: [(); IQ1S_COMPLETION_DESCRIPTOR_CRC32_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCompletion, descriptor_crc32)];
const _: [(); IQ1S_COMPLETION_COMMAND_INDEX_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCompletion, command_index)];
const _: [(); IQ1S_COMPLETION_CYCLES_OFFSET] = [(); core::mem::offset_of!(Iq1sCompletion, cycles)];
const _: [(); IQ1S_COMPLETION_DDR_READ_BYTES_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCompletion, ddr_read_bytes)];
const _: [(); IQ1S_COMPLETION_DDR_WRITE_BYTES_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCompletion, ddr_write_bytes)];
const _: [(); IQ1S_COMPLETION_IQ1S_BLOCKS_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCompletion, iq1s_blocks)];
const _: [(); IQ1S_COMPLETION_GRID_PASSES_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCompletion, grid_passes)];
const _: [(); IQ1S_COMPLETION_DELTA_PASSES_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCompletion, delta_passes)];
const _: [(); IQ1S_COMPLETION_RESULT_FENCE_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCompletion, result_fence)];
const _: [(); IQ1S_COMPLETION_FAULT_DETAIL_OFFSET] =
    [(); core::mem::offset_of!(Iq1sCompletion, fault_detail)];

/// command CRC32 is IEEE CRC32 with bytes 8..11 zeroed.
pub fn iq1s_command_crc32(record: &[u8; IQ1S_COMMAND_BYTES]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for (index, value) in record.iter().copied().enumerate() {
        let byte = if (8..=11).contains(&index) { 0 } else { value };
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & 0u32.wrapping_sub(crc & 1));
        }
    }
    !crc
}
