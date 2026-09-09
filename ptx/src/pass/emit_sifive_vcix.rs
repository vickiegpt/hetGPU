// emit_sifive_vcix.rs - SIFIVE VCIX intrinsic emission for RISC-V IME matrix operations
//
// This module emits LLVM IR VCIX intrinsics (sf.vc.v.vvv / sf.vc.v.vvw) to
// implement PTX matrix multiply-accumulate (mma) and matrix load (ldmatrix)
// instructions on RISC-V hardware with the Integrated Matrix Extension (IME).
//
// The VCIX (XSfvcp) interface encodes custom coprocessor instructions in the
// CUSTOM-2 opcode space. Matrix data lives in the standard RISC-V vector
// register file (v0-v31), and tile dimensions are derived from VLEN and SEW
// via vsetvli.
//
// Compilation flow:
//   PTX mma instruction
//     → detect operand types (f16/f32/s8/u8)
//     → choose VCIX format:
//         - sf.vc.v.vvv for same-width accumulate (fp16 → fp16)
//         - sf.vc.v.vvw for widening accumulate (int8 → int32)
//     → emit LLVM IR intrinsic call
//     → LLVM RISC-V backend assembles VCIX custom instruction

use std::collections::HashMap;
use std::fmt::Write;

/// VCIX opcode constants for the 2-bit bit[27:26] field
pub mod vcix_opcodes {
    /// Matrix multiply-accumulate, signed × signed (maps to smt.vmadot SS)
    pub const MMA_SS: u8 = 3;
    /// Matrix multiply-accumulate, unsigned × unsigned
    pub const MMA_UU: u8 = 0;
    /// Matrix multiply-accumulate, signed × unsigned
    pub const MMA_SU: u8 = 2;
    /// Matrix multiply-accumulate, unsigned × signed
    pub const MMA_US: u8 = 1;
}

/// VCIX instruction format types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VcixFormat {
    /// sf.vc.v.vvv: 3 vector operands, same-width accumulate (EMUL=1)
    /// Used for fp16/bf16 matrix multiply where accumulator matches source width
    Vvv,
    /// sf.vc.v.vvw: 2 narrow vector operands + 1 wide accumulator (widening)
    /// Used for int8→int32 matrix multiply
    Vvw,
    /// sf.vc.vvv: 3 vector operands, no output (fire-and-forget side effect)
    VvvNoOutput,
    /// sf.vc.v.xvv: scalar + 2 vector operands (for slide amount in t0)
    Xvv,
}

/// Matrix element types supported by SIFIVE/IME
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SifiveElementType {
    Int4,
    Int8,
    Uint8,
    Int16,
    Uint16,
    Float16,
    Bfloat16,
    Float32,
    Int32,
}

impl SifiveElementType {
    pub fn sew_bits(&self) -> usize {
        match self {
            SifiveElementType::Int4 => 4,
            SifiveElementType::Int8 | SifiveElementType::Uint8 => 8,
            SifiveElementType::Int16
            | SifiveElementType::Uint16
            | SifiveElementType::Float16
            | SifiveElementType::Bfloat16 => 16,
            SifiveElementType::Float32 | SifiveElementType::Int32 => 32,
        }
    }

    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            SifiveElementType::Int4
                | SifiveElementType::Int8
                | SifiveElementType::Int16
                | SifiveElementType::Int32
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(
            self,
            SifiveElementType::Float16 | SifiveElementType::Bfloat16 | SifiveElementType::Float32
        )
    }

    /// LLVM IR type string for scalable vectors
    pub fn llvm_scalar_type(&self) -> &'static str {
        match self {
            SifiveElementType::Int4 => "i4",
            SifiveElementType::Int8 | SifiveElementType::Uint8 => "i8",
            SifiveElementType::Int16 | SifiveElementType::Uint16 => "i16",
            SifiveElementType::Int32 => "i32",
            SifiveElementType::Float16 => "half",
            SifiveElementType::Bfloat16 => "bfloat",
            SifiveElementType::Float32 => "float",
        }
    }
}

/// Hardware configuration for SIFIVE tile dimensions
#[derive(Debug, Clone)]
pub struct SifiveTileConfig {
    /// VLEN in bits
    pub vlen: usize,
    /// Source element SEW in bits
    pub sew: usize,
    /// Tile M dimension
    pub m: usize,
    /// Tile N dimension
    pub n: usize,
    /// Tile K dimension
    pub k: usize,
    /// Whether Copies=2 mode is active
    pub copies2: bool,
}

impl SifiveTileConfig {
    pub fn new(vlen: usize, sew: usize) -> Self {
        let sqrt_val = (vlen as f64 / 64.0).sqrt();
        let copies2 = (sqrt_val - sqrt_val.floor()).abs() > 1e-9;
        let m = sqrt_val.floor() as usize;
        let n = m;
        let k = vlen / (m * sew);

        Self {
            vlen,
            sew,
            m,
            n,
            k,
            copies2,
        }
    }

    /// Number of elements per vector register for the given SEW
    pub fn vl(&self) -> usize {
        self.vlen / self.sew
    }

    /// Number of scalable vector elements (for LLVM nxv type)
    /// For VLEN=256, SEW=8: vl=32, nxv factor = 32/vscale where vscale=VLEN/64
    pub fn nxv_count(&self) -> usize {
        // RISC-V vector types use vscale = VLEN/64
        // So for VLEN=256: vscale=4, and nxv_count = vl/vscale = 32/4 = 8
        // But for the intrinsic types we need the actual element count
        self.vlen / self.sew
    }
}

impl Default for SifiveTileConfig {
    fn default() -> Self {
        Self::new(256, 8)
    }
}

/// Describes a VCIX intrinsic call to emit
#[derive(Debug, Clone)]
pub struct VcixIntrinsicCall {
    /// The LLVM intrinsic name
    pub intrinsic_name: String,
    /// The VCIX opcode (2-bit, bits[27:26])
    pub opcode: u8,
    /// The format
    pub format: VcixFormat,
    /// Source element type
    pub src_type: SifiveElementType,
    /// Accumulator/destination element type
    pub dst_type: SifiveElementType,
    /// Tile configuration
    pub tile: SifiveTileConfig,
}

/// Generate VCIX intrinsic call for a PTX mma instruction.
///
/// Maps PTX mma operand types to the appropriate VCIX format:
/// - int8 sources with int32 accumulator → sf.vc.v.vvw (widening)
/// - fp16/bf16 sources with fp16/bf16 accumulator → sf.vc.v.vvv (same-width)
/// - fp16 sources with fp32 accumulator → sf.vc.v.vvw (widening)
pub fn plan_mma_vcix(
    a_type: SifiveElementType,
    b_type: SifiveElementType,
    c_type: SifiveElementType,
    d_type: SifiveElementType,
    vlen: usize,
) -> VcixIntrinsicCall {
    let src_sew = a_type.sew_bits();
    let dst_sew = d_type.sew_bits();

    let needs_widening = dst_sew > src_sew;
    let is_float = a_type.is_float();

    let format = if needs_widening {
        VcixFormat::Vvw
    } else {
        VcixFormat::Vvv
    };

    let opcode = if is_float {
        // Floating-point uses OPFMMA encoding
        vcix_opcodes::MMA_SS // fp doesn't have sign variants
    } else {
        // Integer: choose sign encoding based on source types
        match (a_type.is_signed(), b_type.is_signed()) {
            (true, true) => vcix_opcodes::MMA_SS,
            (false, false) => vcix_opcodes::MMA_UU,
            (true, false) => vcix_opcodes::MMA_SU,
            (false, true) => vcix_opcodes::MMA_US,
        }
    };

    let tile = SifiveTileConfig::new(vlen, src_sew);

    // Build LLVM intrinsic name
    // For vvv: @llvm.riscv.sf.vc.v.vvv.se.nxv{N}{src_ty}.i64.nxv{N}{src_ty}.nxv{N}{src_ty}.i64
    // For vvw: @llvm.riscv.sf.vc.v.vvw.se.nxv{Nw}{dst_ty}.i64.nxv{N}{src_ty}.nxv{N}{src_ty}.i64
    let src_scalar = a_type.llvm_scalar_type();
    let dst_scalar = d_type.llvm_scalar_type();

    // Calculate nxv counts
    // For RISC-V V: type is <vscale x N x ty> where N * sizeof(ty) * vscale = VLEN/8
    // vscale is a runtime value, but for type naming we use the minimum guarantee
    // nxv count = LMUL * (64 / SEW) for the minimum vscale=1 case
    let src_nxv = 64 / src_sew; // elements per vscale unit
    let dst_nxv = if needs_widening {
        // Wide dest uses EMUL=2, so same nxv count but wider type
        64 / dst_sew
    } else {
        64 / dst_sew
    };

    let intrinsic_name = match format {
        VcixFormat::Vvv => {
            format!(
                "@llvm.riscv.sf.vc.v.vvv.se.nxv{}{}.i64.nxv{}{}.nxv{}{}.i64",
                dst_nxv, dst_scalar, src_nxv, src_scalar, src_nxv, src_scalar
            )
        }
        VcixFormat::Vvw => {
            format!(
                "@llvm.riscv.sf.vc.v.vvw.se.nxv{}{}.i64.nxv{}{}.nxv{}{}.i64",
                dst_nxv, dst_scalar, src_nxv, src_scalar, src_nxv, src_scalar
            )
        }
        _ => unreachable!(),
    };

    VcixIntrinsicCall {
        intrinsic_name,
        opcode,
        format,
        src_type: a_type,
        dst_type: d_type,
        tile,
    }
}

/// Generate LLVM IR text for a VCIX matrix multiply-accumulate call.
///
/// This produces the inline LLVM IR that calls the appropriate
/// `@llvm.riscv.sf.vc.v.{vvv,vvw}.se` intrinsic.
///
/// Arguments:
/// - `result_reg`: SSA name for the result (e.g., "%mma_result")
/// - `acc_reg`: SSA name for the accumulator input (C matrix, passthru)
/// - `a_reg`: SSA name for the A matrix tile
/// - `b_reg`: SSA name for the B matrix tile
/// - `vl_reg`: SSA name for the vector length value
/// - `call`: The planned VCIX intrinsic call
pub fn emit_vcix_mma_ir(
    result_reg: &str,
    acc_reg: &str,
    a_reg: &str,
    b_reg: &str,
    vl_reg: &str,
    call: &VcixIntrinsicCall,
) -> String {
    let src_scalar = call.src_type.llvm_scalar_type();
    let dst_scalar = call.dst_type.llvm_scalar_type();
    let src_nxv = 64 / call.src_type.sew_bits();
    let dst_nxv = 64 / call.dst_type.sew_bits();

    let src_vec_ty = format!("<vscale x {} x {}>", src_nxv, src_scalar);
    let dst_vec_ty = format!("<vscale x {} x {}>", dst_nxv, dst_scalar);

    match call.format {
        VcixFormat::Vvv => {
            format!(
                "{result_reg} = call {dst_vec_ty} {intrinsic}(\
                 i64 {opcode}, {dst_vec_ty} {acc_reg}, {src_vec_ty} {a_reg}, {src_vec_ty} {b_reg}, i64 {vl_reg})",
                result_reg = result_reg,
                dst_vec_ty = dst_vec_ty,
                intrinsic = call.intrinsic_name,
                opcode = call.opcode,
                acc_reg = acc_reg,
                src_vec_ty = src_vec_ty,
                a_reg = a_reg,
                b_reg = b_reg,
                vl_reg = vl_reg,
            )
        }
        VcixFormat::Vvw => {
            format!(
                "{result_reg} = call {dst_vec_ty} {intrinsic}(\
                 i64 {opcode}, {dst_vec_ty} {acc_reg}, {src_vec_ty} {a_reg}, {src_vec_ty} {b_reg}, i64 {vl_reg})",
                result_reg = result_reg,
                dst_vec_ty = dst_vec_ty,
                intrinsic = call.intrinsic_name,
                opcode = call.opcode,
                acc_reg = acc_reg,
                src_vec_ty = src_vec_ty,
                a_reg = a_reg,
                b_reg = b_reg,
                vl_reg = vl_reg,
            )
        }
        _ => unreachable!("Unsupported VCIX format for MMA"),
    }
}

/// Generate LLVM IR text for the intrinsic declarations needed by SIFIVE.
///
/// These declarations must appear at module level for the VCIX intrinsics
/// to be callable.
pub fn emit_vcix_declarations(vlen: usize) -> String {
    let mut decls = String::new();

    writeln!(
        decls,
        "; SIFIVE/VCIX intrinsic declarations for VLEN={}",
        vlen
    )
    .unwrap();
    writeln!(decls).unwrap();

    // Integer int8 → int32 widening (most common for inference)
    writeln!(
        decls,
        "; sf.vc.v.vvw: int8 sources, int32 accumulator (widening MMA)"
    )
    .unwrap();
    writeln!(
        decls,
        "declare <vscale x 2 x i32> @llvm.riscv.sf.vc.v.vvw.se.nxv2i32.i64.nxv8i8.nxv8i8.i64(\
         i64, <vscale x 2 x i32>, <vscale x 8 x i8>, <vscale x 8 x i8>, i64)"
    )
    .unwrap();

    // Integer int8 → int32 widening, side-effect only (no output)
    writeln!(
        decls,
        "declare void @llvm.riscv.sf.vc.vvw.se.i64.nxv2i32.nxv8i8.nxv8i8.i64(\
         i64, <vscale x 2 x i32>, <vscale x 8 x i8>, <vscale x 8 x i8>, i64)"
    )
    .unwrap();

    // Float16 → float16 same-width MMA
    writeln!(
        decls,
        "\n; sf.vc.v.vvv: fp16 sources, fp16 accumulator (same-width MMA)"
    )
    .unwrap();
    writeln!(
        decls,
        "declare <vscale x 4 x half> @llvm.riscv.sf.vc.v.vvv.se.nxv4half.i64.nxv4half.nxv4half.i64(\
         i64, <vscale x 4 x half>, <vscale x 4 x half>, <vscale x 4 x half>, i64)"
    )
    .unwrap();

    // BFloat16 → bfloat16 same-width MMA
    writeln!(decls, "\n; sf.vc.v.vvv: bf16 sources, bf16 accumulator").unwrap();
    writeln!(
        decls,
        "declare <vscale x 4 x bfloat> @llvm.riscv.sf.vc.v.vvv.se.nxv4bfloat.i64.nxv4bfloat.nxv4bfloat.i64(\
         i64, <vscale x 4 x bfloat>, <vscale x 4 x bfloat>, <vscale x 4 x bfloat>, i64)"
    ).unwrap();

    // Standard RVV vector loads/stores for tile movement
    writeln!(decls, "\n; RVV vector load/store for matrix tiles").unwrap();
    writeln!(
        decls,
        "declare <vscale x 8 x i8> @llvm.riscv.vle.nxv8i8.i64(\
         <vscale x 8 x i8>, ptr, i64)"
    )
    .unwrap();
    writeln!(
        decls,
        "declare void @llvm.riscv.vse.nxv8i8.i64(\
         <vscale x 8 x i8>, ptr, i64)"
    )
    .unwrap();
    writeln!(
        decls,
        "declare <vscale x 4 x half> @llvm.riscv.vle.nxv4half.i64(\
         <vscale x 4 x half>, ptr, i64)"
    )
    .unwrap();
    writeln!(
        decls,
        "declare void @llvm.riscv.vse.nxv4half.i64(\
         <vscale x 4 x half>, ptr, i64)"
    )
    .unwrap();

    // Strided loads for non-contiguous matrix rows
    writeln!(decls, "\n; RVV strided load for matrix row access").unwrap();
    writeln!(
        decls,
        "declare <vscale x 8 x i8> @llvm.riscv.vlse.nxv8i8.i64(\
         <vscale x 8 x i8>, ptr, i64, i64)"
    )
    .unwrap();

    decls
}

/// Generate a complete LLVM IR module for a SIFIVE kernel that uses VCIX
/// matrix operations. Used by the compile_bitcode_sifive pipeline.
pub fn emit_sifive_kernel_module(kernel_name: &str, vlen: usize) -> String {
    let tile = SifiveTileConfig::new(vlen, 8);
    let mut module = String::new();

    writeln!(module, "; ModuleID = 'sifive_{}'", kernel_name).unwrap();
    writeln!(module, "source_filename = \"sifive_{}\"", kernel_name).unwrap();
    writeln!(
        module,
        "target datalayout = \"e-m:e-p:64:64-i64:64-i128:128-n64-S128\""
    )
    .unwrap();
    writeln!(module, "target triple = \"riscv64-unknown-elf\"").unwrap();
    writeln!(module).unwrap();

    // Add VCIX intrinsic declarations
    module.push_str(&emit_vcix_declarations(vlen));

    writeln!(module).unwrap();
    writeln!(
        module,
        "; SIFIVE kernel: {}x{}x{} tile MMA (VLEN={}, SEW=8)",
        tile.m, tile.n, tile.k, vlen
    )
    .unwrap();
    writeln!(
        module,
        "define void @{}(ptr %a, ptr %b, ptr %c, i64 %M, i64 %N, i64 %K) {{",
        kernel_name
    )
    .unwrap();
    writeln!(module, "entry:").unwrap();

    // Set vector length
    writeln!(
        module,
        "  ; VL = {} elements (VLEN={}/SEW=8)",
        tile.vl(),
        vlen
    )
    .unwrap();
    writeln!(module, "  %vl = add i64 0, {}", tile.vl()).unwrap();

    // Load tiles
    writeln!(module, "  ; Load A tile").unwrap();
    writeln!(module, "  %a_tile = call <vscale x 8 x i8> @llvm.riscv.vle.nxv8i8.i64(<vscale x 8 x i8> undef, ptr %a, i64 %vl)").unwrap();
    writeln!(module, "  ; Load B tile").unwrap();
    writeln!(module, "  %b_tile = call <vscale x 8 x i8> @llvm.riscv.vle.nxv8i8.i64(<vscale x 8 x i8> undef, ptr %b, i64 %vl)").unwrap();

    // Zero-init accumulator
    writeln!(module, "  ; Zero accumulator").unwrap();
    writeln!(module, "  %c_zero = bitcast i64 0 to i64").unwrap();
    writeln!(module, "  %acc_init = call <vscale x 2 x i32> @llvm.riscv.sf.vc.v.vvw.se.nxv2i32.i64.nxv8i8.nxv8i8.i64(i64 {}, <vscale x 2 x i32> zeroinitializer, <vscale x 8 x i8> %a_tile, <vscale x 8 x i8> %b_tile, i64 %vl)", vcix_opcodes::MMA_SS).unwrap();

    // Store result
    writeln!(module, "  ; Store result (widened int32 accumulator)").unwrap();
    writeln!(module, "  %c_ptr = bitcast ptr %c to ptr").unwrap();
    writeln!(module, "  store <vscale x 2 x i32> %acc_init, ptr %c_ptr").unwrap();
    writeln!(module, "  ret void").unwrap();
    writeln!(module, "}}").unwrap();

    module
}

// =========================================================================
// Zvbdot / Zvldot block dot product instructions for SiFive spike simulation
// =========================================================================
//
// The SiFive riscv-isa-sim-zvfbfa-plus fork supports these extensions:
//   - Zvqbdot8i:   4-element quad block dot product (int8 → int32)
//   - Zvqbdot16i:  4-element quad block dot product (int16 → int32)
//   - Zvfqbdot8f:  4-element FP8 block dot product (fp8 → fp32)
//   - Zvfwbdot16bf: 2-element widening BF16 block dot product (bf16 → fp32)
//   - Zvfbdot32f:  1-element FP32 block dot product (fp32 → fp32)
//   - Zvqdotq:     4-element quad dot product (int8 → int32)
//
// Block dot products use an 8-register group for the B matrix operand,
// with column indexing via vs2 = (base_reg | column_index).
// This provides tile-level matrix operations.
//
// For PTX mma → Zvbdot mapping:
//   mma.m8n8k32.s8.s8.s32 → vqbdots.vv (signed int8 quad block dot)
//   mma.m8n8k32.u8.u8.s32 → vqbdotu.vv (unsigned int8 quad block dot)
//   mma.m16n16k16.bf16.bf16.f32 → vfwbdot.vv (bf16 widening block dot)
//   mma.m16n16k16.f16.f16.f32 → falls back to VCIX or element-wise

/// Zvbdot instruction types supported by the SiFive spike fork
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZvbdotInstr {
    /// vqbdotu.vv - unsigned int8 4-element quad block dot product → int32
    Vqbdotu,
    /// vqbdots.vv - signed int8 4-element quad block dot product → int32
    Vqbdots,
    /// vfqbdot.vv - FP8 (E5M2) 4-element block dot product → FP32
    Vfqbdot,
    /// vfqbdot_alt.vv - FP8 (E4M3) 4-element block dot product → FP32
    VfqbdotAlt,
    /// vfwbdot.vv - BF16 2-element widening block dot product → FP32
    Vfwbdot,
    /// vfbdot.vv - FP32 1-element block dot product → FP32
    Vfbdot,
    /// vqldotu.vv - unsigned int8 4-element lane dot product → int32
    Vqldotu,
    /// vqldots.vv - signed int8 4-element lane dot product → int32
    Vqldots,
    /// vfwldot.vv - BF16 2-element widening lane dot product → FP32
    Vfwldot,
}

impl ZvbdotInstr {
    /// ISA extension required for this instruction
    pub fn required_extension(&self) -> &'static str {
        match self {
            ZvbdotInstr::Vqbdotu | ZvbdotInstr::Vqbdots => "zvqbdot8i",
            ZvbdotInstr::Vfqbdot | ZvbdotInstr::VfqbdotAlt => "zvfqbdot8f",
            ZvbdotInstr::Vfwbdot => "zvfwbdot16bf",
            ZvbdotInstr::Vfbdot => "zvfbdot32f",
            ZvbdotInstr::Vqldotu | ZvbdotInstr::Vqldots => "zvqldot8i",
            ZvbdotInstr::Vfwldot => "zvfwldot16bf",
        }
    }

    /// Assembly mnemonic
    pub fn mnemonic(&self) -> &'static str {
        match self {
            ZvbdotInstr::Vqbdotu => "vqbdotu.vv",
            ZvbdotInstr::Vqbdots => "vqbdots.vv",
            ZvbdotInstr::Vfqbdot => "vfqbdot.vv",
            ZvbdotInstr::VfqbdotAlt => "vfqbdot_alt.vv",
            ZvbdotInstr::Vfwbdot => "vfwbdot.vv",
            ZvbdotInstr::Vfbdot => "vfbdot.vv",
            ZvbdotInstr::Vqldotu => "vqldotu.vv",
            ZvbdotInstr::Vqldots => "vqldots.vv",
            ZvbdotInstr::Vfwldot => "vfwldot.vv",
        }
    }

    /// Source element type
    pub fn src_element_type(&self) -> SifiveElementType {
        match self {
            ZvbdotInstr::Vqbdotu => SifiveElementType::Uint8,
            ZvbdotInstr::Vqbdots => SifiveElementType::Int8,
            ZvbdotInstr::Vfqbdot | ZvbdotInstr::VfqbdotAlt => SifiveElementType::Int8, // OFP8
            ZvbdotInstr::Vfwbdot | ZvbdotInstr::Vfwldot => SifiveElementType::Bfloat16,
            ZvbdotInstr::Vfbdot => SifiveElementType::Float32,
            ZvbdotInstr::Vqldotu => SifiveElementType::Uint8,
            ZvbdotInstr::Vqldots => SifiveElementType::Int8,
        }
    }

    /// Destination/accumulator element type
    pub fn dst_element_type(&self) -> SifiveElementType {
        match self {
            ZvbdotInstr::Vqbdotu
            | ZvbdotInstr::Vqbdots
            | ZvbdotInstr::Vqldotu
            | ZvbdotInstr::Vqldots => SifiveElementType::Int32,
            _ => SifiveElementType::Float32,
        }
    }
}

/// Select the appropriate Zvbdot instruction for a PTX mma operation
pub fn select_zvbdot_instr(
    a_type: SifiveElementType,
    b_type: SifiveElementType,
    d_type: SifiveElementType,
) -> Option<ZvbdotInstr> {
    match (a_type, d_type) {
        // int8 × int8 → int32: quad block dot product
        (SifiveElementType::Int8, SifiveElementType::Int32) => Some(ZvbdotInstr::Vqbdots),
        (SifiveElementType::Uint8, SifiveElementType::Int32) => Some(ZvbdotInstr::Vqbdotu),

        // bf16 × bf16 → fp32: widening block dot product
        (SifiveElementType::Bfloat16, SifiveElementType::Float32) => Some(ZvbdotInstr::Vfwbdot),

        // fp32 × fp32 → fp32: block dot product
        (SifiveElementType::Float32, SifiveElementType::Float32) => Some(ZvbdotInstr::Vfbdot),

        // fp16 → fp32: no direct Zvbdot, would need conversion + vfwbdot
        (SifiveElementType::Float16, SifiveElementType::Float32) => None,

        _ => None,
    }
}

/// Describes a Zvbdot intrinsic call to emit in LLVM IR
#[derive(Debug, Clone)]
pub struct ZvbdotCall {
    /// The Zvbdot instruction to use
    pub instr: ZvbdotInstr,
    /// LLVM IR intrinsic name (if available) or inline asm
    pub emit_mode: ZvbdotEmitMode,
    /// Source element type
    pub src_type: SifiveElementType,
    /// Dest element type
    pub dst_type: SifiveElementType,
    /// Tile config
    pub tile: SifiveTileConfig,
}

/// How to emit the Zvbdot instruction in LLVM IR
#[derive(Debug, Clone)]
pub enum ZvbdotEmitMode {
    /// Use LLVM intrinsic (if the LLVM build has the extension)
    Intrinsic(String),
    /// Use inline assembly (works with any LLVM)
    InlineAsm(String),
}

/// Plan a Zvbdot instruction for a PTX mma operation (simulation path)
pub fn plan_mma_zvbdot(
    a_type: SifiveElementType,
    b_type: SifiveElementType,
    c_type: SifiveElementType,
    d_type: SifiveElementType,
    vlen: usize,
) -> Option<ZvbdotCall> {
    let instr = select_zvbdot_instr(a_type, b_type, d_type)?;
    let tile = SifiveTileConfig::new(vlen, a_type.sew_bits());

    // Since the Zvbdot extensions are new and may not have LLVM intrinsics yet,
    // we use inline assembly as the primary emission path
    let asm_template = format!(
        "{mnemonic} {{{{$0}}}}, {{{{$1}}}}, {{{{$2}}}}",
        mnemonic = instr.mnemonic()
    );

    Some(ZvbdotCall {
        instr,
        emit_mode: ZvbdotEmitMode::InlineAsm(asm_template),
        src_type: a_type,
        dst_type: d_type,
        tile,
    })
}

/// Generate LLVM IR inline assembly for a Zvbdot block dot product.
///
/// Since these instructions are new SiFive extensions not yet in upstream LLVM,
/// we emit them as inline assembly that the SiFive spike fork can simulate.
pub fn emit_zvbdot_inline_asm(
    result_reg: &str,
    acc_reg: &str,
    a_reg: &str,
    b_reg: &str,
    call: &ZvbdotCall,
) -> String {
    let src_scalar = call.src_type.llvm_scalar_type();
    let dst_scalar = call.dst_type.llvm_scalar_type();
    let src_nxv = 64 / call.src_type.sew_bits();
    let dst_nxv = 64 / call.dst_type.sew_bits();

    let src_vec_ty = format!("<vscale x {} x {}>", src_nxv, src_scalar);
    let dst_vec_ty = format!("<vscale x {} x {}>", dst_nxv, dst_scalar);

    // Emit inline asm: the Zvbdot instructions use standard V-extension encoding
    // vd = vd + bdot(vs2, vs1) where vs2 uses 8-register group block access
    format!(
        "{result} = call {dst_ty} asm sideeffect \"{mnemonic} $0, $1, $2\", \
         \"=&v,v,v,0\"({src_ty} {a}, {src_ty} {b}, {dst_ty} {acc})",
        result = result_reg,
        dst_ty = dst_vec_ty,
        src_ty = src_vec_ty,
        mnemonic = call.instr.mnemonic(),
        a = a_reg,
        b = b_reg,
        acc = acc_reg,
    )
}

/// Generate a SIFIVE kernel module using Zvbdot instructions (for spike simulation).
pub fn emit_sifive_zvbdot_kernel_module(
    kernel_name: &str,
    vlen: usize,
    a_type: SifiveElementType,
    d_type: SifiveElementType,
) -> Option<String> {
    let instr = select_zvbdot_instr(a_type, a_type, d_type)?;
    let tile = SifiveTileConfig::new(vlen, a_type.sew_bits());
    let src_scalar = a_type.llvm_scalar_type();
    let dst_scalar = d_type.llvm_scalar_type();
    let src_nxv = 64 / a_type.sew_bits();
    let dst_nxv = 64 / d_type.sew_bits();

    let src_vec_ty = format!("<vscale x {} x {}>", src_nxv, src_scalar);
    let dst_vec_ty = format!("<vscale x {} x {}>", dst_nxv, dst_scalar);

    let mut module = String::new();

    writeln!(module, "; ModuleID = 'sifive_zvbdot_{}'", kernel_name).unwrap();
    writeln!(
        module,
        "source_filename = \"sifive_zvbdot_{}\"",
        kernel_name
    )
    .unwrap();
    writeln!(
        module,
        "target datalayout = \"e-m:e-p:64:64-i64:64-i128:128-n64-S128\""
    )
    .unwrap();
    writeln!(module, "target triple = \"riscv64-unknown-elf\"").unwrap();
    writeln!(module).unwrap();

    // Standard RVV load/store declarations
    writeln!(module, "; RVV vector load/store").unwrap();
    writeln!(
        module,
        "declare {} @llvm.riscv.vle.nxv{}{}.i64({}, ptr, i64)",
        src_vec_ty, src_nxv, src_scalar, src_vec_ty
    )
    .unwrap();
    writeln!(
        module,
        "declare void @llvm.riscv.vse.nxv{}{}.i64({}, ptr, i64)",
        dst_nxv, dst_scalar, dst_vec_ty
    )
    .unwrap();
    writeln!(module).unwrap();

    writeln!(
        module,
        "; SIFIVE/X390 Zvbdot kernel: {} instruction, VLEN={}, tile={}x{}x{}",
        instr.mnemonic(),
        vlen,
        tile.m,
        tile.n,
        tile.k
    )
    .unwrap();
    writeln!(
        module,
        "define void @{}(ptr %a, ptr %b, ptr %c, i64 %vl) {{",
        kernel_name
    )
    .unwrap();
    writeln!(module, "entry:").unwrap();

    // Load A and B tiles
    writeln!(
        module,
        "  %a_tile = call {} @llvm.riscv.vle.nxv{}{}.i64({} undef, ptr %a, i64 %vl)",
        src_vec_ty, src_nxv, src_scalar, src_vec_ty
    )
    .unwrap();
    writeln!(
        module,
        "  %b_tile = call {} @llvm.riscv.vle.nxv{}{}.i64({} undef, ptr %b, i64 %vl)",
        src_vec_ty, src_nxv, src_scalar, src_vec_ty
    )
    .unwrap();

    // Zvbdot inline asm
    writeln!(module, "  ; {} block dot product", instr.mnemonic()).unwrap();
    writeln!(
        module,
        "  %result = call {} asm sideeffect \"{} $0, $1, $2\", \"=&v,v,v\"({} %a_tile, {} %b_tile)",
        dst_vec_ty,
        instr.mnemonic(),
        src_vec_ty,
        src_vec_ty
    )
    .unwrap();

    // Store result
    writeln!(
        module,
        "  call void @llvm.riscv.vse.nxv{}{}.i64({} %result, ptr %c, i64 %vl)",
        dst_nxv, dst_scalar, dst_vec_ty
    )
    .unwrap();
    writeln!(module, "  ret void").unwrap();
    writeln!(module, "}}").unwrap();

    Some(module)
}

/// Convert PTX ScalarType to SifiveElementType
pub fn ptx_scalar_to_sifive(scalar_type: u32) -> SifiveElementType {
    // Map based on PTX scalar type enum values
    // These match the ptx_parser::ast::ScalarType variants
    match scalar_type {
        // U8
        0 => SifiveElementType::Uint8,
        // S8
        1 => SifiveElementType::Int8,
        // U16
        4 => SifiveElementType::Uint16,
        // S16
        5 => SifiveElementType::Int16,
        // U32
        8 => SifiveElementType::Uint8, // fallback
        // S32
        9 => SifiveElementType::Int32,
        // F16
        12 => SifiveElementType::Float16,
        // F32
        14 => SifiveElementType::Float32,
        // BF16
        18 => SifiveElementType::Bfloat16,
        _ => SifiveElementType::Int8, // default fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_mma_int8_widening() {
        let call = plan_mma_vcix(
            SifiveElementType::Int8,
            SifiveElementType::Int8,
            SifiveElementType::Int32,
            SifiveElementType::Int32,
            256,
        );
        assert_eq!(call.format, VcixFormat::Vvw);
        assert_eq!(call.opcode, vcix_opcodes::MMA_SS);
        assert!(call.intrinsic_name.contains("vvw"));
        assert_eq!(call.tile.m, 2);
        assert_eq!(call.tile.n, 2);
        assert_eq!(call.tile.k, 16);
    }

    #[test]
    fn test_plan_mma_fp16_same_width() {
        let call = plan_mma_vcix(
            SifiveElementType::Float16,
            SifiveElementType::Float16,
            SifiveElementType::Float16,
            SifiveElementType::Float16,
            256,
        );
        assert_eq!(call.format, VcixFormat::Vvv);
        assert!(call.intrinsic_name.contains("vvv"));
    }

    #[test]
    fn test_plan_mma_uint8() {
        let call = plan_mma_vcix(
            SifiveElementType::Uint8,
            SifiveElementType::Uint8,
            SifiveElementType::Int32,
            SifiveElementType::Int32,
            256,
        );
        assert_eq!(call.opcode, vcix_opcodes::MMA_UU);
        assert_eq!(call.format, VcixFormat::Vvw);
    }

    #[test]
    fn test_emit_vcix_ir() {
        let call = plan_mma_vcix(
            SifiveElementType::Int8,
            SifiveElementType::Int8,
            SifiveElementType::Int32,
            SifiveElementType::Int32,
            256,
        );
        let ir = emit_vcix_mma_ir("%result", "%acc", "%a", "%b", "%vl", &call);
        assert!(ir.contains("@llvm.riscv.sf.vc.v.vvw.se"));
        assert!(ir.contains("i64 3")); // opcode = MMA_SS = 3
        assert!(ir.contains("%acc"));
        assert!(ir.contains("%a"));
        assert!(ir.contains("%b"));
    }

    #[test]
    fn test_tile_config_vlen256() {
        let tile = SifiveTileConfig::new(256, 8);
        assert_eq!(tile.m, 2);
        assert_eq!(tile.n, 2);
        assert_eq!(tile.k, 16);
        assert!(!tile.copies2);
        assert_eq!(tile.vl(), 32);
    }

    #[test]
    fn test_tile_config_vlen128() {
        let tile = SifiveTileConfig::new(128, 8);
        assert_eq!(tile.m, 1);
        assert_eq!(tile.k, 16);
        assert!(tile.copies2); // sqrt(128/64) = sqrt(2) is not integer
    }

    #[test]
    fn test_emit_kernel_module() {
        let module = emit_sifive_kernel_module("test_matmul", 256);
        assert!(module.contains("target triple = \"riscv64-unknown-elf\""));
        assert!(module.contains("@llvm.riscv.sf.vc.v.vvw.se"));
        assert!(module.contains("@llvm.riscv.vle.nxv8i8"));
        assert!(module.contains("define void @test_matmul"));
    }

    #[test]
    fn test_declarations() {
        let decls = emit_vcix_declarations(256);
        assert!(decls.contains("sf.vc.v.vvw.se"));
        assert!(decls.contains("sf.vc.v.vvv.se"));
        assert!(decls.contains("vle.nxv8i8"));
        assert!(decls.contains("vse.nxv8i8"));
    }

    // --- Zvbdot tests ---

    #[test]
    fn test_select_zvbdot_int8_signed() {
        let instr = select_zvbdot_instr(
            SifiveElementType::Int8,
            SifiveElementType::Int8,
            SifiveElementType::Int32,
        );
        assert_eq!(instr, Some(ZvbdotInstr::Vqbdots));
    }

    #[test]
    fn test_select_zvbdot_int8_unsigned() {
        let instr = select_zvbdot_instr(
            SifiveElementType::Uint8,
            SifiveElementType::Uint8,
            SifiveElementType::Int32,
        );
        assert_eq!(instr, Some(ZvbdotInstr::Vqbdotu));
    }

    #[test]
    fn test_select_zvbdot_bf16() {
        let instr = select_zvbdot_instr(
            SifiveElementType::Bfloat16,
            SifiveElementType::Bfloat16,
            SifiveElementType::Float32,
        );
        assert_eq!(instr, Some(ZvbdotInstr::Vfwbdot));
    }

    #[test]
    fn test_select_zvbdot_fp16_unsupported() {
        let instr = select_zvbdot_instr(
            SifiveElementType::Float16,
            SifiveElementType::Float16,
            SifiveElementType::Float32,
        );
        assert_eq!(instr, None);
    }

    #[test]
    fn test_plan_mma_zvbdot_int8() {
        let call = plan_mma_zvbdot(
            SifiveElementType::Int8,
            SifiveElementType::Int8,
            SifiveElementType::Int32,
            SifiveElementType::Int32,
            512,
        );
        assert!(call.is_some());
        let call = call.unwrap();
        assert_eq!(call.instr, ZvbdotInstr::Vqbdots);
        assert_eq!(call.instr.required_extension(), "zvqbdot8i");
    }

    #[test]
    fn test_emit_zvbdot_inline_asm() {
        let call = plan_mma_zvbdot(
            SifiveElementType::Int8,
            SifiveElementType::Int8,
            SifiveElementType::Int32,
            SifiveElementType::Int32,
            512,
        )
        .unwrap();
        let ir = emit_zvbdot_inline_asm("%result", "%acc", "%a", "%b", &call);
        assert!(ir.contains("vqbdots.vv"));
        assert!(ir.contains("asm sideeffect"));
        assert!(ir.contains("%a"));
        assert!(ir.contains("%b"));
    }

    #[test]
    fn test_emit_zvbdot_kernel_module() {
        let module = emit_sifive_zvbdot_kernel_module(
            "test_bdot",
            512,
            SifiveElementType::Int8,
            SifiveElementType::Int32,
        );
        assert!(module.is_some());
        let module = module.unwrap();
        assert!(module.contains("vqbdots.vv"));
        assert!(module.contains("target triple = \"riscv64-unknown-elf\""));
        assert!(module.contains("define void @test_bdot"));
    }

    #[test]
    fn test_tile_config_vlen512() {
        let tile = SifiveTileConfig::new(512, 8);
        // sqrt(512/64) = sqrt(8) ≈ 2.83 → floor = 2
        assert_eq!(tile.m, 2);
        assert_eq!(tile.n, 2);
        assert_eq!(tile.k, 32); // 512 / (2 * 8) = 32
        assert!(tile.copies2); // sqrt(8) is not integer
        assert_eq!(tile.vl(), 64); // 512 / 8
    }
}
