// sifive_benchmark.rs - Benchmark for SIFIVE (RISC-V IME) PTX compilation pipeline
//
// Benchmarks:
// 1. PTX parsing time
// 2. PTX → LLVM IR compilation time (per-pass breakdown)
// 3. LLVM IR → RISC-V object code time (via llc)
// 4. Zvbdot kernel module generation time
// 5. Spike simulation time (if spike is available)
//
// Usage:
//   cargo run --example sifive_benchmark --features sifive [-- --spike]

use std::time::Instant;

fn main() {
    let run_spike = std::env::args().any(|a| a == "--spike");

    println!("=== SIFIVE (RISC-V IME) Benchmark ===\n");

    // --- Benchmark 1: PTX parsing ---
    bench_ptx_parsing();

    // --- Benchmark 2: PTX → LLVM IR compilation ---
    bench_ptx_to_llvm();

    // --- Benchmark 3: VCIX intrinsic planning ---
    bench_vcix_planning();

    // --- Benchmark 4: Zvbdot instruction selection ---
    bench_zvbdot_selection();

    // --- Benchmark 5: LLVM IR module generation ---
    bench_module_generation();

    // --- Benchmark 6: Spike simulation (optional) ---
    if run_spike {
        bench_spike_simulation();
    } else {
        println!("[6] Spike simulation: skipped (pass --spike to enable)");
    }

    println!("\n=== Benchmark complete ===");
}

/// Benchmark PTX parsing of a vector_add kernel
fn bench_ptx_parsing() {
    let ptx = VECTOR_ADD_PTX;
    let iterations = 1000;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = ptx_parser::parse_module_checked(ptx);
    }
    let elapsed = start.elapsed();

    println!(
        "[1] PTX parsing (vector_add, {}x): {:.2}ms total, {:.2}us/iter",
        iterations,
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64,
    );
}

/// Benchmark full PTX → LLVM IR pipeline with per-pass timing
fn bench_ptx_to_llvm() {
    let ptx = VECTOR_ADD_PTX;
    let iterations = 100;

    let _ast = ptx_parser::parse_module_checked(ptx).expect("parse failed");

    // Single run with per-pass timing
    let mut pass_times: Vec<(String, f64)> = Vec::new();
    let ast_single = ptx_parser::parse_module_checked(ptx).unwrap();
    let _ = ptx::pass::to_llvm_module(
        ast_single,
        ptx::pass::Attributes {
            clock_rate: 1000000,
            emit_debug_info: false,
        },
        |pass_name| {
            // Collect pass name (timing is approximate here)
            pass_times.push((pass_name.to_string(), 0.0));
        },
    );

    // Timed runs
    let start = Instant::now();
    for _ in 0..iterations {
        let ast = ptx_parser::parse_module_checked(ptx).unwrap();
        let _ = ptx::pass::to_llvm_module(
            ast,
            ptx::pass::Attributes {
                clock_rate: 1000000,
                emit_debug_info: false,
            },
            |_| {},
        );
    }
    let elapsed = start.elapsed();

    println!(
        "[2] PTX → LLVM IR (vector_add, {}x): {:.2}ms total, {:.2}ms/iter",
        iterations,
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / iterations as f64,
    );
    println!(
        "    Passes: {}",
        pass_times
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join(" → ")
    );
}

/// Benchmark VCIX intrinsic call planning
fn bench_vcix_planning() {
    use ptx::pass::emit_sifive_vcix::*;

    let iterations = 100_000;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = plan_mma_vcix(
            SifiveElementType::Int8,
            SifiveElementType::Int8,
            SifiveElementType::Int32,
            SifiveElementType::Int32,
            512,
        );
    }
    let elapsed = start.elapsed();

    println!(
        "[3] VCIX MMA planning (int8→int32, {}x): {:.2}ms total, {:.0}ns/iter",
        iterations,
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_nanos() as f64 / iterations as f64,
    );

    // Also benchmark fp16 and bf16 variants
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = plan_mma_vcix(
            SifiveElementType::Float16,
            SifiveElementType::Float16,
            SifiveElementType::Float16,
            SifiveElementType::Float16,
            512,
        );
    }
    let fp16_ns = start.elapsed().as_nanos() as f64 / iterations as f64;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = plan_mma_vcix(
            SifiveElementType::Bfloat16,
            SifiveElementType::Bfloat16,
            SifiveElementType::Float32,
            SifiveElementType::Float32,
            512,
        );
    }
    let bf16_ns = start.elapsed().as_nanos() as f64 / iterations as f64;

    println!(
        "    Variants: int8→i32: {:.0}ns, fp16→fp16: {:.0}ns, bf16→f32: {:.0}ns",
        elapsed.as_nanos() as f64 / iterations as f64,
        fp16_ns,
        bf16_ns,
    );
}

/// Benchmark Zvbdot instruction selection
fn bench_zvbdot_selection() {
    use ptx::pass::emit_sifive_vcix::*;

    let iterations = 100_000;

    let types = [
        (
            SifiveElementType::Int8,
            SifiveElementType::Int32,
            "int8→i32",
        ),
        (
            SifiveElementType::Uint8,
            SifiveElementType::Int32,
            "uint8→i32",
        ),
        (
            SifiveElementType::Bfloat16,
            SifiveElementType::Float32,
            "bf16→f32",
        ),
        (
            SifiveElementType::Float32,
            SifiveElementType::Float32,
            "f32→f32",
        ),
        (
            SifiveElementType::Float16,
            SifiveElementType::Float32,
            "fp16→f32 (unsupported)",
        ),
    ];

    println!("[4] Zvbdot instruction selection ({}x each):", iterations);
    for (src, dst, label) in &types {
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = select_zvbdot_instr(*src, *src, *dst);
        }
        let ns = start.elapsed().as_nanos() as f64 / iterations as f64;
        let instr = select_zvbdot_instr(*src, *src, *dst);
        println!(
            "    {}: {:.0}ns → {}",
            label,
            ns,
            instr.map_or("None".to_string(), |i| i.mnemonic().to_string()),
        );
    }
}

/// Benchmark LLVM IR module text generation
fn bench_module_generation() {
    use ptx::pass::emit_sifive_vcix::*;

    let iterations = 10_000;

    // VCIX kernel module
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = emit_sifive_kernel_module("bench_matmul", 512);
    }
    let vcix_us = start.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64;

    // Zvbdot kernel module
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = emit_sifive_zvbdot_kernel_module(
            "bench_matmul",
            512,
            SifiveElementType::Int8,
            SifiveElementType::Int32,
        );
    }
    let zvbdot_us = start.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64;

    // VCIX declarations
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = emit_vcix_declarations(512);
    }
    let decl_us = start.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64;

    println!(
        "[5] LLVM IR module generation ({}x): VCIX kernel: {:.1}us, Zvbdot kernel: {:.1}us, declarations: {:.1}us",
        iterations, vcix_us, zvbdot_us, decl_us,
    );

    // Print module sizes
    let vcix_module = emit_sifive_kernel_module("bench", 512);
    let zvbdot_module = emit_sifive_zvbdot_kernel_module(
        "bench",
        512,
        SifiveElementType::Int8,
        SifiveElementType::Int32,
    )
    .unwrap_or_default();
    println!(
        "    Module sizes: VCIX: {} bytes, Zvbdot: {} bytes",
        vcix_module.len(),
        zvbdot_module.len(),
    );
}

/// Benchmark spike simulation (requires spike in PATH)
fn bench_spike_simulation() {
    use ptx::pass::emit_sifive_vcix::*;
    use std::process::Command;

    // Check if spike is available
    let spike_available = Command::new("spike")
        .arg("--help")
        .output()
        .map(|o| !o.stderr.is_empty() || !o.stdout.is_empty())
        .unwrap_or(false);

    if !spike_available {
        println!("[6] Spike simulation: spike not found in PATH");
        return;
    }

    // Generate a test kernel module
    let module_ir = emit_sifive_kernel_module("spike_bench", 512);

    // Write to temp file
    let tmp_dir = tempfile::TempDir::new().expect("tmpdir");
    let ir_path = tmp_dir.path().join("spike_bench.ll");
    let asm_path = tmp_dir.path().join("spike_bench.s");
    let obj_path = tmp_dir.path().join("spike_bench.o");
    let elf_path = tmp_dir.path().join("spike_bench.elf");

    std::fs::write(&ir_path, &module_ir).expect("write IR");

    // Step 1: LLC compilation
    let start = Instant::now();
    let llc_result = Command::new("llc")
        .arg(&ir_path)
        .arg("-o")
        .arg(&asm_path)
        .arg("-march=riscv64")
        .arg("-mattr=+v,+d,+f,+zfbfmin,+zvfbfmin,+zvfbfwma,+zvl512b")
        .arg("-filetype=asm")
        .output();
    let llc_ms = start.elapsed().as_secs_f64() * 1000.0;

    match llc_result {
        Ok(o) if o.status.success() => {
            println!("[6] LLC compilation: {:.2}ms", llc_ms);
        }
        Ok(o) => {
            println!("[6] LLC failed: {}", String::from_utf8_lossy(&o.stderr));
            return;
        }
        Err(e) => {
            println!("[6] LLC not found: {}", e);
            return;
        }
    }

    // Step 2: Assembly
    let start = Instant::now();
    let as_ok = Command::new("riscv64-unknown-elf-as")
        .arg(&asm_path)
        .arg("-o")
        .arg(&obj_path)
        .arg("-march=rv64gcv_zvl512b")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    let as_ms = start.elapsed().as_secs_f64() * 1000.0;

    if as_ok {
        println!("    Assembly: {:.2}ms", as_ms);
    } else {
        println!("    Assembly: riscv64-unknown-elf-as not found");
        return;
    }

    // Step 3: Spike simulation
    let isa = "rv64gcv_zfbfmin_zvfbfmin_zvfbfwma_zvfbfa_zvqbdot8i_zvfqbdot8f_zvfwbdot16bf_zvl512b";

    let start = Instant::now();
    let spike_result = Command::new("spike")
        .arg(format!("--isa={}", isa))
        .arg(format!("--varch=vlen:512,elen:64"))
        .arg("pk")
        .arg(&elf_path)
        .output();
    let spike_ms = start.elapsed().as_secs_f64() * 1000.0;

    match spike_result {
        Ok(o) if o.status.success() => {
            println!("    Spike simulation: {:.2}ms", spike_ms);
            println!(
                "    Total (LLC + asm + spike): {:.2}ms",
                llc_ms + as_ms + spike_ms
            );
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            println!(
                "    Spike exited with code {}: {}",
                o.status,
                &stderr[..stderr.len().min(200)]
            );
        }
        Err(e) => {
            println!("    Spike error: {}", e);
        }
    }
}

// --- Test PTX kernels ---

const VECTOR_ADD_PTX: &str = "\
.version 7.0
.target sm_75
.address_size 64

.visible .entry vector_add(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr
)
{
    .reg .f32 %r<4>;
    .reg .u64 %rd<4>;

    ld.param.u64 %rd1, [a_ptr];
    ld.param.u64 %rd2, [b_ptr];
    ld.param.u64 %rd3, [c_ptr];

    ld.global.f32 %r1, [%rd1];
    ld.global.f32 %r2, [%rd2];

    add.f32 %r3, %r1, %r2;

    st.global.f32 [%rd3], %r3;

    ret;
}
";
