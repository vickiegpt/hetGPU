use sifive_runtime_sys::{
    sifive_CreateDevice, sifive_CreateKernelOnDevice, sifive_CreateProgram, sifive_DestroyDevice,
    sifive_DestroyKernel, sifive_LaunchKernel, sifive_LoadProgram, SifiveComm, SifiveReduceOp,
};
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let real_io = std::env::var("HETGPU_SIFIVE_REAL_IO").ok().as_deref() == Some("1");
    if !real_io {
        std::env::set_var("HETGPU_SIFIVE_DRY_RUN", "1");
        std::env::set_var("HETGPU_SIFIVE_SKIP_HW_REDUCE", "1");
        println!(
            "SIFIVE smoke using dry-run io; set HETGPU_SIFIVE_REAL_IO=1 for driver submission"
        );
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = manifest_dir.join("examples/sifive_gemm_kernel.c");
    let elf = manifest_dir.join("target/sifive_gemm_kernel.elf");

    compile_kernel(&source, &elf)?;
    let elf_bytes = std::fs::read(&elf)?;
    println!("compiled custom SIFIVE ELF: {} bytes", elf_bytes.len());

    let mut launched = 0usize;
    for device_id in 0..4u32 {
        match launch_on_device(device_id, &elf_bytes) {
            Ok(()) => {
                println!("sifive{} launch submitted", device_id);
                launched += 1;
            }
            Err(e) => {
                eprintln!("sifive{} launch failed: {}", device_id, e);
                if strict() {
                    return Err(e);
                }
            }
        }
    }

    if launched != 4 && strict() {
        return Err(format!("strict mode expected 4 launches, got {}", launched).into());
    }

    let input = [1.0f32, 2.0, 3.0, 4.0];
    let mut output = [0.0f32; 4];
    match SifiveComm::init_all()
        .and_then(|comm| comm.all_reduce(&input, &mut output, SifiveReduceOp::Sum))
    {
        Ok(()) => println!("SIFIVE communicator reduce smoke: {:?}", output),
        Err(e) => {
            eprintln!("SIFIVE communicator reduce failed: {}", e);
            if strict() {
                return Err(format!("reduce failed: {}", e).into());
            }
        }
    }

    println!(
        "four-SIFIVE GEMM/reduce smoke complete: {} launch attempts succeeded",
        launched
    );
    Ok(())
}

fn compile_kernel(source: &Path, elf: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = elf.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let compiler =
        std::env::var("SIFIVE_CC").unwrap_or_else(|_| "riscv64-linux-gnu-gcc".to_string());
    let status = Command::new(&compiler)
        .arg("-nostdlib")
        .arg("-static")
        .arg("-march=rv64gcv")
        .arg("-mabi=lp64d")
        .arg("-Wl,-Ttext=0x30080000")
        .arg("-Wl,-e,_start")
        .arg("-o")
        .arg(elf)
        .arg(source)
        .status()?;

    if !status.success() {
        return Err(format!("{} failed with {}", compiler, status).into());
    }

    Ok(())
}

fn launch_on_device(device_id: u32, elf_bytes: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    unsafe {
        let dev = sifive_CreateDevice(device_id);
        if dev.is_null() {
            return Err(format!("sifive_CreateDevice({}) returned null", device_id).into());
        }

        let program = sifive_CreateProgram();
        if program.is_null() {
            sifive_DestroyDevice(dev);
            return Err("sifive_CreateProgram returned null".into());
        }

        let load_result =
            sifive_LoadProgram(program, elf_bytes.as_ptr().cast(), elf_bytes.len() as u64);
        if load_result != 0 {
            sifive_DestroyDevice(dev);
            return Err(format!("sifive_LoadProgram failed: {}", load_result).into());
        }

        let kernel_name = CString::new("sifive_gemm_smoke")?;
        let kernel = sifive_CreateKernelOnDevice(program, dev, kernel_name.as_ptr());
        if kernel.is_null() {
            sifive_DestroyDevice(dev);
            return Err("sifive_CreateKernelOnDevice returned null".into());
        }

        let launch_result = sifive_LaunchKernel(kernel, 1, 1, 1, 1, 1, 1);

        if launch_result != 0 {
            return Err(format!("sifive_LaunchKernel failed: {}", launch_result).into());
        }

        sifive_DestroyKernel(kernel);
        sifive_DestroyDevice(dev);
    }

    Ok(())
}

fn strict() -> bool {
    std::env::var("HETGPU_SIFIVE_STRICT").ok().as_deref() == Some("1")
}
