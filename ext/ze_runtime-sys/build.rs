use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let src_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?).join("src");

    cc::Build::new()
        .file(src_dir.join("runner/ze_runner.c"))
        .include(src_dir.clone())
        .compile("ze_runner");

    if cfg!(windows) {
        println!("cargo:rustc-link-lib=dylib=ze_loader_1");
        let env_val = env::var("CARGO_CFG_TARGET_ENV")?;
        if env_val == "msvc" {
            let mut path = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
            path.push("lib");
            println!("cargo:rustc-link-search=native={}", path.display());
        } else {
            println!("cargo:rustc-link-search=native=C:\\Windows\\System32");
        }
    } else {
        // Only link ze_loader if it is actually present on this system.
        // On non-Intel platforms (RISC-V, ARM, etc.) ze_loader is not installed,
        // and building sifive/gemmini/tenstorrent features should not fail because of it.
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
        let manifest_lib_dir = manifest_dir.join("lib");
        let search_dirs = [
            manifest_lib_dir.as_path(),
            Path::new("/usr/lib/x86_64-linux-gnu"),
            Path::new("/usr/lib/aarch64-linux-gnu"),
            Path::new("/usr/local/lib"),
            Path::new("/usr/lib"),
        ];
        let unversioned = search_dirs
            .iter()
            .map(|d| d.join("libze_loader.so"))
            .find(|p| p.exists());
        let versioned = search_dirs
            .iter()
            .map(|d| d.join("libze_loader.so.1"))
            .find(|p| p.exists());
        if let Some(loader) = unversioned {
            println!(
                "cargo:rustc-link-search=native={}",
                loader.parent().expect("loader path has parent").display()
            );
            println!("cargo:rustc-link-lib=dylib=ze_loader");
        } else if let Some(loader) = versioned {
            let out_dir = PathBuf::from(env::var("OUT_DIR")?);
            let link_name = out_dir.join("libze_loader.so");
            let _ = fs::remove_file(&link_name);
            #[cfg(unix)]
            {
                use std::os::unix::fs::symlink;
                symlink(&loader, &link_name)
                    .or_else(|_| fs::copy(&loader, &link_name).map(|_| ()))?;
            }
            #[cfg(not(unix))]
            {
                fs::copy(&loader, &link_name)?;
            }
            println!("cargo:rustc-link-search=native={}", out_dir.display());
            println!("cargo:rustc-link-lib=dylib=ze_loader");
        } else {
            cc::Build::new()
                .file(src_dir.join("runner/ze_stub.c"))
                .include(src_dir.clone())
                .compile("ze_loader_stub");

            // Provide local failing stubs so cdylibs can be loaded on systems
            // without Level Zero. Consumers that use another backend should
            // not be blocked by unresolved ze_* symbols.
            println!("cargo:warning=ze_loader not found — Intel Level Zero runtime disabled");
        }
    }

    println!("cargo:rerun-if-changed=src/runner/ze_runner.c");
    println!("cargo:rerun-if-changed=src/runner/ze_runner.h");
    println!("cargo:rerun-if-changed=src/runner/ze_stub.c");
    println!("cargo:rerun-if-changed=src/level-zero/ze_api.h");

    Ok(())
}
