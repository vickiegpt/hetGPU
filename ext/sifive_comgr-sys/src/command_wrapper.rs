use lazy_static::lazy_static;
use std::collections::HashMap;
use std::ffi::CString;
use std::fs;
use std::io;
use std::mem;
use std::os::raw::c_void;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::Mutex;
use tempfile::{Builder, TempDir, tempdir};

use crate::{
    sifive_comgr_action_info_t, sifive_comgr_action_kind_s, sifive_comgr_create_data,
    sifive_comgr_data_kind_s, sifive_comgr_data_set_add, sifive_comgr_data_set_bytes,
    sifive_comgr_data_set_name, sifive_comgr_data_set_t, sifive_comgr_language_s,
    sifive_comgr_release_data, sifive_comgr_status_s, sifive_comgr_status_t,
};

pub(crate) struct DataContent {
    pub(crate) kind: sifive_comgr_data_kind_s,
    pub(crate) content: Vec<u8>,
    pub(crate) name: Option<String>,
}

pub(crate) type DataMap = HashMap<u64, DataContent>;

lazy_static! {
    pub(crate) static ref DATA_STORE: Mutex<DataMap> = Mutex::new(HashMap::new());
    pub(crate) static ref DATA_SET_STORE: Mutex<HashMap<u64, Vec<u64>>> =
        Mutex::new(HashMap::new());
    pub(crate) static ref ACTION_INFO_STORE: Mutex<HashMap<u64, ActionInfo>> =
        Mutex::new(HashMap::new());
    pub(crate) static ref NEXT_HANDLE: Mutex<u64> = Mutex::new(1);
}

pub(crate) fn get_next_handle() -> u64 {
    let mut handle = NEXT_HANDLE.lock().unwrap();
    let current = *handle;
    *handle += 1;
    current
}

#[derive(Default, Clone)]
pub(crate) struct ActionInfo {
    pub(crate) language: Option<sifive_comgr_language_s>,
    pub(crate) options: Vec<String>,
    pub(crate) working_directory: Option<String>,
    pub(crate) target: Option<String>,
}

struct ActionContext {
    temp_dir: PathBuf,
    options: Vec<String>,
    language: sifive_comgr_language_s,
    input_files: Vec<(PathBuf, sifive_comgr_data_kind_s)>,
    working_directory: Option<PathBuf>,
    target: Option<String>,
    action_kind: sifive_comgr_action_kind_s,
}

fn preferred_temp_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    for key in ["HETGPU_SIFIVE_TMPDIR", "TMPDIR", "TEMP", "TMP"] {
        if let Ok(value) = std::env::var(key) {
            if !value.is_empty() {
                let path = PathBuf::from(value);
                if !roots.iter().any(|existing| existing == &path) {
                    roots.push(path);
                }
            }
        }
    }

    for candidate in ["/mnt/usb/hetgpu_tmp", "/dev/shm/hetgpu_tmp"] {
        let path = PathBuf::from(candidate);
        if !roots.iter().any(|existing| existing == &path) {
            roots.push(path);
        }
    }

    let system_tmp = std::env::temp_dir();
    if !roots.iter().any(|existing| existing == &system_tmp) {
        roots.push(system_tmp);
    }

    roots
}

fn env_truthy(key: &str) -> bool {
    matches!(
        std::env::var(key).ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

fn create_action_tempdir() -> io::Result<TempDir> {
    let mut last_error: Option<io::Error> = None;

    for root in preferred_temp_roots() {
        if let Err(err) = fs::create_dir_all(&root) {
            last_error = Some(io::Error::new(
                err.kind(),
                format!("{}: {}", root.display(), err),
            ));
            continue;
        }

        match Builder::new().prefix(".tmp").tempdir_in(&root) {
            Ok(dir) => return Ok(dir),
            Err(err) => {
                last_error = Some(io::Error::new(
                    err.kind(),
                    format!("{}: {}", root.display(), err),
                ));
            }
        }
    }

    if let Some(err) = last_error {
        Err(err)
    } else {
        tempdir()
    }
}

fn keep_action_tempdir() -> bool {
    matches!(
        std::env::var("HETGPU_SIFIVE_KEEP_TMP").ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

pub fn perform_action(
    action_kind: sifive_comgr_action_kind_s,
    action_info: sifive_comgr_action_info_t,
    input_set: sifive_comgr_data_set_t,
    output_set: sifive_comgr_data_set_t,
) -> sifive_comgr_status_t {
    let dir = match create_action_tempdir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SIFIVE: Failed to create temporary directory: {}", e);
            return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR);
        }
    };

    let action_info_lock = ACTION_INFO_STORE.lock().unwrap();
    let action_data = match action_info_lock.get(&action_info.handle) {
        Some(data) => data.clone(),
        None => return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT),
    };
    drop(action_info_lock);

    let data_set_lock = DATA_SET_STORE.lock().unwrap();
    let data_handles = match data_set_lock.get(&input_set.handle) {
        Some(handles) => handles.clone(),
        None => return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT),
    };
    drop(data_set_lock);

    let working_directory = action_data.working_directory.as_ref().map(PathBuf::from);
    let data_store_lock = DATA_STORE.lock().unwrap();
    let mut input_files = Vec::new();

    for handle in &data_handles {
        if let Some(data) = data_store_lock.get(handle) {
            let file_name = match &data.name {
                Some(name) => name.clone(),
                None => match data.kind.0 {
                    1 => "input.cpp".to_string(),
                    6 => format!("input_{}.bc", handle),
                    7 => format!("input_{}.o", handle),
                    _ => format!("data_{}", handle),
                },
            };

            if let Some(original_path) =
                resolve_original_input_path(&file_name, working_directory.as_deref())
            {
                input_files.push((original_path, data.kind));
                continue;
            }

            let file_path = dir.path().join(&file_name);
            if let Some(parent) = file_path.parent() {
                if let Err(e) = fs::create_dir_all(parent) {
                    eprintln!(
                        "SIFIVE: Warning: Could not create parent directories for {}: {}",
                        file_path.display(),
                        e
                    );
                }
            }
            if let Err(e) = fs::write(&file_path, &data.content) {
                eprintln!("SIFIVE: Warning: Could not write input file: {}", e);
            }
            input_files.push((file_path, data.kind));
        }
    }
    drop(data_store_lock);

    let ctx = ActionContext {
        temp_dir: dir.path().to_path_buf(),
        options: action_data.options,
        language: action_data
            .language
            .unwrap_or(sifive_comgr_language_s::SIFIVE_COMGR_LANGUAGE_NONE),
        input_files,
        working_directory,
        target: action_data.target,
        action_kind,
    };

    if keep_action_tempdir() {
        eprintln!(
            "SIFIVE: keeping temporary directory {}",
            dir.path().display()
        );
    }

    let result = match action_kind.0 {
        0 => preprocess_source(&ctx),
        1 => Ok(()), // precompiled headers not used
        2 => compile_source_to_bc(&ctx),
        3 => Ok(()), // device libraries handled at link time
        4 => link_bc_to_bc(&ctx),
        5 => optimize_bc(&ctx),
        6 => codegen_to_riscv_sifive(&ctx),
        7 => codegen_to_assembly(&ctx),
        8 => compile_to_fatbin(&ctx),
        _ => {
            eprintln!("SIFIVE: Unknown action kind: {}", action_kind.0);
            return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
        }
    };

    if result.is_ok() {
        add_outputs_to_set(&ctx, output_set)?;
    }

    if keep_action_tempdir() {
        let preserved = dir.keep();
        eprintln!(
            "SIFIVE: preserved temporary directory {}",
            preserved.display()
        );
    }

    result
}

fn add_outputs_to_set(
    ctx: &ActionContext,
    output_set: sifive_comgr_data_set_t,
) -> sifive_comgr_status_t {
    let output_dir = ctx.temp_dir.clone();
    let entries = match fs::read_dir(&output_dir) {
        Ok(entries) => entries,
        Err(_) => return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR),
    };

    let mut added_files = false;

    for entry in entries {
        if let Ok(entry) = entry {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if file_name.contains(".host_stubs.") || file_name.contains(".tmp_shared_stubs.") {
                continue;
            }

            if let Some(extension) = path.extension() {
                let ext = extension.to_string_lossy().to_lowercase();

                if ext == "so" {
                    add_file_to_set(
                        &path,
                        sifive_comgr_data_kind_s::SIFIVE_COMGR_DATA_KIND_EXECUTABLE,
                        output_set,
                    )?;
                    added_files = true;
                } else if ext == "o" || ext == "elf" {
                    add_file_to_set(
                        &path,
                        sifive_comgr_data_kind_s::SIFIVE_COMGR_DATA_KIND_RELOCATABLE,
                        output_set,
                    )?;
                    added_files = true;
                } else if ext == "bc" {
                    add_file_to_set(
                        &path,
                        sifive_comgr_data_kind_s::SIFIVE_COMGR_DATA_KIND_BC,
                        output_set,
                    )?;
                    added_files = true;
                } else if ext == "s" || ext == "asm" {
                    add_file_to_set(
                        &path,
                        sifive_comgr_data_kind_s::SIFIVE_COMGR_DATA_KIND_SOURCE,
                        output_set,
                    )?;
                    added_files = true;
                } else if ext == "fatbin" {
                    add_file_to_set(
                        &path,
                        sifive_comgr_data_kind_s::SIFIVE_COMGR_DATA_KIND_FATBIN,
                        output_set,
                    )?;
                    added_files = true;
                }
            }
        }
    }

    // Create dummy RISC-V ELF if no outputs for codegen action
    if !added_files
        && ctx.action_kind.0
            == sifive_comgr_action_kind_s::SIFIVE_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE.0
    {
        eprintln!("SIFIVE: No output files found - creating dummy RISC-V ELF output");
        let dummy = create_dummy_riscv_elf();

        let mut data = unsafe { mem::zeroed() };
        sifive_comgr_create_data(
            sifive_comgr_data_kind_s::SIFIVE_COMGR_DATA_KIND_RELOCATABLE,
            &mut data,
        )?;
        sifive_comgr_data_set_name(data, c"sifive_output.elf".as_ptr())?;
        sifive_comgr_data_set_bytes(data, dummy.as_ptr() as *const c_void, dummy.len())?;
        sifive_comgr_data_set_add(output_set, data)?;
        sifive_comgr_release_data(data)?;
    }

    Ok(())
}

fn add_file_to_set(
    file_path: &Path,
    kind: sifive_comgr_data_kind_s,
    data_set: sifive_comgr_data_set_t,
) -> sifive_comgr_status_t {
    let mut data = unsafe { mem::zeroed() };
    sifive_comgr_create_data(kind, &mut data)?;

    let name = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| CString::new(n).ok())
        .unwrap_or_else(|| CString::new("output").unwrap());
    sifive_comgr_data_set_name(data, name.as_ptr())?;

    let content =
        fs::read(file_path).map_err(|_| sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR)?;
    sifive_comgr_data_set_bytes(data, content.as_ptr() as *const c_void, content.len())?;
    sifive_comgr_data_set_add(data_set, data)?;
    sifive_comgr_release_data(data)?;

    Ok(())
}

// --- Action implementations targeting RISC-V with VCIX/IME ---

fn preprocess_source(ctx: &ActionContext) -> sifive_comgr_status_t {
    let source_files: Vec<_> = ctx
        .input_files
        .iter()
        .filter(|(_, kind)| kind.0 == sifive_comgr_data_kind_s::SIFIVE_COMGR_DATA_KIND_SOURCE.0)
        .map(|(path, _)| path.clone())
        .collect();

    for input_file in source_files {
        let file_stem = input_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("input");
        let output_file = ctx.temp_dir.join(format!("{}.i", file_stem));
        fs::copy(&input_file, &output_file)
            .map_err(|_| sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR)?;
    }
    Ok(())
}

fn compile_source_to_bc(ctx: &ActionContext) -> sifive_comgr_status_t {
    let source_files: Vec<_> = ctx
        .input_files
        .iter()
        .filter(|(_, kind)| kind.0 == sifive_comgr_data_kind_s::SIFIVE_COMGR_DATA_KIND_SOURCE.0)
        .map(|(path, _)| path.clone())
        .collect();

    if source_files.is_empty() {
        return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    for input_file in source_files {
        let file_stem = input_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("input");
        let output_file = ctx.temp_dir.join(format!("{}.bc", file_stem));

        let compile_result = if is_cxx_source(&input_file) || is_c_source(&input_file) {
            compile_c_family_source_to_bitcode(ctx, &input_file, &output_file)
        } else {
            compile_ptx_to_bitcode(ctx, &input_file, &output_file)
        };

        if let Err(e) = compile_result {
            eprintln!(
                "SIFIVE: source -> LLVM bitcode failed for {}: {}",
                input_file.display(),
                e
            );
            return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR);
        }

        eprintln!("SIFIVE: Created LLVM bitcode: {}", output_file.display());
    }

    Ok(())
}

fn link_bc_to_bc(ctx: &ActionContext) -> sifive_comgr_status_t {
    let bc_files: Vec<_> = ctx
        .input_files
        .iter()
        .filter(|(_, kind)| kind.0 == sifive_comgr_data_kind_s::SIFIVE_COMGR_DATA_KIND_BC.0)
        .map(|(path, _)| path.clone())
        .collect();

    if bc_files.len() < 2 {
        if let Some(input_file) = bc_files.first() {
            let output_file = ctx.temp_dir.join("linked.bc");
            fs::copy(input_file, &output_file)
                .map_err(|_| sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR)?;
        }
        return Ok(());
    }

    let output_file = ctx.temp_dir.join("linked.bc");

    let llvm_link = llvm_link_tool();
    let mut cmd = Command::new(&llvm_link);
    for bc_file in &bc_files {
        cmd.arg(bc_file);
    }
    cmd.arg("-o").arg(&output_file);

    match tool_output(&mut cmd) {
        Ok(output) if output.status.success() => {
            eprintln!("SIFIVE: Successfully linked bitcode files");
            Ok(())
        }
        Ok(output) => {
            let inputs = bc_files
                .iter()
                .map(|path| {
                    let len = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                    format!("{}({} bytes)", path.display(), len)
                })
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!(
                "SIFIVE: llvm-link failed\ntool: {}\nstatus: {}\ninputs: [{}]\noutput: {}\nstdout:\n{}\nstderr:\n{}",
                llvm_link,
                output.status,
                inputs,
                output_file.display(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR)
        }
        Err(err) => {
            eprintln!("SIFIVE: failed to execute llvm-link {}: {err}", llvm_link);
            Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR)
        }
    }
}

fn optimize_bc(ctx: &ActionContext) -> sifive_comgr_status_t {
    let bc_files: Vec<_> = ctx
        .input_files
        .iter()
        .filter(|(_, kind)| kind.0 == sifive_comgr_data_kind_s::SIFIVE_COMGR_DATA_KIND_BC.0)
        .map(|(path, _)| path.clone())
        .collect();

    if bc_files.is_empty() {
        return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    for input_file in bc_files {
        let file_stem = input_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("input");
        let output_file = ctx.temp_dir.join(format!("{}_optimized.bc", file_stem));
        let input_size = fs::metadata(&input_file).map(|m| m.len()).unwrap_or(0);

        if input_size >= 1_000_000 {
            eprintln!(
                "SIFIVE: Skipping opt for large bitcode ({} bytes): {}",
                input_size,
                input_file.display()
            );
            fs::copy(&input_file, &output_file)
                .map_err(|_| sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR)?;
            eprintln!(
                "SIFIVE: Copied bitcode without optimization: {}",
                output_file.display()
            );
            continue;
        }

        let input_for_opt = match sifive_preopt_input(&input_file, &ctx.temp_dir) {
            Ok(path) => path,
            Err(err) => {
                eprintln!(
                    "SIFIVE: failed to prepare {} for opt ({}); using original bitcode",
                    input_file.display(),
                    err
                );
                input_file.clone()
            }
        };

        let mut cmd = Command::new(opt_tool());
        cmd.arg(&input_for_opt)
            .arg("-o")
            .arg(&output_file)
            .arg("-O3")
            .arg("-non-global-value-max-name-size=16384");

        eprintln!(
            "SIFIVE: Running opt on {} -> {}",
            input_for_opt.display(),
            output_file.display()
        );

        match tool_output(&mut cmd) {
            Ok(output) if output.status.success() => {
                eprintln!("SIFIVE: Optimized bitcode: {}", output_file.display());
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                if is_llvm23_attr_mismatch(&stderr) {
                    eprintln!(
                        "SIFIVE: skipping opt for {} due to LLVM23/LLVM20 bitcode mismatch; using unoptimized bitcode",
                        input_file.display()
                    );
                } else {
                    eprintln!(
                        "SIFIVE: opt failed for {} (status {:?}) stderr:\n{}",
                        input_file.display(),
                        output.status.code(),
                        stderr
                    );
                }
                fs::copy(&input_file, &output_file)
                    .map_err(|_| sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR)?;
            }
            Err(err) => {
                eprintln!(
                    "SIFIVE: opt invocation failed for {}: {}",
                    input_file.display(),
                    err
                );
                fs::copy(&input_file, &output_file)
                    .map_err(|_| sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR)?;
            }
        }
    }

    Ok(())
}

fn sifive_preopt_input(input_file: &Path, temp_dir: &Path) -> io::Result<PathBuf> {
    if !bc_requires_sanitized_ll(input_file)? {
        return Ok(input_file.to_path_buf());
    }

    let stem = input_file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("input");
    let ll_file = temp_dir.join(format!("{stem}_sifive_preopt.ll"));

    let llvm_dis = llvm_dis_tool();
    let mut dis_cmd = Command::new(llvm_dis);
    dis_cmd.arg(input_file).arg("-o").arg("-");
    let output = tool_output(&mut dis_cmd)?;
    if !output.status.success() {
        return Ok(input_file.to_path_buf());
    }

    let ll_text = String::from_utf8_lossy(&output.stdout);
    let sanitized = sanitize_llvm23_ir_for_llvm20(&ll_text);
    fs::write(&ll_file, sanitized)?;
    eprintln!(
        "SIFIVE: prepared sanitized pre-opt LLVM IR for {} -> {}",
        input_file.display(),
        ll_file.display()
    );
    Ok(ll_file)
}

fn is_llvm23_attr_mismatch(stderr: &str) -> bool {
    stderr.contains("Unknown attribute kind (105)")
        || (stderr.contains("Producer: 'LLVM23") && stderr.contains("Reader: 'LLVM 20"))
}

fn codegen_to_riscv_sifive(ctx: &ActionContext) -> sifive_comgr_status_t {
    let mut bc_files: Vec<_> = ctx
        .input_files
        .iter()
        .filter(|(_, kind)| kind.0 == sifive_comgr_data_kind_s::SIFIVE_COMGR_DATA_KIND_BC.0)
        .map(|(path, _)| path.clone())
        .collect();

    if bc_files.is_empty() {
        return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    // After LINK_BC_TO_BC + OPTIMIZE_BC_TO_BC we may have per-module bitcode
    // (`main_optimized.bc`, `linked_0_optimized.bc`, ...) and the fully linked
    // aggregate (`linked_optimized.bc`). For the SIFIVE path we only need the
    // aggregate relocatable; compiling each helper module separately drags in
    // AMD-specific helper intrinsics that are irrelevant after linking and can
    // crash the RISC-V backend during instruction selection.
    if let Some(linked) = bc_files
        .iter()
        .find(|path| path.file_name().and_then(|n| n.to_str()) == Some("linked_optimized.bc"))
        .cloned()
    {
        bc_files.clear();
        bc_files.push(linked);
    }

    for input_file in bc_files {
        let file_stem = input_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("input");
        let output_file = ctx.temp_dir.join(format!("{}_sifive.o", file_stem));
        let shared_file = ctx.temp_dir.join(format!("{}_sifive.so", file_stem));

        let used_config = match compile_bc_to_sifive_object(&input_file, &output_file) {
            Ok(config) => config,
            Err(e) => {
                eprintln!(
                    "SIFIVE/XM: failed to generate RISC-V object from {}: {}",
                    input_file.display(),
                    e
                );
                return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR);
            }
        };

        eprintln!(
            "SIFIVE/XM: Generated RISC-V object code with target={} march={}: {}",
            used_config.target_triple,
            used_config.march,
            output_file.display()
        );

        if host_link_sifive_shared_enabled() {
            if let Err(e) = link_sifive_object_to_shared(&output_file, &shared_file) {
                eprintln!(
                    "SIFIVE/XM: failed to host-link SIFIVE shared object from {}: {}",
                    output_file.display(),
                    e
                );
                return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR);
            }
            eprintln!(
                "SIFIVE/XM: Host-linked launchable RISC-V shared object: {}",
                shared_file.display()
            );
        }
    }

    Ok(())
}

fn codegen_to_assembly(ctx: &ActionContext) -> sifive_comgr_status_t {
    let bc_files: Vec<_> = ctx
        .input_files
        .iter()
        .filter(|(_, kind)| kind.0 == sifive_comgr_data_kind_s::SIFIVE_COMGR_DATA_KIND_BC.0)
        .map(|(path, _)| path.clone())
        .collect();

    if bc_files.is_empty() {
        return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    let config = crate::SifiveConfig::xm_hardware();

    for input_file in bc_files {
        let file_stem = input_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("input");
        let output_file = ctx.temp_dir.join(format!("{}.s", file_stem));

        if let Err(e) = compile_bc_to_xm_assembly(&input_file, &output_file, &config) {
            eprintln!(
                "SIFIVE/XM: failed to generate assembly from {}: {}",
                input_file.display(),
                e
            );
            return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR);
        }

        eprintln!(
            "SIFIVE/XM: Generated RISC-V+VCIX assembly: {}",
            output_file.display()
        );
    }

    Ok(())
}

fn compile_to_fatbin(ctx: &ActionContext) -> sifive_comgr_status_t {
    let source_files: Vec<_> = ctx
        .input_files
        .iter()
        .filter(|(_, kind)| kind.0 == sifive_comgr_data_kind_s::SIFIVE_COMGR_DATA_KIND_SOURCE.0)
        .map(|(path, _)| path.clone())
        .collect();

    if source_files.is_empty() {
        return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }

    for input_file in source_files {
        let file_stem = input_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("input");
        let output_file = ctx.temp_dir.join(format!("{}_sifive.fatbin", file_stem));
        let bc_file = ctx.temp_dir.join(format!("{}_fatbin.bc", file_stem));
        let elf_file = ctx.temp_dir.join(format!("{}_fatbin.o", file_stem));

        let compile_result = if is_cxx_source(&input_file) || is_c_source(&input_file) {
            compile_c_family_source_to_bitcode(ctx, &input_file, &bc_file)
        } else {
            compile_ptx_to_bitcode(ctx, &input_file, &bc_file)
        };

        if let Err(e) = compile_result {
            eprintln!(
                "SIFIVE: source -> LLVM bitcode failed for fatbin input {}: {}",
                input_file.display(),
                e
            );
            return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR);
        }

        if let Err(e) = compile_bc_to_sifive_object(&bc_file, &elf_file) {
            eprintln!(
                "SIFIVE/XM: bitcode -> object failed for fatbin input {}: {}",
                input_file.display(),
                e
            );
            return Err(sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR);
        }

        let elf_bytes =
            fs::read(&elf_file).map_err(|_| sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR)?;
        let fatbin_content = create_cuda_fatbin_with_elf(&elf_bytes);

        fs::write(&output_file, fatbin_content)
            .map_err(|_| sifive_comgr_status_s::SIFIVE_COMGR_STATUS_ERROR)?;
        eprintln!("SIFIVE: Created fatbin: {}", output_file.display());
    }

    Ok(())
}

fn compile_ptx_to_bitcode(
    ctx: &ActionContext,
    input_file: &Path,
    output_file: &Path,
) -> io::Result<()> {
    let tool = ensure_ptx_helper_tool("ptx_to_llvm_bc")?;
    let llvm_ir_path = output_file.with_extension("ll");
    let mut cmd = Command::new(tool);
    cmd.arg(input_file)
        .arg(output_file)
        .arg("--llvm-ir")
        .arg(&llvm_ir_path);
    apply_action_context_to_command(ctx, &mut cmd);
    run_command(&mut cmd, "PTX -> LLVM bitcode")
}

fn compile_c_family_source_to_bitcode(
    ctx: &ActionContext,
    input_file: &Path,
    output_file: &Path,
) -> io::Result<()> {
    let config = crate::SifiveConfig::rvv_linux_bf16();
    let clang = sifive_clang_tool("HETGPU_SIFIVE_SOURCE_CLANG");
    let target = ctx
        .target
        .clone()
        .or_else(|| std::env::var("HETGPU_SIFIVE_SOURCE_TARGET").ok())
        .unwrap_or_else(|| config.target_triple.clone());
    let sysroot = std::env::var("HETGPU_SIFIVE_SOURCE_SYSROOT").unwrap_or_else(|_| "/".to_string());
    let gcc_toolchain =
        std::env::var("HETGPU_SIFIVE_SOURCE_GCC_TOOLCHAIN").unwrap_or_else(|_| "/usr".to_string());
    let march = riscv_march_with_required_extensions(
        &std::env::var("HETGPU_SIFIVE_SOURCE_MARCH").unwrap_or_else(|_| config.march.clone()),
    );
    let mabi = std::env::var("HETGPU_SIFIVE_SOURCE_MABI").unwrap_or_else(|_| "lp64d".to_string());

    let mut cmd = Command::new(clang);
    cmd.arg(format!("--target={}", target))
        .arg(format!("--sysroot={}", sysroot))
        .arg(format!("--gcc-toolchain={}", gcc_toolchain))
        .arg("-mllvm")
        .arg("-non-global-value-max-name-size=16384")
        .arg("-emit-llvm")
        .arg("-c")
        .arg("-O3")
        .arg("-menable-experimental-extensions")
        .arg(format!("-march={}", march))
        .arg(format!("-mabi={}", mabi))
        .arg(input_file)
        .arg("-o")
        .arg(output_file);

    if is_cxx_source(input_file) {
        cmd.arg("-std=c++17").arg("-stdlib=libstdc++");
    } else {
        cmd.arg("-std=c11");
    }

    cmd.args(&ctx.options);
    apply_action_context_to_command(ctx, &mut cmd);

    run_command(&mut cmd, "C/C++ source -> LLVM bitcode")
}

fn apply_action_context_to_command(ctx: &ActionContext, cmd: &mut Command) {
    if let Some(dir) = &ctx.working_directory {
        cmd.current_dir(dir);
    }
    cmd.env("HETGPU_SIFIVE_TMPDIR", &ctx.temp_dir)
        .env("TMPDIR", &ctx.temp_dir)
        .env("TEMP", &ctx.temp_dir)
        .env("TMP", &ctx.temp_dir);
}

fn riscv_march_with_required_extensions(march: &str) -> String {
    if !(march.starts_with("rv32") || march.starts_with("rv64")) {
        return march.to_string();
    }
    let require_zbb_disabled = matches!(
        std::env::var("HETGPU_SIFIVE_REQUIRE_ZBB").ok().as_deref(),
        Some("0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF")
    );
    if require_zbb_disabled || env_truthy("HETGPU_SIFIVE_DISABLE_AUTO_ZBB") {
        return march.to_string();
    }

    let has_zbb = march
        .split('_')
        .any(|ext| ext.to_ascii_lowercase().starts_with("zbb"));
    if has_zbb {
        march.to_string()
    } else {
        format!("{march}_zbb")
    }
}

fn sifive_codegen_march(config: &crate::SifiveConfig) -> String {
    std::env::var("HETGPU_SIFIVE_CODEGEN_MARCH")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| config.march.clone())
}

fn host_link_sifive_shared_enabled() -> bool {
    std::env::var("HETGPU_SIFIVE_HOST_LINK_SHARED")
        .ok()
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            !matches!(v.as_str(), "0" | "false" | "no" | "off")
        })
        .unwrap_or(true)
}

const SIFIVE_KERNEL_HOST_STUBS_C: &str = r#"
#include <stdint.h>
#include <stdbool.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <math.h>
#define WEAK __attribute__((weak, visibility("hidden")))
#define WEAK_EXPORT __attribute__((weak, visibility("default")))
struct ShflSyncResult { uint32_t x; uint32_t pred; };
struct DivF32Part1Result { float fma_4; float fma_1; float fma_3; uint8_t numerator_scaled_flag; };
static uint32_t lane_u8(uint32_t x, unsigned lane) { return (x >> (lane * 8)) & 0xffu; }
static int32_t lane_s8(uint32_t x, unsigned lane) { return (int8_t)lane_u8(x, lane); }
static uint32_t pack_lane_u8(uint32_t base, unsigned lane, uint32_t value) {
    uint32_t shift = lane * 8;
    return (base & ~(0xffu << shift)) | ((value & 0xffu) << shift);
}
static uint32_t sat_u8(int32_t v) { return v < 0 ? 0u : (v > 255 ? 255u : (uint32_t)v); }
static int32_t sat_s8(int32_t v) { return v < -128 ? -128 : (v > 127 ? 127 : v); }
WEAK uint32_t f___zluda_ptx_impl_vsub4_u32_u32_u32(uint32_t a, uint32_t b, uint32_t c) {
    (void)c; uint32_t r = 0; for (unsigned i = 0; i < 4; ++i) r = pack_lane_u8(r, i, lane_u8(a, i) - lane_u8(b, i)); return r;
}
WEAK uint32_t f___zluda_ptx_impl_vsub4_u32_u32_u32_sat(uint32_t a, uint32_t b, uint32_t c) {
    (void)c; uint32_t r = 0; for (unsigned i = 0; i < 4; ++i) r = pack_lane_u8(r, i, sat_u8((int32_t)lane_u8(a, i) - (int32_t)lane_u8(b, i))); return r;
}
WEAK uint32_t f___zluda_ptx_impl_vsub4_s32_s32_s32(uint32_t a, uint32_t b, uint32_t c) {
    (void)c; uint32_t r = 0; for (unsigned i = 0; i < 4; ++i) r = pack_lane_u8(r, i, (uint8_t)(lane_s8(a, i) - lane_s8(b, i))); return r;
}
WEAK uint32_t f___zluda_ptx_impl_vsub4_s32_s32_s32_sat(uint32_t a, uint32_t b, uint32_t c) {
    (void)c; uint32_t r = 0; for (unsigned i = 0; i < 4; ++i) r = pack_lane_u8(r, i, (uint8_t)sat_s8(lane_s8(a, i) - lane_s8(b, i))); return r;
}
static uint32_t vset_cmp(uint32_t a, uint32_t b, int op) {
    uint32_t r = 0; for (unsigned i = 0; i < 4; ++i) { uint32_t x = lane_u8(a, i), y = lane_u8(b, i); int p = 0;
    switch (op) { case 0: p = x == y; break; case 1: p = x != y; break; case 2: p = x < y; break; case 3: p = x <= y; break; case 4: p = x > y; break; default: p = x >= y; break; }
    r = pack_lane_u8(r, i, p ? 1u : 0u); } return r;
}
WEAK uint32_t f___zluda_ptx_impl_vset4_u32_u32_eq(uint32_t a, uint32_t b, uint32_t c) { (void)c; return vset_cmp(a, b, 0); }
WEAK uint32_t f___zluda_ptx_impl_vset4_u32_u32_ne(uint32_t a, uint32_t b, uint32_t c) { (void)c; return vset_cmp(a, b, 1); }
WEAK uint32_t f___zluda_ptx_impl_vset4_u32_u32_lt(uint32_t a, uint32_t b, uint32_t c) { (void)c; return vset_cmp(a, b, 2); }
WEAK uint32_t f___zluda_ptx_impl_vset4_u32_u32_le(uint32_t a, uint32_t b, uint32_t c) { (void)c; return vset_cmp(a, b, 3); }
WEAK uint32_t f___zluda_ptx_impl_vset4_u32_u32_gt(uint32_t a, uint32_t b, uint32_t c) { (void)c; return vset_cmp(a, b, 4); }
WEAK uint32_t f___zluda_ptx_impl_vset4_u32_u32_ge(uint32_t a, uint32_t b, uint32_t c) { (void)c; return vset_cmp(a, b, 5); }
WEAK void f___zluda_ptx_impl_bar_sync(uint32_t barrier_id) { (void)barrier_id; __sync_synchronize(); }
WEAK bool f___zluda_ptx_impl_bar_red_and_pred(uint32_t barrier_id, bool predicate, bool invert_predicate) { (void)barrier_id; __sync_synchronize(); return predicate ^ invert_predicate; }
WEAK bool f___zluda_ptx_impl_bar_red_or_pred(uint32_t barrier_id, bool predicate, bool invert_predicate) { (void)barrier_id; __sync_synchronize(); return predicate ^ invert_predicate; }
WEAK uint32_t f___zluda_ptx_impl_activemask(void) { return 1u; }
struct HetgpuLaunchState {
    uint32_t tid[3];
    uint32_t ntid[3];
    uint32_t ctaid[3];
    uint32_t nctaid[3];
};
static struct HetgpuLaunchState hetgpu_launch_states[64];
static unsigned hetgpu_launch_slot(void) {
#if defined(__riscv)
    uintptr_t id = 0;
    __asm__ volatile("mv %0, tp" : "=r"(id));
#else
    uintptr_t id = (uintptr_t)&id;
#endif
    return (unsigned)((id ^ (id >> 6)) & 63u);
}
static struct HetgpuLaunchState *hetgpu_launch_state(void) {
    return &hetgpu_launch_states[hetgpu_launch_slot()];
}
WEAK_EXPORT void f___zluda_ptx_impl_set_launch(uint32_t tid_x, uint32_t tid_y, uint32_t tid_z, uint32_t ntid_x, uint32_t ntid_y, uint32_t ntid_z, uint32_t ctaid_x, uint32_t ctaid_y, uint32_t ctaid_z, uint32_t nctaid_x, uint32_t nctaid_y, uint32_t nctaid_z) {
    struct HetgpuLaunchState *s = hetgpu_launch_state();
    s->tid[0] = tid_x; s->tid[1] = tid_y; s->tid[2] = tid_z;
    s->ntid[0] = ntid_x ? ntid_x : 1u; s->ntid[1] = ntid_y ? ntid_y : 1u; s->ntid[2] = ntid_z ? ntid_z : 1u;
    s->ctaid[0] = ctaid_x; s->ctaid[1] = ctaid_y; s->ctaid[2] = ctaid_z;
    s->nctaid[0] = nctaid_x ? nctaid_x : 1u; s->nctaid[1] = nctaid_y ? nctaid_y : 1u; s->nctaid[2] = nctaid_z ? nctaid_z : 1u;
}
WEAK uint32_t f___zluda_ptx_impl_sreg_tid(uint8_t member) { struct HetgpuLaunchState *s = hetgpu_launch_state(); return member < 3u ? s->tid[member] : 0u; }
WEAK uint32_t f___zluda_ptx_impl_sreg_ntid(uint8_t member) { struct HetgpuLaunchState *s = hetgpu_launch_state(); return member < 3u && s->ntid[member] ? s->ntid[member] : 1u; }
WEAK uint32_t f___zluda_ptx_impl_sreg_ctaid(uint8_t member) { struct HetgpuLaunchState *s = hetgpu_launch_state(); return member < 3u ? s->ctaid[member] : 0u; }
WEAK uint32_t f___zluda_ptx_impl_sreg_nctaid(uint8_t member) { struct HetgpuLaunchState *s = hetgpu_launch_state(); return member < 3u && s->nctaid[member] ? s->nctaid[member] : 1u; }
WEAK uint32_t f___zluda_ptx_impl_sreg_laneid(void) { struct HetgpuLaunchState *s = hetgpu_launch_state(); return s->tid[0] & 31u; }
WEAK uint32_t f___zluda_ptx_impl_sreg_lanemask_eq(void) { struct HetgpuLaunchState *s = hetgpu_launch_state(); uint32_t lane = s->tid[0] & 31u; return 1u << lane; }
WEAK uint32_t f___zluda_ptx_impl_sreg_lanemask_lt(void) { struct HetgpuLaunchState *s = hetgpu_launch_state(); uint32_t lane = s->tid[0] & 31u; return lane == 0u ? 0u : ((1u << lane) - 1u); }
WEAK uint32_t f___zluda_ptx_impl_sreg_lanemask_le(void) { struct HetgpuLaunchState *s = hetgpu_launch_state(); uint32_t lane = s->tid[0] & 31u; return lane == 31u ? ~0u : ((1u << (lane + 1u)) - 1u); }
WEAK uint32_t f___zluda_ptx_impl_sreg_lanemask_ge(void) { struct HetgpuLaunchState *s = hetgpu_launch_state(); uint32_t lane = s->tid[0] & 31u; return ~((lane == 0u ? 0u : ((1u << lane) - 1u))); }
WEAK uint32_t f___zluda_ptx_impl_sreg_lanemask_gt(void) { struct HetgpuLaunchState *s = hetgpu_launch_state(); uint32_t lane = s->tid[0] & 31u; return lane == 31u ? 0u : (~0u << (lane + 1u)); }
WEAK uint32_t f___zluda_ptx_impl_sreg_clock(void) { return 0u; }
WEAK float f___zluda_ptx_impl_sqrt_approx_f32(float x) { return sqrtf(x); }
WEAK float f___zluda_ptx_impl_rsqrt_approx_f32(float x) { return 1.0f / sqrtf(x); }
WEAK float f___zluda_ptx_impl_ex2_approx_f32(float x) { return exp2f(x); }
WEAK float f___zluda_ptx_impl_lg2_approx_f32(float x) { return log2f(x); }
WEAK float f___zluda_ptx_impl_rcp_approx_f32(float x) { return 1.0f / x; }
WEAK void f___zluda_ptx_impl_nanosleep_u32(uint32_t nanoseconds) { (void)nanoseconds; }
WEAK bool f___zluda_ptx_impl_vote_sync_any_pred(bool value, uint32_t membermask) { (void)membermask; return value; }
WEAK bool f___zluda_ptx_impl_vote_sync_any_pred_negate(bool value, uint32_t membermask) { (void)membermask; return !value; }
WEAK bool f___zluda_ptx_impl_vote_sync_all_pred(bool value, uint32_t membermask) { (void)membermask; return value; }
WEAK bool f___zluda_ptx_impl_vote_sync_all_pred_negate(bool value, uint32_t membermask) { (void)membermask; return !value; }
WEAK uint32_t f___zluda_ptx_impl_vote_sync_ballot_b32(bool value, uint32_t membermask) { return value ? (membermask ? membermask : 1u) : 0u; }
WEAK uint32_t f___zluda_ptx_impl_vote_sync_ballot_b32_negate(bool value, uint32_t membermask) { return !value ? (membermask ? membermask : 1u) : 0u; }
WEAK uint32_t f___zluda_ptx_impl_bfe_u32(uint32_t base, uint32_t pos_32, uint32_t len_32) {
    uint32_t pos = pos_32 & 0xffu, len = len_32 & 0xffu; if (pos >= 32u || len == 0u) return 0u; if (len >= 32u) return base >> pos; if (len > 31u) len = 31u; return (base >> pos) & ((1u << len) - 1u);
}
WEAK int32_t f___zluda_ptx_impl_bfe_s32(int32_t base, uint32_t pos_32, uint32_t len_32) {
    uint32_t pos = pos_32 & 0xffu, len = len_32 & 0xffu; if (len == 0u) return 0; if (pos >= 32u) return base >> 31; if (len >= 32u || pos + len >= 32u) return base >> pos; return (base << (32u - pos - len)) >> (32u - len);
}
WEAK uint64_t f___zluda_ptx_impl_bfe_u64(uint64_t base, uint32_t pos, uint32_t len) { if (pos >= 64u || len == 0u) return 0u; if (len >= 64u) return base >> pos; return (base >> pos) & ((1ull << len) - 1ull); }
WEAK int64_t f___zluda_ptx_impl_bfe_s64(int64_t base, uint32_t pos, uint32_t len) { if (len == 0u) return 0; if (pos >= 64u) return base >> 63; if (len >= 64u || pos + len >= 64u) return base >> pos; return (base << (64u - pos - len)) >> (64u - len); }
WEAK uint32_t f___zluda_ptx_impl_bfi_b32(uint32_t insert, uint32_t base, uint32_t pos_32, uint32_t len_32) { uint32_t pos = pos_32 & 0xffu, len = len_32 & 0xffu; if (pos >= 32u || len == 0u) return base; uint32_t mask = (len >= 32u || pos + len >= 32u) ? (~0u << pos) : (((1u << len) - 1u) << pos); return (base & ~mask) | ((insert << pos) & mask); }
WEAK uint64_t f___zluda_ptx_impl_bfi_b64(uint64_t insert, uint64_t base, uint32_t pos, uint32_t len) { if (pos >= 64u || len == 0u) return base; uint64_t mask = (len >= 64u || pos + len >= 64u) ? (~0ull << pos) : (((1ull << len) - 1ull) << pos); return (base & ~mask) | ((insert << pos) & mask); }
WEAK uint32_t f___zluda_ptx_impl_prmt_b32(uint32_t a, uint32_t b, uint32_t c) { uint32_t r = 0; for (unsigned i = 0; i < 4; ++i) { uint32_t sel = (c >> (4 * i)) & 0xfu; uint32_t src = (sel & 4u) ? b : a; uint32_t val = (src >> (8 * (sel & 3u))) & 0xffu; if (sel & 8u) val = (val & 0x80u) ? 0xffu : 0u; r |= val << (8 * i); } return r; }
WEAK struct DivF32Part1Result f___zluda_ptx_impl_div_f32_part1(float lhs, float rhs) { (void)lhs; (void)rhs; return (struct DivF32Part1Result){ 0.0f, 0.0f, 0.0f, 0u }; }
WEAK float f___zluda_ptx_impl_div_f32_part2(float x, float y, float fma_4, float fma_1, float fma_3, uint8_t numerator_scaled_flag) { (void)fma_4; (void)fma_1; (void)fma_3; (void)numerator_scaled_flag; return x / y; }
WEAK struct ShflSyncResult f___zluda_ptx_impl_shfl_sync_bfly_b32_pred(uint32_t input, int32_t delta, uint32_t opts, uint32_t membermask) { (void)delta; (void)opts; (void)membermask; return (struct ShflSyncResult){ input, 1u }; }
WEAK struct ShflSyncResult f___zluda_ptx_impl_shfl_sync_up_b32_pred(uint32_t input, int32_t delta, uint32_t opts, uint32_t membermask) { (void)delta; (void)opts; (void)membermask; return (struct ShflSyncResult){ input, 1u }; }
WEAK struct ShflSyncResult f___zluda_ptx_impl_shfl_sync_down_b32_pred(uint32_t input, int32_t delta, uint32_t opts, uint32_t membermask) { (void)delta; (void)opts; (void)membermask; return (struct ShflSyncResult){ input, 1u }; }
WEAK struct ShflSyncResult f___zluda_ptx_impl_shfl_sync_idx_b32_pred(uint32_t input, int32_t delta, uint32_t opts, uint32_t membermask) { (void)delta; (void)opts; (void)membermask; return (struct ShflSyncResult){ input, 1u }; }
WEAK uint32_t f___zluda_ptx_impl_shfl_sync_bfly_b32(uint32_t input, int32_t delta, uint32_t opts, uint32_t membermask) { (void)delta; (void)opts; (void)membermask; return input; }
WEAK uint32_t f___zluda_ptx_impl_shfl_sync_up_b32(uint32_t input, int32_t delta, uint32_t opts, uint32_t membermask) { (void)delta; (void)opts; (void)membermask; return input; }
WEAK uint32_t f___zluda_ptx_impl_shfl_sync_down_b32(uint32_t input, int32_t delta, uint32_t opts, uint32_t membermask) { (void)delta; (void)opts; (void)membermask; return input; }
WEAK uint32_t f___zluda_ptx_impl_shfl_sync_idx_b32(uint32_t input, int32_t delta, uint32_t opts, uint32_t membermask) { (void)delta; (void)opts; (void)membermask; return input; }
"#;

fn sifive_link_cc_tool() -> String {
    if let Ok(tool) = std::env::var("HETGPU_SIFIVE_HOST_LINK_CC") {
        if !tool.trim().is_empty() {
            return tool;
        }
    }
    preferred_tool(
        "HETGPU_SIFIVE_HOST_LINK_CC",
        &[
            "riscv64-linux-gnu-gcc",
            "/usr/bin/riscv64-linux-gnu-gcc",
            "gcc",
            "cc",
            "clang",
        ],
    )
}

fn command_is_clang(tool: &str) -> bool {
    Path::new(tool)
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.contains("clang"))
        .unwrap_or(false)
}

fn add_riscv_linux_clang_args(cmd: &mut Command) {
    cmd.arg("--target=riscv64-unknown-linux-gnu");
    if let Ok(sysroot) = std::env::var("HETGPU_SIFIVE_SOURCE_SYSROOT") {
        if !sysroot.trim().is_empty() {
            cmd.arg(format!("--sysroot={sysroot}"));
        }
    }
    if let Ok(toolchain) = std::env::var("HETGPU_SIFIVE_SOURCE_GCC_TOOLCHAIN") {
        if !toolchain.trim().is_empty() {
            cmd.arg(format!("--gcc-toolchain={toolchain}"));
        }
    }
}

fn compile_sifive_host_stub(src: &Path, obj: &Path) -> io::Result<()> {
    let cc = sifive_link_cc_tool();
    let mut cmd = Command::new(&cc);
    if command_is_clang(&cc) {
        add_riscv_linux_clang_args(&mut cmd);
    }
    cmd.arg("-O2")
        .arg("-fPIC")
        .arg("-c")
        .arg(src)
        .arg("-o")
        .arg(obj);
    run_command(&mut cmd, "SIFIVE host stub compile")
}

fn nm_tool() -> String {
    bundled_llvm_tool(
        "HETGPU_SIFIVE_NM",
        "llvm-nm",
        &[
            "riscv64-linux-gnu-nm",
            "/usr/bin/riscv64-linux-gnu-nm",
            "llvm-nm",
            "nm",
        ],
    )
}

fn is_valid_c_symbol_name(symbol: &str) -> bool {
    let mut chars = symbol.chars();
    match chars.next() {
        Some('_') | Some('A'..='Z') | Some('a'..='z') => {}
        _ => return false,
    }
    chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

fn object_undefined_symbols(input_obj: &Path) -> io::Result<Vec<String>> {
    let mut cmd = Command::new(nm_tool());
    cmd.arg("-u").arg(input_obj);
    let output = tool_output(&mut cmd)?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "nm failed while scanning {}: {}",
            input_obj.display(),
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    let mut symbols = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if let Some(symbol) = line.split_whitespace().last() {
            if !symbols.iter().any(|existing| existing == symbol) {
                symbols.push(symbol.to_string());
            }
        }
    }
    Ok(symbols)
}

fn kernel_tmp_shared_stub_bytes() -> usize {
    std::env::var("HETGPU_SIFIVE_KERNEL_TMP_SHARED_BYTES")
        .ok()
        .and_then(|v| {
            let trimmed = v.trim();
            let parsed = trimmed
                .strip_prefix("0x")
                .or_else(|| trimmed.strip_prefix("0X"))
                .map(|hex| usize::from_str_radix(hex, 16).ok())
                .unwrap_or_else(|| trimmed.parse().ok())?;
            (4096..=(16 << 20)).contains(&parsed).then_some(parsed)
        })
        .unwrap_or(64 << 10)
}

fn write_tmp_shared_stub_source(input_obj: &Path, stub_src: &Path) -> io::Result<()> {
    let mut out = String::from(
        "#include <stdint.h>\n__attribute__((used)) static unsigned char hetgpu_sifive_tmp_shared_anchor;\n",
    );
    let bytes = kernel_tmp_shared_stub_bytes();
    for symbol in object_undefined_symbols(input_obj)? {
        if !symbol.contains("tmp_shared") || !is_valid_c_symbol_name(&symbol) {
            continue;
        }
        out.push_str(&format!(
            "__attribute__((weak, visibility(\"hidden\"), aligned(16))) unsigned char {symbol}[{bytes}];\n"
        ));
    }
    fs::write(stub_src, out)
}

fn find_riscv_builtins_archive() -> Option<String> {
    if let Ok(path) = std::env::var("HETGPU_SIFIVE_DEVICE_BUILTINS") {
        if !path.trim().is_empty() && Path::new(&path).is_file() {
            return Some(path);
        }
    }

    for candidate in [
        "/usr/lib/llvm-23/lib/clang/23/lib/linux/libclang_rt.builtins-riscv64.a",
        "/usr/lib/llvm-22/lib/clang/22/lib/linux/libclang_rt.builtins-riscv64.a",
        "/usr/lib/llvm-21/lib/clang/21/lib/linux/libclang_rt.builtins-riscv64.a",
        "/usr/lib/llvm-20/lib/clang/20/lib/linux/libclang_rt.builtins-riscv64.a",
        "/usr/lib/llvm-19/lib/clang/19/lib/linux/libclang_rt.builtins-riscv64.a",
        "/usr/lib/llvm-18/lib/clang/18/lib/linux/libclang_rt.builtins-riscv64.a",
    ] {
        if Path::new(candidate).is_file() {
            return Some(candidate.to_string());
        }
    }
    None
}

fn link_sifive_object_to_shared(input_obj: &Path, output_so: &Path) -> io::Result<()> {
    let stem = output_so
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("sifive_kernel");
    let dir = output_so.parent().unwrap_or_else(|| Path::new("."));
    let host_stub_src = dir.join(format!("{stem}.host_stubs.c"));
    let host_stub_obj = dir.join(format!("{stem}.host_stubs.o"));
    let tmp_shared_src = dir.join(format!("{stem}.tmp_shared_stubs.c"));
    let tmp_shared_obj = dir.join(format!("{stem}.tmp_shared_stubs.o"));

    fs::write(&host_stub_src, SIFIVE_KERNEL_HOST_STUBS_C)?;
    compile_sifive_host_stub(&host_stub_src, &host_stub_obj)?;
    write_tmp_shared_stub_source(input_obj, &tmp_shared_src)?;
    compile_sifive_host_stub(&tmp_shared_src, &tmp_shared_obj)?;

    let cc = sifive_link_cc_tool();
    let mut cmd = Command::new(&cc);
    if command_is_clang(&cc) {
        add_riscv_linux_clang_args(&mut cmd);
    }
    if let Ok(linker) = std::env::var("HETGPU_SIFIVE_HOST_LINKER") {
        if !linker.trim().is_empty() {
            cmd.arg(format!("-fuse-ld={}", linker.trim()));
        }
    } else {
        // mold rejects the experimental vendor ISA names emitted for SIFIVE
        // objects in .riscv.attributes. GNU ld.bfd links the same objects.
        cmd.arg("-fuse-ld=bfd");
    }
    if let Ok(flags) = std::env::var("HETGPU_SIFIVE_HOST_LINK_FLAGS") {
        cmd.args(flags.split_ascii_whitespace());
    }
    cmd.arg("-shared")
        .arg("-fPIC")
        .arg("-o")
        .arg(output_so)
        .arg(input_obj)
        .arg(&host_stub_obj)
        .arg(&tmp_shared_obj);
    if let Some(builtins) = find_riscv_builtins_archive() {
        cmd.arg(builtins);
    }
    cmd.arg("-lm").arg("-ldl");
    run_command(&mut cmd, "SIFIVE host object -> shared object")
}

fn resolve_original_input_path(
    file_name: &str,
    working_directory: Option<&Path>,
) -> Option<PathBuf> {
    let path = Path::new(file_name);
    if path.is_absolute() && path.is_file() {
        return Some(path.to_path_buf());
    }

    let candidate = working_directory?.join(path);
    if candidate.is_file() {
        Some(candidate)
    } else {
        None
    }
}

fn compile_bc_to_sifive_object(
    input_file: &Path,
    output_file: &Path,
) -> io::Result<crate::SifiveConfig> {
    let primary = crate::SifiveConfig::xm_hardware();
    match compile_bc_to_xm_object(input_file, output_file, &primary) {
        Ok(()) => Ok(primary),
        Err(primary_err) => {
            let fallback = crate::SifiveConfig::rvv_linux_bf16();
            match compile_bc_to_xm_object(input_file, output_file, &fallback) {
                Ok(()) => {
                    eprintln!(
                        "SIFIVE: primary XM codegen failed for {} ({}); used RVV/Linux fallback",
                        input_file.display(),
                        primary_err
                    );
                    Ok(fallback)
                }
                Err(fallback_err) => Err(io::Error::other(format!(
                    "primary XM codegen failed: {}; RVV/Linux fallback failed: {}",
                    primary_err, fallback_err
                ))),
            }
        }
    }
}

fn compile_bc_to_xm_object(
    input_file: &Path,
    output_file: &Path,
    config: &crate::SifiveConfig,
) -> io::Result<()> {
    if !env_truthy("HETGPU_SIFIVE_DIRECT_BC") {
        eprintln!(
            "SIFIVE: using sanitized textual IR path for {} (set HETGPU_SIFIVE_DIRECT_BC=1 to try direct bitcode codegen)",
            input_file.display()
        );
        return compile_bc_to_xm_object_via_sanitized_ll(input_file, output_file, config);
    }

    if bc_requires_sanitized_ll(input_file)? {
        eprintln!(
            "SIFIVE: forcing sanitized textual IR path for {} due to unsupported source target metadata",
            input_file.display()
        );
        return compile_bc_to_xm_object_via_sanitized_ll(input_file, output_file, config);
    }

    let clang = sifive_clang_tool("HETGPU_SIFIVE_CLANG");
    let march = riscv_march_with_required_extensions(&sifive_codegen_march(config));
    let mut cmd = Command::new(clang);
    cmd.arg("-target")
        .arg(&config.target_triple)
        .arg("-fPIC")
        .arg("-mllvm")
        .arg("-non-global-value-max-name-size=16384")
        .arg("-menable-experimental-extensions")
        .arg(format!("-march={}", march))
        .arg("-Wno-override-module")
        .arg("-c")
        .arg(input_file)
        .arg("-o")
        .arg(output_file);

    if config.target_triple.contains("linux") {
        let sysroot =
            std::env::var("HETGPU_SIFIVE_SOURCE_SYSROOT").unwrap_or_else(|_| "/".to_string());
        let gcc_toolchain = std::env::var("HETGPU_SIFIVE_SOURCE_GCC_TOOLCHAIN")
            .unwrap_or_else(|_| "/usr".to_string());
        cmd.arg(format!("--sysroot={}", sysroot))
            .arg(format!("--gcc-toolchain={}", gcc_toolchain));
    }

    match run_command(&mut cmd, "LLVM bitcode -> RISC-V object") {
        Ok(()) => Ok(()),
        Err(err) => {
            eprintln!(
                "SIFIVE: direct BC->object failed for {}; retrying via sanitized textual IR: {}",
                input_file.display(),
                err
            );
            compile_bc_to_xm_object_via_sanitized_ll(input_file, output_file, config)
        }
    }
}

fn bc_requires_sanitized_ll(input_file: &Path) -> io::Result<bool> {
    let llvm_dis = llvm_dis_tool();
    let mut cmd = Command::new(llvm_dis);
    cmd.arg(input_file).arg("-o").arg("-");
    let output = tool_output(&mut cmd)?;
    if !output.status.success() {
        return Ok(false);
    }
    let ll = String::from_utf8_lossy(&output.stdout);
    Ok(ll.contains("@llvm.amdgcn.wave.barrier")
        || ll.contains("call void @llvm.amdgcn.wave.barrier(")
        || ll.contains("target triple = \"amdgcn")
        || ll.contains("\"target-cpu\"=")
        || ll.contains("\"target-features\"="))
}

fn compile_bc_to_xm_assembly(
    input_file: &Path,
    output_file: &Path,
    config: &crate::SifiveConfig,
) -> io::Result<()> {
    let clang = sifive_clang_tool("HETGPU_SIFIVE_CLANG");
    let march = riscv_march_with_required_extensions(&sifive_codegen_march(config));
    let mut cmd = Command::new(clang);
    cmd.arg("-target")
        .arg(&config.target_triple)
        .arg("-fPIC")
        .arg("-mllvm")
        .arg("-non-global-value-max-name-size=16384")
        .arg("-menable-experimental-extensions")
        .arg(format!("-march={}", march))
        .arg("-Wno-override-module")
        .arg("-S")
        .arg(input_file)
        .arg("-o")
        .arg(output_file);

    if config.target_triple.contains("linux") {
        let sysroot =
            std::env::var("HETGPU_SIFIVE_SOURCE_SYSROOT").unwrap_or_else(|_| "/".to_string());
        let gcc_toolchain = std::env::var("HETGPU_SIFIVE_SOURCE_GCC_TOOLCHAIN")
            .unwrap_or_else(|_| "/usr".to_string());
        cmd.arg(format!("--sysroot={}", sysroot))
            .arg(format!("--gcc-toolchain={}", gcc_toolchain));
    }

    run_command(&mut cmd, "LLVM bitcode -> RISC-V assembly")
}

fn compile_bc_to_xm_object_via_sanitized_ll(
    input_file: &Path,
    output_file: &Path,
    config: &crate::SifiveConfig,
) -> io::Result<()> {
    let llvm_dis = llvm_dis_tool();
    let ll_file = output_file.with_extension("from_bc.ll");
    let sanitized_ll_file = output_file.with_extension("sanitized.ll");

    let mut dis_cmd = Command::new(llvm_dis);
    dis_cmd.arg(input_file).arg("-o").arg(&ll_file);
    run_command(&mut dis_cmd, "LLVM bitcode -> textual LLVM IR")?;

    let ll_text = fs::read_to_string(&ll_file)?;
    let sanitized = sanitize_llvm23_ir_for_llvm20(&ll_text);
    fs::write(&sanitized_ll_file, sanitized)?;

    let clang = sifive_clang_tool("HETGPU_SIFIVE_CLANG");
    if !tool_is_available(&clang) {
        eprintln!(
            "SIFIVE: clang tool '{}' is unavailable; using llc for sanitized LLVM IR codegen",
            clang
        );
        return compile_sanitized_ll_to_xm_object_with_llc(&sanitized_ll_file, output_file, config);
    }

    let march = riscv_march_with_required_extensions(&sifive_codegen_march(config));
    let mut cmd = Command::new(clang);
    cmd.arg("-target")
        .arg(&config.target_triple)
        .arg("-fPIC")
        .arg("-mllvm")
        .arg("-non-global-value-max-name-size=16384")
        .arg("-menable-experimental-extensions")
        .arg(format!("-march={}", march))
        .arg("-Wno-override-module")
        .arg("-c")
        .arg(&sanitized_ll_file)
        .arg("-o")
        .arg(output_file);

    if config.target_triple.contains("linux") {
        let sysroot =
            std::env::var("HETGPU_SIFIVE_SOURCE_SYSROOT").unwrap_or_else(|_| "/".to_string());
        let gcc_toolchain = std::env::var("HETGPU_SIFIVE_SOURCE_GCC_TOOLCHAIN")
            .unwrap_or_else(|_| "/usr".to_string());
        cmd.arg(format!("--sysroot={}", sysroot))
            .arg(format!("--gcc-toolchain={}", gcc_toolchain));
    }

    match run_command(&mut cmd, "sanitized LLVM IR -> RISC-V object") {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            eprintln!(
                "SIFIVE: clang invocation failed for sanitized LLVM IR ({}); retrying with llc",
                err
            );
            compile_sanitized_ll_to_xm_object_with_llc(&sanitized_ll_file, output_file, config)
        }
        Err(err) => Err(err),
    }
}

fn compile_sanitized_ll_to_xm_object_with_llc(
    input_ll: &Path,
    output_file: &Path,
    config: &crate::SifiveConfig,
) -> io::Result<()> {
    let llc = llc_tool();
    let march = riscv_march_with_required_extensions(&sifive_codegen_march(config));
    let mattr = riscv_llc_mattr_from_march(&march);
    let mut cmd = Command::new(llc);
    cmd.arg(format!("-mtriple={}", config.target_triple))
        .arg("-filetype=obj")
        .arg("-relocation-model=pic")
        .arg(format!("-mattr={}", mattr))
        .arg(input_ll)
        .arg("-o")
        .arg(output_file);
    run_command(&mut cmd, "sanitized LLVM IR -> RISC-V object via llc")
}

fn riscv_llc_mattr_from_march(march: &str) -> String {
    let lower = march.to_ascii_lowercase();
    let mut attrs: Vec<String> = Vec::new();
    let mut push_attr = |name: &str| {
        if !name.is_empty() && !attrs.iter().any(|existing| existing == name) {
            attrs.push(name.to_string());
        }
    };

    let rest = lower
        .strip_prefix("rv32")
        .or_else(|| lower.strip_prefix("rv64"))
        .unwrap_or(lower.as_str());
    let mut parts = rest.split('_');
    if let Some(base) = parts.next() {
        if base.contains('g') {
            for ext in ["m", "a", "f", "d"] {
                push_attr(ext);
            }
        }
        for ext in ["m", "a", "f", "d", "c", "v"] {
            if base.contains(ext) {
                push_attr(ext);
            }
        }
    }
    for ext in parts {
        push_attr(ext);
    }

    if attrs.is_empty() {
        "+m,+a,+f,+d,+c".to_string()
    } else {
        attrs
            .into_iter()
            .map(|attr| format!("+{attr}"))
            .collect::<Vec<_>>()
            .join(",")
    }
}

fn sanitize_llvm23_ir_for_llvm20(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for line in input.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("target datalayout =") {
            out.push_str("target datalayout = \"e-m:e-p:64:64-i64:64-i128:128-n32:64-S128\"\n");
            continue;
        }
        if trimmed.starts_with("target triple =") {
            out.push_str("target triple = \"riscv64-unknown-elf\"\n");
            continue;
        }
        if trimmed.contains("llvm.amdgcn.wave.barrier") {
            continue;
        }
        if trimmed.starts_with('@') && trimmed.contains(" = external addrspace(3) global [0 x ") {
            if let Some((name, _rhs)) = trimmed.split_once(" = ") {
                out.push_str(name);
                out.push_str(
                    " = internal addrspace(3) global [65536 x i8] zeroinitializer, align 16\n",
                );
                continue;
            }
        }
        if trimmed.starts_with("attributes #") {
            if let Some((lhs, _rhs)) = line.split_once('=') {
                out.push_str(lhs.trim());
                out.push_str(" = { nounwind }\n");
            } else {
                out.push_str(line);
                out.push('\n');
            }
            continue;
        }

        let mut owned = line.to_string();
        for token in [
            " nocallback",
            " nocreateundeforpoison",
            " nofree",
            " nosync",
            " willreturn",
            " speculatable",
            " mustprogress",
            " memory(none)",
            " memory(argmem: readwrite)",
            " memory(argmem: read)",
            " memory(read)",
            " memory(write)",
            " captures(none)",
            " captures(address)",
            " captures(provenance)",
            " captures(address, provenance)",
            " denormal-fp-math-f32=\"ieee,ieee\"",
            " denormal-fp-math=\"ieee,ieee\"",
            " denormal-fp-math-f64=\"ieee,ieee\"",
            " denormal-fp-math-f16=\"ieee,ieee\"",
            " denormal-fp-math-f32=\"preserve-sign,preserve-sign\"",
            " denormal-fp-math=\"preserve-sign,preserve-sign\"",
            " denormal-fp-math-f64=\"preserve-sign,preserve-sign\"",
            " denormal-fp-math-f16=\"preserve-sign,preserve-sign\"",
            " denormal-fp-math-f32=\"positive-zero,positive-zero\"",
            " denormal-fp-math=\"positive-zero,positive-zero\"",
            " denormal-fpenv(\"dynamic\")",
            " denormal-fpenv(dynamic)",
            " denormal-fpenv(\"ieee\")",
            " denormal-fpenv(ieee)",
        ] {
            owned = owned.replace(token, "");
        }
        out.push_str(&owned);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{riscv_march_with_required_extensions, sanitize_llvm23_ir_for_llvm20};

    #[test]
    fn sanitize_attribute_group_does_not_duplicate_attributes_keyword() {
        let input = "attributes #0 = { nocallback nofree nounwind willreturn memory(none) }\n";
        let output = sanitize_llvm23_ir_for_llvm20(input);
        assert_eq!(output, "attributes #0 = { nounwind }\n");
    }

    #[test]
    fn sanitize_drops_wave_barrier_lines() {
        let input = "declare void @llvm.amdgcn.wave.barrier()\ncall void @llvm.amdgcn.wave.barrier()\nret void\n";
        let output = sanitize_llvm23_ir_for_llvm20(input);
        assert_eq!(output, "ret void\n");
    }

    #[test]
    fn sanitize_materializes_dynamic_shared_memory_symbol() {
        let input = "@smem = external addrspace(3) global [0 x i8], align 4\n";
        let output = sanitize_llvm23_ir_for_llvm20(input);
        assert_eq!(
            output,
            "@smem = internal addrspace(3) global [65536 x i8] zeroinitializer, align 16\n"
        );
    }

    #[test]
    fn sanitize_materializes_named_dynamic_shared_memory_symbol() {
        let input = "@data_mmv = external addrspace(3) global [0 x i8], align 1\n";
        let output = sanitize_llvm23_ir_for_llvm20(input);
        assert_eq!(
            output,
            "@data_mmv = internal addrspace(3) global [65536 x i8] zeroinitializer, align 16\n"
        );
    }

    #[test]
    fn sanitize_rewrites_amdgpu_datalayout_for_riscv_cpu_pointers() {
        let input = "target datalayout = \"e-p:64:64-p5:32:32\"\n";
        let output = sanitize_llvm23_ir_for_llvm20(input);
        assert_eq!(
            output,
            "target datalayout = \"e-m:e-p:64:64-i64:64-i128:128-n32:64-S128\"\n"
        );
    }

    #[test]
    fn sanitize_rewrites_module_target_triple_before_riscv_opt() {
        let input = "target triple = \"amdgcn-amd-amdhsa\"\n";
        let output = sanitize_llvm23_ir_for_llvm20(input);
        assert_eq!(output, "target triple = \"riscv64-unknown-elf\"\n");
    }

    #[test]
    fn riscv_march_adds_zbb_for_backend_orc_b_selection() {
        assert_eq!(
            riscv_march_with_required_extensions(
                "rv64gcv_zvfbfmin_xsfvcp_xsfvfnrclipxfqf_xsfvqmaccqoq"
            ),
            "rv64gcv_zvfbfmin_xsfvcp_xsfvfnrclipxfqf_xsfvqmaccqoq_zbb"
        );
        assert_eq!(
            riscv_march_with_required_extensions("rv64gcv_zbb_zvfbfmin"),
            "rv64gcv_zbb_zvfbfmin"
        );
    }
}

fn source_extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
}

fn is_c_source(path: &Path) -> bool {
    matches!(source_extension(path).as_deref(), Some("c"))
}

fn is_cxx_source(path: &Path) -> bool {
    matches!(
        source_extension(path).as_deref(),
        Some("cc" | "cp" | "cxx" | "cpp" | "c++")
    )
}

fn preferred_tool(env_var: &str, fallbacks: &[&str]) -> String {
    if !env_var.is_empty() {
        if let Ok(value) = std::env::var(env_var) {
            if !value.trim().is_empty() {
                return value;
            }
        }
    }

    for tool in fallbacks {
        if tool.contains('/') {
            if Path::new(tool).is_file() {
                return (*tool).to_string();
            }
        }
    }

    fallbacks.first().copied().unwrap_or("clang").to_string()
}

fn llvm_link_tool() -> String {
    bundled_llvm_tool(
        "HETGPU_SIFIVE_LLVM_LINK",
        "llvm-link",
        &["llvm-link-21", "/usr/bin/llvm-link-21", "llvm-link"],
    )
}

fn opt_tool() -> String {
    bundled_llvm_tool(
        "HETGPU_SIFIVE_OPT",
        "opt",
        &["opt-21", "/usr/bin/opt-21", "opt"],
    )
}

fn llvm_dis_tool() -> String {
    bundled_llvm_tool(
        "HETGPU_SIFIVE_LLVM_DIS",
        "llvm-dis",
        &["llvm-dis-21", "/usr/bin/llvm-dis-21", "llvm-dis"],
    )
}

fn llc_tool() -> String {
    bundled_llvm_tool(
        "HETGPU_SIFIVE_LLC",
        "llc",
        &["llc-21", "/usr/bin/llc-21", "llc"],
    )
}

fn sifive_clang_tool(env_var: &str) -> String {
    bundled_llvm_tool(
        env_var,
        "clang",
        &["clang-21", "/usr/bin/clang-21", "clang"],
    )
}

fn existing_tool(path: PathBuf) -> Option<String> {
    if path.is_file() {
        Some(path.display().to_string())
    } else {
        None
    }
}

fn tool_is_available(tool: &str) -> bool {
    let path = Path::new(tool);
    if path.components().count() > 1 {
        return path.is_file();
    }

    let Some(path_var) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path_var).any(|dir| dir.join(tool).is_file())
}

fn llvm_tool_from_build_dir(build_dir: &Path, tool_name: &str) -> Option<String> {
    let mut roots: Vec<_> = fs::read_dir(build_dir)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .collect();
    roots.sort_by_key(|path| {
        fs::metadata(path)
            .and_then(|metadata| metadata.modified())
            .ok()
    });
    roots.reverse();

    for root in roots {
        for path in [
            root.join("out").join("build").join("bin").join(tool_name),
            root.join("out").join("build").join("tools").join(tool_name),
            root.join("out")
                .join("build")
                .join("Release")
                .join("bin")
                .join(tool_name),
            root.join("out")
                .join("build")
                .join("Release")
                .join("tools")
                .join(tool_name),
            root.join("out")
                .join("build")
                .join("Debug")
                .join("bin")
                .join(tool_name),
            root.join("out")
                .join("build")
                .join("Debug")
                .join("tools")
                .join(tool_name),
            root.join("out").join("bin").join(tool_name),
        ] {
            if let Some(tool) = existing_tool(path) {
                return Some(tool);
            }
        }
    }
    None
}

fn llvm_tool_from_target_dir(target_dir: &Path, tool_name: &str) -> Option<String> {
    for candidate in [
        target_dir.join("debug/build"),
        target_dir.join("release/build"),
        target_dir.join("build"),
    ] {
        if let Some(tool) = llvm_tool_from_build_dir(&candidate, tool_name) {
            return Some(tool);
        }
    }
    None
}

fn llvm_tool_from_prebuilt_dir(prebuilt_dir: &Path, tool_name: &str) -> Option<String> {
    for candidate in [
        prebuilt_dir.join("bin").join(tool_name),
        prebuilt_dir.join("tools").join(tool_name),
        prebuilt_dir.join("build").join("bin").join(tool_name),
        prebuilt_dir.join("build").join("tools").join(tool_name),
        prebuilt_dir
            .join("build")
            .join("Release")
            .join("bin")
            .join(tool_name),
        prebuilt_dir
            .join("build")
            .join("Release")
            .join("tools")
            .join(tool_name),
        prebuilt_dir
            .join("build")
            .join("Debug")
            .join("bin")
            .join(tool_name),
        prebuilt_dir
            .join("build")
            .join("Debug")
            .join("tools")
            .join(tool_name),
        prebuilt_dir
            .join("out")
            .join("build")
            .join("bin")
            .join(tool_name),
        prebuilt_dir
            .join("out")
            .join("build")
            .join("tools")
            .join(tool_name),
        prebuilt_dir
            .join("out")
            .join("build")
            .join("Release")
            .join("bin")
            .join(tool_name),
        prebuilt_dir
            .join("out")
            .join("build")
            .join("Release")
            .join("tools")
            .join(tool_name),
        prebuilt_dir
            .join("out")
            .join("build")
            .join("Debug")
            .join("bin")
            .join(tool_name),
        prebuilt_dir
            .join("out")
            .join("build")
            .join("Debug")
            .join("tools")
            .join(tool_name),
    ] {
        if let Some(tool) = existing_tool(candidate) {
            return Some(tool);
        }
    }
    None
}

fn bundled_llvm_tool(env_var: &str, tool_name: &str, fallbacks: &[&str]) -> String {
    if let Ok(value) = std::env::var(env_var) {
        if !value.trim().is_empty() {
            return value;
        }
    }

    if let Ok(dir) = std::env::var("HETGPU_SIFIVE_LLVM_TOOLS_DIR") {
        if let Some(tool) = existing_tool(PathBuf::from(dir).join(tool_name)) {
            return tool;
        }
    }

    for dir in [
        std::env::var("LLVM_ZLUDA_PREBUILT").ok(),
        option_env!("LLVM_ZLUDA_PREBUILT").map(|s| s.to_string()),
    ]
    .into_iter()
    .flatten()
    {
        if !dir.trim().is_empty() {
            if let Some(tool) = llvm_tool_from_prebuilt_dir(Path::new(&dir), tool_name) {
                return tool;
            }
        }
    }

    for env in ["HETGPU_SIFIVE_HELPER_TARGET_DIR", "CARGO_TARGET_DIR"] {
        if let Ok(dir) = std::env::var(env) {
            if let Some(tool) = llvm_tool_from_target_dir(Path::new(&dir), tool_name) {
                return tool;
            }
        }
    }

    if let Ok(repo_root) = workspace_root() {
        if let Some(tool) = llvm_tool_from_target_dir(&repo_root.join("target"), tool_name) {
            return tool;
        }
    }

    for dir in [
        "/mnt/usb/hetgpu_build_target/releasefix",
        "/mnt/usb/hetgpu_build_target/release",
        "/mnt/usb/hetgpu_build_target",
    ] {
        if let Some(tool) = llvm_tool_from_target_dir(Path::new(dir), tool_name) {
            return tool;
        }
    }

    if tool_name != "clang" {
        let clang = sifive_clang_tool("HETGPU_SIFIVE_CLANG");
        let mut clang_candidates = vec![PathBuf::from(&clang)];
        if let Ok(resolved) = PathBuf::from(&clang).canonicalize() {
            if !clang_candidates.iter().any(|path| path == &resolved) {
                clang_candidates.push(resolved);
            }
        }
        for clang_path in clang_candidates {
            if let Some(parent) = clang_path.parent() {
                for candidate in [
                    parent.join(tool_name),
                    parent.join(format!("{tool_name}-21")),
                ] {
                    if let Some(tool) = existing_tool(candidate) {
                        return tool;
                    }
                }
            }
        }
    }

    preferred_tool(env_var, fallbacks)
}

fn ensure_ptx_helper_tool(tool_name: &str) -> io::Result<PathBuf> {
    let repo_root = workspace_root()?;
    let mut candidate_roots = Vec::new();

    if let Ok(dir) = std::env::var("HETGPU_SIFIVE_HELPER_TARGET_DIR") {
        if !dir.trim().is_empty() {
            candidate_roots.push(PathBuf::from(dir));
        }
    }
    if let Ok(dir) = std::env::var("CARGO_TARGET_DIR") {
        if !dir.trim().is_empty() {
            let path = PathBuf::from(dir);
            if !candidate_roots.iter().any(|p| p == &path) {
                candidate_roots.push(path);
            }
        }
    }
    candidate_roots.push(repo_root.join("target"));

    for root in &candidate_roots {
        let tool_path = root.join("debug").join(tool_name);
        if tool_path.is_file() {
            return Ok(tool_path);
        }
    }

    let build_target_dir = candidate_roots
        .first()
        .cloned()
        .unwrap_or_else(|| repo_root.join("target"));
    let status = Command::new("bash")
        .arg("-lc")
        .arg(format!(
            "cd {} && cargo build -p ptx --features sifive --bin {} --target-dir {}",
            shell_escape_path(&repo_root),
            tool_name,
            shell_escape_path(&build_target_dir),
        ))
        .status()?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "cargo failed while building {}",
            tool_name
        )));
    }

    let tool_path = build_target_dir.join("debug").join(tool_name);
    if !tool_path.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("expected helper binary at {}", tool_path.display()),
        ));
    }

    Ok(tool_path)
}

fn workspace_root() -> io::Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .join("../..")
        .canonicalize()
        .map_err(|e| io::Error::other(format!("failed to locate workspace root: {}", e)))
}

fn run_command(cmd: &mut Command, what: &str) -> io::Result<()> {
    let output = tool_output(cmd)?;
    if output.status.success() {
        return Ok(());
    }

    Err(io::Error::other(format!(
        "{} failed: {}",
        what,
        String::from_utf8_lossy(&output.stderr)
    )))
}

fn tool_output(cmd: &mut Command) -> io::Result<Output> {
    // llama.cpp starts this runtime via LD_PRELOAD=libnvcuda.so. Compiler and
    // binutils subprocesses must not inherit that preload, or the CUDA shim can
    // be injected into clang/gcc/nm and crash inside their allocators.
    cmd.env_remove("LD_PRELOAD")
        .env_remove("DYLD_INSERT_LIBRARIES");
    cmd.output()
}

fn shell_escape_path(path: &Path) -> String {
    let text = path.to_string_lossy();
    format!("'{}'", text.replace('\'', "'\\''"))
}

#[repr(C, align(8))]
struct FatbinHeader {
    magic: u32,
    version: u16,
    header_size: u16,
    files_size: u64,
}

#[repr(C)]
struct FatbinFileHeader {
    kind: u16,
    version: u16,
    header_size: u32,
    padded_payload_size: u32,
    unknown0: u32,
    payload_size: u32,
    unknown1: u32,
    unknown2: u32,
    sm_version: u32,
    bit_width: u32,
    unknown3: u32,
    unknown4: u64,
    unknown5: u64,
    uncompressed_payload: u64,
}

fn create_cuda_fatbin_with_elf(elf_bytes: &[u8]) -> Vec<u8> {
    const FATBIN_MAGIC: u32 = 0xBA55ED50;
    const FATBIN_VERSION: u16 = 0x0001;
    const FATBIN_KIND_ELF: u16 = 0x0002;
    const FATBIN_FILE_HEADER_VERSION_CURRENT: u16 = 0x0101;

    let header_size = mem::size_of::<FatbinHeader>();
    let file_header_size = mem::size_of::<FatbinFileHeader>();
    let padded_payload_size = align_up(elf_bytes.len(), 8);
    let file_header = FatbinFileHeader {
        kind: FATBIN_KIND_ELF,
        version: FATBIN_FILE_HEADER_VERSION_CURRENT,
        header_size: file_header_size as u32,
        padded_payload_size: padded_payload_size as u32,
        unknown0: 0,
        payload_size: elf_bytes.len() as u32,
        unknown1: 0,
        unknown2: 0,
        sm_version: 90,
        bit_width: 64,
        unknown3: 0,
        unknown4: 0,
        unknown5: 0,
        uncompressed_payload: 0,
    };

    let mut files = Vec::with_capacity(file_header_size + padded_payload_size);
    files.extend_from_slice(as_bytes(&file_header));
    files.extend_from_slice(elf_bytes);
    files.resize(file_header_size + padded_payload_size, 0);

    let header = FatbinHeader {
        magic: FATBIN_MAGIC,
        version: FATBIN_VERSION,
        header_size: header_size as u16,
        files_size: files.len() as u64,
    };

    let mut fatbin = Vec::with_capacity(header_size + files.len());
    fatbin.extend_from_slice(as_bytes(&header));
    fatbin.extend_from_slice(&files);
    fatbin
}

fn align_up(value: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return value;
    }
    (value + alignment - 1) & !(alignment - 1)
}

fn as_bytes<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), mem::size_of::<T>()) }
}

fn create_dummy_riscv_elf() -> Vec<u8> {
    let mut elf = Vec::with_capacity(128);

    // ELF header for RISC-V 64-bit
    elf.extend_from_slice(&[
        0x7f, 0x45, 0x4c, 0x46, // ELF magic
        0x02, // 64-bit
        0x01, // Little endian
        0x01, // Current version
        0x00, // System V ABI
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Padding
    ]);
    elf.extend_from_slice(&[0x01, 0x00]); // ET_REL
    elf.extend_from_slice(&[0xf3, 0x00]); // EM_RISCV
    elf.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]); // Version
    elf.extend_from_slice(&[0; 8]); // Entry point
    elf.extend_from_slice(&[0; 8]); // Program header offset
    elf.extend_from_slice(&[0; 8]); // Section header offset
    elf.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // Flags
    elf.extend_from_slice(&[0x40, 0x00]); // ELF header size
    elf.extend_from_slice(&[0x00, 0x00]); // Program header entry size
    elf.extend_from_slice(&[0x00, 0x00]); // Program header entry count
    elf.extend_from_slice(&[0x40, 0x00]); // Section header entry size
    elf.extend_from_slice(&[0x03, 0x00]); // Section header entry count
    elf.extend_from_slice(&[0x02, 0x00]); // Section name string table index

    // SIFIVE marker
    elf.extend_from_slice(b"SIFIVE_RISCV_IME_VCIX\0");

    elf
}
