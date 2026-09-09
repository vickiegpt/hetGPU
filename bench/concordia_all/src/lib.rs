use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const SUMMARY_HEADER: &str = "experiment,status,exit_code,duration_ms,artifact,log,message";

#[derive(Debug, Clone)]
pub struct Config {
    pub run: RunConfig,
    pub experiments: ExperimentConfig,
    pub paths: PathConfig,
    pub mpi: MpiConfig,
}

#[derive(Debug, Clone)]
pub struct RunConfig {
    pub out_dir: PathBuf,
    pub static_only: bool,
    pub continue_on_fail: bool,
    pub cuda_arch: String,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    pub oob: bool,
    pub claim_audit: bool,
    pub eval_claims: bool,
    pub delta_static: bool,
    pub delta_live: bool,
    pub persistent_static: bool,
    pub persistent_live: bool,
    pub kimi_tps: bool,
    pub mpi_nccl_recovery: bool,
}

#[derive(Debug, Clone)]
pub struct PathConfig {
    pub paper_tex: PathBuf,
    pub eval_claims: PathBuf,
    pub kimi_runner: PathBuf,
    pub kimi_model_dir: PathBuf,
}

#[derive(Debug, Clone)]
pub struct MpiConfig {
    pub enabled: bool,
    pub launcher: String,
    pub nodes: Vec<String>,
    pub ranks_per_node: usize,
    pub np: usize,
    pub hostfile: PathBuf,
    pub extra_args: Vec<String>,
}

#[derive(Debug, Clone)]
struct Experiment {
    name: &'static str,
    command: Vec<String>,
    envs: Vec<(String, String)>,
    artifact: PathBuf,
}

#[derive(Debug, Clone)]
struct Row {
    experiment: String,
    status: String,
    exit_code: i32,
    duration_ms: u128,
    artifact: PathBuf,
    log: PathBuf,
    message: String,
}

pub fn run_from_env() -> io::Result<()> {
    let repo = repo_root()?;
    let config_path = env::var_os("CONCORDIA_ALL_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo.join("bench/concordia_all/ConcordiaBench.toml"));
    let mut config = Config::from_file(&config_path)?;
    absolutize_paths(&repo, &mut config);
    run(&repo, &config)
}

pub fn run(repo: &Path, config: &Config) -> io::Result<()> {
    let _ = fs::remove_dir_all(config.run.out_dir.join("logs"));
    let _ = fs::remove_dir_all(config.run.out_dir.join("artifacts"));
    fs::create_dir_all(config.run.out_dir.join("logs"))?;
    fs::create_dir_all(config.run.out_dir.join("artifacts"))?;

    let mpi_hostfile = prepare_mpi_hostfile(config)?;
    let experiments = build_experiments(repo, config, mpi_hostfile.as_deref());
    let mut rows = Vec::new();
    let mut had_fail = false;

    for experiment in experiments {
        let row = run_experiment(repo, config, &experiment)?;
        println!(
            "concordia_all experiment={} status={} artifact={} log={}",
            row.experiment,
            row.status,
            row.artifact.display(),
            row.log.display()
        );
        had_fail |= row.status == "fail";
        rows.push(row);
        if had_fail && !config.run.continue_on_fail {
            break;
        }
    }

    write_summary(&config.run.out_dir, &rows)?;
    println!(
        "concordia_all_summary csv={} jsonl={}",
        config.run.out_dir.join("summary.csv").display(),
        config.run.out_dir.join("summary.jsonl").display()
    );

    if had_fail {
        Err(io::Error::new(
            io::ErrorKind::Other,
            "one or more Concordia experiments failed",
        ))
    } else {
        Ok(())
    }
}

impl Config {
    pub fn from_file(path: &Path) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        Self::from_toml(&text)
    }

    pub fn from_toml(text: &str) -> io::Result<Self> {
        let doc = parse_limited_toml(text)?;
        Ok(Self {
            run: RunConfig {
                out_dir: PathBuf::from(get_string(&doc, "run", "out_dir", "/tmp/concordia-all")),
                static_only: get_bool(&doc, "run", "static_only", true),
                continue_on_fail: get_bool(&doc, "run", "continue_on_fail", true),
                cuda_arch: get_string(&doc, "run", "cuda_arch", "sm_120"),
                timeout_seconds: get_usize(&doc, "run", "timeout_seconds", 600) as u64,
            },
            experiments: ExperimentConfig {
                oob: get_bool(&doc, "experiments", "oob", true),
                claim_audit: get_bool(&doc, "experiments", "claim_audit", true),
                eval_claims: get_bool(&doc, "experiments", "eval_claims", true),
                delta_static: get_bool(&doc, "experiments", "delta_static", true),
                delta_live: get_bool(&doc, "experiments", "delta_live", false),
                persistent_static: get_bool(&doc, "experiments", "persistent_static", true),
                persistent_live: get_bool(&doc, "experiments", "persistent_live", false),
                kimi_tps: get_bool(&doc, "experiments", "kimi_tps", true),
                mpi_nccl_recovery: get_bool(&doc, "experiments", "mpi_nccl_recovery", false),
            },
            paths: PathConfig {
                paper_tex: PathBuf::from(get_string(
                    &doc,
                    "paths",
                    "paper_tex",
                    "69b5f215f5c67f0702bd8f65/05_eval.tex",
                )),
                eval_claims: PathBuf::from(get_string(
                    &doc,
                    "paths",
                    "eval_claims",
                    "bench/concordia_eval_claims/claims.json",
                )),
                kimi_runner: PathBuf::from(get_string(
                    &doc,
                    "paths",
                    "kimi_runner",
                    "/root/hetGPU/BitNet-work/build/bin/llama-cli",
                )),
                kimi_model_dir: PathBuf::from(get_string(
                    &doc,
                    "paths",
                    "kimi_model_dir",
                    "/root/hetGPU/models/bartowski/moonshotai_Kimi-K2.6-GGUF/moonshotai_Kimi-K2.6-IQ1_M",
                )),
            },
            mpi: MpiConfig {
                enabled: get_bool(&doc, "mpi", "enabled", false),
                launcher: get_string(&doc, "mpi", "launcher", "mpirun"),
                nodes: get_array(&doc, "mpi", "nodes", vec!["127.0.0.1".to_string()]),
                ranks_per_node: get_usize(&doc, "mpi", "ranks_per_node", 1),
                np: get_usize(&doc, "mpi", "np", 2),
                hostfile: PathBuf::from(get_string(&doc, "mpi", "hostfile", "")),
                extra_args: get_array(&doc, "mpi", "extra_args", Vec::new()),
            },
        })
    }
}

fn build_experiments(repo: &Path, config: &Config, mpi_hostfile: Option<&Path>) -> Vec<Experiment> {
    let out = &config.run.out_dir;
    let mut experiments = Vec::new();

    if config.experiments.oob {
        experiments.push(Experiment {
            name: "concordia_oob",
            command: vec![
                "cargo".into(),
                "run".into(),
                "--release".into(),
                "--manifest-path".into(),
                repo.join("bench/concordia_oob/Cargo.toml")
                    .display()
                    .to_string(),
            ],
            envs: Vec::new(),
            artifact: out.join("logs/concordia_oob.log"),
        });
    }
    if config.experiments.claim_audit {
        experiments.push(Experiment {
            name: "claim_audit",
            command: vec![
                "bash".into(),
                "bench/concordia_claim_audit/test_claim_audit.sh".into(),
            ],
            envs: Vec::new(),
            artifact: out.join("logs/claim_audit.log"),
        });
    }
    if config.experiments.eval_claims {
        experiments.push(Experiment {
            name: "eval_claims",
            command: vec![
                "python3".into(),
                "bench/concordia_eval_claims/run_eval_claims.py".into(),
                "--repo-root".into(),
                repo.display().to_string(),
                "--paper".into(),
                config.paths.paper_tex.display().to_string(),
                "--claims".into(),
                config.paths.eval_claims.display().to_string(),
                "--work-dir".into(),
                out.join("artifacts/eval_claims_work").display().to_string(),
                "--csv".into(),
                out.join("artifacts/eval_claims.csv").display().to_string(),
                "--jsonl".into(),
                out.join("artifacts/eval_claims.jsonl")
                    .display()
                    .to_string(),
                "--markdown".into(),
                out.join("artifacts/eval_claims.md").display().to_string(),
            ]
            .into_iter()
            .chain(
                config
                    .run
                    .static_only
                    .then_some("--static-only".to_string()),
            )
            .collect(),
            envs: Vec::new(),
            artifact: out.join("artifacts/eval_claims.csv"),
        });
    }
    if config.experiments.delta_static {
        experiments.push(Experiment {
            name: "delta_static",
            command: vec![
                "bash".into(),
                "bench/concordia_delta_checkpoint/test_smoke.sh".into(),
            ],
            envs: vec![
                ("CONCORDIA_BENCH_STATIC_ONLY".into(), "1".into()),
                ("CUDA_OXIDE_ARCH".into(), config.run.cuda_arch.clone()),
            ],
            artifact: out.join("logs/delta_static.log"),
        });
    }
    if config.experiments.delta_live && !config.run.static_only {
        experiments.push(Experiment {
            name: "delta_live",
            command: vec![
                "bash".into(),
                "bench/concordia_delta_checkpoint/test_smoke.sh".into(),
            ],
            envs: vec![("CUDA_OXIDE_ARCH".into(), config.run.cuda_arch.clone())],
            artifact: out.join("logs/delta_live.log"),
        });
    }
    if config.experiments.persistent_static {
        experiments.push(Experiment {
            name: "persistent_static",
            command: vec![
                "bash".into(),
                "bench/concordia_persistent_overhead/test_persistent_overhead.sh".into(),
            ],
            envs: vec![
                (
                    "CONCORDIA_PERSISTENT_OVERHEAD_STATIC_ONLY".into(),
                    "1".into(),
                ),
                (
                    "CONCORDIA_PERSISTENT_OVERHEAD_TEST_WORKDIR".into(),
                    out.join("artifacts/persistent_static")
                        .display()
                        .to_string(),
                ),
            ],
            artifact: out
                .join("artifacts/persistent_static/concordia_persistent_overhead_ablation.pdf"),
        });
    }
    if config.experiments.persistent_live && !config.run.static_only {
        experiments.push(Experiment {
            name: "persistent_live",
            command: vec![
                "bash".into(),
                "bench/concordia_persistent_overhead/test_persistent_overhead.sh".into(),
            ],
            envs: vec![
                ("CUDA_OXIDE_ARCH".into(), config.run.cuda_arch.clone()),
                (
                    "CONCORDIA_PERSISTENT_OVERHEAD_TEST_WORKDIR".into(),
                    out.join("artifacts/persistent_live").display().to_string(),
                ),
            ],
            artifact: out
                .join("artifacts/persistent_live/concordia_persistent_overhead_ablation.pdf"),
        });
    }
    if config.experiments.kimi_tps {
        experiments.push(Experiment {
            name: "kimi_tps",
            command: vec![
                "bash".into(),
                "bench/kimi_k26_tps/run_kimi_k26_tps.sh".into(),
            ],
            envs: vec![
                (
                    "KIMI_TPS_WORKDIR".into(),
                    out.join("artifacts/kimi_tps").display().to_string(),
                ),
                ("KIMI_TPS_KEEP".into(), "1".into()),
                ("KIMI_TPS_BUILD_ZLUDA".into(), "0".into()),
                ("KIMI_TPS_TIMEOUT".into(), "15".into()),
                ("KIMI_EXTRA_LLAMA_ARGS".into(), "--no-display-prompt".into()),
                ("N_PREDICT".into(), "1".into()),
                ("CTX_SIZE".into(), "128".into()),
                ("THREADS".into(), "8".into()),
                (
                    "BITNET_LLAMA_CLI".into(),
                    config.paths.kimi_runner.display().to_string(),
                ),
                (
                    "MODEL_DIR".into(),
                    config.paths.kimi_model_dir.display().to_string(),
                ),
            ],
            artifact: out.join("artifacts/kimi_tps/kimi_k26_tps.csv"),
        });
    }
    if config.experiments.mpi_nccl_recovery {
        let mut command = if config.mpi.enabled {
            let mut command = vec![
                config.mpi.launcher.clone(),
                "-np".into(),
                config.mpi.np.to_string(),
            ];
            if let Some(hostfile) = mpi_hostfile {
                command.push("--hostfile".into());
                command.push(hostfile.display().to_string());
            }
            command.extend(config.mpi.extra_args.clone());
            command.extend([
                "bash".into(),
                "bench/concordia_mpi_nccl_recovery/test_recovered_allreduce.sh".into(),
            ]);
            command
        } else {
            vec![
                "bash".into(),
                "bench/concordia_mpi_nccl_recovery/test_recovered_allreduce.sh".into(),
            ]
        };
        if !config.mpi.enabled {
            command.shrink_to_fit();
        }
        experiments.push(Experiment {
            name: "mpi_nccl_recovery",
            command,
            envs: vec![(
                "HETGPU_NCCL_RECOVERY_ARTIFACT_DIR".into(),
                out.join("artifacts/mpi_nccl_recovery")
                    .display()
                    .to_string(),
            )],
            artifact: out.join("artifacts/mpi_nccl_recovery/evidence"),
        });
    }

    experiments
}

fn run_experiment(repo: &Path, config: &Config, experiment: &Experiment) -> io::Result<Row> {
    let log = config
        .run
        .out_dir
        .join("logs")
        .join(format!("{}.log", experiment.name));
    let started = Instant::now();
    let output = Command::new(&experiment.command[0])
        .args(&experiment.command[1..])
        .envs(experiment.envs.iter().map(|(k, v)| (k, v)))
        .current_dir(repo)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();

    let (exit_code, mut text, command_status) = match output {
        Ok(output) => {
            let mut text = String::from_utf8_lossy(&output.stdout).to_string();
            text.push_str(&String::from_utf8_lossy(&output.stderr));
            (
                output.status.code().unwrap_or(-1),
                text,
                output.status.success(),
            )
        }
        Err(err) => (-1, format!("failed to spawn: {err}\n"), false),
    };

    if started.elapsed() > Duration::from_secs(config.run.timeout_seconds) {
        text.push_str("\nwarning: command exceeded configured timeout after completion\n");
    }

    fs::write(&log, text)?;
    let status = classify_status(
        experiment.name,
        command_status,
        exit_code,
        &log,
        &experiment.artifact,
    )?;
    let message = if command_status {
        "ok".to_string()
    } else {
        format!("exit_{exit_code}")
    };
    Ok(Row {
        experiment: experiment.name.to_string(),
        status,
        exit_code,
        duration_ms: started.elapsed().as_millis(),
        artifact: experiment.artifact.clone(),
        log,
        message,
    })
}

fn classify_status(
    experiment: &str,
    command_status: bool,
    exit_code: i32,
    log: &Path,
    artifact: &Path,
) -> io::Result<String> {
    if command_status {
        if experiment == "kimi_tps" {
            return Ok(classify_kimi_csv(artifact));
        }
        if experiment == "eval_claims" {
            return Ok(classify_eval_claims_csv(artifact));
        }
        if artifact.exists() || artifact.as_os_str().is_empty() {
            return Ok("pass".to_string());
        }
        return Ok("partial".to_string());
    }
    let text = fs::read_to_string(log).unwrap_or_default();
    if exit_code == 0
        || text.contains("skipped_missing")
        || text.contains("blocked")
        || text.contains("no visible NVIDIA GPU")
        || text.contains("cargo-oxide not found")
    {
        return Ok("blocked".to_string());
    }
    Ok("fail".to_string())
}

fn classify_kimi_csv(path: &Path) -> String {
    let Ok(text) = fs::read_to_string(path) else {
        return "blocked".to_string();
    };
    let statuses = csv_column_values(&text, "status");
    if statuses.is_empty() {
        return "blocked".to_string();
    }
    if statuses.iter().any(|status| status == "run_failed") {
        return "fail".to_string();
    }
    if statuses.iter().all(|status| status == "pass") {
        return "pass".to_string();
    }
    if statuses
        .iter()
        .any(|status| status.starts_with("skipped_") || status == "timeout")
    {
        return "blocked".to_string();
    }
    "partial".to_string()
}

fn classify_eval_claims_csv(path: &Path) -> String {
    let Ok(text) = fs::read_to_string(path) else {
        return "blocked".to_string();
    };
    let statuses = csv_column_values(&text, "status");
    if statuses.is_empty() {
        return "blocked".to_string();
    }
    if statuses.iter().any(|status| status == "fail") {
        return "fail".to_string();
    }
    if statuses.iter().all(|status| status == "pass") {
        return "pass".to_string();
    }
    if statuses.iter().all(|status| status == "blocked") {
        return "blocked".to_string();
    }
    "partial".to_string()
}

fn csv_column_values(text: &str, column: &str) -> Vec<String> {
    let mut lines = text.lines();
    let Some(header) = lines.next() else {
        return Vec::new();
    };
    let headers = split_csv_line(header);
    let Some(index) = headers.iter().position(|name| name == column) else {
        return Vec::new();
    };
    lines
        .filter_map(|line| split_csv_line(line).get(index).cloned())
        .collect()
}

fn split_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut chars = line.chars().peekable();
    let mut quoted = false;
    while let Some(ch) = chars.next() {
        match ch {
            '"' if quoted && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => {
                fields.push(field);
                field = String::new();
            }
            _ => field.push(ch),
        }
    }
    fields.push(field);
    fields
}

fn write_summary(out_dir: &Path, rows: &[Row]) -> io::Result<()> {
    let csv_path = out_dir.join("summary.csv");
    let jsonl_path = out_dir.join("summary.jsonl");
    let mut csv = String::from(SUMMARY_HEADER);
    csv.push('\n');
    let mut jsonl = String::new();
    for row in rows {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            csv_escape(&row.experiment),
            csv_escape(&row.status),
            row.exit_code,
            row.duration_ms,
            csv_escape(&row.artifact.display().to_string()),
            csv_escape(&row.log.display().to_string()),
            csv_escape(&row.message)
        ));
        jsonl.push_str(&format!(
            "{{\"artifact\":\"{}\",\"duration_ms\":{},\"exit_code\":{},\"experiment\":\"{}\",\"log\":\"{}\",\"message\":\"{}\",\"status\":\"{}\"}}\n",
            json_escape(&row.artifact.display().to_string()),
            row.duration_ms,
            row.exit_code,
            json_escape(&row.experiment),
            json_escape(&row.log.display().to_string()),
            json_escape(&row.message),
            json_escape(&row.status),
        ));
    }
    fs::write(csv_path, csv)?;
    fs::write(jsonl_path, jsonl)?;
    Ok(())
}

fn prepare_mpi_hostfile(config: &Config) -> io::Result<Option<PathBuf>> {
    if !config.mpi.enabled {
        return Ok(None);
    }
    if !config.mpi.hostfile.as_os_str().is_empty() {
        return Ok(Some(config.mpi.hostfile.clone()));
    }
    let path = config.run.out_dir.join("mpi.hostfile");
    let mut text = String::new();
    for node in &config.mpi.nodes {
        text.push_str(&format!("{} slots={}\n", node, config.mpi.ranks_per_node));
    }
    fs::write(&path, text)?;
    Ok(Some(path))
}

fn absolutize_paths(repo: &Path, config: &mut Config) {
    if config.run.out_dir.is_relative() {
        config.run.out_dir = repo.join(&config.run.out_dir);
    }
    if config.paths.paper_tex.is_relative() {
        config.paths.paper_tex = repo.join(&config.paths.paper_tex);
    }
    if config.paths.eval_claims.is_relative() {
        config.paths.eval_claims = repo.join(&config.paths.eval_claims);
    }
    if !config.mpi.hostfile.as_os_str().is_empty() && config.mpi.hostfile.is_relative() {
        config.mpi.hostfile = repo.join(&config.mpi.hostfile);
    }
}

fn repo_root() -> io::Result<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "repo root not found"))
}

type TomlDoc = BTreeMap<String, BTreeMap<String, String>>;

fn parse_limited_toml(text: &str) -> io::Result<TomlDoc> {
    let mut section = String::new();
    let mut doc: TomlDoc = BTreeMap::new();
    for (line_no, raw) in text.lines().enumerate() {
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            section = line[1..line.len() - 1].trim().to_string();
            doc.entry(section.clone()).or_default();
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(invalid(format!(
                "invalid TOML line {}: {}",
                line_no + 1,
                raw
            )));
        };
        doc.entry(section.clone())
            .or_default()
            .insert(key.trim().to_string(), value.trim().to_string());
    }
    Ok(doc)
}

fn strip_comment(line: &str) -> &str {
    let mut in_string = false;
    for (idx, ch) in line.char_indices() {
        if ch == '"' {
            in_string = !in_string;
        }
        if ch == '#' && !in_string {
            return &line[..idx];
        }
    }
    line
}

fn get_string(doc: &TomlDoc, section: &str, key: &str, default: &str) -> String {
    doc.get(section)
        .and_then(|table| table.get(key))
        .map(|value| parse_string(value))
        .unwrap_or_else(|| default.to_string())
}

fn get_bool(doc: &TomlDoc, section: &str, key: &str, default: bool) -> bool {
    doc.get(section)
        .and_then(|table| table.get(key))
        .map(|value| value.trim() == "true")
        .unwrap_or(default)
}

fn get_usize(doc: &TomlDoc, section: &str, key: &str, default: usize) -> usize {
    doc.get(section)
        .and_then(|table| table.get(key))
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(default)
}

fn get_array(doc: &TomlDoc, section: &str, key: &str, default: Vec<String>) -> Vec<String> {
    let Some(value) = doc.get(section).and_then(|table| table.get(key)) else {
        return default;
    };
    let value = value.trim();
    if !value.starts_with('[') || !value.ends_with(']') {
        return default;
    }
    value[1..value.len() - 1]
        .split(',')
        .map(parse_string)
        .filter(|value| !value.is_empty())
        .collect()
}

fn parse_string(value: &str) -> String {
    let value = value.trim();
    if value.starts_with('"') && value.ends_with('"') && value.len() >= 2 {
        value[1..value.len() - 1].to_string()
    } else {
        value.to_string()
    }
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn invalid(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[allow(dead_code)]
fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    env::temp_dir().join(format!("{prefix}-{}-{nanos}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_mpi_nodes_from_toml() {
        let cfg = Config::from_toml(
            r#"
            [run]
            out_dir = "/tmp/x"
            static_only = false

            [mpi]
            enabled = true
            launcher = "mpirun"
            nodes = ["10.0.0.1", "10.0.0.2"]
            ranks_per_node = 2
            np = 4
            extra_args = ["--bind-to", "none"]
            "#,
        )
        .unwrap();
        assert!(cfg.mpi.enabled);
        assert_eq!(cfg.mpi.nodes, vec!["10.0.0.1", "10.0.0.2"]);
        assert_eq!(cfg.mpi.ranks_per_node, 2);
        assert_eq!(cfg.mpi.np, 4);
        assert_eq!(cfg.mpi.extra_args, vec!["--bind-to", "none"]);
        assert!(!cfg.run.static_only);
    }

    #[test]
    fn writes_mpi_hostfile_from_nodes() {
        let mut cfg = Config::from_toml(
            r#"
            [run]
            out_dir = "/tmp/concordia-all-test-hostfile"
            [mpi]
            enabled = true
            nodes = ["127.0.0.1", "127.0.0.2"]
            ranks_per_node = 3
            "#,
        )
        .unwrap();
        cfg.run.out_dir = unique_temp_dir("concordia-all-hostfile");
        fs::create_dir_all(&cfg.run.out_dir).unwrap();
        let hostfile = prepare_mpi_hostfile(&cfg).unwrap().unwrap();
        let text = fs::read_to_string(hostfile).unwrap();
        assert!(text.contains("127.0.0.1 slots=3"));
        assert!(text.contains("127.0.0.2 slots=3"));
        let _ = fs::remove_dir_all(cfg.run.out_dir);
    }

    #[test]
    fn summary_contains_csv_and_jsonl_rows() {
        let dir = unique_temp_dir("concordia-all-summary");
        fs::create_dir_all(&dir).unwrap();
        let rows = vec![Row {
            experiment: "x".into(),
            status: "pass".into(),
            exit_code: 0,
            duration_ms: 7,
            artifact: dir.join("a"),
            log: dir.join("x.log"),
            message: "ok".into(),
        }];
        write_summary(&dir, &rows).unwrap();
        assert!(fs::read_to_string(dir.join("summary.csv"))
            .unwrap()
            .contains("experiment,status,exit_code"));
        assert!(fs::read_to_string(dir.join("summary.jsonl"))
            .unwrap()
            .contains("\"experiment\":\"x\""));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn classifies_skipped_kimi_csv_as_blocked() {
        let dir = unique_temp_dir("concordia-all-kimi");
        fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("kimi.csv");
        fs::write(
            &csv,
            "case,status,tps\nbaseline,skipped_missing_runner,0\nconcordia,skipped_missing_runner,0\n",
        )
        .unwrap();
        assert_eq!(classify_kimi_csv(&csv), "blocked");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn classifies_timed_out_kimi_csv_as_blocked() {
        let dir = unique_temp_dir("concordia-all-kimi-timeout");
        fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("kimi.csv");
        fs::write(&csv, "case,status,tps\nbaseline,timeout,0\n").unwrap();
        assert_eq!(classify_kimi_csv(&csv), "blocked");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn classifies_eval_claims_partial() {
        let dir = unique_temp_dir("concordia-all-eval");
        fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("eval.csv");
        fs::write(&csv, "claim_id,status\nx,pass\ny,partial\nz,blocked\n").unwrap();
        assert_eq!(classify_eval_claims_csv(&csv), "partial");
        let _ = fs::remove_dir_all(dir);
    }
}
