use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const PHASE_KIND: &str = "iq1s_persistent_phase";
pub(crate) const LIBGGML_REFERENCE_BACKEND: &str = "libggml_dequantize_row_iq1_s";

#[derive(Debug, Clone, Copy, Serialize)]
pub(crate) struct PhaseComparison {
    pub(crate) sampled: bool,
    pub(crate) reference_backend: Option<&'static str>,
    pub(crate) checked_elements: usize,
    pub(crate) max_abs_error: f32,
    pub(crate) max_rel_error: f32,
    pub(crate) nonfinite: u64,
    pub(crate) status: &'static str,
}

impl PhaseComparison {
    pub(crate) fn sampled_pass(
        reference_backend: &'static str,
        checked_elements: usize,
        max_abs_error: f32,
        max_rel_error: f32,
    ) -> Result<Self, String> {
        let comparison = Self {
            sampled: true,
            reference_backend: Some(reference_backend),
            checked_elements,
            max_abs_error,
            max_rel_error,
            nonfinite: 0,
            status: "pass",
        };
        comparison.validate()?;
        Ok(comparison)
    }

    pub(crate) fn finite_only(checked_elements: usize) -> Result<Self, String> {
        let comparison = Self {
            sampled: false,
            reference_backend: None,
            checked_elements,
            max_abs_error: 0.0,
            max_rel_error: 0.0,
            nonfinite: 0,
            status: "finite_only",
        };
        comparison.validate()?;
        Ok(comparison)
    }

    fn validate(&self) -> Result<(), String> {
        let valid_sampled = self.sampled
            && self.reference_backend == Some(LIBGGML_REFERENCE_BACKEND)
            && self.checked_elements > 0
            && self.status == "pass";
        let valid_finite_only = !self.sampled
            && self.reference_backend.is_none()
            && self.checked_elements > 0
            && self.status == "finite_only"
            && self.max_abs_error == 0.0
            && self.max_rel_error == 0.0;
        if (!valid_sampled && !valid_finite_only)
            || self.nonfinite != 0
            || !self.max_abs_error.is_finite()
            || !self.max_rel_error.is_finite()
            || self.max_abs_error < 0.0
            || self.max_rel_error < 0.0
            || self.max_abs_error > 1.0e-4
            || self.max_rel_error > 1.0e-3
        {
            return Err("persistent IQ1_S comparison violates its fail-closed schema".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub(crate) struct PhaseTimingsUs {
    pub(crate) capture: u64,
    pub(crate) route_dma: u64,
    pub(crate) trace_build_or_cache: u64,
    pub(crate) activation_pack: u64,
    pub(crate) activation_sync: u64,
    pub(crate) ring_publish: u64,
    pub(crate) doorbell: u64,
    pub(crate) device_wait: u64,
    pub(crate) completion_sync: u64,
    pub(crate) result_copy: u64,
    pub(crate) reconstruct: u64,
    pub(crate) compare: u64,
    pub(crate) log: u64,
    pub(crate) phase_wall: u64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct PersistentPhaseRecord {
    pub(crate) schema_version: u32,
    pub(crate) kind: &'static str,
    pub(crate) transaction_id: u64,
    pub(crate) layer_id: u32,
    pub(crate) phase: &'static str,
    pub(crate) trace_mode: String,
    pub(crate) session_generation: u64,
    pub(crate) program_sha256: [String; 4],
    pub(crate) semantic_sha256: String,
    pub(crate) commands_per_cu: [usize; 4],
    pub(crate) completions_per_cu: [usize; 4],
    pub(crate) weight_dma_bytes: u64,
    pub(crate) eligible_direct_routes: u64,
    pub(crate) comparison_sampled: bool,
    pub(crate) reference_backend: Option<&'static str>,
    pub(crate) checked_elements: usize,
    pub(crate) max_abs_error: f32,
    pub(crate) max_rel_error: f32,
    pub(crate) nonfinite: u64,
    pub(crate) comparison_status: &'static str,
    pub(crate) timing_us: PhaseTimingsUs,
}

impl PersistentPhaseRecord {
    pub(crate) fn new(
        transaction_id: u64,
        layer_id: u32,
        phase: &'static str,
        trace_mode: String,
        session_generation: u64,
        program_sha256: [String; 4],
        semantic_sha256: String,
        commands_per_cu: [usize; 4],
        completions_per_cu: [usize; 4],
        weight_dma_bytes: u64,
        comparison: PhaseComparison,
        timing_us: PhaseTimingsUs,
    ) -> Result<Self, String> {
        comparison.validate()?;
        let record = Self {
            schema_version: 2,
            kind: PHASE_KIND,
            transaction_id,
            layer_id,
            phase,
            trace_mode,
            session_generation,
            program_sha256,
            semantic_sha256,
            commands_per_cu,
            completions_per_cu,
            weight_dma_bytes,
            eligible_direct_routes: 0,
            comparison_sampled: comparison.sampled,
            reference_backend: comparison.reference_backend,
            checked_elements: comparison.checked_elements,
            max_abs_error: comparison.max_abs_error,
            max_rel_error: comparison.max_rel_error,
            nonfinite: comparison.nonfinite,
            comparison_status: comparison.status,
            timing_us,
        };
        record.validate()?;
        Ok(record)
    }

    fn validate(&self) -> Result<(), String> {
        let comparison = PhaseComparison {
            sampled: self.comparison_sampled,
            reference_backend: self.reference_backend,
            checked_elements: self.checked_elements,
            max_abs_error: self.max_abs_error,
            max_rel_error: self.max_rel_error,
            nonfinite: self.nonfinite,
            status: self.comparison_status,
        };
        if self.schema_version != 2
            || self.transaction_id == 0
            || self.layer_id >= 60
            || !matches!(self.phase, "A" | "B")
            || !matches!(self.trace_mode.as_str(), "handwritten" | "compiler")
            || self.session_generation == 0
            || self.commands_per_cu.iter().any(|count| *count == 0)
            || self.commands_per_cu != self.completions_per_cu
            || self.weight_dma_bytes != 0
            || self.eligible_direct_routes != 0
            || comparison.validate().is_err()
            || self.program_sha256.iter().any(|hash| !is_sha256(hash))
            || !is_sha256(&self.semantic_sha256)
        {
            return Err("persistent IQ1_S phase proof violates its fail-closed schema".to_string());
        }
        Ok(())
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value != "0".repeat(64)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(crate) fn hex_sha256(value: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in value {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

pub(crate) fn checked_proof_path_from_env() -> Result<PathBuf, String> {
    let path = PathBuf::from(
        std::env::var("HETGPU_QWEN_IQ1S_PROOF_LEDGER")
            .map_err(|_| "HETGPU_QWEN_IQ1S_PROOF_LEDGER is required".to_string())?,
    );
    if path.file_name().and_then(|name| name.to_str()) != Some("phase-ledger.jsonl") {
        return Err("persistent IQ1_S proof ledger must be named phase-ledger.jsonl".to_string());
    }
    let parent = path
        .parent()
        .ok_or("persistent IQ1_S proof ledger has no parent")?
        .canonicalize()
        .map_err(|error| format!("canonicalize persistent proof directory: {error}"))?;
    if !parent.starts_with("/mnt/disk0") {
        return Err("persistent IQ1_S proof ledger must be beneath /mnt/disk0".to_string());
    }
    Ok(parent.join("phase-ledger.jsonl"))
}

pub(crate) struct PersistentProofLedger {
    path: PathBuf,
    file: File,
    records: u64,
}

impl PersistentProofLedger {
    pub(crate) fn create(path: &Path) -> Result<Self, String> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| {
                format!("create persistent proof ledger {}: {error}", path.display())
            })?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            records: 0,
        })
    }

    pub(crate) fn append_phase(&mut self, mut record: PersistentPhaseRecord) -> Result<(), String> {
        record.validate()?;
        record.timing_us.log = 0;
        let log_start = Instant::now();
        let _ = serde_json::to_vec(&record)
            .map_err(|error| format!("serialize persistent phase proof preflight: {error}"))?;
        record.timing_us.log = u64::try_from(log_start.elapsed().as_micros()).unwrap_or(u64::MAX);
        let mut line = serde_json::to_vec(&record)
            .map_err(|error| format!("serialize persistent phase proof: {error}"))?;
        line.push(b'\n');
        self.file.write_all(&line).map_err(|error| {
            format!(
                "write persistent proof ledger {}: {error}",
                self.path.display()
            )
        })?;
        self.records = self
            .records
            .checked_add(1)
            .ok_or("persistent proof record count overflow")?;
        eprintln!(
            "[hetgpu-iq1s-proof] tx={} layer={} phase={} status=pass semantic={}",
            record.transaction_id,
            record.layer_id,
            record.phase,
            &record.semantic_sha256[..12]
        );
        Ok(())
    }

    pub(crate) fn sync_boundary(&mut self) -> Result<(), String> {
        self.file
            .flush()
            .and_then(|()| self.file.sync_data())
            .map_err(|error| {
                format!(
                    "sync persistent proof ledger {}: {error}",
                    self.path.display()
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> PersistentPhaseRecord {
        PersistentPhaseRecord::new(
            17,
            7,
            "A",
            "compiler".to_string(),
            9,
            std::array::from_fn(|index| format!("{:064x}", index + 1)),
            format!("{:064x}", 9),
            [3; 4],
            [3; 4],
            0,
            PhaseComparison::sampled_pass("libggml_dequantize_row_iq1_s", 1024, 2.5e-5, 4.0e-4)
                .unwrap(),
            PhaseTimingsUs::default(),
        )
        .unwrap()
    }

    #[test]
    fn iq1s_persistent_proof_writes_one_bounded_line_and_refuses_reuse() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("phase-ledger.jsonl");
        let mut ledger = PersistentProofLedger::create(&path).unwrap();
        assert!(PersistentProofLedger::create(&path).is_err());
        ledger.append_phase(record()).unwrap();
        ledger.sync_boundary().unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.ends_with(b"\n"));
        assert_eq!(bytes.split(|byte| *byte == b'\n').count(), 2);
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed["kind"], PHASE_KIND);
        assert_eq!(parsed["schema_version"], 2);
        assert_eq!(parsed["commands_per_cu"], serde_json::json!([3, 3, 3, 3]));
        assert_eq!(parsed["comparison_sampled"], true);
        assert_eq!(parsed["reference_backend"], "libggml_dequantize_row_iq1_s");
        assert_eq!(parsed["checked_elements"], 1024);
        assert_eq!(parsed["max_abs_error"], 2.5e-5);
        assert_eq!(parsed["max_rel_error"], 4.0e-4);
        assert!(parsed.get("outputs").is_none());
    }

    #[test]
    fn iq1s_persistent_proof_rejects_weight_dma_or_incomplete_cu() {
        let mut weight_dma = record();
        weight_dma.weight_dma_bytes = 1;
        assert!(weight_dma.validate().is_err());
        let mut incomplete = record();
        incomplete.completions_per_cu[3] = 0;
        assert!(incomplete.validate().is_err());
    }

    #[test]
    fn iq1s_persistent_proof_distinguishes_finite_only_from_sampled_oracle() {
        let finite = PhaseComparison::finite_only(4096).unwrap();
        assert!(!finite.sampled);
        assert_eq!(finite.reference_backend, None);
        assert_eq!(finite.checked_elements, 4096);
        assert_eq!(finite.status, "finite_only");

        assert!(
            PhaseComparison::sampled_pass("libggml_dequantize_row_iq1_s", 0, 0.0, 0.0,).is_err()
        );
        assert!(PhaseComparison::sampled_pass("scalar_iq1s", 1, 0.0, 0.0).is_err());
        assert!(
            PhaseComparison::sampled_pass("libggml_dequantize_row_iq1_s", 1, 1.1e-4, 0.0,).is_err()
        );
    }
}
