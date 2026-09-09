use serde::Deserialize;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

const BUILTIN_GPU_MARKERS: &[&str] = &[
    "attention",
    "attn",
    "flash",
    "softmax",
    "soft_max",
    "rope",
    "kq",
    "qk",
    "qkv",
    "query",
    "key",
    "value",
    "kv_cache",
];

const BUILTIN_FFN_MARKERS: &[&str] = &[
    "ffn",
    "feed_forward",
    "mlp",
    "gate_proj",
    "up_proj",
    "down_proj",
    "expert",
    "moe",
    "mul_mat_vec_q",
    "mul_mat_q",
    "mul_mat_f",
];

static ROUTE_LOG_MUTEX: Mutex<()> = Mutex::new(());
static MANIFEST_LOAD_CACHE: OnceLock<Mutex<HashMap<ManifestCacheKey, ManifestCacheEntry>>> =
    OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ManifestCacheKey {
    path: PathBuf,
    strict: bool,
}

#[derive(Debug, Clone)]
struct ManifestCacheEntry {
    manifest: Option<BitnetRouteManifest>,
    error: Option<String>,
    diagnostic_emitted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BitnetRoute {
    CxlTmatmul,
    GpuNative,
    Fallback,
    Reject,
}

impl BitnetRoute {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::CxlTmatmul => "cxl_tmatmul",
            Self::GpuNative => "gpu",
            Self::Fallback => "fallback",
            Self::Reject => "reject",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TmatmulBackend {
    Cxl,
    Xrt,
}

impl TmatmulBackend {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Cxl => "cxl",
            Self::Xrt => "xrt",
        }
    }
}

pub(crate) fn tmatmul_backend_from_env() -> Result<Option<TmatmulBackend>, String> {
    match std::env::var("HETGPU_TMATMUL_BACKEND")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        None | Some("") => Ok(None),
        Some("cxl" | "ia780i" | "hardware" | "hw" | "fpga") => {
            Ok(Some(TmatmulBackend::Cxl))
        }
        Some("xrt") => Ok(Some(TmatmulBackend::Xrt)),
        Some(value) => Err(format!(
            "unsupported HETGPU_TMATMUL_BACKEND={value:?}; expected cxl, ia780i, hardware, hw, fpga, or xrt"
        )),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RouteHardware {
    pub(crate) backend: Option<TmatmulBackend>,
    pub(crate) enabled: bool,
    pub(crate) hardware_matmul_enabled: bool,
}

impl RouteHardware {
    pub(crate) fn xrt(hardware_matmul_enabled: bool) -> Self {
        Self {
            backend: Some(TmatmulBackend::Xrt),
            enabled: true,
            hardware_matmul_enabled,
        }
    }

    pub(crate) fn cxl(enabled: bool, hardware_matmul_enabled: bool) -> Self {
        Self {
            backend: Some(TmatmulBackend::Cxl),
            enabled,
            hardware_matmul_enabled,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BitnetRouteSource {
    Disabled,
    ExplicitGpuEnv,
    Manifest,
    ExplicitCxlEnv,
    BuiltinGpuMarker,
    BuiltinFfnMarker,
    Default,
}

impl BitnetRouteSource {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::ExplicitGpuEnv => "explicit_gpu_env",
            Self::Manifest => "manifest",
            Self::ExplicitCxlEnv => "explicit_cxl_env",
            Self::BuiltinGpuMarker => "builtin_gpu_marker",
            Self::BuiltinFfnMarker => "builtin_ffn_marker",
            Self::Default => "default",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BitnetRouteDecision {
    pub(crate) kernel: String,
    pub(crate) route: BitnetRoute,
    pub(crate) source: BitnetRouteSource,
    pub(crate) matched: Option<String>,
    pub(crate) reason: Option<String>,
    pub(crate) strict: bool,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct BitnetRouteConfig {
    pub(crate) enabled: bool,
    pub(crate) strict: bool,
    pub(crate) gpu_markers: Vec<String>,
    pub(crate) cxl_markers: Vec<String>,
    pub(crate) manifest: Option<BitnetRouteManifest>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub(crate) struct BitnetRouteManifest {
    pub(crate) version: u32,
    #[serde(default = "default_manifest_route")]
    pub(crate) default: ManifestRouteName,
    #[serde(default)]
    pub(crate) routes: Vec<BitnetRouteManifestEntry>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub(crate) struct BitnetRouteManifestEntry {
    #[serde(rename = "match")]
    pub(crate) match_text: String,
    pub(crate) route: ManifestRouteName,
    pub(crate) reason: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ManifestRouteName {
    #[serde(alias = "cxl")]
    CxlTmatmul,
    #[serde(alias = "gpu_native")]
    Gpu,
    Fallback,
    Reject,
}

impl Default for ManifestRouteName {
    fn default() -> Self {
        Self::Fallback
    }
}

impl ManifestRouteName {
    fn to_route(self) -> BitnetRoute {
        match self {
            Self::CxlTmatmul => BitnetRoute::CxlTmatmul,
            Self::Gpu => BitnetRoute::GpuNative,
            Self::Fallback => BitnetRoute::Fallback,
            Self::Reject => BitnetRoute::Reject,
        }
    }
}

fn default_manifest_route() -> ManifestRouteName {
    ManifestRouteName::Fallback
}

pub(crate) fn load_manifest_file(path: &Path) -> Result<BitnetRouteManifest, String> {
    let text =
        std::fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let manifest: BitnetRouteManifest =
        serde_json::from_str(&text).map_err(|err| format!("parse {}: {err}", path.display()))?;
    if manifest.version != 1 {
        return Err(format!(
            "unsupported manifest version {}, expected 1",
            manifest.version
        ));
    }
    if manifest.default == ManifestRouteName::CxlTmatmul {
        return Err(
            "cxl_tmatmul is not allowed as manifest default; use fallback, gpu, or reject"
                .to_string(),
        );
    }
    Ok(manifest)
}

pub(crate) fn config_from_env() -> BitnetRouteConfig {
    let enabled = enabled_from_env();
    let strict = env_truthy("HETGPU_BITNET_DISAGG_STRICT");
    let mut config = BitnetRouteConfig {
        enabled,
        strict,
        gpu_markers: env_list("HETGPU_BITNET_GPU_KERNELS"),
        cxl_markers: env_list("HETGPU_BITNET_CXL_KERNELS"),
        manifest: None,
    };

    if let Ok(path) = std::env::var("HETGPU_BITNET_ROUTE_MANIFEST") {
        config.manifest = cached_manifest_from_path(path.trim(), strict);
    }
    config
}

fn cached_manifest_from_path(path_text: &str, strict: bool) -> Option<BitnetRouteManifest> {
    if path_text.is_empty() {
        return None;
    }

    let load_path = Path::new(path_text);
    let key = ManifestCacheKey {
        path: normalize_manifest_path(load_path),
        strict,
    };
    let mut cache = manifest_load_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let entry = cache
        .entry(key)
        .or_insert_with(|| ManifestCacheEntry::load(load_path, strict));
    if let Some(err) = entry.error.as_ref() {
        if !entry.diagnostic_emitted {
            eprintln!("[BitNet Disagg] manifest disabled: {err}");
            entry.diagnostic_emitted = true;
        }
    }
    entry.manifest.clone()
}

fn manifest_load_cache() -> &'static Mutex<HashMap<ManifestCacheKey, ManifestCacheEntry>> {
    MANIFEST_LOAD_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn normalize_manifest_path(path: &Path) -> PathBuf {
    path.components().collect()
}

impl ManifestCacheEntry {
    fn load(path: &Path, strict: bool) -> Self {
        match load_manifest_file(path) {
            Ok(manifest) => Self {
                manifest: Some(manifest),
                error: None,
                diagnostic_emitted: false,
            },
            Err(err) => Self {
                manifest: if strict {
                    Some(BitnetRouteManifest {
                        version: 1,
                        default: ManifestRouteName::Reject,
                        routes: Vec::new(),
                    })
                } else {
                    None
                },
                error: Some(err),
                diagnostic_emitted: false,
            },
        }
    }
}

pub(crate) fn enabled_from_env() -> bool {
    env_truthy("HETGPU_BITNET_DISAGGREGATE")
        || env_truthy("HETGPU_BITNET_FFN_CXL")
        || env_truthy("HETGPU_TMATMUL_BITNET_DISAGGREGATE")
}

pub(crate) fn classify_kernel_name(
    kernel_name: &str,
    config: &BitnetRouteConfig,
) -> BitnetRouteDecision {
    let name_lower = normalize_kernel_name(kernel_name);
    if !config.enabled {
        return decision(
            kernel_name,
            BitnetRoute::Fallback,
            BitnetRouteSource::Disabled,
            None,
            None,
            config.strict,
        );
    }

    if let Some(marker) = find_marker(&name_lower, config.gpu_markers.iter().map(String::as_str)) {
        return decision(
            kernel_name,
            BitnetRoute::GpuNative,
            BitnetRouteSource::ExplicitGpuEnv,
            Some(marker),
            None,
            config.strict,
        );
    }

    if let Some(manifest_decision) = classify_manifest(kernel_name, &name_lower, config) {
        return manifest_decision;
    }

    if let Some(marker) = find_marker(&name_lower, config.cxl_markers.iter().map(String::as_str)) {
        return decision(
            kernel_name,
            BitnetRoute::CxlTmatmul,
            BitnetRouteSource::ExplicitCxlEnv,
            Some(marker),
            None,
            config.strict,
        );
    }

    if let Some(marker) = find_marker(&name_lower, BUILTIN_GPU_MARKERS.iter().copied()) {
        return decision(
            kernel_name,
            BitnetRoute::GpuNative,
            BitnetRouteSource::BuiltinGpuMarker,
            Some(marker),
            None,
            config.strict,
        );
    }

    if let Some(marker) = find_marker(&name_lower, BUILTIN_FFN_MARKERS.iter().copied()) {
        return decision(
            kernel_name,
            BitnetRoute::CxlTmatmul,
            BitnetRouteSource::BuiltinFfnMarker,
            Some(marker),
            None,
            config.strict,
        );
    }

    let route = if config.strict {
        BitnetRoute::Reject
    } else {
        BitnetRoute::Fallback
    };
    decision(
        kernel_name,
        route,
        BitnetRouteSource::Default,
        None,
        None,
        config.strict,
    )
}

fn normalize_kernel_name(kernel_name: &str) -> String {
    kernel_name.trim().to_ascii_lowercase()
}

fn find_marker<'a>(name_lower: &str, mut markers: impl Iterator<Item = &'a str>) -> Option<String> {
    markers.find_map(|marker| normalized_marker_match(name_lower, marker))
}

fn normalized_marker_match(name_lower: &str, marker: &str) -> Option<String> {
    let marker = marker.trim();
    if marker.is_empty() {
        return None;
    }

    if marker_is_ascii_lowercase(marker) {
        if name_lower.contains(marker) {
            return Some(marker.to_string());
        }
        return None;
    }

    let marker_lower = marker.to_ascii_lowercase();
    if name_lower.contains(marker_lower.as_str()) {
        Some(marker_lower)
    } else {
        None
    }
}

fn marker_is_ascii_lowercase(marker: &str) -> bool {
    marker.bytes().all(|byte| !byte.is_ascii_uppercase())
}

fn classify_manifest(
    kernel_name: &str,
    name_lower: &str,
    config: &BitnetRouteConfig,
) -> Option<BitnetRouteDecision> {
    let manifest = config.manifest.as_ref()?;
    for entry in &manifest.routes {
        if let Some(marker) = normalized_marker_match(name_lower, entry.match_text.as_str()) {
            return Some(decision(
                kernel_name,
                entry.route.to_route(),
                BitnetRouteSource::Manifest,
                Some(marker),
                entry.reason.clone(),
                config.strict,
            ));
        }
    }

    if manifest.default == ManifestRouteName::CxlTmatmul {
        return Some(decision(
            kernel_name,
            BitnetRoute::Fallback,
            BitnetRouteSource::Manifest,
            None,
            Some("cxl_tmatmul manifest default is not allowed; falling back".to_string()),
            config.strict,
        ));
    }

    if manifest.default != ManifestRouteName::Fallback {
        return Some(decision(
            kernel_name,
            manifest.default.to_route(),
            BitnetRouteSource::Manifest,
            None,
            None,
            config.strict,
        ));
    }
    None
}

pub(crate) fn append_route_log_from_env(
    decision: &BitnetRouteDecision,
    hardware: RouteHardware,
) -> Result<(), String> {
    let Ok(path) = std::env::var("HETGPU_BITNET_ROUTE_LOG") else {
        return Ok(());
    };
    if path.trim().is_empty() {
        return Ok(());
    }
    append_route_log(Path::new(path.trim()), decision, hardware)
}

pub(crate) fn append_route_log(
    path: &Path,
    decision: &BitnetRouteDecision,
    hardware: RouteHardware,
) -> Result<(), String> {
    let cxl_enabled = hardware.backend == Some(TmatmulBackend::Cxl) && hardware.enabled;
    let xrt_enabled = hardware.backend == Some(TmatmulBackend::Xrt) && hardware.enabled;
    let record = serde_json::json!({
        "kernel": decision.kernel.as_str(),
        "route": decision.route.as_str(),
        "source": decision.source.as_str(),
        "matched": decision.matched.as_deref(),
        "reason": decision.reason.as_deref(),
        "strict": decision.strict,
        "backend": hardware.backend.map(TmatmulBackend::as_str),
        "cxl_enabled": cxl_enabled,
        "xrt_enabled": xrt_enabled,
        "hardware_matmul_enabled": hardware.hardware_matmul_enabled,
    });
    let mut line = serde_json::to_string(&record)
        .map_err(|err| format!("serialize route log {}: {err}", path.display()))?;
    line.push('\n');

    let _guard = ROUTE_LOG_MUTEX
        .lock()
        .map_err(|_| format!("lock route log {}: mutex poisoned", path.display()))?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| format!("open route log {}: {err}", path.display()))?;
    file.write_all(line.as_bytes())
        .map_err(|err| format!("write route log {}: {err}", path.display()))
}

fn decision(
    kernel_name: &str,
    route: BitnetRoute,
    source: BitnetRouteSource,
    matched: Option<String>,
    reason: Option<String>,
    strict: bool,
) -> BitnetRouteDecision {
    BitnetRouteDecision {
        kernel: kernel_name.to_string(),
        route,
        source,
        matched,
        reason,
        strict,
    }
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn env_list(name: &str) -> Vec<String> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|item| item.trim().to_ascii_lowercase())
                .filter(|item| !item.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    const BITNET_DISAGG_ENV_VARS: &[&str] = &[
        "HETGPU_BITNET_DISAGGREGATE",
        "HETGPU_BITNET_FFN_CXL",
        "HETGPU_TMATMUL_BITNET_DISAGGREGATE",
        "HETGPU_BITNET_DISAGG_STRICT",
        "HETGPU_BITNET_CXL_KERNELS",
        "HETGPU_BITNET_GPU_KERNELS",
        "HETGPU_BITNET_ROUTE_MANIFEST",
        "HETGPU_BITNET_ROUTE_LOG",
        "HETGPU_TMATMUL_BACKEND",
    ];

    struct EnvGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        previous: Vec<(&'static str, Option<String>)>,
    }

    impl EnvGuard {
        fn set(vars: &[(&'static str, Option<&str>)]) -> Self {
            let lock = super::super::test_env::lock();
            let previous = BITNET_DISAGG_ENV_VARS
                .iter()
                .map(|name| (*name, std::env::var(name).ok()))
                .collect::<Vec<_>>();
            for name in BITNET_DISAGG_ENV_VARS {
                std::env::remove_var(name);
            }
            for (name, value) in vars {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
            Self {
                _lock: lock,
                previous,
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (name, value) in self.previous.drain(..) {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }

    fn config(enabled: bool) -> BitnetRouteConfig {
        BitnetRouteConfig {
            enabled,
            strict: false,
            gpu_markers: Vec::new(),
            cxl_markers: Vec::new(),
            manifest: None,
        }
    }

    #[test]
    fn disabled_router_falls_back() {
        let decision = classify_kernel_name("layer_0_ffn_gate_mul_mat", &config(false));
        assert_eq!(decision.route, BitnetRoute::Fallback);
        assert_eq!(decision.source, BitnetRouteSource::Disabled);
    }

    #[test]
    fn builtin_attention_markers_route_to_gpu() {
        let decision = classify_kernel_name("_z13flash_attn_mul_mat_q", &config(true));
        assert_eq!(decision.route, BitnetRoute::GpuNative);
        assert_eq!(decision.source, BitnetRouteSource::BuiltinGpuMarker);
        assert_eq!(decision.matched.as_deref(), Some("attn"));
    }

    #[test]
    fn builtin_ffn_markers_route_to_cxl() {
        let decision = classify_kernel_name("layer_03_ffn_gate_mul_mat", &config(true));
        assert_eq!(decision.route, BitnetRoute::CxlTmatmul);
        assert_eq!(decision.source, BitnetRouteSource::BuiltinFfnMarker);
        assert_eq!(decision.matched.as_deref(), Some("ffn"));
    }

    #[test]
    fn explicit_gpu_markers_override_explicit_cxl_markers() {
        let mut cfg = config(true);
        cfg.gpu_markers = vec!["force_gpu".to_string()];
        cfg.cxl_markers = vec!["ffn_gate".to_string(), "force_gpu".to_string()];

        let decision = classify_kernel_name("layer_04_force_gpu_ffn_gate_mul_mat", &cfg);
        assert_eq!(decision.route, BitnetRoute::GpuNative);
        assert_eq!(decision.source, BitnetRouteSource::ExplicitGpuEnv);
        assert_eq!(decision.matched.as_deref(), Some("force_gpu"));
    }

    #[test]
    fn explicit_gpu_markers_are_trimmed_and_lowercased() {
        let mut cfg = config(true);
        cfg.gpu_markers = vec![" FORCE_GPU ".to_string()];
        cfg.cxl_markers = vec!["ffn_gate".to_string()];

        let decision = classify_kernel_name("layer_force_gpu_ffn_gate", &cfg);
        assert_eq!(decision.route, BitnetRoute::GpuNative);
        assert_eq!(decision.source, BitnetRouteSource::ExplicitGpuEnv);
        assert_eq!(decision.matched.as_deref(), Some("force_gpu"));
    }

    #[test]
    fn explicit_cxl_markers_override_builtin_gpu_markers_only_after_gpu_env() {
        let mut cfg = config(true);
        cfg.cxl_markers = vec!["attention_ffn_probe".to_string()];

        let decision = classify_kernel_name("attention_ffn_probe_mul_mat", &cfg);
        assert_eq!(decision.route, BitnetRoute::CxlTmatmul);
        assert_eq!(decision.source, BitnetRouteSource::ExplicitCxlEnv);
        assert_eq!(decision.matched.as_deref(), Some("attention_ffn_probe"));
    }

    #[test]
    fn unknown_enabled_kernel_falls_back_by_default() {
        let decision = classify_kernel_name("layer_07_layernorm", &config(true));
        assert_eq!(decision.route, BitnetRoute::Fallback);
        assert_eq!(decision.source, BitnetRouteSource::Default);
    }

    #[test]
    fn unknown_enabled_kernel_rejects_in_strict_mode() {
        let mut cfg = config(true);
        cfg.strict = true;

        let decision = classify_kernel_name("layer_07_layernorm", &cfg);
        assert_eq!(decision.route, BitnetRoute::Reject);
        assert_eq!(decision.source, BitnetRouteSource::Default);
    }

    #[test]
    fn manifest_routes_kernel_by_substring() {
        let manifest = BitnetRouteManifest {
            version: 1,
            default: ManifestRouteName::Fallback,
            routes: vec![BitnetRouteManifestEntry {
                match_text: "ffn_gate".to_string(),
                route: ManifestRouteName::CxlTmatmul,
                reason: Some("manifest ffn".to_string()),
            }],
        };
        let mut cfg = config(true);
        cfg.manifest = Some(manifest);

        let decision = classify_kernel_name("layer_0_ffn_gate_mul_mat", &cfg);
        assert_eq!(decision.route, BitnetRoute::CxlTmatmul);
        assert_eq!(decision.source, BitnetRouteSource::Manifest);
        assert_eq!(decision.matched.as_deref(), Some("ffn_gate"));
        assert_eq!(decision.reason.as_deref(), Some("manifest ffn"));
    }

    #[test]
    fn manifest_default_rejects_when_requested() {
        let manifest = BitnetRouteManifest {
            version: 1,
            default: ManifestRouteName::Reject,
            routes: Vec::new(),
        };
        let mut cfg = config(true);
        cfg.manifest = Some(manifest);

        let decision = classify_kernel_name("layer_0_unknown", &cfg);
        assert_eq!(decision.route, BitnetRoute::Reject);
        assert_eq!(decision.source, BitnetRouteSource::Manifest);
    }

    #[test]
    fn loads_manifest_from_json_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routes.json");
        std::fs::write(
            &path,
            r#"{
            "version": 1,
            "default": "fallback",
            "routes": [
                {"match": "flash_attn", "route": "gpu", "reason": "attention"}
            ]
        }"#,
        )
        .unwrap();

        let manifest = load_manifest_file(&path).unwrap();
        assert_eq!(manifest.routes.len(), 1);
        assert_eq!(manifest.routes[0].match_text, "flash_attn");
        assert_eq!(manifest.routes[0].route, ManifestRouteName::Gpu);
    }

    #[test]
    fn rejects_cxl_manifest_default_from_json() {
        for default in ["cxl_tmatmul", "cxl"] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("routes.json");
            std::fs::write(
                &path,
                format!(r#"{{"version": 1, "default": "{default}", "routes": []}}"#),
            )
            .unwrap();

            let err = load_manifest_file(&path).unwrap_err();
            assert!(
                err.contains("cxl_tmatmul is not allowed as manifest default"),
                "{err}"
            );
        }
    }

    #[test]
    fn allows_manifest_route_aliases_from_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routes.json");
        std::fs::write(
            &path,
            r#"{
            "version": 1,
            "default": "fallback",
            "routes": [
                {"match": "ffn_gate", "route": "cxl", "reason": "alias cxl"},
                {"match": "flash_attn", "route": "gpu_native", "reason": "alias gpu"}
            ]
        }"#,
        )
        .unwrap();

        let manifest = load_manifest_file(&path).unwrap();
        assert_eq!(manifest.routes[0].route, ManifestRouteName::CxlTmatmul);
        assert_eq!(manifest.routes[1].route, ManifestRouteName::Gpu);

        let mut cfg = config(true);
        cfg.manifest = Some(manifest);

        let cxl = classify_kernel_name("layer_0_ffn_gate_mul_mat", &cfg);
        assert_eq!(cxl.route, BitnetRoute::CxlTmatmul);
        assert_eq!(cxl.source, BitnetRouteSource::Manifest);
        assert_eq!(cxl.matched.as_deref(), Some("ffn_gate"));
        assert_eq!(cxl.reason.as_deref(), Some("alias cxl"));

        let gpu = classify_kernel_name("layer_0_flash_attn_mul_mat", &cfg);
        assert_eq!(gpu.route, BitnetRoute::GpuNative);
        assert_eq!(gpu.source, BitnetRouteSource::Manifest);
        assert_eq!(gpu.matched.as_deref(), Some("flash_attn"));
        assert_eq!(gpu.reason.as_deref(), Some("alias gpu"));
    }

    #[test]
    fn omitted_manifest_default_is_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routes.json");
        std::fs::write(&path, r#"{"version": 1, "routes": []}"#).unwrap();

        let manifest = load_manifest_file(&path).unwrap();
        assert_eq!(manifest.default, ManifestRouteName::Fallback);
    }

    #[test]
    fn manual_manifest_default_cxl_does_not_route_unmatched_kernel_to_cxl() {
        let manifest = BitnetRouteManifest {
            version: 1,
            default: ManifestRouteName::CxlTmatmul,
            routes: Vec::new(),
        };
        let mut cfg = config(true);
        cfg.manifest = Some(manifest);

        let decision = classify_kernel_name("layer_0_unknown", &cfg);
        assert_eq!(decision.route, BitnetRoute::Fallback);
        assert_eq!(decision.source, BitnetRouteSource::Manifest);
    }

    #[test]
    fn rejects_manifest_version_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routes.json");
        std::fs::write(
            &path,
            r#"{"version": 2, "default": "fallback", "routes": []}"#,
        )
        .unwrap();

        let err = load_manifest_file(&path).unwrap_err();
        assert!(err.contains("unsupported manifest version"));
    }

    #[test]
    fn config_from_env_accepts_all_enable_flags() {
        let _guard = EnvGuard::set(&[
            ("HETGPU_BITNET_DISAGGREGATE", Some("0")),
            ("HETGPU_BITNET_FFN_CXL", Some("yes")),
            ("HETGPU_TMATMUL_BITNET_DISAGGREGATE", None),
            ("HETGPU_BITNET_DISAGG_STRICT", Some("on")),
            ("HETGPU_BITNET_CXL_KERNELS", Some("ffn_gate, mlp_up")),
            ("HETGPU_BITNET_GPU_KERNELS", Some("rope, softmax")),
            ("HETGPU_BITNET_ROUTE_MANIFEST", None),
        ]);

        let cfg = config_from_env();
        assert!(cfg.enabled);
        assert!(cfg.strict);
        assert_eq!(cfg.cxl_markers, vec!["ffn_gate", "mlp_up"]);
        assert_eq!(cfg.gpu_markers, vec!["rope", "softmax"]);
    }

    #[test]
    fn malformed_manifest_is_ignored_without_strict() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.json");
        std::fs::write(&path, r#"{"version": 9, "routes": []}"#).unwrap();
        let path_text = path.to_string_lossy().to_string();
        let _guard = EnvGuard::set(&[
            ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
            ("HETGPU_BITNET_DISAGG_STRICT", None),
            ("HETGPU_BITNET_ROUTE_MANIFEST", Some(&path_text)),
        ]);

        let cfg = config_from_env();
        assert!(cfg.enabled);
        assert!(cfg.manifest.is_none());
    }

    #[test]
    fn config_from_env_caches_valid_manifest_loads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routes.json");
        std::fs::write(
            &path,
            r#"{
            "version": 1,
            "default": "reject",
            "routes": [
                {"match": "flash_attn", "route": "gpu", "reason": "attention"}
            ]
        }"#,
        )
        .unwrap();
        let path_text = path.to_string_lossy().to_string();
        let _guard = EnvGuard::set(&[
            ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
            ("HETGPU_BITNET_ROUTE_MANIFEST", Some(&path_text)),
        ]);

        let first = config_from_env();
        let first_decision = classify_kernel_name("layer_0_flash_attn_mul_mat", &first);
        assert_eq!(first_decision.route, BitnetRoute::GpuNative);
        assert_eq!(first_decision.source, BitnetRouteSource::Manifest);
        assert_eq!(first_decision.reason.as_deref(), Some("attention"));

        std::fs::remove_file(&path).unwrap();
        let second = config_from_env();
        assert_eq!(second.manifest, first.manifest);
        let second_decision = classify_kernel_name("layer_0_flash_attn_mul_mat", &second);
        assert_eq!(second_decision.route, BitnetRoute::GpuNative);
        assert_eq!(second_decision.source, BitnetRouteSource::Manifest);
        assert_eq!(second_decision.reason.as_deref(), Some("attention"));
    }

    #[test]
    fn malformed_manifest_without_strict_stays_disabled_from_cache() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.json");
        std::fs::write(&path, r#"{"version": 9, "routes": []}"#).unwrap();
        let path_text = path.to_string_lossy().to_string();
        let _guard = EnvGuard::set(&[
            ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
            ("HETGPU_BITNET_ROUTE_MANIFEST", Some(&path_text)),
        ]);

        let first = config_from_env();
        assert!(first.manifest.is_none());

        std::fs::write(
            &path,
            r#"{
            "version": 1,
            "default": "reject",
            "routes": [{"match": "flash_attn", "route": "gpu"}]
        }"#,
        )
        .unwrap();
        let second = config_from_env();
        assert!(second.manifest.is_none());
    }

    #[test]
    fn malformed_manifest_sets_strict_reject_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.json");
        std::fs::write(&path, r#"{"version": 9, "routes": []}"#).unwrap();
        let path_text = path.to_string_lossy().to_string();
        let _guard = EnvGuard::set(&[
            ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
            ("HETGPU_BITNET_DISAGG_STRICT", Some("1")),
            ("HETGPU_BITNET_ROUTE_MANIFEST", Some(&path_text)),
        ]);

        let cfg = config_from_env();
        let decision = classify_kernel_name("layer_unknown", &cfg);
        assert_eq!(decision.route, BitnetRoute::Reject);
    }

    #[test]
    fn strict_malformed_manifest_cache_keeps_reject_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.json");
        std::fs::write(&path, r#"{"version": 9, "routes": []}"#).unwrap();
        let path_text = path.to_string_lossy().to_string();
        let _guard = EnvGuard::set(&[
            ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
            ("HETGPU_BITNET_DISAGG_STRICT", Some("1")),
            ("HETGPU_BITNET_ROUTE_MANIFEST", Some(&path_text)),
        ]);

        let first = config_from_env();
        assert_eq!(
            first.manifest,
            Some(BitnetRouteManifest {
                version: 1,
                default: ManifestRouteName::Reject,
                routes: Vec::new(),
            })
        );

        std::fs::write(
            &path,
            r#"{
            "version": 1,
            "default": "fallback",
            "routes": [{"match": "flash_attn", "route": "gpu"}]
        }"#,
        )
        .unwrap();
        let second = config_from_env();
        assert_eq!(second.manifest, first.manifest);
        let decision = classify_kernel_name("layer_unknown", &second);
        assert_eq!(decision.route, BitnetRoute::Reject);
        assert_eq!(decision.source, BitnetRouteSource::Manifest);
    }

    #[test]
    fn appends_jsonl_route_log() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routes.jsonl");
        let decision = BitnetRouteDecision {
            kernel: "layer_0_ffn_gate_mul_mat".to_string(),
            route: BitnetRoute::CxlTmatmul,
            source: BitnetRouteSource::BuiltinFfnMarker,
            matched: Some("ffn".to_string()),
            reason: Some("FFN ternary matmul candidate".to_string()),
            strict: false,
        };

        append_route_log(&path, &decision, RouteHardware::cxl(true, true)).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let value: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(value["kernel"], "layer_0_ffn_gate_mul_mat");
        assert_eq!(value["route"], "cxl_tmatmul");
        assert_eq!(value["source"], "builtin_ffn_marker");
        assert_eq!(value["matched"], "ffn");
        assert_eq!(value["reason"], "FFN ternary matmul candidate");
        assert_eq!(value["strict"], false);
        assert_eq!(value["cxl_enabled"], true);
        assert_eq!(value["hardware_matmul_enabled"], true);

        let fallback = BitnetRouteDecision {
            kernel: "layer_0_layernorm".to_string(),
            route: BitnetRoute::Fallback,
            source: BitnetRouteSource::Default,
            matched: None,
            reason: None,
            strict: true,
        };
        append_route_log(&path, &fallback, RouteHardware::cxl(false, false)).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let values = text
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(values.len(), 2);
        assert!(values[1]["matched"].is_null());
        assert!(values[1]["reason"].is_null());
        assert_eq!(values[1]["strict"], true);
        assert_eq!(values[1]["hardware_matmul_enabled"], false);
    }

    #[test]
    fn backend_env_selects_xrt_without_enabling_cxl() {
        let _guard = EnvGuard::set(&[("HETGPU_TMATMUL_BACKEND", Some("xrt"))]);
        assert_eq!(
            tmatmul_backend_from_env().unwrap(),
            Some(TmatmulBackend::Xrt)
        );
    }

    #[test]
    fn backend_env_maps_hardware_aliases_to_cxl() {
        for alias in ["ia780i", "hardware", "hw", "fpga"] {
            let _guard = EnvGuard::set(&[("HETGPU_TMATMUL_BACKEND", Some(alias))]);
            assert_eq!(
                tmatmul_backend_from_env().unwrap(),
                Some(TmatmulBackend::Cxl),
                "alias={alias}"
            );
        }
    }

    #[test]
    fn backend_env_rejects_unknown_values() {
        let _guard = EnvGuard::set(&[("HETGPU_TMATMUL_BACKEND", Some("cuda"))]);
        assert!(tmatmul_backend_from_env().unwrap_err().contains("cuda"));
    }

    #[test]
    fn xrt_route_log_preserves_logical_route_and_records_physical_backend() {
        let path = tempfile::NamedTempFile::new().unwrap();
        let decision = classify_kernel_name("ffn_mul_mat_q", &config(true));
        append_route_log(path.path(), &decision, RouteHardware::xrt(true)).unwrap();
        let text = std::fs::read_to_string(path.path()).unwrap();
        let value: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(value["route"], "cxl_tmatmul");
        assert_eq!(value["backend"], "xrt");
        assert_eq!(value["xrt_enabled"], true);
        assert_eq!(value["cxl_enabled"], false);
    }

    #[test]
    fn route_and_source_as_str_values_are_stable() {
        assert_eq!(BitnetRoute::CxlTmatmul.as_str(), "cxl_tmatmul");
        assert_eq!(BitnetRoute::GpuNative.as_str(), "gpu");
        assert_eq!(BitnetRoute::Fallback.as_str(), "fallback");
        assert_eq!(BitnetRoute::Reject.as_str(), "reject");

        assert_eq!(BitnetRouteSource::Disabled.as_str(), "disabled");
        assert_eq!(
            BitnetRouteSource::ExplicitGpuEnv.as_str(),
            "explicit_gpu_env"
        );
        assert_eq!(BitnetRouteSource::Manifest.as_str(), "manifest");
        assert_eq!(
            BitnetRouteSource::ExplicitCxlEnv.as_str(),
            "explicit_cxl_env"
        );
        assert_eq!(
            BitnetRouteSource::BuiltinGpuMarker.as_str(),
            "builtin_gpu_marker"
        );
        assert_eq!(
            BitnetRouteSource::BuiltinFfnMarker.as_str(),
            "builtin_ffn_marker"
        );
        assert_eq!(BitnetRouteSource::Default.as_str(), "default");
    }
}
