#!/usr/bin/env python3
"""Fail-closed validator for the fixed Qwen CUDA/handwritten/compiler proof."""

import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path


MODEL_SIZE = 94_155_830_880
MODEL_SHA256 = "0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568"
LLAMA_REVISION = "925e1179947ea0c0ebfb0032df18af3a729822be"
EXPERT_TYPES = {"IQ1_S": 141, "IQ2_XXS": 24, "IQ3_S": 4, "MXFP4": 11}
MODES = ("cuda", "handwritten", "compiler")
HYBRID_MODES = ("handwritten", "compiler")
PROFILES = {
    "one-token": {
        "request_count": 1,
        "max_active": 1,
        "tokens_per_request": 1,
        "measurements": 1,
        "warmups": 0,
    },
    "full": {
        "request_count": 64,
        "max_active": 16,
        "tokens_per_request": 32,
        "measurements": 3,
        "warmups": 1,
    },
}
PROMPT_TOKENS_BY_PROFILE = {"one-token": 16, "full": 256}
REQUEST_COUNT = 64
MAX_ACTIVE = 16
TOKENS_PER_REQUEST = 32
CONTEXT_TOKENS_PER_REQUEST = 512
SERVER_CONTEXT_TOKENS = MAX_ACTIVE * CONTEXT_TOKENS_PER_REQUEST
GENERATED_TOKENS = REQUEST_COUNT * TOKENS_PER_REQUEST
MEASUREMENT_COUNT = 3
MIN_HYBRID_TOKENS_PER_SECOND = 15.0
MODEL_CONTEXT_LIMIT = 262_144
CU_COUNT = 4
MATRIX_TILE_BYTES = 262_144
SHA256_RE = re.compile(r"[0-9a-f]{64}")
EXPECTED_TRACE = [
    ["ldv", "v0", "PARAM_INPUT"],
    ["tmatmul_import", "v0"],
    ["tmatmul_go", "PARAM_MATRIX"],
    ["tmatmul_export", "v0"],
    ["sv", "v0", "PARAM_OUTPUT"],
    ["stall"],
]
TIMING_KEYS = (
    "model_load_ms",
    "prompt_tokens_per_second",
    "ttft_ms",
    "generation_tokens_per_second",
    "single_request_generation_tokens_per_second",
    "end_to_end_ms",
    "queue_ms",
    "service_ms",
    "measured_wall_seconds",
)


class ProofInvalid(ValueError):
    pass


def _fail(message):
    raise ProofInvalid(message)


def _mapping(value, label):
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    return value


def _list(value, label):
    if not isinstance(value, list):
        _fail(f"{label} must be an array")
    return value


def _integer(value, label, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        _fail(f"{label} must be at least {minimum}")
    return value


def _number(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or (positive and value <= 0.0):
        _fail(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _expect(value, expected, label):
    if value != expected:
        _fail(f"{label} must equal {expected!r}")


def _keys(value, keys, label):
    missing = [key for key in keys if key not in value]
    if missing:
        _fail(f"{label} missing keys: {', '.join(missing)}")


def _validate_profile(value):
    profile = _mapping(value, "profile")
    fields = ("name", "request_count", "max_active", "tokens_per_request", "measurements", "warmups")
    _keys(profile, fields, "profile")
    if set(profile) != set(fields):
        _fail("profile contains fields outside the fixed contract")
    name = profile["name"]
    if name not in PROFILES:
        _fail("profile.name is not one-token or full")
    expected = {"name": name, **PROFILES[name]}
    _expect(profile, expected, "profile")
    return dict(profile)


def _sha(value, label, nonzero=False):
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256")
    if nonzero and value == "0" * 64:
        _fail(f"{label} must be nonzero")
    return value


def _read_bytes(path):
    try:
        return Path(path).read_bytes()
    except OSError as error:
        _fail(f"cannot read {path}: {error}")


def _read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        _fail(f"cannot read {path}: {error}")


def _read_json(path):
    try:
        return _mapping(json.loads(_read_text(path)), str(path))
    except ProofInvalid:
        raise
    except json.JSONDecodeError as error:
        _fail(f"cannot parse {path}: {error}")


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _validate_audit(path):
    raw = _read_bytes(path)
    try:
        audit = _mapping(json.loads(raw), "model-tensor-audit.json")
    except ProofInvalid:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        _fail(f"cannot parse {path}: {error}")
    expected = {
        "schema_version": 1,
        "status": "pass",
        "model_sha256": MODEL_SHA256,
        "architecture": "qwen35moe",
        "routed_expert_count": 180,
        "routed_expert_types": EXPERT_TYPES,
        "tq1_0_total": 0,
        "non_expert_iq1s": [],
    }
    for key, value in expected.items():
        _expect(audit.get(key), value, f"model_audit.{key}")
    return audit, _sha256(raw)


def _parse_key_values(path):
    result = {}
    for index, line in enumerate(_read_text(path).splitlines(), 1):
        if not line or "=" not in line:
            _fail(f"{path} line {index} is not key=value")
        key, value = line.split("=", 1)
        if not key or key in result:
            _fail(f"{path} contains an empty or duplicate key")
        result[key] = value
    return result


def _validate_artifacts(root):
    verification = _read_json(root / "model-verification.json")
    _expect(verification.get("path"), "/models/qwen/Qwen3.5-397B-A17B-UD-TQ1_0.gguf", "model_verification.path")
    _expect(verification.get("size"), MODEL_SIZE, "model_verification.size")
    _expect(verification.get("sha256"), MODEL_SHA256, "model_verification.sha256")
    for name in ("device", "inode", "mtime_ns", "ctime_ns"):
        _integer(verification.get(name), f"model_verification.{name}", 1)

    artifacts = _parse_key_values(root / "artifact-hashes.txt")
    required_hashes = (
        "model_sha256", "llama_server_sha256", "libggml_sha256",
        "libnvcuda_sha256", "cuda13_launch_shim_sha256", "xclbin_sha256",
    )
    _keys(artifacts, (*required_hashes, "llama_revision", "build_threads", "threads"), "artifact_hashes")
    for name in required_hashes:
        _sha(artifacts[name], f"artifact_hashes.{name}", nonzero=True)
    _expect(artifacts["model_sha256"], MODEL_SHA256, "artifact_hashes.model_sha256")
    _expect(artifacts["llama_revision"], LLAMA_REVISION, "artifact_hashes.llama_revision")
    try:
        build_threads = int(artifacts["build_threads"])
        runtime_threads = int(artifacts["threads"])
    except ValueError:
        _fail("artifact thread counts must be integers")
    if not 1 <= build_threads <= 32:
        _fail("artifact_hashes.build_threads must be between 1 and 32")
    if runtime_threads < 1:
        _fail("artifact_hashes.threads must be positive")

    preflight = _read_json(root / "qwen-build-preflight.json")
    expected_preflight = {
        "schema_version": 1,
        "build_root": "/qwen-build",
        "libggml_path": "/qwen-build/llama-build/bin/libggml.so.0.22.0",
        "libggml_sha256": artifacts["libggml_sha256"],
        "llama_revision": LLAMA_REVISION,
        "required_symbols": ["dequantize_row_iq1_s", "ggml_init", "ggml_free"],
        "status": "pass",
    }
    _expect(preflight, expected_preflight, "qwen_build_preflight")
    source_hash = _read_text(root / "repository-diff.sha256").strip()
    _sha(source_hash, "repository_diff_sha256", nonzero=True)

    xclbin = _read_text(root / "xclbin-info.txt")
    for name in ("ternip_big_1", "ternip_big_2", "ternip_big_3", "ternip_small_1"):
        if f"Instance:        {name}" not in xclbin:
            _fail(f"xclbin topology is missing {name}")
    pcie = _read_text(root / "pcie-link.txt")
    if re.search(r"LnkSta:.*Speed 8GT/s.*Width x4", pcie) is None:
        _fail("PCIe link evidence must record the observed Gen3 x4 link")
    return {
        "hashes": artifacts,
        "repository_diff_sha256": source_hash,
        "pcie_link": {"observed": "Gen3 x4", "downgrade_risk": True},
    }


def _validate_health(mode, record):
    health = _mapping(record.get("device_health"), f"{mode}.device_health")
    for phase in ("before", "after"):
        item = _mapping(health.get(phase), f"{mode}.device_health.{phase}")
        _expect(item.get("firewall"), "GOOD", f"{mode}.device_health.{phase}.firewall")
        fatal = _list(item.get("fatal_errors"), f"{mode}.device_health.{phase}.fatal_errors")
        if fatal:
            _fail(f"{mode}.device_health.{phase}.fatal_errors must be empty")


def _trace_tokens(assembly):
    result = []
    for raw in assembly.splitlines():
        line = raw.split(";", 1)[0].strip()
        if line:
            result.append([item for item in re.split(r"[\s,]+", line) if item])
    return result


def _semantic_hash(instructions):
    digest = hashlib.sha256(b"hetgpu-tmatmul-semantic-trace-v1\0")
    for instruction in instructions:
        for token in instruction:
            digest.update(token.encode())
            digest.update(b"\0")
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_physical(mode, value):
    completion = _mapping(value, f"{mode}.physical_completion")
    required = (
        "request_id", "cu_index", "stall_code", "dispatch_to_stall_ns",
        "matrix_key_sha256", "matrix_content_sha256", "matrix_address",
        "matrix_cache_hit", "matrix_bytes_transferred", "trace_mode",
        "model_context_limit", "trace_semantic_sha256", "trace_assembly_sha256",
        "replay_safe_program_sha256", "trace_assembly", "trace_instructions",
        "encoded_program_sha256", "encoded_program_hex", "program_address",
        "program_bytes", "program_cache_hit",
    )
    _keys(completion, required, f"{mode}.physical_completion")
    request_id = _integer(completion["request_id"], f"{mode}.physical.request_id", 1)
    cu = _integer(completion["cu_index"], f"{mode}.physical.cu_index", 0)
    if cu >= CU_COUNT:
        _fail(f"{mode}.physical.cu_index is outside four CUs")
    stall = _integer(completion["stall_code"], f"{mode}.physical.stall_code", 1)
    _integer(completion["dispatch_to_stall_ns"], f"{mode}.physical.dispatch_to_stall_ns", 1)
    _sha(completion["matrix_key_sha256"], f"{mode}.physical.matrix_key_sha256", True)
    _sha(completion["matrix_content_sha256"], f"{mode}.physical.matrix_content_sha256", True)
    _integer(completion["matrix_address"], f"{mode}.physical.matrix_address", 1)
    if not isinstance(completion["matrix_cache_hit"], bool):
        _fail(f"{mode}.physical.matrix_cache_hit must be boolean")
    expected_transfer = 0 if completion["matrix_cache_hit"] else MATRIX_TILE_BYTES
    _expect(completion["matrix_bytes_transferred"], expected_transfer, f"{mode}.physical.matrix_bytes_transferred")
    _expect(completion["trace_mode"], mode, f"{mode}.physical.trace_mode")
    _expect(completion["model_context_limit"], MODEL_CONTEXT_LIMIT, f"{mode}.physical.model_context_limit")
    assembly = completion["trace_assembly"]
    instructions = completion["trace_instructions"]
    if not isinstance(assembly, str) or _trace_tokens(assembly) != instructions or instructions != EXPECTED_TRACE:
        _fail(f"{mode}.physical trace body/instructions are not canonical terminal-STALL trace")
    _expect(completion["trace_assembly_sha256"], _sha256(assembly.encode()), f"{mode}.physical.trace_assembly_sha256")
    _expect(completion["trace_semantic_sha256"], _semantic_hash(instructions), f"{mode}.physical.trace_semantic_sha256")
    encoded_hex = completion["encoded_program_hex"]
    try:
        encoded = bytes.fromhex(encoded_hex)
    except (TypeError, ValueError):
        _fail(f"{mode}.physical.encoded_program_hex is invalid")
    program_bytes = _integer(completion["program_bytes"], f"{mode}.physical.program_bytes", 16)
    if program_bytes % 16 or len(encoded) != program_bytes:
        _fail(f"{mode}.physical encoded program length is inconsistent")
    program_sha = _sha256(encoded)
    _expect(completion["encoded_program_sha256"], program_sha, f"{mode}.physical.encoded_program_sha256")
    _expect(completion["replay_safe_program_sha256"], program_sha, f"{mode}.physical.replay_safe_program_sha256")
    _integer(completion["program_address"], f"{mode}.physical.program_address", 1)
    if not isinstance(completion["program_cache_hit"], bool):
        _fail(f"{mode}.physical.program_cache_hit must be boolean")
    return dict(completion)


def _zero_xrt():
    return {
        "submission_count": 0, "completion_count": 0,
        "per_cu_submissions": [0, 0, 0, 0], "per_cu_completions": [0, 0, 0, 0],
        "request_ids": [], "stall_codes": [], "raw_min": 0, "raw_max": 0,
        "reference_checked_components": 0, "operation_count": 0,
        "resident_matrix_hits": 0, "resident_matrix_misses": 0,
        "resident_matrix_bytes_transferred": 0, "program_cache_hits": 0,
        "program_cache_misses": 0, "host_pack_hits": 0, "host_pack_misses": 0,
        "host_pack_bytes_built": 0, "physical_completions": [],
    }


def _validate_xrt(label, value, hybrid, require_reuse):
    xrt = _mapping(value, f"{label}.xrt")
    required = tuple(_zero_xrt())
    _keys(xrt, required, f"{label}.xrt")
    if not hybrid:
        _expect(xrt, _zero_xrt(), f"{label}.xrt")
        return _zero_xrt()
    submissions = _integer(xrt["submission_count"], f"{label}.xrt.submission_count", 1)
    completions = _integer(xrt["completion_count"], f"{label}.xrt.completion_count", 1)
    _expect(completions, submissions, f"{label}.xrt.completion_count")
    per_submit = [_integer(item, f"{label}.xrt.per_cu_submissions", 0) for item in _list(xrt["per_cu_submissions"], f"{label}.xrt.per_cu_submissions")]
    per_complete = [_integer(item, f"{label}.xrt.per_cu_completions", 0) for item in _list(xrt["per_cu_completions"], f"{label}.xrt.per_cu_completions")]
    if len(per_submit) != CU_COUNT or len(per_complete) != CU_COUNT or per_submit != per_complete or sum(per_complete) != completions:
        _fail(f"{label}.xrt per-CU accounting is inconsistent")
    if any(item <= 0 for item in per_complete):
        _fail(f"{label}.xrt must complete work on all four CUs")
    ids = [_integer(item, f"{label}.xrt.request_ids", 1) for item in _list(xrt["request_ids"], f"{label}.xrt.request_ids")]
    stalls = [_integer(item, f"{label}.xrt.stall_codes", 1) for item in _list(xrt["stall_codes"], f"{label}.xrt.stall_codes")]
    if len(ids) != completions or len(set(ids)) != completions or len(stalls) != completions:
        _fail(f"{label}.xrt request/STALL accounting is incomplete")
    raw_min = _integer(xrt["raw_min"], f"{label}.xrt.raw_min")
    raw_max = _integer(xrt["raw_max"], f"{label}.xrt.raw_max")
    if not -4096 <= raw_min <= raw_max <= 4096:
        _fail(f"{label}.xrt raw range is invalid")
    _integer(xrt["reference_checked_components"], f"{label}.xrt.reference_checked_components", 1)
    _integer(xrt["operation_count"], f"{label}.xrt.operation_count", 1)
    physical = [_validate_physical(label.split(".")[0], item) for item in _list(xrt["physical_completions"], f"{label}.xrt.physical_completions")]
    if len(physical) != completions:
        _fail(f"{label}.xrt physical completion count differs")
    if [item["request_id"] for item in physical] != ids or [item["stall_code"] for item in physical] != stalls:
        _fail(f"{label}.xrt physical IDs or STALLs do not bind aggregate evidence")
    physical_per_cu = [sum(item["cu_index"] == cu for item in physical) for cu in range(CU_COUNT)]
    if physical_per_cu != per_complete:
        _fail(f"{label}.xrt physical CU ownership differs from aggregate evidence")
    resident_hits = sum(item["matrix_cache_hit"] for item in physical)
    resident_misses = completions - resident_hits
    program_hits = sum(item["program_cache_hit"] for item in physical)
    program_misses = completions - program_hits
    _expect(xrt["resident_matrix_hits"], resident_hits, f"{label}.xrt.resident_matrix_hits")
    _expect(xrt["resident_matrix_misses"], resident_misses, f"{label}.xrt.resident_matrix_misses")
    _expect(xrt["resident_matrix_bytes_transferred"], resident_misses * MATRIX_TILE_BYTES, f"{label}.xrt.resident_matrix_bytes_transferred")
    _expect(xrt["program_cache_hits"], program_hits, f"{label}.xrt.program_cache_hits")
    _expect(xrt["program_cache_misses"], program_misses, f"{label}.xrt.program_cache_misses")
    host_hits = _integer(xrt["host_pack_hits"], f"{label}.xrt.host_pack_hits", 0)
    host_misses = _integer(xrt["host_pack_misses"], f"{label}.xrt.host_pack_misses", 0)
    _expect(xrt["host_pack_bytes_built"], host_misses * MATRIX_TILE_BYTES, f"{label}.xrt.host_pack_bytes_built")
    if require_reuse and (resident_hits <= 0 or program_hits <= 0 or host_hits <= 0):
        _fail(f"{label}.xrt must prove resident, program, and host cache reuse")
    return {**xrt, "physical_completions": physical}


def _validate_routes(label, value, hybrid):
    routes = _mapping(value, f"{label}.routes")
    _keys(routes, ("eligible", "handled", "fallback", "error", "eligible_kernels"), f"{label}.routes")
    result = {name: _integer(routes[name], f"{label}.routes.{name}", 0) for name in ("eligible", "handled", "fallback", "error")}
    kernels = _list(routes["eligible_kernels"], f"{label}.routes.eligible_kernels")
    if len(kernels) != result["eligible"] or any(
        not isinstance(name, str) or "ggml_type19" not in name.lower()
        or ("mul_mat_q" not in name.lower() and "mul_mat_vec_q" not in name.lower())
        or "stream_k_fixup" in name.lower() for name in kernels
    ):
        _fail(f"{label}.routes eligible kernel set is invalid")
    if hybrid:
        if result["eligible"] <= 0 or result["handled"] != result["eligible"] or result["fallback"] or result["error"]:
            _fail(f"{label} IQ1_S routing is not strict and complete")
    elif any(result.values()) or kernels:
        _fail(f"{label} CUDA route evidence must be zero")
    result["eligible_kernels"] = kernels
    return result


def _validate_ffn(mode, comparison):
    if mode == "cuda":
        _expect(comparison, None, "cuda.sampled_ffn_comparison")
        return None
    comparison = _mapping(comparison, f"{mode}.sampled_ffn_comparison")
    _expect(comparison.get("status"), "pass", f"{mode}.sampled_ffn.status")
    _expect(comparison.get("reference_backend"), "scalar_iq1s", f"{mode}.sampled_ffn.reference_backend")
    _expect(comparison.get("phase"), "pre_timed", f"{mode}.sampled_ffn.phase")
    checked = _integer(comparison.get("checked_elements"), f"{mode}.sampled_ffn.checked_elements", 1)
    atol = _number(comparison.get("atol"), f"{mode}.sampled_ffn.atol")
    rtol = _number(comparison.get("rtol"), f"{mode}.sampled_ffn.rtol")
    _expect(atol, 1e-4, f"{mode}.sampled_ffn.atol")
    _expect(rtol, 1e-3, f"{mode}.sampled_ffn.rtol")
    reference = _list(comparison.get("reference_outputs"), f"{mode}.sampled_ffn.reference_outputs")
    actual = _list(comparison.get("actual_outputs"), f"{mode}.sampled_ffn.actual_outputs")
    if len(reference) != checked or len(actual) != checked:
        _fail(f"{mode}.sampled_ffn vectors are incomplete")
    errors = []
    relatives = []
    for index, (left, right) in enumerate(zip(reference, actual)):
        if (
            isinstance(left, bool)
            or not isinstance(left, (int, float))
            or not math.isfinite(float(left))
            or isinstance(right, bool)
            or not isinstance(right, (int, float))
            or not math.isfinite(float(right))
        ):
            _fail(f"{mode}.sampled_ffn output {index} is invalid")
        reference_raw = float(left)
        right = float(right)
        error = abs(right - reference_raw)
        if error > atol + rtol * abs(reference_raw):
            _fail(f"{mode}.sampled FFN output is outside atol=1e-4, rtol=1e-3")
        errors.append(error)
        relatives.append(error / abs(reference_raw) if reference_raw else error)
    reported_abs = _number(comparison.get("max_absolute_error"), f"{mode}.sampled_ffn.max_absolute_error")
    reported_rel = _number(comparison.get("max_relative_error"), f"{mode}.sampled_ffn.max_relative_error")
    if not math.isclose(reported_abs, max(errors), rel_tol=0.0, abs_tol=1e-7) or not math.isclose(reported_rel, max(relatives), rel_tol=0.0, abs_tol=1e-7):
        _fail(f"{mode}.sampled FFN error summary differs from vectors")
    return comparison


def _validate_measurements(mode, values, profile):
    profile = _validate_profile(profile)
    values = _list(values, f"{mode}.measurements")
    if len(values) != profile["measurements"]:
        _fail(
            f"{mode}.measurements must contain exactly {profile['measurements']} entries"
        )
    timing_keys = tuple(
        key
        for key in TIMING_KEYS
        if profile["name"] == "full" or key != "generation_tokens_per_second"
    )
    columns = {key: [] for key in timing_keys}
    throughput = []
    generated_tokens = profile["request_count"] * profile["tokens_per_request"]
    for index, raw in enumerate(values):
        item = _mapping(raw, f"{mode}.measurements[{index}]")
        _keys(item, (*timing_keys, "request_count", "max_active", "generated_tokens"), f"{mode}.measurements[{index}]")
        for key in timing_keys:
            columns[key].append(_number(item[key], f"{mode}.measurements[{index}].{key}", positive=key != "model_load_ms"))
        _expect(item["request_count"], profile["request_count"], f"{mode}.measurements[{index}].request_count")
        active = _integer(item["max_active"], f"{mode}.measurements[{index}].max_active", 1)
        if active > profile["max_active"]:
            _fail(
                f"{mode}.measurements[{index}].max_active exceeds {profile['max_active']}"
            )
        _expect(item["generated_tokens"], generated_tokens, f"{mode}.measurements[{index}].generated_tokens")
        if profile["name"] == "full":
            computed = generated_tokens / columns["measured_wall_seconds"][-1]
            observed = columns["generation_tokens_per_second"][-1]
            if not math.isclose(computed, observed, rel_tol=1e-9, abs_tol=1e-9):
                _fail(f"{mode}.measurements[{index}] throughput does not equal 2048/wall_seconds")
            if mode in HYBRID_MODES and computed < MIN_HYBRID_TOKENS_PER_SECOND:
                _fail(f"{mode}.measurements[{index}] is below 15 aggregate generated tok/s")
            throughput.append(computed)
    summaries = {}
    for key, column in columns.items():
        mean = statistics.fmean(column)
        summaries[key] = {
            "min": min(column), "max": max(column), "median": statistics.median(column),
            "population_stdev": statistics.pstdev(column),
            "cv": 0.0 if mean == 0.0 else statistics.pstdev(column) / mean,
        }
    return summaries, throughput


def _validate_refill_round(evidence, request_count, max_active, context):
    evidence = _mapping(evidence, context)
    events = _list(evidence.get("admission_events"), f"{context}.admission_events")
    erases = _list(evidence.get("slot_reuse_evidence"), f"{context}.slot_reuse_evidence")
    wall_ms = 1000 * _number(evidence.get("wall_seconds"), f"{context}.wall_seconds", positive=True)
    if len(events) != 2 * request_count or len(erases) != max(0, request_count - max_active):
        _fail(f"{context} missing admission/completion or slot erase evidence")
    live, admissions, completions = {}, {}, {}
    previous_ms = 0.0
    for raw in events:
        event = _mapping(raw, context)
        request_id = _integer(event.get("request_id"), context, 0)
        slot = _integer(event.get("id_slot"), context, 0)
        elapsed = _number(event.get("elapsed_ms"), context)
        if request_id >= request_count or slot != request_id % max_active:
            _fail(f"{context} request/slot identity mismatch")
        if not previous_ms <= elapsed <= wall_ms:
            _fail(f"{context} event timestamp outside ordered measurement window")
        previous_ms = elapsed
        if event.get("event") == "admit":
            if slot in live or request_id in admissions:
                _fail(f"{context} duplicate admission or live slot reuse")
            if request_id >= max_active and request_id - max_active not in completions:
                _fail(f"{context} previous slot generation did not complete")
            admissions[request_id] = elapsed
            live[slot] = request_id
        elif event.get("event") == "complete":
            if live.get(slot) != request_id or request_id in completions:
                _fail(f"{context} completion has no live owner")
            completions[request_id] = elapsed
            del live[slot]
        else:
            _fail(f"{context} failed or unknown admission event")
        if _integer(event.get("active"), context, 0) != len(live) or len(live) > max_active:
            _fail(f"{context} active request accounting mismatch")
    if live or set(completions) != set(range(request_count)):
        _fail(f"{context} incomplete request timeline")
    erased = set()
    for raw in erases:
        item = _mapping(raw, context)
        after = _integer(item.get("after_request_id"), context, 0)
        slot = _integer(item.get("id_slot"), context, 0)
        _integer(item.get("n_erased"), context, 0)
        elapsed = _number(item.get("elapsed_ms"), context)
        if after in erased or after + max_active >= request_count or slot != after % max_active:
            _fail(f"{context} duplicate or wrong slot erase identity")
        if not completions[after] <= elapsed <= admissions[after + max_active]:
            _fail(f"{context} erase did not occur between slot generations")
        erased.add(after)


def _validate_mode(mode, record, audit_sha256):
    record = _mapping(record, mode)
    required = (
        "schema_version", "evidence_kind", "mode", "profile", "model_size", "model_sha256",
        "model_audit_sha256", "llama_revision", "binary_sha256", "placement",
        "prompt_tokens", "prompt_text", "prompt_token_ids", "generated_token_ids",
        "generated_token_ids_by_request", "token_equivalence_measurement",
        "slot_ids_by_request", "slot_erase_evidence", "wave_slot_erase_evidence",
        "semantic", "hardware_probe",
        "sampled_ffn_comparison", "semantic_hardware_gate", "routes", "xrt",
        "gpu_attention_routes", "measurements", "process", "device_health",
        "request_contract", "warmup_count", "request_count", "max_active_requests",
        "generated_tokens_per_request", "context_tokens_per_request",
        "server_context_tokens",
    )
    _keys(record, required, mode)
    profile = _validate_profile(record["profile"])
    scheduler = record.get("scheduler", "waves")
    if scheduler not in ("waves", "slot-refill"):
        _fail(f"{mode} unknown scheduler")
    prompt_tokens = PROMPT_TOKENS_BY_PROFILE[profile["name"]]
    _expect(record["schema_version"], 2, f"{mode}.schema_version")
    _expect(record["evidence_kind"], "iq1s", f"{mode}.evidence_kind")
    _expect(record["mode"], mode, f"{mode}.mode")
    _expect(record["model_size"], MODEL_SIZE, f"{mode}.model_size")
    _expect(record["model_sha256"], MODEL_SHA256, f"{mode}.model_sha256")
    _expect(record["model_audit_sha256"], audit_sha256, f"{mode}.model_audit_sha256")
    _expect(record["llama_revision"], LLAMA_REVISION, f"{mode}.llama_revision")
    binary = _sha(record["binary_sha256"], f"{mode}.binary_sha256", True)
    _expect(record["placement"], {"all_layers_on_gpu": True, "cpu_layers": 0}, f"{mode}.placement")
    _expect(record["prompt_tokens"], prompt_tokens, f"{mode}.prompt_tokens")
    if not isinstance(record["prompt_text"], str) or not record["prompt_text"]:
        _fail(f"{mode}.prompt_text must be nonempty")
    prompt_ids = _list(record["prompt_token_ids"], f"{mode}.prompt_token_ids")
    if len(prompt_ids) != prompt_tokens or any(isinstance(item, bool) or not isinstance(item, int) for item in prompt_ids):
        _fail(f"{mode}.prompt_token_ids must contain {prompt_tokens} integers")
    generated = _list(record["generated_token_ids"], f"{mode}.generated_token_ids")
    if len(generated) != profile["tokens_per_request"]:
        _fail(f"{mode}.generated_token_ids must contain {profile['tokens_per_request']} IDs")
    by_request = _list(record["generated_token_ids_by_request"], f"{mode}.generated_token_ids_by_request")
    if len(by_request) != profile["request_count"]:
        _fail(f"{mode} must retain {profile['request_count']} deterministic request outputs")
    for index, item in enumerate(by_request):
        item = _list(item, f"{mode}.generated_token_ids_by_request[{index}]")
        if len(item) != profile["tokens_per_request"] or any(isinstance(token, bool) or not isinstance(token, int) for token in item):
            _fail(f"{mode}.generated_token_ids_by_request[{index}] must contain {profile['tokens_per_request']} integer IDs")
    _expect(generated, by_request[0], f"{mode}.generated_token_ids request-zero binding")
    _expect(record["token_equivalence_measurement"], 0, f"{mode}.token_equivalence_measurement")
    slot_ids = _list(record["slot_ids_by_request"], f"{mode}.slot_ids_by_request")
    if len(slot_ids) != profile["request_count"]:
        _fail(f"{mode}.slot_ids_by_request must contain {profile['request_count']} assignments")
    for wave_start in range(0, profile["request_count"], profile["max_active"]):
        wave = slot_ids[wave_start:wave_start + profile["max_active"]]
        if sorted(wave) != list(range(len(wave))):
            _fail(f"{mode}.slot_ids_by_request wave {wave_start // profile['max_active']} must be a slot permutation")
    if scheduler == "slot-refill" and slot_ids != [i % profile["max_active"] for i in range(profile["request_count"])]:
        _fail(f"{mode} refill request IDs must retain their recurrent slot")
    slot_erases = _list(record["slot_erase_evidence"], f"{mode}.slot_erase_evidence")
    expected_rounds = profile["warmups"] + profile["measurements"]
    if len(slot_erases) != expected_rounds:
        _fail(f"{mode}.slot_erase_evidence must contain {expected_rounds} pre-batch rounds")
    for round_index, values in enumerate(slot_erases):
        values = _list(values, f"{mode}.slot_erase_evidence[{round_index}]")
        if len(values) != profile["max_active"] or any(_integer(value, f"{mode}.slot_erase_evidence[{round_index}]", 0) < 0 for value in values):
            _fail(f"{mode}.slot_erase_evidence[{round_index}] must bind all {profile['max_active']} slots")
    wave_erases = _list(record["wave_slot_erase_evidence"], f"{mode}.wave_slot_erase_evidence")
    if len(wave_erases) != expected_rounds:
        _fail(f"{mode}.wave_slot_erase_evidence must contain {expected_rounds} batch rounds")
    if scheduler == "slot-refill":
        rounds = _list(record.get("scheduler_evidence"), f"{mode}.scheduler_evidence")
        if len(rounds) != expected_rounds or any(wave_erases):
            _fail(f"{mode} refill must retain every round and no wave barrier")
        for index, evidence in enumerate(rounds):
            _validate_refill_round(evidence, profile["request_count"], profile["max_active"], f"{mode}.refill[{index}]")
            if index >= profile["warmups"]:
                measurement = record["measurements"][index - profile["warmups"]]
                _expect(measurement.get("scheduler"), scheduler, f"{mode}.measurement scheduler")
                _expect(measurement.get("wave_count"), 0, f"{mode}.measurement wave barriers")
                _expect(measurement.get("measured_wall_seconds"), evidence["wall_seconds"], f"{mode}.refill wall binding")
    for batch_index, batch_erases in enumerate(wave_erases):
        batch_erases = _list(batch_erases, f"{mode}.wave_slot_erase_evidence[{batch_index}]")
        wave_count = math.ceil(profile["request_count"] / profile["max_active"])
        if len(batch_erases) != (wave_count - 1 if scheduler == "waves" else 0):
            _fail(f"{mode}.wave_slot_erase_evidence[{batch_index}] must contain {wave_count - 1} wave barriers")
        for wave_index, values in enumerate(batch_erases):
            values = _list(values, f"{mode}.wave_slot_erase_evidence[{batch_index}][{wave_index}]")
            if len(values) != profile["max_active"] or any(_integer(value, f"{mode}.wave_slot_erase_evidence[{batch_index}][{wave_index}]", 0) < 0 for value in values):
                _fail(f"{mode}.wave_slot_erase_evidence[{batch_index}][{wave_index}] must bind all {profile['max_active']} slots")
    _expect(record["request_count"], profile["request_count"], f"{mode}.request_count")
    _expect(record["max_active_requests"], profile["max_active"], f"{mode}.max_active_requests")
    _expect(record["generated_tokens_per_request"], profile["tokens_per_request"], f"{mode}.generated_tokens_per_request")
    _expect(record["context_tokens_per_request"], CONTEXT_TOKENS_PER_REQUEST, f"{mode}.context_tokens_per_request")
    _expect(record["server_context_tokens"], profile["max_active"] * CONTEXT_TOKENS_PER_REQUEST, f"{mode}.server_context_tokens")
    _expect(record["warmup_count"], profile["warmups"], f"{mode}.warmup_count")
    expected_contract = {
        "prompt": prompt_ids, "n_predict": profile["tokens_per_request"], "temperature": 0.0, "seed": 42,
        "ignore_eos": True, "cache_prompt": False, "return_tokens": True, "stream": True,
        "timings_per_token": True, "return_progress": True,
    }
    _expect(record["request_contract"], expected_contract, f"{mode}.request_contract")
    semantic = _mapping(record["semantic"], f"{mode}.semantic")
    _expect(semantic.get("text"), "OK", f"{mode}.semantic.text")
    if len(_list(semantic.get("token_ids"), f"{mode}.semantic.token_ids")) != 1:
        _fail(f"{mode}.semantic.token_ids must contain one ID")
    probe = _mapping(record["hardware_probe"], f"{mode}.hardware_probe")
    _expect(probe.get("n_predict"), 2, f"{mode}.hardware_probe.n_predict")
    if len(_list(probe.get("token_ids"), f"{mode}.hardware_probe.token_ids")) != 2:
        _fail(f"{mode}.hardware_probe.token_ids must contain two IDs")
    process = _mapping(record["process"], f"{mode}.process")
    _expect(process.get("exit_code"), 0, f"{mode}.process.exit_code")
    _validate_health(mode, record)
    hybrid = mode in HYBRID_MODES
    routes = _validate_routes(mode, record["routes"], hybrid)
    xrt = _validate_xrt(mode, record["xrt"], hybrid, require_reuse=hybrid)
    attention = _integer(record["gpu_attention_routes"], f"{mode}.gpu_attention_routes", 0)
    if hybrid and attention <= 0:
        _fail(f"{mode}.gpu_attention_routes must be positive")
    if not hybrid and attention:
        _fail("cuda.gpu_attention_routes must be zero")
    gate = record["semantic_hardware_gate"]
    if not hybrid:
        _expect(gate, None, "cuda.semantic_hardware_gate")
        gate_result = None
    else:
        gate = _mapping(gate, f"{mode}.semantic_hardware_gate")
        gate_routes = _validate_routes(f"{mode}.semantic_hardware_gate", gate.get("routes"), True)
        gate_xrt = _validate_xrt(f"{mode}.semantic_hardware_gate", gate.get("xrt"), True, require_reuse=False)
        _integer(gate.get("gpu_attention_routes"), f"{mode}.semantic_hardware_gate.gpu_attention_routes", 1)
        if routes["handled"] < gate_routes["handled"] or xrt["completion_count"] < gate_xrt["completion_count"]:
            _fail(f"{mode} final evidence does not include its semantic hardware gate")
        gate_result = {"routes": gate_routes, "xrt": gate_xrt}
    comparison = _validate_ffn(mode, record["sampled_ffn_comparison"])
    metrics, throughput = _validate_measurements(mode, record["measurements"], profile)
    return {
        "scheduler": scheduler,
        "binary_sha256": binary, "profile": profile, "prompt_text": record["prompt_text"],
        "prompt_token_ids": prompt_ids, "generated_token_ids": generated,
        "generated_token_ids_by_request": by_request,
        "slot_ids_by_request": slot_ids,
        "slot_erase_evidence": slot_erases,
        "wave_slot_erase_evidence": wave_erases,
        "semantic_token_ids": semantic["token_ids"], "probe_token_ids": probe["token_ids"],
        "routes": routes, "xrt": xrt, "semantic_hardware_gate": gate_result,
        "gpu_attention_routes": attention, "sampled_ffn_comparison": comparison,
        "metrics": metrics, "measured_throughput": throughput,
    }


def validate_proof(proof_root):
    root = Path(proof_root)
    audit, audit_sha = _validate_audit(root / "model-tensor-audit.json")
    artifacts = _validate_artifacts(root)
    modes = {mode: _validate_mode(mode, _read_json(root / f"{mode}.json"), audit_sha) for mode in MODES}
    for mode in MODES:
        _expect(modes[mode]["binary_sha256"], artifacts["hashes"]["llama_server_sha256"], f"{mode}.binary artifact binding")
    reference = modes["cuda"]
    for mode in HYBRID_MODES:
        current = modes[mode]
        for field in ("scheduler", "profile", "prompt_text", "prompt_token_ids", "generated_token_ids", "generated_token_ids_by_request", "slot_erase_evidence", "wave_slot_erase_evidence", "semantic_token_ids", "probe_token_ids"):
            _expect(current[field], reference[field], f"{mode}.{field} CUDA equivalence")
    _expect(modes["handwritten"]["routes"]["eligible_kernels"], modes["compiler"]["routes"]["eligible_kernels"], "hybrid eligible launch set")
    handwritten_semantics = {item["trace_semantic_sha256"] for item in modes["handwritten"]["xrt"]["physical_completions"]}
    compiler_semantics = {item["trace_semantic_sha256"] for item in modes["compiler"]["xrt"]["physical_completions"]}
    _expect(compiler_semantics, handwritten_semantics, "compiler/handwritten semantic trace identities")
    normalized_modes = {}
    for mode in MODES:
        item = modes[mode]
        normalized_modes[mode] = {
            "profile": item["profile"], "measurements": item["profile"]["measurements"], "metrics": item["metrics"],
            "measured_throughput": item["measured_throughput"], "routes": item["routes"],
            "xrt": item["xrt"], "gpu_attention_routes": item["gpu_attention_routes"],
            "sampled_ffn_comparison": item["sampled_ffn_comparison"],
        }
    return {
        "schema_version": 3,
        "status": "pass",
        "model": {
            "size": MODEL_SIZE, "sha256": MODEL_SHA256, "architecture": "qwen35moe",
            "llama_revision": LLAMA_REVISION,
            "binary_sha256": artifacts["hashes"]["llama_server_sha256"],
        },
        "model_audit": audit,
        "model_audit_sha256": audit_sha,
        "artifact_hashes": artifacts["hashes"],
        "repository_diff_sha256": artifacts["repository_diff_sha256"],
        "pcie_link": artifacts["pcie_link"],
        "modes": normalized_modes,
        "token_ids_match": True,
        "sampled_ffn_within_tolerance": True,
        "eligible_route_coverage": 1.0,
        "tensor_eligibility_coverage": EXPERT_TYPES["IQ1_S"] / 180,
        "all_cus_active": all(
            all(value > 0 for value in modes[mode]["xrt"]["per_cu_completions"])
            for mode in HYBRID_MODES
        ),
    }


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    try:
        if len(arguments) != 1:
            _fail("usage: validate_qwen35_iq1s_au250_proof.py PROOF_DIR")
        result = validate_proof(arguments[0])
    except (ProofInvalid, OSError, ValueError, TypeError) as error:
        print(f"QWEN_IQ1S_PROOF_INVALID: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
