#!/usr/bin/env python3
"""Fail-closed validator for compact Qwen IQ1_S persistent phase evidence."""

import argparse
import json
import math
import re
import sys
from pathlib import Path


SHA256 = re.compile(r"[0-9a-f]{64}")
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
TIMING_FIELDS = (
    "capture",
    "route_dma",
    "trace_build_or_cache",
    "activation_pack",
    "activation_sync",
    "ring_publish",
    "doorbell",
    "device_wait",
    "completion_sync",
    "result_copy",
    "reconstruct",
    "compare",
    "log",
    "phase_wall",
)
PHASE_FIELDS = {
    "schema_version",
    "kind",
    "transaction_id",
    "layer_id",
    "phase",
    "trace_mode",
    "session_generation",
    "program_sha256",
    "semantic_sha256",
    "commands_per_cu",
    "completions_per_cu",
    "weight_dma_bytes",
    "eligible_direct_routes",
    "comparison_sampled",
    "reference_backend",
    "checked_elements",
    "max_abs_error",
    "max_rel_error",
    "nonfinite",
    "comparison_status",
    "timing_us",
}
SUMMARY_FIELDS = {
    "schema_version",
    "kind",
    "profile",
    "mode",
    "routing",
    "fallbacks",
    "token_ids",
    "cuda_token_ids",
}
G3_SUMMARY_FIELDS = {
    "schema_version",
    "status",
    "xclbin_uuid",
    "persistent_starts_per_cu",
    "ring_generations_per_cu",
    "per_cu_completions",
    "sticky_fault_codes",
    "quiescent_before_shutdown",
    "result_rows_checked",
    "expected_f32_bits",
    "measured_dma",
}
G3_DMA_FIELDS = {
    "command_ranges",
    "activation_ranges",
    "result_ranges",
    "program_ranges",
    "weight_ranges",
    "weight_bytes",
}
G4_FIELDS = {
    "schema_version",
    "kind",
    "profile",
    "model_sha256",
    "xclbin_sha256",
    "xclbin_uuid",
    "cuda",
    "handwritten",
    "compiler",
}
G4_CUDA_FIELDS = {"generated_tokens", "token_ids", "e2e_latency_ms"}
G4_HYBRID_FIELDS = {
    "generated_tokens",
    "token_ids",
    "routing",
    "eligible_direct_routes",
    "fallbacks",
    "measured_weight_dma_bytes",
    "completions_per_cu",
    "e2e_latency_ms",
}


class ProofInvalid(ValueError):
    pass


def fail(message):
    raise ProofInvalid(message)


def exact_fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != expected:
        fail(f"{label} fields differ from schema")


def integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        fail(f"{label} must be an integer at least {minimum}")
    return value


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        fail(f"{label} must be finite and nonnegative")
    return result


def sha256(value, label):
    if not isinstance(value, str) or SHA256.fullmatch(value) is None or value == "0" * 64:
        fail(f"{label} must be a nonzero lowercase SHA-256")


def four_positive_integers(value, label):
    if not isinstance(value, list) or len(value) != 4:
        fail(f"{label} must contain exactly four CU counts")
    return [integer(item, f"{label}[{index}]", 1) for index, item in enumerate(value)]


def read_json(path, label):
    try:
        raw = path.read_bytes()
    except OSError as error:
        fail(f"cannot read {label}: {error}")
    if not raw.endswith(b"\n"):
        fail(f"{label} is truncated or lacks its final newline")
    try:
        return json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        fail(f"cannot parse {label}: {error}")


def read_ledger(path):
    try:
        raw = path.read_bytes()
    except OSError as error:
        fail(f"cannot read phase ledger: {error}")
    if not raw or not raw.endswith(b"\n"):
        fail("phase ledger is empty, truncated, or lacks its final newline")
    records = []
    for index, line in enumerate(raw.splitlines(), 1):
        if not line:
            fail(f"phase ledger line {index} is empty")
        try:
            records.append(json.loads(line))
        except (UnicodeError, json.JSONDecodeError) as error:
            fail(f"cannot parse phase ledger line {index}: {error}")
    return records


def read_text(path, label):
    try:
        value = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        fail(f"cannot read {label}: {error}")
    if not value.endswith("\n"):
        fail(f"{label} is truncated or lacks its final newline")
    return value


def validate_phase(record, index, mode):
    label = f"phase[{index}]"
    exact_fields(record, PHASE_FIELDS, label)
    if record["schema_version"] != 2 or record["kind"] != "iq1s_persistent_phase":
        fail(f"{label} schema identity is invalid")
    integer(record["transaction_id"], f"{label}.transaction_id", 1)
    layer = integer(record["layer_id"], f"{label}.layer_id")
    if layer >= 60:
        fail(f"{label}.layer_id is outside Qwen")
    if record["phase"] not in ("A", "B"):
        fail(f"{label}.phase must be A or B")
    if record["trace_mode"] != mode:
        fail(f"{label}.trace_mode differs from summary mode")
    integer(record["session_generation"], f"{label}.session_generation", 1)
    programs = record["program_sha256"]
    if not isinstance(programs, list) or len(programs) != 4:
        fail(f"{label}.program_sha256 must contain four hashes")
    for cu, value in enumerate(programs):
        sha256(value, f"{label}.program_sha256[{cu}]")
    sha256(record["semantic_sha256"], f"{label}.semantic_sha256")
    commands = four_positive_integers(record["commands_per_cu"], f"{label}.commands_per_cu")
    completions = four_positive_integers(
        record["completions_per_cu"], f"{label}.completions_per_cu"
    )
    if commands != completions:
        fail(f"{label} command/completion counts differ")
    if integer(record["weight_dma_bytes"], f"{label}.weight_dma_bytes") != 0:
        fail(f"{label} measured weight DMA must be zero")
    if integer(record["eligible_direct_routes"], f"{label}.eligible_direct_routes") != 0:
        fail(f"{label} eligible direct routes must be zero")
    checked_elements = integer(record["checked_elements"], f"{label}.checked_elements", 1)
    max_abs_error = finite_number(record["max_abs_error"], f"{label}.max_abs_error")
    max_rel_error = finite_number(record["max_rel_error"], f"{label}.max_rel_error")
    if max_abs_error > 1.0e-4:
        fail(f"{label} absolute error exceeds tolerance")
    if max_rel_error > 1.0e-3:
        fail(f"{label} relative error exceeds tolerance")
    if integer(record["nonfinite"], f"{label}.nonfinite") != 0:
        fail(f"{label} contains nonfinite outputs")
    sampled = record["comparison_sampled"]
    if not isinstance(sampled, bool):
        fail(f"{label}.comparison_sampled must be boolean")
    if sampled:
        if (
            record["reference_backend"] != "libggml_dequantize_row_iq1_s"
            or record["comparison_status"] != "pass"
        ):
            fail(f"{label} sampled libggml comparison did not pass")
    elif (
        record["reference_backend"] is not None
        or record["comparison_status"] != "finite_only"
        or max_abs_error != 0.0
        or max_rel_error != 0.0
    ):
        fail(f"{label} finite-only validation metadata is invalid")
    _ = checked_elements
    timing = record["timing_us"]
    exact_fields(timing, set(TIMING_FIELDS), f"{label}.timing_us")
    for name in TIMING_FIELDS:
        integer(timing[name], f"{label}.timing_us.{name}")
    return sampled


def validate_summary(summary):
    exact_fields(summary, SUMMARY_FIELDS, "summary")
    if summary["schema_version"] != 1 or summary["kind"] != "iq1s_persistent_summary":
        fail("summary schema identity is invalid")
    mode = summary["mode"]
    if mode not in ("handwritten", "compiler"):
        fail("summary mode must be handwritten or compiler")
    profile = summary["profile"]
    exact_fields(
        profile,
        {"name", "request_count", "max_active", "tokens_per_request", "measurements", "warmups"},
        "summary.profile",
    )
    expected_profiles = {
        "one-token": (1, 1, 1, 1, 0),
        "full": (64, 16, 32, 3, 1),
    }
    name = profile["name"]
    observed = tuple(
        integer(profile[field], f"summary.profile.{field}")
        for field in ("request_count", "max_active", "tokens_per_request", "measurements", "warmups")
    )
    if name not in expected_profiles or observed != expected_profiles[name]:
        fail("summary profile differs from a fixed evaluator profile")
    routing = summary["routing"]
    exact_fields(
        routing,
        {"u250_iq1s_tensors", "gpu_non_iq1s_tensors", "u250_non_iq1s_tensors", "attention"},
        "summary.routing",
    )
    if routing != {
        "u250_iq1s_tensors": 141,
        "gpu_non_iq1s_tensors": 39,
        "u250_non_iq1s_tensors": 0,
        "attention": "gpu",
    }:
        fail("summary routing differs from the fixed GPU/U250 split")
    if integer(summary["fallbacks"], "summary.fallbacks") != 0:
        fail("summary fallbacks must be zero")
    tokens = summary["token_ids"]
    cuda_tokens = summary["cuda_token_ids"]
    for value, label in ((tokens, "token_ids"), (cuda_tokens, "cuda_token_ids")):
        if not isinstance(value, list) or not value:
            fail(f"summary.{label} must be a nonempty array")
        for index, token in enumerate(value):
            integer(token, f"summary.{label}[{index}]")
    if tokens != cuda_tokens:
        fail("summary generated token IDs differ from CUDA")
    if name == "one-token" and len(tokens) != 1:
        fail("one-token profile must contain exactly one generated token ID")
    return mode


def validate_ledger(root):
    summary = read_json(root / "summary.json", "summary.json")
    mode = validate_summary(summary)
    records = read_ledger(root / "phase-ledger.jsonl")
    seen = set()
    transaction_phases = {}
    transaction_layers = {}
    generation = None
    sampled_comparisons = 0
    for index, record in enumerate(records):
        sampled_comparisons += int(validate_phase(record, index, mode))
        identity = (record["transaction_id"], record["phase"])
        if identity in seen:
            fail("phase ledger duplicates a transaction/phase")
        seen.add(identity)
        phases = transaction_phases.setdefault(record["transaction_id"], [])
        phases.append(record["phase"])
        if phases not in (["A"], ["A", "B"]):
            fail("phase ledger transaction order is not A followed by optional B")
        previous_layer = transaction_layers.setdefault(record["transaction_id"], record["layer_id"])
        if previous_layer != record["layer_id"]:
            fail("phase ledger transaction spans multiple layers")
        if generation is None:
            generation = record["session_generation"]
        elif record["session_generation"] != generation:
            fail("phase ledger spans multiple session generations")
    if sampled_comparisons != 1:
        fail("phase ledger must contain exactly one sampled libggml comparison")
    return {
        "gate": "ledger",
        "mode": mode,
        "phase_records": len(records),
        "status": "pass",
    }


def validate_g3(root):
    qualification = read_json(root / "qualification.json", "qualification.json")
    required_qualification = {"status", "sha256", "uuid", "installed_path", "synthesis"}
    if not isinstance(qualification, dict) or not required_qualification.issubset(qualification):
        fail("qualification.json lacks mandatory fields")
    if qualification["status"] != "pass":
        fail("xclbin qualification did not pass")
    sha256(qualification["sha256"], "qualification.sha256")
    image_uuid = qualification["uuid"]
    if not isinstance(image_uuid, str) or UUID.fullmatch(image_uuid) is None:
        fail("qualification.uuid is invalid")
    installed = qualification["installed_path"]
    expected_name = f"qwen397b_iq1s_layer_persistent_{qualification['sha256'][:8]}.xclbin"
    if not isinstance(installed, str) or Path(installed).name != expected_name:
        fail("qualified install name does not contain the image SHA prefix")
    synthesis = qualification["synthesis"]
    expected_instances = {
        "iq1s_layer_big_1",
        "iq1s_layer_big_2",
        "iq1s_layer_big_3",
        "iq1s_layer_small_1",
    }
    if not isinstance(synthesis, dict) or set(synthesis) != expected_instances:
        fail("qualification synthesis set differs from four CUs")
    for instance, evidence in synthesis.items():
        exact_fields(evidence, {"grid_read_success", "log_sha256"}, f"synthesis.{instance}")
        if evidence["grid_read_success"] is not True:
            fail(f"synthesis.{instance} did not read the IQ1_S grid")
        sha256(evidence["log_sha256"], f"synthesis.{instance}.log_sha256")

    summary = read_json(root / "summary.json", "summary.json")
    exact_fields(summary, G3_SUMMARY_FIELDS, "G3 summary")
    if summary["schema_version"] != 1 or summary["status"] != "pass":
        fail("G3 summary did not pass")
    if summary["xclbin_uuid"] != image_uuid:
        fail("G3 summary UUID differs from qualified xclbin")
    expected_vectors = {
        "persistent_starts_per_cu": [1, 1, 1, 1],
        "ring_generations_per_cu": [[0, 1], [0, 1], [0, 1], [0, 1]],
        "per_cu_completions": [2, 2, 2, 2],
        "sticky_fault_codes": [0, 0, 0, 0],
        "quiescent_before_shutdown": [1, 1, 1, 1],
    }
    for field, expected in expected_vectors.items():
        if summary[field] != expected:
            fail(f"G3 summary {field} differs from the strict contract")
    if integer(summary["result_rows_checked"], "G3 result_rows_checked") != 2048:
        fail("G3 must check exactly 2048 result rows")
    integer(summary["expected_f32_bits"], "G3 expected_f32_bits", 1)
    dma = summary["measured_dma"]
    exact_fields(dma, G3_DMA_FIELDS, "G3 measured_dma")
    expected_dma = {
        "command_ranges": 8,
        "activation_ranges": 8,
        "result_ranges": 8,
        "program_ranges": 4,
        "weight_ranges": 0,
        "weight_bytes": 0,
    }
    if dma != expected_dma:
        fail("G3 measured DMA differs from the strict resident-weight contract")

    for filename in ("health-before.txt", "health-after.txt"):
        health = read_text(root / filename, filename)
        if "Level 0 : 0x0 (GOOD)" not in health or re.search(r"\bfatal\b", health, re.I):
            fail(f"{filename} is not healthy")
    xclbin_info = read_text(root / "xclbin-info.txt", "xclbin-info.txt")
    if f"UUID (xclbin):          {image_uuid}" not in xclbin_info:
        fail("xclbin-info UUID differs from qualified xclbin")
    cargo_log = read_text(root / "cargo.log", "cargo.log")
    if "test result: ok." not in cargo_log:
        fail("cargo hardware smoke did not pass")
    return {
        "gate": "g3",
        "per_cu_completions": summary["per_cu_completions"],
        "result_rows_checked": summary["result_rows_checked"],
        "status": "pass",
        "xclbin_uuid": image_uuid,
    }


def one_token_ids(value, label):
    if not isinstance(value, list) or len(value) != 1:
        fail(f"{label} must contain exactly one token ID")
    integer(value[0], f"{label}[0]")
    return value


def validate_g4(root):
    aggregate = read_json(root / "g4-summary.json", "g4-summary.json")
    exact_fields(aggregate, G4_FIELDS, "G4 summary")
    if (
        aggregate["schema_version"] != 1
        or aggregate["kind"] != "iq1s_persistent_g4"
        or aggregate["profile"] != "one-token"
    ):
        fail("G4 summary schema or profile is invalid")
    if aggregate["model_sha256"] != "0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568":
        fail("G4 model SHA-256 differs from the pinned Qwen model")
    sha256(aggregate["xclbin_sha256"], "G4 xclbin_sha256")
    if not isinstance(aggregate["xclbin_uuid"], str) or UUID.fullmatch(aggregate["xclbin_uuid"]) is None:
        fail("G4 xclbin_uuid is invalid")

    cuda = aggregate["cuda"]
    exact_fields(cuda, G4_CUDA_FIELDS, "G4 CUDA")
    if integer(cuda["generated_tokens"], "G4 CUDA generated_tokens") != 1:
        fail("G4 CUDA must generate exactly one token")
    cuda_tokens = one_token_ids(cuda["token_ids"], "G4 CUDA token_ids")
    cuda_latency = finite_number(cuda["e2e_latency_ms"], "G4 CUDA e2e_latency_ms")
    if cuda_latency <= 0.0:
        fail("G4 CUDA latency must be positive")

    latencies = {"cuda": cuda_latency}
    for mode_name in ("handwritten", "compiler"):
        mode = aggregate[mode_name]
        exact_fields(mode, G4_HYBRID_FIELDS, f"G4 {mode_name}")
        if integer(mode["generated_tokens"], f"G4 {mode_name} generated_tokens") != 1:
            fail(f"G4 {mode_name} must generate exactly one token")
        if one_token_ids(mode["token_ids"], f"G4 {mode_name} token_ids") != cuda_tokens:
            fail(f"G4 {mode_name} token IDs differ from CUDA")
        routing = mode["routing"]
        exact_fields(
            routing,
            {"u250_iq1s_tensors", "gpu_non_iq1s_tensors", "u250_non_iq1s_tensors", "attention"},
            f"G4 {mode_name} routing",
        )
        if routing != {
            "u250_iq1s_tensors": 141,
            "gpu_non_iq1s_tensors": 39,
            "u250_non_iq1s_tensors": 0,
            "attention": "gpu",
        }:
            fail(f"G4 {mode_name} routing differs from the fixed split")
        if integer(mode["eligible_direct_routes"], f"G4 {mode_name} eligible_direct_routes") != 0:
            fail(f"G4 {mode_name} contains an eligible direct route")
        if integer(mode["fallbacks"], f"G4 {mode_name} fallbacks") != 0:
            fail(f"G4 {mode_name} contains a fallback")
        if integer(mode["measured_weight_dma_bytes"], f"G4 {mode_name} measured_weight_dma_bytes") != 0:
            fail(f"G4 {mode_name} transferred weights during measurement")
        four_positive_integers(mode["completions_per_cu"], f"G4 {mode_name} completions_per_cu")
        latency = finite_number(mode["e2e_latency_ms"], f"G4 {mode_name} e2e_latency_ms")
        if latency <= 0.0:
            fail(f"G4 {mode_name} latency must be positive")
        latencies[mode_name] = latency

        mode_root = root / f"{mode_name}-mode"
        ledger_result = validate_ledger(mode_root)
        if ledger_result["mode"] != mode_name:
            fail(f"G4 {mode_name} ledger mode differs from its directory")
        phase_records = read_ledger(mode_root / "phase-ledger.jsonl")
        phase_a_layers = {record["layer_id"] for record in phase_records if record["phase"] == "A"}
        phase_b_layers = {record["layer_id"] for record in phase_records if record["phase"] == "B"}
        if len(phase_a_layers) != 50 or len(phase_b_layers) != 41 or not phase_b_layers < phase_a_layers:
            fail(f"G4 {mode_name} phase layers differ from the audited 50/50/41 manifest")
        per_transaction = {}
        per_layer_a = {layer: 0 for layer in phase_a_layers}
        for record in phase_records:
            transaction = per_transaction.setdefault(
                record["transaction_id"], {"layer": record["layer_id"], "phases": []}
            )
            if transaction["layer"] != record["layer_id"]:
                fail(f"G4 {mode_name} transaction spans multiple layers")
            transaction["phases"].append(record["phase"])
            if record["phase"] == "A":
                per_layer_a[record["layer_id"]] += 1
        if len(set(per_layer_a.values())) != 1:
            fail(f"G4 {mode_name} did not traverse every Phase A layer equally")
        for transaction in per_transaction.values():
            expected = ["A", "B"] if transaction["layer"] in phase_b_layers else ["A"]
            if transaction["phases"] != expected:
                fail(f"G4 {mode_name} transaction phases differ from its layer manifest")
        ledger_completions = [
            sum(record["completions_per_cu"][cu] for record in phase_records)
            for cu in range(4)
        ]
        if ledger_completions != mode["completions_per_cu"]:
            fail(f"G4 {mode_name} completion counts differ from its phase ledger")
        if sum(record["weight_dma_bytes"] for record in phase_records) != mode["measured_weight_dma_bytes"]:
            fail(f"G4 {mode_name} weight DMA differs from its phase ledger")
        mode_summary = read_json(mode_root / "summary.json", f"{mode_name} summary.json")
        if mode_summary["token_ids"] != mode["token_ids"]:
            fail(f"G4 {mode_name} aggregate tokens differ from its ledger summary")

    return {
        "gate": "g4",
        "latency_ms": latencies,
        "status": "pass",
        "token_ids": cuda_tokens,
        "xclbin_uuid": aggregate["xclbin_uuid"],
    }


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--gate", choices=("ledger", "g3", "g4", "g5"), required=True)
    result.add_argument("root")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        root = Path(args.root)
        if args.gate == "ledger":
            result = validate_ledger(root)
        elif args.gate == "g3":
            result = validate_g3(root)
        elif args.gate == "g4":
            result = validate_g4(root)
        else:
            fail(f"gate {args.gate} is not implemented by this build")
    except (ProofInvalid, OSError, TypeError, ValueError) as error:
        print(f"QWEN_IQ1S_PERSISTENT_PROOF_INVALID: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
