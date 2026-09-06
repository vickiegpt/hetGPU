#!/usr/bin/env python3
"""Fail-closed validator for compact Qwen IQ1_S persistent phase evidence."""

import argparse
import json
import math
import re
import sys
from pathlib import Path


SHA256 = re.compile(r"[0-9a-f]{64}")
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


def validate_phase(record, index, mode):
    label = f"phase[{index}]"
    exact_fields(record, PHASE_FIELDS, label)
    if record["schema_version"] != 1 or record["kind"] != "iq1s_persistent_phase":
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
    if finite_number(record["max_abs_error"], f"{label}.max_abs_error") > 1.0e-4:
        fail(f"{label} absolute error exceeds tolerance")
    if finite_number(record["max_rel_error"], f"{label}.max_rel_error") > 1.0e-3:
        fail(f"{label} relative error exceeds tolerance")
    if integer(record["nonfinite"], f"{label}.nonfinite") != 0:
        fail(f"{label} contains nonfinite outputs")
    if record["comparison_status"] != "pass":
        fail(f"{label} comparison did not pass")
    timing = record["timing_us"]
    exact_fields(timing, set(TIMING_FIELDS), f"{label}.timing_us")
    for name in TIMING_FIELDS:
        integer(timing[name], f"{label}.timing_us.{name}")


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
    generation = None
    for index, record in enumerate(records):
        validate_phase(record, index, mode)
        identity = (record["transaction_id"], record["phase"])
        if identity in seen:
            fail("phase ledger duplicates a transaction/phase")
        seen.add(identity)
        phases = transaction_phases.setdefault(record["transaction_id"], [])
        phases.append(record["phase"])
        if phases not in (["A"], ["A", "B"]):
            fail("phase ledger transaction order is not A followed by optional B")
        if generation is None:
            generation = record["session_generation"]
        elif record["session_generation"] != generation:
            fail("phase ledger spans multiple session generations")
    return {
        "gate": "ledger",
        "mode": mode,
        "phase_records": len(records),
        "status": "pass",
    }


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--gate", choices=("ledger", "g3", "g4", "g5"), required=True)
    result.add_argument("root")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.gate != "ledger":
            fail(f"gate {args.gate} is not implemented by this build")
        result = validate_ledger(Path(args.root))
    except (ProofInvalid, OSError, TypeError, ValueError) as error:
        print(f"QWEN_IQ1S_PERSISTENT_PROOF_INVALID: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
