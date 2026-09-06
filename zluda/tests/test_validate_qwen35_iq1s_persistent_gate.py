#!/usr/bin/env python3
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest


VALIDATOR = Path(__file__).with_name("validate_qwen35_iq1s_persistent_gate.py")
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


def phase(phase_name="A"):
    return {
        "schema_version": 1,
        "kind": "iq1s_persistent_phase",
        "transaction_id": 17,
        "layer_id": 7,
        "phase": phase_name,
        "trace_mode": "handwritten",
        "session_generation": 1,
        "program_sha256": ["11" * 32] * 4,
        "semantic_sha256": "22" * 32,
        "commands_per_cu": [3, 3, 3, 3],
        "completions_per_cu": [3, 3, 3, 3],
        "weight_dma_bytes": 0,
        "eligible_direct_routes": 0,
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
        "nonfinite": 0,
        "comparison_status": "pass",
        "timing_us": {name: 1 for name in TIMING_FIELDS},
    }


def summary():
    return {
        "schema_version": 1,
        "kind": "iq1s_persistent_summary",
        "profile": {
            "name": "one-token",
            "request_count": 1,
            "max_active": 1,
            "tokens_per_request": 1,
            "measurements": 1,
            "warmups": 0,
        },
        "mode": "handwritten",
        "routing": {
            "u250_iq1s_tensors": 141,
            "gpu_non_iq1s_tensors": 39,
            "u250_non_iq1s_tensors": 0,
            "attention": "gpu",
        },
        "fallbacks": 0,
        "token_ids": [151643],
        "cuda_token_ids": [151643],
    }


def write_bundle(root, phases=None, mode_summary=None):
    root.mkdir()
    records = phases if phases is not None else [phase()]
    (root / "phase-ledger.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (root / "summary.json").write_text(
        json.dumps(mode_summary if mode_summary is not None else summary(), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate(root):
    return subprocess.run(
        [sys.executable, str(VALIDATOR), "--gate", "ledger", str(root)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_compact_persistent_ledger_accepts_exact_one_token_bundle(tmp_path):
    root = tmp_path / "valid"
    write_bundle(root)
    result = validate(root)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output == {
        "gate": "ledger",
        "mode": "handwritten",
        "phase_records": 1,
        "status": "pass",
    }
    assert "tps" not in result.stdout.lower()


@pytest.mark.parametrize("field", list(phase().keys()))
def test_compact_persistent_ledger_rejects_every_missing_phase_field(tmp_path, field):
    broken = phase()
    del broken[field]
    root = tmp_path / field
    write_bundle(root, [broken])
    result = validate(root)
    assert result.returncode != 0
    assert "tps" not in result.stdout.lower()


@pytest.mark.parametrize(
    "mutation",
    (
        lambda phases, mode: phases.append(copy.deepcopy(phases[0])),
        lambda phases, mode: phases[0].__setitem__("commands_per_cu", [3, 3, 3, 0]),
        lambda phases, mode: phases[0].__setitem__("completions_per_cu", [3, 3, 3, 0]),
        lambda phases, mode: phases[0].__setitem__("weight_dma_bytes", 1),
        lambda phases, mode: phases[0].__setitem__("eligible_direct_routes", 1),
        lambda phases, mode: phases[0].__setitem__("max_abs_error", 1.1e-4),
        lambda phases, mode: phases[0].__setitem__("max_rel_error", 1.1e-3),
        lambda phases, mode: phases[0].__setitem__("nonfinite", 1),
        lambda phases, mode: phases[0].__setitem__("comparison_status", "fail"),
        lambda phases, mode: mode["routing"].__setitem__("attention", "u250"),
        lambda phases, mode: mode["routing"].__setitem__("u250_iq1s_tensors", 140),
        lambda phases, mode: mode["routing"].__setitem__("gpu_non_iq1s_tensors", 38),
        lambda phases, mode: mode["routing"].__setitem__("u250_non_iq1s_tensors", 1),
        lambda phases, mode: mode.__setitem__("fallbacks", 1),
        lambda phases, mode: mode.__setitem__("token_ids", [151644]),
    ),
)
def test_compact_persistent_gate_rejects_isolated_semantic_mutations(tmp_path, mutation):
    phases = [phase()]
    mode = summary()
    mutation(phases, mode)
    root = tmp_path / "mutated"
    write_bundle(root, phases, mode)
    result = validate(root)
    assert result.returncode != 0
    assert "tps" not in result.stdout.lower()


def test_compact_persistent_gate_rejects_truncated_jsonl(tmp_path):
    root = tmp_path / "truncated"
    write_bundle(root)
    ledger = root / "phase-ledger.jsonl"
    ledger.write_bytes(ledger.read_bytes()[:-1])
    result = validate(root)
    assert result.returncode != 0
    assert "tps" not in result.stdout.lower()
