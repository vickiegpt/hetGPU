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


def phase(phase_name="A", sampled=True):
    return {
        "schema_version": 2,
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
        "comparison_sampled": sampled,
        "reference_backend": "libggml_dequantize_row_iq1_s" if sampled else None,
        "checked_elements": 1024,
        "max_abs_error": 2.5e-5 if sampled else 0.0,
        "max_rel_error": 4.0e-4 if sampled else 0.0,
        "nonfinite": 0,
        "comparison_status": "pass" if sampled else "finite_only",
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


def validate(root, gate="ledger"):
    return subprocess.run(
        [sys.executable, str(VALIDATOR), "--gate", gate, str(root)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def g3_summary():
    return {
        "schema_version": 1,
        "status": "pass",
        "xclbin_uuid": "b1bafc64-09fd-32b0-a5b4-a881e554ae84",
        "persistent_starts_per_cu": [1, 1, 1, 1],
        "ring_generations_per_cu": [[0, 1], [0, 1], [0, 1], [0, 1]],
        "per_cu_completions": [2, 2, 2, 2],
        "sticky_fault_codes": [0, 0, 0, 0],
        "quiescent_before_shutdown": [1, 1, 1, 1],
        "result_rows_checked": 2048,
        "expected_f32_bits": 1167027804,
        "measured_dma": {
            "command_ranges": 8,
            "activation_ranges": 8,
            "result_ranges": 8,
            "program_ranges": 4,
            "weight_ranges": 0,
            "weight_bytes": 0,
        },
    }


def write_g3_bundle(root, hardware_summary=None):
    root.mkdir()
    qualification = {
        "status": "pass",
        "sha256": "aa" * 32,
        "uuid": "b1bafc64-09fd-32b0-a5b4-a881e554ae84",
        "installed_path": "/au250_xrt/xclbins/qwen397b_iq1s_layer_persistent_aaaaaaaa.xclbin",
        "synthesis": {
            name: {"grid_read_success": True, "log_sha256": "bb" * 32}
            for name in (
                "iq1s_layer_big_1",
                "iq1s_layer_big_2",
                "iq1s_layer_big_3",
                "iq1s_layer_small_1",
            )
        },
    }
    for name, value in (
        ("qualification.json", qualification),
        ("summary.json", hardware_summary if hardware_summary is not None else g3_summary()),
    ):
        (root / name).write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    for name in ("health-before.txt", "health-after.txt"):
        (root / name).write_text("Level 0 : 0x0 (GOOD)\n", encoding="utf-8")
    (root / "xclbin-info.txt").write_text(
        "UUID (xclbin):          b1bafc64-09fd-32b0-a5b4-a881e554ae84\n",
        encoding="utf-8",
    )
    (root / "cargo.log").write_text("test result: ok. 1 passed; 0 failed\n", encoding="utf-8")


def g4_mode(mode):
    return {
        "generated_tokens": 1,
        "token_ids": [151643],
        "routing": {
            "u250_iq1s_tensors": 141,
            "gpu_non_iq1s_tensors": 39,
            "u250_non_iq1s_tensors": 0,
            "attention": "gpu",
        },
        "eligible_direct_routes": 0,
        "fallbacks": 0,
        "measured_weight_dma_bytes": 0,
        "completions_per_cu": [3, 3, 3, 3],
        "e2e_latency_ms": 1234.5 if mode == "handwritten" else 1200.25,
    }


def g4_summary():
    return {
        "schema_version": 1,
        "kind": "iq1s_persistent_g4",
        "profile": "one-token",
        "model_sha256": "0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568",
        "xclbin_sha256": "aa" * 32,
        "xclbin_uuid": "b1bafc64-09fd-32b0-a5b4-a881e554ae84",
        "cuda": {
            "generated_tokens": 1,
            "token_ids": [151643],
            "e2e_latency_ms": 1100.0,
        },
        "handwritten": g4_mode("handwritten"),
        "compiler": g4_mode("compiler"),
    }


def write_g4_bundle(root, aggregate=None):
    root.mkdir()
    data = aggregate if aggregate is not None else g4_summary()
    (root / "g4-summary.json").write_text(
        json.dumps(data, sort_keys=True) + "\n", encoding="utf-8"
    )
    for mode in ("handwritten", "compiler"):
        mode_summary = summary()
        mode_summary["mode"] = mode
        write_bundle(root / f"{mode}-mode", [{**phase(), "trace_mode": mode}], mode_summary)


def test_g4_accepts_exact_three_process_one_token_bundle(tmp_path):
    root = tmp_path / "g4"
    write_g4_bundle(root)
    result = validate(root, "g4")
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["gate"] == "g4"
    assert output["status"] == "pass"
    assert output["token_ids"] == [151643]
    assert output["latency_ms"] == {
        "cuda": 1100.0,
        "handwritten": 1234.5,
        "compiler": 1200.25,
    }
    assert "tps" not in result.stdout.lower()


@pytest.mark.parametrize(
    "mutation",
    (
        lambda data: data["handwritten"].__setitem__("generated_tokens", 2),
        lambda data: data["compiler"].__setitem__("token_ids", [151644]),
        lambda data: data["handwritten"]["routing"].__setitem__("u250_iq1s_tensors", 140),
        lambda data: data["compiler"]["routing"].__setitem__("gpu_non_iq1s_tensors", 38),
        lambda data: data["handwritten"]["routing"].__setitem__("attention", "u250"),
        lambda data: data["compiler"].__setitem__("eligible_direct_routes", 1),
        lambda data: data["handwritten"].__setitem__("fallbacks", 1),
        lambda data: data["compiler"].__setitem__("measured_weight_dma_bytes", 1),
        lambda data: data["handwritten"].__setitem__("completions_per_cu", [3, 3, 3, 0]),
        lambda data: data["cuda"].__setitem__("e2e_latency_ms", float("nan")),
    ),
)
def test_g4_rejects_each_one_token_contract_mutation(tmp_path, mutation):
    root = tmp_path / "g4-bad"
    data = g4_summary()
    mutation(data)
    write_g4_bundle(root, data)
    result = validate(root, "g4")
    assert result.returncode != 0
    assert "tps" not in result.stdout.lower()


def test_g3_accepts_exact_fail_closed_hardware_bundle(tmp_path):
    root = tmp_path / "g3"
    write_g3_bundle(root)
    result = validate(root, "g3")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "gate": "g3",
        "per_cu_completions": [2, 2, 2, 2],
        "result_rows_checked": 2048,
        "status": "pass",
        "xclbin_uuid": "b1bafc64-09fd-32b0-a5b4-a881e554ae84",
    }


@pytest.mark.parametrize(
    "mutation",
    (
        lambda root, data: data.__setitem__("status", "fail"),
        lambda root, data: data.__setitem__("persistent_starts_per_cu", [1, 1, 1, 0]),
        lambda root, data: data.__setitem__("ring_generations_per_cu", [[0, 1]] * 3),
        lambda root, data: data.__setitem__("per_cu_completions", [2, 2, 2, 1]),
        lambda root, data: data.__setitem__("sticky_fault_codes", [0, 0, 0, 1]),
        lambda root, data: data.__setitem__("quiescent_before_shutdown", [1, 1, 1, 0]),
        lambda root, data: data.__setitem__("result_rows_checked", 2047),
        lambda root, data: data["measured_dma"].__setitem__("weight_bytes", 1),
        lambda root, data: data["measured_dma"].__setitem__("program_ranges", 3),
        lambda root, data: (root / "health-after.txt").write_text(
            "Level 0 : 0x1 (TRIPPED)\n", encoding="utf-8"
        ),
    ),
)
def test_g3_rejects_each_hardware_proof_mutation(tmp_path, mutation):
    root = tmp_path / "g3-bad"
    data = g3_summary()
    write_g3_bundle(root, data)
    mutation(root, data)
    (root / "summary.json").write_text(
        json.dumps(data, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = validate(root, "g3")
    assert result.returncode != 0


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


def test_compact_persistent_ledger_accepts_one_sample_and_finite_only_phases(tmp_path):
    first = phase("A", sampled=True)
    second = phase("B", sampled=False)
    root = tmp_path / "sampled-and-finite"
    write_bundle(root, [first, second])
    result = validate(root)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "records",
    (
        [phase("A", sampled=False)],
        [phase("A", sampled=True), {**phase("A", sampled=True), "transaction_id": 18}],
    ),
)
def test_compact_persistent_ledger_requires_exactly_one_libggml_sample(tmp_path, records):
    root = tmp_path / "bad-sample-count"
    write_bundle(root, records)
    result = validate(root)
    assert result.returncode != 0


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
        lambda phases, mode: phases[0].__setitem__("reference_backend", "scalar_iq1s"),
        lambda phases, mode: phases[0].__setitem__("checked_elements", 0),
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
