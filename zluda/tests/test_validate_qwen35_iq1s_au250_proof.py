import importlib.util
import copy
import pytest
from pathlib import Path


VALIDATOR = Path(__file__).with_name("validate_qwen35_iq1s_au250_proof.py")


def load_validator():
    spec = importlib.util.spec_from_file_location("qwen35_iq1s_validator", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def refill_evidence():
    return {
        "wall_seconds": 1.0,
        "admission_events": [
            {"event": "admit", "request_id": 0, "id_slot": 0, "active": 1, "elapsed_ms": 0},
            {"event": "admit", "request_id": 1, "id_slot": 1, "active": 2, "elapsed_ms": 1},
            {"event": "complete", "request_id": 0, "id_slot": 0, "active": 1, "elapsed_ms": 2},
            {"event": "admit", "request_id": 2, "id_slot": 0, "active": 2, "elapsed_ms": 4},
            {"event": "complete", "request_id": 2, "id_slot": 0, "active": 1, "elapsed_ms": 5},
            {"event": "complete", "request_id": 1, "id_slot": 1, "active": 0, "elapsed_ms": 6},
            {"event": "admit", "request_id": 3, "id_slot": 1, "active": 1, "elapsed_ms": 8},
            {"event": "complete", "request_id": 3, "id_slot": 1, "active": 0, "elapsed_ms": 9},
        ],
        "slot_reuse_evidence": [
            {"after_request_id": 0, "id_slot": 0, "n_erased": 2, "elapsed_ms": 3},
            {"after_request_id": 1, "id_slot": 1, "n_erased": 2, "elapsed_ms": 7},
        ],
    }


def test_refill_proof_accepts_overlap_without_a_wave_barrier():
    load_validator()._validate_refill_round(refill_evidence(), 4, 2, "test")


@pytest.mark.parametrize("mutation", ["live_reuse", "active", "missing", "late_erase", "duplicate", "wall"])
def test_refill_proof_rejects_broken_slot_ownership(mutation):
    validator = load_validator()
    evidence = copy.deepcopy(refill_evidence())
    if mutation == "live_reuse":
        evidence["admission_events"][2]["event"] = "admit"
    elif mutation == "active":
        evidence["admission_events"][1]["active"] = 3
    elif mutation == "missing":
        evidence["admission_events"].pop()
    elif mutation == "late_erase":
        evidence["slot_reuse_evidence"][0]["elapsed_ms"] = 5
    elif mutation == "duplicate":
        evidence["slot_reuse_evidence"][1] = evidence["slot_reuse_evidence"][0]
    elif mutation == "wall":
        evidence["wall_seconds"] = 0.001
    with pytest.raises(validator.ProofInvalid):
        validator._validate_refill_round(evidence, 4, 2, "test")


def test_one_token_profile_is_a_fixed_validator_contract():
    validator = load_validator()
    profile = {
        "name": "one-token",
        "request_count": 1,
        "max_active": 1,
        "tokens_per_request": 1,
        "measurements": 1,
        "warmups": 0,
    }

    assert validator._validate_profile(profile) == profile


def test_one_token_measurement_requires_latency_but_not_aggregate_tps():
    validator = load_validator()
    profile = validator._validate_profile({
        "name": "one-token",
        "request_count": 1,
        "max_active": 1,
        "tokens_per_request": 1,
        "measurements": 1,
        "warmups": 0,
    })
    measurement = {
        "model_load_ms": 100.0,
        "prompt_tokens_per_second": 2.0,
        "ttft_ms": 500.0,
        "single_request_generation_tokens_per_second": 1.0,
        "end_to_end_ms": 1000.0,
        "queue_ms": 1.0,
        "service_ms": 999.0,
        "measured_wall_seconds": 1.0,
        "request_count": 1,
        "max_active": 1,
        "generated_tokens": 1,
    }

    summaries, throughput = validator._validate_measurements(
        "handwritten", [measurement], profile
    )

    assert summaries["end_to_end_ms"]["median"] == 1000.0
    assert "generation_tokens_per_second" not in summaries
    assert throughput == []


def test_one_token_record_contract_uses_16_prompt_tokens():
    source = VALIDATOR.read_text(encoding="utf-8")
    assert 'PROMPT_TOKENS_BY_PROFILE = {"one-token": 16, "full": 256}' in source
    assert 'prompt_tokens = PROMPT_TOKENS_BY_PROFILE[profile["name"]]' in source
