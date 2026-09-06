import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "qualify_qwen397b_persistent_xclbin.py"
CANDIDATE_SHA256 = "9c83dcae07b4c7bf1d2e1cebf46ccf0ff1ebf8848a437035fef1451dee7770a3"
CANDIDATE_UUID = "b1bafc64-09fd-32b0-a5b4-a881e554ae84"

VALID_INFO = """
Kernels:                iq1s_layer_small, iq1s_layer_big
UUID (xclbin):          b1bafc64-09fd-32b0-a5b4-a881e554ae84
Kernel: iq1s_layer_small
   Signature: iq1s_layer_small (void* command_ring, void* completion_ring, void* program, void* arena_manifest, void* activation_slab, void* result_slab)
Instance:        iq1s_layer_small_1
   Memory:            bank1 (MEM_DDR4)
Kernel: iq1s_layer_big
   Signature: iq1s_layer_big (void* command_ring, void* completion_ring, void* program, void* arena_manifest, void* activation_slab, void* result_slab)
Instance:        iq1s_layer_big_1
   Memory:            bank0 (MEM_DDR4)
Instance:        iq1s_layer_big_2
   Memory:            bank3 (MEM_DRAM)
Instance:        iq1s_layer_big_3
   Memory:            bank2 (MEM_DDR4)
"""

VALID_TIMING = """
Design Timing Summary
WNS(ns) TNS(ns) TNS Failing Endpoints TNS Total Endpoints WHS(ns) THS(ns)
0.023 0.000 0 933669 0.010 0.000 0 928800 0.000 0.000 0 370164
All user specified timing constraints are met.
"""

VALID_ROUTE = """
# of unrouted nets = 0
# of nets with routing errors = 0
# of nets with overlaps = 0
"""


def load_tool():
    spec = importlib.util.spec_from_file_location("qwen_xclbin_qualifier", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_candidate(path: Path) -> str:
    path.write_bytes(b"persistent-xclbin-fixture")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance():
    return {
        "rtl_head": "14a2e5e583f4823d3dbcf7d86c429514cecf14b8",
        "tracked_diff_sha256": "11" * 32,
        "untracked_sources_sha256": "22" * 32,
        "build_log_sha256": "33" * 32,
        "timing_report_sha256": "44" * 32,
        "route_report_sha256": "55" * 32,
        "build_exit_code": 0,
    }


def test_qualifier_module_exists():
    assert TOOL.is_file(), f"missing qualifier: {TOOL}"


def test_accepts_exact_four_cu_timing_and_provenance(tmp_path):
    tool = load_tool()
    candidate = tmp_path / "candidate.xclbin"
    digest = write_candidate(candidate)
    record = tool.qualify_from_text(
        candidate=candidate,
        info_text=VALID_INFO,
        timing_text=VALID_TIMING,
        route_text=VALID_ROUTE,
        expected_sha256=digest,
        expected_uuid=CANDIDATE_UUID,
        provenance=provenance(),
    )
    assert record["status"] == "pass"
    assert record["sha256"] == digest
    assert record["uuid"] == CANDIDATE_UUID
    assert record["cu_banks"] == {
        "iq1s_layer_big_1": "bank0",
        "iq1s_layer_big_2": "bank3",
        "iq1s_layer_big_3": "bank2",
        "iq1s_layer_small_1": "bank1",
    }
    assert record["arguments"] == [
        "command_ring",
        "completion_ring",
        "program",
        "arena_manifest",
        "activation_slab",
        "result_slab",
    ]
    assert record["timing"] == {
        "wns_ns": 0.023,
        "tns_ns": 0.0,
        "whs_ns": 0.01,
        "ths_ns": 0.0,
    }
    assert record["source_provenance_sha256"] != "0" * 64


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("expected_sha256", "0" * 64, "SHA-256"),
        ("expected_uuid", "00000000-0000-0000-0000-000000000000", "UUID"),
        ("info_text", VALID_INFO.replace("iq1s_layer_big_3", "iq1s_layer_big_4"), "compute unit"),
        ("info_text", VALID_INFO.replace("bank3", "bank2"), "bank"),
        ("info_text", VALID_INFO.replace("void* result_slab", "void* spare"), "signature"),
        ("timing_text", VALID_TIMING.replace("0.023 0.000", "-0.023 -1.000"), "timing"),
        ("timing_text", VALID_TIMING.replace("0.010 0.000", "-0.010 -1.000"), "timing"),
        ("timing_text", VALID_TIMING.replace("All user specified timing constraints are met.", "Timing failed."), "constraints"),
        ("route_text", VALID_ROUTE.replace("unrouted nets = 0", "unrouted nets = 1"), "routing"),
        ("route_text", VALID_ROUTE.replace("overlaps = 0", "overlaps = 1"), "routing"),
    ],
)
def test_rejects_one_static_contract_mutation(tmp_path, field, value, message):
    tool = load_tool()
    candidate = tmp_path / "candidate.xclbin"
    digest = write_candidate(candidate)
    arguments = {
        "candidate": candidate,
        "info_text": VALID_INFO,
        "timing_text": VALID_TIMING,
        "route_text": VALID_ROUTE,
        "expected_sha256": digest,
        "expected_uuid": CANDIDATE_UUID,
        "provenance": provenance(),
    }
    arguments[field] = value
    with pytest.raises(tool.QualificationError, match=message):
        tool.qualify_from_text(**arguments)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"rtl_head": ""}, "provenance"),
        ({"tracked_diff_sha256": "0" * 64}, "provenance"),
        ({"untracked_sources_sha256": "0" * 64}, "provenance"),
        ({"build_log_sha256": "0" * 64}, "provenance"),
        ({"build_exit_code": 1}, "build exit"),
    ],
)
def test_rejects_incomplete_source_provenance(tmp_path, mutation, message):
    tool = load_tool()
    candidate = tmp_path / "candidate.xclbin"
    digest = write_candidate(candidate)
    source = provenance()
    source.update(mutation)
    with pytest.raises(tool.QualificationError, match=message):
        tool.qualify_from_text(
            candidate=candidate,
            info_text=VALID_INFO,
            timing_text=VALID_TIMING,
            route_text=VALID_ROUTE,
            expected_sha256=digest,
            expected_uuid=CANDIDATE_UUID,
            provenance=source,
        )


def test_atomic_install_preserves_existing_destination_on_hash_failure(tmp_path):
    tool = load_tool()
    source = tmp_path / "source.xclbin"
    source.write_bytes(b"new")
    destination = tmp_path / tool.INSTALL_NAME
    destination.write_bytes(b"existing")
    with pytest.raises(tool.QualificationError, match="hash"):
        tool.atomic_install(source, destination, expected_sha256="0" * 64)
    assert destination.read_bytes() == b"existing"
    assert not destination.with_suffix(".xclbin.partial").exists()


def test_atomic_install_writes_exact_bytes_and_rejects_different_existing_file(tmp_path):
    tool = load_tool()
    source = tmp_path / "source.xclbin"
    digest = write_candidate(source)
    destination = tmp_path / tool.INSTALL_NAME
    tool.atomic_install(source, destination, expected_sha256=digest)
    assert destination.read_bytes() == source.read_bytes()
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
    destination.write_bytes(b"different")
    with pytest.raises(tool.QualificationError, match="existing"):
        tool.atomic_install(source, destination, expected_sha256=digest)


def test_write_record_is_canonical_and_atomic(tmp_path):
    tool = load_tool()
    output = tmp_path / "qualification.json"
    tool.write_record(output, {"z": 1, "a": 2})
    assert output.read_text(encoding="utf-8") == json.dumps(
        {"z": 1, "a": 2}, indent=2, sort_keys=True
    ) + "\n"
    assert not output.with_suffix(".json.partial").exists()
