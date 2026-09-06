#!/usr/bin/env python3
"""Fail-closed static qualification for the Qwen397B persistent U250 image.

This tool deliberately does not import or call XRT.  It proves the immutable
image identity, kernel ABI/topology, final routed state, timing closure, and
the source/build provenance before an optional atomic install.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping


CANDIDATE_SHA256 = "9c83dcae07b4c7bf1d2e1cebf46ccf0ff1ebf8848a437035fef1451dee7770a3"
CANDIDATE_UUID = "b1bafc64-09fd-32b0-a5b4-a881e554ae84"
EXPECTED_CUS = {
    "iq1s_layer_big_1": "bank0",
    "iq1s_layer_big_2": "bank3",
    "iq1s_layer_big_3": "bank2",
    "iq1s_layer_small_1": "bank1",
}
EXPECTED_ARGS = (
    "command_ring",
    "completion_ring",
    "program",
    "arena_manifest",
    "activation_slab",
    "result_slab",
)
INSTALL_NAME = "qwen397b_iq1s_layer_persistent_9c83dcae.xclbin"


class QualificationError(RuntimeError):
    """The candidate failed a mandatory static qualification check."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _require_digest(name: str, value: object) -> str:
    text = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", text) or text == "0" * 64:
        raise QualificationError(f"incomplete source provenance: {name}")
    return text


def _validate_provenance(provenance: Mapping[str, object]) -> str:
    head = str(provenance.get("rtl_head", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise QualificationError("incomplete source provenance: rtl_head")
    for field in (
        "tracked_diff_sha256",
        "untracked_sources_sha256",
        "build_log_sha256",
        "timing_report_sha256",
        "route_report_sha256",
        "synthesis_logs_sha256",
    ):
        _require_digest(field, provenance.get(field, ""))
    if provenance.get("build_exit_code") != 0:
        raise QualificationError("build exit status is not zero")
    digest = hashlib.sha256(_canonical_bytes(dict(provenance))).hexdigest()
    if digest == "0" * 64:  # Defensive, although SHA-256 cannot produce this here.
        raise QualificationError("incomplete source provenance digest")
    return digest


def _parse_info(info_text: str) -> tuple[str, dict[str, str]]:
    uuid_match = re.search(r"UUID \(xclbin\):\s*([0-9a-fA-F-]{36})", info_text)
    if uuid_match is None:
        raise QualificationError("xclbin UUID is missing")
    image_uuid = uuid_match.group(1).lower()

    signatures: dict[str, tuple[str, ...]] = {}
    for kernel, raw_args in re.findall(
        r"Signature:\s*(iq1s_layer_(?:small|big))\s*\(([^)]*)\)", info_text
    ):
        args = tuple(
            match.group(1)
            for arg in raw_args.split(",")
            if (match := re.fullmatch(r"\s*void\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*", arg))
        )
        if len(args) != len(raw_args.split(",")):
            raise QualificationError(f"invalid kernel signature for {kernel}")
        signatures[kernel] = args
    expected_kernels = {"iq1s_layer_small", "iq1s_layer_big"}
    if set(signatures) != expected_kernels or any(
        args != EXPECTED_ARGS for args in signatures.values()
    ):
        raise QualificationError("persistent kernel signature does not match the six-argument ABI")

    cu_banks: dict[str, str] = {}
    instance_matches = list(re.finditer(r"^Instance:\s*([A-Za-z0-9_]+)\s*$", info_text, re.MULTILINE))
    for index, match in enumerate(instance_matches):
        instance = match.group(1)
        end = instance_matches[index + 1].start() if index + 1 < len(instance_matches) else len(info_text)
        block = info_text[match.end() : end]
        memories = set(re.findall(r"^\s*Memory:\s*(bank[0-9]+)\b", block, re.MULTILINE))
        if len(memories) != 1:
            raise QualificationError(f"compute unit {instance} has an ambiguous bank mapping")
        cu_banks[instance] = memories.pop()
    if set(cu_banks) != set(EXPECTED_CUS):
        raise QualificationError(
            f"compute unit set mismatch: expected {sorted(EXPECTED_CUS)}, got {sorted(cu_banks)}"
        )
    if cu_banks != EXPECTED_CUS:
        raise QualificationError(f"compute unit bank mapping mismatch: {cu_banks}")
    return image_uuid, cu_banks


def _parse_timing(timing_text: str) -> dict[str, float]:
    if "All user specified timing constraints are met." not in timing_text:
        raise QualificationError("timing constraints-met sentinel is missing")
    summary = re.search(
        r"WNS\(ns\).*?WHS\(ns\).*?\n(?:\s*-+.*?\n)?"
        r"\s*([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+\d+\s+\d+\s+"
        r"([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\b",
        timing_text,
        re.DOTALL,
    )
    if summary is None:
        raise QualificationError("timing summary is missing or malformed")
    wns, tns, whs, ths = (float(item) for item in summary.groups())
    if wns < 0.0 or tns != 0.0 or whs < 0.0 or ths != 0.0:
        raise QualificationError(
            f"timing failed: WNS={wns}, TNS={tns}, WHS={whs}, THS={ths}"
        )
    return {"wns_ns": wns, "tns_ns": tns, "whs_ns": whs, "ths_ns": ths}


def _parse_route(route_text: str) -> dict[str, int]:
    # Unit fixtures use compact report_route_status spelling.
    compact = {
        "unrouted_nets": re.findall(r"# of unrouted nets\s*=\s*(\d+)", route_text, re.IGNORECASE),
        "routing_errors": re.findall(
            r"# of nets with routing errors\s*=\s*(\d+)", route_text, re.IGNORECASE
        ),
        "node_overlaps": re.findall(r"# of nets with overlaps\s*=\s*(\d+)", route_text, re.IGNORECASE),
    }
    if any(compact.values()):
        if not all(len(values) == 1 for values in compact.values()):
            raise QualificationError("routing report is incomplete")
        result = {key: int(values[0]) for key, values in compact.items()}
    else:
        # The Vivado implementation log contains intermediate routing states.
        # Accept only a contiguous zero finalization block that is subsequently
        # verified and closed by the successful router/route_design sentinels.
        zero_block = re.search(
            r"Number of Failed Nets\s*=\s*0.*?"
            r"Number of Unrouted Nets\s*=\s*0.*?"
            r"Number of Partially Routed Nets\s*=\s*0.*?"
            r"Number of Node Overlaps\s*=\s*0.*?"
            r"Verification completed successfully",
            route_text,
            re.DOTALL,
        )
        required = (
            "Router Completed Successfully",
            "route_design completed successfully",
            "Routing Is Done.",
        )
        if zero_block is None or not all(item in route_text for item in required):
            raise QualificationError("routing did not reach a verified zero-error final state")
        if not re.search(r"\b0 Errors encountered\.", route_text):
            raise QualificationError("routing completed with errors")
        result = {"unrouted_nets": 0, "routing_errors": 0, "node_overlaps": 0}
    if any(result.values()):
        raise QualificationError(f"routing failed: {result}")
    return result


def _validate_synthesis_logs(
    synthesis_logs: Mapping[str, str],
) -> dict[str, dict[str, object]]:
    if set(synthesis_logs) != set(EXPECTED_CUS):
        raise QualificationError(
            "synthesis log set mismatch: "
            f"expected {sorted(EXPECTED_CUS)}, got {sorted(synthesis_logs)}"
        )
    result: dict[str, dict[str, object]] = {}
    for instance in sorted(EXPECTED_CUS):
        text = synthesis_logs[instance]
        if "Synth 8-4445" in text:
            raise QualificationError(
                f"{instance} contains Synth 8-4445 missing-readmem evidence"
            )
        if re.search(r"grid_rom.*does not have driver", text, re.IGNORECASE):
            raise QualificationError(f"{instance} contains undriven grid ROM evidence")
        if not re.search(
            r"Synth 8-3876.*\$readmem data file ['\"].*IQ1S_GRID\.memh['\"] "
            r"is read successfully",
            text,
        ):
            raise QualificationError(f"{instance} is missing grid read-success evidence")
        if (
            "Synthesis finished with 0 errors, 0 critical warnings" not in text
            or "synth_design completed successfully" not in text
        ):
            raise QualificationError(f"{instance} is missing clean completion evidence")
        result[instance] = {
            "grid_read_success": True,
            "log_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
    return result


def qualify_from_text(
    *,
    candidate: Path,
    info_text: str,
    timing_text: str,
    route_text: str,
    synthesis_logs: Mapping[str, str],
    expected_sha256: str,
    expected_uuid: str,
    provenance: Mapping[str, object],
) -> dict[str, Any]:
    if not candidate.is_file():
        raise QualificationError(f"candidate does not exist: {candidate}")
    actual_sha256 = sha256(candidate)
    if actual_sha256 != expected_sha256:
        raise QualificationError(
            f"candidate SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    image_uuid, cu_banks = _parse_info(info_text)
    if image_uuid != expected_uuid.lower():
        raise QualificationError(
            f"candidate UUID mismatch: expected {expected_uuid}, got {image_uuid}"
        )
    timing = _parse_timing(timing_text)
    routing = _parse_route(route_text)
    synthesis = _validate_synthesis_logs(synthesis_logs)
    provenance_digest = _validate_provenance(provenance)
    return {
        "arguments": list(EXPECTED_ARGS),
        "cu_banks": cu_banks,
        "routing": routing,
        "synthesis": synthesis,
        "sha256": actual_sha256,
        "source_provenance": dict(provenance),
        "source_provenance_sha256": provenance_digest,
        "status": "pass",
        "timing": timing,
        "uuid": image_uuid,
    }


def atomic_install(source: Path, destination: Path, *, expected_sha256: str) -> None:
    actual_source_hash = sha256(source)
    if actual_source_hash != expected_sha256:
        raise QualificationError(
            f"source hash mismatch before install: expected {expected_sha256}, got {actual_source_hash}"
        )
    if destination.exists():
        if not destination.is_file() or sha256(destination) != expected_sha256:
            raise QualificationError(f"existing install destination conflicts: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        with source.open("rb") as src, temporary.open("xb") as dst:
            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        copied_hash = sha256(temporary)
        if copied_hash != expected_sha256:
            raise QualificationError(
                f"installed candidate hash mismatch: expected {expected_sha256}, got {copied_hash}"
            )
        if destination.exists():
            raise QualificationError(f"existing install destination appeared during copy: {destination}")
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _run_bytes(command: list[str], *, cwd: Path) -> bytes:
    completed = subprocess.run(command, cwd=cwd, check=False, capture_output=True)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise QualificationError(
            f"command failed with exit {completed.returncode}: {' '.join(command)}\n{stderr}"
        )
    return completed.stdout


def _hash_untracked_sources(rtl_root: Path) -> tuple[str, list[dict[str, str]]]:
    raw = _run_bytes(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=rtl_root
    )
    roots = ("rtl/", "dv/", "synth/", "sw_utils/")
    source_suffixes = {
        ".c", ".cc", ".cpp", ".h", ".hpp", ".json", ".py", ".sv", ".svh",
        ".tcl", ".v", ".vh", ".xdc", ".yml", ".yaml",
    }
    entries: list[dict[str, str]] = []
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        relative = encoded.decode("utf-8", errors="strict")
        path = rtl_root / relative
        if relative.startswith(roots) and path.suffix.lower() in source_suffixes and path.is_file():
            entries.append({"path": relative, "sha256": sha256(path)})
    entries.sort(key=lambda item: item["path"])
    return hashlib.sha256(_canonical_bytes(entries)).hexdigest(), entries


def _collect_synthesis_logs(
    build_root: Path,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    log_root = build_root / "_x/logs/link/syn"
    texts: dict[str, str] = {}
    entries: list[dict[str, str]] = []
    for instance in sorted(EXPECTED_CUS):
        path = log_root / f"ulp_{instance}_0_synth_1_runme.log"
        if not path.is_file():
            raise QualificationError(f"required synthesis evidence is missing: {path}")
        texts[instance] = path.read_text(encoding="utf-8", errors="replace")
        entries.append({"instance": instance, "path": str(path), "sha256": sha256(path)})
    return texts, entries


def _collect_provenance(
    rtl_root: Path, build_root: Path
) -> tuple[dict[str, object], Path, Path, Path, dict[str, str]]:
    build_logs = (
        build_root / "v++_kernel.gridrom-fixed.log",
        build_root / "v++_kernel.resume-s2-i8.log",
    )
    build_log = next((path for path in build_logs if path.is_file()), build_logs[0])
    timing_report = (
        build_root / "_x/reports/link/imp/impl_1_hw_bb_locked_timing_summary_postroute_physopted.rpt"
    )
    route_report = build_root / "_x/logs/link/imp/impl_1_runme.log"
    for path in (build_log, timing_report, route_report):
        if not path.is_file():
            raise QualificationError(f"required build evidence is missing: {path}")
    head = _run_bytes(["git", "rev-parse", "HEAD"], cwd=rtl_root).decode().strip()
    tracked_diff = _run_bytes(["git", "diff", "--binary", "HEAD", "--"], cwd=rtl_root)
    untracked_digest, untracked_entries = _hash_untracked_sources(rtl_root)
    synthesis_logs, synthesis_entries = _collect_synthesis_logs(build_root)
    build_text = build_log.read_text(encoding="utf-8", errors="replace")
    exit_matches = re.findall(r"^EXIT=(\d+)\s*$", build_text, re.MULTILINE)
    if not exit_matches:
        raise QualificationError("build exit status sentinel is missing")
    provenance: dict[str, object] = {
        "build_exit_code": int(exit_matches[-1]),
        "build_log": str(build_log),
        "build_log_sha256": sha256(build_log),
        "route_report": str(route_report),
        "route_report_sha256": sha256(route_report),
        "rtl_head": head,
        "synthesis_logs": synthesis_entries,
        "synthesis_logs_sha256": hashlib.sha256(
            _canonical_bytes(synthesis_entries)
        ).hexdigest(),
        "timing_report": str(timing_report),
        "timing_report_sha256": sha256(timing_report),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "untracked_source_count": len(untracked_entries),
        "untracked_sources": untracked_entries,
        "untracked_sources_sha256": untracked_digest,
    }
    return provenance, build_log, timing_report, route_report, synthesis_logs


def _xclbin_info(candidate: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="qwen397b-xclbin-info-") as directory:
        output = Path(directory) / "candidate.info"
        completed = subprocess.run(
            [
                "xclbinutil", "--quiet", "--force", "--info", str(output),
                "--input", str(candidate),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not output.is_file():
            raise QualificationError(
                f"xclbinutil --info failed with exit {completed.returncode}: {completed.stderr}"
            )
        return output.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--rtl-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--install-dir", type=Path)
    args = parser.parse_args()

    candidate = args.candidate.resolve(strict=True)
    rtl_root = args.rtl_root.resolve(strict=True)
    build_root = args.build_root.resolve(strict=True)
    provenance, _, timing_report, route_report, synthesis_logs = _collect_provenance(
        rtl_root, build_root
    )
    record = qualify_from_text(
        candidate=candidate,
        info_text=_xclbin_info(candidate),
        timing_text=timing_report.read_text(encoding="utf-8", errors="replace"),
        route_text=route_report.read_text(encoding="utf-8", errors="replace"),
        synthesis_logs=synthesis_logs,
        expected_sha256=CANDIDATE_SHA256,
        expected_uuid=CANDIDATE_UUID,
        provenance=provenance,
    )
    if args.install_dir is not None:
        destination = args.install_dir.resolve() / INSTALL_NAME
        atomic_install(candidate, destination, expected_sha256=CANDIDATE_SHA256)
        record["installed_path"] = str(destination)
    else:
        record["installed_path"] = None
    write_record(args.output, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QualificationError as error:
        raise SystemExit(f"qualification failed: {error}") from error
