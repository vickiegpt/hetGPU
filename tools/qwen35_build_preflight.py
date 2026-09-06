#!/usr/bin/env python3
"""Fail-closed validation of the pinned Qwen runtime's libggml artifact."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


REQUIRED_SYMBOLS = ("dequantize_row_iq1_s", "ggml_init", "ggml_free")
CONTAINER_BUILD_ROOT = Path("/qwen-build")
SHA256 = re.compile(r"[0-9a-f]{64}")


class PreflightError(ValueError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"cannot read build manifest: {error}") from error
    if not isinstance(value, dict):
        raise PreflightError("build manifest root must be an object")
    return value


def verify_required_symbols(library_path):
    loader_path = str(library_path.parent)
    environment = os.environ.copy()
    if environment.get("LD_LIBRARY_PATH"):
        loader_path = f"{loader_path}:{environment['LD_LIBRARY_PATH']}"
    environment["LD_LIBRARY_PATH"] = loader_path
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import ctypes
import sys

try:
    library = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_LOCAL)
except OSError as error:
    print(error, file=sys.stderr)
    raise SystemExit(2)
for symbol in sys.argv[2:]:
    try:
        getattr(library, symbol)
    except AttributeError:
        print(f"verified libggml is missing required symbol {symbol}", file=sys.stderr)
        raise SystemExit(3)
""",
            str(library_path),
            *REQUIRED_SYMBOLS,
        ],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    detail = probe.stderr.strip() or f"symbol probe exited with status {probe.returncode}"
    if probe.returncode == 3:
        raise PreflightError(detail)
    if probe.returncode != 0:
        raise PreflightError(f"cannot load verified libggml artifact: {detail}")


def verify(manifest_path, build_root, expected_revision):
    try:
        canonical_root = build_root.resolve(strict=True)
    except OSError as error:
        raise PreflightError(f"cannot resolve build root: {error}") from error
    if not canonical_root.is_dir():
        raise PreflightError("build root is not a directory")

    manifest = load_manifest(manifest_path)
    if manifest.get("schema_version") != 1:
        raise PreflightError("build manifest schema mismatch; expected version 1")
    if manifest.get("llama_revision") != expected_revision:
        raise PreflightError("build manifest llama revision mismatch")

    artifact = manifest.get("artifacts", {}).get("libggml")
    if not isinstance(artifact, dict):
        raise PreflightError("build manifest is missing artifacts.libggml")
    raw_path = artifact.get("path")
    recorded_hash = artifact.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise PreflightError("build manifest libggml path is invalid")
    if not isinstance(recorded_hash, str) or SHA256.fullmatch(recorded_hash) is None:
        raise PreflightError("build manifest libggml SHA-256 is invalid")

    library_path = Path(raw_path)
    if library_path.is_relative_to(CONTAINER_BUILD_ROOT):
        library_path = canonical_root / library_path.relative_to(CONTAINER_BUILD_ROOT)
    try:
        canonical_library = library_path.resolve(strict=True)
    except OSError as error:
        raise PreflightError(f"cannot resolve libggml artifact: {error}") from error
    if not canonical_library.is_relative_to(canonical_root):
        raise PreflightError("libggml artifact escapes the canonical build root")
    if library_path.is_symlink():
        raise PreflightError("libggml artifact must not be a symlink")
    if not canonical_library.is_file():
        raise PreflightError("libggml artifact is not a regular file")

    actual_hash = sha256(canonical_library)
    if actual_hash != recorded_hash:
        raise PreflightError("libggml SHA-256 does not match the build manifest")

    try:
        verify_required_symbols(canonical_library)
    except subprocess.TimeoutExpired as error:
        raise PreflightError("verified libggml symbol probe timed out") from error

    return {
        "schema_version": 1,
        "build_root": str(canonical_root),
        "libggml_path": str(canonical_library),
        "libggml_sha256": actual_hash,
        "llama_revision": expected_revision,
        "required_symbols": list(REQUIRED_SYMBOLS),
        "status": "pass",
    }


def write_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--llama-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    try:
        value = verify(args.manifest, args.build_root, args.llama_revision)
        write_atomic(args.output, value)
    except (OSError, PreflightError) as error:
        partial.unlink(missing_ok=True)
        print(f"qwen35_build_preflight: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
