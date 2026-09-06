#!/usr/bin/env python3
"""Behavior tests for the strict Qwen build-artifact preflight."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


PREFLIGHT = Path(__file__).parents[2] / "tools" / "qwen35_build_preflight.py"
PINNED_REVISION = "925e1179947ea0c0ebfb0032df18af3a729822be"
REQUIRED_SYMBOLS = ("dequantize_row_iq1_s", "ggml_init", "ggml_free")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_library(path, symbols=REQUIRED_SYMBOLS):
    source = path.with_suffix(".c")
    source.write_text(
        "\n".join(f"void {symbol}(void) {{}}" for symbol in symbols) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["cc", "-shared", "-fPIC", "-o", str(path), str(source)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def build_library_with_sibling_dependency(path):
    dependency_source = path.parent / "preflight-dependency.c"
    dependency_source.write_text("void preflight_dependency(void) {}\n", encoding="utf-8")
    dependency = path.parent / "libpreflight-dependency.so.0"
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-Wl,-soname,libpreflight-dependency.so.0",
            "-o",
            str(dependency),
            str(dependency_source),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    (path.parent / "libpreflight-dependency.so").symlink_to(dependency.name)
    source = path.with_suffix(".c")
    source.write_text(
        "extern void preflight_dependency(void);\n"
        "void dequantize_row_iq1_s(void) {}\n"
        "void ggml_init(void) { preflight_dependency(); }\n"
        "void ggml_free(void) {}\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-o",
            str(path),
            str(source),
            f"-L{path.parent}",
            "-lpreflight-dependency",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def write_manifest(path, library, *, schema=1, revision=PINNED_REVISION, digest=None):
    path.write_text(
        json.dumps(
            {
                "schema_version": schema,
                "llama_revision": revision,
                "artifacts": {
                    "libggml": {
                        "path": str(library),
                        "sha256": sha256(library) if digest is None else digest,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def run_preflight(build_root, manifest, output):
    return subprocess.run(
        [
            sys.executable,
            str(PREFLIGHT),
            "--manifest",
            str(manifest),
            "--build-root",
            str(build_root),
            "--llama-revision",
            PINNED_REVISION,
            "--output",
            str(output),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_accepts_hashed_regular_libggml_with_required_symbols(tmp_path):
    build_root = tmp_path / "build"
    build_root.mkdir()
    library = build_root / "libggml.so"
    manifest = build_root / "manifest.json"
    output = tmp_path / "verified.json"
    build_library(library)
    write_manifest(manifest, library)

    result = run_preflight(build_root, manifest, output)

    assert result.returncode == 0, result.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record == {
        "schema_version": 1,
        "build_root": str(build_root.resolve()),
        "libggml_path": str(library.resolve()),
        "libggml_sha256": sha256(library),
        "llama_revision": PINNED_REVISION,
        "required_symbols": list(REQUIRED_SYMBOLS),
        "status": "pass",
    }
    assert not output.with_suffix(".json.partial").exists()


def test_remaps_container_libggml_path_to_host_build_root(tmp_path):
    build_root = tmp_path / "build"
    build_root.mkdir()
    library = build_root / "llama-build" / "bin" / "libggml.so"
    library.parent.mkdir(parents=True)
    manifest = build_root / "manifest.json"
    output = tmp_path / "verified.json"
    build_library(library)
    write_manifest(manifest, Path("/qwen-build/llama-build/bin/libggml.so"), digest=sha256(library))

    result = run_preflight(build_root, manifest, output)

    assert result.returncode == 0, result.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["build_root"] == str(build_root.resolve())
    assert record["libggml_path"] == str(library.resolve())
    assert record["libggml_sha256"] == sha256(library)


def test_loads_verified_libggml_with_sibling_dependency(tmp_path):
    build_root = tmp_path / "build"
    build_root.mkdir()
    library = build_root / "llama-build" / "bin" / "libggml.so"
    library.parent.mkdir(parents=True)
    manifest = build_root / "manifest.json"
    output = tmp_path / "verified.json"
    build_library_with_sibling_dependency(library)
    write_manifest(manifest, Path("/qwen-build/llama-build/bin/libggml.so"), digest=sha256(library))

    result = run_preflight(build_root, manifest, output)

    assert result.returncode == 0, result.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "pass"
    assert record["libggml_path"] == str(library.resolve())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_artifact", "libggml"),
        ("wrong_schema", "schema"),
        ("wrong_revision", "revision"),
        ("wrong_hash", "SHA-256"),
        ("directory", "regular file"),
        ("missing_symbol", "ggml_free"),
        ("symlink_inside", "symlink"),
        ("symlink_escape", "build root"),
        ("path_escape", "build root"),
    ],
)
def test_rejects_unqualified_libggml_before_writing_output(tmp_path, mutation, message):
    build_root = tmp_path / "build"
    build_root.mkdir()
    library = build_root / "libggml.so"
    manifest = build_root / "manifest.json"
    output = tmp_path / "verified.json"
    build_library(library)

    if mutation == "missing_artifact":
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "llama_revision": PINNED_REVISION,
                    "artifacts": {},
                }
            ),
            encoding="utf-8",
        )
    elif mutation == "wrong_schema":
        write_manifest(manifest, library, schema=2)
    elif mutation == "wrong_revision":
        write_manifest(manifest, library, revision="0" * 40)
    elif mutation == "wrong_hash":
        write_manifest(manifest, library, digest="0" * 64)
    elif mutation == "directory":
        library.unlink()
        library.mkdir()
        write_manifest(manifest, library, digest="0" * 64)
    elif mutation == "missing_symbol":
        library.unlink()
        build_library(library, REQUIRED_SYMBOLS[:-1])
        write_manifest(manifest, library)
    elif mutation == "symlink_inside":
        target = build_root / "real-libggml.so"
        library.rename(target)
        library.symlink_to(target.name)
        write_manifest(manifest, library, digest=sha256(target))
    elif mutation in ("symlink_escape", "path_escape"):
        outside = tmp_path / "outside-libggml.so"
        library.unlink()
        build_library(outside)
        if mutation == "symlink_escape":
            library.symlink_to(outside)
            write_manifest(manifest, library, digest=sha256(outside))
        else:
            write_manifest(manifest, outside)

    result = run_preflight(build_root, manifest, output)

    assert result.returncode != 0
    assert message.lower() in result.stderr.lower()
    assert not output.exists()
    assert not output.with_suffix(".json.partial").exists()
