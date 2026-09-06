#!/usr/bin/env bash
set -euo pipefail

pinned_revision="925e1179947ea0c0ebfb0032df18af3a729822be"
repo_root=/work
pristine=/llama-pristine
overlay=/qwen-build/llama-overlay
llama_build=/qwen-build/llama-build
rust_target=/qwen-build/hetgpu-target
build_jobs="${QWEN35_BUILD_JOBS:-32}"

if [[ ! "${build_jobs}" =~ ^[0-9]+$ ]] || (( build_jobs < 1 || build_jobs > 32 )); then
    echo "QWEN35_BUILD_JOBS must be an integer from 1 through 32" >&2
    exit 2
fi

test -d "${pristine}/.git" || { echo "missing pristine llama.cpp checkout" >&2; exit 1; }
test "$(git -C "${pristine}" rev-parse HEAD)" = "${pinned_revision}" || {
    echo "llama.cpp revision is not pinned to ${pinned_revision}" >&2
    exit 1
}
test -z "$(git -C "${pristine}" status --porcelain --untracked-files=normal)" || {
    echo "pristine llama.cpp checkout is dirty" >&2
    exit 1
}

"${repo_root}/tools/prepare_au250_qwen35_source.sh" "${pristine}" "${overlay}"

export PATH="/usr/local/cuda-13.0/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"
export RUSTUP_HOME=/qwen-build/rustup
export CARGO_HOME=/qwen-build/cargo
export CARGO_TARGET_DIR="${rust_target}"
export CARGO_BUILD_JOBS="${build_jobs}"
export PATH="${CARGO_HOME}/bin:${PATH}"
if ! command -v cargo >/dev/null 2>&1; then
    install -d "${CARGO_HOME}" "${RUSTUP_HOME}"
    curl --proto '=https' --tlsv1.2 -fsS https://sh.rustup.rs -o /qwen-build/rustup-init.sh
    sh /qwen-build/rustup-init.sh -y --profile minimal --default-toolchain 1.92.0
fi

cmake -S "${overlay}" -B "${llama_build}" \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_F16=ON \
    -DGGML_NATIVE=OFF \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${llama_build}" --target llama-server llama-cli -j"${build_jobs}"
c++ -O2 -std=c++17 \
    -I"${overlay}/ggml/include" \
    -I"${overlay}/ggml/src" \
    -I"${overlay}/ggml/src/ggml-cpu" \
    "${repo_root}/zluda/tests/tq1_upstream_reference.cpp" \
    -L"${llama_build}/bin" \
    -Wl,-rpath,"${llama_build}/bin" \
    -lggml-cpu -lggml-base -lggml -lpthread -ldl -lm \
    -o /qwen-build/tq1_upstream_reference

cargo build -p zluda --release --no-default-features \
    --features nvidia,embed_cudart,evaluation \
    --manifest-path "${repo_root}/Cargo.toml"

llama_server="${llama_build}/bin/llama-server"
llama_cli="${llama_build}/bin/llama-cli"
libggml="$(realpath -e "${llama_build}/bin/libggml.so")"
nvcuda="${rust_target}/release/libnvcuda.so"
cuda13_launch_shim="${rust_target}/release/libqwen35_cuda13_launch_shim.so"
upstream_oracle=/qwen-build/tq1_upstream_reference
cc -O2 -fPIC -shared -Wall -Wextra -Werror \
    "${repo_root}/tools/qwen35_cuda13_launch_shim.c" \
    -Wl,--version-script="${repo_root}/tools/qwen35_cuda13_launch_shim.map" \
    -ldl -lpthread -o "${cuda13_launch_shim}"
test -x "${llama_server}"
test -x "${llama_cli}"
test -s "${libggml}"
test -s "${nvcuda}"
test -s "${cuda13_launch_shim}"
objdump -T "${cuda13_launch_shim}" | grep -Eq 'libcudart\.so\.13[[:space:]]+cudaLaunchKernelExC$' || {
    echo "Qwen launch shim does not export the CUDA 13 launch ABI" >&2
    exit 1
}
objdump -T "${cuda13_launch_shim}" | grep -Eq 'libcudart\.so\.13[[:space:]]+__cudaRegisterFunction$' || {
    echo "Qwen launch shim does not intercept CUDA 13 function registration" >&2
    exit 1
}
test -x "${upstream_oracle}"
symbols="$(nm -D "${nvcuda}" | awk '$3 ~ /^hetgpu_tq1_(evaluate_raw|register_tensor|try_mul_mat_id)_v1$/ { print $3 }' | LC_ALL=C sort -u)"
test "${symbols}" = $'hetgpu_tq1_evaluate_raw_v1\nhetgpu_tq1_register_tensor_v1\nhetgpu_tq1_try_mul_mat_id_v1' || {
    echo "libnvcuda.so does not export exactly the three required TQ1 symbols" >&2
    exit 1
}
iq1s_symbols="$(nm -D "${nvcuda}" | awk \
  '$3 ~ /^hetgpu_iq1s_(bind_device_v1|register_tensor_v1|layer_(begin|set_routes|phase_commit|commit|abort)_v2)$/ { print $3 }' \
  | LC_ALL=C sort -u)"
expected_iq1s_symbols=$'hetgpu_iq1s_bind_device_v1\nhetgpu_iq1s_layer_abort_v2\nhetgpu_iq1s_layer_begin_v2\nhetgpu_iq1s_layer_commit_v2\nhetgpu_iq1s_layer_phase_commit_v2\nhetgpu_iq1s_layer_set_routes_v2\nhetgpu_iq1s_register_tensor_v1'
test "${iq1s_symbols}" = "${expected_iq1s_symbols}" || {
    echo "libnvcuda.so does not export the complete IQ1_S persistent ABI" >&2
    exit 1
}
iq1s_symbols_sha256="$(printf '%s\n' "${iq1s_symbols}" | sha256sum | awk '{print $1}')"
relocations="$(ldd -r "${nvcuda}" 2>&1)"
if grep -Fq 'undefined symbol:' <<<"${relocations}"; then
    printf '%s\n' "${relocations}" >&2
    echo "libnvcuda.so has unresolved runtime symbols" >&2
    exit 1
fi

patch_sha256="$(sha256sum "${repo_root}/tools/llama-qwen35-tq1-hetgpu.patch" | awk '{print $1}')"
hetgpu_commit="$(git -C "${repo_root}" rev-parse HEAD)"
dirty_manifest=/qwen-build/hetgpu-dirty.manifest
{
    git -C "${repo_root}" status --porcelain=v1 --untracked-files=normal --ignore-submodules=all
    git -C "${repo_root}" diff --binary --ignore-submodules=all HEAD
    while IFS= read -r path; do
        sha256sum "${repo_root}/${path}"
    done < <(git -C "${repo_root}" ls-files --others --exclude-standard | LC_ALL=C sort)
} > "${dirty_manifest}"
dirty_manifest_sha256="$(sha256sum "${dirty_manifest}" | awk '{print $1}')"
cuda_math_header_sha256="$(sha256sum /usr/local/cuda-13.0/targets/x86_64-linux/include/crt/math_functions.h | awk '{print $1}')"

PINNED_REVISION="${pinned_revision}" \
PATCH_SHA256="${patch_sha256}" \
HETGPU_COMMIT="${hetgpu_commit}" \
DIRTY_MANIFEST_SHA256="${dirty_manifest_sha256}" \
CUDA_MATH_HEADER_SHA256="${cuda_math_header_sha256}" \
LLAMA_SERVER="${llama_server}" \
LLAMA_CLI="${llama_cli}" \
LIBGGML="${libggml}" \
NVCUDA="${nvcuda}" \
CUDA13_LAUNCH_SHIM="${cuda13_launch_shim}" \
UPSTREAM_ORACLE="${upstream_oracle}" \
IQ1S_SYMBOLS="${iq1s_symbols}" \
IQ1S_SYMBOLS_SHA256="${iq1s_symbols_sha256}" \
python3 - <<'PY'
import hashlib
import json
import os
import pathlib
import subprocess

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def version(command):
    return subprocess.run(command, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=True).stdout.strip()

manifest = {
    "schema_version": 1,
    "llama_revision": os.environ["PINNED_REVISION"],
    "overlay_patch_sha256": os.environ["PATCH_SHA256"],
    "hetgpu_commit": os.environ["HETGPU_COMMIT"],
    "hetgpu_dirty_manifest_sha256": os.environ["DIRTY_MANIFEST_SHA256"],
    "cuda_math_header_sha256": os.environ["CUDA_MATH_HEADER_SHA256"],
    "iq1s_persistent_abi_symbols": os.environ["IQ1S_SYMBOLS"].splitlines(),
    "iq1s_persistent_abi_symbols_sha256": os.environ["IQ1S_SYMBOLS_SHA256"],
    "compilers": {
        "cc": version(["cc", "--version"]).splitlines()[0],
        "cxx": version(["c++", "--version"]).splitlines()[0],
        "nvcc": version(["nvcc", "--version"]),
        "rustc": version(["rustc", "--version"]),
        "cargo": version(["cargo", "--version"]),
    },
    "artifacts": {
        "llama_server": {
            "path": os.environ["LLAMA_SERVER"],
            "sha256": sha256(os.environ["LLAMA_SERVER"]),
        },
        "llama_cli": {
            "path": os.environ["LLAMA_CLI"],
            "sha256": sha256(os.environ["LLAMA_CLI"]),
        },
        "libggml": {
            "path": os.environ["LIBGGML"],
            "sha256": sha256(os.environ["LIBGGML"]),
        },
        "libnvcuda": {
            "path": os.environ["NVCUDA"],
            "sha256": sha256(os.environ["NVCUDA"]),
        },
        "cuda13_launch_shim": {
            "path": os.environ["CUDA13_LAUNCH_SHIM"],
            "sha256": sha256(os.environ["CUDA13_LAUNCH_SHIM"]),
        },
        "tq1_upstream_reference": {
            "path": os.environ["UPSTREAM_ORACLE"],
            "sha256": sha256(os.environ["UPSTREAM_ORACLE"]),
        },
    },
    "build_commands": {
        "cmake": "GGML_CUDA=ON GGML_CUDA_F16=ON CMAKE_CUDA_ARCHITECTURES=120",
        "targets": "llama-server llama-cli",
        "cargo_features": "nvidia,embed_cudart,evaluation",
        "oracle": "pinned llama.cpp quantize_row_q8_K_ref and ggml_vec_dot_tq1_0_q8_K",
    },
}
destination = pathlib.Path("/qwen-build/manifest.json")
temporary = destination.with_suffix(".json.partial")
temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
os.replace(temporary, destination)
PY

test -s /qwen-build/manifest.json
echo "Built Qwen TQ1 AU250 runtime; manifest: /qwen-build/manifest.json"
