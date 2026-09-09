#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
iq1s_validator="${repo_root}/zluda/tests/validate_qwen35_iq1s_au250_proof.py"

if [[ "${1:-}" == "--inside" ]]; then
    [[ $# -eq 3 ]] || { echo "usage: $0 --inside PROOF_DIR XCLBIN" >&2; exit 2; }
    proof_dir=$2
    xclbin=$3
    case "${proof_dir}" in
        /work/.proof/qwen35-iq1s-au250-*) ;;
        *) echo "refusing unexpected proof directory ${proof_dir}" >&2; exit 2 ;;
    esac

    model=/models/qwen/Qwen3.5-397B-A17B-UD-TQ1_0.gguf
    manifest=/qwen-build/manifest.json
    llama_server=/qwen-build/llama-build/bin/llama-server
    libnvcuda=/qwen-build/hetgpu-target/release/libnvcuda.so
    cuda13_launch_shim=/qwen-build/hetgpu-target/release/libqwen35_cuda13_launch_shim.so
    oracle=/qwen-build/tq1_upstream_reference
    evaluator=/work/tools/qwen35_au250_eval.py
    auditor=/work/tools/qwen35_gguf_audit.py
    build_preflight=/work/tools/qwen35_build_preflight.py
    route_manifest=/work/tools/qwen35-iq1s-route-manifest.json
    prompt_seed=/work/zluda/evaluation/fixtures/qwen35_prompt_seed.txt
    iq1s_validator=/work/zluda/tests/validate_qwen35_iq1s_au250_proof.py
    model_size=94155830880
    model_sha256=0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568
    llama_revision=925e1179947ea0c0ebfb0032df18af3a729822be
    fpga_bdf=0000:64:00.1
    profile=one-token
    cu_config='{"version":1,"cus":[{"ip_name":"ternip_big:ternip_big_1","memory_group":0,"lanes":9},{"ip_name":"ternip_big:ternip_big_2","memory_group":3,"lanes":9},{"ip_name":"ternip_big:ternip_big_3","memory_group":2,"lanes":9},{"ip_name":"ternip_small:ternip_small_1","memory_group":1,"lanes":6}]}'

    for required in \
        "${model}" "${manifest}" "${llama_server}" "${libnvcuda}" "${cuda13_launch_shim}" "${oracle}" \
        "${evaluator}" "${auditor}" "${build_preflight}" "${route_manifest}" "${prompt_seed}" \
        "${xclbin}" "${iq1s_validator}"; do
        [[ -f "${required}" ]] || { echo "missing Qwen IQ1_S evaluation input ${required}" >&2; exit 1; }
    done
    install -d "${proof_dir}"

    actual_size="$(stat -c %s "${model}")"
    [[ "${actual_size}" == "${model_size}" ]] || { echo "Qwen model size mismatch" >&2; exit 1; }
    actual_model_sha="$(sha256sum "${model}" | awk '{print $1}')"
    [[ "${actual_model_sha}" == "${model_sha256}" ]] || { echo "Qwen model SHA-256 mismatch" >&2; exit 1; }
    MODEL="${model}" MODEL_SHA256="${model_sha256}" OUTPUT="${proof_dir}/model-verification.json" python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["MODEL"])
stat = path.stat()
record = {
    "path": str(path),
    "size": stat.st_size,
    "device": stat.st_dev,
    "inode": stat.st_ino,
    "mtime_ns": stat.st_mtime_ns,
    "ctime_ns": stat.st_ctime_ns,
    "sha256": os.environ["MODEL_SHA256"],
}
output = Path(os.environ["OUTPUT"])
temporary = output.with_suffix(output.suffix + ".partial")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
    python3 "${auditor}" "${model}" \
        --model-verification "${proof_dir}/model-verification.json" \
        --output "${proof_dir}/model-tensor-audit.json"

    libggml="$(realpath -e /qwen-build/llama-build/bin/libggml.so)"
    LLAMA_SERVER="${llama_server}" LIBGGML="${libggml}" LIBNVCUDA="${libnvcuda}" \
    CUDA13_LAUNCH_SHIM="${cuda13_launch_shim}" \
    ORACLE="${oracle}" \
    MANIFEST="${manifest}" LLAMA_REVISION="${llama_revision}" python3 - <<'PY'
import hashlib
import json
import os

def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

manifest = json.load(open(os.environ["MANIFEST"], encoding="utf-8"))
if manifest.get("schema_version") != 1 or manifest.get("llama_revision") != os.environ["LLAMA_REVISION"]:
    raise SystemExit("build manifest revision/schema mismatch")
for name, variable in (("llama_server", "LLAMA_SERVER"), ("libggml", "LIBGGML"), ("libnvcuda", "LIBNVCUDA"), ("cuda13_launch_shim", "CUDA13_LAUNCH_SHIM"), ("tq1_upstream_reference", "ORACLE")):
    artifact = manifest.get("artifacts", {}).get(name, {})
    if artifact.get("path") != os.environ[variable] or artifact.get("sha256") != digest(os.environ[variable]):
        raise SystemExit(f"build manifest artifact mismatch: {name}")
PY
    python3 "${build_preflight}" \
        --manifest "${manifest}" \
        --build-root /qwen-build \
        --llama-revision "${llama_revision}" \
        --output "${proof_dir}/qwen-build-preflight.json"
    verified_libggml="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["libggml_path"])' "${proof_dir}/qwen-build-preflight.json")"
    libggml_sha256="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["libggml_sha256"])' "${proof_dir}/qwen-build-preflight.json")"
    server_sha256="$(sha256sum "${llama_server}" | awk '{print $1}')"
    libnvcuda_sha256="$(sha256sum "${libnvcuda}" | awk '{print $1}')"
    cuda13_launch_shim_sha256="$(sha256sum "${cuda13_launch_shim}" | awk '{print $1}')"
    xclbin_sha256="$(sha256sum "${xclbin}" | awk '{print $1}')"

    xclbin_info="$(xclbinutil --info --input "${xclbin}" 2>&1)"
    printf '%s\n' "${xclbin_info}" > "${proof_dir}/xclbin-info.txt"
    for cu in ternip_big_1 ternip_big_2 ternip_big_3 ternip_small_1; do
        grep -Fq "Instance:        ${cu}" <<<"${xclbin_info}" || {
            echo "xclbin is missing expected compute unit ${cu}" >&2
            exit 1
        }
    done

    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits \
        > "${proof_dir}/nvidia-memory-preflight.csv"
    MODEL_SIZE="${model_size}" python3 - "${proof_dir}/nvidia-memory-preflight.csv" <<'PY'
import os
import pathlib
import sys

free_mib = 0
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    fields = [field.strip() for field in line.split(",")]
    free_mib += int(fields[-1])
required = int(os.environ["MODEL_SIZE"]) + 2 * 1024**3
if free_mib * 1024**2 < required:
    raise SystemExit(f"insufficient aggregate free GPU memory: {free_mib} MiB free, {required} bytes required")
PY

    xbutil examine -d "${fpga_bdf}" -r dynamic-regions -r error -r firewall -r thermal \
        > "${proof_dir}/xbutil-preflight.txt" 2>&1
    lspci -s "${fpga_bdf}" -vv > "${proof_dir}/pcie-link.txt"
    grep -Fq 'Level 0 : 0x0 (GOOD)' "${proof_dir}/xbutil-preflight.txt" || {
        echo "AU250 firewall is not GOOD during preflight" >&2
        exit 1
    }
    if grep -Eiq '(^|[^[:alpha:]])fatal([^[:alpha:]]|$)' "${proof_dir}/xbutil-preflight.txt"; then
        echo "AU250 reported a fatal preflight error" >&2
        exit 1
    fi

    {
        printf 'model_sha256=%s\n' "${actual_model_sha}"
        printf 'llama_server_sha256=%s\n' "${server_sha256}"
        printf 'libggml_sha256=%s\n' "${libggml_sha256}"
        printf 'libnvcuda_sha256=%s\n' "${libnvcuda_sha256}"
        printf 'cuda13_launch_shim_sha256=%s\n' "${cuda13_launch_shim_sha256}"
        printf 'xclbin_sha256=%s\n' "${xclbin_sha256}"
        printf 'llama_revision=%s\n' "${llama_revision}"
        printf 'build_threads=%s\n' "${QWEN35_BUILD_JOBS:-32}"
        printf 'threads=%s\n' "${QWEN35_THREADS:-$(nproc)}"
    } > "${proof_dir}/artifact-hashes.txt"
    git -C /work status --porcelain=v1 --untracked-files=normal --ignore-submodules=all \
        > "${proof_dir}/repository-status.txt"
    git -C /work diff --binary --ignore-submodules=all HEAD | sha256sum | awk '{print $1}' \
        > "${proof_dir}/repository-diff.sha256"
    nvidia-smi -L > "${proof_dir}/nvidia-gpus.txt"

    export LD_LIBRARY_PATH="/qwen-build/llama-build/bin:/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"
    export HETGPU_XRT_XCLBIN="${xclbin}"
    export HETGPU_XRT_NUM_VECTOR_REGISTERS=4
    export HETGPU_XRT_TIMEOUT_MS=10000
    export HETGPU_XRT_CLOCK_HZ=300000000
    export HETGPU_XRT_CU_CONFIG="${cu_config}"
    export HETGPU_XRT_BAR0_RESOURCE=/sys/bus/pci/devices/0000:64:00.1/resource0
    export HETGPU_QWEN_MODEL_SHA256="${model_sha256}"
    export HETGPU_QWEN35_CUDA_BUFFER_MAX_MIB=49152
    # Keep greedy reference and hybrid modes launch-ordered across fresh server
    # processes.  The recurrent state is explicitly cleared between batches.
    export GGML_CUDA_DISABLE_GRAPHS=1
    export CUDA_LAUNCH_BLOCKING=1
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    threads="${QWEN35_THREADS:-$(nproc)}"

    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
        > "${proof_dir}/cuda-compute-apps-before.csv"
    (
        export HETGPU_QWEN_TQ1_XRT=0
        export HETGPU_QWEN_TQ1_STRICT=0
        export HETGPU_BITNET_DISAGGREGATE=0
        export HETGPU_BITNET_DISAGG_STRICT=0
        export HETGPU_CUDART_PRELAUNCH_NAMED_KERNEL=0
        unset HETGPU_TMATMUL_BACKEND HETGPU_TMATMUL_HARDWARE_MATMUL
        unset HETGPU_BITNET_ROUTE_MANIFEST HETGPU_BITNET_ROUTE_LOG HETGPU_XRT_EXECUTION_LOG
        unset HETGPU_BITNET_GPU_KERNELS HETGPU_BITNET_CXL_KERNELS HETGPU_TQ1_EVIDENCE_LOG
        python3 "${evaluator}" \
            --mode cuda --profile "${profile}" --evidence-kind iq1s \
            --server "${llama_server}" --server-preload "${cuda13_launch_shim}:${libnvcuda}" \
            --model "${model}" --prompt-seed "${prompt_seed}" \
            --model-verification "${proof_dir}/model-verification.json" \
            --model-audit "${proof_dir}/model-tensor-audit.json" \
            --proof-dir "${proof_dir}/cuda-mode" --port 18080 --threads "${threads}" \
            --model-size "${model_size}" --model-sha256 "${model_sha256}" \
            --llama-revision "${llama_revision}" --binary-sha256 "${server_sha256}" \
            --fpga-bdf "${fpga_bdf}"
    )
    cp "${proof_dir}/cuda-mode/cuda.json" "${proof_dir}/cuda.json"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
        > "${proof_dir}/cuda-compute-apps-after.csv"

    run_hybrid_mode() {
        local trace_mode=$1
        local mode_port=$2
        local mode_dir="${proof_dir}/${trace_mode}-mode"
        (
        export HETGPU_QWEN_TQ1_XRT=0
        export HETGPU_QWEN_TQ1_STRICT=0
        export HETGPU_TMATMUL_BACKEND=xrt
        export HETGPU_BITNET_DISAGGREGATE=1
        export HETGPU_BITNET_DISAGG_STRICT=1
        export HETGPU_CUDART_PRELAUNCH_NAMED_KERNEL=1
        export HETGPU_QWEN_IQ1S_DISABLE_CUDA_FUSION=1
        export HETGPU_QWEN_IQ1S_STRICT=1
        export HETGPU_QWEN_IQ1S_PERSISTENT=1
        export HETGPU_QWEN_MODEL_CONTEXT_LIMIT=262144
        export HETGPU_IQ1S_TRACE_MODE="${trace_mode}"
        export HETGPU_LIBGGML="${verified_libggml}"
        export HETGPU_TMATMUL_HARDWARE_MATMUL=1
        export HETGPU_BITNET_ROUTE_MANIFEST="${route_manifest}"
        export HETGPU_BITNET_GPU_KERNELS=attention,attn,flash,softmax,soft_max,rope,kq,qk,qkv,query,key,value,kv_cache
        export HETGPU_BITNET_CXL_KERNELS=ggml_type19
        export HETGPU_BITNET_ROUTE_LOG="${mode_dir}/routes.jsonl"
        export HETGPU_XRT_EXECUTION_LOG="${mode_dir}/xrt.jsonl"
        # The process-global ordinal consumes this one comparison during the
        # semantic/hardware gate. Warm-up and measured batches do no shadow work.
        export HETGPU_XRT_COMPARE_MAX_LAUNCHES=1
        unset HETGPU_TQ1_EVIDENCE_LOG
        python3 "${evaluator}" \
            --mode "${trace_mode}" --profile "${profile}" --evidence-kind iq1s \
            --server "${llama_server}" --server-preload "${cuda13_launch_shim}:${libnvcuda}" \
            --model "${model}" --prompt-seed "${prompt_seed}" \
            --model-verification "${proof_dir}/model-verification.json" \
            --model-audit "${proof_dir}/model-tensor-audit.json" \
            --proof-dir "${mode_dir}" --port "${mode_port}" --threads "${threads}" \
            --model-size "${model_size}" --model-sha256 "${model_sha256}" \
            --llama-revision "${llama_revision}" --binary-sha256 "${server_sha256}" \
            --fpga-bdf "${fpga_bdf}" \
            --route-evidence "${mode_dir}/routes.jsonl" \
            --xrt-evidence "${mode_dir}/xrt.jsonl" \
            --require-routing-evidence
        )
        cp "${mode_dir}/${trace_mode}.json" "${proof_dir}/${trace_mode}.json"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
            > "${mode_dir}/cuda-compute-apps-after.csv"
    }

    run_hybrid_mode handwritten 18081
    run_hybrid_mode compiler 18082

    python3 "${iq1s_validator}" "${proof_dir}" | tee "${proof_dir}/summary.json"
    exit 0
fi

[[ $# -eq 4 ]] || {
    echo "usage: $0 MODEL BUILD_MANIFEST XCLBIN OUTPUT_DIR" >&2
    exit 2
}
model=$1
manifest=$2
xclbin=$3
proof_dir=$4

for required in "${model}" "${manifest}" "${xclbin}"; do
    [[ -f "${required}" ]] || { echo "missing evaluation input ${required}" >&2; exit 1; }
done
case "$(realpath -m "${proof_dir}")" in
    "${repo_root}"/.proof/qwen35-iq1s-au250-*) ;;
    *) echo "output directory must be ${repo_root}/.proof/qwen35-iq1s-au250-*" >&2; exit 2 ;;
esac
[[ ! -e "${proof_dir}" ]] || { echo "output directory already exists: ${proof_dir}" >&2; exit 1; }
[[ "$(realpath "${model}")" == /root/models/qwen35-tq1/Qwen3.5-397B-A17B-UD-TQ1_0.gguf ]] || {
    echo "runner requires the pinned model path" >&2
    exit 1
}
[[ "$(realpath "${manifest}")" == /root/qwen35-au250-build/manifest.json ]] || {
    echo "runner requires the pinned build manifest path" >&2
    exit 1
}
case "$(realpath "${xclbin}")" in
    /au250_xrt/*) ;;
    *) echo "xclbin must be mounted beneath /au250_xrt" >&2; exit 1 ;;
esac

proof_rel="$(realpath -m --relative-to="${repo_root}" "${proof_dir}")"
source /au250_xrt/env.sh >/dev/null
temperature="$(_au250_fpga_temp)"
[[ -z "${temperature}" || "${temperature}" -lt 85 ]] || {
    echo "AU250 temperature ${temperature}C exceeds 85C guard" >&2
    exit 1
}
cd "${repo_root}"
install -d "${proof_dir}"
HETGPU_XRT_XCLBIN="${xclbin}" QWEN35_BUILD_JOBS=32 CARGO_BUILD_JOBS=32 \
    bash "${repo_root}/zluda/tests/run_au250_xrt_iq1s.sh" --inside handwritten \
    > "${proof_dir}/standalone-handwritten.log" 2>&1
HETGPU_XRT_XCLBIN="${xclbin}" QWEN35_BUILD_JOBS=32 CARGO_BUILD_JOBS=32 \
    bash "${repo_root}/zluda/tests/run_au250_xrt_iq1s.sh" --inside compiler \
    > "${proof_dir}/standalone-compiler.log" 2>&1
"${repo_root}/tools/au250_qwen35_run.sh" bash /work/tools/run_qwen35_iq1s_au250_hybrid.sh \
    --inside "/work/${proof_rel}" "$(realpath "${xclbin}")"
python3 "${iq1s_validator}" "${proof_dir}" | tee "${proof_dir}/summary.json"
