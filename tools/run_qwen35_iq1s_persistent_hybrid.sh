#!/usr/bin/env bash
# shellcheck disable=SC2030,SC2031
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
model_sha256=0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568
model_size=94155830880
llama_revision=925e1179947ea0c0ebfb0032df18af3a729822be
device_bdf=0000:64:00.1

if [[ ${1:-} == --inside ]]; then
    [[ $# -eq 7 || $# -eq 8 ]] || {
        echo "usage: $0 --inside PROFILE EXECUTION_KIND PROOF_DIR XCLBIN XCLBIN_SHA256 ACCEPTED_GATE [SCHEDULER]" >&2
        exit 2
    }
    profile=$2
    execution_kind=$3
    proof_dir=$4
    xclbin=$5
    expected_xclbin_sha256=$6
    accepted_gate=$7
    scheduler=${8:-waves}
    case ${scheduler} in waves|slot-refill) ;; *) echo "invalid scheduler: ${scheduler}" >&2; exit 2;; esac
    [[ ${profile} == one-token || ${profile} == full ]] || { echo "inside profile must be one-token or full" >&2; exit 2; }
    [[ ${execution_kind} == correctness || ${execution_kind} == performance ]] || {
        echo "inside execution kind must be correctness or performance" >&2; exit 2;
    }
    if [[ ${execution_kind} == correctness ]]; then
        [[ ${profile} == one-token && -z ${accepted_gate} ]] || {
            echo "correctness execution requires one-token and no accepted gate" >&2; exit 2;
        }
    else
        [[ ${profile} == full && -d ${accepted_gate} ]] || {
            echo "performance execution requires full and an accepted gate" >&2; exit 2;
        }
    fi
    case ${proof_dir} in
        /mnt/disk0/qwen397b-proof/*) ;;
        *) echo "proof directory must be beneath /mnt/disk0/qwen397b-proof" >&2; exit 2 ;;
    esac

    model=/models/qwen/Qwen3.5-397B-A17B-UD-TQ1_0.gguf
    manifest=/qwen-build/manifest.json
    server=/qwen-build/llama-build/bin/llama-server
    libnvcuda=/qwen-build/hetgpu-target/release/libnvcuda.so
    launch_shim=/qwen-build/hetgpu-target/release/libqwen35_cuda13_launch_shim.so
    evaluator=/work/tools/qwen35_au250_eval.py
    auditor=/work/tools/qwen35_gguf_audit.py
    build_preflight=/work/tools/qwen35_build_preflight.py
    route_manifest=/work/tools/qwen35-iq1s-route-manifest.json
    prompt_seed=/work/zluda/evaluation/fixtures/qwen35_prompt_seed.txt
    validator=/work/zluda/tests/validate_qwen35_iq1s_persistent_gate.py
    threads=${QWEN35_THREADS:-32}
    if [[ ! ${threads} =~ ^[0-9]+$ ]] || (( threads < 1 || threads > 32 )); then
        echo "QWEN35_THREADS must be an integer from 1 through 32" >&2
        exit 2
    fi

    for required in "${model}" "${manifest}" "${server}" "${libnvcuda}" "${launch_shim}" \
        "${evaluator}" "${auditor}" "${build_preflight}" "${route_manifest}" \
        "${prompt_seed}" "${validator}" "${xclbin}"; do
        [[ -f ${required} ]] || { echo "missing persistent E2E input ${required}" >&2; exit 1; }
    done
    [[ $(stat -c %s "${model}") == "${model_size}" ]] || { echo "model size mismatch" >&2; exit 1; }
    actual_model_sha256=$(sha256sum "${model}" | awk '{print $1}')
    [[ ${actual_model_sha256} == "${model_sha256}" ]] || { echo "model SHA-256 mismatch" >&2; exit 1; }
    actual_xclbin_sha256=$(sha256sum "${xclbin}" | awk '{print $1}')
    [[ ${actual_xclbin_sha256} == "${expected_xclbin_sha256}" ]] || {
        echo "persistent xclbin SHA-256 changed before E2E" >&2
        exit 1
    }

    install -d "${proof_dir}"
    MODEL=${model} MODEL_SHA256=${model_sha256} OUTPUT=${proof_dir}/model-verification.json \
        python3 - <<'PY'
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
temporary = output.with_suffix(".json.partial")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
    python3 "${auditor}" "${model}" \
        --model-verification "${proof_dir}/model-verification.json" \
        --output "${proof_dir}/model-tensor-audit.json"
    python3 "${build_preflight}" \
        --manifest "${manifest}" --build-root /qwen-build \
        --llama-revision "${llama_revision}" \
        --output "${proof_dir}/qwen-build-preflight.json"
    MANIFEST=${manifest} SERVER=${server} LIBNVCUDA=${libnvcuda} LAUNCH_SHIM=${launch_shim} \
        LLAMA_REVISION=${llama_revision} python3 - <<'PY'
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
for name, variable in (
    ("llama_server", "SERVER"),
    ("libnvcuda", "LIBNVCUDA"),
    ("cuda13_launch_shim", "LAUNCH_SHIM"),
):
    artifact = manifest.get("artifacts", {}).get(name, {})
    path = os.environ[variable]
    if artifact.get("path") != path or artifact.get("sha256") != digest(path):
        raise SystemExit(f"build manifest artifact mismatch: {name}")
PY
    verified_libggml=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["libggml_path"])' \
        "${proof_dir}/qwen-build-preflight.json")
    server_sha256=$(sha256sum "${server}" | awk '{print $1}')
    route_manifest_sha256=$(sha256sum "${route_manifest}" | awk '{print $1}')
    tensor_audit_sha256=$(sha256sum "${proof_dir}/model-tensor-audit.json" | awk '{print $1}')

    if [[ ${execution_kind} == performance ]]; then
        python3 "${validator}" --gate g4 "${accepted_gate}" >/dev/null
        accepted_hashes=${accepted_gate}/artifact-hashes.txt
        [[ -f ${accepted_hashes} ]] || { echo "accepted gate omitted artifact hashes" >&2; exit 1; }
        require_accepted_hash() {
            local name=$1
            local expected=$2
            local actual
            actual=$(sed -n "s/^${name}=//p" "${accepted_hashes}")
            [[ -n ${actual} && ${actual} == "${expected}" ]] || {
                echo "accepted gate ${name} mismatch" >&2
                exit 1
            }
        }
        require_accepted_hash model_sha256 "${model_sha256}"
        require_accepted_hash llama_server_sha256 "${server_sha256}"
        require_accepted_hash xclbin_sha256 "${actual_xclbin_sha256}"
        require_accepted_hash route_manifest_sha256 "${route_manifest_sha256}"
        require_accepted_hash tensor_audit_sha256 "${tensor_audit_sha256}"
    fi

    xclbinutil --info --input "${xclbin}" > "${proof_dir}/xclbin-info.txt" 2>&1
    for cu in iq1s_layer_big_1 iq1s_layer_big_2 iq1s_layer_big_3 iq1s_layer_small_1; do
        grep -Fq "Instance:        ${cu}" "${proof_dir}/xclbin-info.txt" || {
            echo "persistent xclbin is missing ${cu}" >&2
            exit 1
        }
    done
    xclbin_uuid=$(sed -n 's/^[[:space:]]*UUID (xclbin):[[:space:]]*//p' "${proof_dir}/xclbin-info.txt" | head -n1)
    [[ ${xclbin_uuid} =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] || {
        echo "persistent xclbin UUID is absent or malformed" >&2
        exit 1
    }

    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits \
        > "${proof_dir}/nvidia-memory-preflight.csv"
    MODEL_SIZE=${model_size} python3 - "${proof_dir}/nvidia-memory-preflight.csv" <<'PY'
import os
from pathlib import Path
import sys

free_mib = sum(
    int(line.rsplit(",", 1)[-1].strip())
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.strip()
)
required = int(os.environ["MODEL_SIZE"]) + 2 * 1024**3
if free_mib * 1024**2 < required:
    raise SystemExit(
        f"insufficient aggregate free GPU memory: {free_mib} MiB, need {required} bytes"
    )
PY
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits \
        > "${proof_dir}/cuda-compute-apps-before.csv"
    [[ ! -s ${proof_dir}/cuda-compute-apps-before.csv ]] || {
        echo "refusing persistent E2E while another CUDA compute process is active" >&2
        exit 1
    }
    xbutil examine -d "${device_bdf}" -r dynamic-regions -r error -r firewall -r thermal \
        > "${proof_dir}/xbutil-preflight.txt" 2>&1
    lspci -s "${device_bdf}" -vv > "${proof_dir}/pcie-link.txt"
    grep -Fq 'Level 0 : 0x0 (GOOD)' "${proof_dir}/xbutil-preflight.txt" || {
        echo "AU250 firewall is not GOOD before persistent E2E" >&2
        exit 1
    }
    if grep -Eiq '(^|[^[:alpha:]])fatal([^[:alpha:]]|$)' "${proof_dir}/xbutil-preflight.txt"; then
        echo "AU250 reports a fatal error before persistent E2E" >&2
        exit 1
    fi
    {
        printf 'model_sha256=%s\n' "${model_sha256}"
        printf 'llama_server_sha256=%s\n' "${server_sha256}"
        printf 'libnvcuda_sha256=%s\n' "$(sha256sum "${libnvcuda}" | awk '{print $1}')"
        printf 'launch_shim_sha256=%s\n' "$(sha256sum "${launch_shim}" | awk '{print $1}')"
        printf 'xclbin_sha256=%s\n' "${actual_xclbin_sha256}"
        printf 'xclbin_uuid=%s\n' "${xclbin_uuid}"
        printf 'route_manifest_sha256=%s\n' "${route_manifest_sha256}"
        printf 'tensor_audit_sha256=%s\n' "${tensor_audit_sha256}"
        printf 'build_threads=%s\n' "${QWEN35_BUILD_JOBS:-32}"
        printf 'runtime_threads=%s\n' "${threads}"
    } > "${proof_dir}/artifact-hashes.txt"

    export LD_LIBRARY_PATH="/qwen-build/llama-build/bin:/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"
    export GGML_CUDA_DISABLE_GRAPHS=1
    if [[ ${execution_kind} == correctness ]]; then
        export CUDA_LAUNCH_BLOCKING=1
    else
        unset CUDA_LAUNCH_BLOCKING
    fi
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export HETGPU_QWEN_MODEL_SHA256=${model_sha256}
    export HETGPU_QWEN35_CUDA_BUFFER_MAX_MIB=49152
    export HETGPU_XRT_XCLBIN=${xclbin}
    export HETGPU_XRT_TIMEOUT_MS=10000

    (
        export HETGPU_QWEN_IQ1S_PERSISTENT=0
        export HETGPU_QWEN_IQ1S_STRICT=0
        export HETGPU_BITNET_DISAGGREGATE=0
        export HETGPU_BITNET_DISAGG_STRICT=0
        export HETGPU_CUDART_PRELAUNCH_NAMED_KERNEL=0
        unset HETGPU_TMATMUL_BACKEND HETGPU_TMATMUL_HARDWARE_MATMUL
        unset HETGPU_BITNET_ROUTE_MANIFEST HETGPU_BITNET_ROUTE_LOG
        unset HETGPU_QWEN_IQ1S_PROOF_LEDGER HETGPU_QWEN_IQ1S_PROGRESS_LOG
        unset HETGPU_QWEN_IQ1S_SESSION_GENERATION
        python3 "${evaluator}" \
            --mode cuda --profile "${profile}" --scheduler "${scheduler}" --evidence-kind iq1s \
            --server "${server}" --server-preload "${launch_shim}:${libnvcuda}" \
            --model "${model}" --prompt-seed "${prompt_seed}" \
            --model-verification "${proof_dir}/model-verification.json" \
            --model-audit "${proof_dir}/model-tensor-audit.json" \
            --proof-dir "${proof_dir}/cuda-mode" --port 18100 --threads "${threads}" \
            --model-size "${model_size}" --model-sha256 "${model_sha256}" \
            --llama-revision "${llama_revision}" --binary-sha256 "${server_sha256}" \
            --fpga-bdf "${device_bdf}"
    )

    run_hybrid_mode() {
        local trace_mode=$1
        local port=$2
        local generation=$3
        local mode_dir=${proof_dir}/${trace_mode}-mode
        (
            export HETGPU_QWEN_TQ1_XRT=0
            export HETGPU_QWEN_TQ1_STRICT=0
            export HETGPU_TMATMUL_BACKEND=xrt
            export HETGPU_TMATMUL_HARDWARE_MATMUL=1
            export HETGPU_BITNET_DISAGGREGATE=1
            export HETGPU_BITNET_DISAGG_STRICT=1
            export HETGPU_CUDART_PRELAUNCH_NAMED_KERNEL=1
            export HETGPU_QWEN_IQ1S_DISABLE_CUDA_FUSION=1
            export HETGPU_QWEN_IQ1S_STRICT=1
            export HETGPU_QWEN_IQ1S_PERSISTENT=1
            export HETGPU_QWEN_IQ1S_SESSION_GENERATION=${generation}
            export HETGPU_QWEN_IQ1S_RING_CAPACITY=512
            export HETGPU_QWEN_IQ1S_PROOF_LEDGER=${mode_dir}/phase-ledger.jsonl
            export HETGPU_QWEN_IQ1S_PROGRESS_LOG=${mode_dir}/progress.jsonl
            if [[ ${execution_kind} == correctness ]]; then
                export HETGPU_QWEN_IQ1S_PROOF_SYNC_PHASE=1
            else
                export HETGPU_QWEN_IQ1S_PROOF_SYNC_PHASE=0
            fi
            export HETGPU_QWEN_MODEL_CONTEXT_LIMIT=262144
            export HETGPU_IQ1S_TRACE_MODE=${trace_mode}
            export HETGPU_LIBGGML=${verified_libggml}
            export HETGPU_BITNET_ROUTE_MANIFEST=${route_manifest}
            export HETGPU_BITNET_GPU_KERNELS=attention,attn,flash,softmax,soft_max,rope,kq,qk,qkv,query,key,value,kv_cache
            export HETGPU_BITNET_CXL_KERNELS=ggml_type19
            export HETGPU_BITNET_ROUTE_LOG=${mode_dir}/routes.jsonl
            unset HETGPU_XRT_EXECUTION_LOG HETGPU_TQ1_EVIDENCE_LOG
            python3 "${evaluator}" \
                --mode "${trace_mode}" --profile "${profile}" --scheduler "${scheduler}" --evidence-kind iq1s \
                --server "${server}" --server-preload "${launch_shim}:${libnvcuda}" \
                --model "${model}" --prompt-seed "${prompt_seed}" \
                --model-verification "${proof_dir}/model-verification.json" \
                --model-audit "${proof_dir}/model-tensor-audit.json" \
                --proof-dir "${mode_dir}" --port "${port}" --threads "${threads}" \
                --model-size "${model_size}" --model-sha256 "${model_sha256}" \
                --llama-revision "${llama_revision}" --binary-sha256 "${server_sha256}" \
                --fpga-bdf "${device_bdf}" --route-evidence "${mode_dir}/routes.jsonl" \
                --persistent-ledger "${mode_dir}/phase-ledger.jsonl" \
                --require-routing-evidence
        )
    }

    run_hybrid_mode handwritten 18101 1
    run_hybrid_mode compiler 18102 2

    if [[ ${execution_kind} == performance ]]; then
        PROOF_DIR=${proof_dir} python3 - <<'PY'
import json
from pathlib import Path
import os
import sys

sys.path.insert(0, "/work/tools")
from qwen35_au250_eval import enforce_aggregate_target
sys.path.insert(0, "/work/zluda/tests")
from validate_qwen35_iq1s_au250_proof import _validate_refill_round

root = Path(os.environ["PROOF_DIR"])
target_tps = 28.0
max_wall_seconds = 2048.0 / target_tps
records = {
    "cuda": json.loads((root / "cuda-mode" / "cuda.json").read_text()),
    "handwritten": json.loads((root / "handwritten-mode" / "handwritten.json").read_text()),
    "compiler": json.loads((root / "compiler-mode" / "compiler.json").read_text()),
}
reference = records["cuda"]["generated_token_ids_by_request"]
summary = {
    "schema_version": 1,
    "kind": "iq1s_persistent_aggregate_performance",
    "profile": {"request_count": 64, "max_active": 32, "tokens_per_request": 32},
    "scheduler": records["cuda"].get("scheduler", "waves"),
    "status": "pass",
    "modes": {},
}
for mode, record in records.items():
    scheduler = record.get("scheduler", "waves")
    if scheduler != summary["scheduler"]:
        raise SystemExit(f"{mode} scheduler differs from CUDA")
    if scheduler == "slot-refill":
        rounds = record.get("scheduler_evidence", [])
        if len(rounds) != 4:
            raise SystemExit(f"{mode} missing refill warmup/measurement evidence")
        for index, evidence in enumerate(rounds):
            _validate_refill_round(evidence, 64, 32, f"{mode}.refill[{index}]")
            if index > 0 and evidence["wall_seconds"] != record["measurements"][index - 1]["measured_wall_seconds"]:
                raise SystemExit(f"{mode} refill wall time does not match measurement")
    if record["generated_token_ids_by_request"] != reference:
        raise SystemExit(f"{mode} full-run token IDs differ from CUDA")
    measurements = record["measurements"]
    if len(measurements) != 3:
        raise SystemExit(f"{mode} full run omitted three measurements")
    tps = [item["generation_tokens_per_second"] for item in measurements]
    if min(tps) < target_tps:
        raise SystemExit(f"{mode} full run is below the aggregate TPS target")
    expected_waves = 0 if scheduler == "slot-refill" else 2
    if any(item["generated_tokens"] != 2048 or item["wave_count"] != expected_waves
           or item.get("scheduler", "waves") != scheduler for item in measurements):
        raise SystemExit(f"{mode} did not execute the fixed 64x32 workload")
    mode_summary = enforce_aggregate_target(measurements, target_tps)
    if mode_summary["max_wall_seconds"] != max_wall_seconds:
        raise SystemExit(f"{mode} aggregate wall-time budget drifted")
    mode_summary["generated_tokens"] = 2048
    summary["modes"][mode] = mode_summary
for mode in ("handwritten", "compiler"):
    ledger = [
        json.loads(line)
        for line in (root / f"{mode}-mode" / "phase-ledger.jsonl").read_text().splitlines()
    ]
    if not ledger or sum(item["weight_dma_bytes"] for item in ledger) != 0:
        raise SystemExit(f"{mode} ledger is empty or contains measured weight DMA")
(root / "aggregate-throughput.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
        exit 0
    fi

    PROOF_DIR=${proof_dir} MODEL_SHA256=${model_sha256} XCLBIN_SHA256=${actual_xclbin_sha256} \
        XCLBIN_UUID=${xclbin_uuid} python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PROOF_DIR"])
fixed_routing = {
    "u250_iq1s_tensors": 141,
    "gpu_non_iq1s_tensors": 39,
    "u250_non_iq1s_tensors": 0,
    "attention": "gpu",
}

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def latency(record):
    values = record.get("measurements")
    if not isinstance(values, list) or len(values) != 1:
        raise SystemExit("one-token evaluator must emit exactly one measurement")
    value = values[0].get("end_to_end_ms")
    if not isinstance(value, (int, float)) or value <= 0:
        raise SystemExit("one-token evaluator emitted invalid E2E latency")
    return float(value)

cuda_record = read(root / "cuda-mode" / "cuda.json")
cuda_tokens = cuda_record["generated_token_ids"]
aggregate = {
    "schema_version": 1,
    "kind": "iq1s_persistent_g4",
    "profile": "one-token",
    "model_sha256": os.environ["MODEL_SHA256"],
    "xclbin_sha256": os.environ["XCLBIN_SHA256"],
    "xclbin_uuid": os.environ["XCLBIN_UUID"],
    "cuda": {
        "generated_tokens": len(cuda_tokens),
        "token_ids": cuda_tokens,
        "e2e_latency_ms": latency(cuda_record),
    },
}
for mode in ("handwritten", "compiler"):
    mode_root = root / f"{mode}-mode"
    record = read(mode_root / f"{mode}.json")
    tokens = record["generated_token_ids"]
    xrt = record["xrt"]
    routes = record["routes"]
    ledger = [json.loads(line) for line in (mode_root / "phase-ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    weight_dma = sum(item["weight_dma_bytes"] for item in ledger)
    mode_summary = {
        "schema_version": 1,
        "kind": "iq1s_persistent_summary",
        "profile": {
            "name": "one-token", "request_count": 1, "max_active": 1,
            "tokens_per_request": 1, "measurements": 1, "warmups": 0,
        },
        "mode": mode,
        "routing": fixed_routing,
        "fallbacks": routes["fallback"],
        "token_ids": tokens,
        "cuda_token_ids": cuda_tokens,
    }
    (mode_root / "summary.json").write_text(
        json.dumps(mode_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    aggregate[mode] = {
        "generated_tokens": len(tokens),
        "token_ids": tokens,
        "routing": fixed_routing,
        "eligible_direct_routes": 0,
        "fallbacks": routes["fallback"],
        "measured_weight_dma_bytes": weight_dma,
        "completions_per_cu": xrt["per_cu_completions"],
        "e2e_latency_ms": latency(record),
    }
(root / "g4-summary.json").write_text(
    json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
    python3 "${validator}" --gate ledger "${proof_dir}/handwritten-mode"
    python3 "${validator}" --gate ledger "${proof_dir}/compiler-mode"
    python3 "${validator}" --gate g4 "${proof_dir}" | tee "${proof_dir}/validation.json"
    exit 0
fi

gate=
performance=
accepted_gate=
model=
manifest=
xclbin=
proof_dir=
selected_scheduler=${HETGPU_QWEN_SCHEDULER:-slot-refill}
while [[ $# -gt 0 ]]; do
    case $1 in
        --gate) gate=${2:-}; shift 2 ;;
        --performance) performance=${2:-}; shift 2 ;;
        --accepted-gate) accepted_gate=${2:-}; shift 2 ;;
        --model) model=${2:-}; shift 2 ;;
        --manifest) manifest=${2:-}; shift 2 ;;
        --xclbin) xclbin=${2:-}; shift 2 ;;
        --proof-dir) proof_dir=${2:-}; shift 2 ;;
        --scheduler) selected_scheduler=${2:-}; shift 2 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done
case ${selected_scheduler} in waves|slot-refill) ;; *) echo "invalid scheduler: ${selected_scheduler}" >&2; exit 2;; esac
[[ ${gate} == one-token ]] || { echo "--gate must be one-token" >&2; exit 2; }
if [[ -n ${performance} ]]; then
    [[ ${performance} == full ]] || { echo "--performance must be full" >&2; exit 2; }
    [[ -n ${accepted_gate} ]] || { echo "--performance full requires --accepted-gate" >&2; exit 2; }
else
    [[ -z ${accepted_gate} ]] || { echo "--accepted-gate requires --performance full" >&2; exit 2; }
fi
[[ -n ${model} && -n ${manifest} && -n ${xclbin} && -n ${proof_dir} ]] || {
    echo "usage: $0 --gate one-token [--performance full --accepted-gate DIR --scheduler waves|slot-refill] --model MODEL --manifest MANIFEST --xclbin XCLBIN --proof-dir DIR" >&2
    exit 2
}
[[ $(realpath -e "${model}") == /root/models/qwen35-tq1/Qwen3.5-397B-A17B-UD-TQ1_0.gguf ]] || {
    echo "runner requires the pinned Qwen model" >&2; exit 1;
}
[[ $(realpath -e "${manifest}") == /root/qwen35-au250-build/manifest.json ]] || {
    echo "runner requires the pinned Qwen build manifest" >&2; exit 1;
}
case $(realpath -e "${xclbin}") in
    /au250_xrt/xclbins/qwen397b_iq1s_layer_persistent_*.xclbin) ;;
    *) echo "runner requires a SHA-named installed persistent xclbin" >&2; exit 1 ;;
esac
proof_dir=$(realpath -m "${proof_dir}")
if [[ -n ${accepted_gate} ]]; then
    accepted_gate=$(realpath -e "${accepted_gate}")
    case ${accepted_gate} in
        /mnt/disk0/qwen397b-proof/*) ;;
        *) echo "accepted gate must be beneath /mnt/disk0/qwen397b-proof" >&2; exit 2 ;;
    esac
fi
case ${proof_dir} in
    /mnt/disk0/qwen397b-proof/*) ;;
    *) echo "proof directory must be beneath /mnt/disk0/qwen397b-proof" >&2; exit 2 ;;
esac
[[ ! -e ${proof_dir} ]] || { echo "refusing to reuse proof directory ${proof_dir}" >&2; exit 1; }
install -d "${proof_dir}"
xclbin_sha256=$(sha256sum "${xclbin}" | awk '{print $1}')

source /au250_xrt/env.sh >/dev/null
temperature=$(_au250_fpga_temp)
[[ -z ${temperature} || ${temperature} -lt ${AU250_TEMP_LIMIT:-85} ]] || {
    echo "AU250 temperature ${temperature}C exceeds launch guard" >&2
    exit 1
}
if [[ ${performance} == full ]]; then
    execution_kind=performance
    profile=full
    scheduler=${selected_scheduler}
else
    execution_kind=correctness
    profile=one-token
    scheduler=waves
fi
AU250_QWEN_PROOF_ROOT=${proof_dir} QWEN35_BUILD_JOBS=32 CARGO_BUILD_JOBS=32 \
    "${repo_root}/tools/au250_qwen35_run.sh" bash /work/tools/run_qwen35_iq1s_persistent_hybrid.sh \
    --inside "${profile}" "${execution_kind}" "${proof_dir}" "$(realpath -e "${xclbin}")" \
    "${xclbin_sha256}" "${accepted_gate}" "${scheduler}"
