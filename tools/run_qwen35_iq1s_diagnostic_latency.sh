#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 2 ]] || { echo "usage: $0 PROOF_DIR PORT" >&2; exit 2; }
proof_dir=$1
port=$2
case ${proof_dir} in
    /mnt/disk0/qwen397b-proof/*) ;;
    *) echo "diagnostic directory must be beneath /mnt/disk0/qwen397b-proof" >&2; exit 2 ;;
esac

model=/models/qwen/Qwen3.5-397B-A17B-UD-TQ1_0.gguf
server=/qwen-build/llama-build/bin/llama-server
libnvcuda=/qwen-build/hetgpu-target/release/libnvcuda.so
launch_shim=/qwen-build/hetgpu-target/release/libqwen35_cuda13_launch_shim.so
xclbin=/au250_xrt/xclbins/qwen397b_iq1s_layer_persistent_9c83dcae.xclbin
route_manifest=/work/tools/qwen35-iq1s-route-manifest.json
prompt_seed=/work/zluda/evaluation/fixtures/qwen35_prompt_seed.txt

install -d "${proof_dir}" "${proof_dir}/slot-state"
test "$(sha256sum "${xclbin}" | awk '{print $1}')" = \
    9c83dcae07b4c7bf1d2e1cebf46ccf0ff1ebf8848a437035fef1451dee7770a3

export LD_LIBRARY_PATH="/qwen-build/llama-build/bin:/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"
export GGML_CUDA_DISABLE_GRAPHS=1
unset CUDA_LAUNCH_BLOCKING
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HETGPU_QWEN_MODEL_SHA256=0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568
export HETGPU_QWEN35_CUDA_BUFFER_MAX_MIB=49152
export HETGPU_XRT_XCLBIN=${xclbin}
export HETGPU_XRT_TIMEOUT_MS=10000
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
export HETGPU_QWEN_IQ1S_SESSION_GENERATION=1
export HETGPU_QWEN_IQ1S_RING_CAPACITY=512
export HETGPU_QWEN_IQ1S_PROOF_LEDGER=${proof_dir}/phase-ledger.jsonl
export HETGPU_QWEN_IQ1S_PROGRESS_LOG=${proof_dir}/progress.jsonl
export HETGPU_QWEN_IQ1S_PROOF_SYNC_PHASE=0
export HETGPU_QWEN_IQ1S_DIAGNOSTIC_FINITE_ONLY=1
export HETGPU_QWEN_MODEL_CONTEXT_LIMIT=262144
export HETGPU_IQ1S_TRACE_MODE=handwritten
export HETGPU_LIBGGML=/qwen-build/llama-build/bin/libggml.so.0.22.0
export HETGPU_BITNET_ROUTE_MANIFEST=${route_manifest}
export HETGPU_BITNET_GPU_KERNELS=attention,attn,flash,softmax,soft_max,rope,kq,qk,qkv,query,key,value,kv_cache
export HETGPU_BITNET_CXL_KERNELS=ggml_type19
export HETGPU_BITNET_ROUTE_LOG=${proof_dir}/routes.jsonl
unset HETGPU_XRT_EXECUTION_LOG HETGPU_TQ1_EVIDENCE_LOG

env LD_PRELOAD="${launch_shim}:${libnvcuda}" \
    "${server}" --model "${model}" --ctx-size 512 --n-gpu-layers 999 \
    --threads 32 --host 127.0.0.1 --port "${port}" --seed 42 --parallel 1 \
    --reasoning off --verbosity 4 --batch-size 16 --ubatch-size 16 --cache-ram 0 \
    --timeout 7200 \
    --flash-attn on --no-cache-prompt --slot-save-path "${proof_dir}/slot-state" \
    --no-warmup --no-webui >"${proof_dir}/server.stdout.log" \
    2>"${proof_dir}/server.stderr.log" &
server_pid=$!
cleanup() {
    kill -TERM "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT

python3 - "${proof_dir}" "${port}" "${server_pid}" "${prompt_seed}" <<'PY'
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, "/work/tools")
from qwen35_au250_eval import (
    EvaluationError,
    atomic_json,
    completion_request,
    erase_all_slots,
    exact_prompt,
    stream_completion,
)

proof_dir = Path(sys.argv[1])
port = int(sys.argv[2])
pid = int(sys.argv[3])
prompt_seed = sys.argv[4]
base_url = f"http://127.0.0.1:{port}"
deadline = time.monotonic() + 900
last_error = None
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except ProcessLookupError as error:
        raise SystemExit("llama-server exited during diagnostic startup") from error
    try:
        with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
            body = json.load(response)
        if body.get("status") in ("ok", "no slot available"):
            break
    except Exception as error:
        last_error = error
    time.sleep(0.25)
else:
    raise SystemExit(f"llama-server did not become healthy: {last_error}")

prompt_text, prompt_ids = exact_prompt(base_url, prompt_seed, 3600, 16)
atomic_json(proof_dir / "prompt-token-ids.json", prompt_ids)
(proof_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")

warmup_error = None
try:
    warmup = stream_completion(
        base_url,
        completion_request(prompt_ids, 1),
        7200,
        proof_dir / "warmup-stream-events.json",
    )
except EvaluationError as error:
    # Arena initialization can outlive the first HTTP stream.  This is a
    # diagnostic warm-up only: retain the server, wait for the runtime and
    # slot to become ready, and never treat the disconnected request as a
    # measured or proof-eligible token.
    warmup = None
    warmup_error = str(error)
    atomic_json(proof_dir / "warmup-disconnect.json", {
        "schema_version": 1,
        "status": "diagnostic_nonproof",
        "error": warmup_error,
        "server_alive": True,
    })

if warmup is None:
    deadline = time.monotonic() + 3600
    progress_path = proof_dir / "progress.jsonl"
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError as error:
            raise SystemExit("llama-server exited while completing U250 arena initialization") from error
        progress = progress_path.read_text(encoding="utf-8") if progress_path.exists() else ""
        if '"stage":"pool_ready"' in progress:
            break
        time.sleep(1)
    else:
        raise SystemExit("U250 arena did not reach pool_ready after warm-up disconnect")

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
                body = json.load(response)
            if body.get("status") == "ok":
                break
        except Exception:
            pass
        time.sleep(0.25)
    else:
        raise SystemExit("llama-server slot did not become idle after U250 arena initialization")

erase_all_slots(base_url, 3600, 1)
measurement = stream_completion(
    base_url,
    completion_request(prompt_ids, 1),
    7200,
    proof_dir / "measurement-stream-events.json",
)

record = {
    "schema_version": 1,
    "status": "diagnostic_nonproof",
    "reason": "sampled libggml mismatch explicitly accepted for timing only",
    "strict_proof_eligible": False,
    "mode": "handwritten_gpu_attention_u250_ffn",
    "profile": {"request_count": 1, "prompt_tokens": 16, "generated_tokens": 1},
    "warmup": None if warmup is None else {
        "token_ids": warmup["token_ids"],
        "ttft_ms": warmup["ttft_ms"],
        "end_to_end_ms": warmup["end_to_end_ms"],
    },
    "warmup_disconnect": warmup_error,
    "measurement": {
        "token_ids": measurement["token_ids"],
        "ttft_ms": measurement["ttft_ms"],
        "end_to_end_ms": measurement["end_to_end_ms"],
        "timings": measurement["timings"],
    },
}
atomic_json(proof_dir / "diagnostic-latency.json", record)
print(json.dumps(record, sort_keys=True))
PY
