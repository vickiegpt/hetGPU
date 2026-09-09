#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
wrapper="${repo_root}/tools/au250_qwen35_run.sh"
builder="${repo_root}/tools/build_au250_qwen35_runtime.sh"
cublas_shim="${repo_root}/zluda/src/cublas_shim.c"
runner="${repo_root}/tools/run_qwen35_tq1_au250_hybrid.sh"
evaluator="${repo_root}/tools/qwen35_au250_eval.py"
validator="${repo_root}/zluda/tests/validate_qwen35_tq1_au250_proof.py"
iq1s_runner="${repo_root}/tools/run_qwen35_iq1s_au250_hybrid.sh"
iq1s_validator="${repo_root}/zluda/tests/validate_qwen35_iq1s_au250_proof.py"
cuda13_launch_shim="${repo_root}/tools/qwen35_cuda13_launch_shim.c"
cuda13_launch_map="${repo_root}/tools/qwen35_cuda13_launch_shim.map"
cuda13_launch_shim_test="${repo_root}/zluda/tests/qwen35_cuda13_launch_shim_test.c"
build_preflight="${repo_root}/tools/qwen35_build_preflight.py"
function_rs="${repo_root}/zluda/src/impl/function.rs"
iq1s_standalone="${repo_root}/zluda/tests/run_au250_xrt_iq1s.sh"

test -x "${wrapper}"
test -x "${builder}"
test -x "${runner}"
test -x "${evaluator}"
test -x "${iq1s_runner}"
test -x "${iq1s_validator}"
test -f "${build_preflight}"
bash -n "${wrapper}"
bash -n "${builder}"
bash -n "${runner}"
bash -n "${iq1s_runner}"
grep -Fq 'profile=one-token' "${iq1s_runner}"
test "$(grep -Fc -- '--profile "${profile}"' "${iq1s_runner}")" -eq 2
grep -Fq '#define _GNU_SOURCE' "${cublas_shim}"

grep -Fq 'AU250_QWEN_MODEL_ROOT:-/root/models/qwen35-tq1' "${wrapper}"
grep -Fq 'AU250_QWEN_LLAMA_ROOT:-/tmp/llama.cpp-qwen-context' "${wrapper}"
grep -Fq 'AU250_CUDA_ROOT:-/usr/local/cuda-13.0' "${wrapper}"
grep -Fq -- '--git-common-dir' "${wrapper}"
grep -Fq -- '-v "${git_common_dir}":"${git_common_dir}":ro' "${wrapper}"
grep -Fq '_au250_devflags' "${wrapper}"
grep -Fq '_au250_fpga_temp' "${wrapper}"
grep -Fq '/models/qwen:ro' "${wrapper}"
grep -Fq '/llama-pristine:ro' "${wrapper}"
grep -Fq '/usr/local/cuda-13.0:ro' "${wrapper}"
grep -Fq 'cuda-compat' "${wrapper}"
grep -Fq 'cuda_compat_math_header' "${wrapper}"
grep -Fq '/usr/local/cuda-13.0/targets/x86_64-linux/include/crt/math_functions.h:ro' "${wrapper}"
grep -Fq 'double rsqrt(double a) noexcept(true)' "${wrapper}"
grep -Fq 'float rsqrtf(float a) noexcept(true)' "${wrapper}"
grep -Fq 'cmp -s' "${wrapper}"
grep -Fq '/au250_xrt:ro' "${wrapper}"
grep -Fq ':/qwen-build' "${wrapper}"
grep -Fq 'AU250_QWEN_PROOF_ROOT' "${wrapper}"
grep -Fq 'proof root must be beneath /mnt/disk0' "${wrapper}"
grep -Fq 'proof_mount_args=(-v "${proof_root}:${proof_root}")' "${wrapper}"

grep -Fq '925e1179947ea0c0ebfb0032df18af3a729822be' "${builder}"
grep -Fq 'prepare_au250_qwen35_source.sh' "${builder}"
grep -Fq 'QWEN35_BUILD_JOBS:-32' "${builder}"
grep -Fq 'CARGO_BUILD_JOBS="${build_jobs}"' "${builder}"
grep -Fq 'llama_memory_clear(llama_get_memory(ctx_tgt), true)' "${repo_root}/tools/llama-qwen35-tq1-hetgpu.patch"
grep -Fq 'HETGPU Qwen IQ1_S: using verified full sequence removal mode' "${repo_root}/tools/llama-qwen35-tq1-hetgpu.patch"
grep -Fq -- '-j"${build_jobs}"' "${builder}"
grep -Fq 'must be an integer from 1 through 32' "${builder}"
cargo_path_line="$(grep -nF 'export PATH="${CARGO_HOME}/bin:${PATH}"' "${builder}" | head -1 | cut -d: -f1)"
cargo_check_line="$(grep -nF 'command -v cargo' "${builder}" | head -1 | cut -d: -f1)"
test "${cargo_path_line}" -lt "${cargo_check_line}"
grep -Fq -- '-DGGML_CUDA=ON' "${builder}"
grep -Fq -- '-DGGML_CUDA_F16=ON' "${builder}"
grep -Fq -- '-DCMAKE_CUDA_ARCHITECTURES=120' "${builder}"
grep -Fq -- '-DLLAMA_BUILD_SERVER=ON' "${builder}"
grep -Fq -- '--target llama-server llama-cli' "${builder}"
grep -Fq -- '--features nvidia,embed_cudart,evaluation' "${builder}"
grep -Fq 'qwen35_cuda13_launch_shim.c' "${builder}"
grep -Fq 'qwen35_cuda13_launch_shim.map' "${builder}"
grep -Fq 'libqwen35_cuda13_launch_shim.so' "${builder}"
grep -Fq 'libcudart\.so\.13' "${builder}"
grep -Fq '__cudaRegisterFunction' "${builder}"
grep -Fq 'hetgpu_tq1_register_tensor_v1' "${builder}"
grep -Fq 'hetgpu_tq1_try_mul_mat_id_v1' "${builder}"
grep -Fq 'hetgpu_tq1_evaluate_raw_v1' "${builder}"
grep -Fq 'hetgpu_iq1s_register_tensor_v1' "${builder}"
grep -Fq 'hetgpu_iq1s_bind_device_v1' "${builder}"
for symbol in \
    hetgpu_iq1s_layer_begin_v2 \
    hetgpu_iq1s_layer_set_routes_v2 \
    hetgpu_iq1s_layer_phase_commit_v2 \
    hetgpu_iq1s_layer_commit_v2 \
    hetgpu_iq1s_layer_abort_v2; do
    grep -Fq "dlsym(RTLD_DEFAULT, \"${symbol}\")" \
        "${repo_root}/tools/llama-qwen35-tq1-hetgpu.patch"
done
grep -Fq 'struct hetgpu_iq1s_stream_state' "${repo_root}/tools/llama-qwen35-tq1-hetgpu.patch"
grep -Fq 'state.gate_seen && state.up_seen && !state.phase_a_committed' \
    "${repo_root}/tools/llama-qwen35-tq1-hetgpu.patch"
grep -Fq 'hetgpu_iq1s_commit_after_down' "${repo_root}/tools/llama-qwen35-tq1-hetgpu.patch"
grep -Fq 'hetgpu_iq1s_try_close_gpu_down' "${repo_root}/tools/llama-qwen35-tq1-hetgpu.patch"
grep -Fq '!hetgpu_iq1s_try_close_gpu_down(state)' \
    "${repo_root}/tools/llama-qwen35-tq1-hetgpu.patch"
grep -Fq 'hetgpu_iq1s_note_route_weights' "${repo_root}/tools/llama-qwen35-tq1-hetgpu.patch"
grep -Fq 'static std::atomic<uint64_t> hetgpu_iq1s_next_transaction{1}' \
    "${repo_root}/tools/llama-qwen35-tq1-hetgpu.patch"
grep -Fq 'hetgpu_iq1s_graph_boundary("exception")' \
    "${repo_root}/tools/llama-qwen35-tq1-hetgpu.patch"
grep -Fq 'hetgpu_iq1s_abort_active_for_cuda_error();' \
    "${repo_root}/tools/llama-qwen35-tq1-hetgpu.patch"
grep -Fq 'int32_t * expert_ids_device = nullptr;' \
    "${repo_root}/tools/llama-qwen35-tq1-hetgpu.patch"
grep -Fq 'cudaMemcpy2DAsync(' \
    "${repo_root}/tools/llama-qwen35-tq1-hetgpu.patch"
grep -Fq 'constexpr uint32_t max_batch = 32;' \
    "${repo_root}/tools/llama-qwen35-tq1-hetgpu.patch"
grep -Fq 'ids->ne[1] > max_batch' \
    "${repo_root}/tools/llama-qwen35-tq1-hetgpu.patch"
grep -Fq 'max_batch * top_k' \
    "${repo_root}/tools/llama-qwen35-tq1-hetgpu.patch"
grep -Fq 'state.expert_ids_device, top_k_bytes,' \
    "${repo_root}/tools/llama-qwen35-tq1-hetgpu.patch"
grep -Fq 'ids->data, ids->nb[1], top_k_bytes, static_cast<size_t>(ids->ne[1]),' \
    "${repo_root}/tools/llama-qwen35-tq1-hetgpu.patch"
grep -Fq 'static_cast<const int32_t *>(state.expert_ids_device)' \
    "${repo_root}/tools/llama-qwen35-tq1-hetgpu.patch"
python3 - "${repo_root}/tools/llama-qwen35-tq1-hetgpu.patch" <<'PY'
import pathlib
import re
import sys

source = pathlib.Path(sys.argv[1]).read_text()
for member in ("begin", "set_routes", "phase_commit", "commit", "abort"):
    calls = re.findall(rf"^\+.*hooks\.{member}\(.*$", source, re.MULTILINE)
    assert calls, f"missing hooks.{member} call"
    for call in calls:
        assert "const int " in call and "_result =" in call, f"unchecked v2 call: {call}"
PY
grep -Fq 'ldd -r' "${builder}"
grep -Fq 'tq1_upstream_reference.cpp' "${builder}"
grep -Fq 'tq1_upstream_reference' "${builder}"
grep -Fq 'libggml="$(realpath -e "${llama_build}/bin/libggml.so")"' "${builder}"
grep -Fq 'LIBGGML="${libggml}"' "${builder}"
grep -Fq '"libggml": {' "${builder}"
grep -Fq 'sha256sum' "${builder}"
test "$(grep -Fc -- '--ignore-submodules=all' "${builder}")" -ge 2
grep -Fq 'cuda_math_header_sha256' "${builder}"
grep -Fq '/qwen-build/manifest.json' "${builder}"

grep -Fq -- '--ctx-size", str(profile_values["max_active"] * CONTEXT_TOKENS_PER_REQUEST)' "${evaluator}"
grep -Fq -- '--n-gpu-layers", "999"' "${evaluator}"
grep -Fq -- '--verbosity", "4"' "${evaluator}"
grep -Fq -- '--parallel", str(profile_values["max_active"])' "${evaluator}"
grep -Fq 'server_batch = 16 if args.profile == "one-token" else profile_values["max_active"]' "${evaluator}"
grep -Fq -- '"--batch-size", str(server_batch)' "${evaluator}"
grep -Fq -- '"--ubatch-size", str(server_batch)' "${evaluator}"
grep -Fq '"n_predict": tokens_per_request' "${evaluator}"
grep -Fq '"temperature": 0.0' "${evaluator}"
grep -Fq '"seed": 42' "${evaluator}"
grep -Fq '"cache_prompt": False' "${evaluator}"
grep -Fq 'REQUEST_COUNT = PROFILES["full"]["request_count"]' "${evaluator}"
grep -Fq 'MAX_ACTIVE_REQUESTS = PROFILES["full"]["max_active"]' "${evaluator}"
grep -Fq 'CONTEXT_TOKENS_PER_REQUEST = 512' "${evaluator}"
grep -Fq 'PREDICT_TOKENS = PROFILES["full"]["tokens_per_request"]' "${evaluator}"
grep -Fq 'MEASUREMENTS = PROFILES["full"]["measurements"]' "${evaluator}"
grep -Fq 'WARMUPS = PROFILES["full"]["warmups"]' "${evaluator}"
grep -Fq 'Reply with exactly OK and no other text.' "${evaluator}"
grep -Fq 'semantic' "${evaluator}"
grep -Fq 'sha256' "${evaluator}"
grep -Fq 'export GGML_CUDA_DISABLE_GRAPHS=1' "${iq1s_runner}"
grep -Fq 'export CUDA_LAUNCH_BLOCKING=1' "${iq1s_runner}"
grep -Fq 'export CUBLAS_WORKSPACE_CONFIG=:4096:8' "${iq1s_runner}"
test "$(grep -Fc 'install -d "${mode_dir}"' "${iq1s_runner}")" -eq 0
test "$(grep -Fc '/qwen-build/llama-build/bin/llama-server' "${runner}")" -eq 1
grep -Fq 'HETGPU_QWEN_TQ1_XRT=0' "${runner}"
grep -Fq 'HETGPU_QWEN_TQ1_XRT=1' "${runner}"
grep -Fq 'HETGPU_QWEN_TQ1_STRICT=1' "${runner}"
grep -Fq 'run_au250_xrt_tq1.sh' "${runner}"
last_effective_command="$(grep -Ev '^\s*(#|$|echo |printf )' "${runner}" | tail -1)"
test "${last_effective_command}" = 'python3 "${validator}" "${proof_dir}" | tee "${proof_dir}/summary.json"'

grep -Fq 'qwen35_gguf_audit.py' "${iq1s_runner}"
grep -Fq 'qwen35-iq1s-route-manifest.json' "${iq1s_runner}"
test "$(grep -Fc 'HETGPU_QWEN_TQ1_XRT=0' "${iq1s_runner}")" -eq 2
grep -Fq 'HETGPU_TMATMUL_BACKEND=xrt' "${iq1s_runner}"
grep -Fq 'HETGPU_BITNET_DISAGGREGATE=1' "${iq1s_runner}"
grep -Fq 'HETGPU_BITNET_DISAGG_STRICT=1' "${iq1s_runner}"
grep -Fq 'HETGPU_TMATMUL_HARDWARE_MATMUL=1' "${iq1s_runner}"
grep -Fq 'HETGPU_QWEN35_CUDA_BUFFER_MAX_MIB=49152' "${iq1s_runner}"
grep -Fq 'HETGPU_CUDART_PRELAUNCH_NAMED_KERNEL=1' "${iq1s_runner}"
grep -Fq 'HETGPU_QWEN_IQ1S_DISABLE_CUDA_FUSION=1' "${iq1s_runner}"
grep -Fq 'qwen35_build_preflight.py' "${iq1s_runner}"
grep -Fq -- '--build-root /qwen-build' "${iq1s_runner}"
grep -Fq 'libggml="$(realpath -e /qwen-build/llama-build/bin/libggml.so)"' "${iq1s_runner}"
grep -Fq 'HETGPU_QWEN_IQ1S_STRICT=1' "${iq1s_runner}"
grep -Fq 'HETGPU_QWEN_IQ1S_PERSISTENT=1' "${iq1s_runner}"
grep -Fq 'HETGPU_QWEN_MODEL_SHA256="${model_sha256}"' "${iq1s_runner}"
test "$(grep -Fc 'export HETGPU_QWEN_MODEL_SHA256="${model_sha256}"' "${iq1s_runner}")" -eq 1
model_sha_line="$(grep -Fn 'export HETGPU_QWEN_MODEL_SHA256="${model_sha256}"' "${iq1s_runner}" | cut -d: -f1)"
cuda_mode_line="$(grep -Fn 'export HETGPU_QWEN_TQ1_XRT=0' "${iq1s_runner}" | head -1 | cut -d: -f1)"
test "${model_sha_line}" -lt "${cuda_mode_line}"
grep -Fq 'HETGPU_LIBGGML="${verified_libggml}"' "${iq1s_runner}"
grep -Fq 'libggml_sha256=' "${iq1s_runner}"
grep -Fq 'libqwen35_cuda13_launch_shim.so' "${iq1s_runner}"
grep -Fq '"${cuda13_launch_shim}:${libnvcuda}"' "${iq1s_runner}"
test -f "${cuda13_launch_shim}"
test -f "${cuda13_launch_map}"
test -f "${cuda13_launch_shim_test}"
shim_test_dir="$(mktemp -d)"
trap 'rm -rf -- "${shim_test_dir}"' EXIT
cc -O2 -Wall -Wextra -Werror -DHETGPU_QWEN35_LAUNCH_SHIM_TEST \
    "${cuda13_launch_shim}" "${cuda13_launch_shim_test}" \
    -ldl -lpthread -o "${shim_test_dir}/qwen35_cuda13_launch_shim_test"
"${shim_test_dir}/qwen35_cuda13_launch_shim_test"
grep -Fq 'nvidia_capture_modern_iq1s_xrt_moe_mmvq' "${function_rs}"
grep -Fq 'export CARGO_BUILD_JOBS=32' "${iq1s_standalone}"
grep -Fq 'HETGPU_XRT_BAR0_RESOURCE=/sys/bus/pci/devices/0000:64:00.1/resource0' "${iq1s_standalone}"
grep -Fq 'xrt-smi examine -d 0000:64:00.1' "${iq1s_standalone}"
grep -Fq 'xclbin=${HETGPU_XRT_XCLBIN:?' "${iq1s_standalone}"
for cu in ternip_big_1 ternip_big_2 ternip_big_3 ternip_small_1; do
    grep -Fq "${cu}" "${iq1s_standalone}"
    grep -Fq "${cu}" "${iq1s_runner}"
    grep -Fq "${cu}" "${iq1s_validator}"
done
if grep -Eq 'MaxCores_370M|iq1s_layer_(big|small)' "${iq1s_standalone}" "${iq1s_runner}" "${iq1s_validator}"; then
    echo "Qwen IQ1_S workflow still references the rejected persistent xclbin topology" >&2
    exit 1
fi
if grep -Fq 'if (strstr(name, "mul_mat_vec_q_moe") != NULL)' "${cuda13_launch_shim}"; then
    echo "CUDA 13 launch shim still bypasses the multi-token IQ1_S MoE kernel" >&2
    exit 1
fi
test "$(grep -Fc 'run_au250_xrt_iq1s.sh" --inside' "${iq1s_runner}")" -eq 2
grep -Fq 'run_au250_xrt_iq1s.sh" --inside handwritten' "${iq1s_runner}"
grep -Fq 'run_au250_xrt_iq1s.sh" --inside compiler' "${iq1s_runner}"
grep -Fq 'HETGPU_IQ1S_TRACE_MODE="${trace_mode}"' "${iq1s_runner}"
grep -Fq 'HETGPU_XRT_COMPARE_MAX_LAUNCHES=1' "${iq1s_runner}"
grep -Fq -- '--mode "${trace_mode}"' "${iq1s_runner}"
grep -Fq -- '--port "${mode_port}"' "${iq1s_runner}"
grep -Fq 'HETGPU_QWEN_MODEL_CONTEXT_LIMIT=262144' "${iq1s_runner}"
grep -Fq 'HETGPU_XRT_BAR0_RESOURCE=/sys/bus/pci/devices/0000:64:00.1/resource0' "${iq1s_runner}"
test "$(grep -Fc 'QWEN35_BUILD_JOBS=32 CARGO_BUILD_JOBS=32' "${iq1s_runner}")" -eq 2
grep -Fq 'build_threads=%s' "${iq1s_runner}"
grep -Fq '${QWEN35_BUILD_JOBS:-32}' "${iq1s_runner}"
grep -Fq 'lspci -s "${fpga_bdf}" -vv' "${iq1s_runner}"
grep -Fq 'pcie-link.txt' "${iq1s_runner}"
grep -Fq 'validate_qwen35_iq1s_au250_proof.py' "${iq1s_runner}"
iq1s_last_effective_command="$(grep -Ev '^\s*(#|$|echo |printf )' "${iq1s_runner}" | tail -1)"
test "${iq1s_last_effective_command}" = 'python3 "${iq1s_validator}" "${proof_dir}" | tee "${proof_dir}/summary.json"'

echo "PASS: static Qwen AU250 runtime workflow contract"
