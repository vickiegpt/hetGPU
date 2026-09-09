#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
reference="${HETGPU_TEST_LLAMA_SOURCE:-/tmp/llama.cpp-qwen-context}"
prepare="${repo_root}/tools/prepare_au250_qwen35_source.sh"
header="${repo_root}/tools/qwen35-tq1-bridge.h"

test -x "${prepare}"
test -f "${header}"
for relative in \
    src/llama-mmap.h \
    src/llama-mmap.cpp \
    src/llama-model-loader.cpp \
    ggml/src/ggml-cuda/ggml-cuda.cu \
    tools/server/server-context.cpp; do
    test -f "${reference}/${relative}"
done

scratch="$(mktemp -d)"
trap 'rm -rf -- "${scratch}"' EXIT
source="${scratch}/source"
overlay="${scratch}/overlay"
mkdir -p "${source}/src" "${source}/ggml/src/ggml-cuda" "${source}/tools/server"
for relative in \
    src/llama-mmap.h \
    src/llama-mmap.cpp \
    src/llama-model-loader.cpp \
    ggml/src/ggml-cuda/ggml-cuda.cu \
    tools/server/server-context.cpp; do
    cp -- "${reference}/${relative}" "${source}/${relative}"
done
git -C "${source}" init -q
git -C "${source}" config user.email test@example.invalid
git -C "${source}" config user.name "HetGPU overlay test"
git -C "${source}" add .
git -C "${source}" commit -qm fixture
fixture_revision="$(git -C "${source}" rev-parse HEAD)"

HETGPU_OVERLAY_TESTING=1 \
HETGPU_TEST_LLAMA_REVISION="${fixture_revision}" \
    "${prepare}" "${source}" "${overlay}"

grep -Fq 'hetgpu_tq1_register_tensor_v1' "${overlay}/src/llama-model-loader.cpp"
grep -Fq 'hetgpu_iq1s_register_tensor_v1' "${overlay}/src/llama-model-loader.cpp"
grep -Fq 'GGML_TYPE_IQ1_S' "${overlay}/src/llama-model-loader.cpp"
grep -Fq 'hetgpu_tq1_try_mul_mat_id_v1' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq 'hetgpu_iq1s_bind_device_v1' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
for symbol in \
    hetgpu_iq1s_layer_begin_v2 \
    hetgpu_iq1s_layer_set_routes_v2 \
    hetgpu_iq1s_layer_phase_commit_v2 \
    hetgpu_iq1s_layer_commit_v2 \
    hetgpu_iq1s_layer_abort_v2; do
    grep -Fq "dlsym(RTLD_DEFAULT, \"${symbol}\")" \
        "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
done
grep -Fq 'struct hetgpu_iq1s_stream_state' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq 'uint32_t layer =' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq 'uint64_t transaction =' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq 'bool gate_seen =' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq 'bool up_seen =' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq 'bool down_seen =' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq 'state.gate_seen && state.up_seen && !state.phase_a_committed' \
    "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq 'hetgpu_iq1s_commit_after_down' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq 'hetgpu_iq1s_try_close_gpu_down' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq '!hetgpu_iq1s_try_close_gpu_down(state)' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq 'hetgpu_iq1s_note_route_weights' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq 'static std::atomic<uint64_t> hetgpu_iq1s_next_transaction{1}' \
    "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq 'hetgpu_iq1s_graph_boundary("begin")' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq 'hetgpu_iq1s_graph_boundary("end")' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq 'hetgpu_iq1s_graph_boundary("exception")' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq 'hetgpu_iq1s_abort_active_for_cuda_error();' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq 'constexpr uint32_t max_batch = 32;' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq 'ids->ne[1] > max_batch' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq 'max_batch * top_k' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
python3 - "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu" <<'PY'
import pathlib
import re
import sys

source = pathlib.Path(sys.argv[1]).read_text()
for member in ("begin", "set_routes", "phase_commit", "commit", "abort"):
    calls = re.findall(rf"^.*hooks\.{member}\(.*$", source, re.MULTILINE)
    assert calls, f"missing hooks.{member} call"
    for call in calls:
        assert "const int " in call and "_result =" in call, f"unchecked v2 call: {call}"
PY
grep -Fq 'HETGPU_QWEN_IQ1S_DISABLE_CUDA_FUSION' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq 'HETGPU_QWEN35_CUDA_BUFFER_MAX_MIB' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq 'hetgpu_qwen35_cuda_buffer_max_size' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq 'llama_memory_clear(llama_get_memory(ctx_tgt), true)' "${overlay}/tools/server/server-context.cpp"
grep -Fq 'HETGPU Qwen IQ1_S: using verified full sequence removal mode' "${overlay}/tools/server/server-context.cpp"
buffer_helper_line="$(grep -nF 'static size_t hetgpu_qwen35_cuda_buffer_max_size' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu" | cut -d: -f1)"
buffer_allocator_line="$(grep -nF 'static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu" | cut -d: -f1)"
test "${buffer_helper_line}" -lt "${buffer_allocator_line}"
grep -Fq 'const std::string & path() const' "${overlay}/src/llama-mmap.h"
grep -Fq 'HETGPU_TQ1_ABI_VERSION' "${overlay}/tools/qwen35-tq1-bridge.h"
grep -Fq 'HETGPU_IQ1S_ABI_VERSION' "${overlay}/tools/qwen35-tq1-bridge.h"
grep -Fq '925e1179947ea0c0ebfb0032df18af3a729822be' "${prepare}"
test "$(git -C "${source}" rev-parse HEAD)" = "${fixture_revision}"
test -z "$(git -C "${source}" status --short)"

before="$(find "${overlay}" -type f -printf '%P %s %T@\n' | LC_ALL=C sort)"
HETGPU_OVERLAY_TESTING=1 \
HETGPU_TEST_LLAMA_REVISION="${fixture_revision}" \
    "${prepare}" "${source}" "${overlay}"
after="$(find "${overlay}" -type f -printf '%P %s %T@\n' | LC_ALL=C sort)"
test "${before}" = "${after}"

wrong="${scratch}/wrong"
git clone -q --no-hardlinks "${source}" "${wrong}"
git -C "${wrong}" config user.email test@example.invalid
git -C "${wrong}" config user.name "HetGPU overlay test"
printf 'wrong revision\n' > "${wrong}/revision-marker"
git -C "${wrong}" add revision-marker
git -C "${wrong}" commit -qm wrong
if HETGPU_OVERLAY_TESTING=1 HETGPU_TEST_LLAMA_REVISION="${fixture_revision}" \
    "${prepare}" "${wrong}" "${scratch}/wrong-overlay"; then
    echo "wrong source revision unexpectedly accepted" >&2
    exit 1
fi

nonempty="${scratch}/nonempty"
mkdir -p "${nonempty}"
printf 'keep\n' > "${nonempty}/user-file"
if HETGPU_OVERLAY_TESTING=1 HETGPU_TEST_LLAMA_REVISION="${fixture_revision}" \
    "${prepare}" "${source}" "${nonempty}"; then
    echo "nonempty unqualified destination unexpectedly accepted" >&2
    exit 1
fi
test "$(cat "${nonempty}/user-file")" = keep

printf '#include "qwen35-tq1-bridge.h"\nint main(void) { return HETGPU_TQ1_ABI_VERSION != 1 || HETGPU_IQ1S_ABI_VERSION != 1; }\n' \
    > "${scratch}/header.c"
cc -std=c11 -Werror -I"${repo_root}/tools" -c "${scratch}/header.c" -o "${scratch}/header-c.o"
c++ -std=c++17 -Werror -I"${repo_root}/tools" -x c++ -c "${scratch}/header.c" \
    -o "${scratch}/header-cxx.o"

echo "PASS: pinned Qwen llama.cpp overlay preparation"
