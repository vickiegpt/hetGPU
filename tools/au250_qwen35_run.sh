#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
git_common_dir="$(git -C "${repo_root}" rev-parse --path-format=absolute --git-common-dir)"
qwen_model_root="${AU250_QWEN_MODEL_ROOT:-/root/models/qwen35-tq1}"
llama_source_root="${AU250_QWEN_LLAMA_ROOT:-/tmp/llama.cpp-qwen-context}"
qwen_build_root="${AU250_QWEN_BUILD_ROOT:-/root/qwen35-au250-build}"
cuda_root="${AU250_CUDA_ROOT:-/usr/local/cuda-13.0}"
cuda_math_header="${cuda_root}/targets/x86_64-linux/include/crt/math_functions.h"
cuda_compat_dir="${qwen_build_root}/cuda-compat"
cuda_compat_math_header="${cuda_compat_dir}/math_functions.h"
source /au250_xrt/env.sh >/dev/null

proof_mount_args=()
if [[ -n ${AU250_QWEN_PROOF_ROOT:-} ]]; then
    proof_root=$(realpath -e -- "${AU250_QWEN_PROOF_ROOT}")
    [[ -d ${proof_root} ]] || {
        echo "Qwen proof root is not a directory: ${proof_root}" >&2
        exit 1
    }
    case "${proof_root}" in
        /mnt/disk0/*) ;;
        *) echo "proof root must be beneath /mnt/disk0" >&2; exit 1 ;;
    esac
    proof_mount_args=(-v "${proof_root}:${proof_root}")
fi

# The platform helper expects nounset to be disabled while it builds its flag string.
_au250_qwen_devflags() {
    set +u
    _au250_devflags
}

prepare_cuda_math_compat() {
    local temporary="${cuda_compat_math_header}.partial"
    test -f "${cuda_math_header}" || {
        echo "missing CUDA math header ${cuda_math_header}" >&2
        exit 1
    }
    mkdir -p -- "${cuda_compat_dir}"
    sed \
        -e 's/rsqrt(double x);/rsqrt(double x) noexcept(true);/' \
        -e 's/rsqrtf(float x);/rsqrtf(float x) noexcept(true);/' \
        -e 's/double rsqrt(double a));/double rsqrt(double a) noexcept(true));/' \
        -e 's/float rsqrtf(float a));/float rsqrtf(float a) noexcept(true));/' \
        "${cuda_math_header}" > "${temporary}"
    test "$(grep -Fc 'double rsqrt(double a) noexcept(true)' "${temporary}")" -eq 1
    test "$(grep -Fc 'float rsqrtf(float a) noexcept(true)' "${temporary}")" -eq 1
    if [[ -f "${cuda_compat_math_header}" ]] && cmp -s -- "${temporary}" "${cuda_compat_math_header}"; then
        rm -f -- "${temporary}"
    else
        mv -f -- "${temporary}" "${cuda_compat_math_header}"
    fi
}

if [[ "${1:-}" == "--print-docker" ]]; then
    shift
    printf 'docker run --rm --gpus all --privileged %s -v /sys:/sys -v /lib/firmware/xilinx:/lib/firmware/xilinx:ro -v /au250_xrt:/au250_xrt:ro -v %q:/work -v %q:%q:ro -v %q:/models/qwen:ro -v %q:/llama-pristine:ro -v %q:/qwen-build -v %q:/usr/local/cuda-13.0:ro -v %q:/usr/local/cuda-13.0/targets/x86_64-linux/include/crt/math_functions.h:ro app215 %q\n' \
        "$(_au250_qwen_devflags)" "${repo_root}" "${git_common_dir}" "${git_common_dir}" "${qwen_model_root}" \
        "${llama_source_root}" "${qwen_build_root}" "${cuda_root}" \
        "${cuda_compat_math_header}" "$*"
    exit 0
fi

[[ $# -gt 0 ]] || { echo "usage: $0 <command...>" >&2; exit 2; }
[[ -d "${qwen_model_root}" ]] || { echo "missing Qwen model root ${qwen_model_root}" >&2; exit 1; }
[[ -d "${llama_source_root}/.git" ]] || { echo "missing pristine llama.cpp checkout ${llama_source_root}" >&2; exit 1; }
[[ -d "${cuda_root}" ]] || { echo "missing CUDA 13 root ${cuda_root}" >&2; exit 1; }
[[ -d "${git_common_dir}" ]] || { echo "missing Git common directory ${git_common_dir}" >&2; exit 1; }
mkdir -p -- "${qwen_build_root}"
prepare_cuda_math_compat

temperature="$(_au250_fpga_temp)"
[[ -z "${temperature}" || "${temperature}" -lt "${AU250_TEMP_LIMIT:-85}" ]] || {
    echo "AU250 temperature ${temperature}C exceeds guard" >&2
    exit 1
}

docker run --rm --gpus all --privileged $(_au250_qwen_devflags) \
    "${proof_mount_args[@]}" \
    -v /sys:/sys \
    -v /lib/firmware/xilinx:/lib/firmware/xilinx:ro \
    -v /au250_xrt:/au250_xrt:ro \
    -v "${repo_root}":/work -w /work \
    -v "${git_common_dir}":"${git_common_dir}":ro \
    -v "${qwen_model_root}":/models/qwen:ro \
    -v "${llama_source_root}":/llama-pristine:ro \
    -v "${qwen_build_root}":/qwen-build \
    -v "${cuda_root}":/usr/local/cuda-13.0:ro \
    -v "${cuda_compat_math_header}":/usr/local/cuda-13.0/targets/x86_64-linux/include/crt/math_functions.h:ro \
    app215 bash -lc 'source /XRT/build/Release/opt/xilinx/xrt/setup.sh >/dev/null 2>&1; export PATH=/usr/local/cuda-13.0/bin:$PATH; export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}; exec "$@"' _ "$@"
