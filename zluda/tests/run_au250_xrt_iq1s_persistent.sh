#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
candidate=/au250_xrt/xclbins/qwen397b_iq1s_layer_persistent_9c83dcae.xclbin
candidate_sha256=9c83dcae07b4c7bf1d2e1cebf46ccf0ff1ebf8848a437035fef1451dee7770a3
candidate_uuid=b1bafc64-09fd-32b0-a5b4-a881e554ae84
device_bdf=0000:64:00.1

capture_health() {
    local destination=$1
    xbutil examine -d "${device_bdf}" \
        -r dynamic-regions -r error -r firewall -r thermal 2>&1 | tee "${destination}"
}

require_healthy() {
    local report=$1
    grep -Fq "Level 0 : 0x0 (GOOD)" "${report}" || {
        echo "AU250 firewall is not GOOD in ${report}" >&2
        return 1
    }
    if grep -Eiq '(^|[^[:alpha:]])fatal([^[:alpha:]]|$)' "${report}"; then
        echo "AU250 fatal error reported in ${report}" >&2
        return 1
    fi
}

if [[ ${1:-} == --inside ]]; then
    [[ $# -eq 1 ]] || { echo "usage: $0 --inside" >&2; exit 2; }
    [[ ${HETGPU_PERSISTENT_SMOKE_INSIDE:-} == 1 ]] || {
        echo "persistent smoke inside mode requires its container guard" >&2
        exit 2
    }
    proof_dir=/proof
    qualification=${proof_dir}/qualification.json
    for output in cargo.log health-before.txt health-after.txt summary.json xclbin-info.txt; do
        [[ ! -e ${proof_dir}/${output} ]] || {
            echo "refusing to overwrite G3 evidence ${proof_dir}/${output}" >&2
            exit 1
        }
    done
    [[ -f ${qualification} ]] || { echo "missing G3 qualification record" >&2; exit 1; }
    [[ -f ${candidate} ]] || { echo "missing installed persistent xclbin" >&2; exit 1; }
    [[ $(sha256sum "${candidate}" | awk '{print $1}') == "${candidate_sha256}" ]] || {
        echo "installed persistent xclbin SHA-256 mismatch" >&2
        exit 1
    }
    python3 - "${qualification}" "${candidate}" "${candidate_sha256}" "${candidate_uuid}" <<'PY'
import json
import pathlib
import sys

record = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert record["status"] == "pass"
assert record["installed_path"] == sys.argv[2]
assert record["sha256"] == sys.argv[3]
assert record["uuid"] == sys.argv[4]
assert record["cu_banks"] == {
    "iq1s_layer_big_1": "bank0",
    "iq1s_layer_big_2": "bank3",
    "iq1s_layer_big_3": "bank2",
    "iq1s_layer_small_1": "bank1",
}
PY
    xclbinutil --info --input "${candidate}" > "${proof_dir}/xclbin-info.txt" 2>&1
    grep -Fq "UUID (xclbin):          ${candidate_uuid}" "${proof_dir}/xclbin-info.txt"

    capture_health "${proof_dir}/health-before.txt"
    require_healthy "${proof_dir}/health-before.txt"

    export CARGO_BUILD_JOBS=32
    export CARGO_INCREMENTAL=0
    export CARGO_HOME=/qwen-build/cargo
    export RUSTUP_HOME=/qwen-build/rustup
    export CARGO_TARGET_DIR=/qwen-build/hetgpu-target
    export PATH="${CARGO_HOME}/bin:/usr/local/cuda-13.0/bin:${PATH}"
    export LD_LIBRARY_PATH="/qwen-build/llama-build/bin:/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"
    export HETGPU_QWEN_IQ1S_STRICT=1
    export HETGPU_LIBGGML=/qwen-build/llama-build/bin/libggml.so.0.22.0
    export HETGPU_XRT_AU250_IQ1S_PERSISTENT_TEST=1
    export HETGPU_XRT_XCLBIN="${candidate}"
    export HETGPU_XRT_EXPECTED_UUID="${candidate_uuid}"
    export HETGPU_XRT_TIMEOUT_MS=10000
    export HETGPU_XRT_PERSISTENT_SUMMARY="${proof_dir}/summary.json"

    set +e
    cargo test -p zluda --no-default-features --features nvidia,evaluation \
        au250_iq1s_persistent_four_cu_smoke -- --ignored --nocapture \
        2>&1 | tee "${proof_dir}/cargo.log"
    cargo_status=${PIPESTATUS[0]}
    set -e

    capture_health "${proof_dir}/health-after.txt"
    require_healthy "${proof_dir}/health-after.txt"
    (( cargo_status == 0 )) || {
        echo "persistent four-CU cargo smoke failed with status ${cargo_status}" >&2
        exit "${cargo_status}"
    }
    python3 - "${proof_dir}/summary.json" "${candidate_uuid}" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert summary["status"] == "pass"
assert summary["xclbin_uuid"] == sys.argv[2]
assert summary["persistent_starts_per_cu"] == [1, 1, 1, 1]
assert len(summary["command_baselines_per_cu"]) == 4
assert summary["ring_generations_per_cu"] == [
    [baseline, baseline + 1]
    for baseline in summary["command_baselines_per_cu"]
]
assert summary["per_cu_completions"] == [2, 2, 2, 2]
assert summary["sticky_fault_codes"] == [0, 0, 0, 0]
assert summary["quiescent_before_shutdown"] == [1, 1, 1, 1]
assert summary["result_rows_checked"] == 2048
assert summary["measured_dma"]["weight_ranges"] == 0
assert summary["measured_dma"]["weight_bytes"] == 0
assert summary["measured_dma"]["program_ranges"] == 4
PY
    echo "AU250 persistent four-CU G3 smoke PASS"
    exit 0
fi

[[ $# -eq 1 ]] || { echo "usage: $0 PROOF_DIR" >&2; exit 2; }
proof_dir=$(realpath -e "$1")
[[ ${proof_dir} == /mnt/disk0/* ]] || {
    echo "G3 proof directory must already exist beneath /mnt/disk0" >&2
    exit 1
}
[[ -f ${proof_dir}/qualification.json ]] || {
    echo "G3 proof directory is missing qualification.json" >&2
    exit 1
}

set +u
source /au250_xrt/env.sh >/dev/null
temperature=$(_au250_fpga_temp)
device_flags_text=$(_au250_devflags)
set -u
read -r -a device_flags <<< "${device_flags_text}"
[[ -z ${temperature} || ${temperature} -lt ${AU250_TEMP_LIMIT:-85} ]] || {
    echo "AU250 temperature ${temperature}C exceeds launch guard" >&2
    exit 1
}
git_common_dir=$(git -C "${repo_root}" rev-parse --path-format=absolute --git-common-dir)
qwen_build_root=${AU250_QWEN_BUILD_ROOT:-/root/qwen35-au250-build}
cuda_root=${AU250_CUDA_ROOT:-/usr/local/cuda-13.0}

docker run --rm --gpus all --privileged "${device_flags[@]}" \
    -e HETGPU_PERSISTENT_SMOKE_INSIDE=1 \
    -v /sys:/sys \
    -v /lib/firmware/xilinx:/lib/firmware/xilinx:ro \
    -v /au250_xrt:/au250_xrt:ro \
    -v "${repo_root}":/work -w /work \
    -v "${git_common_dir}":"${git_common_dir}":ro \
    -v "${qwen_build_root}":/qwen-build \
    -v "${proof_dir}":/proof \
    -v "${cuda_root}":/usr/local/cuda-13.0:ro \
    app215 bash -lc \
    'source /XRT/build/Release/opt/xilinx/xrt/setup.sh >/dev/null 2>&1; exec bash /work/zluda/tests/run_au250_xrt_iq1s_persistent.sh --inside'
