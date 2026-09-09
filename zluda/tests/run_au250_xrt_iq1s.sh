#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$repo_root"

trace_mode=
if [[ ${1:-} == "--trace-mode" ]]; then
    [[ $# -eq 2 ]] || { echo "usage: $0 --trace-mode handwritten|compiler" >&2; exit 2; }
    trace_mode=$2
elif [[ ${1:-} == "--inside" ]]; then
    [[ $# -eq 2 ]] || { echo "usage: $0 --inside handwritten|compiler" >&2; exit 2; }
    trace_mode=$2
fi
case "$trace_mode" in
    handwritten|compiler) ;;
    *) echo "trace mode must be handwritten or compiler" >&2; exit 2 ;;
esac

if [[ ${1:-} == "--inside" ]]; then
    export CARGO_TARGET_DIR="${repo_root}/target/au250-app215"
    export CARGO_INCREMENTAL=0
    export CARGO_BUILD_JOBS=32

    xclbin=${HETGPU_XRT_XCLBIN:?HETGPU_XRT_XCLBIN must name the qualified legacy IQ1_S xclbin}
    xclbin_info=$(xclbinutil --info --input "$xclbin" 2>&1)
    printf "%s\n" "$xclbin_info"
    require_instance_bank() {
        local instance=$1
        local bank=$2
        awk -v instance="$instance" -v bank="$bank" '
            $1 == "Instance:" { active = ($2 == instance) }
            active && $1 == "Memory:" && $2 == bank { found = 1 }
            END { exit !found }
        ' <<<"$xclbin_info" || {
            echo "qualified legacy topology missing ${instance}->${bank}" >&2
            exit 1
        }
    }
    require_instance_bank ternip_big_1 bank0
    require_instance_bank ternip_big_2 bank3
    require_instance_bank ternip_big_3 bank2
    require_instance_bank ternip_small_1 bank1

    export HETGPU_XRT_AU250_IQ1S_TEST=1
    export HETGPU_XRT_XCLBIN="$xclbin"
    export HETGPU_XRT_NUM_VECTOR_REGISTERS=4
    export HETGPU_XRT_TIMEOUT_MS=10000
    export HETGPU_XRT_CU_CONFIG='{"version":1,"cus":[{"ip_name":"ternip_big:ternip_big_1","memory_group":0,"lanes":9},{"ip_name":"ternip_big:ternip_big_2","memory_group":3,"lanes":9},{"ip_name":"ternip_big:ternip_big_3","memory_group":2,"lanes":9},{"ip_name":"ternip_small:ternip_small_1","memory_group":1,"lanes":6}]}'
    export HETGPU_XRT_BAR0_RESOURCE=/sys/bus/pci/devices/0000:64:00.1/resource0
    export HETGPU_QWEN_IQ1S_STRICT=1
    export HETGPU_IQ1S_TRACE_MODE="$trace_mode"
    export HETGPU_QWEN_MODEL_CONTEXT_LIMIT=262144
    export HETGPU_XRT_EXECUTION_LOG="${repo_root}/target/au250-iq1s-${trace_mode}-live.jsonl"
    rm -f "$HETGPU_XRT_EXECUTION_LOG"

    cargo test -p zluda --features nvidia --no-default-features \
        au250_iq1s_two_by_two_tiles_match_reference -- --ignored --nocapture

    python3 - <<'PY'
import json
import os
from pathlib import Path

records = [json.loads(line) for line in Path(os.environ["HETGPU_XRT_EXECUTION_LOG"]).read_text().splitlines()]
if len(records) != 1:
    raise SystemExit(f"expected one XRT IQ1_S evidence record, got {len(records)}")
evidence = records[0]["evidence"]
if evidence["comparison_status"] != "pass":
    raise SystemExit(f"XRT IQ1_S comparison did not pass: {evidence}")
if not all(value > 0 for value in evidence["per_cu_submissions"]):
    raise SystemExit(f"not all four CUs were used: {evidence}")
if not (-4096 <= evidence["raw_min"] <= evidence["raw_max"] <= 4096):
    raise SystemExit(f"raw component bounds are invalid: {evidence}")
if not evidence["physical_completions"]:
    raise SystemExit("physical completion evidence is empty")
for completion in evidence["physical_completions"]:
    if completion["trace_mode"] != os.environ["HETGPU_IQ1S_TRACE_MODE"]:
        raise SystemExit(f"trace mode mismatch: {completion}")
    if completion["stall_code"] == 0 or completion["program_address"] == 0:
        raise SystemExit(f"unbound program completion: {completion}")
PY

    health_report=$(xrt-smi examine -d 0000:64:00.1 \
        -r dynamic-regions -r error -r firewall -r thermal 2>&1)
    printf "%s\n" "$health_report"
    grep -Fq "Level 0 : 0x0 (GOOD)" <<<"$health_report" || {
        echo "AU250 health gate failed: firewall is not GOOD" >&2
        exit 1
    }
    for cu in ternip_big_1 ternip_big_2 ternip_big_3 ternip_small_1; do
        grep -Eq "${cu}.*\(DONE\)" <<<"$health_report" || {
            echo "AU250 health gate failed: ${cu} is not DONE" >&2
            exit 1
        }
    done
    if grep -Eiq "(^|[^[:alpha:]])fatal([^[:alpha:]]|$)" <<<"$health_report"; then
        echo "AU250 health gate failed: fatal error reported" >&2
        exit 1
    fi
    exit 0
fi

# env.sh currently reads an unset helper while loading, so source it before
# enabling nounset in callers and use its host-side thermal guard here.
set +u
source /au250_xrt/env.sh >/dev/null
set -u
temperature="$(_au250_fpga_temp)"
awk -v temperature="$temperature" 'BEGIN { exit !(temperature < 85) }' || {
    echo "AU250 temperature guard failed before launch: ${temperature} C" >&2
    exit 1
}

bash "${repo_root}/zluda/tests/run_au250_xrt_iq1s.sh" --inside "$trace_mode"

temperature="$(_au250_fpga_temp)"
awk -v temperature="$temperature" 'BEGIN { exit !(temperature < 85) }' || {
    echo "AU250 temperature guard failed after launch: ${temperature} C" >&2
    exit 1
}
echo "AU250 post-run temperature: ${temperature} C"
