#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime="${repo_root}/zluda/src/impl/iq1s_persistent_runtime.rs"
proof="${repo_root}/zluda/src/impl/iq1s_persistent_proof.rs"
module="${repo_root}/zluda/src/impl/mod.rs"
function="${repo_root}/zluda/src/impl/function.rs"
builder="${repo_root}/tools/build_au250_qwen35_runtime.sh"
evaluator="${repo_root}/tools/qwen35_au250_eval.py"
validator="${repo_root}/zluda/tests/validate_qwen35_iq1s_persistent_gate.py"

grep -Fq 'pub(crate) mod iq1s_persistent_proof;' "${module}"
grep -Fq 'HETGPU_QWEN_IQ1S_PROOF_LEDGER' "${proof}"
grep -Fq '.create_new(true)' "${proof}"
grep -Fq 'must be beneath /mnt/disk0' "${proof}"
grep -Fq 'self.ledger.append_phase(proof)?;' "${runtime}"
append_line="$(grep -nF 'self.ledger.append_phase(proof)?;' "${runtime}" | cut -d: -f1)"
copy_line="$(grep -nF 'copy_host_to_cuda(output.cuda_ptr' "${runtime}" | cut -d: -f1)"
test "${append_line}" -lt "${copy_line}" || {
    echo "persistent proof must be written before CUDA result publication" >&2
    exit 1
}
grep -Fq 'eligible_direct_route=1' "${function}"
grep -Fq 'hetgpu_iq1s_layer_phase_commit_v2' "${builder}"
grep -Fq 'iq1s_persistent_abi_symbols_sha256' "${builder}"
grep -Fq '"one-token"' "${evaluator}"
grep -Fq '"full"' "${evaluator}"
grep -Fq 'result.add_argument("--profile"' "${evaluator}"
grep -Fq 'phase ledger is empty, truncated' "${validator}"

echo "Qwen IQ1_S persistent runtime static contract: PASS"
