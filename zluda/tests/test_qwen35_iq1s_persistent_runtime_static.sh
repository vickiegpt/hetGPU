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
runner="${repo_root}/tools/run_qwen35_iq1s_persistent_hybrid.sh"

grep -Fq 'pub(crate) mod iq1s_persistent_proof;' "${module}"
grep -Fq 'HETGPU_QWEN_IQ1S_PROOF_LEDGER' "${proof}"
grep -Fq '.create_new(true)' "${proof}"
grep -Fq 'must be beneath /mnt/disk0' "${proof}"
grep -Fq 'self.ledger.append_phase(proof)?;' "${runtime}"
append_line="$(grep -nF 'self.ledger.append_phase(proof)?;' "${runtime}" | cut -d: -f1)"
copy_line="$(grep -nF 'publish_outputs_with(&mut self.output_publisher' "${runtime}" | cut -d: -f1)"
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
test -x "${runner}"
grep -Fq 'HETGPU_QWEN_IQ1S_PERSISTENT=1' "${runner}"
grep -Fq 'HETGPU_QWEN_IQ1S_PROOF_LEDGER' "${runner}"
grep -Fq -- '--performance' "${runner}"
grep -Fq -- '--accepted-gate' "${runner}"
grep -Fq '${execution_kind} == correctness' "${runner}"
grep -Fq 'export CUDA_LAUNCH_BLOCKING=1' "${runner}"
grep -Fq 'unset CUDA_LAUNCH_BLOCKING' "${runner}"
grep -Fq 'export HETGPU_QWEN_IQ1S_PROOF_SYNC_PHASE=0' "${runner}"
grep -Fq 'export HETGPU_QWEN_IQ1S_PROOF_SYNC_PHASE=1' "${runner}"
grep -Fq 'export HETGPU_QWEN_IQ1S_PROGRESS_LOG=${mode_dir}/progress.jsonl' "${runner}"
grep -Fq 'export GGML_CUDA_DISABLE_GRAPHS=1' "${runner}"
grep -Fq 'publish_outputs_with(&mut self.output_publisher' "${runtime}"
grep -Fq '"phase_prepared"' "${runtime}"
grep -Fq '"phase_published"' "${runtime}"
grep -Fq '"phase_collected"' "${runtime}"
grep -Fq '.prepare_ticket(' "${runtime}"
grep -Fq '.poll_ticket(' "${runtime}"
grep -Fq '.collect_ticket(' "${runtime}"
grep -Fq 'PhasePipeline::lazy(initialize_runtime_from_env)' "${runtime}"
grep -Fq 'global_phase_pipeline().execute(snapshot)' "${runtime}"
if grep -Fq 'with_global_persistent_runtime(|runtime|' "${repo_root}/zluda/src/impl/iq1s_layer.rs"; then
    echo "persistent phase wait still holds the global runtime mutex" >&2
    exit 1
fi
grep -Fq 'copy_cuda_to_host_batch(' "${function}"
grep -Fq 'snapshot.batch_count > 32' "${runtime}"
grep -Fq -- '--persistent-ledger' "${runner}"
grep -Fq -- '--profile "${profile}"' "${runner}"
grep -Fq 'QWEN35_BUILD_JOBS=32' "${runner}"
grep -Fq "s/^[[:space:]]*UUID (xclbin):[[:space:]]*//p" "${runner}"
grep -Fq 'target_tps = 28.0' "${runner}"
grep -Fq 'max_wall_seconds = 2048.0 / target_tps' "${runner}"
grep -Fq 'if min(tps) < target_tps:' "${runner}"

echo "Qwen IQ1_S persistent runtime static contract: PASS"
