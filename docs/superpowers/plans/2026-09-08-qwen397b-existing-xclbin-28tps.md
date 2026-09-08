# Qwen3.5-397B Existing-Xclbin 28 TPS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the strict Qwen3.5-397B GPU/U250 correctness path, qualify CUDA/handwritten/compiler one-token execution, and measure whether the existing persistent U250 xclbin reaches 28 aggregate TPS for the fixed 64-request by 32-token workload.

**Architecture:** Replace invalid CUDA-pointer uniqueness with a content-bound ordered lane identity and exact packed-activation slab reuse. Keep GPU attention and the fixed 141/39 tensor placement, then admit a three-measurement 64-by-32 performance run only from an accepted one-token proof. Diagnose the live PCIe link without mutating hardware; a missed 28 TPS target transitions to a separate fused full-FFN xclbin plan.

**Tech Stack:** Rust/ZLUDA CUDA-launch interception, CUDA Driver API, XRT C API, U250 four-CU persistent kernel, patched llama.cpp, Bash proof runner, Python/pytest evaluator and proof validators.

---

## File map

- `zluda/src/impl/iq1s_layer_trace.rs`: represent and validate content-bound packed activation identities in semantic traces.
- `zluda/src/impl/iq1s_persistent_runtime.rs`: derive ordered lane identities, reuse exact activation slabs, and preserve fail-closed buffer ownership.
- `tools/run_qwen35_iq1s_persistent_hybrid.sh`: enforce the 28 TPS/73.143-second three-measurement admission rule.
- `zluda/tests/test_qwen35_iq1s_persistent_runtime_static.sh`: pin the runner's fixed performance threshold and workload contract.
- `zluda/tests/test_qwen35_au250_eval.py`: retain the exact 64-by-32/two-wave evaluator contract.
- `zluda/tests/test_validate_qwen35_iq1s_persistent_gate.py`: verify that only an accepted one-token proof can admit the full run.
- `/root/qwen35-au250-build/manifest.json`: generated runtime identity; never edit by hand.
- `/mnt/disk0/qwen397b-proof/`: new immutable proof directories for the one-token gate and full measurements.

## Execution invariants

- Use only `/root/models/qwen35-tq1/Qwen3.5-397B-A17B-UD-TQ1_0.gguf`,
  94,155,830,880 bytes with SHA-256
  `0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568`.
- Preserve the model context limit of 262,144 tokens.
- Route all 141 IQ1_S expert tensors to U250 and keep the other 39 routed
  tensors on GPU. Attention remains on GPU.
- Keep the existing xclbin byte-identical with SHA-256
  `9c83dcae07b4c7bf1d2e1cebf46ccf0ff1ebf8848a437035fef1451dee7770a3`
  and UUID `b1bafc64-09fd-32b0-a5b4-a881e554ae84`.
- Never exceed 32 compilation jobs, never reuse a failed proof directory, and
  never report throughput without an accepted sealed proof.

### Task 1: Replace pointer uniqueness with packed activation identity

**Files:**
- Modify: `zluda/src/impl/iq1s_layer_trace.rs:52-57`
- Modify: `zluda/src/impl/iq1s_layer_trace.rs:168-195`
- Test: `zluda/src/impl/iq1s_layer_trace.rs` test module

- [ ] **Step 1: Write trace validation regressions**

Extend the trace fixture so every `ActivationRange` has a nonzero identity:

```rust
ActivationRange {
    cuda_ptr: 0x10_0000,
    source_identity_sha256: [0x41; 32],
    slab_offset: 0,
    bytes: 1024 * 1024,
    stream: 0xabc0,
}
```

Add one test that assigns the same `cuda_ptr` to two ranges with distinct
nonzero `source_identity_sha256` values and nonoverlapping slab offsets, then
requires `compile_layer_phase` to succeed. Add a second test that repeats one
`source_identity_sha256` with a different slab offset and requires the error
`Qwen IQ1_S activation identity maps to inconsistent ranges`.

- [ ] **Step 2: Run the tests and observe RED**

Run:

```bash
CARGO_BUILD_JOBS=32 \
CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/tmp/qwen-cc-allow-shlib-undefined \
cargo test -j 32 -p zluda --no-default-features \
  --features nvidia,embed_cudart,evaluation \
  iq1s_layer_trace::tests -- --nocapture
```

Expected: compilation fails because `ActivationRange` has no
`source_identity_sha256`, or the legal repeated-pointer case fails with
`Qwen IQ1_S activation range repeats a CUDA pointer`.

- [ ] **Step 3: Implement the trace identity contract**

Add the identity field:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ActivationRange {
    pub(crate) cuda_ptr: usize,
    pub(crate) source_identity_sha256: [u8; 32],
    pub(crate) slab_offset: u64,
    pub(crate) bytes: u32,
    pub(crate) stream: usize,
}
```

In `validate_activation_ranges`, remove the `BTreeSet<usize>` pointer check.
Use a `HashMap<[u8; 32], (u64, u32, usize)>` and reject a zero digest. An
existing digest is legal only when its slab offset, byte count, and stream are
identical; otherwise return exactly:

```rust
return Err("Qwen IQ1_S activation identity maps to inconsistent ranges".to_string());
```

Keep the existing pointer lower bound, stream, size, alignment, overflow, and
slab-overlap checks. Include `source_identity_sha256` in the phase semantic
hash immediately after `cuda_ptr` so handwritten and compiler traces bind the
same source identity.

- [ ] **Step 4: Run the trace tests and observe GREEN**

Run the command from Step 2. Expected: all `iq1s_layer_trace::tests` pass,
including legal repeated pointers, inconsistent identity rejection, overlap
rejection, and handwritten/compiler equivalence.

- [ ] **Step 5: Commit the trace contract**

```bash
git add zluda/src/impl/iq1s_layer_trace.rs
git commit -m "fix: identify shared Qwen activations by content"
```

### Task 2: Reuse exact packed activation slabs

**Files:**
- Modify: `zluda/src/impl/iq1s_persistent_runtime.rs:307-516`
- Test: `zluda/src/impl/iq1s_persistent_runtime.rs` test module

- [ ] **Step 1: Write runtime reuse regressions**

Add a fixture with one token routed to two experts in both Gate and Up. Give
all four launches the same activation pointer and bytes while keeping their
weight and output identities distinct. Require:

```rust
let prepared = prepare_phase(&arena, phase_a_shared_snapshot(), "handwritten").unwrap();
assert_eq!(prepared.buffers.activations.len(), 1);
assert_eq!(prepared.compiled.activations.len(), 1);
assert!(prepared.compiled.commands.iter().flatten().all(|command| {
    command.input_offset == prepared.buffers.activations[0].offset
}));
```

Clone the fixture, alter one lane byte for the second expert, and require two
activation buffers with distinct aligned offsets. Clone it again with identical
lane bytes but reversed token order and require two distinct identities.

- [ ] **Step 2: Run the runtime tests and observe RED**

Run:

```bash
CARGO_BUILD_JOBS=32 \
CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/tmp/qwen-cc-allow-shlib-undefined \
cargo test -j 32 -p zluda --no-default-features \
  --features nvidia,embed_cudart,evaluation \
  iq1s_persistent_runtime::tests -- --nocapture
```

Expected: shared Gate/Up/top-10 inputs produce multiple activation buffers or
fail trace pointer validation.

- [ ] **Step 3: Derive an ordered lane-source digest**

Add a private helper whose domain and inputs are fixed:

```rust
fn packed_activation_identity(
    key: &LayerKey,
    lanes: &[(RouteAssignment, CapturedActivationLaunch)],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"hetgpu-qwen-iq1s-packed-activation-v1\0");
    hash.update(key.session_generation.to_le_bytes());
    hash.update(key.transaction_id.to_le_bytes());
    hash.update(key.layer_id.to_le_bytes());
    hash.update((key.stream as u64).to_le_bytes());
    for (route, launch) in lanes {
        hash.update(route.token_id.to_le_bytes());
        hash.update((launch.launch.activation_ptr as u64).to_le_bytes());
        hash.update((launch.packed_activations().len() as u64).to_le_bytes());
        hash.update(Sha256::digest(launch.packed_activations()));
    }
    hash.finalize().into()
}
```

Do not include role or expert ID: identical ordered token inputs must be reusable
by Gate, Up, and different experts. Transaction, layer, stream, pointer, length,
lane order, and captured bytes remain bound. The current CUDA interception ABI
does not expose a separate activation-allocation generation; at this boundary,
the transaction-scoped pointer plus the digest of bytes already copied from that
pointer is the fail-closed activation allocation identity. The existing matrix
allocation generation remains independently checked against the resolved weight.

- [ ] **Step 4: Implement exact cache-backed slab reuse**

In `prepare_phase`, keep a
`HashMap<[u8; 32], (u64, Vec<u8>, usize)>`. After `pack_q8_lanes`, compute the
identity. On a hit, compare the cached bytes byte-for-byte and require the same
diagnostic first-lane pointer; then reuse its `activation_offset` without adding
a `HostRange` or advancing `activation_cursor`. On a miss, allocate the next
aligned offset, add one `HostRange`, one `ActivationRange` carrying the digest,
and cache the bytes. A digest hit with unequal bytes or pointer returns:

```rust
return Err("persistent IQ1_S packed activation identity collision".to_string());
```

Commands and output bindings use the selected offset in both cases. Token maps
remain independently packed in this task because their global-token mapping is
already small and validated.

- [ ] **Step 5: Run runtime and trace tests and observe GREEN**

Run the commands from Task 1 Step 2 and Task 2 Step 2. Expected: both modules
pass, batch sizes 1/6/9/16/32 remain accepted, and batch 33 remains rejected.

- [ ] **Step 6: Commit runtime reuse**

```bash
git add zluda/src/impl/iq1s_persistent_runtime.rs
git commit -m "perf: reuse exact Qwen activation slabs"
```

### Task 3: Enforce the 28 TPS proof threshold

**Files:**
- Modify: `tools/run_qwen35_iq1s_persistent_hybrid.sh:300-345`
- Modify: `tools/qwen35_au250_eval.py:1430-1490`
- Modify: `zluda/tests/test_qwen35_iq1s_persistent_runtime_static.sh`
- Test: `zluda/tests/test_qwen35_au250_eval.py`

- [ ] **Step 1: Write static and evaluator contract tests**

Require the runner to contain fixed values and fail-closed checks:

```bash
grep -Fq 'target_tps = 28.0' tools/run_qwen35_iq1s_persistent_hybrid.sh
grep -Fq 'max_wall_seconds = 2048.0 / target_tps' tools/run_qwen35_iq1s_persistent_hybrid.sh
grep -Fq 'if min(tps) < target_tps:' tools/run_qwen35_iq1s_persistent_hybrid.sh
```

In the evaluator tests, retain exact assertions for 64 requests, 32 active,
32 generated tokens, two waves, and three measurements. Add a test for a new
`enforce_aggregate_target` helper. A synthetic measurement list with TPS
`[28.1, 29.0, 30.0]`, 2,048 tokens, two waves, and wall times below 73.143
seconds must return minimum/median/maximum statistics. Changing the first TPS
to `27.9`, the token count to `2047`, the waves to `3`, or a wall time to
`73.144` must raise `EvaluationError`.

- [ ] **Step 2: Run the tests and observe RED**

Run:

```bash
bash zluda/tests/test_qwen35_iq1s_persistent_runtime_static.sh
python3 -m pytest -q zluda/tests/test_qwen35_au250_eval.py
```

Expected: the helper test fails because `enforce_aggregate_target` does not
exist, and the static threshold assertions fail because the current summary
records only median TPS and never rejects a sub-28 measurement.

- [ ] **Step 3: Add fixed threshold enforcement**

Add this reusable evaluator function:

```python
def enforce_aggregate_target(measurements, target_tps=28.0):
    if len(measurements) != 3:
        raise EvaluationError("full run omitted three measurements")
    max_wall_seconds = 2048.0 / target_tps
    for item in measurements:
        if item["generated_tokens"] != 2048 or item["wave_count"] != 2:
            raise EvaluationError("full run did not execute fixed 64x32 two-wave workload")
        if item["generation_tokens_per_second"] < target_tps:
            raise EvaluationError("full run is below the aggregate TPS target")
        if item["measured_wall_seconds"] > max_wall_seconds:
            raise EvaluationError("full run exceeded the aggregate wall-time budget")
    values = [item["generation_tokens_per_second"] for item in measurements]
    return {
        "target_tps": target_tps,
        "max_wall_seconds": max_wall_seconds,
        "minimum_tps": min(values),
        "median_tps": statistics.median(values),
        "maximum_tps": max(values),
        "measurement_tps": values,
    }
```

In the performance aggregation Python block, import the helper from
`/work/tools/qwen35_au250_eval.py` and retain fixed local constants for static
inspection:

```python
target_tps = 28.0
max_wall_seconds = 2048.0 / target_tps
```

Call `enforce_aggregate_target(record["measurements"], target_tps)` for CUDA,
handwritten, and compiler records, then write its returned values into each
mode summary. Abort before writing `aggregate-throughput.json` if the helper
raises. Keep `max_wall_seconds` in the summary and assert it equals the helper's
value so the runner cannot drift from the evaluator contract.

- [ ] **Step 4: Run proof/evaluator tests and observe GREEN**

Run:

```bash
bash zluda/tests/test_qwen35_iq1s_persistent_runtime_static.sh
python3 -m pytest -q \
  zluda/tests/test_qwen35_au250_eval.py \
  zluda/tests/test_validate_qwen35_iq1s_persistent_gate.py \
  zluda/tests/test_validate_qwen35_iq1s_au250_proof.py
```

Expected: all tests pass with the fixed 64-by-32 profile and strict 28 TPS
threshold.

- [ ] **Step 5: Commit the threshold gate**

```bash
git add tools/run_qwen35_iq1s_persistent_hybrid.sh \
  tools/qwen35_au250_eval.py \
  zluda/tests/test_qwen35_iq1s_persistent_runtime_static.sh \
  zluda/tests/test_qwen35_au250_eval.py
git commit -m "test: require 28 TPS Qwen aggregate proof"
```

### Task 4: Run the complete offline regression set and rebuild

**Files:**
- Verify: `zluda/src/impl/iq1s_layer_trace.rs`
- Verify: `zluda/src/impl/iq1s_persistent_runtime.rs`
- Verify: `zluda/src/impl/xrt_iq1s_persistent.rs`
- Verify: `tools/llama-qwen35-tq1-hetgpu.patch`
- Generate: `/root/qwen35-au250-build/manifest.json`

- [ ] **Step 1: Run focused Rust modules**

Run with at most 32 jobs:

```bash
for filter in \
  iq1s_layer_trace::tests \
  iq1s_layer::tests \
  iq1s_persistent_runtime::tests \
  xrt_iq1s_persistent::tests \
  xrt_tmatmul::tests
do
  CARGO_BUILD_JOBS=32 \
  CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/tmp/qwen-cc-allow-shlib-undefined \
  cargo test -j 32 -p zluda --no-default-features \
    --features nvidia,embed_cudart,evaluation "$filter" -- --nocapture || exit 1
done
```

Expected: every non-hardware test passes; explicitly ignored live-hardware
smokes remain ignored.

- [ ] **Step 2: Run static, evaluator, proof, and overlay tests**

```bash
bash zluda/tests/test_qwen35_iq1s_persistent_runtime_static.sh
bash zluda/tests/test_au250_qwen35_runtime_static.sh
bash zluda/tests/test_prepare_au250_qwen35_source.sh
python3 -m pytest -q \
  zluda/tests/test_qwen35_au250_eval.py \
  zluda/tests/test_validate_qwen35_iq1s_persistent_gate.py \
  zluda/tests/test_validate_qwen35_iq1s_au250_proof.py
git diff --check
```

Expected: all commands exit zero and no whitespace errors are reported.

- [ ] **Step 3: Rebuild the runtime with 32 jobs**

```bash
QWEN35_BUILD_JOBS=32 CARGO_BUILD_JOBS=32 \
bash tools/au250_qwen35_run.sh \
  bash /work/tools/build_au250_qwen35_runtime.sh
```

Expected: exit zero and a new `/root/qwen35-au250-build/manifest.json` whose
artifact hashes match the files under `/root/qwen35-au250-build`.

- [ ] **Step 4: Recheck the immutable image**

```bash
sha256sum /au250_xrt/xclbins/qwen397b_iq1s_layer_persistent_9c83dcae.xclbin
xclbinutil --info --input \
  /au250_xrt/xclbins/qwen397b_iq1s_layer_persistent_9c83dcae.xclbin |
  grep -E 'UUID \(xclbin\)|Instance:'
```

Expected: SHA-256 is
`9c83dcae07b4c7bf1d2e1cebf46ccf0ff1ebf8848a437035fef1451dee7770a3`,
UUID is `b1bafc64-09fd-32b0-a5b4-a881e554ae84`, and all four expected CUs are
present.

### Task 5: Run a fresh strict one-token hardware gate

**Files:**
- Generate: a timestamped `qwen397b-28tps-gate-*` directory under `/mnt/disk0/qwen397b-proof/`

- [ ] **Step 1: Verify idle hardware and enable temporary overcommit**

```bash
pgrep -a llama-server || true
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
free -h
sysctl -w vm.overcommit_memory=1
```

Expected: no stale Qwen server, sufficient GPU memory, and overcommit reports
`vm.overcommit_memory = 1`.

- [ ] **Step 2: Start the new immutable proof directory**

Use a UTC timestamp once, require that the directory does not already exist,
and run:

```bash
gate_stamp=$(date -u +%Y%m%dT%H%M%SZ)
gate_dir=/mnt/disk0/qwen397b-proof/qwen397b-28tps-gate-${gate_stamp}
test ! -e "$gate_dir"
printf '%s\n' "$gate_dir" > /tmp/qwen397b-28tps-gate.path
QWEN35_THREADS=32 QWEN35_BUILD_JOBS=32 CARGO_BUILD_JOBS=32 \
bash tools/run_qwen35_iq1s_persistent_hybrid.sh \
  --gate one-token \
  --model /root/models/qwen35-tq1/Qwen3.5-397B-A17B-UD-TQ1_0.gguf \
  --manifest /root/qwen35-au250-build/manifest.json \
  --xclbin /au250_xrt/xclbins/qwen397b_iq1s_layer_persistent_9c83dcae.xclbin \
  --proof-dir "$gate_dir"
```

Expected: CUDA, handwritten, and compiler print `status: pass`; handwritten
and compiler each record Phase A and Phase B completions from all four CUs,
sampled numerical comparison passes, and all three token sequences match.

- [ ] **Step 3: Restore the host policy on every exit path**

```bash
sysctl -w vm.overcommit_memory=2
```

Expected: `vm.overcommit_memory = 2`, including after a runner failure.

- [ ] **Step 4: Validate and seal the gate**

Run the repository gate validator against the new proof directory. Expected:
`validation.status=accepted`, nonempty authoritative phase ledgers, exactly
59,139,686,400 resident bytes per mode, matching token IDs, no fallback, no
XRT fault, and the pinned artifact identities.

If a new error occurs, preserve the proof directory, collect the exact server
error, progress stage counts, phase-ledger counts, device health, and memory
state, then stop before changing code or hardware.

### Task 6: Diagnose PCIe and run the existing-xclbin baseline

**Files:**
- Generate: a timestamped `qwen397b-28tps-full-*` directory under `/mnt/disk0/qwen397b-proof/`

- [ ] **Step 1: Record the live link without changing it**

Resolve the U250 user BDF from `/sys/bus/pci/drivers/xocl`, then record the
endpoint and its upstream bridge:

```bash
readlink -f /sys/bus/pci/devices/0000:64:00.1
cat /sys/bus/pci/devices/0000:64:00.1/current_link_speed
cat /sys/bus/pci/devices/0000:64:00.1/current_link_width
cat /sys/bus/pci/devices/0000:64:00.1/max_link_speed
cat /sys/bus/pci/devices/0000:64:00.1/max_link_width
lspci -s 64:00.1 -vv
```

Expected current evidence is 8 GT/s x2 with maximum x16. Inspect AER and kernel
logs read-only. Do not retrain, reset, unbind, power-cycle, or reboot in this
step. If restoring x16 requires one of those actions, present the exact endpoint
and upstream target for separate approval.

- [ ] **Step 2: Start full performance from the accepted gate**

After the gate is accepted and hardware identity is unchanged:

```bash
gate_dir=$(< /tmp/qwen397b-28tps-gate.path)
test -d "$gate_dir"
full_stamp=$(date -u +%Y%m%dT%H%M%SZ)
full_dir=/mnt/disk0/qwen397b-proof/qwen397b-28tps-full-${full_stamp}
test ! -e "$full_dir"
printf '%s\n' "$full_dir" > /tmp/qwen397b-28tps-full.path
sysctl -w vm.overcommit_memory=1
QWEN35_THREADS=32 QWEN35_BUILD_JOBS=32 CARGO_BUILD_JOBS=32 \
bash tools/run_qwen35_iq1s_persistent_hybrid.sh \
  --gate one-token \
  --performance full \
  --accepted-gate "$gate_dir" \
  --model /root/models/qwen35-tq1/Qwen3.5-397B-A17B-UD-TQ1_0.gguf \
  --manifest /root/qwen35-au250-build/manifest.json \
  --xclbin /au250_xrt/xclbins/qwen397b_iq1s_layer_persistent_9c83dcae.xclbin \
  --proof-dir "$full_dir"
status=$?
sysctl -w vm.overcommit_memory=2
exit "$status"
```

Expected: three complete measurements for CUDA, handwritten, and compiler;
each measurement contains 2,048 tokens in two waves and no measured weight DMA.

- [ ] **Step 3: Apply the 28 TPS decision gate**

Inspect the sealed `aggregate-throughput.json` and validator record. Success
requires all three handwritten measurements and all three compiler measurements
to be at least 28 TPS, with each wall time no greater than 73.143 seconds.

If the proof is accepted and the minimum is at least 28, run final verification,
commit the intended source/test changes, and push the branch. If correctness is
accepted but any measurement is below 28, preserve the exact phase/DMA/CU/GPU
timing breakdown and write the separate fused full-FFN persistent-xclbin plan
from the measured bottleneck. Do not label a projection or the fastest sample
as 28 TPS.
