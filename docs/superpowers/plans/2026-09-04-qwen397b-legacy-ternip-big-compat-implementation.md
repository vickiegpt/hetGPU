# Qwen3.5-397B Legacy TernIP Big-CU Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and qualify a new legacy `ternip_v1` U250 image whose three 9-lane big CUs and one 6-lane small CU execute the existing IQ1_S traces exactly, then prove one real Qwen decode token and push only the relevant source changes.

**Architecture:** Keep the protected xclbin unchanged and build in a new RTL-repository worktree based on committed Qwen/TernIP build infrastructure. Add one legacy-compatible target descriptor, one exact big/small RTL numerical test, and a fail-closed implementation-report qualifier. Use the existing Qwen software worktree and four-BO XRT executor for single-CU, four-CU, handwritten/compiler, and strict two-token E2E gates.

**Tech Stack:** SystemVerilog, cocotb, Python/pytest, GNU Make, Vivado/Vitis 2026.1, XRT, Rust/Cargo, CUDA-launch interception, llama.cpp.

---

## Repository and artifact boundaries

- Software worktree: `/home/victoryang00/hetGPU/.worktrees/qwen35-tq1-au250-20260826`
- RTL repository: `/home/victoryang00/hetGPU/ternary_matmul`
- New RTL worktree: `/home/victoryang00/hetGPU/.worktrees/ternary-qwen-legacy-compat-20260904`
- New RTL branch: `codex/qwen397b-legacy-ternip-big-compat-20260904`
- RTL base commit: `14a2e5e583f4823d3dbcf7d86c429514cecf14b8`
- Protected xclbin: `/home/esifferm/ternip_bench/kernel.xclbin`
- Staging directory for a qualified image: `/au250_xrt/xclbins/`
- Model: `/root/models/qwen35-tq1/Qwen3.5-397B-A17B-UD-TQ1_0.gguf`

### Task 1: Create the isolated RTL worktree

**Files:**
- Verify only: `/home/victoryang00/hetGPU/ternary_matmul/.git`
- Create worktree: `/home/victoryang00/hetGPU/.worktrees/ternary-qwen-legacy-compat-20260904`

- [ ] **Step 1: Verify the source branch and destination do not conflict**

Run:

```bash
git -C /home/victoryang00/hetGPU/ternary_matmul worktree list --porcelain
git -C /home/victoryang00/hetGPU/ternary_matmul show -s --format='%H %s' 14a2e5e583f4823d3dbcf7d86c429514cecf14b8
test ! -e /home/victoryang00/hetGPU/.worktrees/ternary-qwen-legacy-compat-20260904
```

Expected: the base commit resolves, the persistent worktree remains listed,
and the new path does not exist.

- [ ] **Step 2: Create the branch and worktree**

Run:

```bash
git -C /home/victoryang00/hetGPU/ternary_matmul worktree add \
  -b codex/qwen397b-legacy-ternip-big-compat-20260904 \
  /home/victoryang00/hetGPU/.worktrees/ternary-qwen-legacy-compat-20260904 \
  14a2e5e583f4823d3dbcf7d86c429514cecf14b8
git -C /home/victoryang00/hetGPU/.worktrees/ternary-qwen-legacy-compat-20260904 status --short
```

Expected: the new worktree is clean. Do not copy files from the dirty
layer-persistent worktree.

### Task 2: Add the exact legacy target contract with TDD

**Files:**
- Create: `synth/pynqvivado_au250/targets/pynqvivado_au250_Qwen397B_IQ1S_LegacyCompat.json`
- Modify: `sw_utils/tests/test_target.py`

- [ ] **Step 1: Write the failing descriptor-contract test**

Add a `LEGACY_QWEN_TARGET` path and this test to
`sw_utils/tests/test_target.py`:

```python
def test_qwen_iq1s_legacy_target_exposes_exact_four_cu_contract():
    target = Target(LEGACY_QWEN_TARGET)

    assert target.kernel_names() == ["ternip_big", "ternip_small"]
    assert target.kernel_kind("ternip_big") == "ternip"
    assert target.kernel_kind("ternip_small") == "ternip"
    assert target.kernel_abi("ternip_big") == "ternip_v1"
    assert target.kernel_abi("ternip_small") == "ternip_v1"
    assert target.kernel_count_spec() == "ternip_big:3 ternip_small:1"
    assert target.memory_groups() == [0, 3, 2, 1]
    assert target.slrs() == [0, 3, 2, 1]
    assert target.parameters("ternip_big")["BatchSize"] == 9
    assert target.parameters("ternip_small")["BatchSize"] == 6
    assert target.instance_map_spec() == (
        "ternip_big:1:0:0 ternip_big:2:3:3 "
        "ternip_big:3:2:2 ternip_small:1:1:1"
    )
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python3 -m pytest sw_utils/tests/test_target.py::test_qwen_iq1s_legacy_target_exposes_exact_four_cu_contract -q
```

Expected: FAIL because the descriptor does not exist.

- [ ] **Step 3: Add the minimal descriptor**

Create the descriptor by copying the platform/default fields from
`pynqvivado_au250_MaxCores_370M.json` and set the kernels exactly to:

```json
[
  {
    "name": "ternip_big",
    "kernel_kind": "ternip",
    "bd_script": "bd.tcl",
    "kernel_abi": "ternip_v1",
    "count": 3,
    "slrs": [0, 3, 2],
    "memory_groups": [0, 3, 2],
    "params": {"BatchSize": 9, "NumSeparateAxiInstances": 1}
  },
  {
    "name": "ternip_small",
    "kernel_kind": "ternip",
    "bd_script": "bd.tcl",
    "kernel_abi": "ternip_v1",
    "count": 1,
    "slrs": [1],
    "memory_groups": [1],
    "params": {"BatchSize": 6, "NumSeparateAxiInstances": 1}
  }
]
```

The model field is `Qwen3.5-397B-A17B-UD-TQ1`; the clock remains 3.333 ns.

- [ ] **Step 4: Run the target tests and verify GREEN**

Run:

```bash
python3 -m pytest sw_utils/tests/test_target.py -q
PYTHONPATH=sw_utils python3 -m sw_utils resolve_target --plan \
  synth/pynqvivado_au250/targets/pynqvivado_au250_Qwen397B_IQ1S_LegacyCompat.json \
  pynqvivado_au250
```

Expected: all tests pass and the plan prints three big instances plus one small
instance in bank/SLR order 0/3/2/1.

- [ ] **Step 5: Commit the descriptor contract**

```bash
git add sw_utils/tests/test_target.py \
  synth/pynqvivado_au250/targets/pynqvivado_au250_Qwen397B_IQ1S_LegacyCompat.json
git commit -m "feat: add Qwen legacy four-CU target"
```

### Task 3: Add an exact 9-lane tmatmul RTL regression

**Files:**
- Modify: `dv/cocotb/axi_ternip_batched/test_axi_ternip_batched.py`
- Modify: `dv/cocotb/axi_ternip_batched/Makefile`

- [ ] **Step 1: Add a failing full-program cocotb test**

Add `test_iq1s_component_dot_exact` using the existing `TB`, `Asm`, and AXI
RAM helpers. The test must allocate nonoverlapping matrix/input/output
addresses, pack a 1024x1024 ternary matrix with
`instruction_ternary_to_packed_byte_array`, encode nine distinct lane vectors
with `AlgorithmTree.instruction_vector_to_byte_array`, send this exact program,
and compare all output lanes and rows:

```python
program = [
    ["ldv", "v0", "PARAM_INPUT"],
    ["tmatmul_import", "v0"],
    ["tmatmul_go", "PARAM_MATRIX"],
    ["tmatmul_export", "v0"],
    ["sv", "v0", "PARAM_OUTPUT"],
    ["stall"],
]
expected = torch.matmul(inputs.to(torch.int32), matrix.T.to(torch.int32))
assert torch.equal(actual.to(torch.int32), expected)
assert int(actual.min()) >= -4096
assert int(actual.max()) <= 4096
```

Construct matrix rows and input lanes so the fixture includes zero output,
alternating signs, and exact boundary values `-4096` and `4096`. Assert every
padding byte remains zero.

- [ ] **Step 2: Run the 9-lane test and establish the RTL boundary**

Run with the new descriptor and big kernel:

```bash
NUM_JOBS=32 CARGO_BUILD_JOBS=32 make -C dv/cocotb/axi_ternip_batched clean
NUM_JOBS=32 CARGO_BUILD_JOBS=32 make -C dv/cocotb/axi_ternip_batched \
  TARGET=../../../synth/pynqvivado_au250/targets/pynqvivado_au250_Qwen397B_IQ1S_LegacyCompat.json \
  KERNEL=ternip_big TESTCASE=test_iq1s_component_dot_exact
```

Expected outcomes are constrained:

- FAIL at a specific datapath comparison: retain the transcript and trace the
  first wrong import/decode/accumulate/export/store boundary before editing RTL.
- PASS: record that RTL semantics are correct and make no speculative RTL
  change; continue to implementation qualification because the bad deployed
  xclbin is then a build/artifact issue.

- [ ] **Step 3: Run the 6-lane control**

```bash
NUM_JOBS=32 CARGO_BUILD_JOBS=32 make -C dv/cocotb/axi_ternip_batched clean
NUM_JOBS=32 CARGO_BUILD_JOBS=32 make -C dv/cocotb/axi_ternip_batched \
  TARGET=../../../synth/pynqvivado_au250/targets/pynqvivado_au250_Qwen397B_IQ1S_LegacyCompat.json \
  KERNEL=ternip_small TESTCASE=test_iq1s_component_dot_exact
```

Expected: PASS for all six lanes.

- [ ] **Step 4: Apply one minimal RTL fix only if Step 2 proved divergence**

Modify only the first divergent module named by the waveform/transcript. Keep
the host-visible ABI unchanged. Re-run Steps 2 and 3 until both pass. If Step 2
already passed, this step intentionally changes no production RTL.

- [ ] **Step 5: Commit the regression and any proven RTL correction**

```bash
git add dv/cocotb/axi_ternip_batched/Makefile \
  dv/cocotb/axi_ternip_batched/test_axi_ternip_batched.py rtl third_party/ternip
git diff --cached --check
git commit -m "test: cover nine-lane legacy IQ1S tmatmul"
```

Before committing, unstage every RTL path that was not required by the first
divergent boundary.

### Task 4: Add a fail-closed xclbin implementation qualifier

**Files:**
- Create: `sw_utils/target/qualify_pynqvivado_xclbin.py`
- Create: `sw_utils/tests/test_qualify_pynqvivado_xclbin.py`
- Modify: `sw_utils/__main__.py`

- [ ] **Step 1: Write failing parser and policy tests**

Tests construct temporary timing and xclbin-info text and require:

```python
assert qualify_timing("WNS(ns) 0.012  WHS(ns) 0.004  WPWS(ns) 0.100\n") == {
    "wns_ns": 0.012,
    "whs_ns": 0.004,
    "wpws_ns": 0.100,
}
with pytest.raises(QualificationError, match="setup WNS"):
    qualify_timing("WNS(ns) -0.001  WHS(ns) 0.004  WPWS(ns) 0.100\n")
with pytest.raises(QualificationError, match="routing overlaps"):
    qualify_route_status("Number of Nodes with overlaps = 1\n")
```

The metadata test requires exactly
`ternip_big:ternip_big_1/2/3`, `ternip_small:ternip_small_1`, UUID, and bank
connectivity 0/3/2/1.

- [ ] **Step 2: Run the tests and verify RED**

```bash
python3 -m pytest sw_utils/tests/test_qualify_pynqvivado_xclbin.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the minimal qualifier**

Implement `QualificationError`, `qualify_timing`, `qualify_route_status`,
`qualify_xclbin_info`, and a CLI that accepts:

```text
--xclbin PATH --xclbin-info PATH --timing-summary PATH --route-status PATH
--source-commit HEX --descriptor PATH --output-manifest PATH
```

The CLI refuses missing files, negative WNS/WHS/WPWS, nonzero overlaps,
missing/unexpected CUs, incorrect banks, empty UUID, zero-size xclbin, or a
dirty/unbound source identity. On success it writes a JSON manifest containing
the source commit, descriptor SHA-256, xclbin SHA-256, xclbin UUID, CU/bank
mapping, timing values, and status `pass`.

- [ ] **Step 4: Run qualifier and Make-contract tests**

```bash
python3 -m pytest \
  sw_utils/tests/test_qualify_pynqvivado_xclbin.py \
  sw_utils/tests/test_make_contract.py \
  sw_utils/tests/test_target.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the qualifier**

```bash
git add sw_utils/target/qualify_pynqvivado_xclbin.py \
  sw_utils/tests/test_qualify_pynqvivado_xclbin.py sw_utils/__main__.py
git commit -m "feat: fail closed on U250 implementation reports"
```

### Task 5: Build and statically qualify the four-CU image

**Files:**
- Build output only: the `link_dir` printed by `resolve_target --plan`, followed by `/hw/kernel.xclbin`
- Proof output only: `/home/victoryang00/hetGPU/.worktrees/ternary-qwen-legacy-compat-20260904/.proof/qwen397b-legacy-compat-build-20260904/`

- [ ] **Step 1: Verify no conflicting build owns the chosen output tree**

Run `ps`, `lsof`, and `git status` against the new RTL worktree. The separate
layer-persistent build may remain active because it uses a different worktree
and content-addressed build directory.

- [ ] **Step 2: Build with bounded parallelism**

```bash
export NUM_JOBS=32
export CARGO_BUILD_JOBS=32
make pynqvivado_au250_hw \
  TARGET=synth/pynqvivado_au250/targets/pynqvivado_au250_Qwen397B_IQ1S_LegacyCompat.json \
  MODEL=Qwen3.5-397B-A17B-UD-TQ1 VPP_JOBS=8 VPP_ALLOW_OLD_PLATFORM=1
```

Expected: a complete xclbin in the descriptor-hash build directory. A zero
exit alone does not pass this task.

- [ ] **Step 3: Extract reports and run the qualifier**

Use `xclbinutil --info` for metadata and the routed Vivado run's timing and
route-status reports. Run the new qualifier with the exact paths and write its
manifest under `.proof/`.

Expected: status `pass`, WNS/WHS/WPWS nonnegative, overlaps zero, exact four
CUs, banks 0/3/2/1, and nonzero UUID/SHA-256. If any gate fails, preserve the
reports and stop before programming the board.

- [ ] **Step 4: Install without overwriting the protected image**

Copy the passing image to a temporary file in `/au250_xrt/xclbins/`, verify its
SHA-256, then rename it to
`/au250_xrt/xclbins/qwen397b_legacy_ternip_qualified.xclbin` and install the
qualifier output as
`/au250_xrt/xclbins/qwen397b_legacy_ternip_qualified.json`. Never modify
`/home/esifferm/ternip_bench/kernel.xclbin`.

### Task 6: Pass single-CU and four-CU live numerical gates

**Files:**
- Existing test: `zluda/src/impl/iq1s_xrt.rs`
- Proof output only: `/home/victoryang00/hetGPU/.worktrees/qwen35-tq1-au250-20260826/.proof/qwen397b-legacy-compat-live-20260904/`

- [ ] **Step 1: Verify device ownership and program the qualified image**

Confirm no unrelated container or process owns `0000:64:00.1`, then program
the new xclbin. Verify loaded UUID, firewall `GOOD`, healthy status, four CU
names, and temperature below 85 C.

- [ ] **Step 2: Run `ternip_big_1` only**

Run the ignored Rust fixture with `HETGPU_XRT_CU_CONFIG` containing only
`ternip_big_1`, bank 0, lanes 9. The test may end at the deliberate four-CU
topology assertion, but it must first complete all numerical comparisons and
must not produce a raw-bound or mismatch error.

- [ ] **Step 3: Run `ternip_small_1` control**

Repeat with `ternip_small_1`, bank 1, lanes 6. Require the same numerical
behavior as Step 2.

- [ ] **Step 4: Run strict four-CU handwritten mode**

Set `HETGPU_QWEN_IQ1S_STRICT=1`, trace mode `handwritten`, context limit
262144, exact 9/9/9/6 CU JSON, and the qualified xclbin path. Run
`au250_iq1s_two_by_two_tiles_match_reference`. Expected: PASS, positive and
equal submissions/completions on all four CUs, exact reference output.

- [ ] **Step 5: Run strict four-CU compiler mode**

Repeat Step 4 with trace mode `compiler`. Expected: PASS with the same semantic
coverage and output, with nonzero compiler/assembly/program hashes bound to
completion evidence.

- [ ] **Step 6: Recheck board health**

Require healthy device, firewall `GOOD`, loaded qualified UUID, and no thermal
violation. Any failure blocks E2E.

### Task 7: Run strict Qwen two-token E2E qualification

**Files:**
- Existing proof runner/configuration under the software worktree
- Proof output only: `/home/victoryang00/hetGPU/.worktrees/qwen35-tq1-au250-20260826/.proof/qwen397b-legacy-compat-e2e-20260904/`

- [ ] **Step 1: Re-verify model and runtime identities**

Require the exact model byte count and SHA-256, the Qwen build manifest and
library hashes, compiler/build threads no greater than 32, GPU availability,
qualified xclbin SHA/UUID, and no competing GPU/U250 workloads.

- [ ] **Step 2: Run CUDA-only greedy reference**

Generate two tokens with the same prompt, seed, context, GPU layers, and
sampling settings. Record token IDs and timing, but do not treat prompt-eval
throughput as decode TPS.

- [ ] **Step 3: Run strict hybrid handwritten proof**

Generate two tokens. Require at least one eligible M=1 IQ1_S decode launch,
all four U250 CUs, physical submission/completion equality, zero fallback,
zero errors, GPU attention evidence, exact token equality with CUDA-only, and
complete final event.

- [ ] **Step 4: Run strict hybrid compiler proof**

Repeat Step 3 with compiler trace mode. Require exact token equality with both
CUDA-only and handwritten results plus bound compiler trace/program evidence.

- [ ] **Step 5: Validate proof directories fail closed**

Run the existing proof validator over CUDA, handwritten, and compiler outputs.
No E2E TPS is reported if any event, route, CU, artifact, model, CUDA, or health
record is missing.

### Task 8: Verify, commit, and push only relevant source

**Files:**
- RTL branch changes from Tasks 2-4
- Software branch existing relevant CUDA/backend changes and design/plan docs

- [ ] **Step 1: Run final RTL tests**

Run target, qualifier, Make-contract, big/small cocotb, and any directly
affected RTL regressions. Require all PASS.

- [ ] **Step 2: Run final software tests**

Run focused Rust tests for the CUDA function wrapper, backend environment
aliases, IQ1_S trace builders, XRT executor, and route validation. Require all
PASS.

- [ ] **Step 3: Audit both diffs**

Use `git status --short`, `git diff --check`, `git diff --stat`, and full diffs
in both worktrees. Exclude `.proof/`, logs, model files, build directories,
xclbins, XOs, generated IP, and unrelated dirty changes.

- [ ] **Step 4: Commit any remaining verified source changes**

Commit focused logical units with test-backed messages. Do not amend unrelated
history and do not include binary artifacts.

- [ ] **Step 5: Push both implementation branches**

Push the RTL legacy-compat branch and the Qwen software branch to `origin`
without force. Record remote branch names and commit IDs in the final handoff.

- [ ] **Step 6: Report the proof boundary**

Report the qualified xclbin path/SHA/UUID, timing slacks, single/four-CU
results, CUDA/hybrid token IDs, U250 submission/completion counts, fallback
count, GPU attention evidence, board health, commits, and pushed branches.
State explicitly that 15 tok/s and the 64-by-32 workload remain unclaimed
until that separate benchmark passes.
