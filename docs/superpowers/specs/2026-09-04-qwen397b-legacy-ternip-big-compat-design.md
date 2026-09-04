# Qwen3.5-397B Legacy TernIP Big-CU Compatibility Repair

**Date:** 2026-09-04
**Status:** Conversationally approved; pending written-spec review
**Software checkout:** `/home/victoryang00/hetGPU/.worktrees/qwen35-tq1-au250-20260826`
**RTL source:** a new isolated worktree derived from the current TernIP/Qwen RTL branch
**Protected input artifact:** `/home/esifferm/ternip_bench/kernel.xclbin`

## Goal

Produce a new four-CU Alveo U250 xclbin that preserves the existing
`ternip_v1` MM2S/S2MM instruction ABI and correctly executes the repository's
IQ1_S affine-ternary component traces. The image contains three 9-lane
`ternip_big` CUs and one 6-lane `ternip_small` CU connected to DDR banks
0/3/2/1. It must pass deterministic single-CU and four-CU numerical gates,
post-route implementation gates, and a strict Qwen3.5-397B two-generated-token
decode proof before it may be selected by the Qwen runner.

The original `/home/esifferm/ternip_bench/kernel.xclbin` is immutable. The new
image is written under `/au250_xrt/xclbins/` with its xclbin UUID and SHA-256 in
the filename or adjacent manifest. Source changes are committed and pushed;
xclbins, model files, build products, and `.proof/` evidence are not committed.

## Observed failure and proven boundary

The protected input xclbin has SHA-256
`176fc1daa417ecb04179f2f7fd4e6d563d05327ace0f369790ec32e6e1b2e87a`
and UUID `a3e8bdd4-9ba9-972b-a811-d2b9581b8aa4`. A strict Qwen run loaded the
verified 94 GB model and launched 40 IQ1_S components, then rejected raw value
`-6683` because it was outside the mathematically valid `[-4096, 4096]`
component-dot range.

The same image failed the repository's deterministic four-CU IQ1_S fixture
with raw value `32767`. Single-CU isolation established a narrower boundary:

- `ternip_small:ternip_small_1` completed all numerical comparisons exactly;
- `ternip_big:ternip_big_1` returned saturated value `-32768` for the same
  matrix, input, assembler, program, XRT executor, and U250 device;
- the U250 remained healthy with firewall status `GOOD` after each failure.

Therefore the host IQ1_S decomposition, two-bit matrix encoding, i16 lane
layout, assembler, and small-CU hardware path are working controls. The new
work targets the 9-lane big-CU implementation and its build qualification. It
must not weaken raw bounds, reinterpret saturated values, fall back to CUDA
after selecting an eligible launch, or report a partial token as successful.

## Fixed compatibility contract

The replacement image retains the legacy runtime-visible contract:

- kernel ABI: `ternip_v1`, user-managed control;
- instruction width: 128 bits, little-endian 16-byte words;
- program sequence: `ldv`, `tmatmul_import`, `tmatmul_go`,
  `tmatmul_export`, `sv`, terminal `stall`;
- matrix tile: row-major 1024 by 1024 ternary values packed four 2-bit values
  per byte, with `-1 -> 0b11`, `0 -> 0b00`, and `+1 -> 0b01`;
- activation/output layout: dimension-major i16 values with lanes contiguous;
- topology: `ternip_big_1` on bank/SLR 0, `ternip_big_2` on 3,
  `ternip_big_3` on 2, and `ternip_small_1` on 1;
- lane capacities: 9, 9, 9, and 6 in that order;
- clock target: 300 MHz;
- host runtime: the existing four-BO matrix/input/output/program executor;
- compiler and handwritten traces: both use the repository assembler and the
  same physical program binding;
- Qwen model context limit: 262,144 tokens;
- build parallelism: every Cargo, Make, Vivado, and Vitis invocation is limited
  to at most 32 workers.

The xclbin may change internal pipelining, register placement, fanout trees,
and implementation directives. It may not change any host-visible register,
kernel-name, bank, instruction, packing, or completion behavior.

## Architecture and source isolation

Implementation uses a new git worktree so the in-progress layer-persistent
build and its dirty RTL files remain untouched. The worktree begins from the
current committed TernIP/Qwen RTL lineage and adds one target descriptor named
`pynqvivado_au250_Qwen397B_IQ1S_LegacyCompat.json`. Its two kernel definitions
are:

```text
ternip_big   kind=ternip abi=ternip_v1 count=3 BatchSize=9 banks/slrs=0,3,2
ternip_small kind=ternip abi=ternip_v1 count=1 BatchSize=6 bank/slr=1
```

The descriptor is the single source of truth for packed RTL parameters,
kernel hashes, kernel XML, CU replication, DDR connectivity, and host CU
manifest generation. Tests reject duplicate banks/SLRs, incorrect lane
counts, a non-legacy ABI, or a topology other than 9/9/9/6.

The small-CU RTL is retained as a known-working control. Changes shared by big
and small are allowed only when simulation proves identical small-CU behavior.
Big-only pipelining or hierarchy changes are preferred when the failing path
can be isolated there.

## Root-cause workflow

The implementation does not assume that saturation is caused solely by
timing. It establishes the first divergent boundary in this order:

1. Generate the exact packed configuration for the 9-lane big and 6-lane
   small kernels and archive their hashes.
2. Run the same deterministic ternary matrix/i16 activation vectors through a
   software reference, big-CU RTL simulation, and small-CU RTL simulation.
3. At the big-CU tmatmul boundary, compare imported lane values, decoded
   ternary values, partial accumulators, export values, and store bytes.
4. If RTL simulation diverges, fix the first incorrect pipeline stage and
   repeat the same failing test before proceeding.
5. If RTL simulation passes, build a single 9-lane big XO and inspect synthesis
   and implementation timing, CDC, fanout, and unconstrained-path reports.
6. Only after the single-big live fixture passes may the three-big/one-small
   image be linked and tested.

One hypothesis is tested at a time. A change that does not make the failing
fixture pass is reverted or superseded before another independent change is
attempted. Three failed fix hypotheses trigger an architecture review rather
than a fourth speculative patch.

## RTL numerical test

Before production RTL changes, the work adds or extends a deterministic test
that exercises the real 9-lane `ternip_big` datapath with:

- at least one all-zero matrix/vector case;
- signed extrema and alternating `-1/0/+1` matrix values;
- all nine lanes carrying distinct i8-range values widened to i16;
- a case whose valid raw dot is exactly `-4096` and one exactly `4096`;
- row and lane padding that must remain zero;
- the exact six-instruction program generated by the host assembler.

The test compares every stored i16 element with a software integer reference.
Saturation at `32767` or `-32768`, an unknown value, a missing store, a
nonzero padding value, or a nonterminal STALL is a failure. The same test is
parameterized for six lanes and must remain passing for `ternip_small`.

## Build and implementation qualification

The build flow emits content-addressed big and small XOs and links a new
content-addressed xclbin. Build commands set `NUM_JOBS`, `CARGO_BUILD_JOBS`,
Vivado `general.maxThreads`, `--vivado.synth.jobs`, and
`--vivado.impl.jobs` to values no greater than 32.

Compiler options that prevent automatic clock downscaling may be used to hold
the requested 300 MHz target, but they do not waive acceptance. A generated
xclbin is rejected unless post-route evidence proves all of:

- setup WNS is nonnegative;
- hold WHS is nonnegative;
- pulse-width slack is nonnegative;
- no unrouted nets or routing overlaps;
- no critical DRC or methodology violations relevant to the four kernels;
- every clock used by the kernel is constrained;
- CU names, bank connectivity, SLR placement, lane parameters, and kernel ABI
  match the fixed descriptor;
- the output is a complete regular file with nonzero UUID and SHA-256.

The qualifier records tool versions, command lines, source commit, dirty-state
hash, descriptor hash, XO hashes, timing summaries, xclbin UUID, xclbin
SHA-256, and kernel/connectivity metadata. A build process exiting zero is not
sufficient qualification.

## Live hardware gates

The live gates always verify the target artifact hash before programming and
check that no unrelated process owns the U250. The gates run in this order:

1. Program the new image and confirm its UUID, four expected CUs, DDR mapping,
   healthy device status, and firewall `GOOD`.
2. Run the deterministic fixture on `ternip_big_1` only. Every output must
   match the software reference and remain in `[-4096, 4096]`.
3. Repeat on `ternip_small_1` as the control.
4. Run the tiled fixture across all four CUs. Submission and completion counts
   must be equal and positive for every CU; all reconstructed f32 values must
   match the software reference bit-for-bit where the established test
   requires it.
5. Repeat the four-CU fixture once with handwritten traces and once with
   AlgorithmTree compiler traces. The two modes must have identical semantic
   coverage and numerical outputs.

Any timeout, missing/duplicate completion, unexpected CU, STALL mismatch,
padding error, raw-bound error, comparison error, XRT failure, firewall event,
or artifact-identity mismatch aborts the sequence. The failing artifact is
retained but is never made the runner default.

## Strict Qwen decode-token gate

After all standalone gates pass, the existing strict runner uses the verified
model:

`/root/models/qwen35-tq1/Qwen3.5-397B-A17B-UD-TQ1_0.gguf`

with expected size `94,155,830,880` and SHA-256
`0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568`.
Attention and all non-IQ1_S work remain on the RTX PRO 6000. Only the audited
141 IQ1_S routed-expert tensors are eligible for U250; the other 39 routed
expert tensors remain GPU-native.

The diagnostic requests two generated tokens. The first token consumes
prompt-evaluation logits; the second forces one genuine M=1 decode graph. The
run passes only when:

- model and tensor audits match exactly;
- the model loads with the intended GPU placement;
- at least one eligible IQ1_S decode launch is selected;
- every selected launch executes on U250 with no native CUDA fallback;
- all four physical CUs have positive validated submissions and completions;
- handwritten/compiler program hashes and model-context metadata are bound to
  the actual completion records;
- attention remains GPU-native;
- both tokens complete and match a CUDA-only greedy reference;
- route, XRT, CUDA, model, artifact, health, and timing evidence is complete;
- error and fallback counts are zero.

No E2E TPS is reported from this two-token qualification. Aggregate throughput
remains a later gate using 64 requests, 32 generated tokens each, and at most
16 active requests.

## Artifact publication and git handoff

The protected input xclbin is never overwritten or renamed. A passing new
image and manifest are installed atomically under `/au250_xrt/xclbins/` only
after the qualification checks complete. The runner receives the new path
explicitly; no global symlink or silent default substitution is used.

Git commits contain only reviewed source, tests, descriptor, qualification
scripts, documentation, and small text manifests needed to reproduce the
build. They exclude:

- `*.xclbin`, `*.xo`, checkpoints, routed design databases, and generated IP;
- the 94 GB model;
- `.proof/`, logs, traces, screenshots, and benchmark output;
- unrelated pre-existing dirty changes.

The implementation branch is pushed only after the strict Qwen gate passes.
If RTL, implementation, standalone hardware, or Qwen validation fails, the
failure is reported with its evidence path and no completion/push claim is
made.

## Non-goals

- Changing the IQ1_S mathematical decomposition or raw bounds.
- Moving attention, routing, normalization, KV/recurrent state, embeddings,
  sampling, or non-IQ1_S expert tensors to U250.
- Completing the separate `iq1s_layer_v2` persistent-kernel effort as part of
  this repair.
- Using only the working small CU as a four-CU substitute.
- Overwriting the user-supplied xclbin.
- Claiming the 15 tok/s target before the fixed 64-by-32 workload passes.
