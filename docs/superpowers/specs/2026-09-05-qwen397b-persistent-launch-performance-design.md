# Qwen3.5-397B Persistent Launch-Performance Execution Addendum

**Date:** 2026-09-05

**Status:** Approved direction; pending written-spec review

**Parent design:** `2026-09-02-qwen397b-u250-layer-persistent-design.md`

**Software checkout:** `/home/victoryang00/hetGPU/.worktrees/qwen35-tq1-au250-20260826`

**RTL checkout:** `/home/victoryang00/hetGPU/.worktrees/ternary-qwen-iq1s-persistent-20260902`

## Purpose and authority

This addendum converts the parent architecture into a staged, fail-closed
execution contract for the current launch-performance bug. The parent design
remains authoritative for model topology, numerical tolerances, trace truth,
weight layout, workload, and the final 15 aggregate generated tok/s target.
Where the two documents differ on qualification order, runtime wiring, proof
volume, or use of the current xclbin candidate, this addendum controls.

The immediate goal is not to report full-workload throughput. It is to wire
the already implemented layer lifecycle to the already implemented persistent
ring executor, qualify the existing timing-clean xclbin without replacing any
known image, and pass one exact real-model decode token through GPU attention
and U250 IQ1_S FFN. The 64-request by 32-token benchmark is allowed only after
that gate passes.

## Current evidence and diagnosed bottleneck

The latest partial handwritten run is not an end-to-end result because it did
not generate a token and has no `handwritten.json`. It does, however, isolate
the control-path bottleneck:

| Observation | Measured value |
|---|---:|
| Eligible intercepted CUDA launches in the first semantic prompt | 141 |
| Captured components per eligible launch | 40 |
| Legacy logical XRT executions | 5,640 |
| Physical four-CU submissions | 180,480 |
| Estimated FPGA critical path | 49.169486 s |
| Prompt wall time | 496.51070 s |
| FPGA critical-path share of wall time | 9.903% |
| Combined host pack, BO sync, MMIO, polling, reconstruction, and logging | about 447.34 s |
| Resident-program hit rate over the partial run | 37.84% |
| Detailed XRT records in the partial run | 7,399 |
| `xrt.jsonl` plus duplicated stderr evidence | about 654 MiB |

The 447.34 seconds is a combined residual. The current instrumentation does
not justify assigning that time to any one of packing, synchronization, MMIO,
polling, reconstruction, or logging.

The software source matches the measurement. When no layer transaction is
open, `nvidia_dispatch_captured_iq1s` iterates through all captured components
and calls the direct executor once per component. `DirectCapturedLaunchExecutor`
uses a process-global XRT pool, so the process does retain XRT state, but its
unit of host control remains one component and then one synchronous four-CU
wave. Conversely, the layer lifecycle currently fails at Phase A with
`IQ1_S persistent Phase A executor is not wired`, and `PersistentIq1sPool`
has no production call site.

Therefore the primary fix is the layer-to-ring runtime bridge. Relaxing
`CUDA_LAUNCH_BLOCKING`, adding host component threads, or reducing logs cannot
make the legacy component loop into a layer-persistent implementation.

## Fixed execution boundary

- Exact model: `Qwen3.5-397B-A17B`, GGUF size `94,155,830,880` bytes, SHA-256
  `0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568`.
- Model context support remains valid through 262,144 tokens; checked trace and
  offset arithmetic may not narrow this limit.
- Exactly 141 IQ1_S routed-expert tensors execute on U250. The remaining 39
  routed-expert tensors execute on GPU.
- Attention, recurrent/KV state, router, normalization, embeddings, sampling,
  and non-IQ1_S work remain on GPU.
- CUDA launch interception remains authoritative for live pointers, shapes,
  strides, streams, and dependencies. The layer sideband supplies boundaries
  and route identity, not replacement tensor semantics.
- Handwritten and AlgorithmTree compiler builders share one assembler, one
  immutable weight registry, one resident weight arena, and one four-CU
  persistent executor.
- Strict mode has no eligible CUDA fallback. An error poisons the persistent
  session and aborts the run.
- Every compile command is limited to at most 32 threads.

## Selected xclbin candidate

The approved starting candidate is:

```text
/home/victoryang00/hetGPU/.worktrees/ternary-qwen-iq1s-persistent-20260902/
synth/pynqvivado_au250/build/89933b4ba419/hw-signoff-s2-i8-20260904/
kernel.candidate.xclbin
```

Its currently verified static identity is:

- size: `59,157,345` bytes;
- SHA-256: `9c83dcae07b4c7bf1d2e1cebf46ccf0ff1ebf8848a437035fef1451dee7770a3`;
- xclbin UUID: `b1bafc64-09fd-32b0-a5b4-a881e554ae84`;
- CUs: `iq1s_layer_big_1`, `iq1s_layer_big_2`,
  `iq1s_layer_big_3`, and `iq1s_layer_small_1`;
- bank assignment: big 1/2/3 on DDR 0/3/2, small 1 on DDR 1;
- each CU exposes command ring, completion ring, program, arena manifest,
  activation slab, and result slab arguments;
- final post-route physical-optimization timing: WNS `+0.023 ns`,
  TNS `0`, WHS `+0.010 ns`, THS `0`, with all user constraints met.

Static timing and metadata do not qualify the image for inference. The RTL
worktree is dirty at commit `14a2e5e583f4823d3dbcf7d86c429514cecf14b8`, so the qualification bundle must
also record the tracked diff hash, hashes of untracked source inputs used by
the build, build command/log identities, platform identity, and report hashes.
An unaccounted source input fails qualification.

The candidate is installed under a new SHA-derived name only after static
qualification. Neither of these existing files may be overwritten:

- `/home/esifferm/ternip_bench/kernel.xclbin`, SHA-256
  `176fc1daa417ecb04179f2f7fd4e6d563d05327ace0f369790ec32e6e1b2e87a`;
- `/au250_xrt/xclbins/qwen397b_legacy_ternip_qualified.xclbin`, SHA-256
  `60cf200c50552004ac1991e70d6eaf1f6c27addf930b34f355d17eab7bf3ee4b`.

No new RTL or xclbin build is started unless candidate qualification fails or
a hardware smoke identifies an RTL fault.

## Runtime ownership and data flow

One process-global Qwen persistent runtime is keyed by model SHA, xclbin SHA,
device identity, ABI version, and session generation. It owns the tensor
registry, four bank arenas, program cache, command/completion rings, activation
and result slabs, CUDA staging/events, and sticky fault state. A second model
identity or generation cannot reuse this state.

Both trace modes compile normalized layer transactions into the same command
schema:

```text
GPU router + layer_begin/routes
              |
    one asynchronous route D2H batch
              |
 CUDA interception captures gate/up launches
              |
 Phase A: validate -> trace/cache -> batch activation DMA
              |
       one phase publication to four CU rings
              |
      completion validation + result DMA
              |
        GPU SiLU and multiply
              |
 CUDA interception captures down launch
              |
 Phase B: validate -> trace/cache -> batch activation DMA
              |
       one phase publication to four CU rings
              |
      completion validation + result DMA
              |
 GPU merge/residual -> next GPU attention layer
```

`layer_phase_commit(PHASE_A)` validates the complete gate/up capture, selects
the handwritten or AlgorithmTree builder, resolves only checked arena
relocations, publishes all four CU descriptor ranges, waits for matching
completions, copies results onto the bound CUDA stream, and advances the
transaction. `layer_commit` performs the equivalent Phase B operation for an
IQ1_S down role or verifies the registered GPU-native down role before close.

The audited 50 gate, 50 up, and 41 down tensors imply 50 Phase A and 41 Phase B
layer submissions for a traversal of all affected layers, rather than 5,640
component executions for the observed first prompt. Each phase may contain
many descriptors, but descriptor count does not create host-visible kernel
launches or per-component BO synchronization.

Raw IQ1_S weights are hashed and loaded once before inference, row-sharded
across the four 16-GiB banks, and addressed through the immutable arena
manifest. The measured decode window requires `weight_dma_bytes == 0`; a
cache miss that would reload weights aborts instead of silently perturbing the
measurement.

## Correctness-first synchronization and later pipelining

The first hardware qualification and real-model one-token gate use blocking
phase boundaries. This deliberately minimizes concurrency while validating
ring ownership, completion identity, numerical reconstruction, and CUDA/U250
dependency correctness.

Only after exact one-token parity may a performance configuration keep all
four kernels running while queuing a subsequent independent phase. CUDA
events, not global device synchronization, guard route readiness, activation
readiness, and result consumption. Double-buffered slab generations may
overlap GPU-native projection work with U250 execution when the graph exposes
a real independent dependency. A slab is never reused before both its U250
completion and CUDA consumer event are complete.

`CUDA_LAUNCH_BLOCKING=1` and disabled CUDA graphs may then be relaxed only in
an explicitly labeled performance mode. Strict validation, route identity,
stream ordering, numerical comparison, and no-fallback behavior remain
enabled. Because llama.cpp commonly sequences dependent work on one stream,
the implementation reports measured overlap rather than assuming it.

## Compact fail-closed proof

Numerical comparison remains in memory and fail-closed, but the hot path does
not serialize full component vectors twice. The proof ledger emits one compact
record per layer phase containing:

- transaction, layer, phase, stream, session generation, and builder;
- expected and observed role/expert/token coverage;
- program, resolved-trace, arena-manifest, activation, and result hashes;
- descriptor/completion counts and first/last ring sequence per CU;
- activation, result, command, completion, and weight DMA bytes;
- maximum absolute/relative error, nonfinite count, and compare status;
- phase timing breakdown and sticky fault snapshot.

One explicitly selected diagnostic layer may retain detailed comparison data.
All other vectors remain bounded in-memory validation inputs. JSON evidence is
written once to the proof ledger; stderr contains concise human diagnostics,
not duplicated JSON records. Full-workload proof is staged on `/mnt/disk0`
after checking free space and inode capacity. A missing final manifest,
truncated ledger, failed record count, or root-filesystem proof path invalidates
the run.

Timing fields must separately cover capture, route DMA, trace build/cache,
activation packing, activation sync, ring publication, doorbell, device wait,
completion sync, result copy, reconstruction, comparison, and logging. Phase
wall time and command/completion counts are also recorded. These fields are
observations, not a license to subtract overhead from reported end-to-end TPS.

## Fail-closed conditions

In addition to the parent design, any of the following poisons the process
session, prevents result publication, and aborts the run:

- missing lifecycle symbol or invocation, or any eligible launch reaching the
  legacy direct executor while persistent strict mode is active;
- model, xclbin, CU/bank map, platform, ABI, source provenance, arena, manifest,
  program, or session-generation mismatch;
- incomplete or duplicate role capture, stream mismatch, invalid route, or
  unchecked/narrowed offset arithmetic;
- command-ring full/overwrite, stale sequence, completion mismatch/duplicate,
  timeout, CU fault, or incomplete four-CU phase;
- any weight DMA after the measurement-start record;
- nonfinite output, tolerance failure, exact greedy-token mismatch, or an
  eligible IQ1_S GPU fallback;
- attention or any of the 39 non-IQ1_S routed-expert tensors observed on U250;
- proof storage preflight failure, evidence truncation, or validator failure.

Once poisoned, the runtime cannot resume or fall back within the process. A
new process and new session generation are required.

## Qualification gates

Gates are strictly ordered. A failed gate stops before the next state-changing
operation and preserves its evidence.

### G0: static artifact and provenance

Recompute all three xclbin hashes, extract candidate UUID/kernel/argument/bank
metadata, validate final timing, check routing/unrouted-net reports, record the
dirty-source provenance descriptor, and copy the candidate to a distinct
SHA-derived path. Re-hash source and destination. This gate does not program
the board.

### G1: software bridge and ABI

Use tests first to connect the production layer coordinator to the shared
persistent runtime. Prove Phase A and Phase B state transitions, both trace
builders, common assembler/cache use, four-CU descriptor coverage, poisoning,
and absence of a strict-mode direct-executor route. Add a symbol/ABI test for
all `hetgpu_iq1s_layer_*_v2` exports.

### G2: pinned runtime and llama overlay rebuild

Rebuild `libnvcuda.so` and the pinned llama.cpp overlay from recorded commits
and patch hashes. Verify exported symbols and embedded/current revision
identities before installing into a separate run directory. Build jobs remain
at or below 32. A stale library or old overlay fails this gate.

### G3: four-CU hardware ring smoke

After device-health preflight, program only the qualified candidate. Run a
bounded synthetic descriptor through every CU, validate command and completion
sequences, terminal status, bank-local addresses, numerical output, sticky
fault state, graceful shutdown, and post-test device health. This is hardware
control/data-path evidence, not model E2E evidence.

### G4: exact one-token real-model gate

Run a CUDA reference and then one strict hybrid decode token with the exact
GGUF and deterministic sampling. The hybrid result is accepted only when:

- the 141/39 routing audit and GPU-attention audit pass;
- every eligible operation is inside a real layer lifecycle record;
- the persistent CUs execute handwritten traces with no legacy eligible route
  and no fallback;
- all four CUs have matched nonzero commands and completions;
- weight DMA is zero after measurement start;
- sampled FFN results pass `atol=1e-4`, `rtol=1e-3`;
- the generated token ID exactly matches CUDA;
- the compact proof validator passes.

Repeat this gate with the AlgorithmTree compiler builder before performance
tuning. Failure is reported as a failed one-token gate, never as E2E TPS.

### G5: fixed aggregate-decode benchmark

Only after both G4 modes pass, run 64 requests, at most 16 active requests,
and exactly 32 generated tokens per completed request. After one warm-up, run
three handwritten and three compiler passes. The numerator is exactly 2,048
validated generated tokens and the denominator is last completion minus first
enqueue. Report every completed TPS value even if below target, but declare
the performance gate passed only if every required hybrid pass reaches at
least 15 aggregate generated tok/s and the final validator passes.

## Implementation and test boundaries

Implementation follows test-first slices: lifecycle/pool mocks, shared-cache
and poison tests, ABI/export tests, compact-ledger validator tests, static
runner checks, candidate qualifier, hardware ring smoke, and finally exact
model gates. Runtime work is confined to the isolated software worktree; RTL
work remains in the persistent RTL worktree. Existing dirty files and proof
artifacts are preserved, and commits stage only explicitly reviewed files.

The following are outside this execution slice:

- guaranteeing 15 tok/s before the fixed workload is measured;
- moving attention or non-IQ1_S tensors to U250;
- hiding latency with more than 16 active requests;
- accepting approximate token equality, inferred hardware execution, modeled
  TPS, partial output, fallback, or missing evidence;
- rebuilding RTL before the qualified candidate is shown to be inadequate.

## Review decision

Approval of this written addendum authorizes preparation of a detailed
implementation plan. It does not itself authorize programming the U250 or
starting the real-model run; those occur only at their named fail-closed gates
after the preceding evidence is complete.
