# Qwen3.5-397B GPU/U250 28 TPS Design

## Goal

Qualify at least 28 aggregate decode tokens per second for
Qwen3.5-397B-A17B-UD-TQ1 using GPU attention and U250 IQ1_S FFN execution.
The accepted workload is exactly 64 requests with 32 generated tokens per
request, at most 32 active requests, and two waves. Correctness remains
fail-closed: a projection, partial run, fallback result, or unsealed proof is
not a throughput result.

The workload produces 2,048 measured tokens. Meeting 28 aggregate TPS requires
each accepted measurement to finish its timed region within 73.143 seconds.
With two 32-request waves, this is an average budget of 1.143 seconds per
batch-32 decode step, or about 19.0 milliseconds per model layer across GPU
work, transfers, U250 work, and synchronization.

## Fixed contract

- Model: `/root/models/qwen35-tq1/Qwen3.5-397B-A17B-UD-TQ1_0.gguf`.
- Model SHA-256:
  `0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568`.
- Model context limit: 262,144 tokens.
- GPU: attention, routing, normalization, recurrent layers, and the 39 routed
  tensors that are not IQ1_S.
- U250: all 141 IQ1_S routed-expert tensors.
- Full profile: 64 requests, 32 generated tokens per request, 32 active
  requests, two waves, and three measurements.
- Build parallelism: at most 32 jobs.
- Sampling: deterministic and identical to the accepted CUDA reference.
- Existing xclbin: keep the file byte-identical at
  `/au250_xrt/xclbins/qwen397b_iq1s_layer_persistent_9c83dcae.xclbin`.
- Existing xclbin SHA-256:
  `9c83dcae07b4c7bf1d2e1cebf46ccf0ff1ebf8848a437035fef1451dee7770a3`.
- Existing xclbin UUID: `b1bafc64-09fd-32b0-a5b4-a881e554ae84`.
- A new fused xclbin, if required, receives a new filename and independently
  recorded SHA-256 and UUID.

## Current evidence boundary

The strict run in
`/mnt/disk0/qwen397b-proof/async-generation-fix-gate-20260908T0038Z`
established the following:

- The CUDA reference passed and generated token ID 321.
- The model audit passed with 1,098 tensors and 180 routed tensors: 141 IQ1_S,
  24 IQ2_XXS, 4 IQ3_S, and 11 MXFP4.
- XRT/XDMA made all 992 weight chunks resident and recorded exactly
  59,139,686,400 bytes, followed by `pool_ready`.
- The handwritten Phase A stopped before submission because the trace validator
  rejected a repeated CUDA activation pointer.
- The authoritative phase ledger remained empty; compiler mode and the full
  workload did not run.
- The negotiated U250 link was PCIe 8 GT/s x2 while the endpoint advertised
  x16 capability.

This is a failed gate, not an E2E result. It proves resident weight loading but
does not prove a completed U250 phase or hybrid token.

## Chosen strategy

Use a staged strategy rather than immediately rebuilding the FPGA image:

1. Repair the activation identity model and pass the strict one-token gate.
2. Inspect and, where safely possible, restore the U250 link to PCIe Gen3 x16.
3. Measure the fixed 64-by-32 workload with the existing persistent xclbin.
4. If any of the three measurements is below 28 aggregate TPS, implement a new
   layer-level fused full-FFN persistent kernel and repeat qualification.

This provides a measured baseline before the longer RTL build while retaining
the fused kernel as the planned path when the two-phase image cannot meet the
latency budget.

## Activation identity and reuse

The current trace records the first CUDA pointer of each packed expert group and
requires these pointers to be globally unique. This is incompatible with MoE:
the same token activation is intentionally selected by multiple top-10 experts,
and Gate and Up consume the same input.

Replace this pointer-uniqueness rule with an ordered lane-source identity. The
identity covers:

- transaction, layer, phase, and originating CUDA stream;
- every lane's global token ID in order;
- the captured CUDA allocation identity and byte range;
- the packed activation length and a digest of the already captured host bytes;
- the route identity used to construct the lane set.

Two expert groups may reuse an activation slab only when the complete identity
and packed bytes match. A repeated pointer alone is neither an error nor proof
of equality. Different lane order, length, content, stream, transaction, or
allocation generation creates a distinct range. Slab ranges must remain aligned
and either exactly shared by an identical identity or non-overlapping.

Gate and Up share the existing input capture and, when their expert lane sets
match, the U250 activation slab. Exact lane-set reuse across experts is also
permitted. All mismatches poison the transaction before any command doorbell.

## Existing-xclbin data flow

The existing image retains the real Phase A/Phase B dependency:

1. GPU computes attention, routing, and the top-10 expert IDs and weights.
2. The runtime performs one shared asynchronous capture of route data and the
   Gate/Up activation for the layer transaction.
3. Phase A groups all Gate and Up commands by expert and lane set, merges
   adjacent activation, token-map, program, and command ranges, and publishes
   all four CU ranges before ringing their doorbells.
4. Completion polling visits all four CUs round-robin under one deadline. It
   does not wait for CU 0 before observing the other CUs.
5. All Gate/Up results are reconstructed and submitted to CUDA with one batched
   asynchronous copy on the originating stream.
6. GPU computes SiLU(Gate) times Up, creating the Down input.
7. Phase B performs one corresponding four-CU submission and one batched result
   publication.
8. GPU applies routing weights and completes the residual path.

There is no context-wide CUDA synchronization. Same-stream events express the
true GPU-to-U250 and U250-to-GPU dependencies. Two transaction slots allow one
slot to be collected while the other is being prepared, but a slot is not
reused until its XRT completion and CUDA publication ownership are proven.

## Fused full-FFN persistent kernel

If the existing image misses 28 TPS, create a new four-CU kernel that consumes
one layer transaction and performs:

1. IQ1_S Gate projection.
2. IQ1_S Up projection.
3. SiLU(Gate) multiplied by Up.
4. IQ1_S Down projection.
5. Top-10 route-weighted reduction.

All 141 IQ1_S tensors stay resident in the U250 DDR banks. Expert/component
commands are grouped into one layer program, and each CU continuously consumes
its ring without per-matrix host launches. Weight DMA is forbidden after the
measurement baseline is taken.

For an offloaded layer, the steady-state boundary contains only:

- GPU to U250: the layer activation, ordered top-10 IDs, and route weights;
- U250 to GPU: the final reduced FFN result.

Intermediate Gate, Up, activation, Down, and per-expert output tensors do not
cross PCIe. Layers whose routed tensor is one of the 39 non-IQ1_S tensors remain
entirely on the GPU. Attention remains on the GPU for every layer.

The host publishes a single layer descriptor covering all four CUs, then uses a
single layer completion epoch. The proof still contains per-CU command and
completion counts so a missing CU cannot be hidden by the layer-level boundary.

## PCIe handling

Every gate and performance run records the negotiated and maximum link speed
and width. The current 8 GT/s x2 state is diagnostic evidence, not a reason to
project x16 performance.

Before performance qualification, perform read-only topology and AER checks.
Any retrain, reset, power cycle, slot change, or firmware operation is a separate
hardware action and requires its own exact target validation. After an approved
action, re-read the live BDF, link, xclbin UUID, device health, and memory-bank
state before running the model.

A measurement taken at x2 remains valid for that configuration, but it cannot
be relabeled as x16. The 28 TPS claim depends only on the measured accepted run,
not on link-width projections.

## Measurement and admission

The performance runner may start only from a fresh accepted one-token proof
whose model, runtime, llama binary, route manifest, tensor audit, xclbin SHA/UUID,
and four-CU identity match the performance attempt.

Each of three measurements must satisfy all of the following:

- exactly 64 requests and 32 generated tokens per request;
- exactly 2,048 generated tokens and at most 32 simultaneously active requests;
- two waves with stable request IDs and complete per-request token sequences;
- all token IDs equal to the CUDA reference under deterministic sampling;
- aggregate TPS of at least 28, computed as 2,048 divided by the measured wall
  time for that measurement;
- timed wall time no greater than 73.143 seconds;
- no eligible GPU fallback for an IQ1_S operation;
- zero weight DMA bytes and ranges in the timed window;
- finite outputs, accepted sampled numerical comparison, and complete four-CU
  participation;
- no stale ticket, slot reuse, XRT timeout, firewall error, or device fault.

Report the minimum, median, and maximum of the three accepted TPS values. The
target is met only if the minimum is at least 28; the fastest attempt is never
reported alone.

## Instrumentation

Collect timings at layer and phase boundaries without adding per-command log or
`fsync` operations to the timed path. Required metrics are:

- GPU attention/recurrent duration;
- route capture and activation packing duration;
- GPU-to-U250 and U250-to-GPU bytes and copy batches;
- Phase A and Phase B prepare, publish, device wait, collect, reconstruction,
  and CUDA publication duration for the existing image;
- fused layer prepare, publish, device wait, collect, and publication duration
  for the new image;
- per-CU command, completion, cycle, matrix-byte, input-byte, and output-byte
  counters;
- CU busy balance and trace/program/activation/weight cache hit rates;
- GPU utilization and memory use;
- negotiated PCIe link speed and width.

Correctness mode synchronizes proof publication. Performance mode buffers proof
records and seals them after the timed region. A crash or incomplete seal
invalidates the measurement.

## Failure behavior

The runtime fails closed before publishing a token on any of these conditions:

- activation lane identity or packed bytes differ after claimed reuse;
- a slab range is misaligned, partially overlaps, or exceeds its slot;
- route IDs do not exactly cover the active batch or top-10 contract;
- a registered tensor, allocation generation, model hash, xclbin hash, or CU
  identity changes;
- a completion is missing, duplicated, stale, timed out, or produced by the
  wrong CU/slot generation;
- output reconstruction has gaps, overlaps, non-finite values, or fails sampled
  comparison;
- an IQ1_S launch falls back to GPU;
- the evaluator receives an error, an incomplete SSE stream, or the wrong token
  sequence;
- any measured weight DMA occurs.

Progress records, resident-weight records, process liveness, and a nonempty raw
output directory are diagnostic only. The sealed validator result is the sole
acceptance authority.

## Verification sequence

1. Add a regression that reproduces legal top-10 and Gate/Up pointer sharing,
   and observe the current validator fail.
2. Implement ordered lane-source identity and exact slab reuse; run the focused
   trace, layer coordinator, persistent runtime, and XRT tests.
3. Run all Qwen evaluator, proof, gate, static overlay, and source-preparation
   tests with at most 32 build jobs.
4. Rebuild the runtime and record artifact hashes.
5. Run a new strict CUDA/handwritten/compiler one-token gate. Require matching
   tokens, nonempty Phase A and Phase B ledgers, accepted sampled numerical
   evidence, and complete four-CU activity.
6. Inspect and, if separately approved, repair the PCIe link; revalidate device
   and image identity afterward.
7. Run three full measurements with the existing xclbin.
8. If any measurement is below 28 TPS, implement, build, and qualify the fused
   image without overwriting the existing xclbin.
9. Repeat the strict one-token gate and three full measurements with the fused
   image.
10. Run final validators, verify the clean scope of the intended commit, commit,
    and push only after the complete proof is accepted.

## Completion criteria

The 28 TPS objective is complete only when the sealed proof contains three
accepted 64-by-32 measurements, each at or above 28 aggregate TPS, with matching
CUDA tokens, complete U250 hardware evidence, and no fallback or measured weight
DMA. The report must identify the exact runtime, model, xclbin, live PCIe link,
and proof directory. Until then, status remains partial or failed regardless of
projected throughput.
