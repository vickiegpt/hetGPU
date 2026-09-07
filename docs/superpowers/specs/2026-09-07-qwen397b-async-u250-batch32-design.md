# Qwen3.5-397B Async GPU/U250 Batch-32 Design

## Goal

Run Qwen3.5-397B-A17B-UD-TQ1 with native GPU attention and strict U250 IQ1_S FFN routing, then qualify aggregate continuous-batch decode for 64 requests with 32 generated tokens per request and at most 32 simultaneously active requests. Reduce GPU/U250 synchronization and transfer calls without weakening the fail-closed correctness proof.

## Scope and constraints

- The selected model remains `/root/models/qwen35-tq1/Qwen3.5-397B-A17B-UD-TQ1_0.gguf`, SHA-256 `0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568`.
- The persistent image remains `/au250_xrt/xclbins/qwen397b_iq1s_layer_persistent_9c83dcae.xclbin`, SHA-256 `9c83dcae07b4c7bf1d2e1cebf46ccf0ff1ebf8848a437035fef1451dee7770a3`.
- All 141 IQ1_S routed-expert tensors stay assigned to the U250; the other 39 routed-expert tensors stay on the GPU.
- Attention, routing, normalization, recurrent layers, and non-IQ1_S expert operations remain GPU-native.
- The model context limit remains 262,144 tokens.
- Build parallelism remains limited to 32 jobs.
- The current U250 link is PCIe Gen3 x2 although the endpoint advertises x16 capability. Results must record this condition and must not generalize measured throughput to an x16 link.
- This change does not rebuild or overwrite the xclbin.

## Evidence from the current gate

Run `qwen35-iq1s-persistent-e2e-r15-overcommit1-64m-staging` established these boundaries:

- The CUDA reference model loaded and passed its one-token semantic gate.
- The handwritten mode loaded the same model and recorded 52 routes, including strict IQ1_S `ggml_type19` routes to `cxl_tmatmul`.
- The persistent phase ledger remained empty.
- The first handwritten completion stream ended without a final response or generated token.
- Kernel logs showed no OOM or XRT hardware fault. The container peaked at 20.1 GiB of host memory.
- The evaluator discarded the raw SSE event sequence on failure, so the current evidence does not identify whether the server returned a backend error, an empty stop event, or another terminal event.

This is not an accepted E2E result. Diagnostic preservation is required before changing execution timing.

## Architecture

### Correctness and performance profiles

The runtime exposes two deliberately separate profiles:

1. `one-token` correctness keeps deterministic sampling, strict routing, CUDA launch blocking, synchronous proof publication, sampled numerical comparison, and one active request. This profile is the prerequisite oracle.
2. `full` performance uses 64 total requests, 32 active requests, and 32 generated tokens per request. It preserves strict routing and token/proof validation but removes global CUDA launch blocking and per-phase proof `fsync` from the timed window.

The runner must refuse to start `full` unless a fresh, matching `one-token` proof has `validation.status=accepted`. Binary, model, xclbin, route-manifest, and model-audit hashes must match between the two profiles.

### Per-stream double buffering

Each active CUDA stream owns two reusable transaction slots. Each slot contains:

- pinned host storage for expert IDs, expert bounds, packed Q8 activations, token maps, and reconstructed results;
- CUDA events for input readiness and output-copy enqueue accounting;
- an XRT submission ticket containing transaction, generation, phase, command ranges, expected completion counters, and slot generation;
- one of two disjoint activation, token-map, and output slab regions for every CU.

A slot may return to `free` only after its XRT completions have been validated and all H2D result copies have been enqueued on the originating CUDA stream. Slot generation is checked on every transition so late completions fail closed instead of corrupting a reused slot.

### Phase A data flow

1. Layer begin allocates a free slot and enqueues the existing top-10 route copy.
2. The first IQ1_S gate launch records its CUDA 13 MoE layout rather than synchronously copying IDs, bounds, and activation.
3. The IQ1_S up launch must have an identical route/bounds/activation identity. It contributes its matrix and output bindings while reusing the gate capture.
4. Once gate and up are present, one `cuMemcpyBatchAsync_v2` operation copies the shared IDs, bounds, activation, and route data into pinned slot storage and records one input-ready event.
5. Phase A waits only for that event. It builds or retrieves both projection traces, publishes all four CU command ranges, then rings all four doorbells.
6. Completion polling is round-robin across all CUs against one deadline. The host does not wait for CU 0 to finish before observing CU 1 through CU 3.
7. Completion records and result ranges are synchronized only after their producer counters reach the expected ticket values.
8. Reconstructed gate and up outputs are copied with one batched H2D enqueue on the originating CUDA stream. The runtime does not perform a context-wide synchronization; downstream same-stream CUDA work observes normal stream ordering.

### Phase B data flow

The GPU must compute SiLU(gate) times up before down projection input exists, so Phase B is a real dependency boundary with the current xclbin. Phase B uses the alternate slot when available, captures the down input once, submits all selected experts to the persistent four-CU ring, validates completions, performs route-weighted reconstruction, and enqueues one batched H2D result copy.

This design therefore targets at most one CUDA input event wait and one XRT completion wait per phase, not zero waits per layer. A single-boundary layer requires a future fused gate/up/activation/down/reduce xclbin and is outside this change.

### Batch-32 configuration

- Increase per-stream positional token and expert-ID scratch capacity from 16x10 to 32x10.
- Reject an active batch outside 1 through 32 before any copy or XRT submission.
- Configure `--parallel 32 --batch-size 32 --ubatch-size 32` for the `full` profile.
- Execute 64 requests as two waves of 32 while preserving request IDs and per-request token sequences.
- Keep `one-token` at `--parallel 1`; its scratch uses the same allocation code but only the first ten entries.
- Do not silently lower the active batch after a CUDA allocation or model-fit failure. Such a failure invalidates that performance attempt.

## XRT executor changes

The existing persistent kernels remain continuously running. Host code changes from a single blocking `submit_phase` operation to a ticket lifecycle:

- `prepare_ticket` validates and relocates descriptors, assigns a free slot, and records expected producer counters.
- `publish_ticket` writes merged command, activation, and token-map ranges for all four CUs and rings their doorbells.
- `poll_ticket` checks every CU in round-robin order, validates fault registers and the common timeout, and returns pending or complete.
- `collect_ticket` synchronizes only completed ranges, validates every descriptor/completion identity, and releases the slot after result ownership transfers to the caller.

Adjacent activation, token-map, command, completion, and output ranges are merged before XRT BO synchronization. Weight chunks remain resident; any weight DMA inside the measured window poisons the pool.

The synchronous correctness API remains as a wrapper around the ticket lifecycle so existing proof semantics and unit tests stay valid.

## Diagnostic and proof behavior

- Every completion request writes an atomic `stream-events.json` containing every decoded SSE event, including terminal error payloads. If parsing itself fails, the file contains all events accepted before the failure plus the parser error.
- Persistent initialization emits bounded progress records for arena plan acceptance, each 64 MiB-or-smaller weight chunk, pool readiness, and phase submit/complete. Records contain hashes, sizes, offsets, and stage names, not model data.
- Runtime errors include transaction, layer, phase, stream, and slot generation.
- The phase ledger remains the authoritative hardware proof. Progress records do not count as completions.
- Correctness mode synchronizes each phase ledger append before CUDA output publication.
- Performance mode writes the same records but moves ledger synchronization outside the timed measurement. A crash before final synchronization invalidates the run.
- Empty SSE streams, missing final events, GPU fallback, incomplete CU participation, stale tickets, nonfinite output, sampled numerical mismatch, or token mismatch are terminal failures.

## Measurements

The accepted performance summary reports:

- aggregate decode tokens per second for exactly 64 requests times 32 tokens;
- median and tail request latency, queue latency, TTFT, and service time;
- active batch and number of waves;
- GPU utilization and memory placement;
- per-phase D2H and H2D bytes;
- CUDA copy batches and event waits;
- XRT command, activation, token-map, completion, result, program, and weight DMA ranges;
- four-CU command and completion counts;
- trace/program/weight cache hit rates;
- Phase A and Phase B prepare, publish, device wait, collect, reconstruction, and copy timings;
- negotiated and maximum U250 PCIe speed and width.

The report must label the 15 tok/s target as met only when the sealed `full` proof reports at least 15 aggregate decode tok/s. Projection, one-token latency, process liveness, or unsealed raw output cannot satisfy the target.

## Verification strategy

1. Python tests reproduce an SSE terminal error and prove the raw event/error artifact survives the exception.
2. C/C++ static/runtime tests prove 32x10 scratch bounds, one allocation per stream, shared gate/up capture, and rejection of batch 33.
3. Rust unit tests prove two-slot lifecycle safety, round-robin completion polling, merged range synchronization, timeout/fault poisoning, and no weight DMA during measurement.
4. Existing arena, trace-builder, assembler, route, validator, and proof tests remain green.
5. The runtime is rebuilt with at most 32 compilation jobs and its manifest hashes are revalidated.
6. A new strict one-token hardware run captures the previously lost SSE/backend error or passes with a complete four-CU phase ledger.
7. Only after step 6 passes, run the `full` profile for 64 requests times 32 tokens at 32 active requests and seal the aggregate throughput proof.

## Completion criteria

The work is complete only when:

- the strict one-token CUDA and handwritten/compiler modes pass with matching tokens and accepted numerical evidence;
- the performance run contains no eligible fallback, stale slot, missing CU, weight DMA, or proof error;
- the sealed summary contains aggregate TPS for exactly 2,048 generated tokens;
- all relevant unit, static, validator, and integration tests pass;
- the original xclbin remains byte-identical and its SHA-256 is recorded.
