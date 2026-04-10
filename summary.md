# Bug Summary: FlashInfer CUTLASS BF16 MoE + CUDA Graphs

## What the bug is

A hard CUDA error ("illegal memory access was encountered") when running the
Qwen3-30B-A3B MoE model with **data parallelism**, **CUDA graphs** (both FULL
and PIECEWISE modes), and **FlashInfer CUTLASS BF16** MoE backend. The crash
kills both worker processes and the server. In piecewise mode, instrumentation
shows the crash always occurs during replay of submod_2 — the post-attention
subgraph containing MoE layers, residual connections, and NCCL collectives.

## Crash characteristics

- **Not on the first replay** — the same graph object at the crashing batch
  size has already succeeded many times in the same run. The failure is
  non-deterministic.
- **Both DP ranks crash simultaneously** on the same replay with the same
  batch descriptor.
- **Larger batch sizes** (~144–150 tokens) crash. Smaller decode-only batches
  (e.g. 56 tokens) never crash across hundreds of replays.
- **Crash point ranges from 3% to 90%** through the evaluation across
  different runs.

## Conditions that PASS vs FAIL

### Conditions that always pass

| Config change | Why it matters |
|---|---|
| Single GPU (no DP) | Removes DP padding, NCCL collectives, and DP coordination entirely |
| DP + eager mode (cudagraph=NONE) | Same DP code path, but no graph capture/replay |
| FP8 + DP + graphs (same flashinfer cutlass backend) | Same NCCL ops and DP logic, different data type and kernel variant |
| DP + graphs + flashinfer attention backend | Different attention backend; graph structure changes |
| Single GPU + compute-sanitizer (eager and graph) | No memory errors detected at driver level |

### Conditions that often fail

| Config change | What it tells us |
|---|---|
| DP + graphs + flash_attn (default attention) | The default failing configuration |
| DP + graphs + triton_attn | Different attention backend, same failure — not flash_attn-specific |
| DP + graphs + autotuning disabled | Autotuning is not the cause |
| DP + graphs + cuda.synchronize between every graph replay | Inter-replay races are not the cause |
| DP + graphs + event.record/stream.wait around every PyNCCL op | Stream synchronization around NCCL doesn't help |
| DP + graphs + zeroed padding input_ids and positions | Stale padding data in input_ids/positions is not the cause |

### Edge case

| Config change | Outcome |
|---|---|
| CUDA_LAUNCH_BLOCKING=1 | Often passes but not always — changes timing, shrinks whatever window exists |

## vLLM background relevant to the bug

### Graph replay model

In vLLM, one graph is selected by batch size and replayed sequentially.
Multiple graphs sharing a pool never execute simultaneously.

### Graph pool and allocations

During `torch.cuda.graph(pool=...)`, every `torch.empty()` in the forward
pass allocates from the graph-private pool. This includes NCCL output buffers
(e.g. `cuda_communicator.py:380`), MoE workspace, attention intermediates,
layer norm temporaries, and all other buffers. Freed blocks are recycled
within the same capture. This is safe under single-stream replay because
kernel execution order is preserved.

### MoE workspace lifecycle

The CUTLASS MoE workspace is allocated as a local variable inside
`FusedMoeRunner::runMoe()` (in FlashInfer's
`fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu`). The
allocation chain is `getWorkspaceInfo()` (line 840) → `alloc_tensor()`
(`tvm_ffi_utils.h:316`) → TVM FFI → DLPack → `at::empty()` → PyTorch's
`CUDACachingAllocator`. During graph capture, this lands in the graph-private
pool. The destructor (`tvm_ffi/include/tvm/ffi/container/tensor.h:217`)
frees the block back to the pool when `runMoe()` returns (line 423).

The kernel sequence captured in the graph for each MoE layer is:

1. `fusedBuildExpertMapsSortFirstToken` — routing kernel, writes to workspace
2. `computeStridesTmaWarpSpecializedKernel` — writes TMA strides to workspace
3. GEMM1 — reads workspace
4. GEMM2 — reads workspace

Steps 1–2 fully initialize the workspace before steps 3–4 read it.

### MoE kernel is self-consistent (poison test)

The poison test (`cutlass_bf16_memtest.py`) overwrote the freed workspace
segments with garbage between replays. Result: graph replay still matched
eager output. A memory snapshot around a single capture shows the graph pool
segments are fully freed after `runMoe()` returns:

### Why compute-sanitizer sees nothing

`compute-sanitizer` tracks driver-level `cudaMalloc`/`cudaFree` boundaries.
All graph-pool addresses stay within valid `cudaMalloc`'d regions, so no
hardware-level memory errors are reported — even though graph-pool-internal
reuse could theoretically cause logical corruption.

### PyNCCL uses the capture stream

All PyNCCL operations default to `current_stream()` (`pynccl.py:175-176`).
During graph capture, this is the capture stream. NCCL collectives and
compute kernels are serialized on one stream within the graph.

### Naive DP EP Prepare/Finalize

When DP=true and EP=false, the naive DP EP Prepare/Finalize path is used
(`prepare_finalize/naive_dp_ep.py`). It uses synchronous AllGather (dispatch)
and ReduceScatter (combine) via PyNCCL on the current stream. No async
dispatch/combine, no separate streams.

### DP padding

When CUDA graphs are enabled with DP, all ranks pad to the same token count
(the max across ranks) so they replay the same graph bucket. This
coordination happens in `coordinate_batch_across_dp()` (`dp_utils.py`).
Padding positions get zeroed `input_ids` and `positions`
(`gpu_model_runner.py:3279-3284`).

### Piecewise graph structure

In piecewise mode, the model is split at attention ops:

- submod_0 / submod_1: pre-attention (embedding, RMSNorm, QKV projection)
- Attention runs eagerly between submods
- submod_2: post-attention (MoE with NCCL dispatch/combine, residual, etc.)

### Instrumentation

`torch.cuda.synchronize()` was added around each piecewise CUDA graph replay
in `CUDAGraphWrapper.__call__` (`vllm/compilation/cuda_graph.py`), logging
the submod name, piecewise index, batch descriptor, and DP step metadata
(local num_toks, num_reqs, max_query_len per rank). 4 runs were performed
(DP=2, Qwen3-30B-A3B BF16, piecewise cudagraph, flash_attn backend). All 4
crashed, always during submod_2 replay.

## Hypotheses that can be ruled out

### 1. "CUTLASS kernel bug triggered by DP problem shapes"

**Ruled out.** With DP (AllGather dispatch), all ranks see the same gathered
tokens — DP doesn't change per-expert problem shapes, just the total token
count. Single-GPU tests with matching batch sizes pass. If the kernel itself
were buggy for these shapes, DP eager mode would also fail — it does not.

### 2. "Use-after-free of MoE workspace in graph pool"

**Ruled out.** The workspace is freed back to the graph pool after capture,
but this is standard caching-allocator behavior. The poison test
(`cutlass_bf16_memtest.py`) confirmed that corrupting the freed workspace
address between replays has no effect on MoE output — the kernel fully
reinitializes the workspace before reading it. `compute-sanitizer` also
reports zero errors on single GPU.

### 3. "Race between different submod replays"

**Ruled out.** Inserting `torch.cuda.synchronize()` between every piecewise
graph replay in `CUDAGraphWrapper.__call__`
(`vllm/compilation/cuda_graph.py`) still crashes. The crash is always
detected *within* a submod_2 replay, not between replays.

### 4. "Cross-stream race between NCCL and compute kernels"

**Ruled out.** The naive DP EP path (`prepare_finalize/naive_dp_ep.py`) uses
synchronous dispatch/combine — no async operations, no separate streams.
Inserting explicit stream events around every PyNCCL operation doesn't help.
PyNCCL uses `current_stream()` (`pynccl.py:175-176`), so within a graph
capture, NCCL and compute kernels are on the same stream.

### 5. "NCCL/PyNCCL has a graph-mode bug"

**Effectively ruled out.** NCCL and PyNCCL are well-exercised in the codebase
and trusted. FP8 with DP + graphs uses the exact same NCCL collective
operations (AllGather dispatch, ReduceScatter combine) on the same stream and
passes reliably. If NCCL had a graph-mode bug, FP8 would fail too.

### 6. "Stale data in padding token input_ids/positions"

**Ruled out.** Zeroing `input_ids` and `positions` for DP padding positions
(`gpu_model_runner.py:3279-3284`) before each step still crashes. The crash
is also non-deterministic — the same graph with the same padding layout
succeeds many times before failing. A data-content issue would be
deterministic for a given graph bucket.
