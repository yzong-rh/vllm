#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
cutlass_bf16_memtest_multisize.py — Multi-size workspace poison test for
FlashInfer CUTLASS BF16 MoE under a single CUDA graph memory pool.

This extends cutlass_bf16_memtest.py by:

  - Building the same token-count list as vLLM's default
    ``cudagraph_capture_sizes`` (throughput mode, capped at 512): 51 sizes
    ``[1, 2, 4] + range(8, 256, 8) + range(256, 513, 16)`` (deduped, sorted).
    See vllm/config/vllm.py ``_set_cudagraph_sizes``.

  - Capturing one CUDA graph per size in vLLM order: **largest batch first**
    (reverse of that sorted list), all sharing ``torch.cuda.graph_pool_handle()``,
    matching vllm/v1/cudagraph_dispatcher.py ``get_capture_descs``.

  - After all captures: one memory snapshot, then ``N_ROUNDS`` replay rounds.
    Each round replays all 51 graphs in a random order with poisoned freed segments
    between replays.

Usage:
    python cutlass_bf16_memtest_multisize.py

For a quicker smoke test, lower ``MAX_CUDAGRAPH_CAPTURE_SIZE`` (e.g. 64) to
capture fewer shapes; the full 51-graph run is memory- and time-heavy.
"""

import ctypes
import random
import sys

import torch
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType

# ── Configuration ────────────────────────────────────────────────────────
HIDDEN = 1024
INTERMEDIATE = 1024
NUM_EXPERTS = 64
TOPK = 6
DTYPE = torch.bfloat16
POISON_BYTE = 0xDE

# Match vLLM default decode capture list when max is 512 (51 graphs).
MAX_CUDAGRAPH_CAPTURE_SIZE = 512

SEED = 42
N_ROUNDS = 10000


def vllm_default_cudagraph_capture_sizes(
    max_cudagraph_capture_size: int,
) -> list[int]:
    """Same construction as vllm.config.vllm.VllmConfig._set_cudagraph_sizes."""
    cudagraph_capture_sizes = [i for i in [1, 2, 4] if i <= max_cudagraph_capture_size]
    if max_cudagraph_capture_size >= 8:
        cudagraph_capture_sizes += list(
            range(8, min(max_cudagraph_capture_size + 1, 256), 8)
        )
    if max_cudagraph_capture_size >= 256:
        cudagraph_capture_sizes += list(range(256, max_cudagraph_capture_size + 1, 16))
    return sorted(set(cudagraph_capture_sizes))


# ── Helpers ──────────────────────────────────────────────────────────────


def load_cudart():
    """Load libcudart and configure cudaMemsetAsync prototypes."""
    lib = ctypes.CDLL("libcudart.so")
    lib.cudaMemsetAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib.cudaMemsetAsync.restype = ctypes.c_int
    return lib


def make_per_token_tensors(num_tokens: int):
    """Activations and routing for batch size num_tokens (weights are shared separately)."""
    a = torch.randn(num_tokens, HIDDEN, device="cuda", dtype=DTYPE) / 10

    # All tokens route to the same TopK experts, repeated TOPK times for API shape.
    topk_experts = (
        torch.arange(TOPK, device="cuda", dtype=torch.int32)
        .unsqueeze(0)
        .expand(num_tokens, -1)
    )
    perm = torch.argsort(
        torch.rand(num_tokens, TOPK, device="cuda", dtype=torch.float32), dim=-1
    )
    ti = torch.gather(topk_experts, dim=1, index=perm).contiguous()

    # # All tokens route to the same expert (id 0), repeated TOPK times for API shape.
    # SAME_EXPERT = 0
    # ti = torch.full(
    #     (num_tokens, TOPK), SAME_EXPERT, device="cuda", dtype=torch.int32
    # )
    tw = torch.full(
        (num_tokens, TOPK),
        1.0 / TOPK,
        device="cuda",
        dtype=torch.float32,
    )

    # # Random expert routing
    # scores = torch.randn(num_tokens, NUM_EXPERTS, device="cuda", dtype=DTYPE)
    # tw, ti = torch.topk(scores, TOPK, dim=-1)
    # tw = torch.softmax(tw.float(), dim=-1)
    # ti = ti.to(torch.int32)

    out = torch.zeros(num_tokens, HIDDEN, device="cuda", dtype=DTYPE)
    return a, tw, ti, out


def call_moe(a, w1, w2, tw, ti, out, tune_max_num_tokens: int):
    """One forward pass through FlashInfer CUTLASS BF16 fused-MoE."""
    out.zero_()
    cutlass_fused_moe(
        input=a,
        token_selected_experts=ti,
        token_final_scales=tw,
        fc1_expert_weights=w1,
        fc2_expert_weights=w2,
        output=out,
        output_dtype=DTYPE,
        quant_scales=[],
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        activation_type=ActivationType.Swiglu,
        tune_max_num_tokens=tune_max_num_tokens,
    )


def print_snapshot(snap, label):
    """Pretty-print all segments from a torch.cuda.memory_snapshot."""
    print(f"\n{'─' * 64}")
    print(f"  {label}")
    print(f"{'─' * 64}")
    for seg in snap:
        pool = tuple(seg.get("segment_pool_id", (0, 0)))
        kb = seg["total_size"] // 1024
        alloc = seg["allocated_size"] // 1024
        active = seg["active_size"] // 1024
        sid = seg.get("stream", 0)
        print(
            f"  pool={pool}  stream={sid}  "
            f"size={kb} KB  allocated={alloc} KB  active={active} KB"
        )
        for blk in seg.get("blocks", []):
            req = blk.get("requested_size", 0)
            if req > 0:
                print(
                    f"    └─ requested_size={req // 1024} KB  "
                    f"state={blk.get('state', '?')}"
                )
    if not snap:
        print("  (no segments)")
    print(f"{'─' * 64}\n")


def poison_freed_segments(snap, cudart, stream: torch.cuda.Stream):
    """Overwrite all freed segments with garbage via cudaMemsetAsync on ``stream``."""
    n = 0
    stream_ptr = ctypes.c_void_p(stream.cuda_stream)
    for seg in snap:
        if seg["allocated_size"] == 0:
            addr = seg["address"]
            size = seg["total_size"]
            rc = cudart.cudaMemsetAsync(
                ctypes.c_void_p(addr),
                POISON_BYTE,
                ctypes.c_size_t(size),
                stream_ptr,
            )
            assert rc == 0, f"cudaMemsetAsync error {rc}"
            n += 1
    return n


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    torch.accelerator.set_device(0)
    torch.manual_seed(SEED)
    random.seed(SEED)

    sizes = vllm_default_cudagraph_capture_sizes(MAX_CUDAGRAPH_CAPTURE_SIZE)
    capture_order = list(reversed(sizes))

    print(
        "FlashInfer CUTLASS BF16 MoE — multi-size graph pool poison test\n"
        f"  sizes={len(sizes)} (max_capture={MAX_CUDAGRAPH_CAPTURE_SIZE})  "
        f"hidden={HIDDEN}  inter={INTERMEDIATE}  experts={NUM_EXPERTS}  topk={TOPK}\n"
        f"  capture order: largest-first (e.g. first T={capture_order[0]})\n"
        f"  replay: {N_ROUNDS} round(s), inner order shuffled each round (seed={SEED})\n"
    )

    cudart = load_cudart()

    # One shared weight pair; per-T activations and outputs.
    w1 = (
        torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, device="cuda", dtype=DTYPE)
        / 10
    )
    w2 = torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, device="cuda", dtype=DTYPE) / 10

    tensors: dict[int, tuple] = {}
    for t in sizes:
        tensors[t] = make_per_token_tensors(t)

    eager: dict[int, torch.Tensor] = {}
    print(f"[1/4] Eager reference for {len(sizes)} batch sizes …")
    for t in sizes:
        a, tw, ti, out = tensors[t]
        call_moe(a, w1, w2, tw, ti, out, tune_max_num_tokens=t)
        torch.accelerator.synchronize()
        eager[t] = out.clone()

    pool = torch.cuda.graph_pool_handle()
    stream = torch.cuda.Stream()
    graphs: dict[int, torch.cuda.CUDAGraph] = {}

    print(
        f"[2/4] CUDA-graph capture (shared pool, largest-first), "
        f"{len(capture_order)} graphs …"
    )
    for i, t in enumerate(capture_order):
        a, tw, ti, out = tensors[t]
        graph = torch.cuda.CUDAGraph()

        # Warm-up on capture stream
        with torch.cuda.stream(stream):
            call_moe(a, w1, w2, tw, ti, out, tune_max_num_tokens=t)
        torch.accelerator.synchronize()

        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                call_moe(a, w1, w2, tw, ti, out, tune_max_num_tokens=t)
        torch.accelerator.synchronize()
        graphs[t] = graph
        if (i + 1) % 10 == 0 or i + 1 == len(capture_order):
            print(f"  captured {i + 1}/{len(capture_order)} (T={t})")

    print("[3/4] Memory snapshot …")
    print(f"  graph pool handle = {pool}")
    print(f"  stream = {stream.cuda_stream}")
    snap = torch.cuda.memory_snapshot()
    print_snapshot(snap, "All segments after capture")

    print(f"[4/4] Shuffled replay ({N_ROUNDS} round(s)) …")
    for r in range(N_ROUNDS):
        round_order = sizes.copy()
        random.shuffle(round_order)
        for t in round_order:
            with torch.cuda.stream(stream):
                poison_freed_segments(snap, cudart, stream)
                graphs[t].replay()
            torch.accelerator.synchronize()
            a, tw, ti, out = tensors[t]
            diff = (eager[t] - out).abs()
            if not torch.equal(eager[t], out):
                print(
                    f"\n  FAIL — round={r} T={t}  "
                    f"max |eager-replay|={diff.max().item():.6e}  "
                    f"mean={diff.mean().item():.6e}"
                )
                sys.exit(1)
        if (r + 1) == N_ROUNDS or (r + 1) % max(1, N_ROUNDS // 5) == 0:
            print(f"  round {r + 1}/{N_ROUNDS} OK (all {len(sizes)} graphs)")

    print("\n  PASS — every shuffled replay matches eager after workspace poisoning.")


if __name__ == "__main__":
    main()
