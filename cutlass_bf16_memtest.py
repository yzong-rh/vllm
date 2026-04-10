#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
cutlass_bf16_memtest.py — Workspace poison test for FlashInfer CUTLASS BF16
MoE under CUDA graphs.

Shows two things:

  1. Memory snapshot after graph capture: the CUTLASS workspace segments in
     the graph pool are fully freed (allocated=0, active=0), confirming the
     use-after-free defect described in bug.md.

  2. Overwriting those freed segments with garbage (0xDE) and replaying the
     graph: output still matches eager, proving the MoE kernel writes its
     workspace before reading (routing kernel → TMA strides → GEMMs).

Usage:
    python cutlass_bf16_memtest.py
"""

import ctypes
import sys

import torch
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType

# ── Configuration ────────────────────────────────────────────────────────
# Moderate dimensions that exercise the workspace allocation path without
# requiring excessive GPU memory.
NUM_TOKENS = 384  # > 256 will cause error in cutlass_fused_moe when all tokens route to the same expert
HIDDEN = 1024
INTERMEDIATE = 1024
NUM_EXPERTS = 64
TOPK = 6
DTYPE = torch.bfloat16
POISON_BYTE = 0xDE


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


def make_tensors():
    """Create random BF16 MoE inputs, weights, and routing tensors."""
    a = torch.randn(NUM_TOKENS, HIDDEN, device="cuda", dtype=DTYPE) / 10
    w1 = (
        torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, device="cuda", dtype=DTYPE)
        / 10
    )
    w2 = torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, device="cuda", dtype=DTYPE) / 10

    # Random expert routing
    scores = torch.randn(NUM_TOKENS, NUM_EXPERTS, device="cuda", dtype=DTYPE)
    tw, ti = torch.topk(scores, TOPK, dim=-1)
    tw = torch.softmax(tw.float(), dim=-1)
    ti = ti.to(torch.int32)

    # # All tokens route to the same expert (id 0), repeated TOPK times for API shape.
    # SAME_EXPERT = 0
    # ti = torch.full(
    #     (NUM_TOKENS, TOPK), SAME_EXPERT, device="cuda", dtype=torch.int32
    # )
    # tw = torch.full(
    #     (NUM_TOKENS, TOPK),
    #     1.0 / TOPK,
    #     device="cuda",
    #     dtype=torch.float32,
    # )

    out = torch.zeros(NUM_TOKENS, HIDDEN, device="cuda", dtype=DTYPE)
    return a, w1, w2, tw, ti, out


def call_moe(a, w1, w2, tw, ti, out):
    """One forward pass through FlashInfer CUTLASS BF16 fused-MoE.

    Inside the C++ runMoe(), a workspace tensor is allocated as a local
    variable and freed when runMoe() returns.  During graph capture this
    allocation lands in the graph-private pool and the free returns it —
    leaving the graph with raw pointers into freed memory.
    """
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
        tune_max_num_tokens=NUM_TOKENS,
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
            pool = tuple(seg.get("segment_pool_id", (0, 0)))
            print(f"  poisoned 0x{addr:016x}  {size / (1024**2):.1f} MB  pool={pool}")
            n += 1
    return n


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    torch.accelerator.set_device(0)
    torch.manual_seed(42)

    print(
        f"FlashInfer CUTLASS BF16 MoE — workspace poison test\n"
        f"  tokens={NUM_TOKENS}  hidden={HIDDEN}  inter={INTERMEDIATE}  "
        f"experts={NUM_EXPERTS}  topk={TOPK}\n"
    )

    cudart = load_cudart()
    a, w1, w2, tw, ti, out = make_tensors()

    # 1 ── Eager reference ────────────────────────────────────────────────
    print("[1/6] Eager reference …")
    call_moe(a, w1, w2, tw, ti, out)
    torch.accelerator.synchronize()
    eager = out.clone()

    # 2 ── CUDA-graph capture ─────────────────────────────────────────────
    #   The workspace tensor is allocated and freed *during* capture.
    #   After capture the graph retains raw pointers into freed pool memory.
    print("[2/6] CUDA-graph capture …")
    pool = torch.cuda.graph_pool_handle()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()

    # Warm-up on capture stream (required for autotuning / one-time setup)
    with torch.cuda.stream(stream):
        call_moe(a, w1, w2, tw, ti, out)
    torch.accelerator.synchronize()

    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            call_moe(a, w1, w2, tw, ti, out)
    torch.accelerator.synchronize()

    # 3 ── Memory snapshot ────────────────────────────────────────────────
    #   Graph-pool segments should show allocated=0 / active=0 because the
    #   workspace was freed inside the capture scope.
    print("[3/6] Memory snapshot …")
    print(f"  graph pool handle = {pool}")
    print(f"  stream = {stream.cuda_stream}")
    snap = torch.cuda.memory_snapshot()
    print_snapshot(snap, "All segments after capture")

    # 4 ── Poison freed workspace ─────────────────────────────────────────
    #   Write 0xDE over every freed segment so any kernel that reads stale
    #   workspace would see garbage.
    print("[4/6] Poisoning all freed segments …")
    with torch.cuda.stream(stream):
        n = poison_freed_segments(snap, cudart, stream)
    if n:
        print(f"  → {n} segment(s) poisoned\n")
    else:
        print("  (no freed segments to poison)\n")

    # 5 ── Graph replay ───────────────────────────────────────────────────
    print("[5/6] Graph replay …")
    with torch.cuda.stream(stream):
        graph.replay()
    torch.accelerator.synchronize()
    replayed = out.clone()

    # 6 ── Compare ────────────────────────────────────────────────────────
    diff = (eager - replayed).abs()
    print("[6/6] Comparison")
    print(f"  max  |eager - replay| = {diff.max().item():.6e}")
    print(f"  mean |eager - replay| = {diff.mean().item():.6e}")

    if torch.equal(eager, replayed):
        print("\n  PASS — graph replay matches eager after workspace poisoning.")
    else:
        print("\n  FAIL — graph replay does not match eager after workspace poisoning.")
        sys.exit(1)


if __name__ == "__main__":
    main()
