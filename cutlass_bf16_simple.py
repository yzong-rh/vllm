#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    torch.accelerator.set_device(0)
    torch.manual_seed(42)

    print(
        f"FlashInfer CUTLASS BF16 MoE — simple test\n"
        f"  tokens={NUM_TOKENS}  hidden={HIDDEN}  inter={INTERMEDIATE}  "
        f"experts={NUM_EXPERTS}  topk={TOPK}\n"
    )

    a, w1, w2, tw, ti, out = make_tensors()

    # 1 ── Eager reference ────────────────────────────────────────────────
    print("[1/4] Eager reference …")
    call_moe(a, w1, w2, tw, ti, out)
    eager = out.clone()

    # 2 ── CUDA-graph capture ─────────────────────────────────────────────
    #   The workspace tensor is allocated and freed *during* capture.
    #   After capture the graph retains raw pointers into freed pool memory.
    print("[2/4] CUDA-graph capture …")
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

    # 3 ── Graph replay ───────────────────────────────────────────────────
    print("[3/4] Graph replay …")
    with torch.cuda.stream(stream):
        graph.replay()
    torch.accelerator.synchronize()
    replayed = out.clone()

    # 6 ── Compare ────────────────────────────────────────────────────────
    diff = (eager - replayed).abs()
    print("[4/4] Comparison")
    print(f"  max  |eager - replay| = {diff.max().item():.6e}")
    print(f"  mean |eager - replay| = {diff.mean().item():.6e}")

    if torch.equal(eager, replayed):
        print("\n  PASS — graph replay matches eager after workspace poisoning.")
    else:
        print("\n  FAIL — graph replay does not match eager after workspace poisoning.")
        sys.exit(1)


if __name__ == "__main__":
    main()
