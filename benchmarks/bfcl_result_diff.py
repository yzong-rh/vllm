#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare two BFCL ``*_score.json`` files (NDJSON).

BFCL score files list **only failing** test rows (plus an optional first-line
aggregate without ``id``). Presence of an ``id`` means that run **failed** that
test; absence means **pass** (for a full evaluation of the same category).

Writes two files:

1. ``--out-a`` (default ``a-b.json``): entries **from A** for IDs that **failed
   in A** and **passed in B** (i.e. present in A, absent from B).
2. ``--out-b`` (default ``b-a.json``): entries **from B** for IDs that **failed
   in B** and **passed in A** (present in B, absent from A).

``model_name``, ``test_category``, and ``valid`` are stripped from written rows
to keep diffs smaller.

Example::

    .venv/bin/python benchmarks/bfcl_result_diff.py \\
        benchmarks/bfcl_results/run_a/.../BFCL_v4_multi_turn_base_score.json \\
        benchmarks/bfcl_results/run_b/.../BFCL_v4_multi_turn_base_score.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Omitted from serialized diff rows (comparison uses raw entries before strip).
OMIT_OUTPUT_KEYS = frozenset({"model_name", "test_category", "valid"})


def load_score_entries(path: Path) -> dict[str, dict]:
    """id -> full JSON object (skips aggregate lines without ``id``)."""
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            tid = obj.get("id")
            if tid is None:
                continue
            out[str(tid)] = obj
    return out


def strip_score_entry(obj: dict) -> dict:
    """Return a shallow copy without keys omitted from diff output."""
    return {k: v for k, v in obj.items() if k not in OMIT_OUTPUT_KEYS}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("a", type=Path, help="First BFCL *_score.json (NDJSON)")
    parser.add_argument("b", type=Path, help="Second BFCL *_score.json (NDJSON)")
    parser.add_argument(
        "--out-a",
        type=Path,
        default=Path("a-b.json"),
        help="Output: failed in A, passed in B (lines from A). Default: a-b.json",
    )
    parser.add_argument(
        "--out-b",
        type=Path,
        default=Path("b-a.json"),
        help="Output: failed in B, passed in A (lines from B). Default: b-a.json",
    )
    args = parser.parse_args()

    if not args.a.is_file():
        print(f"ERROR: not found: {args.a}", file=sys.stderr)
        return 1
    if not args.b.is_file():
        print(f"ERROR: not found: {args.b}", file=sys.stderr)
        return 1

    ea = load_score_entries(args.a)
    eb = load_score_entries(args.b)
    keys_a = set(ea)
    keys_b = set(eb)

    # Score files only contain failures; missing id => pass for that run.
    ids_a = keys_a - keys_b
    ids_b = keys_b - keys_a

    with args.out_a.open("w", encoding="utf-8") as f:
        for i in sorted(ids_a):
            f.write(json.dumps(strip_score_entry(ea[i]), ensure_ascii=False) + "\n")
    with args.out_b.open("w", encoding="utf-8") as f:
        for i in sorted(ids_b):
            f.write(json.dumps(strip_score_entry(eb[i]), ensure_ascii=False) + "\n")

    print(
        f"# failed A, passed B: {len(ids_a)} → {args.out_a}",
        file=sys.stderr,
    )
    print(
        f"# failed B, passed A: {len(ids_b)} → {args.out_b}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
