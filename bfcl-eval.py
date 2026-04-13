#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""bfcl-eval.py — BFCL tool-calling evaluation driver for vLLM.

Compares three inference paths against the same model and test categories.
This script starts and stops its own ``vllm serve`` process; BFCL never
manages the server (``skip_server_setup=True`` in all modes).

  chat_completions   Calls the Chat Completions API with tools
                     (OpenAICompletionsHandler).

  responses          Calls the Responses API (OpenAIResponsesHandler).

  completions_oss    Official BFCL OSS path.  Calls the text completions API
                     (client.completions.create) with model-specific prompt
                     formatting and decoding (OSSHandler).

Each mode stores results under ``<results-dir>/<mode>/`` with separate
``result/`` and ``score/`` trees so scores are directly comparable.

A free port is automatically selected for each run, so multiple
invocations can safely run in parallel.  ``--extra-vllm-args`` carries
any additional ``vllm serve`` flags (TP size, tool-call parser, …).

Examples::

    .venv/bin/python bfcl-eval.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --mode chat_completions \\
        --extra-vllm-args "--enable-auto-tool-choice \\
            --tool-call-parser llama3_json --tensor-parallel-size 1 \\
            --max-model-len 32768 --enforce-eager --no-enable-prefix-caching"

    # All three modes (each in a subprocess with its own result dir):
    .venv/bin/python bfcl-eval.py \\
        --model meta-llama/Llama-3.1-8B-Instruct --run-all \\
        --extra-vllm-args "..."

Requirements: ``bfcl-eval>=2025.10.20.1`` and vLLM (from this repo's .venv).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

MODES = ("chat_completions", "responses", "completions_oss")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Ask the OS for an ephemeral port that is currently unused."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_health(port: int, timeout: int = 600) -> None:
    """Block until ``GET /health`` on *port* returns 200."""
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{port}/health"
    t0 = time.monotonic()
    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= timeout:
            raise TimeoutError(f"vLLM server not ready after {timeout}s")
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        if int(elapsed) > 0 and int(elapsed) % 30 == 0:
            print(f"  Still waiting... ({int(elapsed)}s elapsed)")
        time.sleep(2)


def _bfcl_func_defaults(func):
    """Extract keyword defaults from a bfcl-eval Typer-decorated function.

    bfcl-eval wraps ``generate()`` and ``evaluate()`` with Typer, so default
    values are ``typer.models.OptionInfo`` objects instead of plain values.
    """
    import typer

    defaults = {}
    for name, param in inspect.signature(func).parameters.items():
        if param.default is not inspect.Parameter.empty:
            val = param.default
            if isinstance(val, typer.models.OptionInfo):
                val = val.default
            defaults[name] = val
    return defaults


# ---------------------------------------------------------------------------
# Single-mode runner (all three modes share this function)
# ---------------------------------------------------------------------------


def _run_mode(args: argparse.Namespace, mode: str) -> int:
    """Run one BFCL generate + evaluate pass in *mode*.

    Starts a ``vllm serve`` process, runs BFCL with ``skip_server_setup=True``
    (so BFCL never manages the server), then tears the server down.
    """

    is_api = mode in ("chat_completions", "responses")
    categories = [c.strip() for c in args.test_category.split(",")]
    port = _find_free_port()
    extra_tokens = shlex.split(args.extra_vllm_args) if args.extra_vllm_args else []

    # -- 1. Set BFCL_PROJECT_ROOT before any bfcl import (cached at import) -
    project_root = str(Path(args.results_dir).resolve() / mode)
    os.makedirs(project_root, exist_ok=True)
    os.environ["BFCL_PROJECT_ROOT"] = project_root

    if is_api:
        os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
        os.environ["OPENAI_API_KEY"] = "dummy"
    else:
        # OSSHandler reads LOCAL_SERVER_PORT / LOCAL_SERVER_ENDPOINT to build
        # its OpenAI client URL (not OPENAI_BASE_URL).
        os.environ["LOCAL_SERVER_PORT"] = str(port)
        os.environ["LOCAL_SERVER_ENDPOINT"] = "127.0.0.1"

    # -- 2. Import bfcl (reads BFCL_PROJECT_ROOT once at import time) -------
    import bfcl_eval.constants.model_config as bfcl_mc
    from bfcl_eval.__main__ import evaluate, generate
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig

    # -- 3. Register model with the right handler ---------------------------
    if is_api:
        if mode == "chat_completions":
            from bfcl_eval.model_handler.api_inference.openai_completion import (
                OpenAICompletionsHandler,
            )

            handler_cls = OpenAICompletionsHandler
        else:
            from bfcl_eval.model_handler.api_inference.openai_response import (
                OpenAIResponsesHandler,
            )

            handler_cls = OpenAIResponsesHandler

        model_key = args.model
        bfcl_mc.MODEL_CONFIG_MAPPING[model_key] = ModelConfig(
            model_name=args.model,
            display_name=f"{args.model} ({mode}) (vLLM)",
            url=f"https://huggingface.co/{args.model}",
            org="",
            license="apache-2.0",
            model_handler=handler_cls,
            underscore_to_dot=True,
        )
    else:  # completions_oss
        model_key = args.model
        if model_key not in MODEL_CONFIG_MAPPING:
            from bfcl_eval.model_handler.local_inference.quick_testing_oss import (
                QuickTestingOSSHandler,
            )

            model_key = f"{args.model}-oss-eval"
            bfcl_mc.MODEL_CONFIG_MAPPING[model_key] = ModelConfig(
                model_name=args.model,
                display_name=f"{args.model} (OSS completions)",
                url=f"https://huggingface.co/{args.model}",
                org="",
                license="apache-2.0",
                model_handler=QuickTestingOSSHandler,
            )
            print(
                f"NOTE: '{args.model}' not in BFCL model registry; registered "
                f"as '{model_key}' with QuickTestingOSSHandler (generic "
                f"chat-template).  For accurate FC scores, use an official "
                f"BFCL model key."
            )

    # -- 4. Save run metadata -----------------------------------------------
    (Path(project_root) / "run_config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "model_key": model_key,
                "mode": mode,
                "test_category": args.test_category,
                "num_threads": args.num_threads,
                "port": port,
                "extra_vllm_args": args.extra_vllm_args,
                "argv": sys.argv,
            },
            indent=2,
        )
        + "\n"
    )

    # -- 5. Start server, generate, evaluate, cleanup -----------------------
    server = None
    try:
        cmd = ["vllm", "serve", args.model, "--port", str(port)] + extra_tokens
        print(f"Starting vLLM: {' '.join(cmd)}")
        server = subprocess.Popen(cmd)
        print(f"Waiting for vLLM on port {port} (timeout 600s)...")
        _wait_for_health(port)
        print("vLLM server ready.")

        # generate (skip_server_setup=True — BFCL never manages the server)
        print(
            f"=== BFCL generate: {model_key}  mode={mode}  categories={categories} ==="
        )
        gkw = _bfcl_func_defaults(generate)
        gkw["model"] = [model_key]
        gkw["test_category"] = categories
        gkw["num_threads"] = args.num_threads
        gkw["skip_server_setup"] = True
        if not is_api:
            gkw["backend"] = "vllm"
        generate(**gkw)

        # evaluate
        print(
            f"=== BFCL evaluate: {model_key}  mode={mode}  categories={categories} ==="
        )
        ekw = _bfcl_func_defaults(evaluate)
        ekw["model"] = [model_key]
        ekw["test_category"] = categories
        evaluate(**ekw)

        print(f"=== {mode} completed successfully ===")
        return 0

    except Exception as exc:
        print(f"ERROR ({mode}): {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    finally:
        if server is not None:
            print("Stopping vLLM server...")
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        for d in [Path(project_root) / ".file_locks", Path(".file_locks")]:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model id")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--mode", choices=MODES, help="Single evaluation mode")
    mode_group.add_argument(
        "--run-all",
        action="store_true",
        help="Run all three modes sequentially (separate result dirs)",
    )

    parser.add_argument(
        "--results-dir",
        default="./bfcl_results",
        help="Parent results directory (default: ./bfcl_results)",
    )
    parser.add_argument(
        "--test-category",
        default="multi_turn",
        help="Comma-separated BFCL test categories (default: multi_turn)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="Threads for BFCL generate (default: 8)",
    )
    parser.add_argument(
        "--extra-vllm-args",
        default="",
        help="Additional vllm-serve flags as a single quoted string "
        "(e.g. '--tensor-parallel-size 2 --enforce-eager')",
    )
    args = parser.parse_args()

    # --run-all: one subprocess per mode so BFCL_PROJECT_ROOT is fresh.
    if args.run_all:
        worst_rc = 0
        for mode in MODES:
            print(f"\n{'=' * 60}\n  Running mode: {mode}\n{'=' * 60}\n")
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--model",
                args.model,
                "--mode",
                mode,
                "--results-dir",
                args.results_dir,
                "--test-category",
                args.test_category,
                "--num-threads",
                str(args.num_threads),
            ]
            if args.extra_vllm_args:
                cmd += ["--extra-vllm-args", args.extra_vllm_args]
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"Mode {mode} exited with code {rc}", file=sys.stderr)
                worst_rc = rc
        return worst_rc

    return _run_mode(args, args.mode)


if __name__ == "__main__":
    sys.exit(main())
