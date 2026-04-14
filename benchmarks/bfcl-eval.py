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

  all                Runs every mode above sequentially in one invocation.

Each mode stores results under ``<results-dir>/<mode>/`` with separate
``result/`` and ``score/`` trees so scores are directly comparable.

A free port is automatically selected for each run, so multiple
invocations can safely run in parallel.  ``--extra-vllm-args`` carries
any additional ``vllm serve`` flags (TP size, tool-call parser, …).

Greedy decoding (temperature=0) is forced for reproducibility.  The vLLM
server-side seed defaults to 0. For full determinism, set ``VLLM_BATCH_INVARIANT=1``.

Examples::

    # Quick sanity check with a small single-turn category:
    VLLM_BATCH_INVARIANT=1 python bfcl-eval.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --mode chat_completions \\
        --test-category simple_python \\
        --extra-vllm-args "--enable-auto-tool-choice \\
            --tool-call-parser llama3_json \\
            --chat-template examples/tool_chat_template_llama3.1_json.jinja \\
            --max-model-len 4096"

    # Run all modes sequentially (chat_completions, responses, completions_oss)
    # with a single command.  The server is started fresh for each mode.
    # --enable-auto-tool-choice / --tool-call-parser are harmless for
    # completions_oss (the OSS handler uses the text completions API).

    # --- openai/gpt-oss-120b  (TP=4, DP=2) -------------------------
    python bfcl-eval.py \\
        --model openai/gpt-oss-120b \\
        --mode all \\
        --extra-vllm-args "--enable-auto-tool-choice \\
            --tool-call-parser openai \\
            --tensor-parallel-size 4 --data-parallel-size 2 \\
            --max-model-len 32768"

    # --- nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4  (TP=2, DP=4)
    python bfcl-eval.py \\
        --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \\
        --mode all \\
        --extra-vllm-args "--enable-auto-tool-choice \\
            --tool-call-parser hermes \\
            --tensor-parallel-size 2 --data-parallel-size 4 \\
            --max-model-len 32768" --trust-remote-code

    # --- mistralai/Mistral-Small-4-119B-2603  (TP=4, DP=2) ---------
    python bfcl-eval.py \\
        --model mistralai/Mistral-Small-4-119B-2603 \\
        --mode all \\
        --extra-vllm-args "--enable-auto-tool-choice \\
            --tool-call-parser mistral \\
            --tensor-parallel-size 4 --data-parallel-size 2 \\
            --max-model-len 32768"

    # --- Qwen/Qwen3-235B-A22B-GPTQ-Int4  (TP=4, DP=2) -------------
    python bfcl-eval.py \\
        --model Qwen/Qwen3-235B-A22B-GPTQ-Int4 \\
        --mode all \\
        --extra-vllm-args "--enable-auto-tool-choice \\
            --tool-call-parser hermes \\
            --tensor-parallel-size 4 --data-parallel-size 2 \\
            --max-model-len 32768"

    # --- meta-llama/Llama-3.3-70B-Instruct  (TP=2, DP=4) -----------
    python bfcl-eval.py \\
        --model meta-llama/Llama-3.3-70B-Instruct \\
        --mode all \\
        --extra-vllm-args "--enable-auto-tool-choice \\
            --tool-call-parser llama3_json \\
            --chat-template examples/tool_chat_template_llama3.1_json.jinja \\
            --tensor-parallel-size 2 --data-parallel-size 4 \\
            --max-model-len 32768"

Requirements: ``bfcl-eval>=2025.10.20.1`` and vLLM (from this repo's .venv).
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import multiprocessing
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


def _run_mode(args: argparse.Namespace) -> int:
    """Run one BFCL generate + evaluate pass for ``args.mode``.

    Starts a ``vllm serve`` process, runs BFCL with ``skip_server_setup=True``
    (so BFCL never manages the server), then tears the server down.
    """
    mode = args.mode
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
    # Suppress chatty client-side HTTP loggers before importing bfcl/openai.
    for _name in ("httpx", "openai", "httpcore"):
        logging.getLogger(_name).setLevel(logging.WARNING)

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
            print(
                f"SKIP completions_oss: '{args.model}' is not in the BFCL model "
                f"registry. Use an official BFCL model key for OSS completions."
            )
            return 0

    # -- 4. Save run metadata -----------------------------------------------
    (Path(project_root) / "run_config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "model_key": model_key,
                "mode": mode,
                "test_category": args.test_category,
                "temperature": 0.0,
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
        cmd = [
            "vllm",
            "serve",
            args.model,
            "--port",
            str(port),
            "--disable-uvicorn-access-log",
        ] + extra_tokens
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
        gkw["temperature"] = 0.0
        gkw["allow_overwrite"] = True
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
# Subprocess wrapper
# ---------------------------------------------------------------------------

_spawn_ctx = multiprocessing.get_context("spawn")


def _run_mode_worker(args_dict: dict) -> None:
    """Entry point for the spawned child process.

    Reconstructs the argument namespace, runs the mode, and converts the
    return code into a process exit code.
    """
    sys.exit(_run_mode(argparse.Namespace(**args_dict)))


def _run_mode_in_subprocess(args: argparse.Namespace) -> int:
    """Run ``_run_mode`` in an isolated ``spawn`` process.

    The ``spawn`` context starts a fresh Python interpreter, so
    ``bfcl_eval`` is imported from scratch every time and picks up
    environment variables like ``BFCL_PROJECT_ROOT`` correctly.
    """
    proc = _spawn_ctx.Process(
        target=_run_mode_worker,
        args=(vars(args),),
    )
    proc.start()
    proc.join()
    return proc.exitcode


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model id")

    parser.add_argument(
        "--mode",
        required=True,
        choices=(*MODES, "all"),
        help="Evaluation mode ('all' runs every mode sequentially)",
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

    if args.mode != "all":
        return _run_mode_in_subprocess(args)

    worst = 0
    for mode in MODES:
        print(f"\n{'=' * 60}\n  Running mode: {mode}\n{'=' * 60}\n")
        args.mode = mode
        rc = _run_mode_in_subprocess(args)
        if rc:
            print(f"WARNING: mode '{mode}' exited with code {rc}", file=sys.stderr)
        worst = max(worst, rc)
    return worst


if __name__ == "__main__":
    sys.exit(main())
