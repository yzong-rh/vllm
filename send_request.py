#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Send the same conversation to multiple vLLM API endpoints.

Reads a conversation from a JSON file (or inline JSON string) and sends
it to Chat Completions and Responses APIs in both streaming and
non-streaming modes.  Results are dumped to /tmp/vllm-results/.

Two output modes controlled by --raw:
  Default:  SDK-parsed .model_dump() dicts as pretty-printed JSON.
  --raw:    Raw HTTP response bytes (non-streaming) or decoded SSE lines
            (streaming), with zero Pydantic parsing on the response side.

Sampling is hard-coded to greedy decoding: temperature=0, seed=42,
max_tokens=256.  tool_choice is always "auto".
"""

import argparse
import json
import os

from openai import OpenAI

OUTPUT_DIR = "/tmp/vllm-results"


# ---------------------------------------------------------------------------
# Loading and conversion
# ---------------------------------------------------------------------------


def load_conversation(path_or_json: str) -> dict:
    """Load from file path or inline JSON string."""
    try:
        return json.loads(path_or_json)
    except json.JSONDecodeError:
        with open(path_or_json) as f:
            return json.load(f)


def to_responses_input(conversation: dict) -> tuple[list[dict], str | None]:
    """Convert CC messages to Responses API input items and instructions.

    System messages are extracted and concatenated into `instructions`.
    Tool calls, tool results, and reasoning are converted to their
    Responses API equivalents.

    Reverse: construct_chat_messages_with_tool_call() in
    vllm/entrypoints/openai/responses/utils.py
    """
    items: list[dict] = []
    system_parts: list[str] = []
    reasoning_counter = 0

    for m in conversation["messages"]:
        role = m.get("role", "")

        if role == "system":
            system_parts.append(m.get("content", ""))
            continue

        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": m["tool_call_id"],
                    "output": m.get("content", ""),
                }
            )
            continue

        if role == "assistant":
            reasoning = m.get("reasoning") or m.get("reasoning_content")
            if reasoning:
                reasoning_counter += 1
                items.append(
                    {
                        "type": "reasoning",
                        "id": f"reasoning_{reasoning_counter}",
                        "content": [{"type": "text", "text": reasoning}],
                    }
                )

            tool_calls = m.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": tc["id"],
                            **tc["function"],
                        }
                    )
                continue

            items.append({"role": "assistant", "content": m.get("content", "")})
            continue

        items.append({"role": role, "content": m.get("content", "")})

    instructions = "\n\n".join(system_parts) if system_parts else None
    return items, instructions


def to_responses_tools(tools: list[dict]) -> list[dict]:
    """Convert CC tool defs to Responses format (unwrap 'function' nesting).

    Reverse: convert_tool_responses_to_completions_format() in
    vllm/entrypoints/openai/responses/utils.py
    """
    return [
        {"type": "function", **tool["function"]}
        for tool in tools
        if tool.get("type") == "function"
    ]


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def dump(filename: str, data) -> str:
    """Write parsed data as pretty JSON to OUTPUT_DIR/filename."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def dump_bytes(filename: str, data: bytes | str) -> str:
    """Write raw bytes/text verbatim to OUTPUT_DIR/filename."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    if isinstance(data, bytes):
        with open(path, "wb") as f:
            f.write(data)
    else:
        with open(path, "w") as f:
            f.write(data)
    return path


# ---------------------------------------------------------------------------
# Chat Completions
# ---------------------------------------------------------------------------


def send_chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    raw: bool = False,
):
    tool_kwargs: dict = {}
    if tools:
        tool_kwargs["tools"] = tools
        tool_kwargs["tool_choice"] = "auto"
    if raw:
        resp = client.with_raw_response.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            temperature=0.0,
            max_tokens=256,
            seed=42,
            **tool_kwargs,
        )
        path = dump_bytes("chat_completion.json", resp.content)
        print(f"  wrote {path} (raw bytes)")
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            temperature=0.0,
            max_tokens=256,
            seed=42,
            **tool_kwargs,
        )
        path = dump("chat_completion.json", resp.model_dump())
        print(f"  wrote {path}")


def send_chat_completion_stream(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    raw: bool = False,
):
    tool_kwargs: dict = {}
    if tools:
        tool_kwargs["tools"] = tools
        tool_kwargs["tool_choice"] = "auto"
    if raw:
        with client.with_streaming_response.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            temperature=0.0,
            max_tokens=256,
            seed=42,
            **tool_kwargs,
        ) as resp:
            lines = list(resp.iter_lines())
        path = dump_bytes("chat_completion_stream.txt", "\n".join(lines))
        print(f"  wrote {path} ({len(lines)} lines, raw SSE)")
    else:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            temperature=0.0,
            max_tokens=256,
            seed=42,
            **tool_kwargs,
        )
        chunks = [chunk.model_dump() for chunk in stream]
        path = dump("chat_completion_stream.json", chunks)
        print(f"  wrote {path} ({len(chunks)} chunks)")


# ---------------------------------------------------------------------------
# Responses API
# ---------------------------------------------------------------------------


def send_responses(
    client: OpenAI,
    model: str,
    input_items: list[dict],
    instructions: str | None,
    tools: list[dict] | None = None,
    raw: bool = False,
):
    tool_kwargs: dict = {}
    if tools:
        tool_kwargs["tools"] = tools
        tool_kwargs["tool_choice"] = "auto"
    if raw:
        resp = client.with_raw_response.responses.create(
            model=model,
            input=input_items,
            instructions=instructions,
            stream=False,
            temperature=0.0,
            max_output_tokens=256,
            extra_body={"seed": 42},
            **tool_kwargs,
        )
        path = dump_bytes("responses.json", resp.content)
        print(f"  wrote {path} (raw bytes)")
    else:
        resp = client.responses.create(
            model=model,
            input=input_items,
            instructions=instructions,
            stream=False,
            temperature=0.0,
            max_output_tokens=256,
            extra_body={"seed": 42},
            **tool_kwargs,
        )
        path = dump("responses.json", resp.model_dump())
        print(f"  wrote {path}")


def send_responses_stream(
    client: OpenAI,
    model: str,
    input_items: list[dict],
    instructions: str | None,
    tools: list[dict] | None = None,
    raw: bool = False,
):
    tool_kwargs: dict = {}
    if tools:
        tool_kwargs["tools"] = tools
        tool_kwargs["tool_choice"] = "auto"
    if raw:
        with client.with_streaming_response.responses.create(
            model=model,
            input=input_items,
            instructions=instructions,
            stream=True,
            temperature=0.0,
            max_output_tokens=256,
            extra_body={"seed": 42},
            **tool_kwargs,
        ) as resp:
            lines = list(resp.iter_lines())
        path = dump_bytes("responses_stream.txt", "\n".join(lines))
        print(f"  wrote {path} ({len(lines)} lines, raw SSE)")
    else:
        stream = client.responses.create(
            model=model,
            input=input_items,
            instructions=instructions,
            stream=True,
            temperature=0.0,
            max_output_tokens=256,
            extra_body={"seed": 42},
            **tool_kwargs,
        )
        events = []
        for event in stream:
            events.append(
                {
                    "type": event.type,
                    "data": event.model_dump(),
                }
            )
        path = dump("responses_stream.json", events)
        print(f"  wrote {path} ({len(events)} events)")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_all(
    base_url: str,
    model: str,
    conversation: dict,
    api_key: str = "EMPTY",
    apis: list[str] | None = None,
    modes: list[str] | None = None,
    raw: bool = False,
):
    client = OpenAI(base_url=base_url, api_key=api_key)

    messages = list(conversation["messages"])
    input_items, instructions = to_responses_input(conversation)

    cc_tools = conversation.get("tools")
    resp_tools = to_responses_tools(cc_tools) if cc_tools else None

    apis = apis or ["chat_completions", "responses"]
    modes = modes or ["non_streaming", "streaming"]

    dispatch = {
        ("chat_completions", "non_streaming"): lambda: send_chat_completion(
            client, model, messages, cc_tools, raw
        ),
        ("chat_completions", "streaming"): lambda: send_chat_completion_stream(
            client, model, messages, cc_tools, raw
        ),
        ("responses", "non_streaming"): lambda: send_responses(
            client, model, input_items, instructions, resp_tools, raw
        ),
        ("responses", "streaming"): lambda: send_responses_stream(
            client, model, input_items, instructions, resp_tools, raw
        ),
    }

    for api in apis:
        for mode in modes:
            key = (api, mode)
            if key in dispatch:
                label = "raw" if raw else "parsed"
                print(f"\u2192 {api} ({mode}, {label})...")
                dispatch[key]()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Send the same conversation to multiple vLLM APIs"
    )
    parser.add_argument(
        "conversation",
        help="Path to JSON file or inline JSON string with messages",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="vLLM server base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name as registered in vLLM",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key for authentication (default: EMPTY)",
    )
    parser.add_argument(
        "--apis",
        nargs="+",
        choices=["chat_completions", "responses"],
        default=["chat_completions", "responses"],
        help="Which APIs to test",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["non_streaming", "streaming"],
        default=["non_streaming", "streaming"],
        help="Which modes to test",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        default=False,
        help="Dump raw HTTP bytes instead of SDK-parsed JSON",
    )

    args = parser.parse_args()
    conversation = load_conversation(args.conversation)
    run_all(
        base_url=args.base_url,
        model=args.model,
        conversation=conversation,
        api_key=args.api_key,
        apis=args.apis,
        modes=args.modes,
        raw=args.raw,
    )
    print(f"\nAll results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
