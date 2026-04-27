# Debugging vLLM API Discrepancies — Use Guide

Send the same conversation through Chat Completions and Responses APIs (streaming + non-streaming) and capture both client-side responses and server-side token traces.

## 1. Start vLLM with tracing

```bash
vllm serve Qwen/Qwen3-4B --port 8000 \
  --enable-log-requests \
  --enable-log-outputs
```

Both flags are required for the server-side trace file to be written.

## 2. Run `send_request.py`

To send the same request to Chat Completions vs Responses API.

```bash
# All 4 paths (CC + Responses × streaming + non-streaming)
python send_request.py conversation.json --model Qwen/Qwen3-4B

# Only non-streaming
python send_request.py conversation.json --model Qwen/Qwen3-4B --modes non_streaming

# Raw server output, no SDK parsing
python send_request.py conversation.json --model Qwen/Qwen3-4B --raw
```

Sampling is hard-coded: `temperature=0`, `seed=42`, `max_tokens=256`, `tool_choice="auto"`.

## 3. Where results are stored

**Client-side** — `/tmp/vllm-results/`:

| File | Content |
| --- | --- |
| `chat_completion.json` | CC non-streaming response |
| `chat_completion_stream.json` / `.txt` | CC streaming chunks (`.json` parsed, `.txt` raw SSE) |
| `responses.json` | Responses API non-streaming response |
| `responses_stream.json` / `.txt` | Responses API streaming events (`.json` parsed, `.txt` raw SSE) |

**Server-side** — `/tmp/vllm-traces/trace.jsonl`:
