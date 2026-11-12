<!-- markdownlint-disable -->

## Overview

Ensure `--generation-config` and `--override-generation-config` match the stated semantics.

## Claims

- Prevent `generation-config.json` values from being used when using `--generation-config="vllm"`
  - Previously, `eos_token_ids` from `generation-config.json` are still loaded despite `--generation-config="vllm"`
- Apply `--override-generation-config` even when they are not whitelisted sampling params.
  - Previously, `eos_token_ids` from `--override-generation-config` would not be applied. 

## Validation

```bash
.venv/bin/python -m pytest -q \
  tests/config/test_config_generation.py::test_sampling_defaults_include_penalties \
  tests/config/test_config_generation.py::test_sampling_defaults_preserve_penalties_from_generation-config \
  tests/test_config.py::test_generation-config_loading_and_sampling_defaults \
  tests/transformers_utils/test_config.py::test_model_config_generation_fallback_forwards_code_revision \
  tests/entrypoints/openai/test_penalty_default_resolution.py

11 passed, 19 warnings
```

```
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token",
)

harmony_prompt = (
    "<|start|>user<|message|>What is 2 + 2?<|end|>"
    "<|start|>assistant"
)

response = client.completions.create(
    model="openai/gpt-oss-20b",
    prompt=harmony_prompt,
    max_tokens=1024,
    extra_body={
        # Prevent vLLM from adding tokenizer-level special tokens.
        "add_special_tokens": False,
        # Keep Harmony markers in the returned text for client-side parsing.
        "skip_special_tokens": False,
        "seed": 42,
    },
)

print(response.choices[0].text)
```

```
vllm serve openai/gpt-oss-20b

Main:
"<|channel|>analysis<|message|>The user asks "What is 2 + 2?" This is a math question. There's no policy violation. We should answer accordingly: 4.<|end|><|start|>assistant<|channel|>final<|message|>2 + 2 = **4**."

This:
"<|channel|>analysis<|message|>The user asks "What is 2 + 2?" This is a math question. There's no policy violation. We should answer accordingly: 4.<|end|><|start|>assistant<|channel|>final<|message|>2 + 2 = **4**."

vllm serve openai/gpt-oss-20b --override-generation-config '{"eos_token_id": [200005]}'

Main:
"<|channel|>analysis<|message|>The user asks "What is 2 + 2?" This is a math question. There's no policy violation. We should answer accordingly: 4.<|end|><|start|>assistant<|channel|>final<|message|>2 + 2 = **4**."
('{"eos_token_id": [200005]}' had no effect)

This:
""
(empty as expected since `200005` / "<|channel>" stops generation immediately)
```

## Details

In addition to sampling params such as `temperature` and `top_k`, `generation-config.json` is often also used to specify:
- `eos_token_ids`
- `max_denoising_steps` for Diffusion Gemma
- `suppress_tokens` for Gemma 4
- `attn_temperature_tuning` for llama4
- and more.

Previously, `override-generation-config` was only applied to `ModelConfig.get_diff_sampling_param()`, not `ModelConfig.try_get_generation()`. So any non-sampling override didn't apply.

Also, only `ModelCondfig.get_diff_sampling_param()` distinguished between `--generation-config="auto"` vs `--generation-config="vllm"`. So non-sampling params like `eos_token_ids` from `generation-config.json` could have been used even when `--generation-config=vllm`.

The fix is distinguish between `--generation-config="auto"` vs `--generation-config="vllm"` and apply overrides in `ModelConfig.try_get_generation()`. `ModelConfig.get_diff_sampling_param()` calls `ModelConfig.try_get_generation()`.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [ ] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [x] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [ ] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [ ] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>

**BEFORE SUBMITTING, PLEASE READ <https://docs.vllm.ai/en/latest/contributing>** (anything written below this line will be removed by GitHub Actions)
