# Pocket-Agent

> Fine-tuned on-device mobile assistant for structured tool calling — Hackathon Submission

---

## Overview

Pocket-Agent is a fine-tuned language model (≤2B parameters) designed to perform structured tool calls for an on-device mobile assistant. It operates fully offline, runs within latency constraints on CPU, and fits within the quantization size budget.

| Property | Value |
|---|---|
| **Base Model** | `Qwen/Qwen2.5-0.5B-Instruct` |
| **Parameters** | 494M (0.5B) |
| **Fine-tuning Method** | LoRA via Unsloth + TRL SFTTrainer |
| **Quantization** | 4-bit (BitsAndBytes, NF4) |
| **Training Data** | ~567 synthetic examples (generated + adversarial) |
| **Training Time** | ~2 min on Colab T4 (108 steps, 3 epochs) |

---

## Links

| Resource | Link |
|---|---|
| **Colab Notebook** | `[INSERT COLAB LINK HERE]` |
| **Gradio Demo** | `[INSERT GRADIO LINK HERE]` |
| **Quantized Model** | Included in repo under `./quantized_model/` |
| **LoRA Adapter** | Included in repo under `./lora_adapter/` |

---

## Tool Schema

The model emits JSON wrapped in `<tool_call>...</tool_call>` tags for tool requests, or plain text for refusals.

```json
{"tool": "weather",  "args": {"location": "string", "unit": "C|F"}}
{"tool": "calendar", "args": {"action": "list|create", "date": "YYYY-MM-DD", "title": "string?"}}
{"tool": "convert",  "args": {"value": number, "from_unit": "string", "to_unit": "string"}}
{"tool": "currency", "args": {"amount": number, "from": "ISO3", "to": "ISO3"}}
{"tool": "sql",      "args": {"query": "string"}}
```

---

## Setup & Reproduction

### Requirements

```bash
pip install unsloth trl transformers datasets accelerate bitsandbytes peft gradio
```

### End-to-End Reproduction

```bash
# 1. Generate training data
python generate_data.py
python generate_adversarial.py

# 2. Fine-tune
python train.py

# 3. Merge + quantize
python quantize.py

# 4. Evaluate
python eval.py

# 5. Launch demo
python app.py
```

Or run everything at once:

```bash
make all
```

### Run Inference Only

```python
from inference import run

# Single-turn
run("What's the weather in Lahore?", [])
# → <tool_call>{"tool":"weather","args":{"location":"Lahore","unit":"C"}}</tool_call>

# Multi-turn
history = [
    {"role": "user",      "content": "Convert 100 USD to PKR."},
    {"role": "assistant", "content": "<tool_call>{...}</tool_call>"},
]
run("Now convert that to EUR.", history)
# → <tool_call>{"tool":"currency","args":{"amount":100,"from":"USD","to":"EUR"}}</tool_call>

# Refusal
run("Book me a flight to London.", [])
# → "I don't have a flight booking tool. I can help with weather, calendar, conversions, currency, or SQL."
```

---

## Design Decisions

### Model Choice — Qwen2.5-0.5B-Instruct

- Smallest model in the Qwen2.5 family with strong instruction-following capability
- Fits easily within the ≤500 MB quantized size gate (~270 MB at 4-bit)
- Fast inference on CPU — well within the 200ms/turn latency gate
- Native support for Hinglish/Urdu/Arabic code-switched prompts due to multilingual pretraining

### Fine-tuning Strategy — LoRA via Unsloth

- Used `r=8`, `lora_alpha=16` targeting `q_proj`, `v_proj`, `k_proj`, `o_proj`
- Only 0.22% of parameters trained (1.08M / 495M), keeping adapter tiny
- 3 epochs, batch size 16 (4 × 4 gradient accumulation), learning rate 2e-4
- Loss dropped from ~4.0 → ~0.23, indicating strong convergence

### Data Generation Strategy

Training data was fully synthetic, split across two scripts:

**`generate_data.py`** — ~497 examples covering:
- Weather (C/F, typos, Urdu/Hinglish templates)
- Calendar (list, create, today's date edge cases)
- Convert (unit ambiguity, abbreviations)
- Currency (ISO code normalization, Hinglish phrasing)
- SQL (direct query execution)
- Multi-turn (pronoun resolution: "that", "same", "isko", "ye wala")
- Refusals (chitchat, impossible tools, ambiguous references)

**`generate_adversarial.py`** — ~100 additional adversarial examples targeting Slice C:
- Typos (`wether`, `temprature`, `conver`)
- Code-switched prompts (Urdu, Hindi, Spanish, Arabic + English)
- Hinglish pronouns in multi-turn (`isko euros me kar do`)
- Hallucination-bait refusals (cricket scores, gold prices, prayer times)

### Quantization — BitsAndBytes 4-bit

- Used `BitsAndBytesConfig(load_in_4bit=True)` via transformers
- Stays entirely in the HuggingFace/transformers ecosystem
- Grader-compatible: adapter loads cleanly in transformers v5
- No GGUF/llama.cpp dependency required

### Output Post-processing — `_extract_tool_call()`

The model occasionally generates garbage tokens after valid output. Post-processing pipeline:
1. Hard-stop at known special tokens (`<|im_end|>`, `<|endoftext|>`)
2. Truncate everything after `</tool_call>`
3. Validate and parse JSON — normalize ISO codes to uppercase, default weather unit to `C`
4. Strip non-ASCII and garbage ASCII from plain-text refusals
5. Fallback to a safe refusal string if JSON is unrecoverable

---

## Hard Gate Compliance

| Gate | Status |
|---|---|
| Adapter loads on declared base model (≤2B) in transformers v5 | ✅ |
| Quantized model ≤ 500 MB | ✅ (~270 MB at 4-bit) |
| Mean inference latency ≤ 200 ms/turn on Colab CPU | ✅ |
| Training data shares zero prompts with public test set | ✅ (SHA-256 verified) |
| `inference.py` contains no network imports | ✅ |
| Chatbot demo launches and accepts input | ✅ |

---

## 👤 Author
Muhammad Furqan Rauf
Submitted for the Pocket-Agent Hackathon.
