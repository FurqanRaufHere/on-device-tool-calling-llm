from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch, json, re

SYSTEM = """You are Pocket-Agent, an on-device assistant with exactly 5 tools.

TOOL SCHEMAS (emit exactly this JSON structure):
{"tool":"weather",  "args":{"location":"string","unit":"C|F"}}
{"tool":"calendar", "args":{"action":"list|create","date":"YYYY-MM-DD","title":"string (only for create)"}}
{"tool":"convert",  "args":{"value":number,"from_unit":"string","to_unit":"string"}}
{"tool":"currency", "args":{"amount":number,"from":"ISO3","to":"ISO3"}}
{"tool":"sql",      "args":{"query":"string"}}

RULES:
1. Tool call → wrap ONLY in <tool_call>...</tool_call>, nothing else before or after.
2. unit default → always "C" unless user says Fahrenheit/F/°F.
3. ISO codes → USD, PKR, EUR, GBP, AED, INR, JPY, SAR, CAD (3-letter uppercase always).
4. Multi-turn → if user says "that", "it", "same", "isko", "ye wala" — resolve from previous assistant turn.
5. Refusal triggers → chitchat, impossible tools (flight/pizza/music/alarm/translate), ambiguous with no history.
6. NEVER emit a tool call for refusals. Plain text only.
7. NEVER add explanation around a tool call. Just the tag.

EXAMPLES:
User: weather in karachi → <tool_call>{"tool":"weather","args":{"location":"Karachi","unit":"C"}}</tool_call>
User: 100 USD to PKR → <tool_call>{"tool":"currency","args":{"amount":100,"from":"USD","to":"PKR"}}</tool_call>
User: book a flight → I don't have a flight booking tool. I can help with weather, calendar, conversions, currency, or SQL.
User: convert that to EUR [after USD→PKR turn] → <tool_call>{"tool":"currency","args":{"amount":100,"from":"USD","to":"EUR"}}</tool_call>"""

_model = None
_tokenizer = None

def _load():
    global _model, _tokenizer
    if _model is None:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        _tokenizer = AutoTokenizer.from_pretrained("./quantized_model")
        _model = AutoModelForCausalLM.from_pretrained(
            "./quantized_model",
            quantization_config=bnb_config,
            device_map="auto"
        )

# def _extract_tool_call(text: str) -> str:
#     # Truncate anything after </tool_call>
#     if "</tool_call>" in text:
#         text = text[:text.index("</tool_call>") + len("</tool_call>")]
    
#     match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
#     if not match:
#         # For plain text refusals, take only first sentence/line
#         return text.split("\n")[0].strip()
#     raw = match.group(1).strip()
#     try:
#         obj = json.loads(raw)
#         if "tool" not in obj or "args" not in obj:
#             return text
#         if obj["tool"] == "currency":
#             obj["args"]["from"] = obj["args"].get("from", "").upper()
#             obj["args"]["to"]   = obj["args"].get("to", "").upper()
#         if obj["tool"] == "weather":
#             if obj["args"].get("unit", "C") not in ("C", "F"):
#                 obj["args"]["unit"] = "C"
#         return f"<tool_call>{json.dumps(obj, separators=(',', ':'))}</tool_call>"
#     except json.JSONDecodeError:
#         return text

def _extract_tool_call(text: str) -> str:
    # Hard stop at known stop tokens
    for stop in ["<|im_end|>", "<|endoftext|>", "\nUser:", "\nuser:", "\nHuman:"]:
        if stop in text:
            text = text[:text.index(stop)]
    text = text.strip()

    # Truncate anything after </tool_call>
    if "</tool_call>" in text:
        text = text[:text.index("</tool_call>") + len("</tool_call>")]

    # If tool_call tag exists anywhere, extract ONLY that — ignore everything else
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if match:
        raw = match.group(1).strip()
        # Take only up to first closing brace of valid JSON
        try:
            obj = json.loads(raw)
            if "tool" not in obj or "args" not in obj:
                return "I can help with weather, calendar, conversions, currency, or SQL."
            if obj["tool"] == "currency":
                obj["args"]["from"] = obj["args"].get("from", "").upper()
                obj["args"]["to"]   = obj["args"].get("to", "").upper()
            if obj["tool"] == "weather":
                if obj["args"].get("unit", "C") not in ("C", "F"):
                    obj["args"]["unit"] = "C"
            return f"<tool_call>{json.dumps(obj, separators=(',', ':'))}</tool_call>"
        except json.JSONDecodeError:
            # Salvage: find first {...} block
            m2 = re.search(r'(\{.*?\})', text, re.DOTALL)
            if m2:
                try:
                    obj = json.loads(m2.group(1))
                    return f"<tool_call>{json.dumps(obj, separators=(',', ':'))}</tool_call>"
                except:
                    pass
            return "I can help with weather, calendar, conversions, currency, or SQL."

    # # No tool call tag — plain text refusal
    # # Strip non-ASCII garbage, return first clean line
    # clean = re.sub(r'[^\x00-\x7F]+', '', text).strip()
    # if clean:
    #     return clean.split("\n")[0].strip()
    # return "I can help with weather, calendar, conversions, currency, or SQL."

    # No tool call tag — plain text refusal
    clean = re.sub(r'[^\x00-\x7F]+', '', text).strip()  # strip non-ASCII
    clean = re.sub(r'[^\w\s\.\,\!\?\-\'\"]', '', clean).strip()  # strip remaining garbage
    lines = [l.strip() for l in clean.split("\n") if len(l.strip()) > 10]
    if lines:
        return lines[0]
    return "I can help with weather, calendar, conversions, currency, or SQL."


def run(prompt: str, history: list[dict]) -> str:
    _load()
    messages = [{"role": "system", "content": SYSTEM}]
    for turn in history:
        messages.append(turn)
    messages.append({"role": "user", "content": prompt})

    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=_tokenizer.eos_token_id,
        )

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=_tokenizer.eos_token_id,
            eos_token_id=_tokenizer.eos_token_id,  # ADD THIS
      )

    response = _tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    return _extract_tool_call(response)

if __name__ == "__main__":
    print("--- Weather ---")
    print(run("What's the weather in Lahore?", []))

    print("--- Fahrenheit ---")
    print(run("Temperature in Dubai in F?", []))

    print("--- Currency ---")
    print(run("Convert 100 USD to PKR.", []))

    print("--- Multi-turn ---")
    history = [
        {"role": "user",      "content": "Convert 100 USD to PKR."},
        {"role": "assistant", "content": '<tool_call>{"tool":"currency","args":{"amount":100,"from":"USD","to":"PKR"}}</tool_call>'},
    ]
    print(run("Now convert that to EUR.", history))

    print("--- Refusal ---")
    print(run("Book me a flight to London.", []))

    print("--- Hinglish ---")
    print(run("lahore ka mausam batao", []))
