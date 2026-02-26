#!/usr/bin/env python3
import argparse, json, os, sys
from typing import Optional, Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextStreamer,
)

try:
    from peft import PeftModel
except Exception as e:
    print("Please `pip install peft`.", file=sys.stderr)
    raise

# to run the script: `python agent.py --adapter_dir ./phi35-mini-api-lora/checkpoint-418`

SYSTEM = """You are a tool-calling assistant.

When a user mentions a geographic location,
you MUST respond ONLY with valid JSON in this format:

{
  "tool": "geographic_interest",
  "arguments": {
    "location": "<city>"
  }
}

Do not explain.
Do not add extra text.
Return exactly one JSON object.
"""

BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
TIMEZONE = "America/New_York"  # adjust to your runtime TZ needs

def run_background_api_call(area:str):
    pass

def load_model_and_tokenizer(adapter_dir: str, quantize: bool = False):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if quantize:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb, device_map="auto"
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
        )

    model = PeftModel.from_pretrained(base, adapter_dir, local_files_only=True)
    model.eval()
    model.config.use_cache = True
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    return tok, model

@torch.no_grad()
def generate_once(model, tok, user_input: str, max_new_tokens: int,
                  do_sample: bool = False,
                  temperature: float = 0.3,
                  top_p: float = 0.9,
                  stream: bool = False) -> str:

    prompt = (
        f"[SYSTEM]\n{SYSTEM}\n\n"
        f"[USER]\n{user_input}\n\n"
        f"[ASSISTANT]\n"
    )

    inputs = tok(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        do_sample=False,      # IMPORTANT
        temperature=0.0,
        max_new_tokens=200,
        eos_token_id=tok.eos_token_id
    )

    reply1 = tok.decode(output[0], skip_special_tokens=True)

    # Remove the prompt part
    reply = reply1[len(prompt):].strip()       

    return reply

# Very light JSON detector: tries to parse a single JSON object from a string start
def extract_first_json_object(text: str) -> Optional[str]:
    # Find first '{' and attempt brace balancing
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None

def looks_like_json_tool_call(s: str) -> Optional[Dict]:
    """return obj if it resembles a json tool call and has "geographic_interest"""
    s = s.replace(" ", "").replace("\n", "").strip()

    js = extract_first_json_object(s)

    if not js:
        print("not json")
        return None
    try:
        obj = json.loads(js)
        print(f"loaded in {obj}")
        if (
            isinstance(obj, dict)
            and obj.get("tool") == "geographic_interest"
            and isinstance(obj.get("arguments"), dict)
        ):
            # Basic schema check
            args = obj["arguments"]
            if "location" in args:
                return obj
    except Exception:
        return None
    return None


# Chat controller
def chat_loop(adapter_dir: str, quantize: bool = False):
    tok, model = load_model_and_tokenizer(adapter_dir, quantize=quantize)

    print("Chat ready. Type 'exit' to quit.\n")
    while True:
        user = input("You: ").strip()
        if not user or user.lower() in {"exit", "quit"}:
            break

        # 1) First pass:, aiming for JSON tool call (if any)
        reply1 = generate_once(model, tok, user, 200)

        tool_call = looks_like_json_tool_call(reply1)
        if tool_call:
            # RUN TOOL
            args = tool_call["arguments"]
            Area = args["location"]
            result = run_background_api_call(Area)

        else:
            # No tool call → normal chat
            print(f"Assistant: {reply1}\n")


# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fine-tuned Phi-3.5-mini-instruct with tool-calling chat loop.")
    parser.add_argument("--adapter_dir", type=str, required=True,
                        help="Local path to your PEFT adapter checkpoint (e.g., ./phi35-mini-api-lora/checkpoint-198)")
    parser.add_argument("--quantize", action="store_true",
                        help="Load base in 4-bit (QLoRA inference).")
    args = parser.parse_args()

    if not os.path.isdir(args.adapter_dir):
        print(f"Adapter path not found: {args.adapter_dir}", file=sys.stderr)
        sys.exit(1)

    chat_loop(args.adapter_dir, quantize=args.quantize)
