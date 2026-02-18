# train_phi35_lora.py
import os, json, torch, random
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer, default_data_collator)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers.trainer_utils import get_last_checkpoint
from clearml import Task

"""Script to train the LORA model with case A data which is geographic interest. Integrated with ClearML for management and vizualization."""
task = Task.create(project_name="LoRA geography assistant", task_name="LoRA task Version 2 with geographic information", add_task_init_call=True)

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
MAX_LEN  = 2048
VAL_SIZE  = 0.1         # 10% validation
SEED      = 42

SYSTEM = ( """You are a travel assistant interacting with a user. 
Your job is to chat naturally in order to gather information.

You can perform one action:
1) Add geographic area
Trigger when the user wants to create or define a new geographic interest area.
Tool call format (strict):
{"tool":"geographic_interest","arguments":{"location":<Area name>}}

For area, the required field is a geographic area that you will deduce from any background data in the conversation. 
If the value needed field is missing or ambiguous ask one short question—do not call the tool yet.
Do NOT include prose, markdown, or extra keys. Never mix schemas between tools.
After the tool runs, you will receive [TOOL_RESULT] {…}; then produce a 1–2 sentence, grounded summary if sucessful or not.
Never fabricate results; never mix prose with the JSON tool call; keep clarifying questions to a single sentence.
Be concise, precise, and consistent with the schema and formatting above."""
)

caseA_path = "../training_data/CaseA.jsonl"

# --- 4-bit base (QLoRA) ---
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

#model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
#    quantization_config=bnb_cfg,
    dtype=torch.bfloat16,
#    dtype=torch.float16,
    device_map="auto",
)

#peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","v_proj"])
#model = get_peft_model(model, peft_cfg)

# ✅ Important for k-bit training
#model = prepare_model_for_kbit_training(model)

# LoRA targets for Phi-3.5 (attention + MLP projections)
lora_cfg = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05,
#    target_modules=["q_proj","k_proj","v_proj","o_proj", "up_proj","down_proj","gate_proj"],
    target_modules=['down_proj', 'gate_up_proj', 'o_proj', 'qkv_proj'],
#    target_modules=["q_proj","v_proj","o_proj"],
    task_type="CAUSAL_LM"
)


names = set()
for n, m in model.named_modules():
    base = n.split(".")[-1]
    names.add(base)
print(sorted([x for x in names if "proj" in x or "gate" in x]))


model = get_peft_model(model, lora_cfg)

names = set()
for n, m in model.named_modules():
    base = n.split(".")[-1]
    names.add(base)
print(sorted([x for x in names if "proj" in x or "gate" in x]))

# Gradient checkpointing pairs with use_cache=False
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

task.connect(lora_cfg)

# -------------------------
# Tokenizer
# -------------------------
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# -------------------------
# Helpers
# -------------------------
def canonical_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)

def ensure_str_or_json(obj):
    """Return canonical JSON string if obj is list/dict/JSON-string; else return as-is."""
    if isinstance(obj, (list, dict)):
        return canonical_json(obj)
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                return canonical_json(json.loads(s))
            except Exception:
                return obj
        return obj
    return ""

def build_prompt(ex):
    tools_str = ensure_str_or_json(ex.get("tools", []))
    return (
        f"[SYSTEM]\n{SYSTEM}\n\n"
        f"[TOOLS]\n{tools_str}\n\n"
        f"[USER]\n{ex.get('input','')}\n\n"
        f"[ASSISTANT]\n"
    )
    
def tokenize_with_mask(ex):
    """
    Supports:
    - A: assistant_text
    - B: assistant_tool_json
    - C: assistant_tool_json + tool_result + assistant_text_after_result
    Masks everything except assistant spans (JSON and/or final chat).
    """
    prompt = build_prompt(ex)

    # Encode prompt (context only)
    enc_p = tok(prompt, add_special_tokens=False)
    input_ids = enc_p["input_ids"][:]
    attention = [1] * len(input_ids)
    labels    = [-100] * len(input_ids)

    parts = []  # list[(text, supervise_flag)]

    # B/C: tool call JSON span
    if ex.get("assistant_tool_json") not in (None, ""):
        tool_json = ex["assistant_tool_json"]
        if isinstance(tool_json, str):
            try:
                tool_json = json.loads(tool_json)
            except Exception:
                pass
        #input including the json for the reply. canonic tries to rmv spaces, trun to a str.
        y1 = canonical_json(tool_json) if isinstance(tool_json, (dict, list)) else str(tool_json)
        parts.append((y1 + tok.eos_token, True)) # eos = end of sentence.

    # C: tool result context + second assistant span
    if ex.get("tool_result") not in (None, ""):
        tr = ex["tool_result"]
        if isinstance(tr, str):
            try:
                tr = json.loads(tr)
            except Exception:
                pass
        tool_block = "\n\n[TOOL_RESULT]\n" + (canonical_json(tr) if isinstance(tr, (dict, list)) else str(tr)) + "\n\n[ASSISTANT]\n"
        parts.append((tool_block, False))  # context-only

    # A or C final chat span
    y2_text = ex.get("assistant_text") or ex.get("assistant_text_after_result")
    if y2_text:
        parts.append((str(y2_text) + tok.eos_token, True))

    # Must have at least one supervised span
    if not any(flag for _, flag in parts):
        raise ValueError("Row has no supervised assistant span.")

    # Tokenize parts and concat with proper masking
    for text, supervise in parts:
        enc = tok(text, add_special_tokens=False)
        input_ids.extend(enc["input_ids"])
        attention.extend([1]*len(enc["input_ids"]))
        labels.extend(enc["input_ids"] if supervise else [-100]*len(enc["input_ids"]))

    # Truncate consistently (head-only here; customize if needed)
    if len(input_ids) > MAX_LEN:
        input_ids = input_ids[:MAX_LEN]
        attention = attention[:MAX_LEN]
        labels    = labels[:MAX_LEN]

    # Ensure some supervision remains
    if not any(l != -100 for l in labels):
        labels[-1] = tok.eos_token_id

    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}

# -------------------------
# Load each file, split inside, map, then concat
# -------------------------
def load_and_split(path, split_name="train", val_size=VAL_SIZE, seed=SEED):
    ds = load_dataset("json", data_files={split_name: path})[split_name]
    # Shuffle before split for better mix
    ds = ds.shuffle(seed=seed)
    parts = ds.train_test_split(test_size=val_size, seed=seed)
    return parts["train"], parts["test"]

def preprocess_training_data():
    trainA, valA = load_and_split(caseA_path)

    # Tokenize each split separately (removing raw columns)
    proc_trainA = trainA.map(tokenize_with_mask, remove_columns=trainA.column_names)

    proc_valA = valA.map(tokenize_with_mask, remove_columns=valA.column_names)

    # Concatenate per split to preserve A/B/C balance
    train_ds = concatenate_datasets([proc_trainA]).shuffle(seed=SEED) # shuffle is take 10% of examples, use as goal(what u compare against). so like split up the train and val.
    val_ds   = concatenate_datasets([proc_valA]).shuffle(seed=SEED) # changing the weights of the neural network (under the hood)

    ds = DatasetDict(train=train_ds, val=val_ds)

    print("Train size:", len(ds["train"]), " Val size:", len(ds["val"]))
    # quick sanity check
    sample = ds["train"][0]
    assert set(sample.keys()) == {"input_ids","attention_mask","labels"}
    assert any(l != -100 for l in sample["labels"])
    return ds

# --- Training args ---
args = TrainingArguments(
    output_dir="phi35-mini-api-lora",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=32,   # raise/lower for your VRAM
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    num_train_epochs=2,
    bf16=True,
    weight_decay=0.01,
    logging_steps=25,
 #   evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    gradient_checkpointing=True,
    report_to="none"
)

if __name__ == "__main__":
    ds = preprocess_training_data()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        #tokenizer=tok,
        processing_class=tok,
        data_collator=default_data_collator,
    )
    model.print_trainable_parameters()  # sanity: shows nonzero trainable params
    
    # Optional: quickly verify labels aren’t all ignored
    batch = next(iter(trainer.get_train_dataloader()))
    assert "labels" in batch and (batch["labels"] != -100).any(), "All labels masked!"

    print(model.generation_config)  # current defaults
    print("Untrained Model:")

    prompt = (
        f"[SYSTEM]\n{SYSTEM}\n\n"
        f"[USER]\nI'm interested in going to New York\n\n"
        f"[ASSISTANT]\n"
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    print(inputs)

    out = model.generate(
        **inputs,
        do_sample=True,          # <-- enable sampling
        temperature=0.3,         # now respected
        top_p=0.95,              # (optional) nucleus sampling
        max_new_tokens=350
    )
    print(out)
    #out = model.generate(**inputs, max_new_tokens=200, temperature=0.9, streamer=streamer)
    print(tok.decode(out[0], skip_special_tokens=True))

    result = trainer.train()
    trainer.save_model()

    metrics = result.metrics          # dict with train_runtime, train_loss, etc.
    print("Final train metrics:", metrics)
    print(model.generation_config)  # current defaults

