import os, sys, json, glob
os.environ["UNSLOTH_FORCE_FLOAT32"] = "1"
os.system("pip install --pre --upgrade --no-deps unsloth > /dev/null 2>&1")
os.system("pip install --no-deps --force-reinstall git+https://github.com/unslothai/unsloth-zoo.git > /dev/null 2>&1")
os.system("pip install --no-deps trl peft accelerate bitsandbytes datasets > /dev/null 2>&1")

import torch
from unsloth import FastLanguageModel

train_files = glob.glob("/kaggle/input/**/invoice_train_v1_80.jsonl", recursive=True)
test_files = glob.glob("/kaggle/input/**/invoice_test_v1_20.jsonl", recursive=True)
print("TRAIN FILE:", train_files, flush=True)
print("TEST FILE:", test_files, flush=True)
if not train_files or not test_files:
    print("DATASET NOT MOUNTED - re-push needed", flush=True)
    sys.exit(1)

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

train_pairs = load_jsonl(train_files[0])
test_pairs = load_jsonl(test_files[0])
print("train pairs:", len(train_pairs), "test pairs:", len(test_pairs), flush=True)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-4-E4B-it",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    device_map={"": 0},
)

probe = tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False)
print("TEMPLATE PROBE:", repr(probe), flush=True)

model = FastLanguageModel.get_peft_model(
    model,
    r=16, alpha=16, dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)

from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

train_ds = Dataset.from_list([{"messages": p["messages"]} for p in train_pairs])

trainer = train_on_responses_only(
    SFTTrainer(
        model=model,
        train_dataset=train_ds,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=3,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            logging_steps=5,
            output_dir="/kaggle/working/out",
            report_to="none",
        ),
        tokenizer=tokenizer,
    ),
    instruction_part="<|turn>user",
    response_part="<|turn>model",
)

trainer.train()
print("TRAIN HISTORY LAST:", trainer.state.log_history[-1], flush=True)

model.save_pretrained("/kaggle/working/invoice_e4b_lora")
tokenizer.save_pretrained("/kaggle/working/invoice_e4b_lora")
print("LORA SAVED", flush=True)

FastLanguageModel.for_inference(model)
inner = getattr(tokenizer, "tokenizer", tokenizer)

SCHEMA = ["vendor", "invoice_number", "invoice_date", "base_amount", "gst_amount", "total_amount", "po_reference", "payment_terms_days"]

def extract_json(text):
    s = text.find("{")
    if s == -1:
        return None
    depth = 0
    for i in range(s, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[s:i+1])
                except Exception:
                    return None
    return None

field_ok = {k: 0 for k in SCHEMA}
json_valid = 0
schema_ok = 0
all_ok = 0
invented_po = 0
results = []

for n, c in enumerate(test_pairs):
    user = c["messages"][0]["content"]
    gold = json.loads(c["messages"][1]["content"])
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    ids = inner(prompt, return_tensors="pt").input_ids.to("cuda")
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    try:
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        reply = inner.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    except Exception as e:
        reply = "ERROR: " + str(e)
    pred = extract_json(reply)
    row = {"case": n, "json_valid": pred is not None, "reply_head": reply[:200]}
    if pred is not None:
        json_valid += 1
        ok_fields = []
        if set(pred.keys()) == set(SCHEMA):
            schema_ok += 1
        for k in SCHEMA:
            pv = pred.get(k)
            gv = gold.get(k)
            if isinstance(gv, (int, float)) and isinstance(pv, (int, float)) and not isinstance(gv, bool):
                same = abs(float(pv) - float(gv)) < 0.01
            else:
                same = (pv == gv)
            if same:
                field_ok[k] += 1
                ok_fields.append(k)
        if len(ok_fields) == len(SCHEMA):
            all_ok += 1
        if gold.get("po_reference") is None and pred.get("po_reference") is not None:
            invented_po += 1
        row["ok_fields"] = len(ok_fields)
        row["pred"] = pred
    row["gold"] = gold
    results.append(row)
    print("case " + str(n) + ": json_valid=" + str(row["json_valid"]) + " ok_fields=" + str(row.get("ok_fields", 0)) + "/" + str(len(SCHEMA)), flush=True)

total = len(test_pairs)
summary = {
    "model": "gemma-4-E4B-it + LoRA r16 (fine-tuned on 80 invoice pairs)",
    "total": total,
    "json_valid": json_valid,
    "schema_ok": schema_ok,
    "all_fields_exact": all_ok,
    "invented_po_when_missing": invented_po,
    "field_accuracy": {k: round(field_ok[k] / total, 3) for k in SCHEMA},
}
with open("/kaggle/working/eval_finetuned_results.json", "w") as f:
    json.dump({"summary": summary, "cases": results}, f, indent=2)

print("====SUMMARY====", flush=True)
print(json.dumps(summary, indent=2), flush=True)
print("EVAL_DONE", flush=True)
