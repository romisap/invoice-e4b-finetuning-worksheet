import json, os, time, urllib.request

DATA = "/root/workspace/projects/invoice_extract_dataset"
TEST = os.path.join(DATA, "invoice_test_v1_20.jsonl")
OUT = os.path.join(DATA, "eval_base_results.json")
URL = "http://127.0.0.1:8420/v1/chat/completions"
SCHEMA = ["vendor", "invoice_number", "invoice_date", "base_amount", "gst_amount", "total_amount", "po_reference", "payment_terms_days"]

def post(prompt):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 640,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"]

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

cases = []
with open(TEST) as f:
    for line in f:
        line = line.strip()
        if line:
            cases.append(json.loads(line))

results = []
field_ok = {k: 0 for k in SCHEMA}
json_valid = 0
schema_ok = 0
all_ok = 0
invented_po = 0

for n, c in enumerate(cases):
    user = c["messages"][0]["content"]
    gold = json.loads(c["messages"][1]["content"])
    t0 = time.time()
    try:
        reply = post(user)
    except Exception as e:
        reply = "ERROR: " + str(e)
    dt = time.time() - t0
    pred = extract_json(reply)
    row = {"case": n, "seconds": round(dt, 1), "json_valid": pred is not None, "reply_head": reply[:200]}
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
        row["bad_fields"] = [k for k in SCHEMA if k not in ok_fields]
        row["pred"] = pred
    row["gold"] = gold
    results.append(row)
    print("case " + str(n) + ": json_valid=" + str(row["json_valid"]) + " ok_fields=" + str(row.get("ok_fields", 0)) + "/" + str(len(SCHEMA)) + " (" + str(row["seconds"]) + "s)", flush=True)

total = len(cases)
summary = {
    "model": "gemma-4-E4B-it Q4_K_M (base, zero-shot, temp 0)",
    "total": total,
    "json_valid": json_valid,
    "schema_ok": schema_ok,
    "all_fields_exact": all_ok,
    "invented_po_when_missing": invented_po,
    "field_accuracy": {k: round(field_ok[k] / total, 3) for k in SCHEMA},
}
with open(OUT, "w") as f:
    json.dump({"summary": summary, "cases": results}, f, indent=2)

print("---- SUMMARY ----")
print(json.dumps(summary, indent=2))
