import json, random, os, hashlib

SEED = 42
OUT = "/root/workspace/projects/invoice_extract_dataset"
os.makedirs(OUT, exist_ok=True)
random.seed(SEED)
NL = chr(10)

VENDORS = ["Sharma Traders Pvt Ltd", "Gupta Enterprises", "Verma Supplies Co",
    "Mehta and Sons", "Kiran Wholesale Mart", "Deepak Materials Pvt Ltd",
    "Bansal Packaging Industries", "Nanda Electricals", "Prakash Hardware Store",
    "Sundaram Textiles Ltd", "Kapoor Imports", "Reddy Agro Supplies",
    "Chawla Chemicals Pvt Ltd", "Iyer Logistics Solutions", "Grover Steel Traders"]
GST_RATES = [5, 12, 18, 28]
TERMS_OPTIONS = [15, 30, 45, 60]
STYLES = ["email", "short", "tabular", "whatsapp"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def rupee(x):
    return "Rs." + format(int(x), ",")

def rand_date():
    return "2026-%02d-%02d" % (random.randint(1, 12), random.randint(1, 28))

def human_date(iso):
    y, m, d = iso.split("-")
    return "%s %s %s" % (d, MONTHS[int(m) - 1], y)

def render(style, vendor, inv_no, iso, base, gst, total, rate, po, terms, incl_only):
    total_s = rupee(total)
    base_s = rupee(base)
    gst_s = rupee(gst)
    if style == "email":
        s = "Dear Sir, please find attached invoice " + inv_no + " from " + vendor + " dated " + human_date(iso) + ". "
        if incl_only:
            s += "Total payable " + total_s + " inclusive of GST at " + str(rate) + "%. "
        else:
            s += "Base amount " + base_s + " plus GST " + gst_s + ", total payable " + total_s + ". "
        if po:
            s += "PO ref " + po + ". "
        if terms:
            s += "Payment terms " + str(terms) + " days."
        return s
    if style == "short":
        s = "Invoice " + inv_no + " | " + vendor + " | " + human_date(iso) + " | "
        if incl_only:
            s += "Total " + total_s + " (incl. GST " + str(rate) + "%)"
        else:
            s += "Base " + base_s + " + GST " + str(rate) + "% = " + total_s
        if po:
            s += " | PO " + po
        if terms:
            s += " | Terms " + str(terms) + "d"
        return s
    if style == "tabular":
        lines = ["Vendor: " + vendor, "Invoice No: " + inv_no, "Date: " + human_date(iso)]
        if incl_only:
            lines.append("Total (incl. GST " + str(rate) + "%): " + total_s)
        else:
            lines.append("Base Amount: " + base_s)
            lines.append("GST (" + str(rate) + "%): " + gst_s)
            lines.append("Total: " + total_s)
        if po:
            lines.append("PO: " + po)
        if terms:
            lines.append("Terms: " + str(terms) + " days")
        return NL.join(lines)
    s = "bhaiya invoice aa gaya hai - " + inv_no + ", " + vendor + ", date " + human_date(iso) + ". "
    if incl_only:
        s += "Total " + total_s + " (GST " + str(rate) + "% included). "
    else:
        s += "Total " + total_s + " hai, usme GST " + gst_s + " shamil hai. "
    if po:
        s += "PO number " + po + " hai. "
    if terms:
        s += "Payment " + str(terms) + " din ke andar."
    return s

def make_case(i, used_inv):
    vendor = VENDORS[i % len(VENDORS)]
    while True:
        inv_no = "INV-2026-" + str(random.randint(10000, 99999))
        if inv_no not in used_inv:
            used_inv.add(inv_no)
            break
    iso = rand_date()
    rate = GST_RATES[i % len(GST_RATES)]
    base = random.randrange(50, 2000) * 100
    gst = base * rate // 100
    total = base + gst
    incl_only = (i % 7 == 3)
    no_po = (i % 5 == 4)
    po = None
    if not no_po:
        po = "55" + str(random.randint(1000000, 9999999))
    missing_terms = (i % 20 == 9)
    terms = None
    if not missing_terms:
        terms = random.choice(TERMS_OPTIONS)
    style = STYLES[i % 4]
    text = render(style, vendor, inv_no, iso, base, gst, total, rate, po, terms, incl_only)
    answer = json.dumps({
        "vendor": vendor,
        "invoice_number": inv_no,
        "invoice_date": iso,
        "base_amount": float(base),
        "gst_amount": float(gst),
        "total_amount": float(total),
        "po_reference": po,
        "payment_terms_days": terms,
    }, indent=2)
    edges = []
    if no_po:
        edges.append("no_po")
    if incl_only:
        edges.append("incl_total_only")
    if missing_terms:
        edges.append("missing_terms")
    if not edges:
        edges.append("none")
    return {"messages": [
        {"role": "user", "content": "Extract the invoice fields as JSON:" + NL + NL + text},
        {"role": "assistant", "content": answer},
    ], "meta": {"case": i, "format": style, "edges": ",".join(edges)}}

used_inv = set()
golden = [make_case(i, used_inv) for i in range(100)]

SCHEMA = set(["vendor", "invoice_number", "invoice_date", "base_amount", "gst_amount", "total_amount", "po_reference", "payment_terms_days"])
errors = []
prompts = set()
for i, p in enumerate(golden):
    try:
        a = json.loads(p["messages"][1]["content"])
    except Exception:
        errors.append("case " + str(i) + ": assistant JSON unparseable")
        continue
    if set(a.keys()) != SCHEMA:
        errors.append("case " + str(i) + ": schema mismatch")
    if abs(a["base_amount"] + a["gst_amount"] - a["total_amount"]) > 0.001:
        errors.append("case " + str(i) + ": base+gst != total")
    h = hashlib.md5(p["messages"][0]["content"].encode()).hexdigest()
    if h in prompts:
        errors.append("case " + str(i) + ": duplicate prompt")
    prompts.add(h)
if errors:
    print("VALIDATION FAILED:")
    for e in errors:
        print(" - " + e)
    raise SystemExit(1)

srng = random.Random(42)
idx = list(range(100))
srng.shuffle(idx)
train = [golden[i] for i in idx[:80]]
test = [golden[i] for i in idx[80:]]

def write_jsonl(name, rows):
    with open(os.path.join(OUT, name), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + NL)

write_jsonl("invoice_golden_v1_100.jsonl", golden)
write_jsonl("invoice_train_v1_80.jsonl", train)
write_jsonl("invoice_test_v1_20.jsonl", test)

fmt = {}
edges = {}
for p in golden:
    fmt[p["meta"]["format"]] = fmt.get(p["meta"]["format"], 0) + 1
    for e in p["meta"]["edges"].split(","):
        edges[e] = edges.get(e, 0) + 1

readme = """# Invoice Extract Dataset v1 (Golden, 100 handwritten-quality cases)

Task: messy invoice text (email / short / tabular / WhatsApp style) -> strict JSON with
vendor, invoice_number, invoice_date, base_amount, gst_amount, total_amount,
po_reference, payment_terms_days.

Files:
- invoice_golden_v1_100.jsonl : master golden dataset (100 cases, with meta)
- invoice_train_v1_80.jsonl  : train split (80 cases)
- invoice_test_v1_20.jsonl   : held-out test split (20 cases) - never train on these
- make_invoice_dataset.py    : generator (seed 42, fully reproducible)

Edge cases by design: no_po (PO missing -> null), incl_total_only (base must be
computed from inclusive total), missing_terms (terms missing -> null).

Validation: 100/100 assistant outputs parse as JSON, schema exact, base+gst=total
holds, 0 duplicate prompts.

Created: 2026-10-16 for the E4B invoice-extraction teaching track.
Regenerate: python3 make_invoice_dataset.py
"""

with open(os.path.join(OUT, "README.md"), "w") as f:
    f.write(readme)

print("GOLDEN DATASET OK - 100 cases, validation passed")
print("Formats: " + json.dumps(fmt))
print("Edges:   " + json.dumps(edges))
print("Train: " + str(len(train)) + " | Test: " + str(len(test)))
print("---- SAMPLE (case 0) ----")
print("USER:")
print(golden[0]["messages"][0]["content"])
print("ASSISTANT:")
print(golden[0]["messages"][1]["content"])
