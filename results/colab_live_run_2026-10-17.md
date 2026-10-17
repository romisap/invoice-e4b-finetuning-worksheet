# Live Colab Run — Evidence (2026-10-17)

Full end-to-end run of the published worksheet, executed live on **Google Colab (Tesla T4 GPU)** — an independent confirmation of the Kaggle reference run.

- **Notebook:** `worksheet/E4B_Invoice_Finetuning_Worksheet.ipynb` (live outputs embedded)
- **Model:** `unsloth/gemma-4-E4B-it` — QLoRA 4-bit, r=16, alpha=16, lr 2e-4, 3 epochs, batch 2 x grad-accum 8 (15 optimizer steps, ~3 min)
- **Data:** seed-42 generator, 80 train / 20 held-out test cases
- **Eval:** same harness before and after, temperature 0 (deterministic)
- **Environment:** Unsloth 2026.10.6, Transformers 5.18.0, Torch 2.11.0+cu130, CUDA 13.0, Python 3.13, Colab T4 (15 GB)

## Before vs After (live, Colab T4)

| Metric | Before (zero-shot) | After (fine-tuned) |
|---|---|---|
| Valid JSON | 20/20 (100%) | 20/20 (100%) |
| Full schema (all 8 keys) | 0/20 (0%) | 13/20 (65%) |
| All 8 fields exact | 0/20 (0%) | 17/20 (85%) |
| Invented PO (safety, must be 0) | 0 | 0 |
| Average field accuracy | 30.0% | 97.5% |

## Per-field accuracy (live)

| Field | Before | After |
|---|---|---|
| vendor | 20.0% | 100.0% |
| invoice_number | 95.0% | 100.0% |
| invoice_date | 0.0% | 100.0% |
| base_amount | 25.0% | 95.0% |
| gst_amount | 30.0% | 85.0% |
| total_amount | 15.0% | 100.0% |
| po_reference | 45.0% | 100.0% |
| payment_terms_days | 10.0% | 100.0% |

## Raw eval summaries (verbatim from the live notebook outputs)

Before:

```json
{
  "tag": "before",
  "total": 20,
  "json_valid": 20,
  "schema_ok": 0,
  "all_fields_exact": 0,
  "invented_po_when_missing": 0,
  "field_accuracy": {
    "vendor": 0.2,
    "invoice_number": 0.95,
    "invoice_date": 0.0,
    "base_amount": 0.25,
    "gst_amount": 0.3,
    "total_amount": 0.15,
    "po_reference": 0.45,
    "payment_terms_days": 0.1
  }
}
```

After:

```json
{
  "tag": "after",
  "total": 20,
  "json_valid": 20,
  "schema_ok": 13,
  "all_fields_exact": 17,
  "invented_po_when_missing": 0,
  "field_accuracy": {
    "vendor": 1.0,
    "invoice_number": 1.0,
    "invoice_date": 1.0,
    "base_amount": 0.95,
    "gst_amount": 0.85,
    "total_amount": 1.0,
    "po_reference": 1.0,
    "payment_terms_days": 1.0
  }
}
```

## Comparison with the Kaggle reference run

The after-side reproduces almost exactly: 17/20 all-fields-exact (85%) and 97.5% average field accuracy on both machines. The before-side differs in degree, not in kind: on Colab the zero-shot model emitted valid JSON for all 20 cases (vs 55% on Kaggle — different library versions), but still matched almost no fields exactly (avg 30.0% vs 10.6%). Either way, prompting alone produces unusable output for this task; fine-tuning on 80 examples produces business-correct JSON.

Screenshots of the live run: `results/screenshots/`

## Clean full re-run (same day, ~23 minutes) — zero errors confirmed

To confirm the published worksheet runs end-to-end without errors (after the comparison-cell SyntaxError fix in commit 9a2636e), the notebook was re-executed completely fresh: loaded from GitHub, new T4 runtime, Run All.

**Result: all 8 code cells executed clean — no Tracebacks, no SyntaxError.** The training cell ran normally (the earlier "already added LoRA adapters" error was a duplicate-run artifact on an already-trained kernel, not a notebook bug), and the previously-broken comparison cell printed the final table.

Final cell output (verbatim):

```
| Metric | Before | After |
|---|---|---|
| Valid JSON | 20/20 (100.0%) | 20/20 (100.0%) |
| Full schema (all 8 keys) | 0/20 (0.0%) | 13/20 (65.0%) |
| All 8 fields exact | 0/20 (0.0%) | 17/20 (85.0%) |
| Invented PO (safety, must be 0) | 0 | 0 |

| Field | Before | After |
|---|---|---|
| vendor | 20.0% | 100.0% |
| invoice_number | 95.0% | 100.0% |
| invoice_date | 0.0% | 100.0% |
| base_amount | 25.0% | 95.0% |
| gst_amount | 30.0% | 85.0% |
| total_amount | 15.0% | 100.0% |
| po_reference | 45.0% | 100.0% |
| payment_terms_days | 10.0% | 100.0% |

Average field accuracy: 30.0% -> 97.5%
```

Numbers are identical to the first live run of the day — temperature 0 and seed 42 make the whole notebook reproducible.

Screenshots of the clean run (GPU/installs state, execution in progress, final table): `results/screenshots/colab_cleanrun_gpu_installs.png`, `colab_cleanrun_executing.png`, `colab_cleanrun_final_table.png`.
