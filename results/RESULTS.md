# Results — Before vs After (Full Detail)

Reference run: 2026-10-17, Kaggle T4 GPU, unsloth gemma-4-E4B-it; baseline via llama.cpp (Q4_K_M) on a CPU server.

## Headline metrics (20 held-out test cases)

| Metric | Before (zero-shot) | After (fine-tuned) |
|---|---|---|
| json_valid | 11/20 (55%) | 20/20 (100%) |
| schema_ok (all 8 keys) | 0/20 (0%) | 13/20 (65%) |
| all_fields_exact | 0/20 (0%) | 17/20 (85%) |
| invented_po_when_missing | 0 | 0 |

## Per-field accuracy

| Field | Before | After |
|---|---|---|
| vendor | 5% | 100% |
| invoice_number | 25% | 100% |
| invoice_date | 5% | 100% |
| base_amount | 0% | 95% |
| gst_amount | 10% | 85% |
| total_amount | 10% | 100% |
| po_reference | 25% | 100% |
| payment_terms_days | 5% | 100% |
| **Average** | **10.6%** | **97.5%** |

## Timing (Kaggle kernel, T4)

| Stage | Time |
|---|---|
| Setup: pip installs, model load, tokenization | ~3 min |
| Training: 3 epochs, 15 steps | 175.8 s (~3 min) |
| In-kernel eval: 20 cases | ~10 min |
| Total kernel | ~16 min |

## Loss curve

Epoch 1: 0.475 -> Epoch 2: 0.053 -> Epoch 3: 0.022.

## Methodology

- Dataset: 100 cases from data/make_invoice_dataset.py (seed 42); 80/20 split with a second seed-42 shuffle. Test set never used in training.
- Before: zero-shot gemma-4-E4B-it Q4_K_M GGUF via llama.cpp server (eval/eval_base_e4b.py), temperature 0.
- After: unsloth QLoRA r=16 alpha=16, LR 2e-4, 3 epochs, batch 2 x grad-accum 8, train_on_responses_only; eval via transformers generate, temperature 0 (training/train_reference_kaggle.py).
- Metrics identical on both sides: JSON parse, 8-key schema, exact field match (0.01 numeric tolerance), invented-PO counter.

## Honest notes

- schema_ok after fine-tuning is 65%: the model sometimes omits keys whose gold value is null — the values are correct, a key is missing. More no_po/missing_terms emphasis in training data would likely fix it.
- Weakest fields after training: gst_amount (85%) and base_amount (95%) — the incl_total_only edge cases that require computing base from an inclusive total.
- The before-run invented_po of 0 partly reflects that the base model rarely produced complete valid JSON at all; the after-run 0 is the meaningful safety result.