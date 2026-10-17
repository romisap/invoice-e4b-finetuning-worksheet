# Invoice Extract Dataset v1 (Golden, 100 handwritten-quality cases)

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
