# Data Prep Utilities

This folder contains utilities to format your three datasets into JSONL files suitable
for supervised fine-tuning with instruction-following models.

Usage examples:

1. Format advisory dataset JSONL:

```bash
python agents/data_prep/format_datasets.py --mode advisory --input advisory_raw.jsonl --output advisory_train.jsonl
```

2. Format individual-news dataset:

```bash
python agents/data_prep/format_datasets.py --mode news --input bitcoin_individual_raw.jsonl --output bitcoin_individual_train.jsonl
```

3. Format forecast dataset:

```bash
python agents/data_prep/format_datasets.py --mode forecast --input forecast_raw.jsonl --output forecast_train.jsonl
```

Assumptions:

- Input is JSONL with each line a JSON object representing a single example.
- For best results, include explicit `output` or `label` fields when available.

This is a minimal helper; you may need to adapt it to your exact dataset column names.
