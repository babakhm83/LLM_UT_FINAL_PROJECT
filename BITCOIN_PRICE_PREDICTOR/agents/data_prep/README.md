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

# Data Prep Utilities

This folder contains a minimal helper script, `format_datasets.py`, to convert raw
examples into instruction-following JSONL records suitable for fine-tuning
instruction-tuned models. The script supports three dataset flavors used in this
project:

- bitcoin-investment-advisory-dataset (advisory)
- bitcoin-individual-news-dataset (news)
- bitcoin-enhanced-prediction-dataset (forecast)

The repository also provides canonical prompt/instruction helpers in
`agents/model_prompts.py` so you can use identical phrasing for training and
inference.

## Contents

- `format_datasets.py` - CLI to format JSONL -> training JSONL (instruction/input/output)
- `README.md` - this file

## Quick usage

Examples assume you have an input JSONL file where each line is a JSON object
for a single example. The formatter outputs a JSONL file where each line is an
object with at least `instruction`, `input`, and `output` keys.

Format advisory dataset (advisory):

```bash
python agents/data_prep/format_datasets.py --mode advisory --input advisory_raw.jsonl --output advisory_train.jsonl
```

Format individual-news dataset (news):

```bash
python agents/data_prep/format_datasets.py --mode news --input bitcoin_individual_raw.jsonl --output bitcoin_individual_train.jsonl
```

Format forecast dataset (forecast):

```bash
python agents/data_prep/format_datasets.py --mode forecast --input forecast_raw.jsonl --output forecast_train.jsonl
```

## Canonical instructions / prompts

The following exact instruction strings are provided by the project and used to
generate the training examples. They are available programmatically via
`agents.model_prompts` and are included here for clarity.

1. Advisory dataset (bitcoin-investment-advisory-dataset)

Instruction (exact):

```
You are an elite institutional Bitcoin investment advisor. Provide comprehensive investment advisory based on the given market intelligence.
```

Typical input shape (examples of fields we expect in the raw row):

```json
{
  "MARKET_DATA": "Current Price: $...; Price Range: ...; Next 10-Day Price Trend: [...]",
  "NEWS_ANALYSIS": "Structured news summaries and sentiment",
  "DAILY_MARKET_ANALYSIS": "Daily narrative"
}
```

Expected output: the `output` field should contain the advisory text or structured JSON used as the training target. After formatting, each training record will look like:

```json
{
  "instruction": "...advisory instruction...",
  "input": "<context aggregated>",
  "output": "<desired advisory text or JSON>"
}
```

2. Individual news dataset (bitcoin-individual-news-dataset)

Instruction (exact):

```
Analyze Bitcoin news and predict price impact. Return JSON with this exact structure:
{
	"sentiment": "bullish|neutral|bearish",
	"price_direction": "up|sideways|down",
	"impact_strength": "high|medium|low",
	"timeframe": "immediate|short_term|medium_term",
	"confidence": 0.75,
	"key_reason": "Brief explanation of main factor"
}
```

Typical input row format (raw):

```json
{
  "News Title": "...",
  "News Summary": "...",
  "Impact Tags": ["regulation", "liquidity"],
  "Market Context": "Bull 20% | Base 40% | Bear 40%"
}
```

Expected output: a single-line JSON string matching the schema above. After formatting:

```json
{
  "instruction": "...news instruction...",
  "input": "News Title: ...\n\nNews Summary: ...\n\nMarket Context: ...",
  "output": "{\"sentiment\": \"bearish\", ... }"
}
```

3. Forecast dataset (bitcoin-enhanced-prediction-dataset-with-comprehensive-news)

Instruction (exact):

```
You are an expert quantitative crypto analyst. Your tasks:
1) Analyze the context and decide an actionable stance for BTC-USD: BUY, SELL, or HOLD.
2) Forecast the NEXT 10 daily CLOSING prices (USD).

Return a single JSON object with EXACTLY these keys: {"action":"BUY|SELL|HOLD","confidence":<int 1-99>,"stop_loss":<price 2dp>,"take_profit":<price 2dp>,"forecast_10d":[<10 prices 2dp>]}
```

Typical input row (raw):

```json
{
	"Technical Price Analysis": "...",
	"Price History (Last 60 Days USD)": [...],
	"Macro & Commodities Context": {"Gold":...,"Oil":...},
	"Comprehensive News & Market Analysis": "..."
}
```

Expected output: A single JSON object as a string, for example:

```json
{
  "action": "SELL",
  "confidence": 95,
  "stop_loss": 9245.37,
  "take_profit": 11196.83,
  "forecast_10d": [
    8830.75, 9174.91, 8277.01, 6955.27, 7754.0, 7621.3, 8265.59, 8736.98,
    8621.9, 8129.97
  ]
}
```

## Formatting conventions

- Input: JSONL (one JSON object per line). If your raw source is CSV/TSV/JSON, convert to JSONL first.
- Keys: The formatter looks for common key names (see `format_datasets.py`) but is intentionally permissive — adapt it if your column names differ.
- Output: JSONL where each line has `instruction`, `input`, `output`.

## Programmatic helpers

Use the canonical instructions from `agents.model_prompts` to ensure prompt wording is identical between training and inference:

```python
from agents.model_prompts import (
		advisory_instruction,
		individual_news_instruction,
		forecast_instruction,
)

print(advisory_instruction())
```

## Next steps and tips

- Validate a sample of the generated JSONL before starting a large training job.
- Keep `instruction` wording stable across your dataset to avoid instruction drift.
- If outputs are long (multi-paragraph), ensure your training tooling supports long targets (token limits).
- Consider splitting very long input contexts into shorter chunks and label them carefully.

## Contact

If you want me to adapt the formatter to your exact column names or to add additional output formatting (e.g., token-wrapped prompt+input fields), tell me which dataset file you have and I will update the script.
