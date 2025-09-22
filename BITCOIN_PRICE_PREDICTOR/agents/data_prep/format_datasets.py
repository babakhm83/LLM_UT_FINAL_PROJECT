"""Dataset formatting utilities for Bitcoin models.

This script provides functions and a small CLI to convert the three dataset types
described in the user's message into model-specific JSON/JSONL records matching
the prompts/templates used for training:

- bitcoin-investment-advisory-dataset -> advisory JSON prompt inputs
- bitcoin-individual-news-dataset -> news-analysis JSON outputs
- bitcoin-enhanced-prediction-dataset-with-comprehensive-news -> price-forecast training

The script is intentionally minimal and works with local files (CSV/JSON/JSONL).
It does not fetch remote datasets.
"""

from __future__ import annotations
import json
import argparse
from typing import Dict, Any, List


def format_advisory_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single advisory dataset row into model input/output pair.

    Expects `row` to contain keys like 'market_data', 'news_summary', etc.
    Returns a JSON object suitable for supervised fine-tuning where `instruction`
    is the short instruction and `output` is the desired advisory text/JSON.
    """
    instruction = "You are an elite institutional Bitcoin investment advisor. Provide comprehensive investment advisory based on the given market intelligence."
    # Attempt to assemble a compact context from known fields
    context_parts = []
    for k in ("MARKET_DATA", "NEWS_ANALYSIS", "DAILY_MARKET_ANALYSIS"):
        if k in row and row[k]:
            context_parts.append(f"{k}: {row[k]}")
    context = "\n\n".join(context_parts) if context_parts else json.dumps(row)

    return {
        "instruction": instruction,
        "input": context,
        "output": row.get("output", row.get("advisory", "")),
    }


def format_individual_news_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Format a news-analysis row to the small JSON spec used in `bitcoin-individual-news-dataset`.

    The dataset instruction expects a compact JSON output with sentiment, price_direction, impact_strength, timeframe, confidence and key_reason.
    """
    # Build input context
    title = row.get("title") or row.get("News Title") or row.get("news_title", "")
    summary = (
        row.get("summary") or row.get("News Summary") or row.get("news_summary", "")
    )
    market_context = row.get("Market Context") or row.get("market_context", "")

    instruction = (
        "Analyze Bitcoin news and predict price impact. Return JSON with this exact structure:\n"
        '{\n"sentiment": "bullish|neutral|bearish",\n"price_direction": "up|sideways|down",\n"impact_strength": "high|medium|low",\n"timeframe": "immediate|short_term|medium_term",\n"confidence": 0.75,\n"key_reason": "Brief explanation of main factor"\n}\n'
    )

    input_text = f"News Title: {title}\n\nNews Summary: {summary}\n\nMarket Context: {market_context}"

    # If the row already contains a target JSON, use it as `output`, else provide empty placeholder
    output = row.get("output") or row.get("label") or row.get("target") or ""

    return {"instruction": instruction, "input": input_text, "output": output}


def format_forecast_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Format the price-forecast training row.

    The target format is a JSON object with keys: action, confidence, stop_loss, take_profit, forecast_10d
    """
    instruction = "You are an expert quantitative crypto analyst. Given the DAILY CONTEXT produce a JSON with keys: action,confidence,stop_loss,take_profit,forecast_10d"

    context = (
        row.get("context")
        or row.get("Daily Context")
        or json.dumps(row.get("input", {}))
    )

    output = row.get("output") or row.get("label") or ""

    return {"instruction": instruction, "input": context, "output": output}


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(objs: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Format datasets into model training JSONL"
    )
    parser.add_argument(
        "--mode", choices=["advisory", "news", "forecast"], required=True
    )
    parser.add_argument("--input", required=True, help="Path to input JSONL/JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    args = parser.parse_args()

    # For simplicity expect input as JSONL of dicts
    items = read_jsonl(args.input)
    formatted = []
    for row in items:
        if args.mode == "advisory":
            formatted.append(format_advisory_row(row))
        elif args.mode == "news":
            formatted.append(format_individual_news_row(row))
        else:
            formatted.append(format_forecast_row(row))

    write_jsonl(formatted, args.output)


if __name__ == "__main__":
    main()
