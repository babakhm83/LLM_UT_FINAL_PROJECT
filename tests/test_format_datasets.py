import json
from agents.data_prep.format_datasets import (
    format_advisory_row,
    format_individual_news_row,
    format_forecast_row,
)


def test_format_advisory_minimal():
    row = {"MARKET_DATA": "Price: $1000", "output": '{"action": "HOLD"}'}
    out = format_advisory_row(row)
    assert "instruction" in out and "input" in out and "output" in out


def test_format_news_handles_missing_fields():
    row = {"title": "Test", "summary": "Something happened."}
    out = format_individual_news_row(row)
    assert out["instruction"].startswith("Analyze Bitcoin news")
    assert "input" in out


def test_format_forecast_minimal():
    row = {"context": "Daily Context sample", "output": '{"action": "SELL"}'}
    out = format_forecast_row(row)
    assert "instruction" in out and "input" in out and "output" in out
