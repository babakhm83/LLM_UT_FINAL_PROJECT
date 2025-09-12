# Bitcoin Investment Advisory Pipeline – Full Technical Documentation

This directory contains the modular AI pipeline that ingests fresh Bitcoin news, structures and summarizes it, infers directional/effect probabilities, generates short‑horizon forecasts, and synthesizes an institutional‑grade investment advisory.

---

## 1. High‑Level Architecture

```
        ┌────────────┐        ┌──────────────┐        ┌────────────────┐        ┌───────────────┐        ┌─────────────────────┐
        │  News RSS  │  --->  │  Collector   │  --->  │  Bucketer      │  --->  │ Summarization  │  --->  │ Effects Analysis     │
        └────────────┘        └──────────────┘        └────────────────┘        └───────────────┘        └─────────────────────┘
                                                                                                                        │
                                                                                                                        ▼
                                           ┌──────────────────────────┐        ┌─────────────────────┐        ┌─────────────────────────┐
                                           │ Price History Retrieval  │  --->  │  Forecast Generation │  --->  │ Advisory (Structured+MD) │
                                           └──────────────────────────┘        └─────────────────────┘        └─────────────────────────┘
```

Fallback logic ensures graceful degradation if external LLMs or custom endpoints are offline.

---

## 2. Core Components (Files)

| File                             | Purpose                                                                        |
| -------------------------------- | ------------------------------------------------------------------------------ |
| `pipeline_agent.py`              | Orchestrates the end‑to‑end run (ingest → analysis → forecast → advisory).     |
| `prompt_templates.py`            | Houses deterministic, multi‑style prompt generators + JSON validators.         |
| `config.example.json`            | Configuration template (API keys, model names, prompt styles, source control). |
| `bitcoin_prediction_agent.ipynb` | (Optional) Experimental / exploratory notebook for agent logic.                |

---

## 3. Data Structures & Contracts

### 3.1 `NewsArticle` (internal dataclass)

```
{
  title: str,
  url: str,
  content: str,          # truncated to ≤ ~5000 chars
  source: str,           # domain or feed host
  published_date: str|None ("YYYY-MM-DD" or raw feed value)
  summary: str,          # (reserved – not currently filled per-item)
  impact_type: str       # 'short_term' | 'long_term' | 'both'
}
```

### 3.2 Summarization JSON (output of summarization model / fallback)

```
{
  summary: str,                # multi‑paragraph narrative (style‑conditioned)
  daily_summary: str,          # concise 3–5 sentence daily condensation
  sentiment: "bullish"|"bearish"|"neutral"|"mixed",
  market_impact: "high"|"medium"|"low"|"unknown",
  key_events: [str,...],
  risk_factors: [str,...],
  watch_items: [str,...],
  opportunities: [str,...],
  bullish_ratio: float (0–1),
  bearish_ratio: float (0–1),
  neutral_ratio: float (≈ 1 - bullish - bearish),
  high_impact_count: int,
  confidence: float (0–1),
  recommendation: "BUY"|"SELL"|"HOLD"|"ACCUMULATE"|"REDUCE",
  recommendation_confidence: float (0–1)
}
```

Validation & normalization performed by `validate_summary_payload()`.

### 3.3 Aggregated News Analysis

```
{
  date: "YYYY-MM-DD",
  short_term_count: int,
  long_term_count: int,
  total_news_items: int,
  short_term_summary: str,
  long_term_summary: str,
  bullish_ratio: float,
  bearish_ratio: float,
  neutral_ratio: float,
  high_impact_count: int,
  avg_confidence: float,
  key_events: [str,...],
  daily_view: {
     summary: str,
     sentiment: str,
     market_impact: str,
     key_risks: [str,...],
     watch_items: [str,...],
     recommendation_short_term: { action: str, probability: float },
     recommendation_long_term: { action: str, probability: float }
  }
}
```

### 3.4 Effects Analysis (expected from your custom model – mock if unavailable)

```
{
  bull_prob: float,
  bear_prob: float,
  base_prob: float,
  scenarios: { bullish: float, bearish: float, base: float }
  # (merged into news_analysis dict)
}
```

### 3.5 Price Context & Forecast

```
{
  price_history_60d: [float * 60],
  current_price: float,
  next_10_day_prices: [float * 10]
}
```

If forecast endpoint fails, a trend‑biased random walk fallback is produced.

### 3.6 Structured Advisory Blueprint (Stage 1 – JSON)

```
{
  "executive_view": str (≤ ~220 words),
  "core_thesis": ["driver", ...],
  "scenario_matrix": {
     "bullish": {"prob": float, "drivers": [str], "target": float, "invalidations": [str]},
     "base":    {"prob": float, "drivers": [str], "target": float},
     "bearish": {"prob": float, "drivers": [str], "target": float, "stress_points": [str] }
  },
  "risk_register": [{"risk": str, "prob": float, "impact": "low|med|high", "mitigation": str}],
  "tactical_positions": [{"horizon_days": int, "bias": "long|short|neutral", "entry": float, "tp": float, "sl": float, "r_multiple": float}],
  "portfolio_guidance": {"allocation_pct": {"core": float, "tactical": float, "hedge": float}, "leverage_suggestion": str},
  "confidence_score": float
}
```

### 3.7 Final Advisory Narrative (Stage 2 – Markdown)

Sections:

1. Executive Summary
2. Market Structure & Thesis
3. Scenario Framework (formatted table‑like markdown)
4. Tactical Positioning & Trade Plan
5. Risk Register & Mitigations
6. Portfolio Allocation Rationale
7. Catalysts & Monitoring Plan
8. Actionable Checklist

---

## 4. Execution Flow (Normal Path)

1. Collect news feeds (RSS) → parse → basic dedupe.
2. Scrape article bodies via `newspaper3k` then fallback `BeautifulSoup` selection.
3. Heuristic bucket into short vs long term (keyword scoring).
4. Summarize each bucket (Gemini model or fallback mock) using style‐specific prompt.
5. Merge summaries + compute aggregate metrics.
6. Augment with price history (Yahoo Finance) & generate preliminary forecast (model or fallback).
7. Call effects model (if configured) → scenario probabilities.
8. (Optional structured advisory stage) JSON blueprint → narrative expansion.
9. Persist JSON results + advisory text under `output_dir`.

---

## 5. Degradation / Fallback Logic

| Stage                     | Failure Condition   | Fallback Behavior                                  |
| ------------------------- | ------------------- | -------------------------------------------------- |
| News scrape               | network/parse error | Record placeholder content string                  |
| Summarization LLM         | API error / no key  | Deterministic mock JSON ratios per impact type     |
| Effects model             | HTTP error          | Insert static probability template                 |
| Forecast model            | HTTP error          | Trend‑biased random walk using bull vs bear spread |
| Advisory structured stage | JSON parse failure  | Revert to legacy single prompt advisory            |
| Legacy advisory           | OpenAI missing      | Simple summary variant                             |

---

## 6. Configuration (`config.example.json`)

| Key                        | Type  | Description                                            |
| -------------------------- | ----- | ------------------------------------------------------ |
| api_keys.gemini            | [str] | Rotated Gemini API keys (round‑robin).                 |
| api_keys.openai            | str   | OpenAI key for advisory (and optionally future steps). |
| api_keys.openai_base_url   | str   | Override base URL (Azure/OpenAI proxy).                |
| models.summarization_model | str   | Gemini model name for summarization.                   |
| models.effects_model       | str   | HTTP endpoint (POST JSON) for effects analysis.        |
| models.forecast_model      | str   | HTTP endpoint for forecast.                            |
| models.advisory_model      | str   | Chat completion model for advisory.                    |
| news.max_articles          | int   | Upper bound pre‑dedupe.                                |
| news.short_term_count      | int   | Max items in short bucket.                             |
| news.long_term_count       | int   | Max items in long bucket.                              |
| news.sources               | [str] | RSS feed URLs.                                         |
| output_dir                 | str   | Destination folder for results.                        |
| prompt_style               | str   | Summarization style (see below).                       |
| advisory_style             | str   | Advisory style variant (in narrative stage).           |
| use_structured_advisory    | bool  | Enable two‑phase advisory (JSON → Markdown).           |

---

## 7. Prompt Styles (Summarization)

| Style              | Emphasis                                         |
| ------------------ | ------------------------------------------------ |
| comprehensive      | Balanced qualitative depth, multi‑paragraph.     |
| concise            | Fast scan; compressed narrative + bullet arrays. |
| quant_risk         | Vol, liquidity, order flow, statistical framing. |
| institutional_deep | Structural adoption, regulation, capital flows.  |
| trader_flash       | 24–72h catalysts, invalidation levels.           |

`prompt_style` applies to bucket summaries; `advisory_style` influences structured + narrative tone.

---

## 8. Logging & Observability

- Standard library `logging` writes to both stdout and `bitcoin_agent.log`.
- Key lifecycle checkpoints: initialization, feed parse counts, summarization, effects, forecast, advisory generation, persistence.
- Errors downgraded to warnings where safe; fatal errors mark pipeline result `status = failed`.

---

## 9. Running the Pipeline

### 9.1 Environment Setup (macOS / zsh)

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add any missing libs (if you enable extra model SDKs):

```
pip install openai google-generativeai newspaper3k yfinance
```

### 9.2 Execute

```
python agents/pipeline_agent.py --config agents/config.json --date 2025-09-12
```

If `--date` omitted, today’s date is used.

### 9.3 Output Artifacts

```
<output_dir>/bitcoin_advisory_<DATE>.json
```

Top-level JSON includes: counts, aggregated news analysis, probabilities, forecast array, advisory text.

---

## 10. Integrating Your Trained Models

You can replace placeholder endpoints with:

1. Local microservices (Flask/FastAPI) returning JSON.
2. Direct in‑process inference (import your model, run forward pass) — patch `_call_trained_model` or branch logic.
3. Hosted inference (Hugging Face Inference Endpoints, custom cloud function, etc.).

### 10.1 Effects Model Payload (recommended minimal contract)

```
POST /effects
{ "news_analysis": { ...aggregated fields... } }
→ { "bull_prob": 0.62, "bear_prob": 0.22, "base_prob": 0.16, "scenarios": {"bullish":0.62,"bearish":0.22,"base":0.16} }
```

### 10.2 Forecast Model Payload

```
POST /forecast
{ "analysis_data": { merged news + effects + price_history_60d } }
→ { "next_10_day_prices": [float * 10] }
```

---

## 11. Validation & Safety Nets

- JSON resilience: attempt bracket slicing recovery if LLM returns extra tokens.
- Ratio normalization: bullish + bearish + neutral rebased to 1.0.
- Bounds clamping for all probability / confidence fields.
- Advisory structured stage disabled automatically if parsing fails.

---

## 12. Extensibility Roadmap (Suggestions)

| Enhancement                                      | Rationale                                    |
| ------------------------------------------------ | -------------------------------------------- |
| Vector store dedupe (semantic)                   | Reduce repetitive news influence.            |
| Sentiment calibration vs historical realized vol | Improve signal reliability.                  |
| Regime detection (trend / chop / panic)          | Adjust forecast volatility parameter.        |
| On‑chain metrics integration                     | Augment fundamental drivers.                 |
| Strategy backtest harness                        | Validate advisory directional accuracy.      |
| Risk budget allocator                            | Map confidence to position sizing framework. |

---

## 13. Security & Key Management

- Never hard‑code API keys; load via environment or external config file not committed to VCS.
- Consider `.env` + a loader if scaling.
- Rotate Gemini keys when receiving quota errors (round‑robin already implemented).

---

## 14. Performance Considerations

| Area          | Optimization                                                 |
| ------------- | ------------------------------------------------------------ |
| Scraping      | ThreadPoolExecutor (5 workers) – tune based on latency.      |
| Summarization | Batch prompts if moving to local model.                      |
| Forecast      | Cache price history if running multiple intraday updates.    |
| Logging       | Downgrade to WARNING for noisy transient errors once stable. |

---

## 15. Known Limitations

- Heuristic bucketing (keyword score) — upgradeable to classifier (few‑shot or lightweight model).
- Effects & forecast default to mock profiles if endpoints absent (not predictive).
- No persistence layer beyond JSON files (add DB for historical analytics).
- No formal evaluation harness included (metrics like directional hit ratio, regret, calibration pending).

---

## 16. Minimal Code Touch Points for Customization

| Function                       | Purpose                      | Customize When                                    |
| ------------------------------ | ---------------------------- | ------------------------------------------------- |
| `collect_news`                 | Gather + scrape articles     | Add new sources / rate limiting                   |
| `bucket_news`                  | Assign impact types          | Replace heuristic with ML classifier              |
| `summarize_items`              | Per‑bucket LLM summarization | Swap model / style routing                        |
| `_call_summarization_model`    | Gemini + fallback            | Insert local inference                            |
| `analyze_effects`              | Scenario probabilities       | Plug in your directional model                    |
| `forecast_next_10_days`        | Price path                   | Replace with time‑series model (e.g. transformer) |
| `generate_investment_advisory` | Final narrative              | Add structured stage variants                     |

---

## 17. Quick Test (Dry Run Without Real Models)

Run with no API keys: pipeline will produce deterministic mock summaries & advisory (useful for UI integration smoke tests).

---

## 18. Example Result Snippet (Truncated)

```
{
  "status": "completed",
  "news_analysis": { "bullish_ratio": 0.56, ... },
  "effects_analysis": { "bull_prob": 0.65, ... },
  "forecast": [54321.1, 54510.4, ...],
  "advisory": "# Bitcoin Investment Advisory for ..."
}
```

---

## 19. Failure Handling Cheat Sheet

| Symptom                 | Likely Cause                | Fix                                       |
| ----------------------- | --------------------------- | ----------------------------------------- |
| Empty summaries         | Gemini key invalid          | Update `api_keys.gemini` list             |
| All mock outputs        | No external keys configured | Add keys & restart                        |
| Forecast flat / linear  | Endpoint missing            | Implement forecast endpoint               |
| Advisory short & simple | Structured mode failed      | Inspect logs; JSON parse error            |
| Crash on import         | Missing deps                | Reinstall requirements; add optional libs |

---

## 20. License & Attribution

This pipeline is designed for educational / research augmentation. Verify regulatory and compliance requirements before using outputs for live trading decisions.

---

## 21. Fast Start Checklist

- [ ] Create `agents/config.json` from example & insert keys
- [ ] Verify dependencies installed
- [ ] Run pipeline once (`--date` optional)
- [ ] Inspect produced JSON + advisory
- [ ] Plug in effects + forecast endpoints
- [ ] Enable structured advisory (`use_structured_advisory=true`)
- [ ] Iterate on prompt styles & calibration

---

### Contact / Extension Notes

Add internal notes or links here for teammates (e.g., model endpoint docs, credential vault references).

---

**End of Documentation**
