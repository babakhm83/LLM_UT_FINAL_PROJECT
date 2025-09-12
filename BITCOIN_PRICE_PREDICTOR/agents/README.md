# Bitcoin News → Effects → Training → Forecast → Advisory Agent

This agent orchestrates the full daily pipeline you described using your three trained model roles:

- News summarizer model (10 short-term + 10 long-term items)
- Effects analysis model (structured JSON: sentiment, direction, strength, timeframe, confidence, reason)
- 10-day price forecast model
- Final daily advisory generator (prompt style aligned to your notebooks)

The default implementation includes stubbed inference calls; replace them with your trained model endpoints or local inference code.

## Features

- Crawl Bitcoin news via public RSS feeds (no API keys required)
- Bucket into short-term (recent) and long-term (diverse) items
- Summarize items with your news model prompt style
- Convert summaries to your training-sample schema
- Run effects-analysis model to produce structured JSON
- Aggregate and run 10-day forecast model
- Produce a daily advisory for users

## Quickstart

1. Install deps (macOS, zsh)

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r "Files/Final Project/requirements.txt"
```

2. Run the pipeline once:

```
python "Files/Final Project/agents/pipeline_agent.py"
```

You should see a JSON payload containing:

- short_summaries, long_summaries
- short_effects, long_effects (JSON strings in `output`)
- forecast
- advisory

## Wire your trained models

Open `agents/pipeline_agent.py` and replace the stubs with your model calls:

- summarize_items: call your news summarization model
- analyze_effects: call your effects-analysis model (return output as JSON string)
- forecast_next_10_days: call your 10-day forecast model
- generate_daily_advisory: call your advisor model/prompt style

You can expose your models via local HTTP endpoints, Python SDKs, or libraries you used in the notebooks. Keep the input/output contract.

## Prompt style for the news model

The summarizer should produce concise summaries tailored for downstream effects analysis. Consider using the same style as in your training notebooks: focus on

- What happened
- Why it matters for BTC
- Timeframe hint (immediate/short/medium)
- Confidence

## New Prompt & Advisory System

Added modular prompt templates in `prompt_templates.py` with selectable styles:

- comprehensive (default)
- concise
- quant_risk
- institutional_deep
- trader_flash

Config keys:

- prompt_style: controls summarization style
- advisory_style: controls structured advisory narrative depth
- use_structured_advisory: if true, performs two-stage (JSON blueprint + narrative) generation when OpenAI available

### Example Config Snippet

```json
{
  "prompt_style": "quant_risk",
  "advisory_style": "institutional_deep",
  "use_structured_advisory": true
}
```

If Gemini/OpenAI unavailable, system degrades gracefully to deterministic mock + simple advisory.

## Extending

- Add caching of crawl results (e.g., local JSON per day)
- Add retries and rate limiting for model calls
- Parallelize effects analysis for throughput
- Persist outputs to `results_YYYY-MM-DD.json` for auditing

## Troubleshooting

- If imports fail, ensure the virtual environment is active
- Replace stubs with real model calls step-by-step to isolate issues
