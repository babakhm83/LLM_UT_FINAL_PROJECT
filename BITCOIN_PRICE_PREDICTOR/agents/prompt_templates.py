"""
Prompt Templates for Bitcoin Pipeline Agent

Provides standardized, multi-style prompt templates for summarization and advisory generation.

Styles:
- comprehensive: Full institutional-style analysis
- concise: Short, actionable summary
- quant_risk: Emphasis on quantitative metrics and risk decomposition
- institutional_deep: Extended multi-layer thesis & portfolio integration
- trader_flash: Ultra-short tactical view for intraday/short horizon
"""
from __future__ import annotations
import json
from typing import Dict, Any, List

SUMMARY_JSON_SPEC = {
    "required_float_fields": [
        "bullish_ratio", "bearish_ratio", "neutral_ratio", "confidence", "recommendation_confidence"
    ],
    "required_int_fields": ["high_impact_count"],
    "required_string_fields": [
        "summary", "daily_summary", "sentiment", "market_impact", "recommendation"
    ],
    "required_list_fields": [
        "key_events", "risk_factors", "watch_items", "opportunities"
    ]
}

RECOMMENDATION_ENUM = ["BUY", "SELL", "HOLD", "ACCUMULATE", "REDUCE"]
SENTIMENT_ENUM = ["bullish", "bearish", "neutral", "mixed"]
IMPACT_ENUM = ["high", "medium", "low", "unknown"]


def _common_summary_instructions(impact_type: str) -> str:
    return f"""Return ONLY valid JSON with all required keys. Never wrap in markdown fences. All float values in [0,1]. Ensure bullish_ratio+bearish_ratio+neutral_ratio≈1 (±0.02). Use lowercase for sentiment, one of {SENTIMENT_ENUM}. market_impact in {IMPACT_ENUM}. recommendation in {RECOMMENDATION_ENUM}. If insufficient data, lower confidence but fill fields.
Focus strictly on Bitcoin {impact_type} price impacts (exclude unrelated macro unless directly linked). Avoid placeholder text like 'N/A'; provide best-effort content.
""".strip()


def summarization_prompt(news_items: List[Dict[str, Any]], analysis_date: str, impact_type: str, style: str = "comprehensive") -> str:
    style_note = {
        "comprehensive": "Depth 5 paragraphs; balanced qualitative + structured drivers.",
        "concise": "Max 180 words summary, bulletify key arrays.",
        "quant_risk": "Highlight statistical drivers, volatility regimes, liquidity, order flow proxies.",
        "institutional_deep": "Add structural adoption, regulatory trajectory, capital flows, derivative positioning in narrative layers.",
        "trader_flash": "Max 120 words, emphasize catalysts next 24-72h, directional bias & invalidation level."
    }.get(style, "Depth 4-5 paragraphs; balanced analysis.")

    return f"""You are a professional crypto macro analyst. Analyze the following Bitcoin-related news articles for {analysis_date} focusing on {impact_type} impact.
STYLE_DIRECTIVE: {style_note}
ARTICLES_JSON: {json.dumps(news_items, ensure_ascii=False, indent=2)[:18000]}

Produce JSON with keys: summary,daily_summary,sentiment,market_impact,key_events,risk_factors,watch_items,opportunities,bullish_ratio,bearish_ratio,neutral_ratio,high_impact_count,confidence,recommendation,recommendation_confidence.
{_common_summary_instructions(impact_type)}"""


def advisory_json_prompt(analysis_bundle: Dict[str, Any], style: str) -> str:
    """First-stage advisory: produce structured JSON before narrative."""
    return f"""You are an elite Bitcoin portfolio strategist. Given the ANALYSIS_BUNDLE JSON below, synthesize a structured advisory blueprint.
STYLE: {style}
ANALYSIS_BUNDLE: {json.dumps(analysis_bundle, ensure_ascii=False)[:30000]}

Return ONLY JSON with keys:
{{
  "executive_view": "<=220 words distilled institutional summary",
  "core_thesis": ["3-6 bullet core drivers"],
  "scenario_matrix": {{
      "bullish": {{"prob": 0.xx, "drivers": ["..."], "target": <price>, "invalidations": ["..."]}},
      "base": {{"prob": 0.xx, "drivers": ["..."], "target": <price>}},
      "bearish": {{"prob": 0.xx, "drivers": ["..."], "target": <price>, "stress_points": ["..."]}}
  }},
  "risk_register": [{{"risk": "...", "prob": 0.xx, "impact": "low|med|high", "mitigation": "..."}}],
  "tactical_positions": [{{"horizon_days": 5, "bias": "long|short|neutral", "entry": <float>, "tp": <float>, "sl": <float>, "r_multiple": 2.5}}],
  "portfolio_guidance": {{"allocation_pct": {{"core": 0.xx, "tactical": 0.xx, "hedge": 0.xx}}, "leverage_suggestion": "None|Moderate|High"}},
  "confidence_score": 0.xx
}}
Constraints:
- Sum of scenario probs ~1.0 ±0.05
- allocation_pct sums ~1.0 ±0.05
- No empty arrays.
- Use realistic price targets consistent with bundle current_price.
- Return ONLY JSON.
"""


def advisory_narrative_prompt(structured_json: Dict[str, Any], full_analysis: Dict[str, Any], style: str) -> str:
    return f"""You are a senior digital asset strategist. Convert the STRUCTURED_ADVISORY JSON plus RAW_ANALYSIS into a polished institutional narrative.
STYLE={style}
STRUCTURED_ADVISORY_JSON={json.dumps(structured_json, ensure_ascii=False)}
RAW_ANALYSIS={json.dumps(full_analysis, ensure_ascii=False)[:18000]}
Sections (use markdown headers):
1. Executive Summary
2. Market Structure & Thesis
3. Scenario Framework (table-like markdown)
4. Tactical Positioning & Trade Plan
5. Risk Register & Mitigations
6. Portfolio Allocation Rationale
7. Catalysts & Monitoring Plan (next 10/30 days)
8. Actionable Checklist
Keep it precise, avoid hype, cite data points from inputs. Do NOT invent on-chain metrics not provided.
Return plain markdown narrative only (no JSON)."""


def validate_summary_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight structural validator & auto-fixer for summarization output."""
    def clamp(v, lo=0.0, hi=1.0):
        try:
            return max(lo, min(hi, float(v)))
        except Exception:
            return 0.0
    for f in SUMMARY_JSON_SPEC["required_float_fields"]:
        obj[f] = clamp(obj.get(f, 0.0))
    for f in SUMMARY_JSON_SPEC["required_int_fields"]:
        try:
            obj[f] = int(obj.get(f, 0))
        except Exception:
            obj[f] = 0
    for f in SUMMARY_JSON_SPEC["required_string_fields"]:
        obj[f] = str(obj.get(f, ""))[:8000]
    for f in SUMMARY_JSON_SPEC["required_list_fields"]:
        val = obj.get(f, [])
        if not isinstance(val, list):
            val = [str(val)]
        obj[f] = [str(x)[:400] for x in val][:8]
    total = obj["bullish_ratio"] + obj["bearish_ratio"] + obj["neutral_ratio"]
    if total > 0:
        obj["bullish_ratio"] /= total
        obj["bearish_ratio"] /= total
        obj["neutral_ratio"] = 1 - obj["bullish_ratio"] - obj["bearish_ratio"]
    if obj["recommendation"] not in RECOMMENDATION_ENUM:
        obj["recommendation"] = "HOLD"
    if obj["sentiment"] not in SENTIMENT_ENUM:
        obj["sentiment"] = "neutral"
    if obj["market_impact"] not in IMPACT_ENUM:
        obj["market_impact"] = "medium"
    return obj

__all__ = [
    "summarization_prompt",
    "advisory_json_prompt",
    "advisory_narrative_prompt",
    "validate_summary_payload"
]
