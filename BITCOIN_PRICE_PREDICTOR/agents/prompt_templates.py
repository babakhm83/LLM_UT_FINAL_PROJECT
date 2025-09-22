"""
Backward-compatible prompt templates entrypoint.

This module keeps the original public API (`summarization_prompt`,
`advisory_json_prompt`, `advisory_narrative_prompt`, `validate_summary_payload`)
but sources the actual prompt content from `model_prompts.py`. Splitting the
large prompt text into `model_prompts.py` makes maintenance easier and keeps
`prompt_templates.py` stable for external imports.
"""

from __future__ import annotations

# Re-export the concrete prompt implementations and constants from model_prompts
from .model_prompts import (
    SUMMARY_JSON_SPEC,
    RECOMMENDATION_ENUM,
    SENTIMENT_ENUM,
    IMPACT_ENUM,
    _common_summary_instructions,
    summarization_prompt,
    advisory_json_prompt,
    advisory_narrative_prompt,
    validate_summary_payload,
)

__all__ = [
    "summarization_prompt",
    "advisory_json_prompt",
    "advisory_narrative_prompt",
    "validate_summary_payload",
]
