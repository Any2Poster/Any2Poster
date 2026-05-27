# Validates and enriches chart data so generated visuals stay grounded in source text.

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from any2poster.models import VisualSuggestion, VisualType

if TYPE_CHECKING:
    from any2poster.models import AnalyzedContent, AnalyzedSection, ParsedDocument
    from any2poster.providers.openrouter import OpenRouterProvider


_DATA_VISUAL_TYPES = {
    VisualType.BAR_CHART,
    VisualType.LINE_CHART,
    VisualType.TABLE,
    VisualType.COMPARISON,
}

_EXTRACTION_SYSTEM = (
    "You extract exact numerical data from research text for chart generation. "
    "You ONLY return numbers and labels that appear explicitly in the text. "
    "If a value is not stated in the text, you say 'NOT_FOUND'. "
    "You MUST respond with valid JSON only."
)

_EXTRACTION_USER = """Extract exact data points for this visualization.

VISUALIZATION:
- Concept: {concept}
- Type: {visual_type}
- Description: {description}
- Suggested data points: {suggested_points}

SOURCE TEXT (the data MUST come from here):
{source_text}

Respond with this exact JSON:
{{
  "validated_data_points": [
    "<label>: <exact value from text>"
  ],
  "data_found": <true if at least 2 valid data points found>,
  "source_quotes": [
    "<exact quote from source text containing each number>"
  ]
}}

Rules:
- ONLY include numbers that appear literally in the source text
- Include units (%, x, ms, etc.) exactly as stated
- If the source says "up to 70x faster", the data point is "Compiled: up to 70x"
- If a suggested data point has no support in the text, exclude it
- Return at least 2 data points or set data_found to false"""


def validate_visual_data(
    analyzed: "AnalyzedContent",
    parsed: "ParsedDocument",
    provider: "OpenRouterProvider",
) -> "AnalyzedContent":
    if not analyzed.global_analysis or not analyzed.global_analysis.visual_suggestions:
        return analyzed

    section_text_map = {s.section_id: s.content for s in parsed.sections}
    analyzed_section_map = {s.section_id: s for s in analyzed.sections}

    updated_visuals: list[VisualSuggestion] = []

    for vs in analyzed.global_analysis.visual_suggestions:
        if vs.visual_type not in _DATA_VISUAL_TYPES:
            updated_visuals.append(vs)
            continue

        source_text = _gather_relevant_text(vs, analyzed_section_map, section_text_map)

        if not source_text or len(source_text.split()) < 20:
            updated_visuals.append(vs)
            continue

        validated = _extract_and_validate(vs, source_text, provider)
        updated_visuals.append(validated)

    analyzed.global_analysis.visual_suggestions = updated_visuals

    for section in analyzed.sections:
        if section.visual_suggestion and section.visual_suggestion.visual_type in _DATA_VISUAL_TYPES:
            src = section_text_map.get(section.section_id, "")
            if src and len(src.split()) >= 20:
                section.visual_suggestion = _extract_and_validate(
                    section.visual_suggestion, src, provider
                )

    return analyzed


def _gather_relevant_text(
    vs: VisualSuggestion,
    analyzed_sections: dict[str, "AnalyzedSection"],
    section_text_map: dict[str, str],
) -> str:
    parts: list[str] = []

    if vs.target_panel_id:
        for sid, section in analyzed_sections.items():
            if section.section_id == vs.target_panel_id:
                text = section_text_map.get(sid, "")
                if text:
                    parts.append(text[:3000])
                break

    if not parts:
        keywords = vs.concept.lower().split()
        for sid, text in section_text_map.items():
            text_lower = text.lower()
            if any(kw in text_lower for kw in keywords):
                parts.append(text[:2000])
                if len(parts) >= 2:
                    break

    if not parts:
        for sid in list(section_text_map.keys())[:3]:
            parts.append(section_text_map[sid][:1500])

    return "\n\n".join(parts)[:5000]


def _extract_and_validate(
    vs: VisualSuggestion,
    source_text: str,
    provider: "OpenRouterProvider",
) -> VisualSuggestion:
    suggested_str = "\n".join(f"- {dp}" for dp in vs.data_points) if vs.data_points else "None provided"

    user_prompt = _EXTRACTION_USER.format(
        concept=vs.concept,
        visual_type=vs.visual_type.value,
        description=vs.description,
        suggested_points=suggested_str,
        source_text=source_text[:4000],
    )

    try:
        data = provider.complete_json(
            system=_EXTRACTION_SYSTEM,
            user=user_prompt,
            temperature=0.1,
            max_tokens=1024,
        )
    except Exception:
        return vs

    if data.get("data_found", False):
        validated_points = data.get("validated_data_points", [])
        if validated_points and len(validated_points) >= 2:
            return VisualSuggestion(
                concept=vs.concept,
                description=vs.description,
                visual_type=vs.visual_type,
                data_points=validated_points,
                target_panel_id=vs.target_panel_id,
            )

    return vs


def enrich_visual_prompts_with_data(
    analyzed: "AnalyzedContent",
) -> "AnalyzedContent":
    headline = ""
    if analyzed.global_analysis:
        headline = analyzed.global_analysis.headline_result

    if not headline:
        return analyzed

    for section in analyzed.sections:
        if not section.visual_suggestion:
            continue
        vs = section.visual_suggestion
        if vs.visual_type in _DATA_VISUAL_TYPES and not vs.data_points:
            if headline and any(
                kw in vs.concept.lower()
                for kw in ["result", "speed", "accuracy", "performance", "comparison"]
            ):
                vs.data_points = [f"Key result: {headline}"]

    return analyzed