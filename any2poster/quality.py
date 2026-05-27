# Runs VLM quality checks and prompt tweaks for generated visuals.

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from any2poster.models import QualityReport

if TYPE_CHECKING:
    from pathlib import Path
    from any2poster.models import DesignSystem, VisualSuggestion
    from any2poster.providers.openrouter import OpenRouterProvider


QUALITY_SYSTEM = (
    "You are a strict QA reviewer for academic poster visuals. "
    "You assess if the image is publication-quality, readable, and faithful "
    "to the requested concept and data. You MUST respond with valid JSON only."
)

QUALITY_USER = """Evaluate the provided visual.

Context:
- Concept: {concept}
- Type: {visual_type}
- Description: {description}
- Expected data points: {data_points}
- Visual style/mood: {mood}

Return this exact JSON:
{{
  "passed": <true|false>,
  "confidence": <0.0-1.0>,
  "summary": "<short diagnostic summary>",
  "issues": ["<issue 1>", "<issue 2>"],
  "suggested_fixes": ["<fix 1>", "<fix 2>"]
}}

Fail the image if:
- It is blank or mostly empty
- Text is illegible or too small
- Labels are cropped or cut off
- The diagram type is wrong for the request
- Important labels/values are missing or incorrect
- There is severe misalignment or unreadable layout
"""


def evaluate_visual_quality(
    *,
    panel_id: str,
    visual: "VisualSuggestion",
    image_path: "Path",
    design: "DesignSystem",
    provider: "OpenRouterProvider",
    vision_model: str = "",
) -> QualityReport:
    data_points = visual.data_points or []
    data_str = ", ".join(data_points) if data_points else "None"
    mood = ", ".join((design.mood_keywords or [])[:3]) or "professional"

    user_prompt = QUALITY_USER.format(
        concept=visual.concept or "",
        visual_type=visual.visual_type.value,
        description=visual.description or "",
        data_points=data_str,
        mood=mood,
    )

    try:
        data = provider.complete_vision_json(
            system=QUALITY_SYSTEM,
            user=user_prompt,
            image_path=image_path,
            temperature=0.1,
            max_tokens=512,
            model_override=vision_model or None,
        )
    except Exception as exc:
        return QualityReport(
            panel_id=panel_id,
            passed=True,
            confidence=0.0,
            summary=f"Quality check skipped: {exc}",
            issues=["quality_check_error"],
            suggested_fixes=[],
        )

    return QualityReport(
        panel_id=panel_id,
        passed=bool(data.get("passed", True)),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        summary=str(data.get("summary", "")),
        issues=list(data.get("issues", []) or []),
        suggested_fixes=list(data.get("suggested_fixes", []) or []),
    )


class PromptTweak(BaseModel):
    """Simple wrapper to keep prompt adjustments consistent."""
    additions: list[str] = Field(default_factory=list)

    def apply(self, prompt: str) -> str:
        if not self.additions:
            return prompt
        extra = "\n\nQUALITY FIXES:\n" + "\n".join(f"- {a}" for a in self.additions)
        return prompt + extra


def derive_prompt_tweak(report: QualityReport) -> PromptTweak:
    additions: list[str] = []
    if report.issues:
        additions.append("Fix these issues: " + "; ".join(report.issues))
    if report.suggested_fixes:
        additions.append("Apply these fixes: " + "; ".join(report.suggested_fixes))
    additions.append("Ensure all labels are legible and not cropped.")
    additions.append("Avoid large blank regions; fill the canvas with the diagram.")
    return PromptTweak(additions=additions)
