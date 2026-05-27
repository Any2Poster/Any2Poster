"""VLM-as-Judge evaluation (Paper2Poster Appendix F.3 aligned).

Scores a poster image on six criteria:
  - Element Quality
  - Layout Balance
  - Engagement
  - Clarity
  - Content Completeness
  - Logical Flow
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class CriterionPrompt:
    key: str
    name: str
    system: str
    rubric: str


_PROMPTS: list[CriterionPrompt] = [
    CriterionPrompt(
        key="element_quality",
        name="Element Quality",
        system=(
            "You are an extremely discerning visual-element judge. Scrutinize every figure, chart, "
            "and image for any visual or stylistic issue. Always look for even subtle flaws: low contrast, "
            "imperfect resolutions, slightly inconsistent styles, crowded or mislabeled legends, etc. "
            "Be wary of awarding high scores unless the visuals truly meet the strictest standards."
        ),
        rubric=(
            "Element Quality. This criterion evaluates the visual clarity, resolution, and stylistic "
            "consistency of individual graphic elements (figures, charts, icons).\n"
            "Instructions: Five-Point Scale\n"
            "1 Point:\n"
            "- Graphics are blurry, pixelated, or illegible.\n"
            "- Color choices severely hinder interpretation.\n"
            "- Visuals may significantly detract from comprehension.\n"
            "2 Points:\n"
            "- At least one graphic is clear, while others suffer from poor resolution or style.\n"
            "- Legends or labels are missing or too small to read comfortably.\n"
            "- Color schemes create some confusion or difficulty.\n"
            "3 Points:\n"
            "- Most graphics are legible and relevant, but have notable issues with consistency, sizing, or clarity.\n"
            "- Some mismatches in style or color usage detract from cohesion.\n"
            "- Minor but noticeable labeling/legend shortcomings.\n"
            "4 Points:\n"
            "- High-quality graphics with generally consistent styling.\n"
            "- Clear legends and color schemes aid interpretation.\n"
            "- Any remaining flaws are slight and do not significantly hinder understanding.\n"
            "5 Points:\n"
            "- Rarely awarded; strictly reserved for publication-grade visuals.\n"
            "- Crisp resolution with no instances of blurriness.\n"
            "- Harmonious color palette, impeccable labeling, and an exceptionally consistent style.\n"
            "Example Output: {\"reason\": \"...\", \"score\": 4}\n"
            "Think step by step and be conservative with your rating."
        ),
    ),
    CriterionPrompt(
        key="layout_balance",
        name="Layout Balance",
        system=(
            "You are an uncompromising poster-layout judge. Critique the overall arrangement of all visual "
            "components (text blocks, headings, figures, white-space, alignment) that affect readability. "
            "Always scan for subtle alignment issues, uneven spacing, or any layout feature that might disrupt "
            "reader comprehension. Resist giving high scores unless the layout is exceptionally polished."
        ),
        rubric=(
            "Layout Balance. This criterion assesses the overall arrangement, alignment, and spacing of text and "
            "graphics to ensure a coherent and readable poster structure.\n"
            "Instructions: Five-Point Scale\n"
            "1 Point:\n"
            "- Highly disorganized layout; elements overlap, making text or graphics illegible.\n"
            "- Margins are violated or reading path is nearly impossible to follow.\n"
            "- Severely hinders comprehension.\n"
            "2 Points:\n"
            "- Some semblance of structure (columns/rows) but marred by inconsistent alignment or overcrowded sections.\n"
            "- White-space distribution may be haphazard or insufficient.\n"
            "- Reading flow is interrupted, though one can still piece it together.\n"
            "3 Points:\n"
            "- Recognizable structure with mostly consistent alignment and spacing.\n"
            "- Some minor layout distractions remain (e.g., slightly cramped text, uneven spacing, small alignment slips).\n"
            "- Generally readable but not particularly polished.\n"
            "4 Points:\n"
            "- Well-organized grid or arrangement; logical reading path that mostly flows.\n"
            "- Appropriate font sizes, spacing, and alignment; only subtle layout imperfections.\n"
            "- White-space usage clean and deliberate; nearly professional.\n"
            "5 Points:\n"
            "- Very rarely granted; must be a pristine, professional-grade layout.\n"
            "- Seamless alignment, balanced spacing, and expertly guided reading path.\n"
            "- Flawless design synergy that maximizes readability and comprehension.\n"
            "Example Output: {\"reason\": \"...\", \"score\": 4}\n"
            "Think step by step and be tough on small alignment/spacing issues."
        ),
    ),
    CriterionPrompt(
        key="engagement",
        name="Engagement",
        system=(
            "You are an uncompromising poster-aesthetics judge focusing on engagement. Be extremely critical of color "
            "harmony, typography, visual balance, and the poster's ability to grab and hold attention. Always look for "
            "subtle issues-color clashes, overly busy or dull designs, inappropriate font choices, awkward spacing, or "
            "anything that might reduce engagement. Reserve high scores for truly exemplary work."
        ),
        rubric=(
            "Engagement. This criterion judges how effectively the poster's design elements-color, typography, and "
            "composition-capture and sustain viewer attention.\n"
            "Instructions: Five-Point Scale\n"
            "1 Point:\n"
            "- Visually off-putting; clashing colors or crowded design repel viewers.\n"
            "- Typography choice is jarring or illegible at a glance.\n"
            "- Overall fails to engage or entice.\n"
            "2 Points:\n"
            "- Some visually appealing elements exist but are overshadowed by dull or inconsistent design moments.\n"
            "- Font sizes or styles reduce accessibility or attractiveness.\n"
            "- Limited capacity to draw an audience's focus.\n"
            "3 Points:\n"
            "- Shows generally pleasing color scheme and typography, though lacking a \"wow\" factor.\n"
            "- Balance and visual flow are acceptable but reveal minor weaknesses (e.g., slightly crowded or sparse areas).\n"
            "- Engagement is average; neither strong nor particularly weak.\n"
            "4 Points:\n"
            "- Eye-catching design using mostly harmonious colors and effective typography.\n"
            "- Good use of negative space; the layout guides the viewer's eye effectively.\n"
            "- Only minor flaws or bland spots prevent it from being top-tier.\n"
            "5 Points:\n"
            "- Rarely awarded-reserved for truly striking, magazine-cover-caliber visuals.\n"
            "- Flawless color palette and typography; everything works together seamlessly.\n"
            "- Immediately captivating design that retains audience interest without any noticeable weakness.\n"
            "Example Output: {\"reason\": \"...\", \"score\": 4}\n"
            "Think step by step and be very conservative when scoring."
        ),
    ),
    CriterionPrompt(
        key="clarity",
        name="Clarity",
        system=(
            "You are an uncompromising micro-text judge. Critically evaluate sentence-level clarity, grammar, phrasing, "
            "and intra-section coherence. Look for even subtle grammatical slips, confusing jargon, or clumsy phrasing. "
            "Be slow to award top marks unless the text is impeccably polished."
        ),
        rubric=(
            "Clarity. This criterion evaluates sentence-level readability, grammar, and phrasing to ensure the text is "
            "polished and error-free.\n"
            "Instructions: Five-Point Scale\n"
            "1 Point:\n"
            "- Rampant grammatical or spelling errors; sentences may be unreadable.\n"
            "- Overly technical jargon without explanations; fragments or run-ons predominate.\n"
            "- Overall, text quality severely impedes understanding.\n"
            "2 Points:\n"
            "- Meaning is generally discernible, but multiple grammar or syntax problems appear in each section.\n"
            "- Awkward or unclear phrasing disrupts the flow of reading.\n"
            "- Only partial clarity is achieved.\n"
            "3 Points:\n"
            "- Overall readable text with a few noticeable grammar or wording missteps.\n"
            "- Occasional awkward phrasing or redundancies appear, but readers can follow without major confusion.\n"
            "- Average clarity.\n"
            "4 Points:\n"
            "- Well-written, mostly free of grammatical or spelling errors.\n"
            "- Terminology is used properly; text flows smoothly within paragraphs.\n"
            "- Minor slip-ups can be present but do not disrupt understanding.\n"
            "5 Points:\n"
            "- Exceptional text quality, error-free, and elegantly phrased.\n"
            "- Complex ideas conveyed with clear, concise language.\n"
            "- Granted only if absolutely no grammatical, spelling, or stylistic flaws are detected.\n"
            "Example Output: {\"reason\": \"...\", \"score\": 4}\n"
            "Think step by step."
        ),
    ),
    CriterionPrompt(
        key="content_completeness",
        name="Content Completeness",
        system=(
            "You are an uncompromising content-depth judge. Assess whether the poster includes all essential sections "
            "and whether each section presents sufficient detail. Look for any missing or under-developed segments; "
            "do not hesitate to penalize for insufficient depth. Award the highest scores only if the poster expertly "
            "covers every necessary aspect."
        ),
        rubric=(
            "Content Completeness. This criterion measures whether all key sections are included and richly detailed, "
            "reflecting comprehensive coverage of the paper's main contributions.\n"
            "Instructions: Five-Point Scale\n"
            "1 Point:\n"
            "- Critical sections (e.g., objectives or results) are completely missing or trivial.\n"
            "- Data grossly insufficient to comprehend the study or conclusions.\n"
            "- Very poor depth that fails to convey essential information.\n"
            "2 Points:\n"
            "- Most key sections appear but major details (context, data, references) are absent.\n"
            "- Lack of elaboration on methods or results leaves big gaps.\n"
            "- Overall content too shallow to properly inform.\n"
            "3 Points:\n"
            "- All standard sections included with fundamental information.\n"
            "- Some omissions or scant detail in certain areas (e.g., results or methodology).\n"
            "- Only moderate depth; the reader must fill many gaps themselves.\n"
            "4 Points:\n"
            "- All essential sections present, each treated with adequate-to-strong detail.\n"
            "- Robust description of objectives, methods, results, and references.\n"
            "- Only minor improvements needed.\n"
            "5 Points:\n"
            "- Very rarely granted; everything must be comprehensive and thorough.\n"
            "- Exhaustive detail on methodology, results (with statistics), interpretation, references, and future work.\n"
            "- Leaves readers with minimal unanswered questions.\n"
            "Example Output: {\"reason\": \"...\", \"score\": 4}\n"
            "Think step by step."
        ),
    ),
    CriterionPrompt(
        key="logical_flow",
        name="Logical Flow",
        system=(
            "You are an uncompromising macro-logic judge. Examine how well the poster's major sections (Introduction, "
            "Methods, Results, Conclusions, etc.) connect to form a coherent narrative. Pay attention to continuity, "
            "how logically each section flows from the previous, and whether there are any abrupt gaps. Only award the "
            "highest marks if the storyline is perfectly seamless."
        ),
        rubric=(
            "Logical Flow. This criterion examines the coherence and progression of ideas across poster sections, "
            "ensuring a seamless narrative from introduction to conclusion.\n"
            "Instructions: Five-Point Scale\n"
            "1 Point:\n"
            "- Sections are disjointed; little to no logical connection between them.\n"
            "- Key transitions or the central rationale is missing, creating confusion.\n"
            "2 Points:\n"
            "- General sequence recognizable but important logical steps are weak or missing.\n"
            "- Readers must infer key links.\n"
            "3 Points:\n"
            "- Mostly coherent narrative with minor gaps.\n"
            "- Transitions exist but some logical steps are lightly justified.\n"
            "4 Points:\n"
            "- Well-structured storyline; each section clearly builds on the previous.\n"
            "- Transitions are stated, rationale is mostly strong.\n"
            "5 Points:\n"
            "- Extremely rare; flawless logical flow from introduction to conclusion.\n"
            "- Seamless transitions; no inferential leaps.\n"
            "Example Output: {\"reason\": \"...\", \"score\": 4}\n"
            "Think step by step and penalize any noticeable logical gap or awkward transition."
        ),
    ),
]


def _score_from_response(resp: dict[str, Any]) -> tuple[int | None, str]:
    if not isinstance(resp, dict):
        return None, ""
    score = resp.get("score")
    reason = str(resp.get("reason", "")).strip()
    try:
        score_int = int(score)
    except Exception:
        score_int = None
    if score_int is not None:
        score_int = max(1, min(5, score_int))
    return score_int, reason


def evaluate_vlm_judge(
    *,
    poster_image: Path,
    provider,
    model_name: str | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Run VLM-as-Judge evaluation and return a result dict."""
    results: dict[str, Any] = {}

    for prompt in _PROMPTS:
        user = (
            f"{prompt.rubric}\n\n"
            "Return JSON exactly as: {\"reason\": \"...\", \"score\": <1-5>}"
        )
        resp = provider.complete_vision_json(
            system=prompt.system,
            user=user,
            image_path=poster_image,
            temperature=0.0,
            max_tokens=512,
            model_override=model_name or None,
        )
        score, reason = _score_from_response(resp)
        results[prompt.key] = {
            "name": prompt.name,
            "score": score,
            "reason": reason,
        }

    # Aggregates
    def _mean(keys: list[str]) -> float | None:
        vals = [results[k]["score"] for k in keys if results.get(k, {}).get("score") is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    results["aesthetic_score"] = _mean(["element_quality", "layout_balance", "engagement"])
    results["information_score"] = _mean(["clarity", "content_completeness", "logical_flow"])

    all_scores = [
        results[k]["score"]
        for k in ("element_quality", "layout_balance", "engagement", "clarity", "content_completeness", "logical_flow")
        if results.get(k, {}).get("score") is not None
    ]
    results["overall"] = float(sum(all_scores) / len(all_scores)) if all_scores else None

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


def _cli() -> None:
    import argparse
    from any2poster.providers.openrouter import OpenRouterProvider

    parser = argparse.ArgumentParser(description="any2poster VLM-as-Judge evaluator (Paper2Poster-style).")
    parser.add_argument("--poster-image", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--paper-id", default="")
    parser.add_argument("--method-name", default="any2poster")
    parser.add_argument("--out-root", default="eval_results")
    args = parser.parse_args()

    provider = OpenRouterProvider()
    model = args.model.strip() or None

    if args.out:
        out_path = Path(args.out)
    elif args.paper_id:
        out_path = Path(args.out_root) / args.paper_id / args.method_name / "vlm_judge.json"
    else:
        out_path = None

    evaluate_vlm_judge(
        poster_image=Path(args.poster_image),
        provider=provider,
        model_name=model,
        out_path=out_path,
    )


if __name__ == "__main__":
    _cli()

