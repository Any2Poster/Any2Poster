# Generates panel visual assets, runs quality checks, and saves images for compile.

from __future__ import annotations

import time
import uuid
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from any2poster.models import (
    DesignSystem,
    GeneratedPanel,
    GenerationResult,
    QualityReport,
    VisualSuggestion,
    VisualType,
)
from any2poster.quality import (
    evaluate_visual_quality,
    derive_prompt_tweak,
)
from any2poster.prompts import (
    build_visual_prompt,
    get_design_system,
)

if TYPE_CHECKING:
    from any2poster.models import (
        AnalyzedContent,
        PanelCategory,
        PanelSpec,
        PosterConfig,
        PosterPlan,
    )
    from any2poster.providers.openrouter import OpenRouterProvider


def run_generate_stage(
    plan: "PosterPlan",
    analyzed: "AnalyzedContent",
    config: "PosterConfig",
    provider: "OpenRouterProvider",
    output_dir: Path,
    figures_dir: Path | None = None,
) -> GenerationResult:
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)
    design = plan.design or get_design_system(config.style)

    generated: list[GeneratedPanel] = []
    failed: list[str] = []
    quality_reports: list[QualityReport] = []

    if not config.generate_visuals:
        return GenerationResult(
            panels=generated,
            total_generated=0,
            failed_panels=[],
        )

    methodology_summary = ""
    content_domain = ""
    if analyzed.global_analysis:
        methodology_summary = analyzed.global_analysis.methodology_summary or ""
        content_domain = analyzed.global_analysis.paper_domain or ""

    content_panels = [p for p in plan.panels if p.panel_type != "title"]

    for panel in content_panels:
        # Generate visuals for ALL panels — "original" panels get a generated
        # fallback in case the extracted figure file isn't found on disk.
        visual_suggestion = _resolve_visual_suggestion(panel, analyzed)
        if not visual_suggestion:
            continue

        visual_id = f"{panel.id}_visual"
        visual_path = output_dir / f"{visual_id}.png"

        if visual_path.exists() and config.resume:
            generated.append(
                GeneratedPanel(
                    panel_id=visual_id,
                    image_path=visual_path,
                    prompt_used="(resumed)",
                    generation_attempts=0,
                )
            )
            continue

        debug_payload_path = None
        if config.debug:
            debug_dir = output_dir / "debug_requests"
            debug_payload_path = debug_dir / f"{visual_id}.request.json"

        try:
            prompt = build_visual_prompt(
                concept=visual_suggestion.concept,
                description=visual_suggestion.description,
                visual_type=visual_suggestion.visual_type.value,
                data_points=visual_suggestion.data_points,
                design=design,
                methodology_summary=methodology_summary,
                content_domain=content_domain,
            )
            # Append a unique token so identical-concept panels get distinct
            # API responses instead of a cached/repeated image from the model.
            prompt += f"\n\n[panel:{panel.id} uid:{uuid.uuid4().hex[:8]}]"

            max_quality_retries = max(0, int(getattr(config, "quality_check_retries", 0)))
            total_attempts = 0

            while True:
                total_attempts += 1
                attempt_debug = debug_payload_path
                if attempt_debug and total_attempts > 1:
                    attempt_debug = attempt_debug.with_name(
                        attempt_debug.stem + f".retry{total_attempts}" + attempt_debug.suffix
                    )

                _generate_with_retry(
                    provider=provider,
                    prompt=prompt,
                    output_path=visual_path,
                    reference_image=None,
                    aspect_ratio="4:3",
                    max_retries=config.max_retries,
                    debug_payload_path=attempt_debug,
                )

                if not getattr(config, "quality_check", True):
                    break

                report = evaluate_visual_quality(
                    panel_id=visual_id,
                    visual=visual_suggestion,
                    image_path=visual_path,
                    design=design,
                    provider=provider,
                    vision_model=getattr(config, "vision_model", ""),
                )
                quality_reports.append(report)

                if "quality_check_error" in report.issues:
                    logger.warning("Visual QA skipped for %s: %s", visual_id, report.summary)
                    break

                if report.passed:
                    break

                if total_attempts >= 1 + max_quality_retries:
                    logger.warning("Visual QA failed for %s: %s", visual_id, report.summary)
                    break

                tweak = derive_prompt_tweak(report)
                prompt = tweak.apply(prompt)

            generated.append(
                GeneratedPanel(
                    panel_id=visual_id,
                    image_path=visual_path,
                    prompt_used=prompt[:500],
                    generation_attempts=total_attempts,
                )
            )

        except Exception as exc:
            failed.append(visual_id)
            _create_fallback_visual(visual_suggestion, design, visual_path)
            generated.append(
                GeneratedPanel(
                    panel_id=visual_id,
                    image_path=visual_path,
                    prompt_used=f"FALLBACK: {exc}",
                    generation_attempts=config.max_retries,
                )
            )

    # ── Auto-logo generation ──────────────────────────────────
    if getattr(config, "auto_logos", False):
        _generate_auto_logos(
            analyzed=analyzed,
            config=config,
            provider=provider,
            output_dir=output_dir,
            design=design,
        )

    return GenerationResult(
        panels=generated,
        total_generated=len(generated),
        failed_panels=failed,
        quality_reports=quality_reports,
    )


def _resolve_visual_suggestion(
    panel: "PanelSpec",
    analyzed: "AnalyzedContent",
) -> "VisualSuggestion | None":
    """Return the VisualSuggestion for this panel, auto-building one if needed."""
    if panel.visual_suggestion:
        return panel.visual_suggestion

    # Auto-build from section info if the section has no suggestion yet
    section_map = {s.section_id: s for s in analyzed.sections}
    section = section_map.get(panel.section_id)
    if not section:
        return None

    return VisualSuggestion(
        concept=panel.title or section.title,
        description=(
            section.bullets[0][:90]
            if section.bullets
            else f"Diagram for {panel.title}"
        ),
        visual_type=_visual_type_for_category(panel.panel_category),
        data_points=[],
    )


def _visual_type_for_category(category: object) -> VisualType:
    from any2poster.models import PanelCategory as PC
    if category == PC.METHODOLOGY:
        return VisualType.PIPELINE
    if category == PC.ARCHITECTURE:
        return VisualType.ARCHITECTURE_DIAGRAM
    if category == PC.RESULTS:
        return VisualType.COMPARISON
    if category == PC.DATASET:
        return VisualType.BAR_CHART
    if category == PC.ANALYSIS:
        return VisualType.LINE_CHART
    return VisualType.CONCEPT_DIAGRAM


def _generate_with_retry(
    provider: "OpenRouterProvider",
    prompt: str,
    output_path: Path,
    reference_image: "Path | None",
    aspect_ratio: str,
    max_retries: int,
    debug_payload_path: "Path | None" = None,
) -> Path:
    last_error = None
    for attempt in range(max_retries):
        try:
            return provider.generate_image(
                prompt=prompt,
                output_path=output_path,
                reference_image=reference_image,
                aspect_ratio=aspect_ratio,
                debug_payload_path=debug_payload_path,
            )
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))

    raise RuntimeError(
        f"Image generation failed after {max_retries} attempts: {last_error}"
    )


def _create_fallback_visual(
    vs: "VisualSuggestion",
    design: DesignSystem,
    output_path: Path,
) -> None:
    """Create a simple placeholder visual using PIL when generation fails."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = 1400, 1050
    img = Image.new("RGB", (w, h), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 54)
        body_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 36)
    except (OSError, IOError):
        try:
            title_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 54
            )
            body_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36
            )
        except (OSError, IOError):
            title_font = ImageFont.load_default()
            body_font = ImageFont.load_default()

    border = design.tokens.border
    header = design.tokens.header_bar
    text_color = design.tokens.text_primary

    draw.rounded_rectangle([20, 20, w - 20, h - 20], radius=20, outline=border, width=3)
    draw.rounded_rectangle([44, 44, w - 44, 180], radius=14, fill=header)

    title = (vs.concept or "Visual Summary").strip()
    if len(title) > 42:
        title = f"{title[:39]}..."
    draw.text((74, 94), title, fill=design.tokens.text_on_primary, font=title_font)

    # Simple 3-box pipeline placeholder
    top = 260
    box_h = 190
    gap = 40
    box_w = (w - 160 - 2 * gap) // 3
    labels = ["Input", "Process", "Output"]
    for i in range(3):
        left = 80 + i * (box_w + gap)
        right = left + box_w
        bottom = top + box_h
        fill_color = "#F0F4F8" if i % 2 == 0 else "#E8F0F8"
        draw.rounded_rectangle(
            [left, top, right, bottom], radius=14, fill=fill_color, outline=border, width=2
        )
        draw.text((left + 24, top + 24), labels[i], fill=text_color, font=body_font)
        if i < 2:
            ax, ay = right + 8, top + box_h // 2
            bx = ax + gap - 16
            draw.line([ax, ay, bx, ay], fill=design.tokens.accent, width=6)
            draw.polygon(
                [(bx, ay), (bx - 16, ay - 10), (bx - 16, ay + 10)],
                fill=design.tokens.accent,
            )

    desc = (vs.description or "").strip()
    if not desc:
        desc = f"Auto-generated {vs.visual_type.value.replace('_', ' ')}"
    if len(desc) > 78:
        desc = f"{desc[:75]}..."
    draw.text((80, 530), desc, fill=text_color, font=body_font)

    y_pos = 620
    for point in (vs.data_points or [])[:4]:
        line = point.strip()
        if not line:
            continue
        if len(line) > 86:
            line = f"{line[:83]}..."
        draw.text((90, y_pos), f"\u2022 {line}", fill=text_color, font=body_font)
        y_pos += 58

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), quality=95)


def _generate_auto_logos(
    analyzed: "AnalyzedContent",
    config: "PosterConfig",
    provider: "OpenRouterProvider",
    output_dir: Path,
    design: DesignSystem,
) -> None:
    """Generate venue and institution logos when --auto-logos is set.

    Writes logo_left.png and logo_right.png into the output dir
    and patches config.logo_left / config.logo_right so the compile
    stage picks them up.
    """
    venue = ""
    institution = ""
    if analyzed and analyzed.global_analysis:
        venue = (analyzed.global_analysis.venue or "").strip()
        institution = (analyzed.global_analysis.affiliation or "").strip()

    # Skip if user already provided manual logos
    if config.logo_left and config.logo_right:
        return

    is_dark = design.is_dark
    # Determine the banner background so logos blend seamlessly
    white_banner = getattr(design, "white_banner", False)
    banner_bg = "#FFFFFF" if white_banner else design.tokens.primary_dark
    banner_is_light = _perceived_brightness_hex(banner_bg) > 0.5

    # Generate venue logo (left)
    if venue and not config.logo_left:
        venue_path = output_dir / "logo_left.png"
        if not (venue_path.exists() and config.resume):
            _logo_prompt = (
                f"Generate the official logo/emblem for the academic conference '{venue}'. "
                f"CRITICAL: The background MUST be pure solid white (#FFFFFF). "
                f"No off-white, no gray, no gradients — exactly pure white so it blends "
                f"seamlessly when placed on a white banner. "
                f"Use dark/colored elements for the logo itself (NOT white on white). "
                f"Show the conference acronym '{venue.split()[0]}' prominently. "
                f"Professional academic conference logo — simple, clean, geometric, "
                f"recognizable at small sizes. No photographic elements, no shadows, no 3D effects. "
                f"The logo should be compact and centered with generous white margin around it."
            )
            try:
                _generate_with_retry(
                    provider=provider,
                    prompt=_logo_prompt,
                    output_path=venue_path,
                    reference_image=None,
                    aspect_ratio="1:1",
                    max_retries=2,
                )
                config.logo_left = str(venue_path)
            except Exception:
                pass  # Non-critical — poster works without logos

    # Generate institution logo (right)
    if institution and not config.logo_right:
        inst_path = output_dir / "logo_right.png"
        if not (inst_path.exists() and config.resume):
            _logo_prompt = (
                f"Generate the official logo/emblem for the academic institution '{institution}'. "
                f"CRITICAL: The background MUST be pure solid white (#FFFFFF). "
                f"No off-white, no gray, no gradients — exactly pure white so it blends "
                f"seamlessly when placed on a white banner. "
                f"Use dark/colored elements for the logo itself (NOT white on white). "
                f"Show the institution's well-known emblem, crest, or wordmark. "
                f"Professional academic institution logo — clean, authoritative, "
                f"recognizable at small sizes. No photographic elements, no shadows, no 3D effects. "
                f"The logo should be compact and centered with generous white margin around it."
            )
            try:
                _generate_with_retry(
                    provider=provider,
                    prompt=_logo_prompt,
                    output_path=inst_path,
                    reference_image=None,
                    aspect_ratio="1:1",
                    max_retries=2,
                )
                config.logo_right = str(inst_path)
            except Exception:
                pass  # Non-critical


def _perceived_brightness_hex(hex_color: str) -> float:
    """Return perceived brightness in [0,1] for a hex color."""
    h = hex_color.lstrip("#")
    if len(h) < 6:
        h = h.ljust(6, "0")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
