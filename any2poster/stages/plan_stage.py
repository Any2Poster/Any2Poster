"""Stage 3: PLAN

Layout engine for both professional and casual poster modes.

Professional (default):
  - Full-width title banner (10% of poster height)
  - 3 equal-width columns, panels assigned left-to-right
  - Panels with visuals/figures receive 1.5x height allocation
  - Enforced min/max panel height, gutter spacing

Casual:
  - Taller title banner (13% of poster height)
  - 2 equal-width columns, max 6 panels (fewer, larger)
  - Panels with visuals get even more height (1.6x)
  - More generous gutter spacing for breathing room
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from any2poster.models import (
    PanelCategory,
    PanelPosition,
    PanelSpec,
    PosterPlan,
)
from any2poster.prompts import resolve_theme

if TYPE_CHECKING:
    from any2poster.models import AnalyzedContent, PosterConfig

# ── Professional mode constants ──────────────────────────────
_TITLE_HEIGHT_RATIO = 0.10
_FOOTER_HEIGHT_RATIO = 0.020
_OUTER_MARGIN = 0.12
_GUTTER = 0.40
_NUM_COLUMNS = 3
_FIGURE_WEIGHT = 1.3
_VISUAL_WEIGHT = 1.2
_TEXT_ONLY_WEIGHT = 0.90
_MIN_PANEL_HEIGHT_RATIO = 0.10
_MAX_PANEL_HEIGHT_RATIO = 0.48

# ── Casual mode constants ────────────────────────────────────
_CASUAL_TITLE_HEIGHT_RATIO = 0.13
_CASUAL_GUTTER = 0.28
_CASUAL_OUTER_MARGIN = 0.20
_CASUAL_NUM_COLUMNS = 2
_CASUAL_MAX_PANELS = 7
_CASUAL_FIGURE_WEIGHT = 1.5
_CASUAL_VISUAL_WEIGHT = 1.4
_CASUAL_TEXT_ONLY_WEIGHT = 0.85
_CASUAL_MIN_PANEL_HEIGHT_RATIO = 0.12
_CASUAL_MAX_PANEL_HEIGHT_RATIO = 0.52


def run_plan_stage(
    analyzed: "AnalyzedContent",
    config: "PosterConfig",
    provider: object = None,
) -> PosterPlan:
    width = config.poster_width
    height = config.poster_height
    is_casual = config.poster_mode == "casual"

    suggested = ""
    paper_domain = ""
    if analyzed and analyzed.global_analysis:
        suggested = analyzed.global_analysis.suggested_color_theme or ""
        paper_domain = analyzed.global_analysis.paper_domain or ""
    design = resolve_theme(config, suggested_style=suggested, paper_domain=paper_domain)
    style_name = design.name

    title_ratio = _CASUAL_TITLE_HEIGHT_RATIO if is_casual else _TITLE_HEIGHT_RATIO
    title_panel = _create_title_panel(analyzed, width, height, title_ratio)
    content_panels = _create_content_panels(analyzed, is_casual)
    _ensure_unique_panel_titles(content_panels)

    if not content_panels:
        return PosterPlan(
            panels=[title_panel],
            width_inches=width,
            height_inches=height,
            design=design,
            style=style_name,
        )

    # Casual: limit to fewer, larger panels
    if is_casual and len(content_panels) > _CASUAL_MAX_PANELS:
        # Keep the most important panels, then restore reading order
        indexed = list(enumerate(content_panels))
        indexed.sort(key=lambda x: x[1].weight, reverse=True)
        top = indexed[:_CASUAL_MAX_PANELS]
        top.sort(key=lambda x: x[0])  # restore original order
        content_panels = [p for _, p in top]

    num_panels = len(content_panels)
    if is_casual:
        num_cols = _CASUAL_NUM_COLUMNS if num_panels > 1 else 1
    elif num_panels <= 2:
        num_cols = num_panels
    else:
        num_cols = _NUM_COLUMNS

    columns = _distribute_to_columns(content_panels, num_cols)

    gutter = _CASUAL_GUTTER if is_casual else _GUTTER
    outer_margin = _CASUAL_OUTER_MARGIN if is_casual else _OUTER_MARGIN
    footer_ratio = _FOOTER_HEIGHT_RATIO if getattr(config, "show_footer", False) else 0.0
    min_ratio = _CASUAL_MIN_PANEL_HEIGHT_RATIO if is_casual else _MIN_PANEL_HEIGHT_RATIO
    max_ratio = _CASUAL_MAX_PANEL_HEIGHT_RATIO if is_casual else _MAX_PANEL_HEIGHT_RATIO

    _assign_positions(
        columns, width, height,
        title_ratio=title_ratio,
        footer_ratio=footer_ratio,
        gutter=gutter,
        outer_margin=outer_margin,
        min_panel_ratio=min_ratio,
        max_panel_ratio=max_ratio,
        is_professional=not is_casual,
    )

    all_panels = [title_panel]
    for col in columns:
        all_panels.extend(col)

    return PosterPlan(
        panels=all_panels,
        width_inches=width,
        height_inches=height,
        num_columns=num_cols,
        style=style_name,
        design=design,
    )


def _create_title_panel(
    analyzed: "AnalyzedContent",
    width: float,
    height: float,
    title_ratio: float = _TITLE_HEIGHT_RATIO,
) -> PanelSpec:
    return PanelSpec(
        id="title",
        panel_type="title",
        panel_category=PanelCategory.TITLE,
        title=analyzed.poster_title,
        position=PanelPosition(
            x=0,
            y=0,
            width=width,
            height=height * title_ratio,
        ),
        weight=1.0,
    )


def _create_content_panels(
    analyzed: "AnalyzedContent",
    is_casual: bool = False,
) -> list[PanelSpec]:
    fig_w = _CASUAL_FIGURE_WEIGHT if is_casual else _FIGURE_WEIGHT
    vis_w = _CASUAL_VISUAL_WEIGHT if is_casual else _VISUAL_WEIGHT
    txt_w = _CASUAL_TEXT_ONLY_WEIGHT if is_casual else _TEXT_ONLY_WEIGHT

    panels: list[PanelSpec] = []
    for i, section in enumerate(analyzed.sections):
        weight = section.importance
        has_visual = section.visual_suggestion is not None
        has_figure = section.has_figure

        # Text-only categories: no generated diagrams — let them be text-rich panels
        # BUT always prefer original figures even for text-heavy categories
        _text_only_categories = {PanelCategory.MOTIVATION, PanelCategory.CONCLUSION}
        ct = getattr(section, "content_type", "bullets")

        if has_figure:
            # Always use original figures — they are the paper's own visuals
            weight *= fig_w
            visual_source = "original"
        elif has_visual and section.panel_category not in _text_only_categories:
            weight *= vis_w
            visual_source = "generated"
        else:
            weight *= txt_w
            visual_source = "none"

        # Small random perturbation so identical papers produce varied layouts
        weight *= random.uniform(0.95, 1.05)

        # Determine layout_mode from content_type + visual_source
        if ct == "prose" and visual_source == "none":
            layout_mode = "prose_only"
        elif ct == "prose" and visual_source in ("original", "generated"):
            layout_mode = "prose_with_figure"
        elif ct == "mixed" and visual_source == "original" and section.panel_category in {PanelCategory.RESULTS, PanelCategory.ANALYSIS}:
            layout_mode = "figure_dominant"
        elif ct == "mixed" and visual_source in ("original", "generated") and section.panel_category in {PanelCategory.METHODOLOGY, PanelCategory.ARCHITECTURE}:
            layout_mode = "side_by_side"
        elif visual_source == "none":
            layout_mode = "bullets_only"
        else:
            layout_mode = "standard"

        panels.append(
            PanelSpec(
                id=f"panel_{i + 1}",
                panel_type="content",
                panel_category=section.panel_category,
                title=section.title,
                section_id=section.section_id,
                has_figure=has_figure,
                figure_ids=section.figure_ids,
                visual_suggestion=section.visual_suggestion,
                weight=weight,
                visual_source=visual_source,
                layout_mode=layout_mode,
            )
        )
    return panels


_TITLE_BASE = {
    PanelCategory.MOTIVATION: "Motivation & Problem",
    PanelCategory.METHODOLOGY: "Method Overview",
    PanelCategory.ARCHITECTURE: "Architecture Overview",
    PanelCategory.RESULTS: "Key Results",
    PanelCategory.ANALYSIS: "Analysis & Insights",
    PanelCategory.DATASET: "Dataset & Setup",
    PanelCategory.CONCLUSION: "Conclusion & Takeaways",
    PanelCategory.CONTENT: "Overview",
}

_TITLE_ALTS = {
    PanelCategory.MOTIVATION: [
        "Research Gap",
        "Problem Setting",
        "Why It Matters",
        "Challenge Overview",
    ],
    PanelCategory.METHODOLOGY: [
        "Approach",
        "Method Pipeline",
        "Algorithm Steps",
        "Method Details",
    ],
    PanelCategory.ARCHITECTURE: [
        "Model Design",
        "System Architecture",
        "Architecture Details",
        "Design Highlights",
    ],
    PanelCategory.RESULTS: [
        "Experimental Results",
        "Performance Highlights",
        "Ablation Highlights",
        "Result Summary",
    ],
    PanelCategory.ANALYSIS: [
        "Ablation Analysis",
        "Insights & Discussion",
        "Qualitative Analysis",
        "Analysis Summary",
    ],
    PanelCategory.DATASET: [
        "Data & Benchmarks",
        "Experimental Setup",
        "Datasets",
        "Setup Details",
    ],
    PanelCategory.CONCLUSION: [
        "Takeaways",
        "Implications",
        "Limitations & Future Work",
        "Closing Summary",
    ],
    PanelCategory.CONTENT: [
        "Background",
        "Key Concepts",
        "Technical Details",
        "Additional Context",
    ],
}


def _ensure_unique_panel_titles(panels: list[PanelSpec]) -> None:
    """Ensure panel titles are unique and meaningful (no '(cont.)')."""
    used: set[str] = set()
    for panel in panels:
        title = (panel.title or "").strip()
        if not title:
            title = _TITLE_BASE.get(panel.panel_category, "Overview")
        norm = title.lower()
        if norm not in used:
            panel.title = title
            used.add(norm)
            continue

        candidates = _TITLE_ALTS.get(panel.panel_category, [])
        new_title = ""
        for cand in candidates:
            if cand.lower() not in used:
                new_title = cand
                break

        if not new_title:
            base = _TITLE_BASE.get(panel.panel_category, "Overview")
            if base.lower() not in used:
                new_title = base
            else:
                for suffix in ("Insights", "Details", "Summary", "Highlights"):
                    cand = f"{base} {suffix}"
                    if cand.lower() not in used:
                        new_title = cand
                        break

        if not new_title:
            base = _TITLE_BASE.get(panel.panel_category, "Overview")
            idx = 2
            while True:
                cand = f"{base} {idx}"
                if cand.lower() not in used:
                    new_title = cand
                    break
                idx += 1

        panel.title = new_title
        used.add(new_title.lower())


def _distribute_to_columns(
    panels: list[PanelSpec], num_columns: int = _NUM_COLUMNS
) -> list[list[PanelSpec]]:
    columns: list[list[PanelSpec]] = [[] for _ in range(num_columns)]
    column_weights = [0.0] * num_columns

    for i, panel in enumerate(panels):
        if i < num_columns:
            col_idx = i
        else:
            # Content-aware column preference
            preferred = None
            if num_columns >= 3:
                if panel.layout_mode in ("figure_dominant", "side_by_side"):
                    preferred = 1  # center column for figure-heavy panels
                elif panel.panel_category in (PanelCategory.RESULTS, PanelCategory.ANALYSIS):
                    preferred = num_columns - 1  # rightmost for results

            if preferred is not None and column_weights[preferred] <= min(column_weights) * 1.30:
                col_idx = preferred
            else:
                col_idx = column_weights.index(min(column_weights))

        columns[col_idx].append(panel)
        column_weights[col_idx] += panel.weight
        panel.column = col_idx
        panel.row_in_column = len(columns[col_idx]) - 1

    return columns


def _assign_positions(
    columns: list[list[PanelSpec]],
    poster_width: float,
    poster_height: float,
    *,
    title_ratio: float = _TITLE_HEIGHT_RATIO,
    footer_ratio: float = _FOOTER_HEIGHT_RATIO,
    gutter: float = _GUTTER,
    outer_margin: float = _OUTER_MARGIN,
    min_panel_ratio: float = _MIN_PANEL_HEIGHT_RATIO,
    max_panel_ratio: float = _MAX_PANEL_HEIGHT_RATIO,
    is_professional: bool = True,
) -> None:
    num_cols = len(columns)
    content_top = poster_height * title_ratio + gutter
    # Extra 0.15in bottom breathing room so last panel doesn't touch edge
    content_bottom = poster_height - (poster_height * footer_ratio) - outer_margin - 0.15
    content_height = content_bottom - content_top
    usable_width = poster_width - (2 * outer_margin)
    col_width = (usable_width - (gutter * (num_cols - 1))) / num_cols

    min_h = content_height * min_panel_ratio
    max_h = content_height * max_panel_ratio

    for col_idx, col_panels in enumerate(columns):
        if not col_panels:
            continue

        col_x = outer_margin + col_idx * (col_width + gutter)
        total_weight = sum(p.weight for p in col_panels)
        available_height = content_height - (gutter * max(0, len(col_panels) - 1))

        if total_weight == 0:
            total_weight = 1.0

        raw_heights = [(p.weight / total_weight) * available_height for p in col_panels]

        # Clamp only to absolute min/max — let weights drive organic height variation
        clamped_heights = [max(min_h, min(max_h, h)) for h in raw_heights]
        total_clamped = sum(clamped_heights)
        if total_clamped > 0:
            scale = available_height / total_clamped
            final_heights = [h * scale for h in clamped_heights]
        else:
            even = available_height / len(col_panels)
            final_heights = [even] * len(col_panels)

        current_y = content_top
        for panel, panel_height in zip(col_panels, final_heights):
            panel.position = PanelPosition(
                x=col_x,
                y=current_y,
                width=col_width,
                height=panel_height,
            )
            current_y += panel_height + gutter
