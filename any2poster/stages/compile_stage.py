"""Stage 5: COMPILE

Build a self-contained HTML poster and render it to PDF + PNG.

Architecture:
  - All text is real HTML/CSS (deterministic, no AI text rendering)
  - 3-column flexbox layout matches the academic poster standard
  - Typography is locked via a CSS scale ladder (pt units for print)
  - Figures and generated visuals are embedded as base64 data URIs
  - Playwright renders HTML → PDF (vector, print-quality) + PNG (raster preview)
  - WeasyPrint is used as fallback if Playwright is unavailable

Poster structure:
  ┌─────────────────── Title Banner (13%) ───────────────────┐
  │ Column 1         │ Column 2         │ Column 3            │
  │ ┌─ Panel ──────┐ │ ┌─ Panel ──────┐ │ ┌─ Panel ─────────┐│
  │ │ Header bar   │ │ │ Header bar   │ │ │ Header bar      ││
  │ │ • Bullet 1   │ │ │ • Bullet 1   │ │ │ • Bullet 1      ││
  │ │ • Bullet 2   │ │ │ • Bullet 2   │ │ │ • Bullet 2      ││
  │ │ [Visual slot]│ │ │ [Visual slot]│ │ │ [Visual slot]   ││
  │ └──────────────┘ │ └──────────────┘ │ └─────────────────┘│
  └──────────────────────────── Footer ──────────────────────┘
"""

from __future__ import annotations

import base64
import io
import html as _html_mod
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

if TYPE_CHECKING:
    from any2poster.models import (
        AnalyzedContent,
        DesignSystem,
        GenerationResult,
        PanelSpec,
        PosterConfig,
        PosterPlan,
    )

# ──────────────────────────────────────────────────────────────
# Typography scale (pt) — physical print sizes
# ──────────────────────────────────────────────────────────────
_FONT_TITLE_PT = 76          # Poster title in banner
_FONT_AUTHORS_PT = 42        # Authors line
_FONT_AFFIL_PT = 33          # Affiliation line
_FONT_HEADLINE_PT = 28       # (unused by default; reserved)
_FONT_PANEL_HDR_PT = 34      # Panel header bar text (stronger hierarchy)
_FONT_BULLET_PT = 27         # Bullet/body text (slightly smaller for contrast)
_FONT_CAPTION_PT = 19        # Figure captions
_FONT_FOOTER_PT = 14         # Footer text

# Poster geometry (fraction of poster height)
_TITLE_FRAC = 0.10
_FOOTER_FRAC = 0.02

# Spacing (inches)
_MARGIN_IN = 0.12
_GUTTER_IN = 0.40


# ══════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════

def run_compile_stage(
    plan: "PosterPlan",
    generation: "GenerationResult",
    config: "PosterConfig",
    output_path: Path,
    analyzed: "AnalyzedContent | None" = None,
) -> Path:
    """Build HTML poster, render to PDF + PNG via Playwright."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figures_dir = Path(config.checkpoint_dir) / "figures"

    html_content = _build_html_poster(
        plan=plan,
        generation=generation,
        config=config,
        analyzed=analyzed,
        figures_dir=figures_dir,
    )

    # Save the canonical HTML artifact
    html_path = output_path.with_suffix(".html")
    html_path.write_text(html_content, encoding="utf-8")

    # Render to PDF/PNG
    if output_path.suffix.lower() == ".pdf":
        pdf_path = output_path
        png_path = output_path.with_suffix(".png")
    else:
        png_path = output_path
        pdf_path = output_path.with_suffix(".pdf")

    rendered = _render_html(html_path, pdf_path, png_path, plan, config)
    return rendered


# ══════════════════════════════════════════════════════════════
# HTML builder
# ══════════════════════════════════════════════════════════════

def _build_html_poster(
    plan: "PosterPlan",
    generation: "GenerationResult",
    config: "PosterConfig",
    analyzed: "AnalyzedContent | None",
    figures_dir: Path,
) -> str:
    design = plan.design
    tokens = design.tokens

    width_in = plan.width_inches      # e.g. 48
    height_in = plan.height_inches    # e.g. 36
    is_casual = config.poster_mode == "casual"
    margin = 0.20 if is_casual else _MARGIN_IN
    if getattr(config, "column_gutter_in", 0.0) and config.column_gutter_in > 0:
        gutter = float(config.column_gutter_in)
    else:
        gutter = 0.26 if is_casual else _GUTTER_IN
    num_cols = plan.num_columns or 3

    # Build lookup maps
    panel_map = {g.panel_id: g for g in generation.panels}
    section_map = {s.section_id: s for s in analyzed.sections} if analyzed else {}

    # Poster metadata
    poster_title = (analyzed.poster_title if analyzed else "") or plan.panels[0].title if plan.panels else "Research Poster"
    poster_authors = (analyzed.poster_authors if analyzed else "") or ""
    affiliation = ""
    if analyzed and analyzed.global_analysis:
        affiliation = analyzed.global_analysis.affiliation or ""

    source_name = Path(config.input_path).stem if config.input_path else ""

    # Venue: use the LLM-extracted field first, then fall back to regex over narrative_arc
    venue_name = ""
    if analyzed and analyzed.global_analysis:
        venue_name = (analyzed.global_analysis.venue or "").strip()
        if not venue_name:
            import re as _re
            _arc = analyzed.global_analysis.narrative_arc or ""
            _m = _re.search(
                r'\b(NeurIPS|ICML|ICLR|CVPR|ICCV|ECCV|ACL|EMNLP|NAACL|AAAI|IJCAI|SIGIR|WWW|KDD|ICRA|RSS|CORL)\s*\d{4}\b',
                _arc, _re.IGNORECASE,
            )
            venue_name = _m.group(0) if _m else ""

    # Group content panels by column
    content_panels = [p for p in plan.panels if p.panel_type != "title"]
    col_groups: dict[int, list["PanelSpec"]] = {}
    for p in content_panels:
        col_groups.setdefault(p.column, []).append(p)
    for col in col_groups.values():
        col.sort(key=lambda p: p.row_in_column)

    # Normalize flex-grow weights within each column so no single panel
    # dominates: cap any panel at MAX_PANEL_FLEX_FRAC of the column total,
    # then re-normalise until stable.  This prevents a heavy figure panel
    # from pushing adjacent panels out of view.
    MAX_PANEL_FLEX_FRAC = 0.48  # no panel may take more than 48 % of column
    col_flex: dict[int, dict[str, float]] = {}  # panel_id -> clamped flex-grow
    for col_idx, panels_in_col in col_groups.items():
        if not panels_in_col:
            continue
        weights = {p.id: p.weight for p in panels_in_col}
        n = len(panels_in_col)
        max_frac = MAX_PANEL_FLEX_FRAC if n > 1 else 1.0
        # Iterative clamp: repeat until no weight exceeds the cap
        for _ in range(20):
            total = sum(weights.values())
            if total <= 0:
                break
            changed = False
            for pid, w in list(weights.items()):
                if w / total > max_frac:
                    weights[pid] = total * max_frac
                    changed = True
            if not changed:
                break
        col_flex[col_idx] = weights

    # Build each column's HTML
    cols_html = ""
    for col_idx in range(num_cols):
        panels_in_col = col_groups.get(col_idx, [])
        col_html = ""
        flex_map = col_flex.get(col_idx, {})
        for panel_spec in panels_in_col:
            flex_grow = flex_map.get(panel_spec.id, panel_spec.weight)
            col_html += _build_panel_html(
                panel_spec=panel_spec,
                section_map=section_map,
                panel_map=panel_map,
                figures_dir=figures_dir,
                config=config,
                flex_grow=flex_grow,
            )
        # Keep equal column widths; panel weights handle vertical distribution within each column
        cols_html += f'<div class="col">\n{col_html}</div>\n'

    # ── Banner HTML ──────────────────────────────────────────
    safe_title = _esc(poster_title)
    safe_authors = _esc(poster_authors)
    safe_affil = _esc(affiliation)

    # ── Logo zones ────────────────────────────────────────────
    # Embed user-supplied logo files as base64 data URLs.
    # When no logos are provided but a venue is detected, print a helpful hint.
    def _logo_img_html(path_str: str) -> str:
        if not path_str:
            return ""
        p = Path(path_str)
        data_url = _img_to_data_url(p, white_bg=False)
        if not data_url:
            return ""
        return f'<img src="{data_url}" class="banner-logo-img" alt="logo">'

    left_logo_html = _logo_img_html(config.logo_left)
    right_logo_html = _logo_img_html(config.logo_right)

    if not left_logo_html and not right_logo_html and venue_name:
        import logging as _logging
        _logging.getLogger(__name__).info(
            "Detected venue '%s'. To add logos, use: "
            "--logo-left path/to/%s_logo.png --logo-right path/to/institution_logo.png",
            venue_name, venue_name.split()[0].lower(),
        )

    left_logo_div = (
        f'<div class="banner-logo-left">{left_logo_html}</div>'
        if left_logo_html
        else '<div class="banner-logo-left" style="display:none"></div>'
    )
    right_logo_div = (
        f'<div class="banner-logo-right">{right_logo_html}</div>'
        if right_logo_html
        else '<div class="banner-logo-right" style="display:none"></div>'
    )
    affil_div = f'<div class="banner-affil">{safe_affil}</div>' if safe_affil else ""

    banner_html = f"""<div class="title-banner">
  {left_logo_div}
  <div class="banner-center">
    <div class="banner-title">{safe_title}</div>
    <div class="banner-authors">{safe_authors}</div>
    {affil_div}
  </div>
  {right_logo_div}
</div>"""

    # Footer: omit entirely unless explicitly enabled
    show_footer = getattr(config, "show_footer", False)
    if show_footer:
        footer_html = f"""<div class="footer">
  <span>{_esc(source_name)}</span>
  <span>{_esc(venue_name)}</span>
</div>"""
    else:
        footer_html = ""

    title_frac = 0.13 if is_casual else _TITLE_FRAC
    footer_frac = _FOOTER_FRAC if show_footer else 0.0

    css = _build_css(
        design=design,
        poster_mode=config.poster_mode,
        width_in=width_in,
        height_in=height_in,
        margin_in=margin,
        gutter_in=gutter,
        title_frac=title_frac,
        footer_frac=footer_frac,
    )

    mode_class = "casual" if is_casual else "professional"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width">
<title>{_esc(poster_title)}</title>
<style>
{css}
</style>
</head>
<body>
<div class="poster {mode_class}">
{banner_html}
<div class="content-grid cols-{num_cols}">
{cols_html}
</div>
{footer_html}
</div>
</body>
</html>"""


def _build_panel_html(
    panel_spec: "PanelSpec",
    section_map: dict,
    panel_map: dict,
    figures_dir: Path,
    config: "PosterConfig",
    flex_grow: float | None = None,
) -> str:
    section = section_map.get(panel_spec.section_id)
    bullets = section.bullets if section else []
    sub_headers = section.sub_headers if section else []
    lead_paragraph = section.lead_paragraph if section else ""
    content_type = section.content_type if section else "bullets"

    # Resolve visual image
    visual_data_url: str | None = None
    visual_caption = ""

    if panel_spec.visual_source == "generated":
        visual_id = f"{panel_spec.id}_visual"
        gen = panel_map.get(visual_id)
        if gen and gen.image_path and Path(gen.image_path).exists():
            visual_data_url = _img_to_data_url(Path(gen.image_path), white_bg=True)
    elif panel_spec.visual_source == "original":
        fig_path = _resolve_figure_path(panel_spec.figure_ids, figures_dir, config)
        if fig_path:
            visual_data_url = _img_to_data_url(fig_path, white_bg=True)

    if panel_spec.visual_source == "original" and not visual_data_url:
        visual_id = f"{panel_spec.id}_visual"
        gen = panel_map.get(visual_id)
        if gen and gen.image_path and Path(gen.image_path).exists():
            visual_data_url = _img_to_data_url(Path(gen.image_path), white_bg=True)

    has_visual = visual_data_url is not None

    # Use layout_mode from plan stage
    layout_mode = panel_spec.layout_mode or "standard"
    # Downgrade to text-only if visual didn't resolve
    if not has_visual and layout_mode in ("standard", "side_by_side", "prose_with_figure", "figure_dominant"):
        layout_mode = "prose_only" if content_type == "prose" else "bullets_only"

    # Text-only layout modes should never render a visual slot
    if layout_mode in ("prose_only", "bullets_only"):
        has_visual = False
        visual_data_url = None

    css_layout = f"layout-{layout_mode.replace('_', '-')}"

    # ── Build lead paragraph HTML ────────────────────────
    lead_html = ""
    if lead_paragraph and content_type in ("prose", "mixed"):
        lead_html = f'<p class="lead-para">{_render_inline_md(lead_paragraph)}</p>'

    # ── Build bullets HTML with sub-headers ──────────────
    is_casual = config.poster_mode == "casual"
    if layout_mode == "figure_dominant":
        max_bullets = 2
    elif layout_mode in ("prose_only", "prose_with_figure"):
        max_bullets = 2
    elif layout_mode in ("bullets_only",):
        max_bullets = 4 if not is_casual else 3
    else:
        max_bullets = 3 if not is_casual else 2

    def _word_count(text: str) -> int:
        return len(re.findall(r"\b\w+\b", text or ""))

    text_words = _word_count(lead_paragraph) + sum(_word_count(str(b)) for b in bullets)

    # Density-aware cap for visual panels: only reduce to 2 bullets if text is heavy.
    if has_visual and bullets and len(bullets) > 2 and text_words >= 140:
        max_bullets = min(max_bullets, 2)

    bullets_html = ""
    if bullets:
        sh_map: dict[int, str] = {}
        for sh in sub_headers:
            sh_map[sh.after_bullet_index] = sh.text

        items = ""
        shown = 0
        for i, b in enumerate(bullets):
            if shown >= max_bullets:
                break
            items += f"  <li>{_render_inline_md(str(b))}</li>\n"
            shown += 1
            if i in sh_map and shown < max_bullets:
                sh_text = _render_inline_md(sh_map[i])
                items += f'  <li class="subheader-item">{sh_text}</li>\n'

        bullets_html = f'<ul class="bullets">\n{items}</ul>'

    # ── Assemble text block ──────────────────────────────
    text_inner = lead_html + bullets_html
    text_html = ""
    if text_inner:
        wrapper_class = "text-content"
        if has_visual and layout_mode not in ("side_by_side",):
            wrapper_class = "text-content with-visual"
        text_html = f'<div class="{wrapper_class}">\n    {text_inner}\n  </div>'

    # ── Visual slot HTML ──────────────────────────────────
    visual_html = ""
    if has_visual:
        cap_html = f'<p class="caption">{_esc(visual_caption)}</p>' if visual_caption else ""
        visual_html = (
            f'<div class="visual-slot">'
            f'<img src="{visual_data_url}" alt="figure" loading="eager">'
            f'{cap_html}'
            f'</div>'
        )

    panel_title = _esc(panel_spec.title or "Section")

    # ── Side-by-side uses CSS grid: text left, figure right ──
    if layout_mode == "side_by_side" and has_visual:
        body_html = f"""<div class="panel-body-grid">
    {text_html}
    {visual_html}
  </div>"""
    else:
        body_html = f"""<div class="panel-body {css_layout}">
    {text_html}
    {visual_html}
  </div>"""

    # Scale up text in low-density panels to fill available space and improve readability.
    # Text-only panels can scale more aggressively since there's no visual competing for space.
    panel_font_scale = 1.0
    if text_words > 0:
        if has_visual:
            if text_words <= 45:
                panel_font_scale = 1.08
            elif text_words <= 70:
                panel_font_scale = 1.05
            elif text_words <= 95:
                panel_font_scale = 1.03
        else:
            # Text-only: more aggressive scaling to fill panel height
            if text_words <= 40:
                panel_font_scale = 1.18
            elif text_words <= 60:
                panel_font_scale = 1.13
            elif text_words <= 90:
                panel_font_scale = 1.08
            elif text_words <= 130:
                panel_font_scale = 1.04

    # Avoid over-scaling in very tight layouts
    if layout_mode == "figure_dominant":
        panel_font_scale = min(panel_font_scale, 1.03)
    if layout_mode == "side_by_side":
        panel_font_scale = min(panel_font_scale, 1.05)

    # For text-only panels, expand line height to fill vertical space more naturally
    panel_line_scale = 1.0
    if not has_visual and text_words > 0:
        if text_words <= 50:
            panel_line_scale = 1.30
        elif text_words <= 80:
            panel_line_scale = 1.20
        elif text_words <= 120:
            panel_line_scale = 1.10

    _flex = flex_grow if flex_grow is not None else panel_spec.weight
    return f"""<div class="panel" data-panel-id="{panel_spec.id}" style="flex-grow:{_flex:.4f}; --panel-font-scale:{panel_font_scale:.3f}; --panel-line-scale:{panel_line_scale:.3f}; --panel-text-scale:1; --panel-visual-scale:1; --panel-visual-flex:1;">
  <div class="panel-hdr"><span class="panel-title">{panel_title}</span></div>
  {body_html}
</div>
"""


# ══════════════════════════════════════════════════════════════
# CSS builder
# ══════════════════════════════════════════════════════════════

def _build_css(
    design,
    poster_mode: str,
    width_in: float,
    height_in: float,
    margin_in: float,
    gutter_in: float,
    title_frac: float,
    footer_frac: float,
) -> str:
    tokens = design.tokens
    is_dark = design.is_dark

    title_h_in = height_in * title_frac
    footer_h_in = height_in * footer_frac

    # Bullet line height
    bullet_lh = 1.50
    # Vertical spacing between panels (tighter than column gap)
    row_gap_in = 0.14 if poster_mode != "casual" else 0.12

    # ── Derive all colors from design tokens ──────────────────
    bg = tokens.background   # outer poster canvas (e.g. #EBEBEB for academic_gray)
    surface = tokens.surface  # panel card surface (e.g. #FFFFFF)

    # Compute border radius — tighter for gray/neutral themes
    panel_radius_in = 0.04 if bg.upper() in ("#EBEBEB", "#E8E8E8", "#EDEDED", "#F0F0F0") else 0.06
    hdr_bg = tokens.header_bar
    hdr_text = tokens.text_on_primary
    title_bg = tokens.primary_dark
    text_primary = tokens.text_primary
    text_secondary = tokens.text_secondary
    marker_color = tokens.primary_dark  # always dark for contrast on white panel backgrounds
    border_color = tokens.border

    # Dark vs light mode styling differences
    banner_border_bottom = "none"
    if is_dark:
        # Title banner: accent-colored title text on dark background
        banner_bg_css = title_bg
        title_text_color = tokens.accent
        author_color = f"rgba({_hex_to_css_rgb(tokens.text_primary)},0.85)"
        affil_color = f"rgba({_hex_to_css_rgb(tokens.text_primary)},0.70)"
        panel_shadow = "none"
        panel_border = f"1px solid {border_color}"
        footer_bg = tokens.surface
        footer_text = tokens.text_secondary
        bold_color = tokens.accent
        img_border = f"1px solid {border_color}"
    else:
        white_banner = getattr(design, "white_banner", False)
        if white_banner:
            # White banner: clean background, no visible border
            banner_bg_css = "#FFFFFF"
            title_text_color = tokens.primary_dark
            author_color = tokens.text_secondary
            affil_color = f"rgba({_hex_to_css_rgb(tokens.text_secondary)},0.80)"
            banner_border_bottom = "none"
        else:
            # Colored banner: auto-detect text color based on banner brightness
            banner_bg_css = title_bg
            title_text_color = _auto_text_on_bg(title_bg, light="#FFFFFF", dark="#0C0C0C")
            author_color = "rgba(255,255,255,0.95)" if title_text_color == "#FFFFFF" else "rgba(12,12,12,0.85)"
            affil_color = "rgba(255,255,255,0.90)" if title_text_color == "#FFFFFF" else "rgba(12,12,12,0.70)"
        # Gray-background themes (academic_gray etc.): border for panel depth; no shadow
        if bg.upper() in ("#EBEBEB", "#E8E8E8", "#EDEDED", "#F0F0F0", "#E5E5E5"):
            panel_shadow = "none"
            panel_border = f"1px solid {border_color}"
        else:
            panel_shadow = "0 1px 4px rgba(0,0,0,0.06)"
            panel_border = "none"
        footer_bg = _shift_color_css(bg, -8) if bg not in ("#FFFFFF", "#FAFAFA") else "#F0F0F0"
        footer_text = "#444444"
        bold_color = "inherit"
        img_border = "none"

    css = f"""
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
/* ── Reset ─────────────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
:root {{
  --font-title: "Inter", "Noto Sans", sans-serif;
  --font-heading: "Inter", "Helvetica Neue", sans-serif;
  --font-body: "Inter", "Helvetica Neue", sans-serif;
  --font-accent: "Inter", "Helvetica Neue", sans-serif;
}}
html, body {{
  width: {width_in}in;
  height: {height_in}in;
  overflow: hidden;
  background: {bg};
  font-family: var(--font-body);
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}}

/* ── Poster canvas ──────────────────────────────────────── */
.poster {{
  width: {width_in}in;
  height: {height_in}in;
  display: flex;
  flex-direction: column;
  padding: {margin_in}in {margin_in}in {margin_in + 0.15:.2f}in {margin_in}in;
  gap: {gutter_in}in;
  background: {bg};
  font-family: var(--font-body);
  overflow: hidden;
}}

/* ── Title banner ───────────────────────────────────────── */
.title-banner {{
  flex-shrink: 0;
  height: {title_h_in:.4f}in;
  background: {banner_bg_css};
  border-bottom: {banner_border_bottom};
  border-radius: 0;
  display: flex;
  flex-direction: row;
  align-items: center;
  padding: 0.10in 0.3in;
  overflow: hidden;
}}
.banner-logo-left {{
  flex-shrink: 0;
  width: 8%;
  display: flex;
  align-items: center;
  justify-content: flex-start;
}}
.banner-logo-right {{
  flex-shrink: 0;
  width: 8%;
  display: flex;
  align-items: center;
  justify-content: flex-end;
}}
.banner-center {{
  flex: 1;
  min-width: 0;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.04in;
  overflow: hidden;
}}
.banner-logo-img {{
  max-height: 70%;
  max-width: 90%;
  width: auto;
  height: auto;
  object-fit: contain;
  display: block;
}}
.banner-title {{
  font-size: {_FONT_TITLE_PT}pt;
  font-weight: 800;
  font-family: var(--font-title);
  color: {title_text_color};
  text-align: center;
  line-height: 1.15;
  letter-spacing: -0.5pt;
  word-break: break-word;
  max-height: 55%;
  overflow: hidden;
}}
.banner-authors {{
  font-size: {_FONT_AUTHORS_PT}pt;
  font-weight: 600;
  font-family: var(--font-heading);
  color: {author_color};
  text-align: center;
  line-height: 1.25;
  overflow: hidden;
  max-width: 95%;
}}
.banner-affil {{
  font-size: {_FONT_AFFIL_PT}pt;
  font-weight: 400;
  font-family: var(--font-body);
  color: {affil_color};
  text-align: center;
  line-height: 1.25;
  overflow: hidden;
  max-width: 95%;
}}
/* ── Content grid (3-column flexbox) ───────────────────── */
  .content-grid {{
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: row;
    gap: {gutter_in}in;
    overflow: hidden;
  }}

/* ── Column ─────────────────────────────────────────────── */
  .col {{
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: {row_gap_in}in;
    min-height: 0;
    overflow: hidden;
  }}

/* ── Panel card ─────────────────────────────────────────── */
.panel {{
  flex-grow: 1;
  min-height: 0;
  background: {surface};
  border: {panel_border};
  border-radius: {panel_radius_in}in;
  box-shadow: {panel_shadow};
  display: flex;
  flex-direction: column;
  overflow: hidden;
}}

/* ── Panel header bar ───────────────────────────────────── */
.panel-hdr {{
  flex-shrink: 0;
  background: {hdr_bg};
  border-radius: {panel_radius_in}in {panel_radius_in}in 0 0;
  padding: 0.07in 0.18in;
  display: flex;
  align-items: center;
}}
.panel-title {{
  font-size: calc({_FONT_PANEL_HDR_PT}pt * var(--panel-font-scale, 1));
  font-weight: 700;
  font-family: var(--font-heading);
  color: {hdr_text};
  line-height: 1.2;
  letter-spacing: 0.3pt;
  word-break: break-word;
}}

/* ── Panel body ─────────────────────────────────────────── */
.panel-body {{
  flex: 1 1 0;
  min-height: 0;
  padding: 0.14in 0.18in;
  display: flex;
  flex-direction: column;
  gap: 0.04in;
  overflow: hidden;
  height: 0;
}}
/* Side-by-side grid layout (text left, figure right) */
.panel-body-grid {{
  flex: 1 1 0;
  min-height: 0;
  height: 0;
  padding: 0.14in 0.18in;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.10in;
  overflow: hidden;
}}
.panel-body-grid .text-content {{
  overflow: hidden;
}}
.panel-body-grid .visual-slot {{
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}}

/* ── Panel layout modes ──────────────────────────────────── */
/* prose_only / bullets_only — text fills full panel body */
.panel-body.layout-prose-only,
.panel-body.layout-bullets-only {{
  justify-content: space-between;
}}
.panel-body.layout-prose-only .text-content,
.panel-body.layout-bullets-only .text-content {{
  flex: 1 1 0;
  min-height: 0;
  max-height: none;
  overflow: hidden;
}}
/* prose_with_figure — paragraph + figure below */
.panel-body.layout-prose-with-figure .text-content {{
  flex: 0 0 auto;
  max-height: 46%;
  overflow: hidden;
}}
.panel-body.layout-prose-with-figure .visual-slot {{
  flex: 1 1 0;
  min-height: 0;
}}
/* figure_dominant — small text, large figure */
.panel-body.layout-figure-dominant .text-content {{
  flex: 0 0 auto;
  max-height: 28%;
  overflow: hidden;
}}
.panel-body.layout-figure-dominant .visual-slot {{
  flex: 1 1 0;
  min-height: 0;
}}
/* standard — text ~50%, figure ~50% */
.panel-body.layout-standard .text-content.with-visual {{
  flex: 0 0 auto;
  max-height: 50%;
  overflow: hidden;
}}

/* ── Lead paragraph (prose content) ─────────────────────── */
.lead-para {{
  font-size: calc({_FONT_BULLET_PT}pt * var(--panel-font-scale, 1));
  font-weight: 500;
  font-family: var(--font-body);
  color: {text_primary};
  line-height: calc(1.70 * var(--panel-line-scale, 1));
  margin-bottom: 0.04in;
  overflow: hidden;
  display: -webkit-box;
  -webkit-line-clamp: 6;
  -webkit-box-orient: vertical;
}}

/* ── Bullet list ────────────────────────────────────────── */
.bullets {{
  list-style: none;
  padding: 0;
  overflow: hidden;
  min-height: 0;
}}
.bullets li {{
  font-size: calc({_FONT_BULLET_PT}pt * var(--panel-font-scale, 1));
  font-weight: 500;
  font-family: var(--font-body);
  color: {text_primary};
  line-height: calc({bullet_lh} * var(--panel-line-scale, 1));
  padding: 0.03in 0 0.03in 0.32in;
  position: relative;
  word-break: break-word;
  overflow: hidden;
  display: -webkit-box;
  -webkit-line-clamp: 4;
  -webkit-box-orient: vertical;
}}
.bullets li::before {{
  content: '\\2022';
  color: {marker_color};
  position: absolute;
  left: 0.09in;
  top: 0.04in;
  font-size: calc({int(_FONT_BULLET_PT * 0.65)}pt * var(--panel-font-scale, 1));
  font-weight: 700;
}}
/* Sub-header list items (bold section break within bullets) */
.bullets li.subheader-item {{
    font-weight: 700;
    font-size: calc({int(_FONT_BULLET_PT * 1.12)}pt * var(--panel-font-scale, 1));
    color: {marker_color};
    padding-left: 0;
    margin-top: 0.12in;
  }}
.bullets li.subheader-item::before {{
  content: none;
}}

/* ── Text content wrapper ──────────────────────────────── */
.text-content {{
  flex: 0 1 auto;
  min-height: 0;
  overflow: hidden;
}}
.text-content.with-visual {{
  flex: 0 0 auto;
  max-height: 52%;
  overflow: hidden;
}}
.panel-body strong, .panel-body-grid strong {{
  font-weight: 700;
  color: {bold_color};
}}

/* ── Visual / figure slot ───────────────────────────────── */
.visual-slot {{
  flex: var(--panel-visual-flex, 1) 1 0;
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0.03in 0.03in 0.02in;
  overflow: hidden;
  background: #FFFFFF;
}}
.visual-slot img {{
  object-fit: contain;
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: auto;
  border-radius: 0;
  border: none;
  display: block;
}}
.caption {{
  font-size: calc({_FONT_CAPTION_PT}pt * var(--panel-font-scale, 1));
  font-family: var(--font-body);
  color: {text_secondary};
  text-align: center;
  margin-top: 0.03in;
  line-height: 1.3;
  font-style: italic;
  max-width: 95%;
  overflow: hidden;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}}

/* ── Footer ─────────────────────────────────────────────── */
.footer {{
  flex-shrink: 0;
  height: {footer_h_in:.4f}in;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 0.1in;
  background: {footer_bg};
  overflow: hidden;
}}
.footer span {{
  font-size: {_FONT_FOOTER_PT}pt;
  font-family: var(--font-body);
  color: {footer_text};
}}

/* ── Print media (PDF via Playwright) ───────────────────── */
@media print {{
  @page {{
    size: {width_in}in {height_in}in;
    margin: 0;
  }}
  html, body {{
    width: {width_in}in;
    height: {height_in}in;
  }}
}}
"""

    # ── Casual mode CSS overrides ─────────────────────────────
    if poster_mode == "casual":
        # Compute casual-specific accent colors + gradients
        panel_accent = tokens.header_bar
        casual_hdr_text = _auto_text_on_bg(panel_accent, light="#FFFFFF", dark="#0C0C0C")
        title_bg_end = _shift_color_css(title_bg, 22 if is_dark else 28)
        accent_line = panel_accent
        bg_grad_end = _shift_color_css(bg, 10 if is_dark else -6)
        card_bg = _shift_color_css(surface, 8 if is_dark else -4)
        card_bg_end = _shift_color_css(card_bg, 6 if is_dark else -6)
        accent_soft = _shift_color_css(panel_accent, 30 if not is_dark else 14)
        accent_glow = f"rgba({_hex_to_css_rgb(panel_accent)},0.14)"
        accent_glow_soft = f"rgba({_hex_to_css_rgb(panel_accent)},0.08)"

        casual_title_h = height_in * 0.13
        casual_hdr_pt = _FONT_PANEL_HDR_PT + 5
        casual_bullet_pt = _FONT_BULLET_PT + 4
        casual_dot_pt = int(_FONT_BULLET_PT * 0.50)
        casual_title_pt = _FONT_TITLE_PT + 10

        css += f"""
/* ══ Casual mode overrides ══════════════════════════════════ */

  .poster.casual {{
    padding: {margin_in}in;
    gap: {gutter_in}in;
    position: relative;
    background:
      radial-gradient(1200px 700px at 8% 6%, {accent_glow} 0%, rgba(0,0,0,0) 60%),
      radial-gradient(900px 600px at 88% 12%, {accent_glow_soft} 0%, rgba(0,0,0,0) 55%),
      linear-gradient(135deg, {bg} 0%, {bg_grad_end} 100%);
    --font-title: "Inter", "Noto Sans", sans-serif;
    --font-heading: "Inter", "Helvetica Neue", sans-serif;
    --font-body: "Inter", "Helvetica Neue", sans-serif;
    --font-accent: "Inter", "Helvetica Neue", sans-serif;
  }}

/* ── Title banner: gradient + decorative accent line ──────── */
.poster.casual .title-banner {{
  height: {casual_title_h:.4f}in;
  background: linear-gradient(135deg, {title_bg} 0%, {title_bg_end} 100%);
  border-radius: 0.16in;
  border-bottom: 6pt solid {accent_line};
  padding: 0.22in 0.5in;
  gap: 0.07in;
  box-shadow: 0 10px 26px rgba(0,0,0,{'0.30' if is_dark else '0.12'});
}}
.poster.casual .banner-title {{
  font-size: {casual_title_pt}pt;
  font-weight: 800;
  letter-spacing: -1pt;
}}
.poster.casual .banner-authors {{
  font-size: {_FONT_AUTHORS_PT + 2}pt;
}}

/* ── Content grid: wider gaps ─────────────────────────────── */
  .poster.casual .content-grid {{
    display: grid;
    column-gap: {gutter_in}in;
    row-gap: {row_gap_in}in;
  }}
.poster.casual .content-grid.cols-1 {{ grid-template-columns: 1fr; }}
.poster.casual .content-grid.cols-2 {{ grid-template-columns: 1.15fr 0.85fr; }}
.poster.casual .content-grid.cols-3 {{ grid-template-columns: 1fr 1fr 1fr; }}
  .poster.casual .col {{
    gap: {row_gap_in}in;
  }}

/* ── Panel card: left accent bar, rounded, airy ──────────── */
.poster.casual .panel {{
  border-radius: 0.16in;
  border: 1.5pt solid {border_color};
  background: linear-gradient(180deg, {card_bg} 0%, {card_bg_end} 100%);
  box-shadow: 0 10px 26px rgba(0,0,0,{'0.22' if is_dark else '0.08'});
  position: relative;
  overflow: hidden;
}}
.poster.casual .panel::before {{
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  height: 0.10in;
  width: 100%;
  background: linear-gradient(90deg, {panel_accent} 0%, {accent_soft} 100%);
}}

/* ── Panel header: transparent bg, pill label ─────────────── */
.poster.casual .panel-hdr {{
  background: transparent;
  border-bottom: none;
  border-radius: 0;
  padding: 0.18in 0.24in 0.10in;
}}
.poster.casual .panel-title {{
  display: inline-block;
  color: {casual_hdr_text};
  font-size: {casual_hdr_pt}pt;
  font-weight: 700;
  letter-spacing: 0.6pt;
  background: {panel_accent};
  padding: 0.04in 0.16in;
  border-radius: 999px;
  box-shadow: 0 4px 12px rgba(0,0,0,{'0.20' if is_dark else '0.10'});
}}

/* ── Panel body: more padding ─────────────────────────────── */
.poster.casual .panel-body {{
  padding: 0.24in 0.28in;
  gap: 0.12in;
}}

/* ── Bullets: colored dot markers, roomier ────────────────── */
.poster.casual .bullets li {{
  font-size: {casual_bullet_pt}pt;
  line-height: 1.60;
  padding: 0.04in 0 0.04in 0.36in;
}}
.poster.casual .bullets li::before {{
  content: '\\25CF';
  font-size: {casual_dot_pt}pt;
  top: 0.075in;
  color: {panel_accent};
}}

/* ── Bold text: accent-colored in dark casual ─────────────── */
.poster.casual .panel-body strong {{
  color: {tokens.accent if is_dark else 'inherit'};
}}

/* ── Visual slot: larger, more prominent ──────────────────── */
.poster.casual .visual-slot {{
  flex: 1 1 0;
  min-height: 0;
  padding: 0.08in;
  background: {accent_glow_soft};
  border-radius: 0.12in;
  border: 1px solid {border_color};
}}
.poster.casual .text-content.with-visual {{
  flex: 0 0 auto;
  max-height: 30%;
  overflow: hidden;
}}
.poster.casual .visual-slot img {{
  border-radius: 0.10in;
  box-shadow: 0 6px 16px rgba(0,0,0,{'0.20' if is_dark else '0.08'});
}}

/* ── Footer: subtler ──────────────────────────────────────── */
.poster.casual .footer {{
  background: transparent;
}}
.poster.casual .footer span {{
  color: {tokens.text_secondary};
  font-size: {_FONT_FOOTER_PT - 1}pt;
}}
"""

    return css


def _hex_to_css_rgb(hex_color: str) -> str:
    """Convert '#RRGGBB' to 'R,G,B' for use in rgba()."""
    h = hex_color.lstrip("#")
    if len(h) < 6:
        h = h.ljust(6, "0")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"


def _hex_to_rgb_tuple(hex_color: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' to (r, g, b) ints."""
    h = hex_color.lstrip("#")
    if len(h) < 6:
        h = h.ljust(6, "0")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _perceived_brightness(hex_color: str) -> float:
    """Return perceived brightness in [0,1] for a hex color."""
    r, g, b = _hex_to_rgb_tuple(hex_color)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


def _auto_text_on_bg(
    bg_hex: str,
    *,
    light: str = "#FFFFFF",
    dark: str = "#0C0C0C",
    threshold: float = 0.58,
) -> str:
    """Pick light or dark text based on background brightness."""
    return dark if _perceived_brightness(bg_hex) > threshold else light


def _shift_color_css(hex_color: str, amount: int) -> str:
    """Shift all RGB channels by amount, clamped to 0-255."""
    h = hex_color.lstrip("#")
    if len(h) < 6:
        h = h.ljust(6, "0")
    r = max(0, min(255, int(h[0:2], 16) + amount))
    g = max(0, min(255, int(h[2:4], 16) + amount))
    b = max(0, min(255, int(h[4:6], 16) + amount))
    return f"#{r:02x}{g:02x}{b:02x}"


# ══════════════════════════════════════════════════════════════
# Image helpers
# ══════════════════════════════════════════════════════════════

def _img_to_data_url(path: Path, *, white_bg: bool = False) -> str | None:
    """Base64-encode an image file into a data URL for HTML embedding.

    When white_bg is True, attempt a conservative background whitening
    to help figures blend into white poster panels. This only triggers
    for images with a near-uniform light background.
    """
    if not path or not path.exists():
        return None
    try:
        ext = path.suffix.lower().lstrip(".")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/png")
        if white_bg and _PIL_AVAILABLE and ext in ("png", "jpg", "jpeg", "webp"):
            with _PILImage.open(path) as img:
                img = img.convert("RGBA")
                img = _maybe_whiten_background(img)
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                return f"data:image/png;base64,{b64}"
        with open(path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def _maybe_whiten_background(img: "_PILImage.Image") -> "_PILImage.Image":
    """Whiten near-uniform light backgrounds while preserving content."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    w, h = img.size
    if w < 10 or h < 10:
        return img

    # Sample border pixels to estimate background color
    border = []
    px = img.load()
    step = max(1, min(w, h) // 50)
    for x in range(0, w, step):
        border.append(px[x, 0])
        border.append(px[x, h - 1])
    for y in range(0, h, step):
        border.append(px[0, y])
        border.append(px[w - 1, y])

    if not border:
        return img

    # Compute mean and variance of border RGB
    rs = [p[0] for p in border]
    gs = [p[1] for p in border]
    bs = [p[2] for p in border]
    mean_r = sum(rs) / len(rs)
    mean_g = sum(gs) / len(gs)
    mean_b = sum(bs) / len(bs)

    var_r = sum((r - mean_r) ** 2 for r in rs) / len(rs)
    var_g = sum((g - mean_g) ** 2 for g in gs) / len(gs)
    var_b = sum((b - mean_b) ** 2 for b in bs) / len(bs)

    # Only whiten if the background is light and uniform
    if min(mean_r, mean_g, mean_b) < 200:
        return img
    if max(var_r, var_g, var_b) > 120:
        return img

    # Replace pixels close to background with pure white
    tol = 28
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a == 0:
                px[x, y] = (255, 255, 255, 255)
                continue
            if (
                abs(r - mean_r) <= tol
                and abs(g - mean_g) <= tol
                and abs(b - mean_b) <= tol
            ):
                px[x, y] = (255, 255, 255, 255)

    return img


def _resolve_figure_path(
    figure_ids: list[str],
    figures_dir: Path,
    config: "PosterConfig",
) -> "Path | None":
    """Find the best available image file for the given figure IDs.

    Returns None if no file is found or if the best candidate is below the
    minimum resolution threshold (300×300 px) — tiny figures look blurry
    when embedded and are better replaced by a generated visual.
    """
    if not figure_ids or not figures_dir.exists():
        return None

    for fig_id in figure_ids:
        num = fig_id.replace("fig_", "").replace("figure_", "")
        for pattern in [
            f"figure_{num}.png", f"figure_{num}.jpg", f"figure_{num}.jpeg",
            f"fig_{num}.png", f"fig_{num}.jpg",
            f"figure{num}.png", f"figure{num}.jpg",
            f"{fig_id}.png", f"{fig_id}.jpg",
        ]:
            candidate = figures_dir / pattern
            if candidate.exists() and _meets_min_resolution(candidate):
                return candidate

    return None

    return None


def _meets_min_resolution(path: Path, min_px: int = 300) -> bool:
    """Return True if the image meets minimum resolution and acceptable aspect ratio."""
    if not _PIL_AVAILABLE:
        return True  # can't check — allow it through
    try:
        with _PILImage.open(path) as img:
            w, h = img.size
        if w < min_px or h < min_px:
            return False
        # Reject extreme aspect ratios (>4:1 or <1:4) — these look broken in panels
        ratio = w / h if h > 0 else 0
        if ratio > 4.0 or ratio < 0.25:
            return False
        return True
    except Exception:
        return False


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return _html_mod.escape(str(text or ""))


def _render_inline_md(text: str) -> str:
    """HTML-escape and convert **bold** markdown to <strong> tags."""
    safe = _html_mod.escape(str(text or ""))
    safe = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', safe)
    return safe


# ══════════════════════════════════════════════════════════════
# HTML → PDF/PNG rendering
# ══════════════════════════════════════════════════════════════

def _render_html(
    html_path: Path,
    pdf_path: Path,
    png_path: Path,
    plan: "PosterPlan",
    config: "PosterConfig",
) -> Path:
    """Render the HTML file to PDF + PNG. Returns the primary output path."""
    width_in = plan.width_inches
    height_in = plan.height_inches
    dpi = config.dpi

    primary = pdf_path if pdf_path.suffix.lower() == ".pdf" else png_path

    rendered = _render_with_playwright(html_path, pdf_path, png_path, width_in, height_in, dpi)
    if not rendered:
        _render_fallback_pil(html_path, pdf_path, png_path, plan, config)

    return primary


def _find_system_browser() -> str | None:
    """Return path to an installed Chromium-based browser (Edge / Chrome) on Windows."""
    import shutil
    candidates = [
        # Edge (ships with Windows 10/11)
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        # Chrome
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        # Brave
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    # Also check PATH
    for name in ("msedge", "google-chrome", "chromium-browser", "chromium"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _render_with_playwright(
    html_path: Path,
    pdf_path: Path,
    png_path: Path,
    width_in: float,
    height_in: float,
    dpi: int,
) -> bool:
    """Render HTML to PDF + PNG using Playwright.

    Tries the system-installed Edge/Chrome first (no extra download needed on
    Windows 10/11). Falls back to the Playwright-managed Chromium only if a
    system browser is not found.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False

    try:
        html_path = html_path.resolve()
        if not html_path.exists():
            raise FileNotFoundError(f"HTML not found: {html_path}")

        # Viewport in CSS pixels (96 DPI base — CSS uses 'in' units)
        vp_w = int(width_in * 96)
        vp_h = int(height_in * 96)

        # Device scale factor → PNG output at the target DPI
        scale = max(dpi / 96, 1.0)

        # Use a proper file:// URI so Playwright can load local HTML reliably
        url = html_path.as_uri()

        system_exe = _find_system_browser()
        launch_kwargs: dict = {
            "args": ["--no-sandbox", "--disable-setuid-sandbox"],
        }
        if system_exe:
            launch_kwargs["executable_path"] = system_exe

        with sync_playwright() as pw:
            browser = pw.chromium.launch(**launch_kwargs)

            # ── PDF rendering ────────────────────────────────
            pdf_ctx = browser.new_context(viewport={"width": vp_w, "height": vp_h})
            pdf_page = pdf_ctx.new_page()
            pdf_page.goto(url, wait_until="load", timeout=60_000)
            pdf_page.wait_for_load_state("domcontentloaded")

            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_page.pdf(
                path=str(pdf_path),
                width=f"{width_in}in",
                height=f"{height_in}in",
                print_background=True,
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            )
            pdf_ctx.close()

            # ── PNG rendering ────────────────────────────────
            png_ctx = browser.new_context(
                viewport={"width": vp_w, "height": vp_h},
                device_scale_factor=scale,
            )
            png_page = png_ctx.new_page()
            png_page.goto(url, wait_until="load", timeout=60_000)
            png_page.wait_for_load_state("domcontentloaded")

            png_path.parent.mkdir(parents=True, exist_ok=True)
            png_page.screenshot(
                path=str(png_path),
                full_page=False,
                clip={"x": 0, "y": 0, "width": vp_w, "height": vp_h},
            )
            png_ctx.close()
            browser.close()

        return True

    except Exception:
        import traceback
        traceback.print_exc()
        return False


def _render_fallback_pil(
    html_path: Path,
    pdf_path: Path,
    png_path: Path,
    plan: "PosterPlan",
    config: "PosterConfig",
) -> None:
    """Last-resort fallback: create a minimal placeholder image noting the HTML path."""
    from PIL import Image, ImageDraw, ImageFont

    dpi = config.dpi
    w = int(plan.width_inches * dpi)
    h = int(plan.height_inches * dpi)
    img = Image.new("RGB", (w, h), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", max(w // 60, 24))
    except (OSError, IOError):
        font = ImageFont.load_default()

    msg = (
        f"Poster HTML saved to:\n{html_path}\n\n"
        "Install Playwright to render to PDF/PNG:\n"
        "  pip install playwright\n"
        "  playwright install chromium"
    )
    draw.text((w // 20, h // 4), msg, fill="#333333", font=font)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(png_path), quality=95, dpi=(dpi, dpi))

    try:
        from reportlab.lib.units import inch
        from reportlab.pdfgen import canvas as rl_canvas

        c = rl_canvas.Canvas(
            str(pdf_path),
            pagesize=(plan.width_inches * inch, plan.height_inches * inch),
        )
        tmp_png = pdf_path.with_suffix(".tmp.png")
        img.save(str(tmp_png), quality=95, dpi=(dpi, dpi))
        c.drawImage(
            str(tmp_png), 0, 0,
            width=plan.width_inches * inch,
            height=plan.height_inches * inch,
            preserveAspectRatio=True,
            anchor="sw",
        )
        c.save()
        tmp_png.unlink(missing_ok=True)
    except Exception:
        pass
