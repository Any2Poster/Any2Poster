"""Stage 2: ANALYZE

Two-pass LLM analysis with visual suggestion generation.

Pass 1 — Global: Extract poster title, headline result, key contribution,
         section importance, panel categories, and visual suggestions.
Pass 2 — Per-section: Extract poster-ready bullets with global context,
         enforce 15-word bullet limit and 3-4 bullet max per panel.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from any2poster.models import (
    AnalyzedContent,
    AnalyzedSection,
    BulletProvenance,
    ExtractedSection,
    GlobalAnalysis,
    PanelCategory,
    PanelSubHeader,
    SectionType,
    ChunkedDocument,
    TextChunk,
    VisualSuggestion,
    VisualType,
)
from any2poster.prompts import (
    GLOBAL_ANALYSIS_SYSTEM,
    GLOBAL_ANALYSIS_USER,
    SECTION_ANALYSIS_SYSTEM,
    SECTION_ANALYSIS_USER,
)

if TYPE_CHECKING:
    from any2poster.models import ParsedDocument, PosterConfig
    from any2poster.providers.openrouter import OpenRouterProvider

_CATEGORY_MAP = {
    "motivation": PanelCategory.MOTIVATION,
    "methodology": PanelCategory.METHODOLOGY,
    "architecture": PanelCategory.ARCHITECTURE,
    "results": PanelCategory.RESULTS,
    "analysis": PanelCategory.ANALYSIS,
    "conclusion": PanelCategory.CONCLUSION,
    "dataset": PanelCategory.DATASET,
    "content": PanelCategory.CONTENT,
}

_VISUAL_TYPE_MAP = {
    "flowchart": VisualType.FLOWCHART,
    "architecture_diagram": VisualType.ARCHITECTURE_DIAGRAM,
    "bar_chart": VisualType.BAR_CHART,
    "line_chart": VisualType.LINE_CHART,
    "pipeline": VisualType.PIPELINE,
    "comparison": VisualType.COMPARISON,
    "concept_diagram": VisualType.CONCEPT_DIAGRAM,
    "table": VisualType.TABLE,
    "infographic": VisualType.INFOGRAPHIC,
    "matrix": VisualType.MATRIX,
}

_MAX_BULLETS_HIGH = 4
_MAX_BULLETS_STANDARD = 3
_MAX_BULLET_WORDS = 10
_HIGH_IMPORTANCE_THRESHOLD = 1.3
_MIN_VISUAL_SUGGESTIONS = 6
_MAX_VISUAL_SUGGESTIONS = 12

# Section title keywords that indicate boilerplate to skip on the poster.
# These appear in Project Gutenberg PDFs, legal documents, etc.
_BOILERPLATE_KEYWORDS = frozenset({
    "copyright", "license", "licence", "legal", "disclaimer", "terms of use",
    "terms and conditions", "project gutenberg", "gutenberg", "public domain",
    "about the author", "about the publisher", "acknowledgment", "acknowledgement",
    "support", "donation", "how to donate", "colophon", "bibliograph",
    "mission statement", "foundation", "publisher's note", "editor's note",
    "transcriber", "errata",
})


def _is_boilerplate(section) -> bool:
    """Return True if the section looks like legal/publisher boilerplate to skip."""
    title_lower = section.title.lower().strip()
    return any(kw in title_lower for kw in _BOILERPLATE_KEYWORDS)


_PROSE_SEGMENT_SYSTEM = (
    "You are a document analyst. Identify the main thematic or narrative segments "
    "of the given document and extract the most representative passage from each. "
    "You MUST respond with valid JSON only."
)

_PROSE_SEGMENT_USER = """\
This document has very few section headers but rich content. Identify exactly 8
distinct thematic, narrative, or topical segments that would make excellent poster panels.
For each segment, provide:
  - A short, evocative title (2-5 words, suitable as a poster panel heading)
  - The most representative 200-350 words extracted verbatim from the document

Document content:
{document_text}

Respond with this exact JSON:
{{
  "segments": [
    {{"title": "<panel title>", "text": "<200-350 verbatim words from the document>"}},
    ...
  ]
}}

REQUIRED segment structure — you MUST produce ALL 8:
  1. Opening / Inciting Incident — the very beginning, introducing the situation or problem
  2. Characters & Setting — who the key people/entities are and where they exist
  3. Central Conflict / Development — how the central problem unfolds or escalates
  4. A Key Scene or Turning Point — a pivotal moment that changes the direction of the story
  5. Deepening Complications — later complications, secondary conflicts, or developments
  6. Climax or Crisis — the peak tension or decisive moment
  7. Resolution & Ending — what ultimately happens; how the story concludes for each character
  8. Themes & Significance — the deeper meaning, major themes, and what the work says about
     the human condition (write this as character names + theme labels, not a plot summary)

Critical rules:
- Segments MUST span the ENTIRE document from start to finish — do NOT cluster in the opening
- Segment 7 (Resolution) MUST reflect the actual ending of the document
- Segment 8 (Themes) MUST name the themes explicitly (e.g. "alienation", "identity", "family")
- Titles must be specific to THIS document's content
- Text must be verbatim excerpts (not paraphrases or summaries)\
"""


def _get_segmentation_text(parsed: "ParsedDocument", max_words: int = 10000) -> str:
    """Return a representative sample of the full document for prose segmentation.

    Samples beginning (50%) + middle (20%) + end (30%) so the segmentation LLM
    always sees how the story ends, regardless of document length.
    """
    full = parsed.get_full_text(max_tokens=0)  # no truncation
    words = full.split()
    if len(words) <= max_words:
        return full

    begin_count = max_words * 5 // 10
    mid_count = max_words * 2 // 10
    end_count = max_words - begin_count - mid_count

    begin = " ".join(words[:begin_count])
    mid_start = max(begin_count, len(words) // 2 - mid_count // 2)
    mid_end = min(len(words) - end_count, mid_start + mid_count)
    middle = " ".join(words[mid_start:mid_end])
    end = " ".join(words[-end_count:])

    return (
        begin
        + "\n\n[... story continues ...]\n\n"
        + middle
        + "\n\n[... story continues ...]\n\n"
        + end
    )


def _maybe_segment_prose(
    parsed: "ParsedDocument",
    provider: "OpenRouterProvider",
) -> "ParsedDocument":
    """If the document has < 4 usable sections (e.g. continuous fiction prose),
    ask the LLM to identify 8 narrative segments and synthesize them as sections."""
    content_sections = [
        s for s in parsed.sections
        if s.section_type not in (SectionType.REFERENCES, SectionType.ACKNOWLEDGMENTS)
        and not _is_boilerplate(s)
        and s.word_count > 50
    ]
    if len(content_sections) >= 4:
        return parsed

    full_text = _get_segmentation_text(parsed)
    if not full_text or not full_text.strip():
        return parsed

    try:
        result = _safe_json_call(
            provider,
            _PROSE_SEGMENT_SYSTEM,
            _PROSE_SEGMENT_USER.format(document_text=full_text),
            temperature=0.2,
            max_tokens=6000,
        )
        segments = result.get("segments", [])
    except Exception:
        return parsed

    if len(segments) < 4:
        return parsed

    new_sections: list[ExtractedSection] = []
    for i, seg in enumerate(segments[:9]):
        title = str(seg.get("title", f"Section {i + 1}")).strip()
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        new_sections.append(ExtractedSection(
            section_id=f"prose_seg_{i + 1}",
            title=title,
            section_type=SectionType.OTHER,
            content=text,
            level=1,
            word_count=len(text.split()),
        ))

    if len(new_sections) < 4:
        return parsed

    # Keep non-boilerplate non-content sections (e.g. figures) but replace prose body
    other_sections = [
        s for s in parsed.sections
        if _is_boilerplate(s) is False
        and s.section_id not in {s2.section_id for s2 in content_sections}
    ]
    return parsed.model_copy(update={"sections": new_sections + other_sections})


def run_analyze_stage(
    parsed: "ParsedDocument",
    config: "PosterConfig",
    provider: "OpenRouterProvider",
    chunked: "ChunkedDocument | None" = None,
) -> AnalyzedContent:
    parsed = _maybe_segment_prose(parsed, provider)
    global_analysis = _pass_one_global(parsed, provider)
    if global_analysis.poster_title:
        global_analysis.poster_title = _shorten_poster_title(global_analysis.poster_title)
    global_analysis.visual_suggestions = _ensure_visual_suggestions(parsed, global_analysis)
    sections = _pass_two_sections(parsed, global_analysis, provider, chunked, config)
    _assign_reading_order(sections)

    return AnalyzedContent(
        global_analysis=global_analysis,
        sections=sections,
        poster_title=_shorten_poster_title(global_analysis.poster_title or parsed.title),
        poster_authors=_format_authors_for_banner(global_analysis.authors or parsed.authors),
        poster_key_message=global_analysis.key_contribution,
    )


def _pass_one_global(
    parsed: "ParsedDocument",
    provider: "OpenRouterProvider",
) -> GlobalAnalysis:
    document_text = parsed.get_full_text(max_tokens=12000)

    sections_list = "\n".join(
        f"- {s.section_id}: {s.title} (type={s.section_type.value}, words={s.word_count})"
        for s in parsed.sections
        if s.section_type not in (SectionType.REFERENCES, SectionType.ACKNOWLEDGMENTS)
        and not _is_boilerplate(s)
    )

    figures_list = "\n".join(
        f"- {f.figure_id}: {f.caption[:100]}" if f.caption else f"- {f.figure_id}: (no caption)"
        for f in parsed.figures
    ) or "No figures extracted."

    user_prompt = GLOBAL_ANALYSIS_USER.format(
        document_text=document_text,
        sections_list=sections_list,
        figures_list=figures_list,
    )

    data = _safe_json_call(provider, GLOBAL_ANALYSIS_SYSTEM, user_prompt, 0.2, 2048)

    visual_suggestions = []
    for vs in data.get("visual_suggestions", []):
        if not isinstance(vs, dict):
            continue
        vtype_str = vs.get("visual_type", "concept_diagram")
        visual_suggestions.append(
            VisualSuggestion(
                concept=vs.get("concept", ""),
                description=vs.get("description", ""),
                visual_type=_VISUAL_TYPE_MAP.get(vtype_str, VisualType.CONCEPT_DIAGRAM),
                data_points=vs.get("data_points", []),
            )
        )

    section_categories = {}
    for sid, cat_str in data.get("section_categories", {}).items():
        section_categories[sid] = cat_str if cat_str in _CATEGORY_MAP else "content"

    return GlobalAnalysis(
        poster_title=data.get("poster_title", parsed.title),
        authors=data.get("authors", parsed.authors),
        affiliation=data.get("affiliation", ""),
        key_contribution=data.get("key_contribution", ""),
        headline_result=data.get("headline_result", ""),
        summary=data.get("summary", ""),
        narrative_arc=data.get("narrative_arc", ""),
        paper_domain=data.get("paper_domain", ""),
        methodology_summary=data.get("methodology_summary", ""),
        results_summary=data.get("results_summary", ""),
        suggested_color_theme=data.get("suggested_color_theme", "paper_clean"),
        venue=data.get("venue", ""),
        sections_to_include=data.get("sections_to_include", []),
        section_importance=data.get("section_importance", {}),
        section_categories=section_categories,
        essential_figure_ids=data.get("essential_figure_ids", []),
        visual_suggestions=visual_suggestions,
    )


def _pass_two_sections(
    parsed: "ParsedDocument",
    global_ctx: GlobalAnalysis,
    provider: "OpenRouterProvider",
    chunked: "ChunkedDocument | None",
    config: "PosterConfig",
) -> list[AnalyzedSection]:
    sections_to_analyze = list(global_ctx.sections_to_include or [])
    min_panels = max(1, config.min_panels)
    max_panels = max(min_panels, config.max_panels)

    # Complexity-based panel count: vary between 7-9 based on paper structure
    meaningful_sections = sum(
        1 for s in parsed.sections
        if s.section_type not in (SectionType.REFERENCES, SectionType.ACKNOWLEDGMENTS)
        and s.word_count > 100
    )
    if meaningful_sections <= 5:
        target_panels = min(max_panels, max(min_panels, 7))
    elif meaningful_sections <= 8:
        target_panels = min(max_panels, max(min_panels, 8))
    else:
        target_panels = max_panels
    max_panels = target_panels

    if not sections_to_analyze:
        sections_to_analyze = [
            s.section_id
            for s in parsed.sections
            if s.section_type not in (SectionType.REFERENCES, SectionType.ACKNOWLEDGMENTS)
            and not _is_boilerplate(s)
            and s.word_count > 50
        ][:max_panels]

    # Always strip boilerplate even if the LLM selected it
    section_map_bp = {s.section_id: s for s in parsed.sections}
    sections_to_analyze = [
        sid for sid in sections_to_analyze
        if not _is_boilerplate(section_map_bp.get(sid, type("_", (), {"title": ""})()))
    ]

    # Push toward max_panels by adding more sections if available
    if len(sections_to_analyze) < max_panels:
        candidates = [
            s for s in parsed.sections
            if s.section_id not in sections_to_analyze
            and s.section_type not in (SectionType.REFERENCES, SectionType.ACKNOWLEDGMENTS)
            and not _is_boilerplate(s)
            and s.word_count > 50
        ]
        candidates.sort(
            key=lambda s: (
                global_ctx.section_importance.get(s.section_id, 1.0),
                s.word_count,
            ),
            reverse=True,
        )
        for s in candidates:
            if len(sections_to_analyze) >= max_panels:
                break
            sections_to_analyze.append(s.section_id)

    if len(sections_to_analyze) > max_panels:
        sections_to_analyze = sections_to_analyze[:max_panels]

    # ── Deduplication: drop sections with near-duplicate titles ──
    section_map_tmp = {s.section_id: s for s in parsed.sections}
    seen_titles: set[str] = set()
    deduplicated: list[str] = []
    for sid in sections_to_analyze:
        section = section_map_tmp.get(sid)
        if not section:
            continue
        normalized = re.sub(r'\W+', ' ', section.title.lower()).strip()
        is_dup = False
        for seen in seen_titles:
            if normalized in seen or seen in normalized:
                is_dup = True
                break
        if not is_dup:
            seen_titles.add(normalized)
            deduplicated.append(sid)
    sections_to_analyze = deduplicated

    section_map = {s.section_id: s for s in parsed.sections}
    chunk_map: dict[str, list[TextChunk]] = {}
    if chunked:
        for ch in chunked.chunks:
            chunk_map.setdefault(ch.section_id, []).append(ch)
    visual_map = _map_visuals_to_sections(global_ctx)
    if not visual_map and global_ctx.visual_suggestions:
        for idx, sid in enumerate(sections_to_analyze):
            if idx >= len(global_ctx.visual_suggestions):
                break
            visual_map[sid] = global_ctx.visual_suggestions[idx]
    analyzed: list[AnalyzedSection] = []

    for section_id in sections_to_analyze:
        section = section_map.get(section_id)
        if not section:
            continue

        base_importance = global_ctx.section_importance.get(section_id, 1.0)
        if section.word_count:
            work_boost = min(section.word_count / 1500.0, 0.6)
        else:
            work_boost = 0.0
        importance = base_importance + (work_boost if base_importance <= 1.0 else 0.0)
        category_str = global_ctx.section_categories.get(section_id, "content")
        category = _CATEGORY_MAP.get(category_str, PanelCategory.CONTENT)

        section_text = section.content
        use_chunks = len(section_text.split()) > 3000 and section_id in chunk_map

        if use_chunks:
            data = _analyze_section_chunks(
                section=section,
                chunks=chunk_map[section_id],
                global_ctx=global_ctx,
                provider=provider,
                importance=importance,
                section_category=category_str,
            )
        else:
            data = _analyze_section_text(
                section=section,
                section_text=section_text,
                global_ctx=global_ctx,
                provider=provider,
                importance=importance,
                section_category=category_str,
            )

        # Parse content_type from LLM response
        content_type = str(data.get("content_type", "bullets")).strip().lower()
        if content_type not in ("prose", "bullets", "mixed"):
            content_type = "bullets"
        lead_paragraph = str(data.get("lead_paragraph", "")).strip()

        # Cap bullets based on content type
        if content_type == "prose":
            max_bullets = 2
        elif content_type == "mixed":
            max_bullets = 3
        else:
            max_bullets = (
                _MAX_BULLETS_HIGH if importance >= _HIGH_IMPORTANCE_THRESHOLD else _MAX_BULLETS_STANDARD
            )
        bullets = _enforce_bullet_constraints(data.get("bullets", []), max_bullets)

        poster_title = _shorten_panel_title(data.get("poster_section_title", section.title))

        provenance: list[BulletProvenance] = []
        for prov_entry in data.get("provenance", []):
            if isinstance(prov_entry, dict):
                provenance.append(
                    BulletProvenance(
                        source_section_id=section_id,
                        source_chunk_id=prov_entry.get("source_chunk_id"),
                        source_text_span=prov_entry.get("source_text_span", ""),
                    )
                )

        rec_figs = data.get("recommended_figure_ids", [])
        fig_ids = list(set(section.figure_ids + rec_figs))
        has_fig = bool(fig_ids) or any(
            fid in global_ctx.essential_figure_ids for fid in fig_ids
        )

        visual_suggestion = visual_map.get(section_id)

        # Parse sub_headers from LLM data
        sub_headers: list[PanelSubHeader] = []
        for sh_entry in data.get("sub_headers", []):
            if not isinstance(sh_entry, dict):
                continue
            idx = sh_entry.get("after_bullet_index")
            text = str(sh_entry.get("text", "")).strip()
            if text and idx is not None and 0 <= int(idx) < len(bullets):
                words = text.split()
                if len(words) > 6:
                    text = " ".join(words[:6])
                sub_headers.append(PanelSubHeader(after_bullet_index=int(idx), text=text))

        analyzed.append(
            AnalyzedSection(
                section_id=section_id,
                title=poster_title,
                section_type=section.section_type,
                panel_category=category,
                content_type=content_type,
                lead_paragraph=lead_paragraph,
                bullets=bullets,
                sub_headers=sub_headers,
                provenance=provenance,
                importance=importance,
                has_figure=has_fig,
                figure_ids=fig_ids if has_fig else [],
                key_message=data.get("key_message", ""),
                visual_suggestion=visual_suggestion,
            )
        )

    # Distribute any unassigned extracted figures to panels that lack visuals
    _distribute_unassigned_figures(analyzed, parsed, global_ctx)

    return analyzed


def _analyze_section_text(
    section,
    section_text: str,
    global_ctx: GlobalAnalysis,
    provider: "OpenRouterProvider",
    importance: float,
    section_category: str,
) -> dict:
    if len(section_text.split()) > 3000:
        section_text = " ".join(section_text.split()[:3000])

    user_prompt = SECTION_ANALYSIS_USER.format(
        poster_title=global_ctx.poster_title,
        key_contribution=global_ctx.key_contribution,
        headline_result=global_ctx.headline_result,
        paper_domain=global_ctx.paper_domain,
        section_importance=importance,
        section_category=section_category,
        section_title=section.title,
        section_type=section.section_type.value,
        section_text=section_text,
    )

    return _safe_json_call(provider, SECTION_ANALYSIS_SYSTEM, user_prompt, 0.3, 1536)


def _analyze_section_chunks(
    section,
    chunks: list[TextChunk],
    global_ctx: GlobalAnalysis,
    provider: "OpenRouterProvider",
    importance: float,
    section_category: str,
) -> dict:
    merged_bullets: list[str] = []
    merged_provenance: list[dict] = []
    merged_rec_figs: list[str] = []
    poster_section_title = ""
    key_message = ""

    for chunk in chunks:
        chunk_text = chunk.content
        if len(chunk_text.split()) > 2500:
            chunk_text = " ".join(chunk_text.split()[:2500])

        user_prompt = SECTION_ANALYSIS_USER.format(
            poster_title=global_ctx.poster_title,
            key_contribution=global_ctx.key_contribution,
            headline_result=global_ctx.headline_result,
            paper_domain=global_ctx.paper_domain,
            section_importance=importance,
            section_category=section_category,
            section_title=f"{section.title} (chunk {chunk.chunk_id})",
            section_type=section.section_type.value,
            section_text=chunk_text,
        )

        data = _safe_json_call(provider, SECTION_ANALYSIS_SYSTEM, user_prompt, 0.3, 1536)

        if not poster_section_title and data.get("poster_section_title"):
            poster_section_title = data.get("poster_section_title")
        if not key_message and data.get("key_message"):
            key_message = data.get("key_message")

        bullets = data.get("bullets", [])
        for b in bullets:
            if isinstance(b, str) and b.strip():
                merged_bullets.append(b.strip())

        for prov_entry in data.get("provenance", []):
            if isinstance(prov_entry, dict):
                merged_provenance.append(
                    {
                        "source_chunk_id": chunk.chunk_id,
                        "source_text_span": prov_entry.get("source_text_span", ""),
                    }
                )

        merged_rec_figs.extend(data.get("recommended_figure_ids", []))

    # Deduplicate bullets preserving order
    seen = set()
    unique_bullets: list[str] = []
    for b in merged_bullets:
        key = b.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique_bullets.append(b)

    return {
        "poster_section_title": poster_section_title,
        "bullets": unique_bullets,
        "key_message": key_message,
        "provenance": merged_provenance,
        "recommended_figure_ids": list(set(merged_rec_figs)),
    }


def _enforce_bullet_constraints(bullets: list[str], max_count: int) -> list[str]:
    valid = []
    for b in bullets:
        if not isinstance(b, str) or not b.strip():
            continue
        cleaned = b.strip().rstrip(".").strip()
        words = cleaned.split()
        if len(words) > _MAX_BULLET_WORDS:
            cleaned = " ".join(words[:_MAX_BULLET_WORDS])
        valid.append(cleaned)
        if len(valid) >= max_count:
            break
    return valid


def _shorten_panel_title(title: str) -> str:
    if not title:
        return "Section"
    # Keep at most 4 words for compact panel headers
    words = title.strip().split()
    short = " ".join(words[:4])
    return short


def _shorten_poster_title(title: str) -> str:
    if not title:
        return "Research Poster"
    # Keep title readable while allowing wider horizontal composition.
    words = title.strip().split()
    if len(words) > 10:
        words = words[:10]
    shortened = " ".join(words)
    if len(shortened) > 72:
        shortened = shortened[:69].rstrip() + "..."
    return shortened


def _format_authors_for_banner(authors: str, max_authors: int = 6) -> str:
    if not authors:
        return ""

    normalized = " ".join(authors.split())
    parts = [
        p.strip()
        for p in re.split(r"\s*(?:,|;|\band\b)\s*", normalized, flags=re.IGNORECASE)
        if p and p.strip()
    ]
    if not parts:
        return normalized

    if len(parts) <= max_authors:
        result = ", ".join(parts)
    else:
        result = ", ".join(parts[:4]) + ", et al."

    # Only hard-truncate if extremely long (banner can wrap to 2 lines)
    if len(result) > 90:
        result = ", ".join(parts[:4]) + ", et al."
    return result


def _fallback_visual_suggestions(
    parsed: "ParsedDocument",
    global_ctx: GlobalAnalysis,
    target_count: int = _MAX_VISUAL_SUGGESTIONS,
) -> list[VisualSuggestion]:
    suggestions: list[VisualSuggestion] = []
    # Map section types to appropriate visual types for variety
    _type_for_section = {
        SectionType.METHODS: (VisualType.PIPELINE, "pipeline"),
        SectionType.EXPERIMENTS: (VisualType.BAR_CHART, "bar chart"),
        SectionType.RESULTS: (VisualType.COMPARISON, "comparison"),
        SectionType.INTRODUCTION: (VisualType.CONCEPT_DIAGRAM, "concept diagram"),
        SectionType.DISCUSSION: (VisualType.CONCEPT_DIAGRAM, "concept diagram"),
    }
    section_map = {s.section_id: s for s in parsed.sections}
    ordered_sections = []
    ordered_ids: set[str] = set()
    for sid in global_ctx.sections_to_include:
        section = section_map.get(sid)
        if section:
            ordered_sections.append(section)
            ordered_ids.add(section.section_id)
    for section in parsed.sections:
        if section.section_id in ordered_ids:
            continue
        ordered_sections.append(section)
        ordered_ids.add(section.section_id)

    for section in ordered_sections:
        if section.section_type in (SectionType.REFERENCES, SectionType.ACKNOWLEDGMENTS):
            continue
        vtype, vdesc = _type_for_section.get(
            section.section_type, (VisualType.CONCEPT_DIAGRAM, "concept diagram")
        )
        suggestions.append(
            VisualSuggestion(
                concept=section.title or "Key concept",
                description=f"{vdesc} of {section.title}",
                visual_type=vtype,
                data_points=[],
            )
        )
        if len(suggestions) >= target_count:
            break
    return suggestions


def _ensure_visual_suggestions(
    parsed: "ParsedDocument",
    global_ctx: GlobalAnalysis,
) -> list[VisualSuggestion]:
    target = _target_visual_suggestion_count(parsed, global_ctx)
    existing: list[VisualSuggestion] = []
    seen: set[tuple[str, str]] = set()

    for vs in global_ctx.visual_suggestions:
        if not vs.concept or not vs.description:
            continue
        key = (vs.concept.strip().lower(), vs.visual_type.value)
        if key in seen:
            continue
        seen.add(key)
        existing.append(vs)
        if len(existing) >= target:
            return existing

    supplemental = _fallback_visual_suggestions(
        parsed=parsed,
        global_ctx=global_ctx,
        target_count=min(target * 2, _MAX_VISUAL_SUGGESTIONS),
    )
    for vs in supplemental:
        key = (vs.concept.strip().lower(), vs.visual_type.value)
        if key in seen:
            continue
        seen.add(key)
        existing.append(vs)
        if len(existing) >= target:
            break

    return existing


def _target_visual_suggestion_count(
    parsed: "ParsedDocument",
    global_ctx: GlobalAnalysis,
) -> int:
    included_sections = len(global_ctx.sections_to_include)
    if included_sections <= 0:
        included_sections = 8

    # Target a visual for nearly every section
    target = max(_MIN_VISUAL_SUGGESTIONS, included_sections)

    figure_count = len(parsed.figures)
    if figure_count >= 6:
        target = max(target, 7)
    if figure_count >= 10:
        target = max(target, 9)

    return min(target, _MAX_VISUAL_SUGGESTIONS)


def _map_visuals_to_sections(global_ctx: GlobalAnalysis) -> dict[str, VisualSuggestion]:
    result: dict[str, VisualSuggestion] = {}
    categories_order = [
        PanelCategory.METHODOLOGY,
        PanelCategory.ARCHITECTURE,
        PanelCategory.RESULTS,
        PanelCategory.ANALYSIS,
        PanelCategory.MOTIVATION,
        PanelCategory.DATASET,
    ]
    available_visuals = list(global_ctx.visual_suggestions)

    if not global_ctx.section_categories and available_visuals:
        # Fallback: assign visuals to the first few sections in order
        return {
            sid: available_visuals[idx]
            for idx, sid in enumerate(global_ctx.sections_to_include[: len(available_visuals)])
        }

    for cat in categories_order:
        if not available_visuals:
            break
        for sid, cat_str in global_ctx.section_categories.items():
            if _CATEGORY_MAP.get(cat_str) == cat and sid not in result and available_visuals:
                result[sid] = available_visuals.pop(0)

    return result


def _distribute_unassigned_figures(
    analyzed: list[AnalyzedSection],
    parsed: "ParsedDocument",
    global_ctx: GlobalAnalysis,
) -> None:
    """Assign extracted figures to panels that don't yet have visuals."""
    # Collect all figure IDs already assigned
    assigned_figs: set[str] = set()
    for section in analyzed:
        assigned_figs.update(section.figure_ids)

    # Only distribute figures the LLM explicitly marked as essential.
    # Non-essential extracted figures are skipped — generated visuals are
    # cleaner and better suited for poster scale.
    essential_ids = set(global_ctx.essential_figure_ids)
    available_figs = [
        f for f in parsed.figures
        if f.figure_id not in assigned_figs
        and f.figure_id in essential_ids
        and f.is_available
    ]
    if not available_figs:
        return

    # Only assign to panels that have no visual at all
    empty_panels = [
        s for s in analyzed
        if not s.has_figure and not s.visual_suggestion
    ]

    for panel, fig in zip(empty_panels, available_figs):
        panel.figure_ids = [fig.figure_id]
        panel.has_figure = True


def _assign_reading_order(sections: list[AnalyzedSection]) -> None:
    category_order = [
        PanelCategory.MOTIVATION,
        PanelCategory.METHODOLOGY,
        PanelCategory.ARCHITECTURE,
        PanelCategory.DATASET,
        PanelCategory.RESULTS,
        PanelCategory.ANALYSIS,
        PanelCategory.CONCLUSION,
        PanelCategory.CONTENT,
    ]
    priority = {cat: i for i, cat in enumerate(category_order)}
    sections.sort(key=lambda s: priority.get(s.panel_category, 99))
    for idx, section in enumerate(sections):
        section.poster_section_number = idx + 1


def _safe_json_call(
    provider: "OpenRouterProvider",
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    try:
        return provider.complete_json(
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except (json.JSONDecodeError, ValueError):
        try:
            raw = provider.complete(
                system=system,
                user=user,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return _extract_json(raw)
        except Exception:
            return {"bullets": [], "key_message": ""}


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start: end + 1])
        except json.JSONDecodeError:
            pass
    return {"bullets": [], "key_message": ""}
