# Stores prompt templates and theme helpers used by analysis and generation stages.

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from any2poster.models import DesignSystem, DesignToken

if TYPE_CHECKING:
    from any2poster.models import PosterConfig


# ══════════════════════════════════════════════════════════════
# LLM prompt templates
# ══════════════════════════════════════════════════════════════

GLOBAL_ANALYSIS_SYSTEM = (
    "You are an expert information poster designer who creates visual summaries "
    "for any type of content — academic research papers, news articles, business "
    "reports, educational materials, and creative or narrative works. "
    "You analyze documents and produce structured JSON that guides poster creation. "
    "You identify the core message, the single most impactful element, and determine "
    "which sections and visuals best represent the content on a poster. "
    "You MUST respond with valid JSON only. Never refuse to process a document — "
    "every document has a story worth visualizing."
)

GLOBAL_ANALYSIS_USER = """Analyze this document for poster creation.

DOCUMENT:
{document_text}

Respond with this exact JSON structure:
{{
  "poster_title": "<concise, accurate poster title, max 12 words — use the actual document title>",
  "authors": "<author, narrator, or creator names — empty string if none>",
  "affiliation": "<primary institution, publication, or source — empty string if none>",
  "key_contribution": "<single sentence: the ONE core message, central argument, or main contribution of this document>",
  "headline_result": "<the single most striking fact, statistic, quote, or narrative moment — can be a number, a quote, a key event, or a pivotal scene>",
  "summary": "<2 sentence summary of what this document is about and why it matters>",
  "narrative_arc": "<one sentence: from the opening situation or problem to its resolution, conclusion, or impact>",
  "paper_domain": "<content domain or genre, e.g. NLP, fiction, news, business, biology, history>",
  "methodology_summary": "<1 sentence describing the approach, structure, narrative method, or how the document makes its case>",
  "results_summary": "<1 sentence describing the key findings, outcomes, resolution, or conclusion>",
  "suggested_color_theme": "<one of: steel_blue, arctic, periwinkle, aquamarine, forest, ruby, apricot, blush, onyx>",
  "venue": "<conference, publication, or context if explicitly stated — empty string if not found>",
  "sections_to_include": ["<section_id>"],
  "section_importance": {{"<section_id>": <float 0.5-2.0>}},
  "section_categories": {{"<section_id>": "<one of: motivation, methodology, architecture, results, analysis, conclusion, dataset, content>"}},
  "essential_figure_ids": ["<fig_id>"],
  "visual_suggestions": [
    {{
      "concept": "<what to visualize>",
      "description": "<5-10 word description of the diagram>",
      "visual_type": "<flowchart|architecture_diagram|bar_chart|line_chart|pipeline|comparison|concept_diagram|infographic|matrix>",
      "data_points": ["<label: value>"]
    }}
  ]
}}

AVAILABLE SECTIONS:
{sections_list}

AVAILABLE FIGURES:
{figures_list}

Rules:
- Include 7-9 sections on the poster. NEVER more than 9:
  * Short/focused documents (few distinct sections): 7 panels
  * Standard documents: 8 panels
  * Complex documents with many distinct sections: 9 panels
- CRITICAL: Each section must cover a DISTINCT aspect — NO overlapping content between sections
  * If two sections would say similar things, MERGE them into one or DROP the weaker one
  * "Performance Results" and "Performance Analysis" should be ONE panel, not two
  * "Impact & Future" and "Conclusion" should be ONE panel, not two
- Skip references, acknowledgments, and pure metadata sections
- headline_result should be the most impactful single element: a key statistic, a pivotal quote, a striking fact, or a memorable narrative moment
- CRITICAL: If two sections would cover overlapping content, merge them and keep the richer one
- Suggest 6-8 visuals that would help explain the content — aim for a visual in EVERY section
- CRITICAL VISUAL DIVERSITY: Use at LEAST 3 different visual_type values. Do NOT suggest more than 2 of the same type. Mix flowcharts, bar_charts, comparison, concept_diagrams, architecture_diagrams, infographics, etc.
- Visual suggestions should be NEW diagrams, not copies of existing figures
- For fiction/narrative: use concept_diagram for character relationships, infographic for plot timeline, comparison for contrasting themes
- ORIGINAL FIGURES: Only include a figure in essential_figure_ids if it is a HIGH-QUALITY, self-explanatory chart, graph, or architecture diagram that would genuinely enhance the poster. When in doubt, prefer a freshly generated visual.
- section_categories maps each included section to its poster role. Use "content" for narrative/general sections that don't fit other categories."""

SECTION_ANALYSIS_SYSTEM = (
    "You are an expert at distilling any document into poster-ready content. "
    "You decide whether each section is best presented as flowing prose, "
    "structured bullets, or a mix — matching the style of professional informational posters. "
    "Each bullet must be self-contained, specific with facts, names, quotes, or data, and "
    "under 10 words. Keep text very concise — posters need short, punchy text. "
    "You MUST respond with valid JSON only."
)

SECTION_ANALYSIS_USER = """Extract poster-ready content from this section.

GLOBAL CONTEXT:
- Title: {poster_title}
- Core message: {key_contribution}
- Headline element: {headline_result}
- Domain/genre: {paper_domain}
- This section's importance: {section_importance}
- This section's poster role: {section_category}

SECTION: {section_title} (type: {section_type})
SECTION TEXT:
{section_text}

Respond with this exact JSON:
{{
  "poster_section_title": "<short punchy title for the poster panel, max 4 words>",
  "content_type": "<one of: prose, bullets, mixed>",
  "lead_paragraph": "<1-2 sentence flowing paragraph with **bold** key terms — only for prose/mixed types, empty string for bullets type>",
  "bullets": [
    "<bullet 1, max 12 words>",
    "<bullet 2>",
    "<bullet 3>"
  ],
  "sub_headers": [
    {{"after_bullet_index": 1, "text": "<bold sub-header label, max 6 words>"}}
  ],
  "key_message": "<one sentence takeaway>",
  "provenance": [
    {{"bullet_index": 0, "source_text_span": "<exact supporting quote>"}}
  ],
  "recommended_figure_ids": ["<fig_id>"]
}}

Rules:

CONTENT TYPE SELECTION (critical for layout variety):
- "prose": Use for background, introduction, motivation, challenge, conclusion, setting, context, character introduction, or narrative framing sections. Write a strong lead_paragraph (1-2 sentences with **bold** on key terms — method names, character names, key quotes, or defining facts). Include 0-2 supporting bullets.
- "bullets": Use for results, contributions, key findings, events, key facts, or core takeaways. Write 3-4 specific bullets. lead_paragraph must be "".
- "mixed": Use for methodology, architecture, experiments, plot development, or sections that combine narrative with specifics. Write a lead_paragraph (1 sentence) followed by 2-3 bullets with specifics.

BULLET RULES:
- Each bullet MUST contain a specific fact: a number/metric, a name, a direct quote fragment, a concrete event, a comparison, or a causal claim
- Vague bullets are FORBIDDEN — be specific and concrete
- For research: first bullet of results panels should be the headline quantitative finding with **bold** number
- For fiction/narrative: bullets should capture specific plot events, character actions, or key moments with named entities
- For news: bullets should capture specific facts — who, what, when, where, numerical figures
- For business: bullets should cite specific metrics, decisions, or outcomes
- Each bullet MUST be under 10 words, use action verbs
- Use **bold** on the most important term, metric, character name, or keyword in each bullet
- poster_section_title should be a clean label — e.g., "The Challenge", "Key Results", "The Transformation", "Setting"
- Do not include citations or references
- Each bullet must convey UNIQUE information — never repeat what other sections cover

LEAD PARAGRAPH RULES:
- Must name the specific method, character, event, policy, or concept this section is about
- Use **bold** markdown for the most important term or phrase
- Must read like the opening sentence of a feature article about this section
- MAXIMUM 1 sentence, keep very concise — space on posters is extremely limited

SUB-HEADERS:
- ONLY add if the section has 2+ genuinely distinct sub-topics. Most sections should have EMPTY sub_headers []. Max 1 per panel."""


POSTER_TITLE_PROMPT = """Generate an academic conference poster title banner in an ultra-wide landscape format.

Style: {mood} aesthetic. {background_description}.

Content to render (ALL of these MUST appear in the final image):
- Title: "{title}"
- Authors: "{authors}"
- Affiliation: "{affiliation}"
{headline_instruction}

Layout rules:
- This must appear as a full-width poster banner, not a small title card.
- Use bold {title_font} for the title and keep it clearly larger than authors.
- Title should be moderate and compact: about 8-11% of banner height and up to 82% of banner width.
- Prefer a wide horizontal title composition across the center; do not keep it as a narrow block.
- AUTHORS LINE IS MANDATORY: Render authors directly below title, much smaller and lighter, about 26-32% of title size.
- Affiliation line below authors in a smaller sans-serif, about 70-80% of author size.
- Keep the full text group compact in the central horizontal band so it survives wide-banner cropping.
- If title is long, split into two balanced lines; do not compress letters.
- Center text block horizontally with tight, consistent line spacing.
- Keep line gaps small (roughly 2-3% of banner height) and avoid large vertical gaps.
- Leave safe padding from edges; no clipping.
- Use a clean print-style aesthetic: smooth background, no flashy effects or decorative clipart.
- Do not invent logos, badges, or watermark text.
- If authors string already contains "et al.", keep it exactly as provided.
{headline_render}

CRITICAL: Do not overlap text. Do not let any text get cut off. Keep all text well inside the banner bounds. The authors line MUST be visible and fully readable. Prioritize readability over large font size. Use smaller text if needed."""

POSTER_PANEL_PROMPT = """Generate a single panel for an academic research poster.

Style: {mood}, clean and professional. Flat white background panel with a {header_style} at the top containing the section title in white text. No outer card border, no drop shadow, and no rounded card frame.

Section title: "{section_title}"

Bullets (render exactly, one line each):
{bullets_formatted}

Layout rules:
- Use a clean sans-serif font for all text.
- Keep typography consistent with other panels in the same poster.
- Header is large and bold; bullets are compact but readable.
- Header bar should be consistent and not oversized (about 12-14% of panel height).
- Use a consistent header title text size across panels; avoid panel-to-panel variation.
- Use a consistent bullet/body text size across panels; keep line spacing uniform.
- Keep margins of ~5-6% and no overlap.
- Fill the panel fully and minimize empty white space.
- {space_instruction}

{icon_instruction}
{corner_style}. {style_reference_instruction}"""

VISUAL_GENERATION_PROMPT = """Generate a professional visual for an information poster.

Subject: {concept}
Description: {description}
Type: {visual_type}
Content domain: {content_domain}

{data_instruction}
{extreme_ratio_instruction}

{methodology_context}

Style requirements:
- {mood} illustration on {background_instruction}
- Use {color_description} color palette with color-coded components
- Clean vector-style illustration — avoid photographic textures or stock imagery
- Avoid large blank margins; crop tightly to useful content
- Bold, readable sans-serif labels on all components (large enough to read at poster scale)
- Fill the entire image area — the visual should use most of the available space
- Use high contrast between text and background for readability

{domain_specific_rules}

VISUAL TYPE SPECIFIC RULES:
- For bar_chart/line_chart: Use actual numbers and metrics from the content. Label axes clearly. Highlight the key value in a distinct accent color.
- For comparison: Use a clear side-by-side or table format. Show actual values or contrasting attributes.
- For concept_diagram: Show relationships between concepts with spatial layout. Use icons or symbols, not just boxes.
- For architecture_diagram: Show the system or structure with clear data flow, named components.
- For flowchart/pipeline: Keep it horizontal, use distinct colors per stage, label arrows.
- For infographic: Use large bold key facts, icons, and short labels. Color-coded sections with strong visual hierarchy.
- For scene_illustration: Render a clean, evocative scene relevant to the content. Use flat illustration style, minimal detail, strong silhouettes and color.

CRITICAL BACKGROUND RULE: The visual background MUST exactly match the poster panel background so it blends seamlessly. No visible edges, no borders, no frames, no drop shadows. The image should look like it was drawn directly on the poster surface.

The visual must be self-explanatory and faithful to the content. Make it specific to THIS document's actual content — not a generic template."""

VISUAL_TYPE_INSTRUCTIONS = {
    "flowchart": "Show the process as connected steps flowing left-to-right or top-to-bottom with bold directional arrows between stages. Each stage in a color-filled rounded rectangle with a short bold label. Use different fill colors for input, processing, and output stages. Add small annotations on arrows describing transformations.",
    "architecture_diagram": "Show the system architecture as stacked or connected components with clear hierarchy. Use color-filled rounded rectangles for modules, bold directional arrows for data flow, and clear labels inside each component. Group related components with subtle background shading. Label all connections. Show input on the left and output on the right.",
    "bar_chart": "Show a bar chart comparing the values. Use distinctly colored bars with clear bold axis labels and value annotations on top of each bar. Add a subtle grid. Use the paper's actual metric names as labels.",
    "line_chart": "Show a line chart with clearly marked data points as circles, bold axis labels, smooth trend lines, and a clear legend. Use distinct colors per series. Add a subtle background grid.",
    "pipeline": "Show the processing pipeline as a horizontal chain of color-coded stages connected by bold arrows. Each stage as a distinct colored rounded rectangle with a bold label inside. Add small descriptions or icons within each stage. Show data transformations on the arrows.",
    "comparison": "Show a side-by-side comparison with clear color-coded columns. Use green for the proposed method and gray/red for the baseline. Include specific metric values and a clear visual indicator of improvement (arrows, checkmarks).",
    "concept_diagram": "Show the concept as a clean schematic with color-filled labeled components, clear relationship arrows, and annotations. Use a logical spatial layout that mirrors the conceptual hierarchy. Group related elements with subtle background boxes.",
    "table": "Show a clean data table with alternating row shading, bold colored headers, properly aligned columns, and the best values highlighted in bold or with a colored background.",
    "infographic": "Show key statistics and findings in a visually engaging infographic style with large bold numbers, icons, and short labels. Use color-coded sections and visual hierarchy to guide the eye from most to least important findings.",
    "matrix": "Show a matrix or grid layout comparing multiple methods across multiple metrics. Use color-coded cells (green=best, red=worst) with values inside. Clear row and column headers.",
}


# ══════════════════════════════════════════════════════════════
# Design systems — 7 light + 3 dark presets
# ══════════════════════════════════════════════════════════════

DESIGN_SYSTEMS: dict[str, DesignSystem] = {
    # ── Light presets ─────────────────────────────────────────
    "steel_blue": DesignSystem(
        name="steel_blue",
        is_dark=False,
        white_banner=True,
        tokens=DesignToken(
            primary="#9FB6C3",
            primary_dark="#2C3E50",
            accent="#9FB6C3",
            background="#FFFFFF",
            surface="#FFFFFF",
            text_primary="#1A1A1A",
            text_secondary="#555555",
            text_on_primary="#FFFFFF",
            header_bar="#9FB6C3",
            border="#E8EDF2",
        ),
        title_font_style="bold clean sans-serif",
        heading_font_style="bold clean sans-serif",
        body_font_style="regular readable sans-serif",
        corner_radius="subtle rounded corners",
        panel_style="clean white card with soft shadow",
        mood_keywords=["professional", "clean", "academic", "readable"],
    ),
    "periwinkle": DesignSystem(
        name="periwinkle",
        is_dark=False,
        tokens=DesignToken(
            primary="#9B7EDE",
            primary_dark="#37197C",
            accent="#37197C",
            background="#FAFAFF",
            surface="#FFFFFF",
            text_primary="#1C1828",
            text_secondary="#5A5470",
            text_on_primary="#0C0C0C",
            header_bar="#9B7EDE",
            border="#E4DFF2",
        ),
        title_font_style="bold modern serif",
        heading_font_style="bold clean sans-serif",
        body_font_style="regular readable sans-serif",
        corner_radius="subtle rounded corners",
        panel_style="clean white card with soft shadow",
        mood_keywords=["elegant", "refined", "academic", "sophisticated"],
    ),
    "aquamarine": DesignSystem(
        name="aquamarine",
        is_dark=False,
        white_banner=True,
        tokens=DesignToken(
            primary="#47E5BC",
            primary_dark="#065B44",
            accent="#065B44",
            background="#F4FBF8",
            surface="#FFFFFF",
            text_primary="#0E2A22",
            text_secondary="#4A6E62",
            text_on_primary="#0B1F18",
            header_bar="#47E5BC",
            border="#D4EDE6",
        ),
        title_font_style="bold clean sans-serif",
        heading_font_style="bold fresh sans-serif",
        body_font_style="regular clear sans-serif",
        corner_radius="rounded corners",
        panel_style="crisp white card with cool shadow",
        mood_keywords=["fresh", "clean", "professional", "natural"],
    ),
    "ruby": DesignSystem(
        name="ruby",
        is_dark=False,
        tokens=DesignToken(
            primary="#B10F2E",
            primary_dark="#B10F2E",
            accent="#ED4A68",
            background="#FFFAFA",
            surface="#FFFFFF",
            text_primary="#1A1216",
            text_secondary="#5A4A4E",
            text_on_primary="#FFFFFF",
            header_bar="#B10F2E",
            border="#ECD8DC",
        ),
        title_font_style="bold condensed sans-serif",
        heading_font_style="bold sharp sans-serif",
        body_font_style="regular neutral sans-serif",
        corner_radius="subtle rounded corners",
        panel_style="clean white card with subtle border",
        mood_keywords=["bold", "striking", "professional", "high-contrast"],
    ),
    "apricot": DesignSystem(
        name="apricot",
        is_dark=False,
        tokens=DesignToken(
            primary="#FFCF99",
            primary_dark="#AF712A",
            accent="#FFCF99",
            background="#FFFCF6",
            surface="#FFFFFF",
            text_primary="#2A1E10",
            text_secondary="#6B5840",
            text_on_primary="#FFFFFF",
            header_bar="#AF712A",
            border="#E8DCC8",
        ),
        title_font_style="bold elegant serif",
        heading_font_style="bold warm sans-serif",
        body_font_style="regular warm sans-serif",
        corner_radius="gently rounded corners",
        panel_style="warm white card with soft shadow",
        mood_keywords=["warm", "elegant", "professional", "inviting"],
    ),
    "blush": DesignSystem(
        name="blush",
        is_dark=False,
        tokens=DesignToken(
            primary="#FFC2E2",
            primary_dark="#F3618F",
            accent="#F3618F",
            background="#FFF8FB",
            surface="#FFFFFF",
            text_primary="#1A1018",
            text_secondary="#6A5462",
            text_on_primary="#0C0C0C",
            header_bar="#F3618F",
            border="#F0D8E4",
        ),
        title_font_style="bold modern serif",
        heading_font_style="bold clean sans-serif",
        body_font_style="regular readable sans-serif",
        corner_radius="rounded corners",
        panel_style="clean white card with soft shadow",
        mood_keywords=["elegant", "vibrant", "professional", "polished"],
    ),
    "onyx": DesignSystem(
        name="onyx",
        is_dark=False,
        tokens=DesignToken(
            primary="#0C0F0A",
            primary_dark="#1A1D18",
            accent="#CFCFCF",
            background="#F5F5F5",
            surface="#FFFFFF",
            text_primary="#0C0F0A",
            text_secondary="#4F4F4F",
            text_on_primary="#FFFFFF",
            header_bar="#0C0F0A",
            border="#CFCFCF",
        ),
        title_font_style="bold geometric sans-serif",
        heading_font_style="bold condensed sans-serif",
        body_font_style="regular neutral sans-serif",
        corner_radius="sharp corners",
        panel_style="minimal card with subtle border",
        mood_keywords=["minimal", "sharp", "monochrome", "high-contrast"],
    ),
    "arctic": DesignSystem(
        name="arctic",
        is_dark=False,
        white_banner=True,
        tokens=DesignToken(
            primary="#5B8DB8",
            primary_dark="#1A3A5C",
            accent="#5B8DB8",
            background="#F7F9FB",
            surface="#FFFFFF",
            text_primary="#0D1B2A",
            text_secondary="#4A6070",
            text_on_primary="#FFFFFF",
            header_bar="#5B8DB8",
            border="#DDE5ED",
        ),
        title_font_style="bold clean sans-serif",
        heading_font_style="bold clean sans-serif",
        body_font_style="regular readable sans-serif",
        corner_radius="subtle rounded corners",
        panel_style="clean white card with soft shadow",
        mood_keywords=["precise", "clean", "technical", "authoritative"],
    ),
    "forest": DesignSystem(
        name="forest",
        is_dark=False,
        tokens=DesignToken(
            primary="#3A7D4B",
            primary_dark="#1B4A2A",
            accent="#3A7D4B",
            background="#F5F8F5",
            surface="#FFFFFF",
            text_primary="#0F2017",
            text_secondary="#3D5C45",
            text_on_primary="#FFFFFF",
            header_bar="#3A7D4B",
            border="#D0DDD2",
        ),
        title_font_style="bold clean sans-serif",
        heading_font_style="bold clean sans-serif",
        body_font_style="regular readable sans-serif",
        corner_radius="subtle rounded corners",
        panel_style="clean white card with soft shadow",
        mood_keywords=["natural", "confident", "academic", "grounded"],
    ),

    "academic_gray": DesignSystem(
        name="academic_gray",
        is_dark=False,
        white_banner=True,
        tokens=DesignToken(
            primary="#2C2C2C",
            primary_dark="#1A1A1A",
            accent="#4A4A4A",
            background="#EBEBEB",
            surface="#FFFFFF",
            text_primary="#1A1A1A",
            text_secondary="#555555",
            text_on_primary="#FFFFFF",
            header_bar="#2C2C2C",
            border="#C8C8C8",
            shadow="#00000012",
        ),
        title_font_style="bold clean sans-serif",
        heading_font_style="bold clean sans-serif",
        body_font_style="regular readable sans-serif",
        corner_radius="subtle rounded corners",
        panel_style="white card with gray border",
        mood_keywords=["academic", "clean", "professional", "traditional"],
    ),

    # ── Dark presets ──────────────────────────────────────────
    "frozen_lake": DesignSystem(
        name="frozen_lake",
        is_dark=True,
        tokens=DesignToken(
            primary="#75D1FF",
            primary_dark="#000000",
            accent="#75D1FF",
            background="#000000",
            surface="#0F1116",
            text_primary="#E2E4EA",
            text_secondary="#8A92A2",
            text_on_primary="#000000",
            header_bar="#75D1FF",
            border="#20242C",
        ),
        title_font_style="bold geometric sans-serif",
        heading_font_style="bold modern sans-serif",
        body_font_style="regular clean sans-serif",
        corner_radius="rounded corners",
        panel_style="dark card with subtle glow",
        mood_keywords=["sleek", "modern", "cool-toned", "premium"],
    ),
    "neon_ice": DesignSystem(
        name="neon_ice",
        is_dark=True,
        tokens=DesignToken(
            primary="#26FFE6",
            primary_dark="#000000",
            accent="#26FFE6",
            background="#000000",
            surface="#0F1116",
            text_primary="#E2E4EA",
            text_secondary="#8A92A2",
            text_on_primary="#000000",
            header_bar="#26FFE6",
            border="#20242C",
        ),
        title_font_style="bold geometric sans-serif",
        heading_font_style="bold modern sans-serif",
        body_font_style="regular clean sans-serif",
        corner_radius="rounded corners",
        panel_style="dark card with neon accent",
        mood_keywords=["futuristic", "tech", "cool-toned", "vibrant"],
    ),
    "bubblegum_dark": DesignSystem(
        name="bubblegum_dark",
        is_dark=True,
        tokens=DesignToken(
            primary="#E85D75",
            primary_dark="#000000",
            accent="#E85D75",
            background="#000000",
            surface="#0F1116",
            text_primary="#E2E4EA",
            text_secondary="#8A92A2",
            text_on_primary="#000000",
            header_bar="#E85D75",
            border="#20242C",
        ),
        title_font_style="bold modern sans-serif",
        heading_font_style="bold clean sans-serif",
        body_font_style="regular readable sans-serif",
        corner_radius="rounded corners",
        panel_style="dark card with warm accent",
        mood_keywords=["bold", "modern", "warm-accent", "premium"],
    ),
}

# Preset groupings for random selection
LIGHT_PRESETS = [
    "steel_blue", "periwinkle", "aquamarine", "ruby",
    "apricot", "blush", "onyx", "arctic", "forest", "academic_gray",
]
DARK_PRESETS = ["frozen_lake", "neon_ice", "bubblegum_dark"]

ALL_PRESET_NAMES = LIGHT_PRESETS + DARK_PRESETS

# Domain → theme mapping for --style auto
_DOMAIN_THEME_MAP: dict[str, str] = {
    "computer vision": "arctic",
    "cv": "arctic",
    "nlp": "periwinkle",
    "natural language processing": "periwinkle",
    "rl": "forest",
    "reinforcement learning": "forest",
    "systems": "onyx",
    "biology": "aquamarine",
    "neuroscience": "aquamarine",
}


def _domain_to_theme(paper_domain: str) -> str:
    """Map a paper domain string to a theme preset name, or '' if no match."""
    if not paper_domain:
        return ""
    domain_lower = paper_domain.strip().lower()
    for key, theme in _DOMAIN_THEME_MAP.items():
        if key in domain_lower:
            return theme
    return ""


# Backward-compat aliases for old --style names
_STYLE_ALIASES: dict[str, str] = {
    "paper_clean": "steel_blue",
    "academic": "steel_blue",
    "modern": "steel_blue",
    "earth": "aquamarine",
    "ocean": "aquamarine",
    "slate": "onyx",
    "warm": "apricot",
}


# ══════════════════════════════════════════════════════════════
# Theme resolution
# ══════════════════════════════════════════════════════════════

def resolve_theme(
    config: "PosterConfig",
    suggested_style: str | None = None,
    paper_domain: str = "",
) -> DesignSystem:
    """Resolve the user's --style flag into a concrete DesignSystem.

    Handles: "light" (random), "dark" (random), "custom" (from hex codes),
    "auto" (from analysis), specific preset names, and backward-compat aliases.
    If style is "auto", cross-references LLM suggestion with domain mapping:
      - If both agree → use that theme.
      - If they disagree → prefer LLM suggestion, log both.
      - If only domain mapping matches → use domain mapping.
    When poster_mode is "casual", overlays creative mood keywords for visuals.
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)

    style = config.style.strip().lower()
    if style in ("auto", "suggested"):
        llm_suggested = (suggested_style or "").strip().lower()
        # Normalize LLM suggestion through aliases
        llm_suggested = _STYLE_ALIASES.get(llm_suggested, llm_suggested)
        llm_valid = llm_suggested in DESIGN_SYSTEMS

        domain_theme = _domain_to_theme(paper_domain)

        if llm_valid and domain_theme:
            if llm_suggested == domain_theme:
                _log.debug("Theme: LLM and domain both suggest '%s'.", llm_suggested)
            else:
                _log.debug(
                    "Theme: LLM suggests '%s', domain mapping suggests '%s'. Using LLM.",
                    llm_suggested, domain_theme,
                )
            style = llm_suggested
        elif llm_valid:
            style = llm_suggested
        elif domain_theme:
            style = domain_theme
        else:
            style = "light"

    if style == "light":
        name = random.choice(LIGHT_PRESETS)
        design = DESIGN_SYSTEMS[name]
    elif style == "dark":
        name = random.choice(DARK_PRESETS)
        design = DESIGN_SYSTEMS[name]
    elif style == "custom":
        design = _build_custom_design(config)
    elif style in DESIGN_SYSTEMS:
        design = DESIGN_SYSTEMS[style]
    elif style in _STYLE_ALIASES:
        design = DESIGN_SYSTEMS[_STYLE_ALIASES[style]]
    else:
        design = DESIGN_SYSTEMS[random.choice(LIGHT_PRESETS)]

    # Casual mode: overlay creative/illustrative mood for visual generation
    if getattr(config, "poster_mode", "professional") == "casual":
        design = design.model_copy(update={
            "mood_keywords": design.mood_keywords[:2] + [
                "creative", "illustrative", "eye-catching", "vibrant"
            ],
            "panel_style": "rounded card with accent border, asymmetrical layout",
        })

    return design


def _build_custom_design(config: "PosterConfig") -> DesignSystem:
    """Build a DesignSystem from user-supplied hex codes."""
    bg = config.custom_bg or "#FFFFFF"
    header_color = config.custom_header_color or "#2C3E50"
    subheader_color = config.custom_subheader_color or header_color
    text_color = config.custom_text_color or ""
    header_text_color = config.custom_header_text_color or ""

    # Detect dark mode from background luminance
    is_dark = _perceived_brightness(bg) < 0.35

    # Auto-detect text colors if not provided
    if not text_color:
        text_color = "#E2E4EA" if is_dark else "#1A1A1A"
    if not header_text_color:
        header_text_color = (
            "#FFFFFF" if _perceived_brightness(subheader_color) < 0.45 else "#0C0C0C"
        )

    # Derive secondary/surface/border colors
    if is_dark:
        surface = _shift_color(bg, 16)
        border = _shift_color(bg, 32)
        text_secondary = _shift_color(text_color, -50)
        title_banner = _shift_color(bg, 8)
    else:
        surface = "#FFFFFF"
        border = _shift_color(bg, -25)
        text_secondary = _shift_color(text_color, 60)
        # Title banner: darken the header color
        title_banner = _shift_color(header_color, -40)

    return DesignSystem(
        name="custom",
        is_dark=is_dark,
        tokens=DesignToken(
            primary=header_color,
            primary_dark=title_banner,
            accent=subheader_color,
            background=bg,
            surface=surface,
            text_primary=text_color,
            text_secondary=text_secondary,
            text_on_primary=header_text_color,
            header_bar=subheader_color,
            border=border,
        ),
        mood_keywords=["custom", "professional", "clean"],
    )


# ── Color utility helpers ─────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) < 6:
        h = h.ljust(6, "0")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{max(0, min(255, r)):02x}{max(0, min(255, g)):02x}{max(0, min(255, b)):02x}"


def _shift_color(hex_color: str, amount: int) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return _rgb_to_hex(r + amount, g + amount, b + amount)


def _perceived_brightness(hex_color: str) -> float:
    r, g, b = _hex_to_rgb(hex_color)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


# ══════════════════════════════════════════════════════════════
# Public helpers (used by other stages)
# ══════════════════════════════════════════════════════════════

def get_design_system(name: str) -> DesignSystem:
    """Get a design system by name, with backward-compat alias support."""
    if name in DESIGN_SYSTEMS:
        return DESIGN_SYSTEMS[name]
    alias = _STYLE_ALIASES.get(name)
    if alias:
        return DESIGN_SYSTEMS[alias]
    return DESIGN_SYSTEMS["steel_blue"]


def get_style(name: str) -> dict[str, str]:
    ds = get_design_system(name)
    return {
        "bg_color": ds.tokens.primary,
        "accent_color": ds.tokens.accent,
        "text_color": ds.tokens.text_primary,
        "header_text_color": ds.tokens.text_on_primary,
        "background": ds.tokens.background,
        "surface": ds.tokens.surface,
        "border": ds.tokens.border,
        "header_bar": ds.tokens.header_bar,
    }


def format_bullets_for_prompt(bullets: list[str]) -> str:
    return "\n".join(f'- "{b}"' for b in bullets)


def build_title_prompt(
    title: str,
    authors: str,
    affiliation: str,
    headline_result: str,
    design: DesignSystem,
) -> str:
    mood = ", ".join(design.mood_keywords[:3])
    bg_desc = (
        "Clean conference-style white/light-neutral banner with strong readability "
        "for print"
    )
    headline_instruction = ""
    headline_render = ""
    if headline_result:
        headline_instruction = f'- Highlight: "{headline_result}"'
        headline_render = (
            "The highlight line can appear as a compact single line, but do not increase "
            "overall title typography size or spacing."
        )
    return POSTER_TITLE_PROMPT.format(
        mood=mood,
        background_description=bg_desc,
        title=title,
        authors=authors,
        affiliation=affiliation,
        headline_instruction=headline_instruction,
        title_font=design.title_font_style,
        headline_render=headline_render,
    )


def build_panel_prompt(
    section_title: str,
    bullets: list[str],
    design: DesignSystem,
    reserve_visual_space: bool = False,
    has_style_reference: bool = False,
) -> str:
    mood = ", ".join(design.mood_keywords[:3])
    bullets_formatted = format_bullets_for_prompt(bullets)
    if design.name == "paper_clean":
        header_style = "solid dark navy-blue header bar"
    else:
        header_style = f"colored header bar in a {mood} tone"
    if reserve_visual_space:
        space_instruction = (
            "Place all text strictly in the top 55-60% of the panel. "
            "Leave the bottom 40-45% as clean white space reserved for a visual. "
            "Do not place any text in the reserved lower region."
        )
    else:
        space_instruction = (
            "Use the full panel height. Distribute bullets evenly with comfortable spacing. "
            "Fill the space and avoid large empty areas."
        )
    icon_instruction = "Do not add decorative icons or ornaments."
    style_ref = ""
    if has_style_reference:
        style_ref = (
            "Match the visual style, color palette, and typography "
            "of the provided reference image for consistency."
        )
    return POSTER_PANEL_PROMPT.format(
        mood=mood,
        header_style=header_style,
        section_title=section_title,
        bullets_formatted=bullets_formatted,
        space_instruction=space_instruction,
        icon_instruction=icon_instruction,
        corner_style=design.corner_radius,
        style_reference_instruction=style_ref,
    )


_NARRATIVE_DOMAINS = frozenset({
    "fiction", "narrative", "literary", "novel", "story", "memoir",
    "creative writing", "short story", "poetry",
})
_NEWS_DOMAINS = frozenset({
    "news", "journalism", "investigative", "report", "article",
})
_BUSINESS_DOMAINS = frozenset({
    "business", "finance", "corporate", "policy", "economics",
})


def _domain_specific_visual_rules(content_domain: str) -> str:
    d = content_domain.lower()
    if any(k in d for k in _NARRATIVE_DOMAINS):
        return (
            "CONTENT DOMAIN: Fiction / Narrative\n"
            "- Visuals should be evocative illustrations, NOT abstract data diagrams\n"
            "- For concept_diagram: draw a character relationship map or thematic web using"
            " named characters, silhouettes, and connecting lines showing relationships\n"
            "- For infographic: create a visual plot timeline or 'key moments' panel with"
            " icons and short scene descriptions\n"
            "- For scene_illustration: render a flat-style scene from the story — use"
            " strong silhouettes, atmospheric color, and minimal text\n"
            "- Use flat, editorial illustration style (similar to book cover art or"
            " New Yorker illustrations) — clean shapes, strong color blocks, no photorealism\n"
            "- Include the actual character names, place names, and key objects from the story"
        )
    if any(k in d for k in _NEWS_DOMAINS):
        return (
            "CONTENT DOMAIN: News / Journalism\n"
            "- Visuals should be informative infographics, NOT abstract academic diagrams\n"
            "- For infographic: use bold statistics, icons, and short factual labels\n"
            "- For concept_diagram: show key actors, organizations, or causal chains\n"
            "- For comparison: show before/after, then/now, or competing sides\n"
            "- Use journalistic infographic style — clear, bold, fact-forward"
        )
    if any(k in d for k in _BUSINESS_DOMAINS):
        return (
            "CONTENT DOMAIN: Business / Finance\n"
            "- Visuals should be clean business infographics or data charts\n"
            "- For bar_chart/line_chart: use actual figures from the document\n"
            "- For infographic: bold KPIs, icons, and concise labels\n"
            "- Use professional business presentation style"
        )
    return (
        "CONTENT DOMAIN: Research / Technical\n"
        "- Use precise, scientifically accurate diagrams matching top conference poster quality\n"
        "- Include concrete names: model names, method names, variable names — NOT placeholders\n"
        "- Use rounded rectangles for modules, directional arrows with labels for data flow\n"
        "- Add annotations where relevant (loss functions, dimensions, parameters)"
    )


def build_visual_prompt(
    concept: str,
    description: str,
    visual_type: str,
    data_points: list[str],
    design: DesignSystem,
    methodology_summary: str = "",
    content_domain: str = "",
) -> str:
    mood = ", ".join(design.mood_keywords[:3])
    type_instruction = VISUAL_TYPE_INSTRUCTIONS.get(visual_type, VISUAL_TYPE_INSTRUCTIONS["concept_diagram"])
    data_instruction = ""
    extreme_ratio_instruction = ""
    if data_points:
        formatted = "\n".join(f"- {dp}" for dp in data_points)
        data_instruction = f"Data to include:\n{formatted}\n\n{type_instruction}"
    else:
        data_instruction = type_instruction
    if visual_type == "bar_chart":
        ratio = _estimate_ratio_from_points(data_points)
        if ratio and ratio > 100:
            extreme_ratio_instruction = (
                "For extreme ratios (>100x), do NOT use proportional heights. "
                "Use similar-height bars with bold labels like '1x' and '3000x' "
                "and an annotation arrow to show the difference."
            )
    methodology_context = ""
    if methodology_summary:
        methodology_context = f"Paper context (use real names from this): {methodology_summary}"
    color_desc = f"{mood}-toned"

    # Dark mode: generate diagrams on dark background to blend with panels
    if design.is_dark:
        bg_instruction = (
            f"dark background (use exact color {design.tokens.surface}). "
            f"The diagram will be embedded on a {design.tokens.surface} panel surface, "
            "so use that exact background color to blend seamlessly. "
            "Use light-colored text labels (#E0E0E0 or white) for readability. "
            "No borders, no frames, no drop shadows around the diagram."
        )
    else:
        poster_bg = design.tokens.background
        if poster_bg.upper() == "#FFFFFF":
            bg_instruction = (
                "pure white (#FFFFFF) background. The diagram will be placed on a white poster, "
                "so the background MUST be pure white with NO gray tint, NO off-white, NO borders "
                "or frames around the diagram. The image should blend seamlessly into a white surface."
            )
        else:
            bg_instruction = (
                f"background color {poster_bg} to match the poster. "
                f"The diagram will be embedded directly on a {poster_bg} poster surface, "
                "so use that exact background color — no borders, no frames, no drop shadows."
            )

    domain_rules = _domain_specific_visual_rules(content_domain)

    return VISUAL_GENERATION_PROMPT.format(
        concept=concept,
        description=description,
        visual_type=visual_type,
        content_domain=content_domain or "general",
        data_instruction=data_instruction,
        extreme_ratio_instruction=extreme_ratio_instruction,
        methodology_context=methodology_context,
        mood=mood,
        color_description=color_desc,
        background_instruction=bg_instruction,
        domain_specific_rules=domain_rules,
    )


def _estimate_ratio_from_points(data_points: list[str]) -> float | None:
    import re
    values = []
    for dp in data_points:
        if not isinstance(dp, str):
            continue
        matches = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", dp.lower())
        for m in matches:
            try:
                values.append(float(m))
            except ValueError:
                continue
    if len(values) < 2:
        return None
    min_v = min(values)
    max_v = max(values)
    if min_v <= 0:
        return None
    return max_v / min_v


def build_figure_instruction(
    has_figure: bool,
    figure_caption: str = "",
    **_kwargs: object,
) -> str:
    if not has_figure:
        return ""
    caption_line = ""
    if figure_caption:
        caption_line = f'Include a small caption below the figure: "{figure_caption}"'
    return (
        "Include the provided reference image integrated into the lower "
        "portion of the panel. Size it to complement the text above without "
        f"overwhelming the bullet points. {caption_line}"
    )
