# Pydantic models for pipeline data and configuration.

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class SectionType(str, Enum):
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    BACKGROUND = "background"
    METHODS = "methods"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    ACKNOWLEDGMENTS = "acknowledgments"
    OTHER = "other"


class PanelCategory(str, Enum):
    TITLE = "title"
    MOTIVATION = "motivation"
    METHODOLOGY = "methodology"
    ARCHITECTURE = "architecture"
    RESULTS = "results"
    ANALYSIS = "analysis"
    CONCLUSION = "conclusion"
    DATASET = "dataset"
    CONTENT = "content"


class VisualType(str, Enum):
    FLOWCHART = "flowchart"
    ARCHITECTURE_DIAGRAM = "architecture_diagram"
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    TABLE = "table"
    COMPARISON = "comparison"
    PIPELINE = "pipeline"
    CONCEPT_DIAGRAM = "concept_diagram"
    INFOGRAPHIC = "infographic"
    MATRIX = "matrix"
    NONE = "none"


class VisualSuggestion(BaseModel):
    concept: str
    description: str
    visual_type: VisualType = VisualType.CONCEPT_DIAGRAM
    data_points: list[str] = Field(default_factory=list)
    target_panel_id: Optional[str] = None


class ExtractedFigure(BaseModel):
    figure_id: str
    caption: str = ""
    page_number: Optional[int] = None
    path: Optional[Path] = None
    bbox: Optional[list[float]] = None

    @property
    def is_available(self) -> bool:
        return self.path is not None and self.path.exists()


class ExtractedTable(BaseModel):
    table_id: str
    caption: str = ""
    content: str = ""
    page_number: Optional[int] = None


class ExtractedSection(BaseModel):
    section_id: str
    title: str
    section_type: SectionType = SectionType.OTHER
    content: str
    level: int = 1
    page_number: Optional[int] = None
    word_count: int = 0
    figure_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    title: str = ""
    authors: str = ""
    affiliation: str = ""
    abstract: str = ""
    sections: list[ExtractedSection] = Field(default_factory=list)
    figures: list[ExtractedFigure] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
    raw_text: str = ""
    source_path: str = ""
    source_format: str = ""
    total_words: int = 0
    total_pages: Optional[int] = None

    def get_full_text(self, max_tokens: int = 0) -> str:
        parts = []
        if self.title:
            parts.append(self.title)
        if self.authors:
            parts.append(self.authors)
        if self.affiliation:
            parts.append(f"Affiliation: {self.affiliation}")
        if self.abstract:
            parts.append(self.abstract)
        for section in self.sections:
            if section.section_type == SectionType.REFERENCES:
                continue
            parts.append(f"\n## {section.title}\n{section.content}")
        text = "\n\n".join(parts)
        if max_tokens > 0:
            words = text.split()
            if len(words) > max_tokens:
                text = " ".join(words[:max_tokens])
        return text

    def get_section_by_type(self, section_type: SectionType) -> list[ExtractedSection]:
        return [s for s in self.sections if s.section_type == section_type]


class TextChunk(BaseModel):
    chunk_id: str
    section_id: str
    content: str
    token_count: int = 0
    context_before: str = ""
    context_after: str = ""
    figure_ids: list[str] = Field(default_factory=list)


class ChunkedDocument(BaseModel):
    chunks: list[TextChunk] = Field(default_factory=list)
    total_chunks: int = 0


class BulletProvenance(BaseModel):
    source_section_id: str
    source_chunk_id: Optional[str] = None
    source_text_span: str = ""


class GlobalAnalysis(BaseModel):
    poster_title: str = ""
    authors: str = ""
    affiliation: str = ""
    key_contribution: str = ""
    headline_result: str = ""
    summary: str = ""
    narrative_arc: str = ""
    sections_to_include: list[str] = Field(default_factory=list)
    section_importance: dict[str, float] = Field(default_factory=dict)
    section_categories: dict[str, str] = Field(default_factory=dict)
    essential_figure_ids: list[str] = Field(default_factory=list)
    visual_suggestions: list[VisualSuggestion] = Field(default_factory=list)
    methodology_summary: str = ""
    results_summary: str = ""
    paper_domain: str = ""
    suggested_color_theme: str = "paper_clean"
    venue: str = ""


class PanelSubHeader(BaseModel):
    after_bullet_index: int
    text: str


class AnalyzedSection(BaseModel):
    section_id: str
    title: str
    section_type: SectionType = SectionType.OTHER
    panel_category: PanelCategory = PanelCategory.CONTENT
    content_type: str = "bullets"
    lead_paragraph: str = ""
    bullets: list[str] = Field(default_factory=list)
    sub_headers: list[PanelSubHeader] = Field(default_factory=list)
    provenance: list[BulletProvenance] = Field(default_factory=list)
    importance: float = 1.0
    has_figure: bool = False
    figure_ids: list[str] = Field(default_factory=list)
    key_message: str = ""
    visual_suggestion: Optional[VisualSuggestion] = None
    poster_section_number: int = 0


class AnalyzedContent(BaseModel):
    global_analysis: GlobalAnalysis = Field(default_factory=GlobalAnalysis)
    sections: list[AnalyzedSection] = Field(default_factory=list)
    poster_title: str = ""
    poster_authors: str = ""
    poster_key_message: str = ""


class DesignToken(BaseModel):
    primary: str = "#2C5F7C"
    primary_dark: str = "#1A3A4F"
    accent: str = "#4A9B8E"
    background: str = "#F5F0EB"
    surface: str = "#FFFFFF"
    text_primary: str = "#1A1A2E"
    text_secondary: str = "#4A4A6A"
    text_on_primary: str = "#FFFFFF"
    header_bar: str = "#2C5F7C"
    border: str = "#D8D0C8"
    shadow: str = "#00000015"


class DesignSystem(BaseModel):
    name: str = "academic"
    tokens: DesignToken = Field(default_factory=DesignToken)
    title_font_style: str = "bold modern serif"
    heading_font_style: str = "bold clean sans-serif"
    body_font_style: str = "regular readable sans-serif"
    corner_radius: str = "subtle rounded corners"
    panel_style: str = "clean white card with soft shadow"
    mood_keywords: list[str] = Field(default_factory=lambda: [
        "professional", "clean", "academic", "readable"
    ])
    is_dark: bool = False
    white_banner: bool = False  # True → white banner with accent border-bottom instead of colored fill


class PanelPosition(BaseModel):
    x: float
    y: float
    width: float
    height: float


class PanelSpec(BaseModel):
    id: str
    panel_type: str = "content"
    panel_category: PanelCategory = PanelCategory.CONTENT
    title: str = ""
    position: PanelPosition = Field(
        default_factory=lambda: PanelPosition(x=0, y=0, width=0, height=0)
    )
    section_id: Optional[str] = None
    has_figure: bool = False
    figure_ids: list[str] = Field(default_factory=list)
    visual_suggestion: Optional[VisualSuggestion] = None
    weight: float = 1.0
    column: int = 0
    row_in_column: int = 0
    # HTML rendering
    visual_source: str = "none"  # "original" | "generated" | "none"
    layout_mode: str = "standard"  # "standard" | "prose_only" | "bullets_only" | "prose_with_figure" | "side_by_side" | "figure_dominant"
    original_figure_path: Optional[str] = None


class PosterPlan(BaseModel):
    panels: list[PanelSpec] = Field(default_factory=list)
    width_inches: float = 48.0
    height_inches: float = 36.0
    style: str = "academic"
    num_columns: int = 3
    design: DesignSystem = Field(default_factory=DesignSystem)


class GeneratedPanel(BaseModel):
    panel_id: str
    image_path: Path
    prompt_used: str = ""
    generation_attempts: int = 1


class GenerationResult(BaseModel):
    panels: list[GeneratedPanel] = Field(default_factory=list)
    total_generated: int = 0
    failed_panels: list[str] = Field(default_factory=list)
    style_reference_path: Optional[Path] = None
    quality_reports: list["QualityReport"] = Field(default_factory=list)


class QualityReport(BaseModel):
    panel_id: str
    passed: bool = True
    confidence: float = 0.0
    summary: str = ""
    issues: list[str] = Field(default_factory=list)
    suggested_fixes: list[str] = Field(default_factory=list)


class ValidationScore(BaseModel):
    panel_id: str
    text_accuracy: float = 0.0
    expected_text: str = ""
    detected_text: str = ""
    passed: bool = False


class ValidationReport(BaseModel):
    scores: list[ValidationScore] = Field(default_factory=list)
    overall_accuracy: float = 0.0
    panels_passed: int = 0
    panels_failed: int = 0


class PosterConfig(BaseModel):
    input_path: str
    output_path: str = "poster.pdf"
    style: str = "light"
    poster_mode: str = "professional"
    llm_provider: str = "openrouter"
    llm_model: str = "anthropic/claude-sonnet-4"
    image_provider: str = "openrouter"
    image_model: str = "google/gemini-3-pro-image-preview"
    vision_model: str = "openai/gpt-4o-mini"
    poster_width: float = 48.0
    poster_height: float = 36.0
    dpi: int = 250
    max_panels: int = 9
    min_panels: int = 7
    parser: str = "auto"
    checkpoint_dir: str = ".any2poster_cache"
    resume: bool = False
    debug: bool = False
    max_retries: int = 3
    generate_visuals: bool = True
    style_chain: bool = True
    quality_check: bool = True
    quality_check_retries: int = 1
    custom_bg: str = ""
    custom_header_color: str = ""
    custom_subheader_color: str = ""
    custom_header_text_color: str = ""
    custom_text_color: str = ""
    logo_left: str = ""   # path to left banner logo image (e.g. conference logo)
    logo_right: str = ""  # path to right banner logo image (e.g. institution logo)
    auto_logos: bool = False  # auto-generate venue + institution logos from paper metadata
    show_footer: bool = False  # show footer with source name and venue
    # Layout tuning
    column_gutter_in: float = 0.0  # override column gap in inches (0 = default)
    # VLM feedback loop (panel refinement)
    enable_feedback: bool = False
    feedback_max_iters: int = 2
    feedback_model: str = ""
