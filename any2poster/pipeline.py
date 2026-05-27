# Orchestrates the end to end poster pipeline with caching and stage sequencing.

from __future__ import annotations

from pathlib import Path
import logging
import re
from typing import TYPE_CHECKING

from any2poster.checkpoint import (
    get_figures_dir,
    get_panels_dir,
    load_checkpoint,
    save_checkpoint,
    should_run_stage,
)
from any2poster.models import (
    AnalyzedContent,
    ChunkedDocument,
    GenerationResult,
    ParsedDocument,
    PosterConfig,
    PosterPlan,
)

if TYPE_CHECKING:
    from any2poster.providers.openrouter import OpenRouterProvider


def run_pipeline(config: PosterConfig) -> Path:
    logger = logging.getLogger(__name__)
    provider = _create_provider(config)

    parsed = _run_parse(config)
    chunked = _run_chunk(config, parsed)
    analyzed = _run_analyze(config, parsed, provider, chunked)
    if config.generate_visuals:
        try:
            from any2poster.data_extractor import (
                enrich_visual_prompts_with_data,
                validate_visual_data,
            )
            analyzed = validate_visual_data(analyzed, parsed, provider)
            analyzed = enrich_visual_prompts_with_data(analyzed)
        except Exception as exc:
            logger.warning("Visual data validation failed; proceeding without enrichment: %s", exc)
            if config.debug:
                logger.exception("Visual data validation failed")
    plan = _run_plan(config, analyzed, provider)
    panels_dir = get_panels_dir(config)
    figures_dir = get_figures_dir(config)
    generation = _run_generate(config, plan, analyzed, provider, panels_dir, figures_dir)
    output_path = _run_compile(config, plan, generation, analyzed)
    if getattr(config, "enable_feedback", False):
        from any2poster.stages.feedback_stage import run_feedback_stage
        html_path = Path(output_path).with_suffix(".html")
        output_path = run_feedback_stage(
            html_path=html_path,
            output_path=Path(output_path),
            plan=plan,
            config=config,
            provider=provider,
        )

    return output_path


def _create_provider(config: PosterConfig) -> "OpenRouterProvider":
    from any2poster.providers.openrouter import OpenRouterProvider

    return OpenRouterProvider(
        llm_model=config.llm_model,
        image_model=config.image_model,
        vision_model=config.vision_model,
    )


def _run_parse(config: PosterConfig) -> ParsedDocument:
    if not should_run_stage(config, "parse"):
        cached = load_checkpoint(config, "parse", ParsedDocument)
        if cached:
            return cached

    from any2poster.stages.parse_stage import run_parse_stage

    parsed = run_parse_stage(config)
    save_checkpoint(config, "parse", parsed)
    return parsed


def _run_analyze(
    config: PosterConfig,
    parsed: ParsedDocument,
    provider: "OpenRouterProvider",
    chunked: ChunkedDocument,
) -> AnalyzedContent:
    if not should_run_stage(config, "analyze"):
        cached = load_checkpoint(config, "analyze", AnalyzedContent)
        if cached:
            return cached

    from any2poster.stages.analyze_stage import run_analyze_stage

    analyzed = run_analyze_stage(parsed, config, provider, chunked)
    save_checkpoint(config, "analyze", analyzed)
    return analyzed


def _run_chunk(
    config: PosterConfig,
    parsed: ParsedDocument,
) -> ChunkedDocument:
    if not should_run_stage(config, "chunk"):
        cached = load_checkpoint(config, "chunk", ChunkedDocument)
        if cached:
            return cached

    from any2poster.stages.chunk_stage import run_chunk_stage

    chunked = run_chunk_stage(parsed)
    save_checkpoint(config, "chunk", chunked)
    return chunked


def _run_plan(
    config: PosterConfig,
    analyzed: AnalyzedContent,
    provider: "OpenRouterProvider",
) -> PosterPlan:
    if not should_run_stage(config, "plan"):
        cached = load_checkpoint(config, "plan", PosterPlan)
        if cached:
            return cached

    from any2poster.stages.plan_stage import run_plan_stage

    plan = run_plan_stage(analyzed, config, provider)
    save_checkpoint(config, "plan", plan)
    return plan


def _run_generate(
    config: PosterConfig,
    plan: PosterPlan,
    analyzed: AnalyzedContent,
    provider: "OpenRouterProvider",
    panels_dir: Path,
    figures_dir: Path,
) -> GenerationResult:
    if not should_run_stage(config, "generate"):
        cached = load_checkpoint(config, "generate", GenerationResult)
        if cached:
            return cached

    from any2poster.stages.generate_stage import run_generate_stage

    generation = run_generate_stage(
        plan=plan,
        analyzed=analyzed,
        config=config,
        provider=provider,
        output_dir=panels_dir,
        figures_dir=figures_dir,
    )
    save_checkpoint(config, "generate", generation)
    return generation


def _run_compile(
    config: PosterConfig,
    plan: PosterPlan,
    generation: GenerationResult,
    analyzed: "AnalyzedContent | None" = None,
) -> Path:
    from any2poster.stages.compile_stage import run_compile_stage

    output_path = _unique_output_path(Path(config.output_path))
    return run_compile_stage(plan, generation, config, output_path, analyzed)


def _unique_output_path(base_path: Path) -> Path:
    """Return a unique output path.

    For `poster.pdf`, use `poster1.pdf`, `poster2.pdf`, ... to keep naming clean.
    For other stems, fallback to `<stem>_1.<ext>` style.
    """
    if not base_path.exists():
        return base_path
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    m = re.match(r"^(.*?poster)(\d*)$", stem, flags=re.IGNORECASE)
    if m:
        base_no_num = m.group(1)
        start = 1
        if m.group(2):
            try:
                start = int(m.group(2)) + 1
            except ValueError:
                start = 1
        counter = start
        while True:
            candidate = parent / f"{base_no_num}{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1
