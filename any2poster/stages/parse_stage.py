# Parses source input into structured sections, figures, and tables.

from pathlib import Path
import logging

from any2poster.models import PosterConfig, ParsedDocument
from any2poster.parsers import parse_document
from any2poster.checkpoint import get_figures_dir


def run_parse_stage(config: PosterConfig) -> ParsedDocument:
    logger = logging.getLogger(__name__)
    figures_dir = get_figures_dir(config)

    parsed = parse_document(
        input_path=config.input_path,
        output_dir=figures_dir,
        parser_override=config.parser,
    )

    if not parsed.sections and not parsed.raw_text:
        raise ValueError(
            f"Could not extract any content from: {config.input_path}\n"
            "Please check that the file is readable and contains text."
        )

    if config.debug:
        fmt = parsed.source_format.upper() if parsed.source_format else "UNKNOWN"
        logger.info("Parsed: %s", fmt)
        logger.info("Title: %s...", parsed.title[:50])
        logger.info("Sections: %d", len(parsed.sections))
        logger.info("Figures: %d", len(parsed.figures))
        logger.info("Tables: %d", len(parsed.tables))
        logger.info("Words: %d", parsed.total_words)

    return parsed
