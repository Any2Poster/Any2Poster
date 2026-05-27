# CLI entry point that builds config and runs the poster pipeline.

from __future__ import annotations

import sys
from pathlib import Path

import click

_ALL_STYLES = [
    "light", "dark", "custom", "auto",
    "steel_blue", "arctic", "periwinkle", "aquamarine", "forest", "ruby", "apricot", "blush", "onyx", "academic_gray",
    "frozen_lake", "neon_ice", "bubblegum_dark",
    "paper_clean", "academic", "modern", "earth", "ocean", "slate", "warm",
]


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    default="",
    help="Output file path (PDF or PNG). Default: <paper>poster.pdf",
)
@click.option("--style", default="light", type=click.Choice(_ALL_STYLES, case_sensitive=False),
              help="Color theme: light (random), dark (random), custom, auto (LLM-suggested), or a preset name")
@click.option("--mode", default="professional", type=click.Choice(
    ["professional", "casual"], case_sensitive=False
), help="Poster mode: professional (academic grid) or casual (creative layout)")
@click.option("--width", default=48.0, help="Poster width in inches")
@click.option("--height", default=36.0, help="Poster height in inches")
@click.option("--dpi", default=250, help="Output resolution")
@click.option("--resume/--no-resume", default=False, help="Resume from checkpoint")
@click.option("--debug/--no-debug", default=False, help="Debug mode with verbose output")
@click.option("--generate-visuals/--no-generate-visuals", default=True, help="Generate AI diagrams")
@click.option("--style-chain/--no-style-chain", default=True, help="Use first panel as style reference")
@click.option("--llm-model", default="anthropic/claude-sonnet-4", help="LLM model ID")
@click.option("--image-model", default="google/gemini-3-pro-image-preview", help="Image model ID")
@click.option("--vision-model", default="", help="Vision model ID for quality checks (OpenRouter)")
@click.option("--max-retries", default=3, help="Max retries per image generation")
@click.option("--quality-check/--no-quality-check", default=True, help="Run VLM quality checks on generated visuals")
@click.option("--quality-retries", default=1, help="Extra re-generation attempts if quality checks fail")
@click.option("--feedback/--no-feedback", default=False, help="Enable VLM feedback loop on rendered panels")
@click.option("--feedback-iters", default=2, help="Max iterations for VLM feedback loop")
@click.option("--feedback-model", default="", help="Vision model for feedback loop (defaults to --vision-model)")
@click.option("--column-gap", default=0.0, help="Override column gap (inches). 0 = default")
# Custom color flags (used with --style custom)
@click.option("--bg-color", default="", help="Custom background hex color (e.g. '#FFFFFF')")
@click.option("--header-color", default="", help="Custom title banner hex color")
@click.option("--subheader-color", default="", help="Custom panel header bar hex color")
@click.option("--header-text-color", default="", help="Custom header text hex color")
@click.option("--text-color", default="", help="Custom body text hex color")
@click.option("--logo-left", default="", type=click.Path(),
              help="Path to left banner logo image (e.g. conference logo). Supports PNG/JPG/WEBP.")
@click.option("--logo-right", default="", type=click.Path(),
              help="Path to right banner logo image (e.g. institution logo). Supports PNG/JPG/WEBP.")
@click.option("--auto-logos/--no-auto-logos", default=False,
              help="Auto-generate venue and institution logos from paper metadata")
@click.option("--footer/--no-footer", default=False,
              help="Show footer with source name and venue (hidden by default)")
@click.option("--interactive/--no-interactive", default=False,
              help="Interactive prompts for mode/style selection")
def main(
    input_path: str,
    output: str,
    style: str,
    mode: str,
    width: float,
    height: float,
    dpi: int,
    resume: bool,
    debug: bool,
    generate_visuals: bool,
    style_chain: bool,
    llm_model: str,
    image_model: str,
    vision_model: str,
    max_retries: int,
    quality_check: bool,
    quality_retries: int,
    feedback: bool,
    feedback_iters: int,
    feedback_model: str,
    column_gap: float,
    bg_color: str,
    header_color: str,
    subheader_color: str,
    header_text_color: str,
    text_color: str,
    logo_left: str,
    logo_right: str,
    auto_logos: bool,
    footer: bool,
    interactive: bool,
) -> None:
    _validate_environment()

    if not output or not output.strip():
        output = _default_output_for_input(input_path)

    if interactive:
        mode = _prompt_choice(
            "Choose poster mode (p=professional, c=casual)",
            {"p": "professional", "c": "casual"},
        )
        style = _prompt_choice(
            "Choose color style (l=light, d=dark, a=auto, c=custom, or type preset name)",
            {"l": "light", "d": "dark", "a": "auto", "c": "custom"},
            allow_full=True,
        )
        if style == "custom":
            bg_color = _prompt_hex("Background hex (e.g. #FFFFFF)")
            header_color = _prompt_hex("Title banner hex (e.g. #2C3E50)")
            subheader_color = _prompt_hex("Panel header bar hex (optional, Enter to skip)", allow_empty=True)
            header_text_color = _prompt_hex("Header text hex (optional, Enter to skip)", allow_empty=True)
            text_color = _prompt_hex("Body text hex (optional, Enter to skip)", allow_empty=True)
        click.echo(f"Selected: mode={mode}, style={style}, output={output}")

    # Validate custom mode has required colors
    if style.lower() == "custom" and not bg_color and not header_color:
        click.echo(
            "Error: --style custom requires at least --bg-color and --header-color.",
            err=True,
        )
        sys.exit(1)

    from any2poster.models import PosterConfig

    config = PosterConfig(
        input_path=str(Path(input_path).resolve()),
        output_path=output,
        style=style.lower(),
        poster_mode=mode.lower(),
        poster_width=width,
        poster_height=height,
        dpi=dpi,
        resume=resume,
        debug=debug,
        generate_visuals=generate_visuals,
        style_chain=style_chain,
        llm_model=llm_model,
        image_model=image_model,
        vision_model=vision_model.strip(),
        max_retries=max_retries,
        quality_check=quality_check,
        quality_check_retries=quality_retries,
        enable_feedback=feedback,
        feedback_max_iters=feedback_iters,
        feedback_model=feedback_model.strip(),
        column_gutter_in=column_gap,
        custom_bg=bg_color.strip().strip("'\""),
        custom_header_color=header_color.strip().strip("'\""),
        custom_subheader_color=subheader_color.strip().strip("'\""),
        custom_header_text_color=header_text_color.strip().strip("'\""),
        custom_text_color=text_color.strip().strip("'\""),
        logo_left=str(Path(logo_left).resolve()) if logo_left else "",
        logo_right=str(Path(logo_right).resolve()) if logo_right else "",
        auto_logos=auto_logos,
        show_footer=footer,
    )

    from any2poster.pipeline import run_pipeline

    try:
        result = run_pipeline(config)
        click.echo(f"Poster saved to: {result}")
    except KeyboardInterrupt:
        click.echo("\nInterrupted. Use --resume to continue from last checkpoint.")
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        if debug:
            raise
        sys.exit(1)


def _validate_environment() -> None:
    import os

    try:
        from dotenv import load_dotenv
        if Path(".env").exists():
            load_dotenv()
    except ImportError:
        pass

    if not os.getenv("OPENROUTER_API_KEY"):
        click.echo(
            "OPENROUTER_API_KEY not set. "
            "Add it to .env or export it in your shell.",
            err=True,
        )
        sys.exit(1)


def _default_output_for_input(input_path: str) -> str:
    import re
    stem = Path(input_path).stem.strip()
    slug = re.sub(r"[^\w]+", "_", stem).strip("_").lower()
    if len(slug) > 40:
        slug = slug[:40].rstrip("_")
    if not slug:
        slug = "poster"
    return f"{slug}_poster.pdf"


def _prompt_choice(
    prompt: str,
    short_map: dict[str, str],
    *,
    allow_full: bool = False,
) -> str:
    while True:
        resp = click.prompt(prompt, default="", show_default=False)
        if resp is None:
            continue
        val = str(resp).strip().lower()
        if not val:
            continue
        if val in short_map:
            return short_map[val]
        if allow_full and val in _ALL_STYLES:
            return val
        click.echo("Invalid choice. Try again.")


def _prompt_hex(prompt: str, *, allow_empty: bool = False) -> str:
    import re
    pattern = re.compile(r"^#?[0-9a-fA-F]{6}$")
    while True:
        resp = click.prompt(prompt, default="", show_default=False)
        if resp is None:
            continue
        val = str(resp).strip()
        if allow_empty and val == "":
            return ""
        if pattern.match(val):
            return val if val.startswith("#") else f"#{val}"
        click.echo("Invalid hex. Use 6-digit hex like #AABBCC.")


if __name__ == "__main__":
    main()
