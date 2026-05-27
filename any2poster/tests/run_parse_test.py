# tests/run_parse_test.py
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from any2poster.models import PosterConfig
from any2poster.stages.parse_stage import run_parse_stage


SAMPLE_MARKDOWN = """\
---
title: Sample Paper
authors: [Alice Example, Bob Example]
---

# Introduction
This is a short sample to validate the parsing stage.

## Methods
We used a simple approach to demonstrate parsing.

![diagram](https://example.com/diagram.png)

## Results
| Metric | Value |
|-------:|:-----|
| Accuracy | 0.95 |
"""


def _write_sample_markdown() -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="any2poster_parse_test_"))
    sample_path = temp_dir / "sample.md"
    sample_path.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
    return sample_path


def _safe_model_dump_json(obj, indent: int = 2) -> str:
    if hasattr(obj, "model_dump_json"):
        return obj.model_dump_json(indent=indent)  # Pydantic v2
    return obj.json(indent=indent)  # Pydantic v1 fallback


def main() -> int:
    parser = argparse.ArgumentParser(description="Run parse stage against a document.")
    parser.add_argument(
        "--input",
        default=os.environ.get("ANY2POSTER_TEST_INPUT"),
        help="Input path or URL (defaults to ANY2POSTER_TEST_INPUT).",
    )
    parser.add_argument(
        "--output",
        default="output/parse_test.pdf",
        help="Output PDF path (used for output directory).",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip writing parsed_preview.json.",
    )
    args = parser.parse_args()

    if args.input:
        input_path = args.input
        if not input_path.startswith(("http://", "https://")) and not Path(input_path).exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
    else:
        input_path = str(_write_sample_markdown())
        print("No input provided; using sample markdown:", input_path)

    cfg = PosterConfig(
        input_path=input_path,
        output_path=args.output,
        debug=True,
    )

    try:
        parsed = run_parse_stage(cfg)
    except ImportError as exc:
        raise SystemExit(f"{exc}\nTip: install missing dependencies for this file type.") from exc

    # Quick summary
    print("TITLE:", parsed.title)
    print("AUTHORS:", parsed.authors)
    print("PAGE COUNT:", parsed.total_pages)
    print("SECTIONS:", len(parsed.sections))
    for s in parsed.sections[:10]:
        print(f"- {s.section_id} | {s.title} | {len(s.content.split())} words")
    print("FIGURES:", len(parsed.figures))
    for f in parsed.figures:
        caption = (f.caption or "")[:120]
        print(f"- {f.figure_id} -> {f.path} | caption: {caption}")

    # Optional: save parsed JSON for inspection
    if not args.no_json:
        out_json = Path(args.output).with_suffix(".parsed_preview.json")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(_safe_model_dump_json(parsed, indent=2), encoding="utf-8")
        print("Saved parsed_preview.json to", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
