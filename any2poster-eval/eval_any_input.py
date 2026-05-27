"""End-to-end evaluation pipeline for any input format.

Accepts any input that any2poster supports (PDF, PPTX, DOCX, LaTeX, Notebook,
URL, YouTube video, local video, Markdown, plain text), generates a poster,
auto-generates PaperQuiz QA from the source content, and runs both PaperQuiz
and VLM-as-Judge evaluation — no Paper2Poster dataset required.

Usage examples
--------------
  # LaTeX source
  python scripts/eval_any_input.py paper.tex

  # YouTube talk
  python scripts/eval_any_input.py https://www.youtube.com/watch?v=XXXX

  # ArXiv HTML abstract
  python scripts/eval_any_input.py https://arxiv.org/abs/2301.00001

  # PPTX slides
  python scripts/eval_any_input.py slides.pptx --method-name pptx

  # Already-generated poster (skip generation, run eval only)
  python scripts/eval_any_input.py paper.tex --skip-generate \\
      --poster-png results/paper/any2poster.png \\
      --poster-html results/paper/any2poster.html

Output
------
  <out-dir>/<paper-id>/
    poster.pdf / poster.html / poster.png   ← generated poster
    source_text.md                          ← extracted source text (QA source)
    qa.json                                 ← auto-generated QA
    paperquiz.json                          ← PaperQuiz scores
    vlm_judge.json                          ← VLM-as-Judge scores
    summary.json                            ← aggregated summary
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    text = re.sub(r"[^\w]+", "_", text.strip().lower())
    return text.strip("_")[:80] or "document"


def _derive_paper_id(input_path: str, override: str) -> str:
    if override:
        return _slugify(override)
    if input_path.startswith(("http://", "https://")):
        # Strip scheme and use the path/query as a slug
        clean = re.sub(r"https?://", "", input_path)
        return _slugify(clean)
    return _slugify(Path(input_path).stem)


def _print(msg: str) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ──────────────────────────────────────────────────────────────────────────────

def _parse_source(input_path: str, parse_dir: Path) -> str:
    """Parse the input and return the full extracted text for QA generation."""
    from any2poster.parsers import parse_document

    _print(f"[parse] Extracting content from: {input_path}")
    parse_dir.mkdir(parents=True, exist_ok=True)
    doc = parse_document(input_path, parse_dir)

    if not doc.sections:
        raise ValueError("Parser returned no sections — check the input file.")

    full_text = doc.get_full_text()
    _print(
        f"[parse] Done — {doc.source_format} | {len(doc.sections)} sections "
        f"| {doc.total_words} words"
    )
    return full_text


def _run_any2poster(input_path: str, out_dir: Path, args: argparse.Namespace) -> Path:
    """Run the full any2poster pipeline and return the output PDF path."""
    from any2poster.models import PosterConfig
    from any2poster.pipeline import run_pipeline

    output_pdf = out_dir / "poster.pdf"
    config = PosterConfig(
        input_path=str(Path(input_path).resolve()) if not input_path.startswith(("http://", "https://")) else input_path,
        output_path=str(output_pdf),
        resume=args.resume,
        enable_feedback=args.feedback,
        feedback_max_iters=args.feedback_iters,
        feedback_model=args.feedback_model or "",
    )
    _print("[generate] Running any2poster pipeline…")
    output_path = run_pipeline(config)
    _print(f"[generate] Poster saved: {output_path}")
    return Path(output_path)


def _generate_qa(source_text: str, qa_path: Path, provider) -> dict:
    """Generate verbatim + interpretive QA from source text, with caching."""
    from any2poster.eval.paperquiz import generate_qa_from_markdown

    if qa_path.exists():
        _print(f"[qa] Reusing cached QA: {qa_path}")
        return json.loads(qa_path.read_text(encoding="utf-8"))

    _print("[qa] Generating PaperQuiz QA from source text (2 LLM calls)…")

    # generate_qa_from_markdown returns {"verbatim": [QAItem...], "interpretive": [QAItem...]}
    # We need to serialise QAItems to plain dicts for caching.
    qa_items = generate_qa_from_markdown(source_text, provider=provider)

    # Convert to the Paper2Poster-compatible JSON format for caching
    def _items_to_dict(items) -> dict:
        out = {}
        for item in items:
            out[item.qid] = {
                "aspect": item.aspect or "",
                "question": item.question,
                "options": item.options,
                "answer": item.answer,
            }
        return out

    qa_serialised = {
        "detail": {"questions": _items_to_dict(qa_items["verbatim"])},
        "understanding": {"questions": _items_to_dict(qa_items["interpretive"])},
    }

    # Add answers into the questions dict (Paper2Poster format has them separate,
    # but load_paper2poster_qa also reads from the question object directly)
    for section in ("detail", "understanding"):
        for qid, q in qa_serialised[section]["questions"].items():
            q["answer"] = q.get("answer", "")

    qa_path.write_text(json.dumps(qa_serialised, indent=2), encoding="utf-8")
    _print(
        f"[qa] Generated {len(qa_items['verbatim'])} verbatim + "
        f"{len(qa_items['interpretive'])} interpretive questions → {qa_path}"
    )
    return qa_serialised


def _run_paperquiz(
    poster_png: Path,
    poster_html: Path,
    qa_obj: dict,
    out_path: Path,
    args: argparse.Namespace,
    provider,
) -> dict:
    from any2poster.eval.paperquiz import (
        evaluate_paperquiz,
        extract_poster_text_from_html,
        load_paper2poster_qa,
    )

    if out_path.exists():
        _print(f"[paperquiz] Reusing cached results: {out_path}")
        return json.loads(out_path.read_text(encoding="utf-8"))

    _print("[paperquiz] Running PaperQuiz evaluation…")
    qa_items = load_paper2poster_qa(qa_obj)
    poster_text = extract_poster_text_from_html(poster_html.read_text(encoding="utf-8"))
    model_list = [m.strip() for m in args.paperquiz_models.split(",") if m.strip()]

    results = evaluate_paperquiz(
        poster_image=poster_png,
        poster_text=poster_text,
        qa_items=qa_items,
        provider=provider,
        model_names=model_list,
        median_words=args.median_words,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _print(f"[paperquiz] Done → {out_path}")
    return results


def _run_vlm_judge(
    poster_png: Path,
    out_path: Path,
    args: argparse.Namespace,
    provider,
) -> dict:
    from any2poster.eval.vlm_judge import evaluate_vlm_judge

    if out_path.exists():
        _print(f"[vlm_judge] Reusing cached results: {out_path}")
        return json.loads(out_path.read_text(encoding="utf-8"))

    _print("[vlm_judge] Running VLM-as-Judge evaluation (6 criteria)…")
    results = evaluate_vlm_judge(
        poster_image=poster_png,
        provider=provider,
        model_name=args.judge_model or None,
        out_path=out_path,
    )
    _print(f"[vlm_judge] Done → {out_path}")
    return results


def _print_summary(pq_results: dict, vj_results: dict, out_path: Path) -> None:
    summary: dict = {}

    # PaperQuiz summary
    avg = pq_results.get("avg", {})
    summary["paperquiz"] = {
        "verbatim_accuracy": round(avg.get("verbatim_raw", 0.0), 4),
        "interpretive_accuracy": round(avg.get("interpretive_raw", 0.0), 4),
        "overall_accuracy": round(avg.get("overall_accuracy", 0.0), 4),
        "overall_aug": round(avg.get("overall_aug", 0.0), 4),
    }

    # VLM Judge summary
    summary["vlm_judge"] = {
        "element_quality": vj_results.get("element_quality", {}).get("score"),
        "layout_balance": vj_results.get("layout_balance", {}).get("score"),
        "engagement": vj_results.get("engagement", {}).get("score"),
        "clarity": vj_results.get("clarity", {}).get("score"),
        "content_completeness": vj_results.get("content_completeness", {}).get("score"),
        "logical_flow": vj_results.get("logical_flow", {}).get("score"),
        "aesthetic_score": vj_results.get("aesthetic_score"),
        "information_score": vj_results.get("information_score"),
        "overall": vj_results.get("overall"),
    }

    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _print("\n" + "=" * 60)
    _print("EVAL SUMMARY")
    _print("=" * 60)
    _print("\nPaperQuiz:")
    _print(f"  Verbatim accuracy    : {summary['paperquiz']['verbatim_accuracy']:.1%}")
    _print(f"  Interpretive accuracy: {summary['paperquiz']['interpretive_accuracy']:.1%}")
    _print(f"  Overall accuracy     : {summary['paperquiz']['overall_accuracy']:.1%}")
    _print(f"  Overall (aug.)       : {summary['paperquiz']['overall_aug']:.4f}")
    _print("\nVLM-as-Judge (1–5):")
    for key in ("element_quality", "layout_balance", "engagement",
                "clarity", "content_completeness", "logical_flow"):
        val = summary["vlm_judge"].get(key)
        label = key.replace("_", " ").title()
        _print(f"  {label:<22}: {val}")
    _print(f"  {'Aesthetic (avg)':<22}: {summary['vlm_judge']['aesthetic_score']}")
    _print(f"  {'Information (avg)':<22}: {summary['vlm_judge']['information_score']}")
    _print(f"  {'Overall (avg)':<22}: {summary['vlm_judge']['overall']}")
    _print("=" * 60)
    _print(f"\nFull results: {out_path.parent}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate any2poster on any input format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    parser.add_argument("input", help="File path or URL to convert to a poster")
    parser.add_argument("--paper-id", default="", help="Custom slug for output directory naming")
    parser.add_argument("--method-name", default="any2poster", help="Label for this run in results")
    parser.add_argument("--out-dir", default="any_eval_results", help="Root output directory")

    # Generation control
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip any2poster generation; use --poster-png and --poster-html instead")
    parser.add_argument("--poster-png", default="",
                        help="Path to existing poster PNG (used with --skip-generate)")
    parser.add_argument("--poster-html", default="",
                        help="Path to existing poster HTML (used with --skip-generate)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume any2poster from checkpoints if they exist")
    parser.add_argument("--feedback", action="store_true")
    parser.add_argument("--feedback-iters", type=int, default=2)
    parser.add_argument("--feedback-model", default="")

    # QA control
    parser.add_argument("--skip-qa-gen", action="store_true",
                        help="Skip QA generation and reuse qa.json if it exists")
    parser.add_argument("--qa-json", default="",
                        help="Path to an existing QA JSON file (skips generation entirely)")

    # Eval models
    parser.add_argument("--paperquiz-models", default="openai/gpt-4o",
                        help="Comma-separated VLM model names for PaperQuiz answering")
    parser.add_argument("--judge-model", default="openai/gpt-4o",
                        help="VLM model for VLM-as-Judge scoring")
    parser.add_argument("--median-words", type=int, default=774,
                        help="Median poster word count for density-augmented score")

    # Eval toggles
    parser.add_argument("--skip-paperquiz", action="store_true")
    parser.add_argument("--skip-vlm-judge", action="store_true")

    args = parser.parse_args()

    # ── Paths ──────────────────────────────────────────────────────────────────
    paper_id = _derive_paper_id(args.input, args.paper_id)
    run_dir = Path(args.out_dir) / paper_id / args.method_name
    run_dir.mkdir(parents=True, exist_ok=True)

    _print(f"\n[init] Paper ID : {paper_id}")
    _print(f"[init] Method   : {args.method_name}")
    _print(f"[init] Output   : {run_dir}\n")

    from any2poster.providers.openrouter import OpenRouterProvider
    provider = OpenRouterProvider()

    # ── Step 1: Extract source text for QA generation ──────────────────────────
    source_text_path = run_dir / "source_text.md"

    if source_text_path.exists():
        _print(f"[parse] Reusing cached source text: {source_text_path}")
        source_text = source_text_path.read_text(encoding="utf-8")
    else:
        try:
            source_text = _parse_source(args.input, run_dir / "_parse_tmp")
            source_text_path.write_text(source_text, encoding="utf-8")
        except Exception as exc:
            _print(f"[parse] ERROR: {exc}")
            traceback.print_exc()
            sys.exit(1)

    # ── Step 2: Generate poster ────────────────────────────────────────────────
    if args.skip_generate:
        if not args.poster_png or not args.poster_html:
            _print("[generate] --skip-generate requires --poster-png and --poster-html")
            sys.exit(1)
        poster_pdf = Path(args.poster_png).with_suffix(".pdf")
        poster_png = Path(args.poster_png)
        poster_html = Path(args.poster_html)
        _print(f"[generate] Using existing poster: {poster_png}")
    else:
        try:
            poster_pdf = _run_any2poster(args.input, run_dir, args)
            poster_png = poster_pdf.with_suffix(".png")
            poster_html = poster_pdf.with_suffix(".html")
        except Exception as exc:
            _print(f"[generate] ERROR: {exc}")
            traceback.print_exc()
            sys.exit(1)

    if not poster_png.exists():
        _print(f"[generate] ERROR: Poster PNG not found at {poster_png}")
        sys.exit(1)
    if not poster_html.exists():
        _print(f"[generate] ERROR: Poster HTML not found at {poster_html}")
        sys.exit(1)

    # ── Step 3: QA generation ─────────────────────────────────────────────────
    pq_results: dict = {}
    vj_results: dict = {}

    if not args.skip_paperquiz:
        qa_path = run_dir / "qa.json"

        if args.qa_json:
            # Use an externally supplied QA file (e.g. from Paper2Poster dataset)
            qa_obj = json.loads(Path(args.qa_json).read_text(encoding="utf-8"))
            _print(f"[qa] Using supplied QA file: {args.qa_json}")
        elif args.skip_qa_gen and qa_path.exists():
            qa_obj = json.loads(qa_path.read_text(encoding="utf-8"))
            _print(f"[qa] Reusing cached QA: {qa_path}")
        else:
            try:
                qa_obj = _generate_qa(source_text, qa_path, provider)
            except Exception as exc:
                _print(f"[qa] ERROR generating QA: {exc}")
                traceback.print_exc()
                _print("[qa] Skipping PaperQuiz evaluation.")
                args.skip_paperquiz = True

    # ── Step 4: PaperQuiz evaluation ──────────────────────────────────────────
    if not args.skip_paperquiz:
        try:
            pq_results = _run_paperquiz(
                poster_png=poster_png,
                poster_html=poster_html,
                qa_obj=qa_obj,
                out_path=run_dir / "paperquiz.json",
                args=args,
                provider=provider,
            )
        except Exception as exc:
            _print(f"[paperquiz] ERROR: {exc}")
            traceback.print_exc()

    # ── Step 5: VLM-as-Judge evaluation ───────────────────────────────────────
    if not args.skip_vlm_judge:
        try:
            vj_results = _run_vlm_judge(
                poster_png=poster_png,
                out_path=run_dir / "vlm_judge.json",
                args=args,
                provider=provider,
            )
        except Exception as exc:
            _print(f"[vlm_judge] ERROR: {exc}")
            traceback.print_exc()

    # ── Step 6: Summary ───────────────────────────────────────────────────────
    if pq_results or vj_results:
        _print_summary(pq_results, vj_results, run_dir / "summary.json")
    else:
        _print("\n[done] No eval results to summarise (both evals were skipped or failed).")

    _print(f"\n[done] All outputs in: {run_dir}\n")


if __name__ == "__main__":
    main()
