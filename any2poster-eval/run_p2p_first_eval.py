"""Download one Paper2Poster sample and (optionally) run any2poster + evals.

Usage (download only):
  python scripts/run_p2p_first_eval.py --index 0

Run full pipeline + evals:
  python scripts/run_p2p_first_eval.py --index 0 --run-all
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _slugify(text: str) -> str:
    text = re.sub(r"[^\w]+", "_", text.strip().lower())
    return text.strip("_") or "paper"


def _download(url: str, out_path: Path) -> None:
    import requests

    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://openreview.net/",
    }
    resp = requests.get(url, headers=headers, timeout=120, allow_redirects=True)
    if resp.status_code == 403 and "openreview.net" in url:
        alt = url
        if "download=1" not in alt:
            alt = alt + ("&" if "?" in alt else "?") + "download=1"
        resp = requests.get(alt, headers=headers, timeout=120, allow_redirects=True)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)


def _load_row(index: int, split: str):
    from datasets import load_dataset

    ds = load_dataset("Paper2Poster/Paper2Poster", split=split)
    return ds[index]


def _resolve_paper_id(row: dict, override: str | None) -> str:
    if override:
        return _slugify(override)
    title = str(row.get("title", "")).strip()
    conf = str(row.get("conference", "")).strip()
    year = str(row.get("year", "")).strip()
    return _slugify("_".join([t for t in [title, conf, year] if t]))


def _run_any2poster(paper_pdf: Path, out_dir: Path, args) -> Path:
    from any2poster.models import PosterConfig
    from any2poster.pipeline import run_pipeline

    output_pdf = out_dir / "any2poster.pdf"
    config = PosterConfig(
        input_path=str(paper_pdf.resolve()),
        output_path=str(output_pdf),
        resume=bool(args.resume),
        enable_feedback=args.feedback,
        feedback_max_iters=args.feedback_iters,
        feedback_model=args.feedback_model or "",
    )
    return run_pipeline(config)


def _run_evals(
    paper_id: str,
    poster_pdf: Path,
    poster_png: Path,
    poster_html: Path,
    gt_png: Path,
    qa_path: Path,
    args,
) -> None:
    from any2poster.eval.stats import evaluate_stats
    from any2poster.eval.paperquiz import (
        evaluate_paperquiz,
        extract_poster_text_from_html,
        load_paper2poster_qa,
    )
    from any2poster.eval.vlm_judge import evaluate_vlm_judge
    from any2poster.providers.openrouter import OpenRouterProvider

    out_root = Path(args.out_root)
    md_root = Path(args.md_root)

    evaluate_stats(
        paper_id=paper_id,
        method_name="any2poster",
        gen_poster_png=poster_png,
        gen_poster_html=poster_html,
        gt_poster_png=gt_png,
        out_root=out_root,
        md_root=md_root,
        enable_vlm_ppl=not args.disable_vlm_ppl,
        enable_textual_ppl=not args.disable_textual_ppl,
    )

    qa_obj = json.loads(qa_path.read_text(encoding="utf-8"))
    qa_items = load_paper2poster_qa(qa_obj)
    poster_text = extract_poster_text_from_html(poster_html.read_text(encoding="utf-8"))

    provider = OpenRouterProvider()
    model_list = [m.strip() for m in args.paperquiz_models.split(",") if m.strip()]
    pq_res = evaluate_paperquiz(
        poster_image=poster_png,
        poster_text=poster_text,
        qa_items=qa_items,
        provider=provider,
        model_names=model_list,
        median_words=args.median_words,
    )
    pq_out = out_root / paper_id / "any2poster" / "paperquiz.json"
    pq_out.parent.mkdir(parents=True, exist_ok=True)
    pq_out.write_text(json.dumps(pq_res, indent=2), encoding="utf-8")

    vj_out = out_root / paper_id / "any2poster" / "vlm_judge.json"
    evaluate_vlm_judge(
        poster_image=poster_png,
        provider=provider,
        model_name=args.judge_model or None,
        out_path=vj_out,
    )

    print(f"[ok] Eval outputs written to: {out_root / paper_id / 'any2poster'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run first Paper2Poster sample eval.")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--paper-id", default="")
    parser.add_argument("--out-dir", default="paper2poster_sample")
    parser.add_argument("--run-any2poster", action="store_true")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse any2poster checkpoints")
    parser.add_argument("--feedback", action="store_true")
    parser.add_argument("--feedback-iters", type=int, default=2)
    parser.add_argument("--feedback-model", default="")
    parser.add_argument("--paperquiz-models", default="openai/gpt-4o")
    parser.add_argument("--judge-model", default="openai/gpt-4o")
    parser.add_argument("--median-words", type=int, default=774)
    parser.add_argument("--out-root", default="eval_results")
    parser.add_argument("--md-root", default="eval_poster_markdown")
    parser.add_argument("--disable-vlm-ppl", action="store_true")
    parser.add_argument("--disable-textual-ppl", action="store_true")
    args = parser.parse_args()

    if args.run_all:
        args.run_any2poster = True
        args.run_eval = True

    row = _load_row(args.index, args.split)
    paper_id = _resolve_paper_id(row, args.paper_id)

    out_dir = Path(args.out_dir) / paper_id
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_url = row.get("paper_url")
    image_url = row.get("image_url")
    qa_raw = row.get("qa")

    if not paper_url or not image_url or not qa_raw:
        raise RuntimeError("Row is missing paper_url, image_url, or qa.")

    paper_pdf = out_dir / "paper.pdf"
    gt_png = out_dir / "gt.png"
    qa_path = out_dir / "qa.json"

    _download(str(paper_url), paper_pdf)
    _download(str(image_url), gt_png)

    qa_obj = qa_raw
    if isinstance(qa_raw, str):
        qa_obj = json.loads(qa_raw)
    qa_path.write_text(json.dumps(qa_obj, indent=2), encoding="utf-8")

    meta = {
        "paper_id": paper_id,
        "title": row.get("title"),
        "conference": row.get("conference"),
        "year": row.get("year"),
        "paper_url": paper_url,
        "image_url": image_url,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[ok] Downloaded sample to: {out_dir}")

    poster_pdf = out_dir / "any2poster.pdf"
    if args.run_any2poster:
        output_path = _run_any2poster(paper_pdf, out_dir, args)
        poster_pdf = Path(output_path)
        print(f"[ok] any2poster output: {poster_pdf}")

    if args.run_eval:
        poster_png = poster_pdf.with_suffix(".png")
        poster_html = poster_pdf.with_suffix(".html")
        if not (poster_png.exists() and poster_html.exists()):
            raise RuntimeError("Poster PNG/HTML not found. Run any2poster first.")
        _run_evals(
            paper_id=paper_id,
            poster_pdf=poster_pdf,
            poster_png=poster_png,
            poster_html=poster_html,
            gt_png=gt_png,
            qa_path=qa_path,
            args=args,
        )

    print("[next] If you want full eval, re-run with --run-all.")


if __name__ == "__main__":
    main()
