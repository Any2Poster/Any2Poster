"""Phase 1 (stats) evaluation for any2poster.

Implements Paper2Poster-style "stats" metrics:
  - CLIP_similarity (AltCLIP cosine similarity between GT and generated posters)
  - textual_ppl (Llama-2-7b-hf on poster text)
  - mixtual_ppl (PPL on markdown that includes image refs)
  - visual_relevance (AltCLIP similarity between panel images and panel text)
  - visual_ppl / interleaved_ppl / poster_image_ppl (Qwen2.5-VL via vLLM)
"""

from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import base64
import io
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Sequence, cast

from PIL import Image


# ----------------------------
# Data containers
# ----------------------------


@dataclass
class PanelContent:
    title: str
    text: str
    images: list[Image.Image]


# ----------------------------
# HTML parsing helpers
# ----------------------------


_PANEL_SPLIT_RE = re.compile(r'<div class="panel"[^>]*>', re.IGNORECASE)
_TITLE_RE = re.compile(r'class="panel-title"\s*>\s*(.*?)\s*<', re.IGNORECASE | re.DOTALL)
_LEAD_RE = re.compile(r'class="lead-para"\s*>\s*(.*?)\s*<', re.IGNORECASE | re.DOTALL)
_LI_RE = re.compile(r"<li[^>]*>\s*(.*?)\s*</li>", re.IGNORECASE | re.DOTALL)
_IMG_RE = re.compile(r'<img\s+[^>]*src="([^"]+)"', re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(text: str) -> str:
    return unescape(_TAG_RE.sub("", text)).strip()


def _decode_data_uri(data_uri: str) -> bytes | None:
    if not data_uri.startswith("data:"):
        return None
    try:
        header, b64 = data_uri.split(",", 1)
    except ValueError:
        return None
    if "base64" not in header:
        return None
    try:
        return base64.b64decode(b64)
    except Exception:
        return None


def _load_image_from_src(src: str) -> Image.Image | None:
    data = _decode_data_uri(src)
    if data is not None:
        return Image.open(io.BytesIO(data)).convert("RGB")
    if os.path.isfile(src):
        return Image.open(src).convert("RGB")
    return None


def extract_panels_from_html(html_text: str) -> list[PanelContent]:
    """Extract panel text + images from any2poster HTML posters."""
    if not html_text:
        return []

    chunks = _PANEL_SPLIT_RE.split(html_text)
    panels: list[PanelContent] = []
    for chunk in chunks[1:]:  # first chunk is header
        title_match = _TITLE_RE.search(chunk)
        title = _strip_tags(title_match.group(1)) if title_match else "Section"

        lead = ""
        lead_match = _LEAD_RE.search(chunk)
        if lead_match:
            lead = _strip_tags(lead_match.group(1))

        bullets = []
        for li in _LI_RE.findall(chunk):
            cleaned = _strip_tags(li)
            if cleaned:
                bullets.append(cleaned)

        text_parts = []
        if lead:
            text_parts.append(lead)
        text_parts.extend(bullets)
        panel_text = "\n".join(text_parts).strip()

        images: list[Image.Image] = []
        for src in _IMG_RE.findall(chunk):
            img = _load_image_from_src(src)
            if img is not None:
                images.append(img)

        panels.append(PanelContent(title=title, text=panel_text, images=images))

    return panels


# ----------------------------
# Markdown generation
# ----------------------------


def build_eval_markdown(
    panels: Sequence[PanelContent],
    *,
    out_dir: Path,
    paper_id: str,
) -> tuple[dict[str, dict], str, str]:
    """Write markdown + images for interleaved metrics.

    Returns:
      images_meta: mapping of relative image path -> {"image": PIL.Image, "text": panel_text}
      poster_text: plain text (used for textual PPL)
      mixtual_md: markdown with image refs (used for mixtual PPL)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    poster_text_lines: list[str] = []
    images_meta: dict[str, dict] = {}

    img_index = 0
    for p_idx, panel in enumerate(panels, start=1):
        panel_header = f"## {panel.title}".strip()
        if panel_header:
            lines.append(panel_header)
            poster_text_lines.append(panel_header)

        if panel.text:
            lines.append(panel.text)
            poster_text_lines.append(panel.text)

        for img in panel.images:
            img_index += 1
            rel_path = f"images/{paper_id}_panel{p_idx}_img{img_index}.png"
            abs_path = img_dir / f"{paper_id}_panel{p_idx}_img{img_index}.png"
            img.save(abs_path)
            lines.append(f"![panel-image]({rel_path})")
            images_meta[rel_path] = {"image": img, "text": panel.text}

        lines.append("")

    poster_text = "\n".join(poster_text_lines).strip()
    mixtual_md = "\n".join(lines).strip()

    with (out_dir / f"{paper_id}-with-image-refs.md").open("w", encoding="utf-8") as f:
        f.write(mixtual_md)

    with (out_dir / f"{paper_id}-text.md").open("w", encoding="utf-8") as f:
        f.write(poster_text)

    return images_meta, poster_text, mixtual_md


def md_to_blocks(md: str, base_dir: str = "") -> list[dict]:
    """Convert markdown with image refs into multimodal blocks."""
    from PIL import Image

    blocks: list[dict] = []
    pos = 0
    pat = re.compile(r"!\[.*?\]\((.*?)\)", re.DOTALL)
    for m in pat.finditer(md):
        txt = md[pos:m.start()].strip()
        if txt:
            blocks.append({"type": "text", "text": txt})
        img_path = os.path.join(base_dir, m.group(1))
        img = Image.open(img_path)
        blocks.append({"type": "image_url", "image_url": {"url": pil_to_data_uri(img, fmt="PNG")}})
        pos = m.end()
    tail = md[pos:].strip()
    if tail:
        blocks.append({"type": "text", "text": tail})
    return blocks


# ----------------------------
# CLIP (AltCLIP) helpers
# ----------------------------


_ALTCLIP_CACHE: tuple | None = None


def _load_altclip():
    global _ALTCLIP_CACHE
    if _ALTCLIP_CACHE is not None:
        return _ALTCLIP_CACHE
    try:
        import torch
        from transformers import AltCLIPProcessor, AltCLIPModel
    except ImportError as exc:
        raise RuntimeError(
            "AltCLIP requires torch + transformers. Install with:\n"
            "  pip install torch transformers\n"
            "and ensure a compatible CUDA/CPU build."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "BAAI/AltCLIP"
    model = AltCLIPModel.from_pretrained(model_name)
    model = cast(Any, model).to(device)
    processor = AltCLIPProcessor.from_pretrained(model_name)
    model.eval()
    _ALTCLIP_CACHE = (model, processor, device)
    return _ALTCLIP_CACHE


def _clip_image_embedding(image: Image.Image):
    import torch

    model, processor, device = _load_altclip()
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    return feats[0].cpu().numpy()


def _clip_text_embedding(text: str):
    import torch

    model, processor, device = _load_altclip()
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=processor.tokenizer.model_max_length,
    ).to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    return feats[0].cpu().numpy()


def _cosine_similarity(vec1, vec2) -> float:
    import numpy as np

    dot = np.dot(vec1, vec2)
    norm = (np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-8
    return float(dot / norm)


def _collect_images(path: Path) -> list[Image.Image]:
    if path.is_dir():
        images = []
        for p in sorted(path.iterdir()):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                images.append(Image.open(p).convert("RGB"))
        return images
    if path.is_file():
        return [Image.open(path).convert("RGB")]
    return []


def compute_clip_similarity(gt_path: Path, gen_path: Path) -> float | None:
    """Average cosine similarity between GT and generated posters."""
    gt_images = _collect_images(gt_path)
    gen_images = _collect_images(gen_path)
    if not gt_images or not gen_images:
        return None

    gt_embs = [_clip_image_embedding(img) for img in gt_images]
    gen_embs = [_clip_image_embedding(img) for img in gen_images]

    sims: list[float] = []
    for e1 in gt_embs:
        for e2 in gen_embs:
            sims.append(_cosine_similarity(e1, e2))
    return float(sum(sims) / len(sims)) if sims else None


def compute_visual_relevance(panels: Sequence[PanelContent]) -> float:
    """Average AltCLIP image↔text similarity across all panel images."""
    sims: list[float] = []
    for panel in panels:
        if not panel.text:
            continue
        text_emb = _clip_text_embedding(panel.text)
        for img in panel.images:
            img_emb = _clip_image_embedding(img)
            sims.append(_cosine_similarity(img_emb, text_emb))
    if not sims:
        return 0.0
    return float(sum(sims) / len(sims))


# ----------------------------
# Textual PPL (Llama-2-7b-hf)
# ----------------------------


_LLAMA_CACHE: tuple | None = None


def compute_textual_ppl(text: str, model_name: str = "meta-llama/Llama-2-7b-hf") -> float:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "Textual PPL requires torch + transformers. Install with:\n"
            "  pip install torch transformers"
        ) from exc

    if not text:
        return float("nan")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global _LLAMA_CACHE
    if _LLAMA_CACHE is None or _LLAMA_CACHE[0] != model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = cast(Any, model).to(device)
        model.eval()
        _LLAMA_CACHE = (model_name, tokenizer, model)
    else:
        _, tokenizer, model = _LLAMA_CACHE

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    # Sliding window to handle long inputs safely
    max_len = getattr(model.config, "max_position_embeddings", 2048)
    stride = max_len // 2
    nlls = []
    total_tokens = 0

    for start in range(0, input_ids.size(1), stride):
        end = min(start + max_len, input_ids.size(1))
        input_chunk = input_ids[:, start:end]
        target_chunk = input_chunk.clone()
        if start > 0:
            target_chunk[:, : -stride] = -100
        with torch.no_grad():
            outputs = model(input_chunk, labels=target_chunk)
        n_tokens = (target_chunk != -100).sum()
        neg_log_likelihood = outputs.loss * n_tokens
        nlls.append(neg_log_likelihood)
        total_tokens += int(n_tokens)
        if end == input_ids.size(1):
            break

    ppl = torch.exp(torch.stack(nlls).sum() / max(1, total_tokens))
    return float(ppl.item())


# ----------------------------
# VLM PPL (Qwen2.5-VL via vLLM)
# ----------------------------


def pil_to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.save(buf, format="JPEG", quality=90)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG")
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:{mime};base64,{b64}"


def compute_vlm_ppl(content, *, base_url: str, model: str) -> float:
    """Compute PPL using vLLM prompt_logprobs (Paper2Poster-style)."""
    try:
        from openai import OpenAI
        from httpx import Timeout
    except ImportError as exc:
        raise RuntimeError(
            "VLM PPL requires openai + httpx. Install with:\n"
            "  pip install openai httpx"
        ) from exc

    client = OpenAI(
        api_key="EMPTY",
        base_url=base_url,
        timeout=Timeout(5000),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        max_tokens=1,
        logprobs=True,
        extra_body={"prompt_logprobs": 1, "echo": True},
    )
    payload = resp.to_dict()
    lp_list = cast(list[dict], payload.get("prompt_logprobs", []))
    if lp_list is None:
        raise RuntimeError("vLLM response missing prompt_logprobs; ensure vLLM is configured.")
    total_lp = 0.0
    n_text = 0
    for token_entry in lp_list:
        if not token_entry:
            continue
        token_info = next(v for v in token_entry.values() if v["rank"] == 1)
        tok = token_info["decoded_token"]
        lp = token_info["logprob"]
        if re.fullmatch(r"<\\|?image[^>]*\\|?>", tok):
            continue
        total_lp += lp
        n_text += 1
    return math.exp(-total_lp / max(1, n_text))


def get_visual_ppl(image: Image.Image, text: str, *, base_url: str, model: str) -> float:
    img_uri = pil_to_data_uri(image, fmt="PNG")
    content = [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": img_uri}}]
    return compute_vlm_ppl(content, base_url=base_url, model=model)


def estimate_visual_tokens(
    images: Sequence[Image.Image],
    *,
    resized_height: int | None = None,
    resized_width: int | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> list[int]:
    counts: list[int] = []
    for img in images:
        h, w = img.height, img.width
        if resized_height and resized_width:
            h, w = resized_height, resized_width
        if min_pixels and h * w < min_pixels:
            scale = (min_pixels / (h * w)) ** 0.5
            h, w = int(h * scale), int(w * scale)
        if max_pixels and h * w > max_pixels:
            scale = (max_pixels / (h * w)) ** 0.5
            h, w = int(h * scale), int(w * scale)
        h = math.ceil(h / 28) * 28
        w = math.ceil(w / 28) * 28
        counts.append((h // 28) * (w // 28))
    return counts


def _image_memory_size(img: Image.Image, fmt: str = "JPEG") -> int:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.tell()


def truncate_images_to_fit(images: Sequence[Image.Image], *, max_ctx: int) -> list[Image.Image]:
    tokens = estimate_visual_tokens(images)
    max_size = 45 * 1024 * 1024
    total_size = 0
    keep: list[Image.Image] = []
    total = 0
    for img, n_tok in zip(images, tokens):
        if total + n_tok > max_ctx:
            break
        img_size = _image_memory_size(img)
        if total_size + img_size > max_size:
            break
        keep.append(img)
        total += n_tok
        total_size += img_size
    return keep


def compute_poster_image_ppl(
    images: Sequence[Image.Image], *, base_url: str, model: str
) -> float:
    max_ctx = 128_000
    imgs = truncate_images_to_fit(images, max_ctx=max_ctx)
    img_uris = [pil_to_data_uri(img, fmt="PNG") for img in imgs]
    content = [{"type": "image_url", "image_url": {"url": uri}} for uri in img_uris]
    return compute_vlm_ppl(content, base_url=base_url, model=model)


# ----------------------------
# High-level evaluation
# ----------------------------


def evaluate_stats(
    *,
    paper_id: str,
    method_name: str,
    gen_poster_png: Path,
    gen_poster_html: Path | None,
    gt_poster_png: Path | None,
    out_root: Path = Path("eval_results"),
    md_root: Path = Path("eval_poster_markdown"),
    enable_vlm_ppl: bool = True,
    enable_textual_ppl: bool = True,
    vllm_base_url: str | None = None,
    vllm_model: str | None = None,
) -> dict:
    """Run Paper2Poster-style stats metrics for a single poster."""
    out_dir = out_root / paper_id / method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_file = out_dir / "stats_result.json"

    stats_result: dict = {}
    if stats_file.exists():
        stats_result = json.loads(stats_file.read_text(encoding="utf-8"))

    # 1) CLIP similarity (requires GT poster)
    if gt_poster_png and "CLIP_similarity" not in stats_result:
        stats_result["CLIP_similarity"] = compute_clip_similarity(gt_poster_png, gen_poster_png)

    # 2) Textual + visual metrics (requires HTML for panel parsing)
    panels: list[PanelContent] = []
    poster_text = ""
    mixtual_md = ""
    images_meta: dict[str, dict] = {}

    need_panel_metrics = any(
        k not in stats_result
        for k in ("textual_ppl", "mixtual_ppl", "visual_relevance", "visual_ppl", "interleaved_ppl")
    )

    if need_panel_metrics and gen_poster_html and gen_poster_html.exists():
        html_text = gen_poster_html.read_text(encoding="utf-8")
        panels = extract_panels_from_html(html_text)
        md_out_dir = md_root / paper_id / method_name
        images_meta, poster_text, mixtual_md = build_eval_markdown(
            panels, out_dir=md_out_dir, paper_id=paper_id
        )
    else:
        if need_panel_metrics:
            stats_result.setdefault("textual_ppl", None)
            stats_result.setdefault("mixtual_ppl", None)
            stats_result.setdefault("visual_relevance", None)
            stats_result.setdefault("visual_ppl", None)
            stats_result.setdefault("interleaved_ppl", None)

    if enable_textual_ppl and poster_text and "textual_ppl" not in stats_result:
        stats_result["textual_ppl"] = compute_textual_ppl(poster_text)

    if enable_textual_ppl and mixtual_md and "mixtual_ppl" not in stats_result:
        stats_result["mixtual_ppl"] = compute_textual_ppl(mixtual_md)

    if panels and "visual_relevance" not in stats_result:
        stats_result["visual_relevance"] = compute_visual_relevance(panels)

    # VLM-based PPLs (optional, needs vLLM running)
    if enable_vlm_ppl and panels:
        base_url = vllm_base_url or os.getenv("ANY2POSTER_VLLM_BASE_URL", "http://localhost:7000/v1")
        model = vllm_model or os.getenv("ANY2POSTER_VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

        if "visual_ppl" not in stats_result:
            visual_ppls = []
            for panel in panels:
                if not panel.text:
                    continue
                for img in panel.images:
                    visual_ppls.append(get_visual_ppl(img, poster_text, base_url=base_url, model=model))
            stats_result["visual_ppl"] = float(sum(visual_ppls) / len(visual_ppls)) if visual_ppls else 0.0

        if "interleaved_ppl" not in stats_result and mixtual_md:
            md_path = md_root / paper_id / method_name / f"{paper_id}-with-image-refs.md"
            parts = md_to_blocks(md_path.read_text(encoding="utf-8"), base_dir=str(md_path.parent))
            stats_result["interleaved_ppl"] = compute_vlm_ppl(parts, base_url=base_url, model=model)

        if "poster_image_ppl" not in stats_result:
            from PIL import Image

            poster_img = Image.open(gen_poster_png).convert("RGB")
            stats_result["poster_image_ppl"] = compute_poster_image_ppl(
                [poster_img], base_url=base_url, model=model
            )
    else:
        stats_result.setdefault("visual_ppl", None)
        stats_result.setdefault("interleaved_ppl", None)
        stats_result.setdefault("poster_image_ppl", None)

    stats_file.write_text(json.dumps(stats_result, indent=2), encoding="utf-8")
    return stats_result


# ----------------------------
# CLI
# ----------------------------


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="any2poster stats evaluation (Paper2Poster-style).")
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--gen-poster-png", required=True)
    parser.add_argument("--gen-poster-html", default=None)
    parser.add_argument("--gt-poster-png", default=None)
    parser.add_argument("--out-root", default="eval_results")
    parser.add_argument("--md-root", default="eval_poster_markdown")
    parser.add_argument("--disable-vlm-ppl", action="store_true")
    parser.add_argument("--vllm-base-url", default=None)
    parser.add_argument("--vllm-model", default=None)

    args = parser.parse_args()

    evaluate_stats(
        paper_id=args.paper_id,
        method_name=args.method_name,
        gen_poster_png=Path(args.gen_poster_png),
        gen_poster_html=Path(args.gen_poster_html) if args.gen_poster_html else None,
        gt_poster_png=Path(args.gt_poster_png) if args.gt_poster_png else None,
        out_root=Path(args.out_root),
        md_root=Path(args.md_root),
        enable_vlm_ppl=not args.disable_vlm_ppl,
        vllm_base_url=args.vllm_base_url,
        vllm_model=args.vllm_model,
    )


if __name__ == "__main__":
    _cli()
