"""Stage: VLM feedback loop for panel-level refinement.

This stage crops each panel, asks a VLM to detect layout issues using a
fixed vocabulary, and applies deterministic HTML/CSS mutations. The goal
is to reduce overflow, blank panels, and poor figure/text balance while
keeping changes minimal and reversible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html as _html_mod
import re
from typing import Any

from PIL import Image


@dataclass
class PanelBox:
    panel_id: str
    x: float
    y: float
    width: float
    height: float


_ALLOWED_ISSUES = {
    "good",
    "overflow",
    "too_blank",
    "visual_too_small",
    "visual_too_large",
}


_COMMENTER_SYSTEM = (
    "You are a strict poster layout reviewer. Only report real, visible issues. "
    "You MUST respond with valid JSON and use only the allowed issue codes."
)

_COMMENTER_USER = """Analyze the panel image and return any issues.

Allowed issue codes (use ONLY these):
- good (no issues)
- overflow (text looks cramped, clipped, or overflowing)
- too_blank (excess empty space; panel feels underfilled)
- visual_too_small (figure/diagram is too small relative to text)
- visual_too_large (figure/diagram dominates too much)

Return JSON exactly as:
{
  "issues": ["<code1>", "<code2>"]
}

Rules:
- If there are no issues, return {"issues": ["good"]}.
- Do not invent additional issue codes.
- Be conservative: only flag a problem if clearly visible.
"""


def run_feedback_stage(
    *,
    html_path: Path,
    output_path: Path,
    plan,
    config,
    provider,
) -> Path:
    """Run VLM feedback loop and re-render the poster outputs."""
    if not html_path.exists():
        raise FileNotFoundError(f"HTML not found for feedback: {html_path}")

    max_iters = max(1, int(getattr(config, "feedback_max_iters", 2)))
    feedback_model = getattr(config, "feedback_model", "") or getattr(config, "vision_model", "")

    for _iter in range(max_iters):
        # Render current HTML to PNG (and PDF) for inspection
        _render_outputs(html_path, output_path, plan, config)

        panel_boxes, scale = _get_panel_boxes(html_path, plan, config)
        if not panel_boxes:
            break

        panel_crops = _crop_panels(output_path, panel_boxes, scale)
        issues_map = _judge_panels(panel_crops, provider, feedback_model)

        # Apply fixes; if no fixes applied, exit early
        html_text = html_path.read_text(encoding="utf-8")
        updated_text, changed = _apply_fixes(html_text, issues_map)
        if not changed:
            break
        html_path.write_text(updated_text, encoding="utf-8")

    # Final render after feedback loop
    _render_outputs(html_path, output_path, plan, config)
    return output_path


def _render_outputs(html_path: Path, output_path: Path, plan, config) -> None:
    from any2poster.stages.compile_stage import _render_html

    if output_path.suffix.lower() == ".png":
        png_path = output_path
        pdf_path = output_path.with_suffix(".pdf")
    else:
        pdf_path = output_path
        png_path = output_path.with_suffix(".png")

    _render_html(html_path, pdf_path, png_path, plan, config)


def _get_panel_boxes(html_path: Path, plan, config) -> tuple[list[PanelBox], float]:
    """Compute panel bounding boxes via Playwright DOM query."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return [], 1.0

    from any2poster.stages.compile_stage import _find_system_browser

    width_in = plan.width_inches
    height_in = plan.height_inches
    dpi = config.dpi
    scale = max(dpi / 96.0, 1.0)
    vp_w = int(width_in * 96)
    vp_h = int(height_in * 96)
    url = html_path.resolve().as_uri()

    system_exe = _find_system_browser()
    launch_kwargs: dict[str, Any] = {
        "args": ["--no-sandbox", "--disable-setuid-sandbox"],
    }
    if system_exe:
        launch_kwargs["executable_path"] = system_exe

    boxes: list[PanelBox] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(**launch_kwargs)
        ctx = browser.new_context(viewport={"width": vp_w, "height": vp_h})
        page = ctx.new_page()
        page.goto(url, wait_until="load", timeout=60_000)
        page.wait_for_load_state("domcontentloaded")
        panel_info = page.evaluate(
            """
            () => Array.from(document.querySelectorAll('.panel')).map(el => {
              const r = el.getBoundingClientRect();
              return {
                id: el.getAttribute('data-panel-id') || '',
                x: r.x, y: r.y, width: r.width, height: r.height
              };
            })
            """
        )
        for item in panel_info:
            pid = str(item.get("id") or "")
            if not pid or pid == "title":
                continue
            boxes.append(
                PanelBox(
                    panel_id=pid,
                    x=float(item["x"]),
                    y=float(item["y"]),
                    width=float(item["width"]),
                    height=float(item["height"]),
                )
            )
        ctx.close()
        browser.close()

    return boxes, scale


def _crop_panels(poster_path: Path, boxes: list[PanelBox], scale: float) -> dict[str, Image.Image]:
    png_path = poster_path if poster_path.suffix.lower() == ".png" else poster_path.with_suffix(".png")
    img = Image.open(png_path).convert("RGB")
    crops: dict[str, Image.Image] = {}
    for box in boxes:
        left = max(0, int(box.x * scale) - 2)
        top = max(0, int(box.y * scale) - 2)
        right = min(img.width, int((box.x + box.width) * scale) + 2)
        bottom = min(img.height, int((box.y + box.height) * scale) + 2)
        crops[box.panel_id] = img.crop((left, top, right, bottom))
    return crops


def _judge_panels(
    panel_images: dict[str, Image.Image],
    provider,
    model_name: str,
) -> dict[str, list[str]]:
    from concurrent.futures import ThreadPoolExecutor

    def _run(panel_id: str, image: Image.Image) -> tuple[str, list[str]]:
        try:
            data = provider.complete_vision_json(
                system=_COMMENTER_SYSTEM,
                user=_COMMENTER_USER,
                image_path=_save_temp_image(panel_id, image),
                temperature=0.0,
                max_tokens=256,
                model_override=model_name or None,
            )
            issues: list[str] = []
            if isinstance(data, dict):
                raw = data.get("issues", [])
                if isinstance(raw, list):
                    issues = [str(x).strip() for x in raw if str(x).strip()]
            issues = [i for i in issues if i in _ALLOWED_ISSUES]
            if not issues:
                issues = ["good"]
            return panel_id, issues
        except Exception:
            # Fail-open: if the VLM is unavailable, skip feedback for this panel.
            return panel_id, ["good"]

    issues_map: dict[str, list[str]] = {}
    items = list(panel_images.items())
    if not items:
        return issues_map

    max_workers = min(8, len(items))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for panel_id, issues in ex.map(lambda x: _run(*x), items):
            issues_map[panel_id] = issues
    return issues_map


def _save_temp_image(panel_id: str, image: Image.Image) -> Path:
    tmp_dir = Path(".any2poster_cache") / "feedback_panels"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"{panel_id}.png"
    image.save(path)
    return path


def _apply_fixes(html_text: str, issues_map: dict[str, list[str]]) -> tuple[str, bool]:
    changed = False
    for panel_id, issues in issues_map.items():
        if issues == ["good"]:
            continue
        block = _find_panel_block(html_text, panel_id)
        if not block:
            continue
        updated = block

        if "overflow" in issues:
            updated, removed = _remove_last_bullet(updated)
            if not removed:
                updated = _truncate_lead_paragraph(updated)
            updated = _adjust_style_var(updated, "--panel-font-scale", -0.05, min_val=0.90, max_val=1.0)
            changed = True

        if "too_blank" in issues:
            # Fill empty space by gently increasing text and visual scale.
            updated = _adjust_style_var(updated, "--panel-text-scale", 0.06, min_val=0.85, max_val=1.15)
            updated = _adjust_style_var(updated, "--panel-font-scale", 0.03, min_val=0.90, max_val=1.10)
            updated = _adjust_style_var(updated, "--panel-visual-scale", 0.08, min_val=0.85, max_val=1.35)
            updated = _adjust_style_var(updated, "--panel-visual-flex", 0.08, min_val=0.85, max_val=1.35)
            changed = True

        if "visual_too_small" in issues:
            updated = _adjust_style_var(updated, "--panel-visual-scale", 0.15, min_val=0.90, max_val=1.40)
            updated = _adjust_style_var(updated, "--panel-visual-flex", 0.15, min_val=0.85, max_val=1.40)
            updated = _adjust_style_var(updated, "--panel-text-scale", -0.04, min_val=0.80, max_val=1.05)
            changed = True

        if "visual_too_large" in issues:
            updated = _adjust_style_var(updated, "--panel-visual-scale", -0.10, min_val=0.75, max_val=1.20)
            updated = _adjust_style_var(updated, "--panel-visual-flex", -0.10, min_val=0.75, max_val=1.20)
            updated = _adjust_style_var(updated, "--panel-text-scale", 0.05, min_val=0.80, max_val=1.20)
            changed = True

        if updated != block:
            html_text = html_text.replace(block, updated)

    return html_text, changed


def _find_panel_block(html_text: str, panel_id: str) -> str | None:
    pattern = re.compile(
        rf'<div class="panel"[^>]*data-panel-id="{re.escape(panel_id)}"[^>]*>',
        re.IGNORECASE,
    )
    m = pattern.search(html_text)
    if not m:
        return None
    start = m.start()

    # find matching closing </div> by tracking nested divs
    depth = 0
    for match in re.finditer(r"</?div\\b", html_text[start:], flags=re.IGNORECASE):
        tag = match.group(0).lower()
        if tag.startswith("<div"):
            depth += 1
        else:
            depth -= 1
        if depth == 0:
            end = start + match.end()
            return html_text[start:end]
    return None


def _remove_last_bullet(panel_html: str) -> tuple[str, bool]:
    li_matches = list(re.finditer(r"<li[^>]*>.*?</li>", panel_html, flags=re.IGNORECASE | re.DOTALL))
    if not li_matches:
        return panel_html, False
    last = li_matches[-1]
    return panel_html[: last.start()] + panel_html[last.end() :], True


def _truncate_lead_paragraph(panel_html: str) -> str:
    match = re.search(
        r"(<p class=\"lead-para\">)(.*?)(</p>)",
        panel_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return panel_html

    inner = match.group(2).strip()
    if not inner:
        return panel_html

    first_sentence = _first_sentence_html(inner)
    if first_sentence == inner:
        plain = re.sub(r"<[^>]+>", "", inner)
        if len(plain) > 160:
            truncated = plain[:150].rstrip() + "..."
            first_sentence = _html_mod.escape(truncated)

    if first_sentence == inner:
        return panel_html

    return panel_html[: match.start(2)] + first_sentence + panel_html[match.end(2) :]


def _first_sentence_html(html_text: str) -> str:
    parts = re.split(r"([.!?])\s+", html_text.strip())
    if len(parts) >= 2:
        return (parts[0] + parts[1]).strip()
    return html_text.strip()


def _adjust_style_var(
    panel_html: str,
    var_name: str,
    delta: float,
    *,
    min_val: float,
    max_val: float,
) -> str:
    # find opening panel tag
    open_tag_match = re.search(r"<div class=\"panel\"[^>]*>", panel_html, flags=re.IGNORECASE)
    if not open_tag_match:
        return panel_html
    open_tag = open_tag_match.group(0)

    style_match = re.search(r'style="([^"]*)"', open_tag, flags=re.IGNORECASE)
    style = style_match.group(1) if style_match else ""

    styles = _parse_style(style)
    current = float(styles.get(var_name, "1") or 1.0)
    new_val = max(min_val, min(max_val, current + delta))
    styles[var_name] = f"{new_val:.3f}"

    new_style = _style_to_string(styles)
    if style_match:
        new_open_tag = open_tag.replace(style_match.group(0), f'style="{new_style}"')
    else:
        new_open_tag = open_tag[:-1] + f' style="{new_style}">'

    return panel_html.replace(open_tag, new_open_tag, 1)


def _parse_style(style: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in style.split(";"):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k:
            result[k] = v
    return result


def _style_to_string(styles: dict[str, str]) -> str:
    return "; ".join(f"{k}: {v}" for k, v in styles.items())
