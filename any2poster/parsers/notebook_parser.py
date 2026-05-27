"""Jupyter Notebook parser for .ipynb files.

Extracts markdown cells as section prose, code cells as methodology snippets,
and base64-encoded plot outputs as real figures — giving the downstream pipeline
genuine content and images to work with.

Structure of a .ipynb file (nbformat 4):
  {
    "cells": [
      {"cell_type": "markdown", "source": ["# Heading\\n", "body text..."]},
      {"cell_type": "code",     "source": ["..."], "outputs": [...]},
      ...
    ],
    "metadata": {"kernelspec": {...}, ...}
  }

Output images are stored under outputs[*].data as base64-encoded PNG/JPEG
with keys "image/png" or "image/jpeg".

No external dependencies required beyond Pillow (already in the project).
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from any2poster.parsers.base import BaseParser
from any2poster.models import (
    ParsedDocument,
    ExtractedSection,
    ExtractedFigure,
    ExtractedTable,
    SectionType,
)


# Markdown header pattern
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Markdown image reference
_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

# Minimum image dimension to keep (filters out tiny icons / progress bars)
_MIN_IMAGE_PX = 80


class NotebookParser(BaseParser):
    """Parse Jupyter Notebook files into poster-ready ParsedDocuments."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".ipynb"]

    def parse(self, input_path: str, output_dir: Path) -> ParsedDocument:
        output_dir.mkdir(parents=True, exist_ok=True)
        nb_path = Path(input_path)

        try:
            nb = json.loads(nb_path.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:
            raise ValueError(f"Could not parse notebook JSON: {nb_path}") from exc

        # Support both nbformat 4 (cells at top level) and nbformat 3 (cells
        # nested under worksheets[0].cells).
        cells = nb.get("cells", [])
        if not cells:
            worksheets = nb.get("worksheets", [])
            if worksheets and isinstance(worksheets, list):
                cells = worksheets[0].get("cells", [])

        if not cells:
            raise ValueError(
                f"Notebook file is empty (no cells): {nb_path.name}\n"
                "Please replace it with a real Jupyter notebook that has content cells."
            )

        nb_metadata = nb.get("metadata", {})
        figures: list[ExtractedFigure] = []
        fig_counter = 0

        # ── Pass 1: group cells under markdown headers ──────────────────────
        # Each group is (header_level, header_title, [cell]) where cells may be
        # markdown, code, or raw.  A sentinel group captures content before the
        # first header.

        groups: list[tuple[int, str, list[dict]]] = []
        current_level = 1
        current_title = "_preamble"
        current_cells: list[dict] = []

        for cell in cells:
            cell_type = cell.get("cell_type", "")
            source = self._cell_source(cell)

            if cell_type == "markdown":
                first_header = _HEADER_RE.search(source)
                if first_header:
                    # Flush current group up to where this header appears
                    pre_header = source[: first_header.start()].strip()
                    if pre_header:
                        current_cells.append(
                            {"cell_type": "markdown", "source": pre_header, "_outputs": []}
                        )
                    groups.append((current_level, current_title, current_cells))

                    current_level = len(first_header.group(1))
                    current_title = first_header.group(2).strip()
                    # Remaining text after the header line is this section's first cell
                    remainder = source[first_header.end():].strip()
                    current_cells = []
                    if remainder:
                        current_cells.append(
                            {"cell_type": "markdown", "source": remainder, "_outputs": []}
                        )
                    continue

            current_cells.append(
                {"cell_type": cell_type, "source": source, "_outputs": cell.get("outputs", [])}
            )

        groups.append((current_level, current_title, current_cells))

        # ── Pass 2: build sections from groups ─────────────────────────────
        sections: list[ExtractedSection] = []
        section_id = 0
        tables: list[ExtractedTable] = []

        for level, title, group_cells in groups:
            if title == "_preamble":
                # Preamble: extract title/authors from first markdown cell if present
                for c in group_cells:
                    if c["cell_type"] == "markdown" and c["source"].strip():
                        title_candidate, authors_candidate = self._parse_preamble(
                            c["source"]
                        )
                        if title_candidate:
                            nb_metadata["_extracted_title"] = title_candidate
                        if authors_candidate:
                            nb_metadata["_extracted_authors"] = authors_candidate
                        break
                continue

            content_parts: list[str] = []
            section_fig_ids: list[str] = []

            for c in group_cells:
                ctype = c["cell_type"]
                src = c["source"].strip()
                outputs = c.get("_outputs", [])

                if ctype == "markdown" and src:
                    content_parts.append(src)

                elif ctype == "code":
                    # Extract any images from output
                    for out in outputs:
                        img_b64, img_ext = self._extract_image_from_output(out)
                        if img_b64 is None:
                            continue
                        fig_counter += 1
                        img_path = self._save_image(
                            img_b64, img_ext, output_dir, fig_counter
                        )
                        if img_path is None:
                            continue
                        fig_id = f"fig_{fig_counter}"
                        figures.append(
                            ExtractedFigure(
                                figure_id=fig_id,
                                caption=f'Output from cell in "{title}"',
                                path=img_path,
                            )
                        )
                        section_fig_ids.append(fig_id)

                    # Include short code snippets as methodology prose
                    code_prose = self._code_to_prose(src)
                    if code_prose:
                        content_parts.append(code_prose)

                    # Stream / text outputs as supporting text
                    for out in outputs:
                        stream_text = self._extract_stream_text(out)
                        if stream_text:
                            content_parts.append(stream_text)

            if not content_parts and not section_fig_ids:
                continue

            content = "\n\n".join(content_parts).strip()
            # Strip markdown image references (local paths won't resolve)
            content = _IMG_RE.sub("", content).strip()

            section_id += 1
            section_type = SectionType(self._detect_section_type(title))

            sections.append(
                ExtractedSection(
                    section_id=f"section_{section_id}",
                    title=title,
                    section_type=section_type,
                    content=content,
                    level=level,
                    word_count=self._count_words(content),
                    figure_ids=section_fig_ids,
                )
            )

        # ── Fallback: no headers found ─────────────────────────────────────
        if not sections:
            all_text = "\n\n".join(
                self._cell_source(c) for c in cells if c.get("cell_type") == "markdown"
            ).strip()
            if all_text:
                sections.append(
                    ExtractedSection(
                        section_id="section_1",
                        title="Notebook Content",
                        section_type=SectionType.OTHER,
                        content=all_text,
                        level=1,
                        word_count=self._count_words(all_text),
                    )
                )

        # ── Metadata ────────────────────────────────────────────────────────
        title = (
            nb_metadata.get("_extracted_title")
            or self._infer_title(sections, nb_path)
        )
        authors = nb_metadata.get("_extracted_authors", "")

        full_text_parts = [title] + [s.content for s in sections if s.content]
        full_text = "\n\n".join(p for p in full_text_parts if p).strip()

        return ParsedDocument(
            source_path=str(nb_path),
            source_format="notebook",
            title=title,
            authors=authors,
            sections=sections,
            figures=figures,
            tables=tables,
            raw_text=full_text,
            total_words=self._count_words(full_text),
        )

    # ── Cell helpers ──────────────────────────────────────────────────────────

    def _cell_source(self, cell: dict) -> str:
        # nbformat 4 uses "source"; nbformat 3 code cells use "input"
        src = cell.get("source") or cell.get("input", "")
        if isinstance(src, list):
            return "".join(src)
        return str(src)

    # ── Image extraction ──────────────────────────────────────────────────────

    def _extract_image_from_output(
        self, output: dict
    ) -> tuple[bytes | None, str]:
        """Return (raw_bytes, extension) or (None, '') if no image is present."""
        output_type = output.get("output_type", "")
        if output_type not in ("display_data", "execute_result"):
            return None, ""

        data = output.get("data", {})
        for mime, ext in (("image/png", "png"), ("image/jpeg", "jpg")):
            raw = data.get(mime)
            if raw:
                if isinstance(raw, list):
                    raw = "".join(raw)
                try:
                    return base64.b64decode(raw), ext
                except Exception:
                    pass
        return None, ""

    def _save_image(
        self, img_bytes: bytes, ext: str, output_dir: Path, idx: int
    ) -> Path | None:
        path = output_dir / f"nb_figure_{idx}.{ext}"
        try:
            path.write_bytes(img_bytes)
        except Exception:
            return None

        # Reject images below the minimum size threshold
        try:
            from PIL import Image
            with Image.open(path) as img:
                w, h = img.size
            if w < _MIN_IMAGE_PX or h < _MIN_IMAGE_PX:
                path.unlink(missing_ok=True)
                return None
        except Exception:
            pass  # If Pillow can't open it, keep it anyway

        return path

    # ── Stream / text output extraction ──────────────────────────────────────

    def _extract_stream_text(self, output: dict) -> str:
        output_type = output.get("output_type", "")
        if output_type == "stream":
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            text = text.strip()
            # Only keep short, human-readable outputs (not huge data dumps)
            if text and len(text) < 800 and not text.startswith("<"):
                return text
        if output_type in ("display_data", "execute_result"):
            text = output.get("data", {}).get("text/plain", "")
            if isinstance(text, list):
                text = "".join(text)
            text = text.strip()
            if text and len(text) < 600 and not text.startswith("<"):
                return text
        return ""

    # ── Code cell summarisation ───────────────────────────────────────────────

    def _code_to_prose(self, code: str) -> str:
        """Extract docstrings and comments from code as readable prose."""
        lines = code.splitlines()
        prose_lines: list[str] = []
        in_docstring = False
        docstring_char = ""

        for line in lines:
            stripped = line.strip()

            # Start of triple-quoted docstring
            if not in_docstring:
                for q in ('"""', "'''"):
                    if stripped.startswith(q):
                        in_docstring = True
                        docstring_char = q
                        inner = stripped[len(q):]
                        if inner.endswith(q) and len(inner) > len(q):
                            prose_lines.append(inner[: -len(q)].strip())
                            in_docstring = False
                        elif inner.strip():
                            prose_lines.append(inner.strip())
                        break
            else:
                if docstring_char in stripped:
                    before = stripped[: stripped.index(docstring_char)].strip()
                    if before:
                        prose_lines.append(before)
                    in_docstring = False
                else:
                    if stripped:
                        prose_lines.append(stripped)
                continue

            # Standalone comments
            if stripped.startswith("#") and not in_docstring:
                comment = stripped.lstrip("#").strip()
                if len(comment) > 5:
                    prose_lines.append(comment)

        result = " ".join(prose_lines).strip()
        # Don't include trivially short or purely numeric outputs
        return result if len(result.split()) >= 6 else ""

    # ── Preamble / metadata helpers ───────────────────────────────────────────

    def _parse_preamble(self, text: str) -> tuple[str, str]:
        """Try to extract a title and author list from a YAML frontmatter block
        or the first H1 heading in the preamble cell."""
        title = ""
        authors = ""

        # YAML frontmatter
        fm = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
        if fm:
            yaml_block = fm.group(1)
            t = re.search(r"^title:\s*['\"]?([^'\"\n]+)", yaml_block, re.MULTILINE)
            if t:
                title = t.group(1).strip()
            a = re.search(r"^authors?:\s*['\"]?([^'\"\n]+)", yaml_block, re.MULTILINE)
            if a:
                authors = a.group(1).strip()

        # First H1 heading as fallback title
        if not title:
            h1 = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
            if h1:
                title = h1.group(1).strip()

        return title, authors

    def _infer_title(self, sections: list[ExtractedSection], nb_path: Path) -> str:
        if sections:
            return sections[0].title
        return nb_path.stem.replace("_", " ").replace("-", " ").title()
