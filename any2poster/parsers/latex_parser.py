"""LaTeX / .tex parser for arXiv-style research papers.

Handles the standard academic LaTeX document structure produced by arXiv:

  \\title{}, \\author{}, \\begin{abstract}, \\section{}, \\subsection{},
  \\begin{figure}, \\includegraphics{}, \\caption{},
  \\begin{table}, \\begin{tabular}, \\begin{itemize/enumerate},
  \\input{} / \\include{} multi-file documents.

All LaTeX markup is stripped to plain prose before the content is handed to
the downstream pipeline, so the analyze stage receives the same quality of
clean text it gets from a well-parsed PDF.

No external LaTeX libraries required — everything is done with stdlib.
"""

from __future__ import annotations

import base64
import re
import shutil
from pathlib import Path

from any2poster.parsers.base import BaseParser
from any2poster.models import (
    ParsedDocument,
    ExtractedSection,
    ExtractedFigure,
    ExtractedTable,
    SectionType,
)


# ──────────────────────────────────────────────────────────────────────────────
# Regex helpers (compiled once at import time)
# ──────────────────────────────────────────────────────────────────────────────

# Strip % comments (but not \% escaped percent signs)
_COMMENT_RE = re.compile(r"(?<!\\)%[^\n]*")

# Common text-formatting macros whose content we want to keep
_INLINE_MACROS = re.compile(
    r"\\(?:textbf|textit|emph|textrm|texttt|textsc|underline|text)\{([^{}]*)\}"
)

# Macros to remove entirely (content discarded)
_DISCARD_MACROS = re.compile(
    r"\\(?:label|ref|eqref|cite(?:p|t|alt)?|footnote|footnotemark|"
    r"footnotetext|vspace|hspace|vskip|hskip|noindent|centering|"
    r"raggedright|raggedleft|clearpage|newpage|pagebreak|linebreak|"
    r"newline|medskip|bigskip|smallskip|par)\s*(?:\{[^{}]*\})?"
)

# Remaining standalone commands with no meaningful text
_MISC_CMDS = re.compile(r"\\[a-zA-Z]+\*?\s*")

# Collapse excessive whitespace
_WHITESPACE_RE = re.compile(r"[ \t]{2,}")
_BLANKLINES_RE = re.compile(r"\n{3,}")


# ──────────────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────────────

class LaTeXParser(BaseParser):
    """Parse .tex files into poster-ready ParsedDocuments."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".tex"]

    def parse(self, input_path: str, output_dir: Path) -> ParsedDocument:
        output_dir.mkdir(parents=True, exist_ok=True)
        tex_path = Path(input_path)
        source_dir = tex_path.parent

        raw = tex_path.read_text(encoding="utf-8", errors="replace")

        # Inline \input{} and \include{} before anything else
        raw = self._resolve_includes(raw, source_dir)

        # Strip comments first so they don't confuse later regexes
        raw = _COMMENT_RE.sub("", raw)

        # Extract the document body (between \begin{document} and \end{document})
        body = self._extract_body(raw)

        title = self._extract_command(raw, "title") or tex_path.stem.replace("_", " ").title()
        authors = self._extract_authors(raw)
        affiliation = self._extract_affiliation(raw)
        abstract = self._extract_environment(body, "abstract")

        figures = self._extract_figures(body, source_dir, output_dir)
        tables = self._extract_tables(body)
        sections = self._extract_sections(body, abstract)

        full_text_parts: list[str] = [title]
        if authors:
            full_text_parts.append(authors)
        if abstract:
            full_text_parts.append(abstract)
        full_text_parts += [s.content for s in sections if s.content]
        full_text = "\n\n".join(p for p in full_text_parts if p).strip()

        # Associate figures with sections by ordinal proximity
        self._associate_figures(sections, figures, body)

        return ParsedDocument(
            source_path=str(tex_path),
            source_format="latex",
            title=title,
            authors=authors,
            affiliation=affiliation,
            abstract=abstract,
            sections=sections,
            figures=figures,
            tables=tables,
            raw_text=full_text,
            total_words=self._count_words(full_text),
        )

    # ── Include resolution ────────────────────────────────────────────────────

    def _resolve_includes(self, source: str, base_dir: Path, depth: int = 0) -> str:
        if depth > 8:
            return source

        def _replace(m: re.Match) -> str:
            rel = m.group(1).strip()
            for candidate in (rel, rel + ".tex"):
                p = base_dir / candidate
                if p.exists():
                    try:
                        child = p.read_text(encoding="utf-8", errors="replace")
                        child = _COMMENT_RE.sub("", child)
                        return self._resolve_includes(child, p.parent, depth + 1)
                    except Exception:
                        pass
            return ""

        source = re.sub(r"\\input\{([^}]+)\}", _replace, source)
        source = re.sub(r"\\include\{([^}]+)\}", _replace, source)
        return source

    # ── Structural extraction ─────────────────────────────────────────────────

    def _extract_body(self, source: str) -> str:
        m = re.search(
            r"\\begin\{document\}(.*?)\\end\{document\}",
            source,
            re.DOTALL,
        )
        return m.group(1) if m else source

    def _extract_command(self, source: str, cmd: str) -> str:
        """Extract the content of \\cmd{...}, handling nested braces."""
        pattern = re.compile(r"\\" + re.escape(cmd) + r"\s*\{")
        m = pattern.search(source)
        if not m:
            return ""
        content = self._extract_braced(source, m.end() - 1)
        return self._clean_text(content)

    def _extract_authors(self, source: str) -> str:
        raw = self._extract_command(source, "author")
        if not raw:
            return ""
        # Split on \and (common multi-author separator)
        parts = re.split(r"\\and\b", raw, flags=re.IGNORECASE)
        authors: list[str] = []
        for part in parts:
            name = self._clean_text(part).strip()
            # Drop pure affiliation lines (email addresses, numbers)
            name = re.sub(r"\S+@\S+", "", name).strip()
            name = re.sub(r"\\\\\s*\d+", "", name).strip()
            if name and len(name.split()) <= 8:
                authors.append(name)
        return ", ".join(authors) if authors else self._clean_text(raw)

    def _extract_affiliation(self, source: str) -> str:
        for cmd in ("affiliation", "institute", "address", "institution"):
            val = self._extract_command(source, cmd)
            if val:
                return val[:200]
        return ""

    def _extract_environment(self, body: str, env: str) -> str:
        pattern = re.compile(
            r"\\begin\{" + re.escape(env) + r"\}(.*?)\\end\{" + re.escape(env) + r"\}",
            re.DOTALL | re.IGNORECASE,
        )
        m = pattern.search(body)
        if not m:
            return ""
        return self._clean_text(m.group(1))

    # ── Section extraction ────────────────────────────────────────────────────

    def _extract_sections(self, body: str, abstract: str) -> list[ExtractedSection]:
        # Remove figure and table environments before parsing prose
        body_clean = re.sub(
            r"\\begin\{(?:figure|table|algorithm|lstlisting|verbatim)\*?\}.*?"
            r"\\end\{(?:figure|table|algorithm|lstlisting|verbatim)\*?\}",
            "",
            body,
            flags=re.DOTALL | re.IGNORECASE,
        )

        section_re = re.compile(
            r"\\(section|subsection|subsubsection)\*?\{([^}]*)\}",
            re.IGNORECASE,
        )

        level_map = {"section": 1, "subsection": 2, "subsubsection": 3}

        chunks: list[tuple[int, str, str]] = []  # (level, title, content)
        last_end = 0
        pending_level = 0
        pending_title = ""

        for m in section_re.finditer(body_clean):
            if pending_title:
                content = body_clean[last_end:m.start()]
                chunks.append((pending_level, pending_title, content))
            elif last_end == 0 and m.start() > 0:
                # Content before the first section (preamble / abstract block)
                pre = body_clean[:m.start()].strip()
                if pre:
                    chunks.append((1, "_pre", pre))

            pending_level = level_map.get(m.group(1).lower(), 1)
            pending_title = self._clean_text(m.group(2))
            last_end = m.end()

        # Remainder after last section header
        if pending_title:
            content = body_clean[last_end:]
            chunks.append((pending_level, pending_title, content))
        elif not chunks and body_clean.strip():
            chunks.append((1, "Content", body_clean))

        sections: list[ExtractedSection] = []
        section_id = 0

        # Prepend abstract as its own section if present
        if abstract:
            section_id += 1
            sections.append(
                ExtractedSection(
                    section_id=f"section_{section_id}",
                    title="Abstract",
                    section_type=SectionType.ABSTRACT,
                    content=abstract,
                    level=1,
                    word_count=self._count_words(abstract),
                )
            )

        for level, title, raw_content in chunks:
            if title == "_pre":
                continue

            # Skip reference / bibliography sections
            if re.search(r"\b(reference|bibliograph)\b", title, re.IGNORECASE):
                continue

            content = self._clean_text(raw_content)
            if not content.strip():
                continue

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
                )
            )

        return sections

    # ── Figure extraction ─────────────────────────────────────────────────────

    def _extract_figures(
        self, body: str, source_dir: Path, output_dir: Path
    ) -> list[ExtractedFigure]:
        figures: list[ExtractedFigure] = []
        fig_env_re = re.compile(
            r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", re.DOTALL | re.IGNORECASE
        )
        graphics_re = re.compile(
            r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", re.IGNORECASE
        )
        caption_re = re.compile(r"\\caption\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

        counter = 0
        for m in fig_env_re.finditer(body):
            env = m.group(1)
            caption = ""
            cap_m = caption_re.search(env)
            if cap_m:
                caption = self._clean_text(cap_m.group(1))

            for gfx_m in graphics_re.finditer(env):
                img_ref = gfx_m.group(1).strip()
                img_path = self._resolve_image(img_ref, source_dir, output_dir, counter + 1)
                counter += 1
                figures.append(
                    ExtractedFigure(
                        figure_id=f"fig_{counter}",
                        caption=caption,
                        path=img_path,
                    )
                )

        return figures

    def _resolve_image(
        self, ref: str, source_dir: Path, output_dir: Path, idx: int
    ) -> Path | None:
        extensions = ["", ".pdf", ".png", ".jpg", ".jpeg", ".eps", ".PNG", ".JPG"]
        for ext in extensions:
            candidate = source_dir / (ref + ext)
            if candidate.exists():
                dest_suffix = ".png" if candidate.suffix.lower() in (".pdf", ".eps") else candidate.suffix
                dest = output_dir / f"figure_{idx}{dest_suffix}"
                try:
                    if candidate.suffix.lower() in (".pdf", ".eps"):
                        # Convert to PNG via Pillow if possible
                        try:
                            from PIL import Image
                            with Image.open(str(candidate)) as img:
                                img.save(str(dest), "PNG")
                            return dest
                        except Exception:
                            pass
                    shutil.copy2(candidate, dest)
                    return dest
                except Exception:
                    return None
        return None

    # ── Table extraction ──────────────────────────────────────────────────────

    def _extract_tables(self, body: str) -> list[ExtractedTable]:
        tables: list[ExtractedTable] = []
        table_env_re = re.compile(
            r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", re.DOTALL | re.IGNORECASE
        )
        caption_re = re.compile(r"\\caption\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
        tabular_re = re.compile(
            r"\\begin\{tabular\*?\}(?:\[[^\]]*\])?\{[^}]*\}(.*?)\\end\{tabular\*?\}",
            re.DOTALL | re.IGNORECASE,
        )

        counter = 0
        for m in table_env_re.finditer(body):
            env = m.group(1)
            caption = ""
            cap_m = caption_re.search(env)
            if cap_m:
                caption = self._clean_text(cap_m.group(1))

            tab_m = tabular_re.search(env)
            if not tab_m:
                continue

            md_table = self._tabular_to_markdown(tab_m.group(1))
            if not md_table:
                continue

            counter += 1
            tables.append(
                ExtractedTable(
                    table_id=f"table_{counter}",
                    caption=caption or f"Table {counter}",
                    content=md_table,
                )
            )

        return tables

    def _tabular_to_markdown(self, tabular_body: str) -> str:
        """Convert LaTeX tabular content to a markdown table."""
        # Strip LaTeX formatting, keep text and alignment separators
        body = re.sub(r"\\hline\b", "", tabular_body)
        body = re.sub(r"\\cline\{[^}]*\}", "", body)
        body = re.sub(r"\\multicolumn\{[^}]*\}\{[^}]*\}\{([^}]*)\}", r"\1", body)
        body = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}\{([^}]*)\}", r"\1", body)
        body = self._clean_text(body)

        rows: list[list[str]] = []
        for line in body.split("\\\\"):
            line = line.strip()
            if not line:
                continue
            cells = [c.strip() for c in line.split("&")]
            if any(c for c in cells):
                rows.append(cells)

        if not rows:
            return ""

        # Normalise column count
        n_cols = max(len(r) for r in rows)
        rows = [r + [""] * (n_cols - len(r)) for r in rows]

        header = "| " + " | ".join(rows[0]) + " |"
        sep = "| " + " | ".join(["---"] * n_cols) + " |"
        body_lines = ["| " + " | ".join(r) + " |" for r in rows[1:]]
        return "\n".join([header, sep] + body_lines)

    # ── Figure–section association ────────────────────────────────────────────

    def _associate_figures(
        self,
        sections: list[ExtractedSection],
        figures: list[ExtractedFigure],
        body: str,
    ) -> None:
        """Assign figure_ids to sections by matching figure order to section order."""
        if not sections or not figures:
            return

        # Find character positions of each section header in the body
        section_re = re.compile(r"\\(?:sub)*section\*?\{[^}]*\}", re.IGNORECASE)
        positions = [m.start() for m in section_re.finditer(body)]

        # Find positions of each \includegraphics
        graphics_re = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]+\}", re.IGNORECASE)
        fig_positions = [m.start() for m in graphics_re.finditer(body)]

        content_sections = [s for s in sections if s.section_type != SectionType.ABSTRACT]
        if not content_sections or not positions:
            return

        for fig_idx, fig_pos in enumerate(fig_positions):
            if fig_idx >= len(figures):
                break
            # Find which section this figure falls after
            sec_idx = 0
            for i, pos in enumerate(positions):
                if pos <= fig_pos:
                    sec_idx = i
            if sec_idx < len(content_sections):
                sec = content_sections[sec_idx]
                if figures[fig_idx].figure_id not in sec.figure_ids:
                    sec.figure_ids.append(figures[fig_idx].figure_id)

    # ── Text cleaning ─────────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """Convert LaTeX markup to clean readable prose."""
        # Preserve content of formatting macros
        text = _INLINE_MACROS.sub(r"\1", text)

        # Convert list environments to plain bullet text
        text = re.sub(r"\\begin\{itemize\}", "", text)
        text = re.sub(r"\\end\{itemize\}", "", text)
        text = re.sub(r"\\begin\{enumerate\}", "", text)
        text = re.sub(r"\\end\{enumerate\}", "", text)
        text = re.sub(r"\\item\s*\[?[^\]]*\]?\s*", "\n- ", text)

        # Strip equation environments (keep a placeholder so prose isn't run together)
        text = re.sub(
            r"\\begin\{(?:equation|align|gather|multline|eqnarray)\*?\}.*?"
            r"\\end\{(?:equation|align|gather|multline|eqnarray)\*?\}",
            " [eq] ",
            text,
            flags=re.DOTALL,
        )
        text = re.sub(r"\$\$.*?\$\$", " [eq] ", text, flags=re.DOTALL)
        text = re.sub(r"\$[^$\n]+\$", " [eq] ", text)

        # Strip remaining environments we don't want to render
        text = re.sub(
            r"\\begin\{(?:verbatim|lstlisting|algorithm|algorithmic|tikzpicture|"
            r"tabular|array|minipage|wrapfigure)\*?\}.*?"
            r"\\end\{(?:verbatim|lstlisting|algorithm|algorithmic|tikzpicture|"
            r"tabular|array|minipage|wrapfigure)\*?\}",
            "",
            text,
            flags=re.DOTALL,
        )
        text = re.sub(r"\\begin\{[^}]+\}", "", text)
        text = re.sub(r"\\end\{[^}]+\}", "", text)

        # Strip specific macros (content discarded)
        text = _DISCARD_MACROS.sub(" ", text)

        # Strip braces remaining from macros we didn't otherwise handle
        text = re.sub(r"\\[a-zA-Z]+\*?\s*\{([^{}]*)\}", r"\1", text)
        text = re.sub(r"\\[a-zA-Z]+\*?\s*\{([^{}]*)\}", r"\1", text)  # second pass

        # Remove remaining LaTeX commands
        text = _MISC_CMDS.sub(" ", text)

        # Clean up structural characters
        text = text.replace("~", " ")
        text = text.replace("``", '"').replace("''", '"')
        text = text.replace("`", "'")
        text = re.sub(r"---?", "—", text)
        text = re.sub(r"\{|\}", "", text)
        text = re.sub(r"\\\\", "\n", text)  # explicit newlines
        text = re.sub(r"\\,|\\;|\\:|\\!", " ", text)  # spacing commands

        # Normalise whitespace
        text = _WHITESPACE_RE.sub(" ", text)
        text = _BLANKLINES_RE.sub("\n\n", text)
        return self._normalize_text(text.strip())

    def _extract_braced(self, source: str, open_pos: int) -> str:
        """Extract content from a balanced-brace block starting at open_pos."""
        if open_pos >= len(source) or source[open_pos] != "{":
            return ""
        depth = 0
        start = open_pos + 1
        for i in range(open_pos, len(source)):
            c = source[i]
            if i > 0 and source[i - 1] == "\\":
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return source[start:i]
        return source[start:]
