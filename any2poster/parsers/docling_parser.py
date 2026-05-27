"""Docling-based PDF parser for layout-aware document extraction.

Uses IBM's Docling library to parse PDFs with spatial awareness,
producing typed blocks (heading, paragraph, table, figure, equation)
with page numbers and bounding boxes. This matches the parsing
approach used by Paper2Poster (NeurIPS 2025) and RAG-Anything.
"""

import os
import shutil
import sys
from pathlib import Path

# Windows: patch os.symlink to fall back to file copy on privilege errors
# (WinError 1314 = symlink privilege not held; requires Developer Mode or admin).
if sys.platform == "win32":
    _orig_symlink = os.symlink

    def _symlink_with_copy_fallback(src, dst, target_is_directory=False, *, dir_fd=None):
        try:
            _orig_symlink(src, dst, target_is_directory=target_is_directory)
        except OSError as exc:
            if exc.winerror == 1314:  # ERROR_PRIVILEGE_NOT_HELD
                # Resolve src relative to dst's parent so shutil.copy2 gets the real path
                abs_src = src if os.path.isabs(src) else os.path.normpath(
                    os.path.join(os.path.dirname(dst), src)
                )
                shutil.copy2(abs_src, dst)
            else:
                raise

    os.symlink = _symlink_with_copy_fallback

from any2poster.models import (
    ExtractedFigure,
    ExtractedSection,
    ExtractedTable,
    ParsedDocument,
    SectionType,
)
from any2poster.parsers.base import BaseParser


class DoclingParser(BaseParser):

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def parse(self, input_path: str, output_dir: Path) -> ParsedDocument:
        from docling.document_converter import DocumentConverter

        output_dir.mkdir(parents=True, exist_ok=True)
        converter = DocumentConverter()
        result = converter.convert(input_path)
        doc = result.document

        raw_text = doc.export_to_markdown()
        raw_text = self._normalize_text(raw_text)

        sections: list[ExtractedSection] = []
        figures: list[ExtractedFigure] = []
        tables: list[ExtractedTable] = []

        current_heading = ""
        current_level = 1
        current_content_parts: list[str] = []
        current_figure_ids: list[str] = []
        current_table_ids: list[str] = []
        section_counter = 0
        figure_counter = 0
        table_counter = 0

        for item, _level in doc.iterate_items():
            item_type = type(item).__name__

            if item_type in ("SectionHeaderItem", "HeadingItem") or (
                hasattr(item, "label") and "head" in str(getattr(item, "label", "")).lower()
            ):
                if current_heading or current_content_parts:
                    section_counter += 1
                    content = self._normalize_text("\n\n".join(current_content_parts))
                    sections.append(
                        ExtractedSection(
                            section_id=f"sec_{section_counter}",
                            title=current_heading or f"Section {section_counter}",
                            section_type=SectionType(
                                self._detect_section_type(current_heading)
                            ),
                            content=content,
                            level=current_level,
                            word_count=self._count_words(content),
                            figure_ids=current_figure_ids.copy(),
                            table_ids=current_table_ids.copy(),
                        )
                    )
                    current_content_parts = []
                    current_figure_ids = []
                    current_table_ids = []

                current_heading = self._normalize_text(item.text if hasattr(item, "text") else str(item))
                current_level = _level if _level else 1

            elif item_type == "PictureItem" or (
                hasattr(item, "label") and "picture" in str(getattr(item, "label", "")).lower()
            ):
                figure_counter += 1
                fig_id = f"fig_{figure_counter}"
                caption = ""
                if hasattr(item, "caption") and item.caption:
                    caption = self._normalize_text(str(item.caption))
                elif hasattr(item, "text") and item.text:
                    caption = self._normalize_text(item.text)

                image_path = None
                if hasattr(item, "image") and item.image:
                    try:
                        img_filename = output_dir / f"figure_{figure_counter}.png"
                        if hasattr(item.image, "pil_image") and item.image.pil_image:
                            item.image.pil_image.save(img_filename)
                            image_path = img_filename
                    except Exception:
                        pass

                page_num = None
                if hasattr(item, "prov") and item.prov:
                    prov = item.prov[0] if isinstance(item.prov, list) else item.prov
                    if hasattr(prov, "page_no"):
                        page_num = prov.page_no

                figures.append(
                    ExtractedFigure(
                        figure_id=fig_id,
                        caption=caption,
                        page_number=page_num,
                        path=image_path,
                    )
                )
                current_figure_ids.append(fig_id)

            elif item_type == "TableItem" or (
                hasattr(item, "label") and "table" in str(getattr(item, "label", "")).lower()
            ):
                table_counter += 1
                tab_id = f"tab_{table_counter}"
                caption = ""
                content = ""
                if hasattr(item, "caption") and item.caption:
                    caption = self._normalize_text(str(item.caption))
                if hasattr(item, "export_to_markdown"):
                    try:
                        content = item.export_to_markdown(doc)
                    except TypeError:
                        content = item.export_to_markdown()
                elif hasattr(item, "text"):
                    content = self._normalize_text(item.text)

                tables.append(
                    ExtractedTable(
                        table_id=tab_id,
                        caption=caption,
                        content=content,
                    )
                )
                current_table_ids.append(tab_id)

            else:
                text = ""
                if hasattr(item, "text"):
                    text = item.text
                elif hasattr(item, "export_to_markdown"):
                    try:
                        text = item.export_to_markdown(doc)
                    except TypeError:
                        text = item.export_to_markdown()
                if text:
                    current_content_parts.append(self._normalize_text(text))

        if current_heading or current_content_parts:
            section_counter += 1
            content = self._normalize_text("\n\n".join(current_content_parts))
            sections.append(
                ExtractedSection(
                    section_id=f"sec_{section_counter}",
                    title=current_heading or f"Section {section_counter}",
                    section_type=SectionType(
                        self._detect_section_type(current_heading)
                    ),
                    content=content,
                    level=current_level,
                    word_count=self._count_words(content),
                    figure_ids=current_figure_ids.copy(),
                    table_ids=current_table_ids.copy(),
                )
            )

        title = ""
        authors = ""
        abstract = ""

        if hasattr(doc, "title") and doc.title:
            title = self._normalize_text(str(doc.title))
        elif sections:
            first = sections[0]
            if first.word_count < 30 and first.section_type == SectionType.OTHER:
                title = first.title
                sections = sections[1:]

        abstract_sections = [
            s for s in sections if s.section_type == SectionType.ABSTRACT
        ]
        if abstract_sections:
            abstract = abstract_sections[0].content

        total_words = sum(s.word_count for s in sections)

        return ParsedDocument(
            title=title,
            authors=authors,
            abstract=abstract,
            sections=sections,
            figures=figures,
            tables=tables,
            raw_text=raw_text,
            source_path=str(input_path),
            source_format="pdf",
            total_words=total_words,
            total_pages=doc.num_pages() if callable(getattr(doc, "num_pages", None)) else getattr(doc, "page_count", None),
        )
