
from pathlib import Path
import re
from typing import TYPE_CHECKING

from any2poster.parsers.base import BaseParser
from any2poster.models import (
    ParsedDocument,
    ExtractedSection,
    ExtractedFigure,
    ExtractedTable,
    SectionType,
)

if TYPE_CHECKING:
    pass


class PDFParser(BaseParser):
    
    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]
    
    def parse(self, input_path: str, output_dir: Path) -> ParsedDocument:
        #trying api models
        
        try:
            from marker.convert import convert_single_pdf  # type: ignore
            from marker.models import load_all_models  # type: ignore
            use_legacy_api = True
        except Exception:
            use_legacy_api = False
        
        if not use_legacy_api:
            try:
                from marker.models import create_model_dict  # type: ignore
                from marker.converters.pdf import PdfConverter  # type: ignore
                from marker.output import text_from_rendered  # type: ignore
            except Exception:
                raise ImportError(
                    "marker-pdf is required for PDF parsing. "
                    "Install with: pip install marker-pdf"
                )
        
        input_path = Path(input_path)
        
        if use_legacy_api:
            models = load_all_models()
            
            full_text, images, metadata = convert_single_pdf(
                str(input_path),
                models,
                max_pages=None,
                parallel_factor=1,
            )
        else:
            models = create_model_dict()
            converter = PdfConverter(
                artifact_dict=models,
                config={},
            )
            rendered = converter(str(input_path))
            full_text, _ext, images = text_from_rendered(rendered)
            metadata = getattr(rendered, "metadata", {}) or {}
        
        figures = []
        figures_dir = output_dir
        figures_dir.mkdir(exist_ok=True)
        
        for img_name, img_data in images.items():
            img_path = figures_dir / img_name

            img_data.save(img_path)
            
            figures.append(ExtractedFigure(
                figure_id=img_name.replace(".png", "").replace(".jpg", ""),
                path=img_path,
                caption="",
            ))
        
        # Parse the markdown text into sections
        sections = self._parse_sections(full_text)
        
        # Try to extract title and authors from first section or metadata
        title = self._extract_title(full_text, metadata)
        authors_list = self._extract_authors(full_text, metadata)
        authors = ", ".join(authors_list) if isinstance(authors_list, list) else str(authors_list or "")
        
        # Match figure captions to figures
        self._extract_figure_captions(full_text, figures)
        
        # Extract tables
        tables = self._extract_tables(full_text)
        
        page_count = metadata.get("page_count")
        if page_count is None and not use_legacy_api:
            page_count = getattr(converter, "page_count", None)
        
        return ParsedDocument(
            source_path=str(input_path),
            source_format="pdf",
            title=title,
            authors=authors,
            sections=sections,
            figures=figures,
            tables=tables,
            raw_text=full_text,
            total_pages=page_count,
            total_words=self._count_words(full_text),
        )
    
    def _parse_sections(self, text: str) -> list[ExtractedSection]:
        sections = []
        
        # Split by markdown headers
        # Match ## Header or # Header
        header_pattern = r'^(#{1,3})\s+(.+)$'
        
        lines = text.split('\n')
        current_section = None
        current_content = []
        section_id = 0
        
        for line in lines:
            header_match = re.match(header_pattern, line)
            
            if header_match:
                # Save previous section
                if current_section:
                    current_section.content = '\n'.join(current_content).strip()
                    if current_section.content:  # Only add if has content
                        sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                section_id += 1
                
                current_section = ExtractedSection(
                    section_id=f"section_{section_id}",
                    title=title,
                    content="",
                    section_type=SectionType(self._detect_section_type(title)),
                    level=level,
                )
                current_content = []
            else:
                current_content.append(line)
        
        # Don't forget the last section
        if current_section:
            current_section.content = '\n'.join(current_content).strip()
            if current_section.content:
                sections.append(current_section)
        
        # If no sections found, create one with all content
        if not sections and text.strip():
            sections.append(ExtractedSection(
                section_id="section_1",
                title="Content",
                content=text.strip(),
                section_type=SectionType.OTHER,
                level=1,
            ))
        
        return sections
    
    def _extract_title(self, text: str, metadata: dict) -> str:
        # Try metadata first
        if metadata.get("title"):
            return metadata["title"]
        
        # Try first line if it looks like a title
        lines = text.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            # Remove markdown header symbols
            first_line = re.sub(r'^#+\s*', '', first_line)
            if len(first_line) < 200:  # Reasonable title length
                return first_line
        
        return "Untitled Document"
    
    def _extract_authors(self, text: str, metadata: dict) -> list[str]:
        """Extract author names."""
        # Try metadata first
        if metadata.get("authors"):
            return metadata["authors"]
        
        # Look for author patterns in first few lines
        lines = text.strip().split('\n')[:20]
        
        for line in lines:
            # Look for lines with multiple names separated by commas or "and"
            if re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+.*,', line):
                # Clean up and split
                line = re.sub(r'\*+', '', line)  # Remove asterisks
                line = re.sub(r'\d+', '', line)  # Remove numbers
                
                # Split by comma or "and"
                parts = re.split(r',|\s+and\s+', line)
                authors = [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]
                
                if len(authors) >= 1:
                    return authors[:10]  # Limit to 10 authors
        
        return []
    
    def _extract_figure_captions(self, text: str, figures: list[ExtractedFigure]) -> None:
        """Extract and match figure captions to figures."""
        # Look for "Figure N: caption" or "Fig. N. caption" patterns
        caption_pattern = r'(?:Figure|Fig\.?)\s*(\d+)[.:\s]+([^\n]+)'
        
        matches = re.findall(caption_pattern, text, re.IGNORECASE)
        
        for fig_num, caption in matches:
            # Try to match to a figure
            for fig in figures:
                if fig_num in fig.figure_id:
                    fig.caption = caption.strip()
                    break
    
    def _extract_tables(self, text: str) -> list[ExtractedTable]:
        """Extract tables from markdown text."""
        tables = []
        
        # Look for markdown tables (lines with |)
        table_pattern = r'(\|[^\n]+\|\n)+(\|[-:| ]+\|\n)?(\|[^\n]+\|\n)+'
        
        matches = re.findall(table_pattern, text)
        
        for i, match in enumerate(matches):
            table_content = ''.join(match)
            
            # Try to find caption above or below table
            caption = f"Table {i + 1}"
            
            tables.append(ExtractedTable(
                table_id=f"table_{i + 1}",
                caption=caption,
                content=table_content,
            ))
        
        return tables
