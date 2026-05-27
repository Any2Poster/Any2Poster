"""
Markdown parser using markdown-it-py.
"""

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


class MarkdownParser(BaseParser):
    """Parse Markdown files."""
    
    @property
    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown", ".txt"]
    
    def parse(self, input_path: str, output_dir: Path) -> ParsedDocument:
        """Parse a Markdown file and extract content."""
        input_path = Path(input_path)
        
        # Read the file
        text = input_path.read_text(encoding="utf-8")
        
        # Extract sections
        sections = self._parse_sections(text)
        
        # Extract images (references in markdown)
        figures = self._extract_images(text, input_path.parent, output_dir)
        
        # Extract tables
        tables = self._extract_tables(text)
        
        # Get title
        title = self._extract_title(text, sections)
        
        # Get authors (look for YAML frontmatter)
        authors_list = self._extract_authors(text)
        authors = ", ".join(authors_list) if isinstance(authors_list, list) else str(authors_list or "")
        
        return ParsedDocument(
            source_path=str(input_path),
            source_format="markdown",
            title=title,
            authors=authors,
            sections=sections,
            figures=figures,
            tables=tables,
            raw_text=text,
            total_words=self._count_words(text),
        )
    
    def _parse_sections(self, text: str) -> list[ExtractedSection]:
        """Parse markdown into sections based on headers."""
        sections = []
        
        # Remove YAML frontmatter if present
        text = re.sub(r'^---\n.*?\n---\n', '', text, flags=re.DOTALL)
        
        # Split by markdown headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
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
                    if current_section.content:
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
        
        # Save last section
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
    
    def _extract_images(
        self, 
        text: str, 
        source_dir: Path,
        output_dir: Path
    ) -> list[ExtractedFigure]:
        """Extract image references from markdown."""
        figures = []
        figures_dir = output_dir
        figures_dir.mkdir(exist_ok=True)
        
        # Find markdown image syntax: ![alt](path) or ![alt](url)
        img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        
        matches = re.findall(img_pattern, text)
        
        for i, (alt_text, img_path) in enumerate(matches):
            # Check if it's a local file
            if not img_path.startswith(('http://', 'https://')):
                full_path = source_dir / img_path
                
                if full_path.exists():
                    # Copy to output directory
                    dest_path = figures_dir / f"figure_{i + 1}{full_path.suffix}"
                    dest_path.write_bytes(full_path.read_bytes())
                    
                    figures.append(ExtractedFigure(
                        figure_id=f"fig_{i + 1}",
                        path=dest_path,
                        caption=alt_text,
                    ))
            else:
                # URL - we could download, but skip for now
                figures.append(ExtractedFigure(
                    figure_id=f"fig_{i + 1}",
                    path=None,
                    caption=alt_text,
                ))
        
        return figures
    
    def _extract_tables(self, text: str) -> list[ExtractedTable]:
        """Extract markdown tables."""
        tables = []
        
        # Look for markdown table blocks
        table_pattern = r'(\|[^\n]+\|\n)(\|[-:| ]+\|\n)(\|[^\n]+\|\n)+'
        
        for i, match in enumerate(re.finditer(table_pattern, text)):
            tables.append(ExtractedTable(
                table_id=f"table_{i + 1}",
                caption=f"Table {i + 1}",
                content=match.group(0),
            ))
        
        return tables
    
    def _extract_title(self, text: str, sections: list[ExtractedSection]) -> str:
        """Extract document title."""
        # Check YAML frontmatter
        frontmatter_match = re.search(r'^---\n(.*?)\n---', text, re.DOTALL)
        if frontmatter_match:
            yaml_text = frontmatter_match.group(1)
            title_match = re.search(r'^title:\s*["\']?([^"\'\n]+)', yaml_text, re.MULTILINE)
            if title_match:
                return title_match.group(1).strip()
        
        # Use first h1 header
        h1_match = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
        if h1_match:
            return h1_match.group(1).strip()
        
        # Use first section
        if sections:
            return sections[0].title
        
        return "Untitled Document"
    
    def _extract_authors(self, text: str) -> list[str]:
        """Extract authors from YAML frontmatter."""
        frontmatter_match = re.search(r'^---\n(.*?)\n---', text, re.DOTALL)
        if frontmatter_match:
            yaml_text = frontmatter_match.group(1)
            
            # Look for author or authors field
            author_match = re.search(r'^authors?:\s*(.+)$', yaml_text, re.MULTILINE)
            if author_match:
                authors_str = author_match.group(1).strip()
                # Handle YAML list format
                if authors_str.startswith('['):
                    authors_str = authors_str.strip('[]')
                return [a.strip().strip('"\'') for a in authors_str.split(',')]
        
        return []
