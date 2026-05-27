"""
URL/HTML parser using trafilatura for web content extraction.
"""

from pathlib import Path
import re
from typing import TYPE_CHECKING

from any2poster.parsers.base import BaseParser
from any2poster.models import (
    ParsedDocument,
    ExtractedSection,
    ExtractedFigure,
    SectionType,
)

if TYPE_CHECKING:
    pass


class URLParser(BaseParser):
    """Parse web pages and HTML files using trafilatura."""
    
    @property
    def supported_extensions(self) -> list[str]:
        return [".html", ".htm"]
    
    def parse(self, input_path: str, output_dir: Path) -> ParsedDocument:
        """
        Parse a URL or HTML file and extract content.
        
        Uses trafilatura which is excellent at extracting main content
        from web pages while ignoring navigation, ads, etc.
        """
        try:
            import trafilatura
            from trafilatura.settings import use_config
        except ImportError:
            raise ImportError(
                "trafilatura is required for URL parsing. "
                "Install with: pip install trafilatura"
            )
        
        # Configure trafilatura
        config = use_config()
        config.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")
        
        # Fetch and extract
        if input_path.startswith(('http://', 'https://')):
            # It's a URL
            downloaded = trafilatura.fetch_url(input_path)
            if not downloaded:
                raise ValueError(f"Could not download URL: {input_path}")
            source_format = "url"
        else:
            # It's a local HTML file
            downloaded = Path(input_path).read_text(encoding="utf-8")
            source_format = "html"
        
        # Extract main content
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            include_images=True,
            include_links=False,
            output_format="markdown",
            config=config,
        )
        
        if not text:
            raise ValueError(f"Could not extract content from: {input_path}")
        
        # Get metadata
        metadata = trafilatura.extract_metadata(downloaded)
        
        # Parse into sections
        sections = self._parse_sections(text)
        
        # Extract and download images from raw HTML
        figures = self._extract_images(downloaded, input_path, output_dir)
        
        # Get title
        title = "Untitled"
        if metadata:
            title = metadata.title or metadata.sitename or "Untitled"
        elif sections:
            title = sections[0].title
        
        # Get authors
        authors = ""
        if metadata and metadata.author:
            authors = ", ".join(a.strip() for a in metadata.author.split(',') if a.strip())
        
        return ParsedDocument(
            source_path=input_path,
            source_format=source_format,
            title=title,
            authors=authors,
            sections=sections,
            figures=figures,
            tables=[],
            raw_text=text,
            total_words=self._count_words(text),
        )
    
    def _parse_sections(self, text: str) -> list[ExtractedSection]:
        """Parse markdown text into sections."""
        sections = []
        
        # Split by headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        lines = text.split('\n')
        current_section = None
        current_content = []
        section_id = 0
        
        for line in lines:
            header_match = re.match(header_pattern, line)
            
            if header_match:
                if current_section:
                    current_section.content = '\n'.join(current_content).strip()
                    if current_section.content:
                        sections.append(current_section)
                
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
        
        if current_section:
            current_section.content = '\n'.join(current_content).strip()
            if current_section.content:
                sections.append(current_section)
        
        if not sections and text.strip():
            sections.append(ExtractedSection(
                section_id="section_1",
                title="Content",
                content=text.strip(),
                section_type=SectionType.OTHER,
                level=1,
            ))
        
        return sections
    
    def _extract_images(self, raw_html: str, base_url: str, output_dir: Path) -> list[ExtractedFigure]:
        """Download images found in the raw HTML into output_dir."""
        figures = []
        seen: set[str] = set()
        counter = 0

        for tag_match in re.finditer(r'<img\s[^>]*?>', raw_html, re.IGNORECASE | re.DOTALL):
            tag = tag_match.group(0)
            src_m = re.search(r'\bsrc=["\']([^"\']+)["\']', tag, re.IGNORECASE)
            if not src_m:
                continue
            src = src_m.group(1).strip()
            alt_m = re.search(r'\balt=["\']([^"\']*)["\']', tag, re.IGNORECASE)
            alt = alt_m.group(1) if alt_m else ""

            if not src or src in seen or src.startswith('data:'):
                continue
            seen.add(src)

            img_path = self._fetch_image(src, base_url, output_dir, counter + 1)
            if img_path is None:
                continue
            counter += 1
            figures.append(ExtractedFigure(
                figure_id=f"fig_{counter}",
                caption=alt,
                path=img_path,
            ))

        return figures

    def _fetch_image(self, src: str, base_url: str, output_dir: Path, idx: int) -> "Path | None":
        from urllib.parse import urljoin

        lower = src.lower()
        if any(lower.endswith(ext) for ext in ('.svg', '.ico')):
            return None

        if base_url.startswith(('http://', 'https://')):
            full_url = urljoin(base_url, src)
            if not full_url.startswith(('http://', 'https://')):
                return None
            return self._download_image(full_url, output_dir, idx)
        else:
            if src.startswith(('http://', 'https://')):
                return self._download_image(src, output_dir, idx)
            base_dir = Path(base_url).parent if base_url else Path('.')
            candidate = (base_dir / src).resolve()
            if candidate.exists():
                return self._copy_image(candidate, output_dir, idx)
            return None

    def _download_image(self, url: str, output_dir: Path, idx: int) -> "Path | None":
        try:
            import requests
            resp = requests.get(
                url, timeout=10,
                headers={'User-Agent': 'Mozilla/5.0'},
                stream=True,
            )
            if resp.status_code != 200:
                return None
            ct = resp.headers.get('content-type', '')
            ext = self._ext_from_content_type(ct) or self._ext_from_url(url) or 'jpg'
            path = output_dir / f"web_fig_{idx}.{ext}"
            path.write_bytes(resp.content)
            if not self._is_image_large_enough(path, min_px=100):
                path.unlink(missing_ok=True)
                return None
            return path
        except Exception:
            return None

    def _copy_image(self, src_path: Path, output_dir: Path, idx: int) -> "Path | None":
        try:
            import shutil
            dest = output_dir / f"web_fig_{idx}{src_path.suffix}"
            shutil.copy2(src_path, dest)
            if not self._is_image_large_enough(dest, min_px=100):
                dest.unlink(missing_ok=True)
                return None
            return dest
        except Exception:
            return None

    def _is_image_large_enough(self, path: Path, min_px: int = 100) -> bool:
        try:
            from PIL import Image
            with Image.open(path) as img:
                w, h = img.size
            return w >= min_px and h >= min_px
        except Exception:
            return True

    def _ext_from_content_type(self, ct: str) -> str:
        for mime, ext in (
            ('image/jpeg', 'jpg'), ('image/png', 'png'),
            ('image/gif', 'gif'), ('image/webp', 'webp'),
        ):
            if mime in ct:
                return ext
        return ''

    def _ext_from_url(self, url: str) -> str:
        from urllib.parse import urlparse
        suffix = Path(urlparse(url).path).suffix.lstrip('.').lower()
        return suffix if suffix in ('jpg', 'jpeg', 'png', 'gif', 'webp') else ''
