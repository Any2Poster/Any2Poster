"""Document parsers for any input format.

Supported formats:
- PDF via Docling (primary) or marker-pdf (fallback)
- DOCX via python-docx
- Markdown / plain text
- URL / HTML via trafilatura
- PPTX via python-pptx
- Video URLs (YouTube, Vimeo) and local video files via youtube-transcript-api / yt-dlp / Whisper
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from any2poster.models import ParsedDocument
    from any2poster.parsers.base import BaseParser

_VIDEO_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.|m\.)?(?:"
    r"youtube\.com/watch|youtu\.be/|youtube\.com/embed/|youtube\.com/v/|"
    r"vimeo\.com/\d|player\.vimeo\.com/video/"
    r")",
    re.IGNORECASE,
)


def get_parser(input_path: str, parser_override: str = "auto") -> "BaseParser":
    if input_path.startswith(("http://", "https://")):
        if _VIDEO_URL_RE.search(input_path):
            from any2poster.parsers.video_parser import VideoParser
            return VideoParser()
        from any2poster.parsers.url_parser import URLParser
        return URLParser()

    path = Path(input_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return _get_pdf_parser(parser_override)

    parser_map: dict[str, type] = {}

    try:
        from any2poster.parsers.docx_parser import DOCXParser
        parser_map[".docx"] = DOCXParser
        parser_map[".doc"] = DOCXParser
    except ImportError:
        pass

    try:
        from any2poster.parsers.markdown_parser import MarkdownParser
        parser_map[".md"] = MarkdownParser
        parser_map[".markdown"] = MarkdownParser
        parser_map[".txt"] = MarkdownParser
    except ImportError:
        pass

    try:
        from any2poster.parsers.url_parser import URLParser
        parser_map[".html"] = URLParser
        parser_map[".htm"] = URLParser
    except ImportError:
        pass

    try:
        from any2poster.parsers.pptx_parser import PPTXParser
        parser_map[".pptx"] = PPTXParser
    except ImportError:
        pass

    try:
        from any2poster.parsers.video_parser import VideoParser
        for _vext in (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv", ".wmv"):
            parser_map[_vext] = VideoParser
    except ImportError:
        pass

    try:
        from any2poster.parsers.latex_parser import LaTeXParser
        parser_map[".tex"] = LaTeXParser
    except ImportError:
        pass

    try:
        from any2poster.parsers.notebook_parser import NotebookParser
        parser_map[".ipynb"] = NotebookParser
    except ImportError:
        pass

    parser_class = parser_map.get(ext)
    if parser_class is None:
        supported = [".pdf"] + list(parser_map.keys())
        raise ValueError(
            f"Unsupported file format: {ext}\n"
            f"Supported formats: {', '.join(sorted(set(supported)))}"
        )

    return parser_class()


def _get_pdf_parser(parser_override: str) -> "BaseParser":
    if parser_override == "marker":
        from any2poster.parsers.pdf_parser import PDFParser
        return PDFParser()

    if parser_override == "docling":
        from any2poster.parsers.docling_parser import DoclingParser
        return DoclingParser()

    try:
        from any2poster.parsers.docling_parser import DoclingParser
        return DoclingParser()
    except ImportError:
        pass

    try:
        from any2poster.parsers.pdf_parser import PDFParser
        return PDFParser()
    except ImportError:
        pass

    raise ImportError(
        "No PDF parser available. Install one of:\n"
        "  pip install docling        (recommended)\n"
        "  pip install marker-pdf     (fallback)"
    )


def parse_document(
    input_path: str,
    output_dir: Path,
    parser_override: str = "auto",
) -> "ParsedDocument":
    parser = get_parser(input_path, parser_override)
    return parser.parse(input_path, output_dir)
