"""Base parser interface for all document parsers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from any2poster.models import ParsedDocument


SECTION_TYPE_MAP: dict[str, str] = {
    "abstract": "abstract",
    "introduction": "introduction",
    "background": "background",
    "related work": "background",
    "literature review": "background",
    "prior work": "background",
    "preliminary": "background",
    "preliminaries": "background",
    "method": "methods",
    "methods": "methods",
    "methodology": "methods",
    "approach": "methods",
    "model": "methods",
    "proposed method": "methods",
    "framework": "methods",
    "architecture": "methods",
    "algorithm": "methods",
    "formulation": "methods",
    "experiment": "experiments",
    "experiments": "experiments",
    "experimental setup": "experiments",
    "experimental settings": "experiments",
    "setup": "experiments",
    "implementation": "experiments",
    "implementation details": "experiments",
    "evaluation": "experiments",
    "result": "results",
    "results": "results",
    "findings": "results",
    "main results": "results",
    "quantitative results": "results",
    "qualitative results": "results",
    "ablation": "results",
    "ablation study": "results",
    "discussion": "discussion",
    "analysis": "discussion",
    "limitation": "discussion",
    "limitations": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "summary": "conclusion",
    "concluding remarks": "conclusion",
    "future work": "conclusion",
    "conclusion and future work": "conclusion",
    "reference": "references",
    "references": "references",
    "bibliography": "references",
    "acknowledgment": "acknowledgments",
    "acknowledgments": "acknowledgments",
    "acknowledgements": "acknowledgments",
}


class BaseParser(ABC):

    @abstractmethod
    def parse(self, input_path: str, output_dir: Path) -> "ParsedDocument":
        pass

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        pass

    def _detect_section_type(self, title: str) -> str:
        cleaned = title.lower().strip()
        cleaned = cleaned.lstrip("0123456789.-) ").strip()
        if cleaned in SECTION_TYPE_MAP:
            return SECTION_TYPE_MAP[cleaned]
        for keyword, section_type in SECTION_TYPE_MAP.items():
            if keyword in cleaned:
                return section_type
        return "other"

    def _normalize_text(self, text: str) -> str:
        try:
            import ftfy
            text = ftfy.fix_text(text)
        except ImportError:
            pass
        text = text.replace("\x00", "")
        text = text.replace("\ufffd", "")
        return text

    def _count_words(self, text: str) -> int:
        return len(text.split())