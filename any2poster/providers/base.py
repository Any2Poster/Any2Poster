"""Abstract base provider interface.

All LLM and image generation providers implement this contract.
Currently implemented: OpenRouterProvider (single key for both).
Future: DirectAnthropicProvider, DirectGoogleProvider, OpenAIProvider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseLLMProvider(ABC):

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: str | None = None,
    ) -> str:
        pass

    @abstractmethod
    def complete_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        pass


class BaseImageProvider(ABC):

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        output_path: Path,
        reference_image: Path | None = None,
        aspect_ratio: str = "16:9",
    ) -> Path:
        pass