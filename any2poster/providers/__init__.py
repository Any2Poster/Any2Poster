"""LLM and image generation providers.

Currently supports OpenRouter (recommended) which provides
unified access to both LLM (Claude) and image generation
(Nano Banana Pro) with a single API key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from any2poster.providers.openrouter import OpenRouterProvider


def get_provider(
    provider_name: str = "openrouter",
    llm_model: str = "",
    image_model: str = "",
) -> "OpenRouterProvider":
    if provider_name == "openrouter":
        from any2poster.providers.openrouter import OpenRouterProvider
        return OpenRouterProvider(llm_model=llm_model, image_model=image_model)
    raise ValueError(f"Unknown provider: {provider_name}. Use 'openrouter'.")