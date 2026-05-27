# Loads config and API keys for model providers.

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_CONFIG_CACHE: dict[str, Any] | None = None

_DEFAULTS: dict[str, Any] = {
    "llm_model": "anthropic/claude-sonnet-4",
    "image_model": "google/gemini-3-pro-image-preview",
    "vision_model": "openai/gpt-4o-mini",
    "base_url": "https://openrouter.ai/api/v1",
    "request_timeout": 120,
    "max_retries": 3,
}

_ENV_KEYS: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}

_BASE_URLS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta",
}


def load_config() -> dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    _load_dotenv()

    config = dict(_DEFAULTS)

    env_overrides = {
        "llm_model": os.getenv("ANY2POSTER_LLM_MODEL"),
        "image_model": os.getenv("ANY2POSTER_IMAGE_MODEL"),
        "vision_model": os.getenv("ANY2POSTER_VISION_MODEL"),
        "base_url": os.getenv("OPENROUTER_BASE_URL"),
        "request_timeout": os.getenv("ANY2POSTER_TIMEOUT"),
        "max_retries": os.getenv("ANY2POSTER_MAX_RETRIES"),
    }

    for key, val in env_overrides.items():
        if val is not None:
            if key in ("request_timeout", "max_retries"):
                try:
                    config[key] = int(val)
                except ValueError:
                    pass
            else:
                config[key] = val

    _CONFIG_CACHE = config
    return config


def get_api_key(provider: str = "openrouter") -> str:
    _load_dotenv()
    env_var = _ENV_KEYS.get(provider, f"{provider.upper()}_API_KEY")
    key = os.getenv(env_var, "")
    if not key:
        raise ValueError(
            f"API key not found. Set {env_var} in your environment or .env file."
        )
    # Guard against accidental trailing newlines/spaces in .env or shell exports.
    return key.strip()


def get_base_url(provider: str = "openrouter") -> str:
    _load_dotenv()
    env_url = os.getenv(f"{provider.upper()}_BASE_URL")
    if env_url:
        return env_url.strip()
    return _BASE_URLS.get(provider, _BASE_URLS["openrouter"])


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            parent_env = Path.cwd().parent / ".env"
            if parent_env.exists():
                load_dotenv(parent_env)
    except ImportError:
        pass
