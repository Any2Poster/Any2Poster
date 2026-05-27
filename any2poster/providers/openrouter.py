# OpenRouter client used for text, vision, and image generation requests.

from __future__ import annotations

import base64
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from any2poster.config import get_api_key, get_base_url, load_config


def _try_repair_json(raw: str, error_pos: int | None = None) -> "dict | None":
    """Repair a malformed JSON response using multiple strategies.

    Handles two common failure modes:
    1. Trailing-comma truncation: response cut after the last complete item
    2. Bad string content: unescaped quotes inside a value (common with code
       snippets in notebook QA). Uses error_pos to search only BEFORE the
       corruption, not after it.
    """
    # Determine where to stop searching — anything at/after the error is bad
    corrupt_at = error_pos if error_pos is not None else len(raw)

    # Strategy 1: strip last comma before the corruption and close container
    for end_char, opener in (("]", "["), ("}", "{")):
        idx = raw.rfind(",", 0, corrupt_at)
        if idx > 0:
            candidate = raw[:idx] + "\n" + end_char
            if raw.lstrip().startswith(opener):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

    # Strategy 2: find the last clean '}' BEFORE the corruption and close
    # the outer structure around it.
    search_start = min(corrupt_at - 1, len(raw) - 1)
    search_limit = max(0, search_start - 20000)
    found: list[int] = []
    for i in range(search_start, search_limit, -1):
        if raw[i] == "}":
            found.append(i)
        if len(found) >= 40:
            break

    for pos in found:
        fragment = raw[: pos + 1]
        for suffix in ["\n  ]\n}", "\n]\n}", "\n]}", "]}", ""]:
            try:
                return json.loads(fragment + suffix)
            except json.JSONDecodeError:
                continue

    # Strategy 3: scan character-by-character and escape unescaped internal
    # quotes. Handles verbatim dialogue copied into JSON string values.
    sanitized = _escape_internal_quotes(raw)
    if sanitized != raw:
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError as _e2:
            # Try strategies 1+2 on the sanitized version too
            result = _try_repair_json(sanitized, error_pos=_e2.pos)
            if result is not None:
                return result

    return None


def _escape_internal_quotes(raw: str) -> str:
    """Escape unescaped double-quote characters inside JSON string values.

    Walks character-by-character tracking string context. Any `"` that is not
    a string delimiter and is not already backslash-escaped is replaced with
    `\\"` so the result parses as valid JSON.
    """
    result: list[str] = []
    in_string = False
    i = 0
    while i < len(raw):
        c = raw[i]
        if c == "\\" and i + 1 < len(raw):
            result.append(c)
            result.append(raw[i + 1])
            i += 2
            continue
        if c == '"':
            if not in_string:
                in_string = True
                result.append(c)
            else:
                # Closing quote if followed (past whitespace) by :, ,, }, ], ", or EOI.
                # Include '"' because a value string is often followed by the next key.
                rest = raw[i + 1 :].lstrip()
                if not rest or rest[0] in (":", ",", "}", "]", '"'):
                    in_string = False
                    result.append(c)
                else:
                    result.append('\\"')
        else:
            result.append(c)
        i += 1
    return "".join(result)


class OpenRouterProvider:

    def __init__(
        self,
        llm_model: str = "",
        image_model: str = "",
        vision_model: str = "",
    ):
        config = load_config()
        self._api_key = get_api_key("openrouter")
        self._base_url = get_base_url("openrouter")
        self._llm_model = llm_model or config.get(
            "llm_model", "anthropic/claude-sonnet-4"
        )
        self._image_model = image_model or config.get(
            "image_model", "google/gemini-3-pro-image-preview"
        )
        self._vision_model = vision_model or config.get(
            "vision_model", "openai/gpt-4o-mini"
        )
        self._timeout = config.get("request_timeout", 120)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/any2poster",
            "X-Title": "any2poster",
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type(
            (requests.ConnectionError, requests.Timeout)
        ),
    )
    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: str | None = None,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
        )

        kwargs: dict[str, Any] = {
            "model": self._llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from LLM")
        return content.strip()

    def complete_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        raw = self.complete(
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json",
        )
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = lines[1:] if lines[0].startswith("```") else lines
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as _exc:
            repaired = _try_repair_json(raw, error_pos=_exc.pos)
            if repaired is not None:
                return repaired
            raise

    def complete_vision_json(
        self,
        system: str,
        user: str,
        image_path: Path,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        model_override: str | None = None,
    ) -> dict[str, Any]:
        """Run a vision-capable LLM call with an attached image and return JSON."""
        if not image_path or not image_path.exists():
            raise FileNotFoundError(f"Image not found for vision check: {image_path}")

        model = model_override or self._vision_model or self._llm_model

        b64, mime = self._encode_image_file(image_path)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]},
        ]

        from openai import OpenAI
        client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = lines[1:] if lines[0].startswith("```") else lines
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
        return json.loads(content)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=3, min=5, max=60),
        retry=retry_if_exception_type(
            (requests.ConnectionError, requests.Timeout)
        ),
    )
    def generate_image(
        self,
        prompt: str,
        output_path: Path,
        reference_image: Path | None = None,
        aspect_ratio: str = "16:9",
        debug_payload_path: Path | None = None,
    ) -> Path:
        messages_content: list[dict[str, Any]] = []

        if reference_image and reference_image.exists():
            b64, mime = self._encode_image_file(reference_image)
            messages_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )

        messages_content.append({"type": "text", "text": prompt})

        payload: dict[str, Any] = {
            "model": self._image_model,
            "messages": [{"role": "user", "content": messages_content}],
            "modalities": ["image", "text"],
        }

        if aspect_ratio:
            payload["image_config"] = {"aspect_ratio": aspect_ratio}

        if debug_payload_path:
            debug_payload_path.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_payload_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

        resp = requests.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        image_data = self._extract_image_data(data)
        if not image_data:
            raise ValueError(
                f"No image in response from {self._image_model}. "
                f"Keys: {list(data.keys())}"
            )

        img = self._decode_image(image_data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(output_path), quality=95)
        return output_path

    def _encode_image_file(self, path: Path) -> tuple[str, str]:
        with open(path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        suffix = path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
        return b64, mime_map.get(suffix, "image/png")

    def _extract_image_data(self, data: dict[str, Any]) -> str | None:
        try:
            choices = data.get("choices", [])
            if not choices:
                return None
            message = choices[0].get("message", {})

            images = message.get("images", [])
            if images:
                first = images[0]
                if isinstance(first, dict):
                    url = first.get("image_url", {}).get("url")
                    if url:
                        return url
                    return first.get("url") or first.get("data")
                if isinstance(first, str):
                    return first

            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type", "")
                    if ptype == "image_url":
                        url = part.get("image_url", {}).get("url")
                        if url:
                            return url
                    if ptype == "image":
                        src = part.get("source", {})
                        if isinstance(src, dict) and src.get("data"):
                            return src["data"]
                        if part.get("data"):
                            return part["data"]

            if isinstance(content, str):
                if content.startswith("data:image"):
                    return content
                if len(content) > 1000 and not content.startswith("{"):
                    try:
                        base64.b64decode(content[:100])
                        return content
                    except Exception:
                        pass

        except (KeyError, IndexError, TypeError):
            pass

        return None

    def _decode_image(self, image_data: str) -> Image.Image:
        if image_data.startswith("data:"):
            _, b64_data = image_data.split(",", 1)
            img_bytes = base64.b64decode(b64_data)
        elif image_data.startswith("http"):
            resp = requests.get(image_data, timeout=30)
            resp.raise_for_status()
            img_bytes = resp.content
        else:
            img_bytes = base64.b64decode(image_data)

        return Image.open(BytesIO(img_bytes))
