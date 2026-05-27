"""
Google provider for Gemini LLM and Nano Banana Pro image generation.

This provider uses the official Google Generative AI SDK.
"""

import base64
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from any2poster.providers.base import BaseLLMProvider, BaseImageProvider
from any2poster.config import get_api_key, load_config


class GoogleProvider(BaseLLMProvider):
    """Google Gemini LLM provider."""
    
    def __init__(self, model: str = "gemini-pro"):
        super().__init__(model)
        
        api_key = get_api_key("google")
        genai.configure(api_key=api_key)
        
        self.model_instance = genai.GenerativeModel(model)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate completion using Google Gemini."""
        
        # Combine system and user prompt
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        
        response = self.model_instance.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        
        return response.text


class GoogleImageProvider(BaseImageProvider):
    """
    Google Nano Banana Pro image generation provider.
    
    Uses the Gemini 3 Pro Image model for high-quality image generation
    with excellent text rendering.
    """
    
    def __init__(self, model: str = "gemini-3-pro-image-preview"):
        super().__init__(model)
        
        api_key = get_api_key("google")
        genai.configure(api_key=api_key)
        
        # Use the image-capable model
        self.model_instance = genai.GenerativeModel(model)
        self.config = load_config()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
    )
    def generate_image(
        self,
        prompt: str,
        reference_image: Optional[Path] = None,
        width_px: int = 1024,
        height_px: int = 1024,
    ) -> bytes:
        """
        Generate image using Nano Banana Pro.
        
        The Gemini 3 Pro Image model has excellent text rendering capabilities
        (94% accuracy) making it ideal for poster panel generation.
        """
        
        # Build content parts
        content_parts = []
        
        # Add reference image if provided
        if reference_image and reference_image.exists():
            image_data = reference_image.read_bytes()
            
            # Determine mime type
            suffix = reference_image.suffix.lower()
            mime_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }.get(suffix, "image/png")
            
            content_parts.append({
                "mime_type": mime_type,
                "data": base64.b64encode(image_data).decode()
            })
        
        # Add text prompt
        full_prompt = self._build_image_prompt(prompt, width_px, height_px)
        content_parts.append(full_prompt)
        
        # Generate
        response = self.model_instance.generate_content(
            content_parts,
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,  # Lower temperature for more consistent output
            ),
        )
        
        # Extract image from response
        for part in response.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                return part.inline_data.data
        
        # If no inline data, check for text response with base64
        if response.text:
            image_bytes = self._extract_image_from_text(response.text)
            if image_bytes:
                return image_bytes
        
        raise ValueError("No image generated in response")
    
    def _build_image_prompt(
        self, 
        prompt: str, 
        width_px: int, 
        height_px: int
    ) -> str:
        """Build optimized prompt for Nano Banana Pro."""
        
        # Nano Banana Pro responds well to structured, explicit prompts
        return f"""Generate a high-quality image for an academic poster panel.

SPECIFICATIONS:
- Resolution: {width_px}x{height_px} pixels
- Purpose: Academic conference poster
- Quality: Publication-ready, professional

CONTENT:
{prompt}

CRITICAL TEXT RENDERING INSTRUCTIONS:
- ALL text must be spelled EXACTLY as provided
- Use clear, legible sans-serif fonts (like Helvetica or Arial)
- Ensure high contrast between text and background
- Text should be large enough to read from 3 feet away
- Do NOT paraphrase, abbreviate, or modify any specified text

Generate the image now."""
    
    def _extract_image_from_text(self, text: str) -> bytes | None:
        """Try to extract base64 image data from text response."""
        import re
        
        # Look for base64 data
        match = re.search(r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)', text)
        if match:
            try:
                return base64.b64decode(match.group(1))
            except:
                pass
        
        return None
