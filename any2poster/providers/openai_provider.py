"""
OpenAI provider for LLM completions.
"""

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from any2poster.providers.base import BaseLLMProvider
from any2poster.config import get_api_key, get_base_url


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""
    
    def __init__(self, model: str = "gpt-4o"):
        super().__init__(model)
        
        api_key = get_api_key("openai")
        base_url = get_base_url("openai")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url if base_url else None,
        )
    
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
        """Generate completion using OpenAI."""
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return response.choices[0].message.content
