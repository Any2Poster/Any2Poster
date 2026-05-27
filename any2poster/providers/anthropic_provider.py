"""
Anthropic provider for Claude LLM completions.
"""

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from any2poster.providers.base import BaseLLMProvider
from any2poster.config import get_api_key


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider."""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        super().__init__(model)
        
        api_key = get_api_key("anthropic")
        
        self.client = anthropic.Anthropic(api_key=api_key)
    
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
        """Generate completion using Anthropic Claude."""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system if system else "You are a helpful assistant.",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        
        return response.content[0].text
