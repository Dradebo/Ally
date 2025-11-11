"""Anthropic provider implementation."""

from typing import Any, Optional
from app.src.providers.base_provider import BaseProvider


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic's Claude models."""

    @property
    def name(self) -> str:
        return "anthropic"

    def create_llm(
        self,
        model_name: str,
        temperature: float,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Create Anthropic LLM instance.

        Args:
            model_name: Claude model name (e.g., "claude-sonnet-4-5-20250929")
            temperature: Temperature for generation
            api_key: Anthropic API key
            **kwargs: Additional configuration options

        Returns:
            ChatAnthropic instance
        """
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic package is required for Anthropic provider. "
                "Install it with: pip install langchain-anthropic"
            )

        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            timeout=None,
            max_retries=5,
            api_key=api_key,
            **kwargs
        )

    def get_required_env_vars(self) -> list[str]:
        return ["ANTHROPIC_API_KEY"]

    def validate_config(self, config: dict) -> tuple[bool, Optional[str]]:
        """Validate Anthropic configuration."""
        model_name = config.get("model_name")
        if not model_name:
            return False, "model_name is required for Anthropic provider"

        api_key = config.get("api_key")
        if not api_key:
            return False, "ANTHROPIC_API_KEY environment variable must be set"

        return True, None

    def get_display_name(self) -> str:
        return "Anthropic (Claude)"

    def get_default_model(self) -> Optional[str]:
        return "claude-sonnet-4-5-20250929"

    def get_available_models(self) -> list[str]:
        return [
            "claude-sonnet-4-5-20250929",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ]
