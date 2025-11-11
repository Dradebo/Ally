"""OpenAI provider implementation."""

from typing import Any, Optional
from app.src.providers.base_provider import BaseProvider


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI's GPT models."""

    @property
    def name(self) -> str:
        return "openai"

    def create_llm(
        self,
        model_name: str,
        temperature: float,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Create OpenAI LLM instance.

        Args:
            model_name: OpenAI model name (e.g., "gpt-4o", "gpt-4o-mini")
            temperature: Temperature for generation
            api_key: OpenAI API key
            **kwargs: Additional configuration options

        Returns:
            ChatOpenAI instance
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai package is required for OpenAI provider. "
                "Install it with: pip install langchain-openai"
            )

        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            timeout=None,
            max_retries=5,
            api_key=api_key,
            **kwargs
        )

    def get_required_env_vars(self) -> list[str]:
        return ["OPENAI_API_KEY"]

    def validate_config(self, config: dict) -> tuple[bool, Optional[str]]:
        """Validate OpenAI configuration."""
        model_name = config.get("model_name")
        if not model_name:
            return False, "model_name is required for OpenAI provider"

        api_key = config.get("api_key")
        if not api_key:
            return False, "OPENAI_API_KEY environment variable must be set"

        return True, None

    def get_display_name(self) -> str:
        return "OpenAI"

    def get_default_model(self) -> Optional[str]:
        return "gpt-4o"

    def get_available_models(self) -> list[str]:
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ]
