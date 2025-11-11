"""Cerebras provider implementation."""

from typing import Any, Optional
from app.src.providers.base_provider import BaseProvider


class CerebrasProvider(BaseProvider):
    """Provider for Cerebras cloud inference."""

    @property
    def name(self) -> str:
        return "cerebras"

    def create_llm(
        self,
        model_name: str,
        temperature: float,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Create Cerebras LLM instance.

        Args:
            model_name: Cerebras model name
            temperature: Temperature for generation
            api_key: Cerebras API key
            **kwargs: Additional configuration options

        Returns:
            ChatCerebras instance
        """
        try:
            from langchain_cerebras import ChatCerebras
        except ImportError:
            raise ImportError(
                "langchain-cerebras package is required for Cerebras provider. "
                "Install it with: pip install langchain-cerebras"
            )

        if not api_key:
            raise ValueError("CEREBRAS_API_KEY environment variable is required")

        return ChatCerebras(
            model=model_name,
            temperature=temperature,
            timeout=None,
            max_retries=5,
            api_key=api_key,
            **kwargs
        )

    def get_required_env_vars(self) -> list[str]:
        return ["CEREBRAS_API_KEY"]

    def validate_config(self, config: dict) -> tuple[bool, Optional[str]]:
        """Validate Cerebras configuration."""
        model_name = config.get("model_name")
        if not model_name:
            return False, "model_name is required for Cerebras provider"

        api_key = config.get("api_key")
        if not api_key:
            return False, "CEREBRAS_API_KEY environment variable must be set"

        return True, None

    def get_display_name(self) -> str:
        return "Cerebras"

    def get_default_model(self) -> Optional[str]:
        return "llama3.1-70b"

    def get_available_models(self) -> list[str]:
        return [
            "llama3.1-70b",
            "llama3.1-8b",
        ]
