"""Ollama provider implementation for local LLM inference."""

from typing import Any, Optional
from app.src.providers.base_provider import BaseProvider


class OllamaProvider(BaseProvider):
    """Provider for Ollama local LLM runtime."""

    @property
    def name(self) -> str:
        return "ollama"

    def create_llm(
        self,
        model_name: str,
        temperature: float,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Create Ollama LLM instance.

        Args:
            model_name: Ollama model name (e.g., "qwen2.5-coder:7b")
            temperature: Temperature for generation
            api_key: Not used (Ollama is local)
            **kwargs: Additional configuration options

        Returns:
            ChatOllama instance
        """
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-ollama package is required for Ollama provider. "
                "Install it with: pip install langchain-ollama"
            )

        return ChatOllama(
            model=model_name,
            temperature=temperature,
            validate_model_on_init=True,
            reasoning=False,
            **kwargs
        )

    def get_required_env_vars(self) -> list[str]:
        """Ollama doesn't require any API keys (runs locally)."""
        return []

    def validate_config(self, config: dict) -> tuple[bool, Optional[str]]:
        """Validate Ollama configuration.

        Args:
            config: Configuration dictionary

        Returns:
            (True, None) if valid, (False, error_message) if invalid
        """
        model_name = config.get("model_name")
        if not model_name:
            return False, "model_name is required for Ollama provider"

        return True, None

    def get_display_name(self) -> str:
        return "Ollama (Local)"

    def get_default_model(self) -> Optional[str]:
        return "qwen2.5-coder:7b"

    def get_available_models(self) -> list[str]:
        """Return commonly used Ollama models."""
        return [
            "qwen2.5-coder:7b",
            "qwen2.5:latest",
            "llama3.2:latest",
            "codellama:latest",
            "mistral:latest",
        ]
