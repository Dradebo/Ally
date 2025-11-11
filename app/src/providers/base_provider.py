"""Base provider class for extensible LLM provider system.

This module defines the abstract interface that all provider plugins must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseProvider(ABC):
    """Abstract base class for LLM provider implementations.

    All provider plugins must inherit from this class and implement
    the required methods to integrate with Ally's agent system.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique identifier for this provider.

        Returns:
            str: Provider name (e.g., "ollama", "openai", "minimax")
        """
        pass

    @abstractmethod
    def create_llm(
        self,
        model_name: str,
        temperature: float,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Create and return a LangChain-compatible LLM instance.

        Args:
            model_name: The name/identifier of the model to use
            temperature: Temperature setting for model responses (0.0-1.0)
            api_key: API key for authentication (if required)
            **kwargs: Additional provider-specific configuration options

        Returns:
            A LangChain-compatible chat model instance

        Raises:
            ValueError: If configuration is invalid
            ImportError: If required dependencies are not installed
        """
        pass

    @abstractmethod
    def get_required_env_vars(self) -> list[str]:
        """Return list of required environment variable names.

        Returns:
            list[str]: Environment variables needed by this provider
                      (e.g., ["OPENAI_API_KEY"] or ["MINIMAX_API_KEY", "MINIMAX_GROUP_ID"])
        """
        pass

    @abstractmethod
    def validate_config(self, config: dict) -> tuple[bool, Optional[str]]:
        """Validate provider-specific configuration.

        Args:
            config: Configuration dictionary containing provider settings

        Returns:
            tuple: (is_valid: bool, error_message: Optional[str])
                   If valid, returns (True, None)
                   If invalid, returns (False, "error description")
        """
        pass

    def get_display_name(self) -> str:
        """Return human-readable display name for this provider.

        Returns:
            str: Display name (defaults to capitalized provider name)
        """
        return self.name.capitalize()

    def supports_streaming(self) -> bool:
        """Indicate whether this provider supports streaming responses.

        Returns:
            bool: True if streaming is supported, False otherwise
        """
        return True  # Most modern providers support streaming

    def get_default_model(self) -> Optional[str]:
        """Return the default model name for this provider.

        Returns:
            Optional[str]: Default model identifier, or None if no default
        """
        return None

    def get_available_models(self) -> list[str]:
        """Return list of available models for this provider.

        Returns:
            list[str]: List of model identifiers (empty if not applicable)
        """
        return []
