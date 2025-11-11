"""MiniMax M2 provider implementation.

MiniMax is a Chinese AI company providing high-quality language models.
This provider supports the MiniMax-M2 model, optimized for coding and agentic workflows.

Requirements:
    - MINIMAX_API_KEY: API key from MiniMax platform
    - MINIMAX_GROUP_ID: 19-digit group identifier from your account

Get credentials at: https://platform.minimax.io/
"""

import os
from typing import Any, Optional
from app.src.providers.base_provider import BaseProvider


class MinimaxProvider(BaseProvider):
    """Provider for MiniMax's M2 model via LangChain community integration."""

    @property
    def name(self) -> str:
        return "minimax"

    def create_llm(
        self,
        model_name: str,
        temperature: float,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Create MiniMax LLM instance.

        Args:
            model_name: MiniMax model name (e.g., "minimax-m2")
            temperature: Temperature for generation
            api_key: MiniMax API key (from MINIMAX_API_KEY env var)
            **kwargs: Additional configuration options. Can include:
                     - group_id: MiniMax Group ID (19-digit number)

        Returns:
            Minimax LLM instance from langchain_community

        Raises:
            ValueError: If required credentials are missing
            ImportError: If langchain-community is not installed
        """
        try:
            from langchain_community.llms import Minimax
        except ImportError:
            raise ImportError(
                "langchain-community package is required for MiniMax provider. "
                "Install it with: pip install langchain-community"
            )

        if not api_key:
            raise ValueError(
                "MINIMAX_API_KEY environment variable is required. "
                "Get your API key from https://platform.minimax.io/"
            )

        # Get Group ID from kwargs or environment
        group_id = kwargs.pop("group_id", None) or os.getenv("MINIMAX_GROUP_ID")
        if not group_id:
            raise ValueError(
                "MINIMAX_GROUP_ID environment variable is required. "
                "This is your 19-digit account identifier from https://platform.minimax.io/"
            )

        return Minimax(
            minimax_api_key=api_key,
            minimax_group_id=group_id,
            temperature=temperature,
            **kwargs
        )

    def get_required_env_vars(self) -> list[str]:
        """MiniMax requires API key and Group ID."""
        return ["MINIMAX_API_KEY", "MINIMAX_GROUP_ID"]

    def validate_config(self, config: dict) -> tuple[bool, Optional[str]]:
        """Validate MiniMax configuration.

        Args:
            config: Configuration dictionary

        Returns:
            (True, None) if valid, (False, error_message) if invalid
        """
        model_name = config.get("model_name")
        if not model_name:
            return False, "model_name is required for MiniMax provider"

        api_key = config.get("api_key")
        if not api_key:
            return (
                False,
                "MINIMAX_API_KEY environment variable must be set. "
                "Get it from https://platform.minimax.io/",
            )

        # Check for Group ID in config or environment
        group_id = config.get("group_id") or os.getenv("MINIMAX_GROUP_ID")
        if not group_id:
            return (
                False,
                "MINIMAX_GROUP_ID environment variable must be set. "
                "This is your 19-digit account identifier.",
            )

        # Validate Group ID format (should be 19 digits)
        if not group_id.isdigit() or len(group_id) != 19:
            return (
                False,
                f"MINIMAX_GROUP_ID should be a 19-digit number, got: {group_id}",
            )

        return True, None

    def get_display_name(self) -> str:
        return "MiniMax M2"

    def get_default_model(self) -> Optional[str]:
        return "minimax-m2"

    def get_available_models(self) -> list[str]:
        """Return available MiniMax models."""
        return [
            "minimax-m2",  # Primary coding-optimized model
        ]

    def supports_streaming(self) -> bool:
        """MiniMax supports streaming responses."""
        return True
