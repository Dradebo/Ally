"""Google GenAI provider implementation."""

import os
from typing import Any, Optional
from app.src.providers.base_provider import BaseProvider


class GoogleProvider(BaseProvider):
    """Provider for Google's Gemini models."""

    @property
    def name(self) -> str:
        return "google"

    def create_llm(
        self,
        model_name: str,
        temperature: float,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Create Google GenAI LLM instance.

        Args:
            model_name: Gemini model name (e.g., "gemini-2.0-flash-exp")
            temperature: Temperature for generation
            api_key: Google API key
            **kwargs: Additional configuration options

        Returns:
            ChatGoogleGenerativeAI instance
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai package is required for Google provider. "
                "Install it with: pip install langchain-google-genai"
            )

        if not api_key:
            raise ValueError("GOOGLE_GEN_AI_API_KEY environment variable is required")

        # Suppress gRPC verbosity
        os.environ["GRPC_VERBOSITY"] = "NONE"
        os.environ["GRPC_CPP_VERBOSITY"] = "NONE"

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            timeout=None,
            max_retries=5,
            google_api_key=api_key,
            **kwargs
        )

    def get_required_env_vars(self) -> list[str]:
        return ["GOOGLE_GEN_AI_API_KEY"]

    def validate_config(self, config: dict) -> tuple[bool, Optional[str]]:
        """Validate Google configuration."""
        model_name = config.get("model_name")
        if not model_name:
            return False, "model_name is required for Google provider"

        api_key = config.get("api_key")
        if not api_key:
            return False, "GOOGLE_GEN_AI_API_KEY environment variable must be set"

        return True, None

    def get_display_name(self) -> str:
        return "Google Gemini"

    def get_default_model(self) -> Optional[str]:
        return "gemini-2.0-flash-exp"

    def get_available_models(self) -> list[str]:
        return [
            "gemini-2.0-flash-exp",
            "gemini-2.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]
