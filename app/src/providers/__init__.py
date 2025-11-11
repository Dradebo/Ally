"""Provider registry and auto-discovery system for extensible LLM providers.

This module manages the registration and discovery of both built-in and custom
provider implementations, enabling easy addition of new LLM providers without
modifying core code.
"""

import importlib
import pkgutil
from typing import Dict, Optional, Type
from pathlib import Path

from .base_provider import BaseProvider
from .builtin import (
    OllamaProvider,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    CerebrasProvider,
)


class ProviderRegistry:
    """Registry for managing LLM provider implementations.

    Automatically discovers and registers both built-in and custom providers,
    providing a centralized access point for provider instantiation.
    """

    def __init__(self):
        """Initialize the provider registry and load all providers."""
        self._providers: Dict[str, Type[BaseProvider]] = {}
        self._load_builtin_providers()
        self._discover_custom_providers()

    def _load_builtin_providers(self):
        """Load built-in provider implementations."""
        builtin_providers = [
            OllamaProvider,
            OpenAIProvider,
            AnthropicProvider,
            GoogleProvider,
            CerebrasProvider,
        ]

        for provider_class in builtin_providers:
            provider_instance = provider_class()
            self._providers[provider_instance.name] = provider_class

    def _discover_custom_providers(self):
        """Automatically discover and load custom provider plugins.

        Scans the providers/custom/ directory for Python modules that implement
        BaseProvider and automatically registers them.
        """
        custom_dir = Path(__file__).parent / "custom"
        if not custom_dir.exists():
            return

        # Add custom directory to Python path for imports
        import sys
        if str(custom_dir.parent) not in sys.path:
            sys.path.insert(0, str(custom_dir.parent))

        # Discover all Python modules in custom directory
        for _, module_name, _ in pkgutil.iter_modules([str(custom_dir)]):
            try:
                # Import the module
                module = importlib.import_module(f"custom.{module_name}")

                # Find all BaseProvider subclasses in the module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)

                    # Check if it's a class that inherits from BaseProvider
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseProvider)
                        and attr is not BaseProvider
                    ):
                        # Instantiate and register the provider
                        provider_instance = attr()
                        provider_name = provider_instance.name

                        # Warn if overriding a built-in provider
                        if provider_name in self._providers:
                            print(
                                f"Warning: Custom provider '{provider_name}' "
                                f"from {module_name}.py overrides built-in provider"
                            )

                        self._providers[provider_name] = attr

            except Exception as e:
                print(
                    f"Warning: Failed to load custom provider from {module_name}.py: {e}"
                )

    def get_provider(self, provider_name: str) -> Optional[Type[BaseProvider]]:
        """Get a provider class by name.

        Args:
            provider_name: Name of the provider (e.g., "ollama", "openai")

        Returns:
            Provider class if found, None otherwise
        """
        return self._providers.get(provider_name.lower())

    def list_providers(self) -> list[str]:
        """Get list of all registered provider names.

        Returns:
            List of provider names
        """
        return sorted(self._providers.keys())

    def is_provider_available(self, provider_name: str) -> bool:
        """Check if a provider is registered.

        Args:
            provider_name: Name of the provider

        Returns:
            True if provider is registered, False otherwise
        """
        return provider_name.lower() in self._providers

    def get_provider_info(self, provider_name: str) -> Optional[dict]:
        """Get detailed information about a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Dictionary with provider information, or None if not found
        """
        provider_class = self.get_provider(provider_name)
        if not provider_class:
            return None

        provider_instance = provider_class()
        return {
            "name": provider_instance.name,
            "display_name": provider_instance.get_display_name(),
            "required_env_vars": provider_instance.get_required_env_vars(),
            "default_model": provider_instance.get_default_model(),
            "available_models": provider_instance.get_available_models(),
            "supports_streaming": provider_instance.supports_streaming(),
        }


# Global provider registry instance
_provider_registry = None


def get_provider_registry() -> ProviderRegistry:
    """Get the global provider registry instance.

    Returns:
        ProviderRegistry: The singleton provider registry
    """
    global _provider_registry
    if _provider_registry is None:
        _provider_registry = ProviderRegistry()
    return _provider_registry


__all__ = [
    "BaseProvider",
    "ProviderRegistry",
    "get_provider_registry",
]
