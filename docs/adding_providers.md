# Adding Custom Providers to Ally

Ally features an extensible provider system that allows you to easily add support for new LLM providers without modifying core code. This guide explains how to add custom providers to your Ally installation.

## Table of Contents

- [Overview](#overview)
- [Quick Start: Using MiniMax M2](#quick-start-using-minimax-m2)
- [Creating a Custom Provider](#creating-a-custom-provider)
- [Provider Architecture](#provider-architecture)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

Ally's provider system has three tiers of extensibility:

### Tier 1: Using Built-in Providers (Easiest)
Built-in providers are ready to use out of the box:
- **Ollama** - Local LLM runtime (no API key required)
- **OpenAI** - GPT models
- **Anthropic** - Claude models
- **Google** - Gemini models
- **Cerebras** - Cloud inference

Simply configure in `config.json` and add API keys to `.env`.

### Tier 2: Using Custom Provider Plugins (Recommended)
Add new providers by creating a Python module in `app/src/providers/custom/`. The provider is automatically discovered and registered.

### Tier 3: Config-Based Custom Endpoints (Advanced)
For OpenAI/Anthropic-compatible APIs, you can configure custom base URLs directly in `config.json` without writing code (future feature).

## Quick Start: Using MiniMax M2

MiniMax M2 is included as an example custom provider. Here's how to use it:

### Step 1: Get MiniMax Credentials

1. Sign up at [https://platform.minimax.io/](https://platform.minimax.io/)
2. Navigate to Account → Your Profile to get your **Group ID** (19-digit number)
3. Go to API Keys → Create New Secret Key to generate your **API Key**

### Step 2: Add Credentials to .env

Edit your `.env` file and add:

```bash
MINIMAX_API_KEY=your_api_key_here
MINIMAX_GROUP_ID=your_19_digit_group_id
```

### Step 3: Configure in config.json

Update your `config.json` to use MiniMax:

```json
{
    "provider": "minimax",
    "model": "minimax-m2",
    "models": {
        "general": "minimax-m2",
        "code_gen": "minimax-m2",
        "brainstormer": "minimax-m2",
        "web_searcher": "minimax-m2"
    },
    "temperatures": {
        "general": 0.7,
        "code_gen": 0.1,
        "brainstormer": 0.9,
        "web_searcher": 0.5
    }
}
```

### Step 4: Install Dependencies

MiniMax uses `langchain-community`:

```bash
pip install langchain-community
```

### Step 5: Run Ally

```bash
ally
```

That's it! Ally will automatically detect and use the MiniMax provider.

## Creating a Custom Provider

Let's walk through creating a custom provider for a hypothetical "MyLLM" service.

### Step 1: Create Provider Module

Create a new file: `app/src/providers/custom/myllm.py`

```python
"""MyLLM provider implementation."""

from typing import Any, Optional
from app.src.providers.base_provider import BaseProvider


class MyLLMProvider(BaseProvider):
    """Provider for MyLLM service."""

    @property
    def name(self) -> str:
        """Return the unique identifier for this provider."""
        return "myllm"  # This is the name you'll use in config.json

    def create_llm(
        self,
        model_name: str,
        temperature: float,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Create and return a LangChain-compatible LLM instance.

        This is where you instantiate your LLM client using the appropriate
        LangChain integration package.
        """
        try:
            from langchain_community.llms import MyLLM  # Example
        except ImportError:
            raise ImportError(
                "langchain-community package is required. "
                "Install it with: pip install langchain-community"
            )

        if not api_key:
            raise ValueError("MYLLM_API_KEY environment variable is required")

        return MyLLM(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            **kwargs
        )

    def get_required_env_vars(self) -> list[str]:
        """Return list of required environment variables."""
        return ["MYLLM_API_KEY"]

    def validate_config(self, config: dict) -> tuple[bool, Optional[str]]:
        """Validate provider-specific configuration.

        Returns:
            (is_valid: bool, error_message: Optional[str])
        """
        model_name = config.get("model_name")
        if not model_name:
            return False, "model_name is required for MyLLM provider"

        api_key = config.get("api_key")
        if not api_key:
            return False, "MYLLM_API_KEY environment variable must be set"

        return True, None

    def get_display_name(self) -> str:
        """Return human-readable display name."""
        return "MyLLM"

    def get_default_model(self) -> Optional[str]:
        """Return the default model name."""
        return "myllm-base"

    def get_available_models(self) -> list[str]:
        """Return list of available models."""
        return [
            "myllm-base",
            "myllm-large",
            "myllm-code",
        ]
```

### Step 2: Add Environment Variables

Add to your `.env` file:

```bash
MYLLM_API_KEY=your_api_key_here
```

Update `.env.example` for others:

```bash
# MyLLM (Optional)

MYLLM_API_KEY=...
```

### Step 3: Update main.py API Keys

Edit `main.py` and add your provider to the `api_keys` dictionary:

```python
api_keys = {
    "cerebras": os.getenv("CEREBRAS_API_KEY"),
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "google": os.getenv("GOOGLE_GEN_AI_API_KEY"),
    "minimax": os.getenv("MINIMAX_API_KEY"),
    "myllm": os.getenv("MYLLM_API_KEY")  # Add this line
}
```

### Step 4: Configure and Use

Update `config.json`:

```json
{
    "provider": "myllm",
    "model": "myllm-code"
}
```

Run Ally:

```bash
ally
```

## Provider Architecture

### BaseProvider Interface

All providers must inherit from `BaseProvider` and implement these abstract methods:

#### Required Methods

##### `name` (property)
```python
@property
@abstractmethod
def name(self) -> str:
    """Unique identifier for this provider (e.g., 'openai', 'minimax')."""
    pass
```

##### `create_llm`
```python
@abstractmethod
def create_llm(
    self,
    model_name: str,
    temperature: float,
    api_key: Optional[str] = None,
    **kwargs
) -> Any:
    """Create and return a LangChain-compatible LLM instance."""
    pass
```

##### `get_required_env_vars`
```python
@abstractmethod
def get_required_env_vars(self) -> list[str]:
    """Return list of required environment variable names."""
    pass
```

##### `validate_config`
```python
@abstractmethod
def validate_config(self, config: dict) -> tuple[bool, Optional[str]]:
    """Validate provider-specific configuration.

    Returns:
        tuple: (is_valid: bool, error_message: Optional[str])
    """
    pass
```

#### Optional Methods (with defaults)

##### `get_display_name`
```python
def get_display_name(self) -> str:
    """Human-readable display name (default: capitalized provider name)."""
    return self.name.capitalize()
```

##### `supports_streaming`
```python
def supports_streaming(self) -> bool:
    """Whether provider supports streaming responses (default: True)."""
    return True
```

##### `get_default_model`
```python
def get_default_model(self) -> Optional[str]:
    """Default model identifier (default: None)."""
    return None
```

##### `get_available_models`
```python
def get_available_models(self) -> list[str]:
    """List of available models (default: empty list)."""
    return []
```

### Auto-Discovery System

The provider registry automatically:

1. **Discovers** all modules in `app/src/providers/custom/`
2. **Imports** each Python file
3. **Finds** classes that inherit from `BaseProvider`
4. **Registers** each provider by its `name` property
5. **Warns** if a custom provider overrides a built-in provider

No manual registration required - just drop the file and it works!

## Examples

### Example 1: Together AI (OpenAI-Compatible)

```python
"""Together AI provider using OpenAI-compatible endpoint."""

from typing import Any, Optional
from app.src.providers.base_provider import BaseProvider


class TogetherProvider(BaseProvider):
    """Provider for Together AI using OpenAI compatibility."""

    @property
    def name(self) -> str:
        return "together"

    def create_llm(
        self,
        model_name: str,
        temperature: float,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("langchain-openai required for Together AI")

        if not api_key:
            raise ValueError("TOGETHER_API_KEY required")

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url="https://api.together.xyz/v1",  # Custom base URL
            **kwargs
        )

    def get_required_env_vars(self) -> list[str]:
        return ["TOGETHER_API_KEY"]

    def validate_config(self, config: dict) -> tuple[bool, Optional[str]]:
        if not config.get("model_name"):
            return False, "model_name required"
        if not config.get("api_key"):
            return False, "TOGETHER_API_KEY must be set"
        return True, None

    def get_display_name(self) -> str:
        return "Together AI"

    def get_available_models(self) -> list[str]:
        return [
            "meta-llama/Llama-3-70b-chat-hf",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "codellama/CodeLlama-34b-Instruct-hf",
        ]
```

### Example 2: Groq (High-Speed Inference)

```python
"""Groq provider for ultra-fast inference."""

from typing import Any, Optional
from app.src.providers.base_provider import BaseProvider


class GroqProvider(BaseProvider):
    """Provider for Groq's high-speed LLM inference."""

    @property
    def name(self) -> str:
        return "groq"

    def create_llm(
        self,
        model_name: str,
        temperature: float,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError(
                "langchain-groq required. Install: pip install langchain-groq"
            )

        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable required")

        return ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            **kwargs
        )

    def get_required_env_vars(self) -> list[str]:
        return ["GROQ_API_KEY"]

    def validate_config(self, config: dict) -> tuple[bool, Optional[str]]:
        if not config.get("model_name"):
            return False, "model_name required for Groq"
        if not config.get("api_key"):
            return False, "GROQ_API_KEY must be set"
        return True, None

    def get_display_name(self) -> str:
        return "Groq"

    def get_default_model(self) -> Optional[str]:
        return "llama-3.3-70b-versatile"

    def get_available_models(self) -> list[str]:
        return [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ]
```

### Example 3: Local Provider (No API Key)

```python
"""LocalAI provider for self-hosted inference."""

from typing import Any, Optional
from app.src.providers.base_provider import BaseProvider


class LocalAIProvider(BaseProvider):
    """Provider for LocalAI self-hosted server."""

    @property
    def name(self) -> str:
        return "localai"

    def create_llm(
        self,
        model_name: str,
        temperature: float,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("langchain-openai required")

        # LocalAI uses OpenAI-compatible API
        # Default to localhost:8080 unless provided in kwargs
        base_url = kwargs.pop("base_url", "http://localhost:8080/v1")

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            api_key="not-needed",  # LocalAI doesn't require API key
            **kwargs
        )

    def get_required_env_vars(self) -> list[str]:
        return []  # No API key required

    def validate_config(self, config: dict) -> tuple[bool, Optional[str]]:
        if not config.get("model_name"):
            return False, "model_name required"
        return True, None

    def get_display_name(self) -> str:
        return "LocalAI (Self-Hosted)"
```

## Troubleshooting

### Provider Not Found

**Error:** `Unknown provider: myprovider. Available providers: ...`

**Solutions:**
1. Check that your provider file is in `app/src/providers/custom/`
2. Ensure the file ends with `.py`
3. Verify your provider class inherits from `BaseProvider`
4. Check that the `name` property returns the correct string
5. Restart Ally (providers are discovered on startup)

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'langchain_xyz'`

**Solution:** Install the required LangChain integration package:
```bash
pip install langchain-xyz
```

Common packages:
- `langchain-openai` - For OpenAI and compatible APIs
- `langchain-anthropic` - For Anthropic Claude
- `langchain-google-genai` - For Google Gemini
- `langchain-community` - For many community integrations
- `langchain-groq` - For Groq
- `langchain-ollama` - For Ollama

### API Key Not Found

**Error:** `MYPROVIDER_API_KEY environment variable is required`

**Solutions:**
1. Add the API key to your `.env` file:
   ```bash
   MYPROVIDER_API_KEY=your_key_here
   ```
2. Add the provider to `main.py` api_keys dictionary:
   ```python
   api_keys = {
       ...,
       "myprovider": os.getenv("MYPROVIDER_API_KEY")
   }
   ```
3. Restart Ally to reload environment variables

### Provider Validation Warnings

**Warning:** `Provider validation warning: ...`

This is usually informational. Common causes:
- API key not set (for providers that require it)
- Missing additional credentials (like MiniMax Group ID)
- Invalid configuration values

For providers that don't require API keys (like Ollama, LocalAI), these warnings can be ignored if the provider works correctly.

### Custom Provider Overrides Built-in

**Warning:** `Custom provider 'ollama' from my_ollama.py overrides built-in provider`

This happens when a custom provider has the same `name` as a built-in provider. The custom provider will take precedence.

**Solutions:**
1. Rename your provider to something unique
2. If intentional (e.g., customizing a built-in), this is fine

### Provider Works But Gives Strange Responses

**Checklist:**
1. Verify the model name is correct for the provider
2. Check temperature settings (0.0-1.0 range)
3. Ensure your API key has access to the specified model
4. Test the provider directly with a simple prompt
5. Check provider's official documentation for supported features

## Best Practices

### 1. Error Handling

Always wrap LLM instantiation in try-except:

```python
def create_llm(self, model_name, temperature, api_key, **kwargs):
    try:
        from langchain_xyz import ChatXYZ
    except ImportError:
        raise ImportError("langchain-xyz required: pip install langchain-xyz")

    if not api_key:
        raise ValueError("XYZ_API_KEY environment variable required")

    return ChatXYZ(...)
```

### 2. Validation

Implement thorough validation:

```python
def validate_config(self, config):
    # Check required fields
    if not config.get("model_name"):
        return False, "model_name is required"

    # Check credentials
    if not config.get("api_key") and self.get_required_env_vars():
        return False, f"{self.get_required_env_vars()[0]} must be set"

    # Provider-specific validation
    if config.get("custom_field") and not validate_custom_field(config["custom_field"]):
        return False, "Invalid custom_field value"

    return True, None
```

### 3. Documentation

Add comprehensive docstrings:

```python
"""Provider Name provider implementation.

This provider supports [Provider Name]'s API for [purpose].

Requirements:
    - API_KEY: Get from https://provider.com/api-keys
    - ADDITIONAL_ID: 16-digit identifier from account settings

Supported Models:
    - model-a: Fast, general purpose
    - model-b: Large, high quality
    - model-code: Optimized for code generation

Example:
    Add to .env:
        PROVIDER_API_KEY=abc123

    Configure config.json:
        {
            "provider": "provider_name",
            "model": "model-a"
        }
"""
```

### 4. Model Lists

Provide accurate model lists:

```python
def get_available_models(self) -> list[str]:
    """Return list of available models.

    Keep this updated as the provider adds new models.
    """
    return [
        "model-v1",      # Stable, recommended
        "model-v2-beta", # Beta, may change
        "model-code",    # Code-specialized
    ]
```

## Advanced Topics

### Custom kwargs Handling

Some providers need special parameters:

```python
def create_llm(self, model_name, temperature, api_key, **kwargs):
    # Extract custom parameters
    custom_endpoint = kwargs.pop("endpoint", None)
    max_tokens = kwargs.pop("max_tokens", 4096)

    # Use with your LLM
    return ChatProvider(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        endpoint=custom_endpoint,
        max_tokens=max_tokens,
        **kwargs  # Pass remaining kwargs
    )
```

### Multi-Credential Providers

For providers requiring multiple credentials (like MiniMax):

```python
import os

def create_llm(self, model_name, temperature, api_key, **kwargs):
    # Get additional credential from environment
    group_id = kwargs.pop("group_id", None) or os.getenv("PROVIDER_GROUP_ID")

    if not group_id:
        raise ValueError("PROVIDER_GROUP_ID required")

    return ChatProvider(
        api_key=api_key,
        group_id=group_id,
        model=model_name,
        temperature=temperature,
        **kwargs
    )
```

### Provider-Specific Features

Handle unique provider capabilities:

```python
def supports_streaming(self) -> bool:
    """This provider doesn't support streaming."""
    return False

def supports_tools(self) -> bool:
    """Custom method for tool calling support."""
    return True

def get_max_tokens(self, model_name: str) -> int:
    """Get max tokens for a specific model."""
    token_limits = {
        "model-small": 4096,
        "model-large": 128000,
    }
    return token_limits.get(model_name, 4096)
```

## Contributing

If you create a provider for a popular service, consider contributing it back to Ally:

1. Test thoroughly with multiple models
2. Add comprehensive documentation
3. Include example configuration
4. Submit a pull request to move it from `custom/` to `builtin/`

Popular providers we'd love to see:
- Perplexity AI
- Replicate
- HuggingFace Inference API
- Cohere
- AI21 Labs
- Fireworks AI

## Additional Resources

- [LangChain Provider Documentation](https://python.langchain.com/docs/integrations/platforms/)
- [Ally GitHub Repository](https://github.com/YassWorks/Ally)
- [BaseProvider API Reference](../app/src/providers/base_provider.py)
- [Built-in Provider Examples](../app/src/providers/builtin/)

## Questions?

If you have questions or run into issues:
1. Check this documentation first
2. Review existing provider implementations in `app/src/providers/builtin/`
3. Open an issue on GitHub
4. Join the community Discord (link in README)

Happy coding with Ally! 🤖
