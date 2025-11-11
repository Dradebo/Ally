"""Custom provider implementations.

This directory is for user-added provider plugins.
Simply drop a new provider module here and it will be automatically discovered.

Example providers included:
    - minimax.py: MiniMax M2 model support

To add a new provider:
1. Create a new .py file in this directory
2. Implement a class that inherits from BaseProvider
3. Implement all required abstract methods
4. The provider will be automatically discovered and registered

See minimax.py for a complete example implementation.
"""

# This file intentionally left minimal to allow auto-discovery
# Custom providers are loaded dynamically by the ProviderRegistry
