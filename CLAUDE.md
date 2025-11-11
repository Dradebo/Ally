# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Ally** is a lightweight, private AI service assistant built in Python. It's a CLI tool with agentic capabilities designed for tasks ranging from everyday operations to complex projects. The system supports multiple LLM providers (Ollama, OpenAI, Anthropic, Google GenAI, Cerebras) with a focus on privacy and local operation.

## Running the Application

### Installation and Setup
```bash
# Clone and setup
git clone https://github.com/YassWorks/Ally.git
cd Ally

# Run setup script (creates virtual environment, installs dependencies)
./setup.sh  # Linux/Mac
# or
setup.cmd   # Windows

# Configure config.json with your provider/model settings
# Configure .env with your API keys

# Run Ally
ally              # Start default chat
ally -h           # View help
ally -p "message" # Start with a specific prompt
ally -i <id>      # Continue existing session
ally -d <path>    # Set working directory
ally --create-project  # Start project generation workflow
```

### Development Commands
```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Run from source
python main.py

# The CLI is configured through config.json - no build step required
```

## Architecture

### Multi-Agent System

Ally uses a **multi-agent architecture** built on LangGraph. There are four specialized agent types:

1. **GeneralAgent** (`app/src/agents/general/`) - Default chat interface with file operations, web access, and command execution
2. **CodeGenAgent** (`app/src/agents/code_gen/`) - Specialized for code generation
3. **BrainstormerAgent** (`app/src/agents/brainstormer/`) - Creates project specifications and context
4. **WebSearcherAgent** (`app/src/agents/web_searcher/`) - Performs web searches

All agents inherit from `BaseAgent` (`app/src/core/base.py`) which provides:
- Chat session management with thread persistence
- Message handling and state management
- Custom command registration system
- RAG integration capabilities
- Permission-based tool execution

### Agent Creation Pattern

Agents are created using the **AgentFactory** pattern (`app/src/core/agent_factory.py`):

```python
agent = AgentFactory.create_agent(
    agent_type="general",  # or "code_gen", "brainstormer", "web_searcher"
    config={
        "model_name": "gpt-4o",
        "api_key": "...",
        "provider": "openai",
        "temperature": 0.7,
        "system_prompt": None  # Uses default if None
    }
)
```

The factory handles:
- Agent instantiation based on type
- Configuration validation
- System prompt loading from `app/src/agents/<type>/config/system_prompt.md`
- LangGraph state machine compilation

### LangGraph State Machines

Each agent type defines its own LangGraph state machine in `app/src/agents/<type>/config/config.py`. The state machine:
- Compiles into a `CompiledStateGraph`
- Manages message flow between user input, LLM, and tools
- Handles tool calling and response integration
- Maintains conversation state with checkpointing via SQLite

The compiled agent is accessed via:
```python
# In BaseAgent.start_chat()
for event in self.agent.stream(inputs, self.configuration, stream_mode="values"):
    # Process events
```

### Tool System

Tools are implemented as LangChain tools using the `@tool` decorator across multiple modules:
- `app/src/tools/file_tools.py` - File/directory operations
- `app/src/tools/exec_tools.py` - Command execution
- `app/src/tools/web_tools.py` - Web scraping and search
- `app/src/tools/git_tools.py` - Git operations
- `app/src/tools/find_tools.py` - File search and content search

**Critical Pattern**: All tools use the **permission manager** (`app/src/core/permissions.py`):

```python
@tool
def create_file(file_path: str, content: str) -> str:
    if not permission_manager.get_permission(tool_name="create_file", path=file_path):
        raise PermissionDeniedException()
    # ... perform operation
```

The permission system:
- Prompts user for approval before executing potentially dangerous operations
- Can be bypassed with `--allow-all-tools` flag (WARNING: security risk)
- Tracks approved/denied permissions during session

### Configuration System

Configuration is split between two files:

**config.json** - Model and agent settings:
```json
{
    "provider": "openai",              // Default provider
    "model": "gpt-4o",                 // Default model
    "provider_per_model": {            // Override provider per agent
        "general": "ollama",
        "code_gen": "anthropic"
    },
    "models": {                        // Override model per agent
        "general": "gpt-oss:20b",
        "code_gen": "claude-sonnet-3.5"
    },
    "temperatures": {...},
    "system_prompts": {...},           // null uses defaults
    "embedding_provider": "hf",        // For RAG
    "embedding_model": "...",
    "scraping_method": "simple"        // or "docling"
}
```

**NOTE**: `null` values in `provider_per_model`, `models`, and `system_prompts` auto-fill from the default `provider`, `model`, or built-in prompts.

**.env** - API credentials:
```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_GEN_AI_API_KEY=...
CEREBRAS_API_KEY=...
NLP_CLOUD_API_KEY=...
GOOGLE_SEARCH_API_KEY=...
SEARCH_ENGINE_ID=...
```

Configuration loading happens in `main.py` which:
1. Loads environment variables
2. Reads and validates `config.json`
3. Auto-fills null values with defaults
4. Creates API key mappings per agent
5. Instantiates the CLI with complete configuration

### RAG System Architecture

The RAG system is modular with three key components:

**1. Embedding Functions** (`app/src/embeddings/embedding_functions/`):
- `HFEmbedder` - Local Hugging Face models
- `OllamaEmbedder` - Ollama embedding models
- `OpenAIEmbedder` - OpenAI embeddings
- `NLPCloudEmbedder` - NLP Cloud hosted embeddings

**2. Scrapers** (`app/src/embeddings/scrapers/`):
- `SimpleScraper` - Basic text extraction (default, lightweight)
- `DoclingScraper` - Advanced PDF/DOCX parsing with OCR (heavy, requires additional setup)

Both implement `AbstractScraper` interface with `scrape(path) -> list[Document]`

**3. Database Client** (`app/src/embeddings/db_client.py`):
- Wraps ChromaDB for vector storage
- Manages collections and indexing
- Handles document chunking and retrieval

**RAG Workflow**:
```python
# In CLI initialization:
cli = CLI(
    embedding_provider="hf",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    scraping_method="simple"
)

# User commands in chat:
/embed <path> <collection_name>  # Index documents
/start_rag                       # Enable RAG mode
# ... agent answers from indexed docs only
/stop_rag                        # Disable RAG mode

/index <collection>              # Add collection to retrieval
/unindex <collection>            # Remove from retrieval
/list                            # View all collections
/delete <collection>             # Delete collection
/purge                           # Delete all collections
```

When RAG is active:
1. User query is embedded
2. Top-k similar documents retrieved from indexed collections
3. Retrieved context prepended to prompt
4. Agent constrained to answer only from provided documents

### Custom Commands

Agents support custom slash commands registered via:
```python
# In BaseAgent subclass
self._custom_commands["/custom"] = self._handle_custom
```

Built-in commands (handled in `BaseAgent.start_chat()`):
- `/model` - Change model mid-session
- `/id [new_id]` - View/change thread ID
- `/continue` - Continue from last checkpoint
- `/start_rag`, `/stop_rag` - Toggle RAG
- `/embed`, `/index`, `/unindex`, `/list`, `/delete`, `/purge` - RAG management
- `/project` - Launch CodeGen workflow

### Orchestration Units

Complex workflows are handled by "Units" (`app/src/orchestration/units/`):

**CodeGenUnit** (`orchestrated_codegen.py`):
- Orchestrates Brainstormer � CodeGen � WebSearcher pipeline
- Manages project specification generation
- Handles interactive refinement loops
- Creates project directories and files

Units inherit from `BaseUnit` which provides:
- Multi-agent management
- Exception handling with `AgentExceptionHandler`
- Session state coordination

### Exception Handling

`AgentExceptionHandler` (`app/src/core/exception_handler.py`) provides centralized error handling for:
- LangGraph recursion limits
- Permission denials
- RAG setup failures
- API errors (rate limits, invalid keys)
- Tool execution failures

Errors are displayed via the `AgentUI` system (`app/src/core/ui.py`) with styled Rich console output.

## Important Files and Modules

### Entry Points
- `main.py` - Application entry point, configuration loading
- `app/__init__.py` - Exports CLI and default_ui
- `app/src/cli/cli.py` - CLI class with chat orchestration

### Core Infrastructure
- `app/src/core/base.py` - BaseAgent with chat session management
- `app/src/core/agent_factory.py` - Agent creation factory
- `app/src/core/permissions.py` - Tool permission system
- `app/src/core/ui.py` - Terminal UI with Rich formatting
- `app/src/core/exception_handler.py` - Centralized error handling
- `app/src/core/create_base_agent.py` - LLM instantiation per provider

### Agent Implementations
- `app/src/agents/general/general.py` - GeneralAgent
- `app/src/agents/code_gen/code_gen.py` - CodeGenAgent
- `app/src/agents/brainstormer/brainstormer.py` - BrainstormerAgent
- `app/src/agents/web_searcher/web_searcher.py` - WebSearcherAgent
- `app/src/agents/<type>/config/config.py` - LangGraph state machine definitions
- `app/src/agents/<type>/config/system_prompt.md` - Default system prompts

### Tools
- `app/src/tools/file_tools.py` - File/directory CRUD operations
- `app/src/tools/exec_tools.py` - Shell command execution (with timeout)
- `app/src/tools/web_tools.py` - Web scraping, Google search
- `app/src/tools/git_tools.py` - Git operations
- `app/src/tools/find_tools.py` - File/content search

### RAG System
- `app/src/embeddings/db_client.py` - ChromaDB wrapper
- `app/src/embeddings/handle_commands.py` - RAG command handlers
- `app/src/embeddings/embedding_functions/` - Embedding implementations
- `app/src/embeddings/scrapers/` - Document parsing implementations

### Orchestration
- `app/src/orchestration/units/orchestrated_codegen.py` - CodeGenUnit workflow
- `app/src/orchestration/integrate_web_search.py` - Web search integration

### Configuration & Constants
- `config.json` - Model/agent configuration (NOT committed, user-specific)
- `.env` - API keys (NOT committed, user-specific)
- `app/utils/constants.py` - Application constants (timeouts, limits, themes)
- `app/utils/ui_messages.py` - User-facing message templates
- `app/src/cli/flags.py` - CLI argument parsing

## Key Design Patterns

### Adding a New Agent Type

1. Create directory: `app/src/agents/<new_type>/`
2. Add config: `app/src/agents/<new_type>/config/config.py` with `get_agent()` function
3. Add prompt: `app/src/agents/<new_type>/config/system_prompt.md`
4. Implement agent: `app/src/agents/<new_type>/<new_type>.py` inheriting from `BaseAgent`
5. Register in `AgentFactory.create_agent()` in `app/src/core/agent_factory.py`
6. Update `AGENT_TYPES` in `main.py`

### Adding a New Tool

1. Add to appropriate module in `app/src/tools/`
2. Use `@tool` decorator with comprehensive docstring
3. Add permission check: `permission_manager.get_permission(tool_name="...", ...)`
4. Raise `PermissionDeniedException()` if denied
5. Import and add to tools list in agent's `config/config.py`

### Adding Custom Slash Commands

In your agent subclass `__init__`:
```python
self._custom_commands["/mycommand"] = self._handle_mycommand

def _handle_mycommand(self, user_input: str, ...) -> bool:
    # Return True to exit chat, False to continue
    ...
```

### Environment Variables for Storage

Ally respects these environment variables for data storage:
- `ALLY_HISTORY_DIR` - Chat history location
- `ALLY_DATABASE_DIR` - ChromaDB storage location
- `ALLY_EMBEDDING_MODELS_DIR` - HuggingFace models cache
- `ALLY_PARSING_MODELS_DIR` - Docling models cache

Defaults:
- Windows: `%LOCALAPPDATA%\Ally\...`
- Linux/Mac: `~/.local/share/Ally/...`

## Important Notes

- **RAG models are downloaded on-demand** the first time RAG features are used after configuring embedding settings
- **Docling scraper is CPU-only by default** but can be configured for GPU
- **Permission system can be bypassed** with `--allow-all-tools` flag (use with caution)
- **Thread IDs persist sessions** - use `/id` to view current ID and `-i <id>` to continue
- **System prompts default to built-in** when `null` in config.json
- **The agent's state graph is compiled once** during agent creation, not per-message

