# ai-llm-agents-secretary-v2

A local LLM agent stack built around four core components:

- **ContextManager** — manages the model context window
- **LLMServer** — starts and maintains a local `llama.cpp` server
- **ToolManager** — defines, indexes, and executes tools for the LLM
- **StatusManager** — tracks the current LLM phase

The system is designed to run a local model through an OpenAI-compatible API, expose tools dynamically, and manage limited context space by deciding what stays in active context and what gets cached.

---

## Architecture

### ContextManager

A server to manage the model context window.

It is responsible for storing:

- **system prompt**
- **tools** with added metadata such as:
  - `reason_added`
  - `id_added`
- **relevant data** with added metadata such as:
  - `reason_added`
  - `id_added`
- **message history**
  - each message gets an `id_added`
- **cache index**

The intended behavior is a **dynamic garbage collector / caching mechanism** that moves less relevant data out of the active context while reserving an indexed entry in context.

#### Caching model

Instead of keeping everything in the main prompt window, older or lower-priority items can be moved into cache.

Caching is intended for items such as:

- tools
- older messages
- older context data

Each cache entry has a **hit rate**. Based on that hit rate, the manager dynamically:

- reloads useful data into the active context
- unloads stale or less useful data
- keeps a lightweight indexed reference in context

This allows the system to preserve context-window space without fully losing traceability of prior tools, data, or messages.

#### Current implementation in this repo

The provided implementation in `src/ContextManager/ContextManager.py` currently supports:

- configurable context size
- master prompt storage
- context data storage
- chat history storage
- optional compression flag
- persistent session logging
- session state dumps
- simple usage statistics and bars

State is persisted under:

```text
contexts/<timestamp>/
```

with:

- `context.json`
- `context.log`

A FastAPI service is exposed through:

```text
src/ContextManager/run_context_manager.py
```

#### ContextManager API

The current API includes endpoints for:

- reading full state
- setting and getting context size
- setting and getting the master prompt
- adding and listing context data
- adding and listing chat messages
- toggling compression
- viewing assembled context
- viewing CLI and HTML usage bars

Run it with:

```bash
python src/ContextManager/run_context_manager.py
```

Default URLs:

- `http://localhost:7999/`
- `http://localhost:7999/docs`
- `http://localhost:7999/api/context/print`
- `http://localhost:7999/api/context/gui_bar`

---

### LLMServer

Starts and maintains the local LLM server using `llama.cpp`.

This component:

- resolves a model path
- downloads approved Gemma 4 GGUF models from Hugging Face when needed
- starts `llama-server`
- can run in the foreground or via a helper in the background
- tracks PID and logs for lifecycle management

#### Supported model sources

The allowed Hugging Face repos are:

- `ggml-org/gemma-4-E2B-it-GGUF`
- `ggml-org/gemma-4-E4B-it-GGUF`
- `ggml-org/gemma-4-26B-A4B-it-GGUF`
- `ggml-org/gemma-4-31B-it-GGUF`

Supported aliases include:

- `e2b`
- `e4b`
- `26b`
- `31b`
- `5b`
- `8b`
- `25b`
- `31b_params`
- `1.7b`

#### Main files

- `src/LLMServer/LLMServer.py`
- `src/LLMServer/start_llm_server.py`
- `src/LLMServer/stop_llm_server.py`

#### Usage

Start the server:

```bash
python src/LLMServer/start_llm_server.py --model e4b
```

Start with extra llama.cpp parameters:

```bash
python src/LLMServer/start_llm_server.py --model e4b --parameters "--ctx-size 8192 --n-gpu-layers 99"
```

Stop the server:

```bash
python src/LLMServer/stop_llm_server.py
```

Force stop:

```bash
python src/LLMServer/stop_llm_server.py --force
```

---

### ToolManager

Defines and stores all tools available to the LLM.

The tool layer is built around a recursive loader that scans the toolbox and registers every valid tool module.

A tool module must expose:

- `tool_schema`
- `execute(...)`

Tool names are inferred from their relative file paths.

Examples:

- `toolbox/math/add.py` → `math_add`
- `toolbox/data/db/get/table_schema.py` → `data_db_get_table_schema`

#### Tool loading behavior

The ToolManager:

- recursively scans the toolbox directory
- imports Python files dynamically
- injects the inferred function name into each tool schema
- stores the schema for LLM exposure
- stores the execute function for runtime dispatch

#### Intended tool-index behavior

The intended architecture is:

- the **tool index** is exposed in context
- full **tool descriptions** are loaded only on demand
- unused tools are stored and later cached
- only the most relevant tools stay fully expanded in active context

This reduces context usage while still allowing the LLM to discover available capabilities.

#### Current implementation in this repo

The current `ToolManager` implementation supports:

- dynamic recursive tool registration
- schema collection
- dynamic tool execution by name
- JSON argument parsing
- basic execution error handling

The current code does **not** include a visible implementation of:

- `select_best_tool` API
- lazy-loading tool descriptions into context
- tool hit-rate based cache eviction

Those behaviors fit the intended architecture, but they are not present in the pasted source as-is.

#### Available tools in this repo

**Math**
- `math_add`
- `math_power`

**Shell**
- `sh_run_script`
  - allows only:
    - `ls`
    - `cd`
    - `pwd`

**Database**
- `data_db_get_list_of_tables`
- `data_db_get_table_schema`
- `data_db_select`
- `data_db_select_educations`

---

### StatusManager

Dynamic status of the current LLM phase.

This component is intended to track where the agent is in its execution cycle, for example:

- preparing context
- selecting tools
- waiting for model output
- executing tools
- updating memory or cache
- returning final response

This is useful for:

- observability
- UI integration
- debugging agent flow
- exposing current internal phase to the operator

#### Current implementation note

A `StatusManager` source file was **not included** in the code you pasted. So this section reflects the intended architecture, not a confirmed implementation in the current repository snapshot.

---

## Main agent loop

The runtime entrypoint is:

```text
scripts/main.py
```

It:

1. initializes `ToolManager`
2. connects to a local OpenAI-compatible endpoint
3. sends conversation history and tool schemas to the model
4. streams assistant output
5. reconstructs fragmented tool calls from streaming chunks
6. executes requested tools
7. appends tool results back into message history
8. repeats until the model stops calling tools

The client is configured for:

- base URL: `http://localhost:8080/v1`
- API key: `sk-local`
- model: `local-model`

---

## Project structure

```text
.
├── scripts/
│   └── main.py
└── src/
    ├── ContextManager/
    │   ├── ContextManager.py
    │   └── run_context_manager.py
    ├── LLMServer/
    │   ├── LLMServer.py
    │   ├── start_llm_server.py
    │   └── stop_llm_server.py
    └── ToolManager/
        ├── ToolManager.py
        └── toolbox/
            ├── math/
            │   ├── add.py
            │   └── power.py
            ├── sh/
            │   └── run_script.py
            └── data/
                └── db/
                    ├── select.py
                    ├── get/
                    │   ├── list_of_tables.py
                    │   └── table_schema.py
                    └── select/
                        └── educations.py
```

---

## Requirements

- Python 3.10+
- `llama-server` available in `PATH`
- local machine capable of running GGUF models
- Supabase credentials for DB-related tools

Python packages used in the provided code:

- `openai`
- `python-dotenv`
- `huggingface_hub`
- `psycopg2`
- `supabase`
- `fastapi`
- `uvicorn`

Example install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai python-dotenv huggingface_hub psycopg2-binary supabase fastapi uvicorn
```

---

## Environment variables

Create a `.env` file in the project root.

Example:

```env
HF_TOKEN=your_huggingface_token

MODEL=e4b
PORT=8080
PARAMETERS=
CACHE_DIR=./llm/cache
MODELS_DIR=./llm/models

SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_key

STOP_TIMEOUT=10
```

### Supabase note

The current code uses `SUPABASE_URL` inconsistently:

- some tools expect it to be the Supabase project URL
- `src/ToolManager/toolbox/data/db/select.py` uses it like a PostgreSQL connection string

This should be split into separate variables, such as:

- `SUPABASE_URL`
- `SUPABASE_DB_URL`

---

## Security notes

### Shell tool

The shell tool is restricted to:

- `ls`
- `cd`
- `pwd`

### SQL tool

The SQL tool attempts to enforce read-only execution by:

- allowing only queries that start with `SELECT` or `WITH`
- blocking mutation/DDL keywords
- setting the PostgreSQL session to read-only

Even so, database credentials should remain tightly scoped.

---

## Known gaps between architecture and current code

The README architecture above reflects both the intended design and the code you shared.

From the pasted source, these items are **described architecturally but not yet visibly implemented**:

- dynamic cache hit-rate loading/unloading
- indexed placeholder entries for cached context items
- tool descriptions loaded on demand
- `select_best_tool` API
- `StatusManager` implementation

The current code already implements the local server, tool registry, agent loop, and a persistent context service, but some of the more advanced memory-management and status-tracking ideas appear to still be planned rather than completed.

---

## License

No license file was included in the provided source snapshot. Add a `LICENSE` file if you plan to distribute the repository.
