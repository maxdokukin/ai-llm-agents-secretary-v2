# ai-llm-agents-secretary-v2

A local Python agent framework that connects an OpenAI-compatible LLM server to a dynamically loaded tool system.

This project does three main things:

1. Runs a local `llama.cpp`-based OpenAI-compatible server for Gemma GGUF models.
2. Registers tools automatically from a filesystem toolbox.
3. Loops between the model and tools until the model stops making tool calls.

It also includes a standalone context-management service for storing prompt state, context blocks, chat history, and simple usage visualizations.

---

## What this project does

The agent in `scripts/main.py` sends a chat request to a local OpenAI-compatible endpoint:

- Base URL: `http://localhost:8080/v1`
- API key: `sk-local`
- Model name: `local-model`

It attaches all discovered tool schemas to the request, streams the model response, reconstructs fragmented tool calls from the stream, executes those tools locally, appends tool outputs back into conversation history, and repeats until the model returns a final answer without any tool calls.

---

## Features

- Local LLM serving through `llama-server`
- Automatic model download from allowed Hugging Face GGUF repos
- Recursive tool discovery from the `toolbox` directory
- OpenAI function/tool calling support
- Streaming assistant output and streamed tool-call reconstruction
- Shell, math, and database tools
- Separate context manager service with persistence and simple visualizations

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