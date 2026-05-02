import uuid
import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI, APIConnectionError

# --- Custom System Architecture Imports ---
from src.ContextManager.ContextManager import ContextManager
from src.ToolManager.ToolManager import ToolManager

# --- Path Resolution ---
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

STATIC_DIR = ROOT_DIR / "static"
TEMPLATES_DIR = ROOT_DIR / "templates"

app = FastAPI(title="Local LLM Orchestrator")

# Ensure static/templates exist before mounting to avoid startup errors
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def get_index():
    # Make sure you have an index.html in your templates folder!
    return FileResponse(TEMPLATES_DIR / "index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # 1. Init System Components
    # Assuming tools are in a folder named 'toolbox' relative to main.py
    manager = ToolManager("../src/ToolManager/toolbox")

    # Point this to your local inference server (LM Studio, Ollama, vLLM, etc.)
    client = AsyncOpenAI(base_url="http://localhost:8080/v1", api_key="sk-local")

    # 2. Init Context Manager (Memory Ledger)
    ctx = ContextManager(max_context_size=32768)

    # Bootstrapping the context
    ctx.add_master_prompt(
        "You are a capable AI assistant. Utilize your tools efficiently to answer the user's request.")

    # Register dynamically discovered tools into the context memory
    for schema in manager.get_schemas():
        ctx.add_tool(schema)

    # (Optional) Seed data index
    ctx.add_data_index("System Administrator: Maxim Dokukin")

    try:
        while True:
            # 3. Wait for User Input
            user_text = await websocket.receive_text()

            # Trace the user message ID so we can associate tools/data with it later
            user_msg_id = ctx.add_message("user", user_text)

            await websocket.send_json({"type": "system", "content": "[SYSTEM] Agent loop initiated..."})

            # --- AGENTIC REACT LOOP ---
            while True:
                # 4. Fetch the perfectly constructed context directly from the ledger
                request_payload = {
                    "model": "local-model",  # The specific name usually doesn't matter for local proxies
                    "messages": ctx.get_context(),
                    "stream": True
                }

                # Only inject tools into the payload if the manager actually found some
                available_tools = manager.get_schemas()
                if available_tools:
                    request_payload["tools"] = available_tools
                    request_payload["tool_choice"] = "auto"

                turn_content = ""
                assembled_tool_calls = {}

                try:
                    response = await client.chat.completions.create(**request_payload)

                    async for chunk in response:
                        if not chunk.choices:
                            continue

                        delta = chunk.choices[0].delta

                        # Handle DeepSeek/Open-source reasoning traces if supported
                        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                            await websocket.send_json({
                                "type": "agent_thinking_chunk",
                                "content": delta.reasoning_content
                            })

                        # Handle standard conversational output
                        if hasattr(delta, 'content') and delta.content:
                            await websocket.send_json({
                                "type": "agent_chunk",
                                "content": delta.content
                            })
                            turn_content += delta.content

                        # Handle Tool Call streams
                        if hasattr(delta, 'tool_calls') and delta.tool_calls:
                            for tc_chunk in delta.tool_calls:
                                idx = tc_chunk.index
                                if idx not in assembled_tool_calls:
                                    tc_id = tc_chunk.id if tc_chunk.id else f"call_{uuid.uuid4().hex[:8]}"
                                    assembled_tool_calls[idx] = {"id": tc_id, "name": "", "arguments": ""}

                                if tc_chunk.function.name:
                                    assembled_tool_calls[idx]["name"] += tc_chunk.function.name
                                if tc_chunk.function.arguments:
                                    assembled_tool_calls[idx]["arguments"] += tc_chunk.function.arguments

                except APIConnectionError:
                    error_msg = "⚠️ Could not connect to the local LLM server at http://localhost:8080/v1"
                    await websocket.send_json({"type": "system", "content": error_msg})
                    break
                except Exception as e:
                    await websocket.send_json({"type": "system", "content": f"⚠️ Inference Error: {str(e)}"})
                    break

                # Record the agent's conversational response in the ledger
                if turn_content:
                    ctx.add_message("assistant", turn_content)

                # 5. Loop Break Condition (No tools called)
                if not assembled_tool_calls:
                    stats = ctx.calculate_free_space()
                    stat_str = f"Context: {stats['used']} / {stats['max']} chars (Free: {stats['free']})"

                    await websocket.send_json({"type": "stats", "content": stat_str})
                    await websocket.send_json({"type": "done"})
                    break  # Exit the ReAct loop and wait for the next user message

                # 6. Tool Execution Phase
                for idx, tc in assembled_tool_calls.items():
                    tool_name = tc["name"]
                    tool_args = tc["arguments"]

                    await websocket.send_json({"type": "system", "content": f"[PROCESS] Executing: {tool_name}..."})

                    # Execute the tool via the ToolManager
                    result = manager.execute_tool(tool_name, tool_args)

                    # Append tool call results, tightly coupled to the original user query ID
                    ctx.add_tool_result(
                        tool_name=tool_name,
                        result=str(result),
                        associated_id=user_msg_id
                    )

                    await websocket.send_json({"type": "system", "content": f"[PROCESS] Complete: {tool_name}."})

                # The `while True` loop repeats here!
                # The LLM will now be prompted again, and `ctx.get_context()` will automatically
                # include the freshly executed tool results so the LLM can interpret them.

    except WebSocketDisconnect:
        print("Client disconnected from WebSocket.")


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Local LLM Orchestrator on http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)