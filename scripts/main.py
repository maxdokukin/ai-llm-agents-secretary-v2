import uuid
import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

# 1. Import your custom Context Manager
# Because main.py is in scripts/, and src/ is one level up,
# you might need to adjust your PYTHONPATH or use relative imports
# if this still throws a ModuleNotFoundError. Assuming your IDE handles it:
from src.ContextManager.ContextManager import ContextManager

# --- Path Resolution ---
# This dynamically finds the root directory of your project based on where main.py lives
BASE_DIR = Path(__file__).resolve().parent       # This is the /scripts directory
ROOT_DIR = BASE_DIR.parent                       # This is the root project directory

STATIC_DIR = ROOT_DIR / "static"
TEMPLATES_DIR = ROOT_DIR / "templates"

# --- Dummy ToolManager for demonstration purposes ---
class ToolManager:
    def get_schemas(self):
        return []  # Return your actual schemas here

    def execute_tool(self, tool_name, arguments_json):
        return f"Executed {tool_name} successfully."
# ----------------------------------------------------

app = FastAPI()

# Mount the static directory.
# Arg 1: The URL path ("/static").
# Arg 2: The absolute file path to the directory on your drive.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def get_index():
    # Serve the frontend HTML file using the absolute path
    return FileResponse(TEMPLATES_DIR / "index.html")

# --- Internal Helper to rebuild OpenAI Payload from ContextManager ---
def build_openai_messages(ctx: ContextManager) -> list:
    messages = []

    if ctx.get_master_prompt():
        messages.append({"role": "system", "content": ctx.get_master_prompt()})

    context_data = ctx.get_context_data()
    if context_data:
        data_strings = [f"{d['label']}:\n{d['data']}" for d in context_data]
        messages.append({"role": "system", "content": "AVAILABLE CONTEXT DATA:\n" + "\n\n".join(data_strings)})

    for msg in ctx.get_message_history():
        if msg["role"] == "user":
            messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "llm":
            messages.append({"role": "assistant", "content": msg["content"]})
        elif msg["role"] == "assistant_tool_call":
            messages.append(json.loads(msg["content"]))
        elif msg["role"] == "tool_result":
            messages.append(json.loads(msg["content"]))

    return messages

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    manager = ToolManager()
    client = AsyncOpenAI(base_url="http://localhost:8080/v1", api_key="sk-local")

    ctx = ContextManager()
    ctx.set_context_size(8192)
    ctx.set_master_prompt("You are a capable AI secretary. Utilize your tools efficiently.")

    try:
        while True:
            user_text = await websocket.receive_text()
            ctx.add_user_message(user_text)

            await websocket.send_json({"type": "system", "content": "[SYSTEM] Agent loop initiated..."})

            while True:
                request_payload = {
                    "model": "local-model",
                    "messages": build_openai_messages(ctx),
                    "tools": manager.get_schemas(),
                    "stream": True
                }

                if not request_payload["tools"]:
                    request_payload.pop("tools")
                else:
                    request_payload["tool_choice"] = "auto"

                response = await client.chat.completions.create(**request_payload)

                turn_content = ""
                assembled_tool_calls = {}

                async for chunk in response:
                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta

                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        await websocket.send_json({
                            "type": "agent_thinking_chunk",
                            "content": delta.reasoning_content
                        })

                    if hasattr(delta, 'content') and delta.content:
                        await websocket.send_json({
                            "type": "agent_chunk",
                            "content": delta.content
                        })
                        turn_content += delta.content

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

                if not assembled_tool_calls:
                    ctx.add_llm_message(turn_content if turn_content else "")
                    await websocket.send_json({"type": "stats", "content": ctx.get_cli_bar(width=60)})
                    await websocket.send_json({"type": "done"})
                    break

                formatted_tool_calls = []
                for idx, tc in assembled_tool_calls.items():
                    formatted_tool_calls.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]}
                    })

                assistant_msg_dict = {
                    "role": "assistant",
                    "content": turn_content if turn_content else None,
                    "tool_calls": formatted_tool_calls
                }

                ctx._add_message("assistant_tool_call", json.dumps(assistant_msg_dict))

                for tc in formatted_tool_calls:
                    tool_name = tc["function"]["name"]
                    tool_args = tc["function"]["arguments"]
                    tool_id = tc["id"]

                    await websocket.send_json({"type": "system", "content": f"[PROCESS] Executing: {tool_name}..."})

                    result = manager.execute_tool(tool_name, tool_args)

                    tool_result_dict = {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": str(result)
                    }
                    ctx._add_message("tool_result", json.dumps(tool_result_dict))
                    ctx.add_context_data(f"Tool Output ({tool_name})", str(result))

                    await websocket.send_json({"type": "system", "content": f"[PROCESS] Complete: {tool_name}."})

    except WebSocketDisconnect:
        print("Client disconnected.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)