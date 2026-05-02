import uuid
import json
import asyncio
import importlib.util
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

# 1. Import your custom Context Manager
from src.ContextManager.ContextManager import ContextManager

# --- Path Resolution ---
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

STATIC_DIR = ROOT_DIR / "static"
TEMPLATES_DIR = ROOT_DIR / "templates"


# --- Integrated ToolManager ---
class ToolManager:
    def __init__(self, toolbox_dir: str = "toolbox"):
        self.toolbox_dir = BASE_DIR / toolbox_dir
        self.tools = {}
        self.schemas = []
        self._register_all_tools()

    def _register_all_tools(self):
        if not self.toolbox_dir.exists():
            print(f"Directory '{self.toolbox_dir}' does not exist.")
            return

        for py_file in self.toolbox_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            rel_path = py_file.relative_to(self.toolbox_dir)
            inferred_name = "_".join(rel_path.with_suffix('').parts)
            module_name = f"toolbox_{inferred_name}"

            spec = importlib.util.spec_from_file_location(module_name, py_file)
            module = importlib.util.module_from_spec(spec)

            try:
                spec.loader.exec_module(module)
                if hasattr(module, 'tool_schema') and hasattr(module, 'execute'):
                    schema = module.tool_schema
                    if "function" not in schema:
                        schema["function"] = {}
                    schema["function"]["name"] = inferred_name

                    self.schemas.append(schema)
                    self.tools[inferred_name] = module.execute
                    print(f"Registered tool: {inferred_name}")
                else:
                    print(f"Skipping {rel_path}: Missing 'tool_schema' or 'execute()'.")
            except Exception as e:
                print(f"Error loading {rel_path}: {e}")

    def get_schemas(self) -> list:
        return self.schemas

    def execute_tool(self, tool_name: str, arguments_json: str) -> str:
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found."

        try:
            args = json.loads(arguments_json) if arguments_json else {}
            result = self.tools[tool_name](**args)
            return str(result)
        except json.JSONDecodeError:
            return "Error: Invalid JSON arguments provided."
        except Exception as e:
            return f"Error executing '{tool_name}': {str(e)}"


# ----------------------------------------------------

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def get_index():
    return FileResponse(TEMPLATES_DIR / "index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # 1. Init Tools
    manager = ToolManager()
    client = AsyncOpenAI(base_url="http://localhost:8080/v1", api_key="sk-local")

    # 2. Init Context Manager & Architecture Segments
    ctx = ContextManager(max_context_size=8192)

    # Pass master prompt
    ctx.add_master_prompt("You are a capable AI secretary. Utilize your tools efficiently.")

    # Pass tools to the context segment
    for schema in manager.get_schemas():
        ctx.add_tool(schema)

    # Pass data index (dummy for now)
    ctx.add_data_index(json.dumps({"name": "maxim dokukin"}))

    try:
        while True:
            # 3. Handle incoming message
            user_text = await websocket.receive_text()

            # Trace the user message ID so we can associate tools/data with it
            user_msg_id = ctx.add_message("user", user_text)

            await websocket.send_json({"type": "system", "content": "[SYSTEM] Agent loop initiated..."})

            while True:
                # 4. Get the perfectly constructed context directly from the manager
                request_payload = {
                    "model": "local-model",
                    "messages": ctx.get_context(),
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

                # If the agent spoke before calling a tool (or just replied normally), add it to history
                if turn_content:
                    ctx.add_message("assistant", turn_content)

                # 5. Tool Call Execution Loop
                if not assembled_tool_calls:
                    # Agent is done (no tools called), break the loop
                    stats = ctx.calculate_free_space()
                    stat_str = f"Context: {stats['used']} / {stats['max']} (Free: {stats['free']})"

                    await websocket.send_json({"type": "stats", "content": stat_str})
                    await websocket.send_json({"type": "done"})
                    break

                # The agent called tools. Execute them and add to Context Manager.
                for idx, tc in assembled_tool_calls.items():
                    tool_name = tc["name"]
                    tool_args = tc["arguments"]

                    await websocket.send_json({"type": "system", "content": f"[PROCESS] Executing: {tool_name}..."})

                    result = manager.execute_tool(tool_name, tool_args)

                    # 6. Append tool call results, associated with the original user query that caused it
                    ctx.add_tool_result(
                        tool_name=tool_name,
                        result=str(result),
                        associated_id=user_msg_id
                    )

                    await websocket.send_json({"type": "system", "content": f"[PROCESS] Complete: {tool_name}."})

                # The agent loop `while True` repeats here!
                # Because we called `ctx.add_tool_result`, `ctx.get_context()` will now automatically
                # include the results in the payload on the next iteration.

    except WebSocketDisconnect:
        print("Client disconnected.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)