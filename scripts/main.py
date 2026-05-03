import uuid
import json
import httpx
import asyncio
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

# --- Path Resolution ---
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
STATIC_DIR = ROOT_DIR / "static"
TEMPLATES_DIR = ROOT_DIR / "templates"

app = FastAPI(title="Local LLM Orchestrator")

# Ensure static/templates exist
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Logic Configuration ---
from src.ToolManager.ToolManager import ToolManager

CTX_SERVER = "http://localhost:7999/api/context"
SESSION_ID = f"session_{uuid.uuid4().hex[:8]}"

# --- Sample Data Index ---
DATA_INDEX = {
    "user.name": "max dokukin",
    "user.email": "maxdokukin@icloud.com",
}


@app.get("/")
async def get_index():
    return FileResponse(TEMPLATES_DIR / "index.html")


def build_data_index_text(data_index: dict) -> str:
    lines = ["Data Index"]

    for key, value in data_index.items():
        lines.append(f"- {key}: {value}")

    return "\n".join(lines)


def tool_returns_data(t_manager: ToolManager, tool_name: str) -> bool:
    """
    Returns whether a registered tool is a data-returning tool.

    Primary source:
        ToolManager.tool_returns_data

    Fallback:
        function.returns_data inside the tool schema
    """
    if hasattr(t_manager, "tool_returns_data"):
        return bool(t_manager.tool_returns_data.get(tool_name, False))

    for schema in t_manager.get_schemas():
        function_schema = schema.get("function", {})
        if function_schema.get("name") == tool_name:
            return bool(function_schema.get("returns_data", False))

    return False


async def append_tool_result_to_context(
    http: httpx.AsyncClient,
    *,
    session_id: str,
    tool_name: str,
    result: str,
    associated_id: str,
    returns_data: bool,
) -> None:
    """
    Routes tool output into the correct context bucket.

    Data tools:
        Appended as system-level fetched data.

    Non-data tools:
        Appended as regular tool results.
    """
    if returns_data:
        await http.post(f"{CTX_SERVER}/message", json={
            "session_id": session_id,
            "role": "system",
            "content": (
                f"Fetched Data\n"
                f"Tool: {tool_name}\n"
                f"Associated Message ID: {associated_id}\n\n"
                f"{result}"
            )
        })
    else:
        await http.post(f"{CTX_SERVER}/tool_results", json={
            "session_id": session_id,
            "tool_name": tool_name,
            "result": result,
            "associated_id": associated_id
        })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # 1. Init Local Tool Discovery
    t_manager = ToolManager(toolbox_dir="../src/ToolManager/toolbox")

    # 2. LLM Inference Client
    llm_client = AsyncOpenAI(base_url="http://localhost:8080/v1", api_key="sk-local")

    # 3. Bootstrap Context Server
    print(f"\n--- INITIALIZING CONTEXT SESSION: {SESSION_ID} ---")

    async with httpx.AsyncClient() as http:
        try:
            await http.post(f"{CTX_SERVER}/master_prompt", json={
                "session_id": SESSION_ID,
                "text": "You are a capable AI secretary. Use tools to satisfy requests."
            })

            await http.post(f"{CTX_SERVER}/message", json={
                "session_id": SESSION_ID,
                "role": "system",
                "content": build_data_index_text(DATA_INDEX),
            })

            schemas = t_manager.get_schemas()
            for schema in schemas:
                await http.post(f"{CTX_SERVER}/tools", json={
                    "session_id": SESSION_ID,
                    "tool_schema": schema
                })

            print(f"✅ Context {SESSION_ID} successfully initialized on Port 7999.")
            print(f"🛠️  Registered {len(schemas)} tools.")
            print(f"📇 Loaded data index with {len(DATA_INDEX)} entries.")

            await websocket.send_json({
                "type": "system",
                "content": f"[CONNECTED] Session ID: {SESSION_ID}"
            })

        except httpx.ConnectError:
            print(f"❌ FAILED to connect to Context Server at {CTX_SERVER}")
            await websocket.send_json({
                "type": "system",
                "content": "❌ Error: Context Server (7999) is offline."
            })
            return

    try:
        while True:
            user_text = await websocket.receive_text()

            # Step A: Register User Message
            async with httpx.AsyncClient() as http:
                resp = await http.post(f"{CTX_SERVER}/message", json={
                    "session_id": SESSION_ID,
                    "role": "user",
                    "content": user_text
                })
                user_msg_id = resp.json()["id"]

            # --- AGENT REACT LOOP ---
            while True:
                # Step B: Fetch Compiled Context Ledger
                async with httpx.AsyncClient() as http:
                    ctx_resp = await http.get(f"{CTX_SERVER}/assemble/{SESSION_ID}")
                    current_messages = ctx_resp.json()["messages"]

                # Step C: LLM Stream
                schemas = t_manager.get_schemas()
                response = await llm_client.chat.completions.create(
                    model="local-model",
                    messages=current_messages,
                    tools=schemas if schemas else None,
                    stream=True
                )

                turn_content = ""
                tool_calls = {}

                async for chunk in response:
                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta

                    if delta.content:
                        turn_content += delta.content
                        await websocket.send_json({
                            "type": "agent_chunk",
                            "content": delta.content
                        })

                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index

                            if idx not in tool_calls:
                                tool_calls[idx] = {
                                    "id": tc.id,
                                    "name": "",
                                    "arguments": ""
                                }

                            if tc.function.name:
                                tool_calls[idx]["name"] += tc.function.name

                            if tc.function.arguments:
                                tool_calls[idx]["arguments"] += tc.function.arguments

                # Step D: Log Assistant Reply & Ensure Tool Intent is Recorded
                assistant_log = turn_content
                if tool_calls:
                    for tc in tool_calls.values():
                        assistant_log += f"\n[Action: Executed tool '{tc['name']}' with args: {tc['arguments']}]"

                assistant_msg_id = None
                if assistant_log.strip():
                    async with httpx.AsyncClient() as http:
                        resp = await http.post(f"{CTX_SERVER}/message", json={
                            "session_id": SESSION_ID,
                            "role": "assistant",
                            "content": assistant_log.strip()
                        })
                        assistant_msg_id = resp.json()["id"]

                # Step E: Handle Completion or Tools
                if not tool_calls:
                    async with httpx.AsyncClient() as http:
                        stats = await http.get(f"{CTX_SERVER}/stats/{SESSION_ID}")
                        s = stats.json()

                        await websocket.send_json({
                            "type": "stats",
                            "content": f"Session: {SESSION_ID} | Context: {s['used']}/{s['max']} chars"
                        })

                    await websocket.send_json({"type": "done"})
                    break

                for tc in tool_calls.values():
                    tool_name = tc["name"]
                    returns_data = tool_returns_data(t_manager, tool_name)

                    await websocket.send_json({
                        "type": "system",
                        "content": f"[EXE] Calling {tool_name}..."
                    })

                    result = str(t_manager.execute_tool(tool_name, tc["arguments"]))

                    async with httpx.AsyncClient() as http:
                        await append_tool_result_to_context(
                            http,
                            session_id=SESSION_ID,
                            tool_name=tool_name,
                            result=result,
                            associated_id=assistant_msg_id or user_msg_id,
                            returns_data=returns_data,
                        )

                    destination = "fetched data" if returns_data else "tool results"

                    await websocket.send_json({
                        "type": "system",
                        "content": f"[OK] {tool_name} appended to {destination}."
                    })

    except WebSocketDisconnect:
        print(f"Client disconnected from session: {SESSION_ID}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")