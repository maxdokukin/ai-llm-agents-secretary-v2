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
LLM_SERVER = "http://localhost:8080/v1"

# --- Sample Data Index ---
from src.data.supabase import fetch_db_index


@app.get("/")
async def get_index():
    return FileResponse(TEMPLATES_DIR / "index.html")


def tool_returns_data(t_manager: ToolManager, tool_name: str) -> bool:
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
    if returns_data:
        await http.post(f"{CTX_SERVER}/fetched_data", json={
            "session_id": session_id,
            "tool_name": tool_name,
            "data": result,
            "associated_id": associated_id
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

    # Generate a fresh session ID for this specific WebSocket connection
    session_id = f"session_{uuid.uuid4().hex[:8]}"

    t_manager = ToolManager(toolbox_dir="../src/ToolManager/toolbox")
    llm_client = AsyncOpenAI(base_url=LLM_SERVER, api_key="sk-local")

    print(f"\n--- INITIALIZING CONTEXT SESSION: {session_id} ---")

    async with httpx.AsyncClient() as http:
        try:
            with open("/Users/max/Codebase/github/ai-llm-agents-secretary-v2/llm/prompts/secretary_prompt.txt") as f:
                master_prompt = f.read()

            await http.post(f"{CTX_SERVER}/master_prompt", json={
                "session_id": session_id,
                "text": master_prompt
            })

            await http.post(f"{CTX_SERVER}/data_index", json={
                "session_id": session_id,
                "index_data": fetch_db_index(),
            })

            schemas = t_manager.get_schemas()
            for schema in schemas:
                await http.post(f"{CTX_SERVER}/tools", json={
                    "session_id": session_id,
                    "tool_schema": schema
                })

            print(f"✅ Context {session_id} successfully initialized on Port 7999.")
            print(f"🛠️  Registered {len(schemas)} tools.")

            await websocket.send_json({
                "type": "system",
                "content": f"[CONNECTED] Session ID: {session_id}"
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

            async with httpx.AsyncClient() as http:
                resp = await http.post(f"{CTX_SERVER}/message", json={
                    "session_id": session_id,
                    "role": "user",
                    "content": user_text
                })
                user_msg_id = resp.json()["id"]

            while True:
                async with httpx.AsyncClient() as http:
                    ctx_resp = await http.get(f"{CTX_SERVER}/assemble/{session_id}")
                    current_messages = ctx_resp.json()["messages"]

                if current_messages and current_messages[-1].get("role") != "user":
                    current_messages.append({
                        "role": "user",
                        "content": "Action completed. Please evaluate the context and continue."
                    })

                schemas = t_manager.get_schemas()
                response = await llm_client.chat.completions.create(
                    model="local-model",
                    messages=current_messages,
                    tools=schemas if schemas else None,
                    stream=True
                )

                turn_content = ""
                tool_calls = {}
                is_thinking = False
                has_native_reasoning = False

                async for chunk in response:
                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta

                    # 1. Handle Native API Reasoning (DeepSeek API format)
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        if not has_native_reasoning:
                            turn_content += "<think>\n"
                            has_native_reasoning = True
                        turn_content += reasoning
                        await websocket.send_json({
                            "type": "agent_thinking_chunk",
                            "content": reasoning
                        })

                    # 2. Handle Text Content and Legacy <think> tags
                    if delta.content:
                        if has_native_reasoning and not is_thinking:
                            turn_content += "\n</think>\n"
                            has_native_reasoning = False

                        content_piece = delta.content
                        turn_content += content_piece

                        if "<think>" in content_piece:
                            is_thinking = True
                            content_piece = content_piece.replace("<think>", "")

                        if "</think>" in content_piece:
                            is_thinking = False
                            content_piece = content_piece.replace("</think>", "")
                            if content_piece.strip():
                                await websocket.send_json({
                                    "type": "agent_chunk",
                                    "content": content_piece
                                })
                            continue

                        if is_thinking:
                            await websocket.send_json({
                                "type": "agent_thinking_chunk",
                                "content": content_piece
                            })
                        else:
                            await websocket.send_json({
                                "type": "agent_chunk",
                                "content": content_piece
                            })

                    # 3. Handle Tool Calls
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

                assistant_log = turn_content
                if tool_calls:
                    for tc in tool_calls.values():
                        assistant_log += f"\n[Action: Executed tool '{tc['name']}' with args: {tc['arguments']}]"

                assistant_msg_id = None
                if assistant_log.strip():
                    async with httpx.AsyncClient() as http:
                        resp = await http.post(f"{CTX_SERVER}/message", json={
                            "session_id": session_id,
                            "role": "assistant",
                            "content": assistant_log.strip()
                        })
                        assistant_msg_id = resp.json()["id"]

                if not tool_calls:
                    async with httpx.AsyncClient() as http:
                        stats = await http.get(f"{CTX_SERVER}/stats/{session_id}")
                        s = stats.json()

                        await websocket.send_json({
                            "type": "stats",
                            "content": f"Session: {session_id} | Context: {s['used']}/{s['max']} chars"
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
                            session_id=session_id,
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
        print(f"Client disconnected from session: {session_id}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")