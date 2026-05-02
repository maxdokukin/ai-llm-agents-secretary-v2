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
# Generate a unique Session ID for this run
SESSION_ID = f"session_{uuid.uuid4().hex[:8]}"


@app.get("/")
async def get_index():
    return FileResponse(TEMPLATES_DIR / "index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # 1. Init Local Tool Discovery
    t_manager = ToolManager(toolbox_dir="../src/ToolManager/toolbox")

    # 2. LLM Inference Client (llama-server / Ollama)
    llm_client = AsyncOpenAI(base_url="http://localhost:8080/v1", api_key="sk-local")

    # 3. Bootstrap Context Server (Port 7999)
    print(f"\n--- INITIALIZING CONTEXT SESSION: {SESSION_ID} ---")

    async with httpx.AsyncClient() as http:
        try:
            # Initialize with Master Prompt
            await http.post(f"{CTX_SERVER}/master_prompt", json={
                "session_id": SESSION_ID,
                "text": "You are a capable AI secretary. Use tools to satisfy requests."
            })

            # Register discovered tools
            schemas = t_manager.get_schemas()
            for schema in schemas:
                await http.post(f"{CTX_SERVER}/tools", json={
                    "session_id": SESSION_ID,
                    "tool_schema": schema
                })

            print(f"✅ Context {SESSION_ID} successfully initialized on Port 7999.")
            print(f"🛠️  Registered {len(schemas)} tools.")

            # Inform the UI
            await websocket.send_json({
                "type": "system",
                "content": f"[CONNECTED] Session ID: {SESSION_ID}"
            })

        except httpx.ConnectError:
            print(f"❌ FAILED to connect to Context Server at {CTX_SERVER}")
            await websocket.send_json({"type": "system", "content": "❌ Error: Context Server (7999) is offline."})
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
                response = await llm_client.chat.completions.create(
                    model="local-model",
                    messages=current_messages,
                    tools=t_manager.get_schemas() if t_manager.get_schemas() else None,
                    stream=True
                )

                turn_content = ""
                tool_calls = {}

                async for chunk in response:
                    if not chunk.choices: continue
                    delta = chunk.choices[0].delta

                    if delta.content:
                        turn_content += delta.content
                        await websocket.send_json({"type": "agent_chunk", "content": delta.content})

                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls:
                                tool_calls[idx] = {"id": tc.id, "name": "", "arguments": ""}
                            if tc.function.name: tool_calls[idx]["name"] += tc.function.name
                            if tc.function.arguments: tool_calls[idx]["arguments"] += tc.function.arguments

                # Step D: Log Assistant Reply
                if turn_content:
                    async with httpx.AsyncClient() as http:
                        await http.post(f"{CTX_SERVER}/message", json={
                            "session_id": SESSION_ID,
                            "role": "assistant",
                            "content": turn_content
                        })

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
                    await websocket.send_json({"type": "system", "content": f"[EXE] Calling {tc['name']}..."})
                    result = t_manager.execute_tool(tc["name"], tc["arguments"])

                    async with httpx.AsyncClient() as http:
                        await http.post(f"{CTX_SERVER}/tool_results", json={
                            "session_id": SESSION_ID,
                            "tool_name": tc["name"],
                            "result": str(result),
                            "associated_id": user_msg_id
                        })
                    await websocket.send_json({"type": "system", "content": f"[OK] {tc['name']} returned data."})

    except WebSocketDisconnect:
        print(f"Client disconnected from session: {SESSION_ID}")


if __name__ == "__main__":
    import uvicorn

    # Added colors to uvicorn logs for easier reading
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")