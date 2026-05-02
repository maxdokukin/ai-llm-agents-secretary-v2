from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict
import uvicorn
import json

# Import the logic engine
from src.ContextManager.ContextManager import ContextManager

app = FastAPI(title="Context Engineering Server")

# Dictionary to hold multiple sessions if needed
# For now, we'll use "default" as the main session
sessions: Dict[str, ContextManager] = {
    "default": ContextManager(max_context_size=32768)
}

def get_cm(session_id: str = "default") -> ContextManager:
    if session_id not in sessions:
        sessions[session_id] = ContextManager(max_context_size=32768)
    return sessions[session_id]

# --- Pydantic Models ---
class TextPayload(BaseModel):
    text: str
    session_id: str = "default"

class ToolPayload(BaseModel):
    tool_schema: dict
    session_id: str = "default"

class ToolResultPayload(BaseModel):
    tool_name: str
    result: str
    associated_id: Optional[str] = None
    session_id: str = "default"

class MessagePayload(BaseModel):
    role: str
    content: str
    session_id: str = "default"

@app.get("/api/context/assemble/{session_id}")
async def assemble_context(session_id: str):
    return {"messages": get_cm(session_id).get_context()}

@app.post("/api/context/master_prompt")
async def add_master_prompt(payload: TextPayload):
    id = get_cm(payload.session_id).add_master_prompt(payload.text)
    return {"id": id}

@app.post("/api/context/tools")
async def add_tool(payload: ToolPayload):
    id = get_cm(payload.session_id).add_tool(payload.tool_schema)
    return {"id": id}

@app.post("/api/context/message")
async def add_message(payload: MessagePayload):
    id = get_cm(payload.session_id).add_message(payload.role, payload.content)
    return {"id": id}

@app.post("/api/context/tool_results")
async def add_tool_result(payload: ToolResultPayload):
    id = get_cm(payload.session_id).add_tool_result(payload.tool_name, payload.result, payload.associated_id)
    return {"id": id}

@app.get("/api/context/stats/{session_id}")
async def get_stats(session_id: str):
    return get_cm(session_id).calculate_free_space()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7999)