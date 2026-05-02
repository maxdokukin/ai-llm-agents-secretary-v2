from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict
import uvicorn
import json

# Import the logic engine
from src.ContextManager.ContextManager import ContextManager

app = FastAPI(title="Context Engineering Server")

# Storage for active sessions
sessions: Dict[str, ContextManager] = {
    "default": ContextManager(max_context_size=32768)
}


def get_cm(session_id: str) -> ContextManager:
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


# --- UI COMPONENTS ---

def get_shared_styles():
    return """
    <style>
        body { font-family: 'Inter', -apple-system, sans-serif; background: #1e1e2e; color: #cdd6f4; padding: 20px; line-height: 1.5; }
        .container { max-width: 1000px; margin: 0 auto; }
        h1 { color: #b4befe; display: flex; align-items: center; gap: 10px; }
        .card { background: #181825; border: 1px solid #313244; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
        .progress-container { background: #313244; height: 24px; border-radius: 12px; overflow: hidden; margin: 15px 0; border: 1px solid #45475a; }
        .progress-fill { height: 100%; transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1); }
        pre { background: #11111b; padding: 15px; border-radius: 8px; border: 1px solid #313244; overflow: auto; max-height: 500px; font-size: 13px; color: #a6adc8; }
        .badge { background: #45475a; color: #f5e0dc; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: bold; }
        a { color: #89b4fa; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
    </style>
    """


@app.get("/", response_class=HTMLResponse)
async def master_dashboard():
    """Directory of all active sessions."""
    links = ""
    for s_id in sessions.keys():
        stats = sessions[s_id].calculate_free_space()
        links += f"""
        <div class="card">
            <h3>Session: <a href="/{s_id}">{s_id}</a></h3>
            <p>Usage: {stats['used']} characters</p>
            <a href="/{s_id}" style="background: #89b4fa; color: #11111b; padding: 8px 16px; border-radius: 6px; font-weight: bold;">Open Dashboard</a>
        </div>
        """

    return f"""
    <html>
        <head>{get_shared_styles()}<title>Context Hub</title></head>
        <body>
            <div class="container">
                <h1>🧠 Context Hub</h1>
                <div class="grid">{links}</div>
            </div>
        </body>
    </html>
    """


@app.get("/{session_id}", response_class=HTMLResponse)
async def individual_dashboard(session_id: str):
    """Deep dive dashboard for a single session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    cm = sessions[session_id]
    stats = cm.calculate_free_space()
    used, total = stats["used"], stats["max"]
    perc = min(100, (used / total) * 100) if total > 0 else 0

    # Dynamic Color
    color = "#cba6f7"
    if perc > 75: color = "#f9e2af"
    if perc > 90: color = "#f38ba8"

    return f"""
    <html>
        <head>
            {get_shared_styles()}
            <title>Context: {session_id}</title>
            <meta http-equiv="refresh" content="3">
        </head>
        <body>
            <div class="container">
                <p><a href="/">← Back to Hub</a></p>
                <h1>Session Dashboard <span class="badge">{session_id}</span></h1>

                <div class="card">
                    <h3>Memory Usage</h3>
                    <div class="progress-container">
                        <div class="progress-fill" style="width: {perc}%; background: {color};"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 14px;">
                        <span>Used: <b>{used}</b> chars</span>
                        <span>Limit: <b>{total}</b> chars</span>
                    </div>
                </div>

                <div class="card">
                    <h3>Compiled Context Ledger</h3>
                    <pre>{json.dumps(cm.get_context(), indent=2)}</pre>
                </div>
            </div>
        </body>
    </html>
    """


# --- API ENDPOINTS ---

@app.get("/api/context/assemble/{session_id}")
async def assemble_context(session_id: str):
    return {"messages": get_cm(session_id).get_context()}


@app.post("/api/context/master_prompt")
async def add_master_prompt(payload: TextPayload):
    entry_id = get_cm(payload.session_id).add_master_prompt(payload.text)
    return {"status": "success", "id": entry_id}


@app.post("/api/context/tools")
async def add_tool(payload: ToolPayload):
    entry_id = get_cm(payload.session_id).add_tool(payload.tool_schema)
    return {"status": "success", "id": entry_id}


@app.post("/api/context/message")
async def add_message(payload: MessagePayload):
    entry_id = get_cm(payload.session_id).add_message(payload.role, payload.content)
    return {"status": "success", "id": entry_id}


@app.post("/api/context/tool_results")
async def add_tool_result(payload: ToolResultPayload):
    entry_id = get_cm(payload.session_id).add_tool_result(
        payload.tool_name, payload.result, payload.associated_id
    )
    return {"status": "success", "id": entry_id}


@app.get("/api/context/stats/{session_id}")
async def get_stats(session_id: str):
    return get_cm(session_id).calculate_free_space()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7999)