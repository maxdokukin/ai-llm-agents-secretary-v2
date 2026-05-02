from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import json

# Adjust import path based on your project structure
from src.ContextManager.ContextManager import ContextManager

app = FastAPI(title="Context Engineering Tool API")

# Singleton instance for eternal state across requests (32768 chars ≈ 8192 tokens)
cm = ContextManager(max_context_size=32768)

# --- Pydantic Models ---
class SizePayload(BaseModel):
    value: int

class TextPayload(BaseModel):
    text: str

class ToolPayload(BaseModel):
    tool_schema: dict

class ToolResultPayload(BaseModel):
    tool_name: str
    result: str
    associated_id: Optional[str] = None

class FetchedDataPayload(BaseModel):
    label: str
    data: str
    associated_id: Optional[str] = None

class MessagePayload(BaseModel):
    role: str  # "user" or "assistant"
    content: str

# --- Root State Endpoint ---
@app.get("/")
def get_all_context_data():
    """Prints all the segmented data currently held in the context buffer."""
    return {"segments": cm.segments}

# --- Size Endpoints ---
@app.post("/api/context/size")
def set_context_size(payload: SizePayload):
    cm.max_context_size = payload.value
    return {"status": "success", "max_context_size_chars": cm.max_context_size}

@app.get("/api/context/size")
def get_context_size():
    return {"max_context_size_chars": cm.max_context_size}

# --- 1. Master Prompt Endpoints ---
@app.post("/api/context/master_prompt")
def add_master_prompt(payload: TextPayload):
    entry_id = cm.add_master_prompt(payload.text)
    return {"status": "success", "id": entry_id}

@app.delete("/api/context/master_prompt/{entry_id}")
def remove_master_prompt(entry_id: str):
    success = cm.remove_master_prompt(entry_id)
    if not success: raise HTTPException(status_code=404, detail="Entry not found")
    return {"status": "success"}

# --- 2. Tools Endpoints ---
@app.post("/api/context/tools")
def add_tool(payload: ToolPayload):
    entry_id = cm.add_tool(payload.tool_schema)
    return {"status": "success", "id": entry_id}

@app.delete("/api/context/tools/{entry_id}")
def remove_tool(entry_id: str):
    success = cm.remove_tool(entry_id)
    if not success: raise HTTPException(status_code=404, detail="Entry not found")
    return {"status": "success"}

# --- 3. Tool Results Endpoints ---
@app.post("/api/context/tool_results")
def add_tool_result(payload: ToolResultPayload):
    entry_id = cm.add_tool_result(payload.tool_name, payload.result, payload.associated_id)
    return {"status": "success", "id": entry_id}

@app.delete("/api/context/tool_results/{entry_id}")
def remove_tool_result(entry_id: str):
    success = cm.remove_tool_result(entry_id)
    if not success: raise HTTPException(status_code=404, detail="Entry not found")
    return {"status": "success"}

# --- 4. Data Index Endpoints ---
@app.post("/api/context/data_index")
def add_data_index(payload: TextPayload):
    entry_id = cm.add_data_index(payload.text)
    return {"status": "success", "id": entry_id}

@app.delete("/api/context/data_index/{entry_id}")
def remove_data_index(entry_id: str):
    success = cm.remove_data_index(entry_id)
    if not success: raise HTTPException(status_code=404, detail="Entry not found")
    return {"status": "success"}

# --- 5. Fetched Data Endpoints ---
@app.post("/api/context/fetched_data")
def add_fetched_data(payload: FetchedDataPayload):
    entry_id = cm.add_fetched_data(payload.label, payload.data, payload.associated_id)
    return {"status": "success", "id": entry_id}

@app.delete("/api/context/fetched_data/{entry_id}")
def remove_fetched_data(entry_id: str):
    success = cm.remove_fetched_data(entry_id)
    if not success: raise HTTPException(status_code=404, detail="Entry not found")
    return {"status": "success"}

# --- 6. Message History Endpoints ---
@app.post("/api/context/message")
def add_message(payload: MessagePayload):
    entry_id = cm.add_message(payload.role, payload.content)
    return {"status": "success", "id": entry_id}

@app.delete("/api/context/message/{entry_id}")
def remove_message(entry_id: str):
    success = cm.remove_message(entry_id)
    if not success: raise HTTPException(status_code=404, detail="Entry not found")
    return {"status": "success"}

# --- Operations & Visualizations ---
@app.get("/api/context/assemble")
def assemble_context():
    """Returns the compiled OpenAI-compatible messages array."""
    return {"messages": cm.get_context()}

@app.get("/api/context/gui_bar", response_class=HTMLResponse)
def get_nicer_bar():
    """Returns a nicely formatted HTML/CSS split graph bar."""
    stats = cm.calculate_free_space()
    used = stats["used"]
    free = stats["free"]
    max_size = stats["max"]

    # Calculate percentage
    p_used = min(100, (used / max_size) * 100) if max_size else 0

    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: sans-serif; padding: 20px; background: #1e1e2e; color: #cdd6f4; }}
            .progress-container {{ width: 100%; max-width: 800px; height: 30px; background-color: #313244; border-radius: 8px; overflow: hidden; display: flex; margin-top: 20px; box-shadow: inset 0 1px 3px rgba(0,0,0,0.5); }}
            .segment {{ height: 100%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: #11111b; transition: width 0.3s ease; }}
            .used {{ background-color: #cba6f7; width: {p_used}%; }}
            .legend {{ display: flex; gap: 15px; margin-top: 10px; font-size: 14px; }}
            .dot {{ height: 12px; width: 12px; border-radius: 50%; display: inline-block; margin-right: 5px; }}
        </style>
    </head>
    <body>
        <h2>Context Memory Usage</h2>
        <p>Total Context Window Limit: <b>{max_size}</b> characters</p>

        <div class="progress-container">
            <div class="segment used"></div>
        </div>

        <div class="legend">
            <div><span class="dot" style="background-color: #cba6f7;"></span>Used Space ({used} chars)</div>
            <div><span class="dot" style="background-color: #313244;"></span>Free Space ({free} chars)</div>
        </div>

        <h3 style="margin-top: 30px;">Raw Compilation Preview</h3>
        <pre style="background: #11111b; padding: 15px; border-radius: 8px; overflow-x: auto; font-size: 12px;">{json.dumps(cm.get_context(), indent=2)}</pre>
    </body>
    </html>
    """
    return html_content

# --- Execution Block ---
if __name__ == "__main__":
    print("Starting Context Engineering Server...")
    print("API Documentation available at: http://localhost:7999/docs")
    print("GUI Bar & Compilation Preview available at: http://localhost:7999/api/context/gui_bar")
    uvicorn.run("run_context_manager:app", host="0.0.0.0", port=7999, reload=True)