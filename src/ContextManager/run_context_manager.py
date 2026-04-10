from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from ContextManager import ContextManager
import uvicorn  # Imported to run the server directly

app = FastAPI(title="Context Engineering Tool API")

# Singleton instance for eternal state across requests
cm = ContextManager()

# --- Pydantic Models ---
class IntValue(BaseModel):
    value: int

class StringValue(BaseModel):
    value: str

class ContextDataModel(BaseModel):
    label: str
    data: str

class BooleanValue(BaseModel):
    value: bool

# --- Root State Endpoint ---
@app.get("/")
def get_all_context_data():
    """Prints all the data we currently have about the context."""
    return cm.get_full_state()

# --- Size Endpoints ---
@app.post("/api/context/size")
def set_context_size(payload: IntValue):
    cm.set_context_size(payload.value)
    return {"status": "success", "context_size": cm.get_context_size()}

@app.get("/api/context/size")
def get_context_size():
    return {"context_size": cm.get_context_size()}

# --- Master Prompt Endpoints ---
@app.post("/api/context/master_prompt")
def set_master_prompt(payload: StringValue):
    cm.set_master_prompt(payload.value)
    return {"status": "success"}

@app.get("/api/context/master_prompt")
def get_master_prompt():
    return {"master_prompt": cm.get_master_prompt()}

# --- Context Data Endpoints ---
@app.post("/api/context/data")
def add_context_data(payload: ContextDataModel):
    entry_id = cm.add_context_data(payload.label, payload.data)
    return {"status": "success", "entry_id": entry_id}

@app.get("/api/context/data")
def get_context_data():
    return {"context_data": cm.get_context_data()}

@app.post("/api/context/data/limit")
def set_context_data_limit(payload: IntValue):
    cm.set_context_data_limit(payload.value)
    return {"status": "success", "data_limit": payload.value}

# --- Chat History Endpoints ---
@app.post("/api/context/chat/user")
def add_user_message(payload: StringValue):
    entry_id = cm.add_user_message(payload.value)
    return {"status": "success", "entry_id": entry_id}

@app.post("/api/context/chat/llm")
def add_llm_message(payload: StringValue):
    entry_id = cm.add_llm_message(payload.value)
    return {"status": "success", "entry_id": entry_id}

@app.post("/api/context/chat/limit")
def set_chat_history_limit(payload: IntValue):
    cm.set_chat_history_limit(payload.value)
    return {"status": "success", "chat_history_limit": payload.value}

@app.get("/api/context/chat")
def get_message_history(author: str = Query("all", description="Can be 'all', 'user', or 'llm'")):
    return {"chat_history": cm.get_message_history(author)}

# --- Operations & Visualizations ---
@app.post("/api/context/compression")
def set_context_compression(payload: BooleanValue):
    cm.set_context_compression(payload.value)
    return {"status": "success", "compression_enabled": payload.value}

@app.get("/api/context/print")
def print_context():
    return {"assembled_context": cm.print_context()}

@app.get("/api/context/cli_bar")
def get_cli_bar():
    return {"cli_bar": cm.get_cli_bar()}

@app.get("/api/context/gui_bar", response_class=HTMLResponse)
def get_nicer_bar():
    """Returns a nicely formatted HTML/CSS split graph bar."""
    prompt_len, data_len, history_len, total_used = cm.get_usage_stats()
    max_size = cm.get_context_size()

    # Calculate percentages
    p_prompt = min(100, (prompt_len / max_size) * 100) if max_size else 0
    p_data = min(100, (data_len / max_size) * 100) if max_size else 0
    p_history = min(100, (history_len / max_size) * 100) if max_size else 0

    # Cap total at 100% for visual sanity if over limits
    total_p = p_prompt + p_data + p_history
    if total_p > 100:
        scale = 100 / total_p
        p_prompt *= scale
        p_data *= scale
        p_history *= scale

    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: sans-serif; padding: 20px; background: #1e1e2e; color: #cdd6f4; }}
            .progress-container {{ width: 100%; max-width: 800px; height: 30px; background-color: #313244; border-radius: 8px; overflow: hidden; display: flex; margin-top: 20px; }}
            .segment {{ height: 100%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: #11111b; transition: width 0.3s ease; }}
            .prompt {{ background-color: #f38ba8; width: {p_prompt}%; }}
            .data {{ background-color: #f9e2af; width: {p_data}%; }}
            .history {{ background-color: #89b4fa; width: {p_history}%; }}
            .legend {{ display: flex; gap: 15px; margin-top: 10px; font-size: 14px; }}
            .dot {{ height: 12px; width: 12px; border-radius: 50%; display: inline-block; margin-right: 5px; }}
        </style>
    </head>
    <body>
        <h2>Context Memory Usage</h2>
        <p>Total Size limit: <b>{max_size}</b> chars | Used: <b>{total_used}</b> chars</p>

        <div class="progress-container">
            <div class="segment prompt"></div>
            <div class="segment data"></div>
            <div class="segment history"></div>
        </div>

        <div class="legend">
            <div><span class="dot" style="background-color: #f38ba8;"></span>Master Prompt ({prompt_len}c)</div>
            <div><span class="dot" style="background-color: #f9e2af;"></span>Data Block ({data_len}c)</div>
            <div><span class="dot" style="background-color: #89b4fa;"></span>Chat History ({history_len}c)</div>
        </div>
    </body>
    </html>
    """
    return html_content


# --- Execution Block ---
if __name__ == "__main__":
    print("Starting Context Engineering Server...")
    print(f"Logging sessions to: {cm.session_dir}")
    print("Full Context Dump available at: http://localhost:7999/")
    print("API Documentation available at: http://localhost:7999/docs")
    print("GUI Bar available at: http://localhost:7999/api/context/gui_bar")
    uvicorn.run("run_context_manager:app", host="0.0.0.0", port=7999, reload=True)