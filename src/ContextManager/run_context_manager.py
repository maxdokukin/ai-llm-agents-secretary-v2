from pathlib import Path
from typing import Dict, Optional, List, Any
import json
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# --- IMPORT FROM OUR SEPARATED FILE ---
from src.ContextManager.ContextManager import ContextManager, MAX_CONTEXT_SIZE

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Context Engineering Server")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# --- SESSION REGISTRY ---
sessions: Dict[str, ContextManager] = {
    "default": ContextManager(session_id="default", max_context_size=MAX_CONTEXT_SIZE)
}


def get_cm(session_id: str) -> ContextManager:
    if session_id not in sessions:
        sessions[session_id] = ContextManager(session_id=session_id, max_context_size=MAX_CONTEXT_SIZE)
    return sessions[session_id]


def add_usage(session_id: str, category: str, amount: int) -> None:
    get_cm(session_id).update_usage(category, amount)


def get_stat_value(stats: dict, keys: List[str], default: int = 0) -> int:
    for key in keys:
        value = stats.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return default


def classify_message(msg: dict) -> str:
    role = str(msg.get("role", "")).lower()
    if role == "tool": return "results"
    if role == "user": return "user"
    if role == "assistant": return "assistant"
    return "master"


def estimate_usage_from_messages(messages: List[dict]) -> Dict[str, int]:
    counts = {
        "master": 0, "tools": 0, "results": 0,
        "index": 0, "data": 0, "user": 0, "assistant": 0
    }
    for msg in messages:
        if isinstance(msg, dict) and msg.get("content"):
            category = classify_message(msg)
            counts[category] += len(str(msg["content"]))
    return counts


def calculate_usage(session_id: str) -> dict:
    cm = get_cm(session_id)
    stats = cm.calculate_free_space()
    messages = cm.get_context()

    max_context = get_stat_value(stats, ["max", "maximum", "max_context_size", "total", "limit"], MAX_CONTEXT_SIZE)
    stats_used = get_stat_value(stats, ["used", "usage", "consumed"], 0)

    tracked_counts = cm.session_usage.get(session_id, {
        "master": 0, "tools": 0, "results": 0,
        "index": 0, "data": 0, "user": 0, "assistant": 0
    }).copy()

    tracked_total = sum(tracked_counts.values())

    if tracked_total == 0:
        tracked_counts = estimate_usage_from_messages(messages)
        tracked_total = sum(tracked_counts.values())

    effective_used = max(stats_used, tracked_total)

    if effective_used > tracked_total:
        tracked_counts["assistant"] += effective_used - tracked_total
        tracked_total = effective_used

    free = max(0, max_context - tracked_total)

    return {
        "max": max_context,
        "used": tracked_total,
        "free": free,
        "counts": tracked_counts,
        "stats": stats,
    }


# --- Pydantic Models ---
class TextPayload(BaseModel):
    text: str
    session_id: str = "default"


class DataIndexPayload(BaseModel):
    index_data: dict
    session_id: str = "default"


class FetchedDataPayload(BaseModel):
    tool_name: str
    data: Any
    associated_id: Optional[str] = None
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

class PngPayload(BaseModel):
    session_id: str = "default"
    folder_path: str


# --- UI ROUTES ---
@app.get("/")
async def master_dashboard(request: Request):
    session_cards = []
    for session_id in sessions.keys():
        usage = calculate_usage(session_id)
        session_cards.append({
            "session_id": session_id,
            "used": usage["used"],
            "max": usage["max"],
            "free": usage["free"],
        })
    return templates.TemplateResponse(request=request, name="index.html", context={"sessions": session_cards})


@app.get("/{session_id}")
async def individual_dashboard(request: Request, session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "session_id": session_id,
            "session_id_json": json.dumps(session_id)
        }
    )


# --- API ENDPOINTS ---
@app.get("/api/context/assemble/{session_id}")
async def assemble_context(session_id: str):
    return {"messages": get_cm(session_id).get_context()}


@app.get("/api/context/usage/{session_id}")
async def get_usage(session_id: str):
    return calculate_usage(session_id)


@app.post("/api/context/master_prompt")
async def add_master_prompt(payload: TextPayload):
    entry_id = get_cm(payload.session_id).add_master_prompt(payload.text)
    add_usage(payload.session_id, "master", len(payload.text))
    return {"status": "success", "id": entry_id}


@app.post("/api/context/data_index")
async def add_data_index(payload: DataIndexPayload):
    entry_id = get_cm(payload.session_id).add_data_index(payload.index_data)
    add_usage(payload.session_id, "index", len(json.dumps(payload.index_data)))
    return {"status": "success", "id": entry_id}


@app.post("/api/context/fetched_data")
async def add_fetched_data(payload: FetchedDataPayload):
    entry_id = get_cm(payload.session_id).add_fetched_data(payload.tool_name, payload.data, payload.associated_id)
    add_usage(payload.session_id, "data", len(str(payload.data)))
    return {"status": "success", "id": entry_id}


@app.post("/api/context/tools")
async def add_tool(payload: ToolPayload):
    serialized_tool = json.dumps(payload.tool_schema, ensure_ascii=False)
    entry_id = get_cm(payload.session_id).add_tool(payload.tool_schema)
    add_usage(payload.session_id, "tools", len(serialized_tool))
    return {"status": "success", "id": entry_id}


@app.post("/api/context/message")
async def add_message(payload: MessagePayload):
    entry_id = get_cm(payload.session_id).add_message(payload.role, payload.content)
    usage_category = classify_message({"role": payload.role, "content": payload.content})
    add_usage(payload.session_id, usage_category, len(payload.content))
    return {"status": "success", "id": entry_id}


@app.post("/api/context/tool_results")
async def add_tool_result(payload: ToolResultPayload):
    entry_id = get_cm(payload.session_id).add_tool_result(payload.tool_name, payload.result, payload.associated_id)
    add_usage(payload.session_id, "results", len(payload.result))
    return {"status": "success", "id": entry_id}


@app.get("/api/context/stats/{session_id}")
async def get_stats(session_id: str):
    return get_cm(session_id).calculate_free_space()


@app.post("/api/context/save_png")
async def save_png(payload: PngPayload):
    try:
        # Get usage so the PNG mirrors the exact math from the frontend
        usage_data = calculate_usage(payload.session_id)
        cm = get_cm(payload.session_id)
        filepath = cm.save_context_as_png(
            folder_path=payload.folder_path,
            usage_counts=usage_data["counts"],
            max_size=usage_data["max"]
        )
        return {"status": "success", "filepath": filepath}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7999)