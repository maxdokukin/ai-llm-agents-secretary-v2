from pathlib import Path
from typing import Dict, Optional, List

import json
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Import the logic engine
from src.ContextManager.ContextManager import ContextManager


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

MAX_CONTEXT_SIZE = 32768

app = FastAPI(title="Context Engineering Server")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# Storage for active sessions
sessions: Dict[str, ContextManager] = {
    "default": ContextManager(max_context_size=MAX_CONTEXT_SIZE)
}


# Sidecar usage ledger for the utilization bar.
# This tracks usage by category when data is added through these API endpoints.
session_usage: Dict[str, Dict[str, int]] = {
    "default": {
        "master": 0,
        "tools": 0,
        "results": 0,
        "index": 0,
        "data": 0,
        "user": 0,
        "assistant": 0,
    }
}


def empty_usage() -> Dict[str, int]:
    return {
        "master": 0,
        "tools": 0,
        "results": 0,
        "index": 0,
        "data": 0,
        "user": 0,
        "assistant": 0,
    }


def normalize_usage_counts(counts: Optional[Dict[str, int]]) -> Dict[str, int]:
    normalized = empty_usage()

    if not counts:
        return normalized

    for key in normalized.keys():
        value = counts.get(key, 0)
        if isinstance(value, (int, float)):
            normalized[key] = max(0, int(value))

    # Backward compatibility for any in-memory sessions that still have "history".
    legacy_history = counts.get("history", 0)
    if isinstance(legacy_history, (int, float)) and legacy_history > 0:
        normalized["assistant"] += int(legacy_history)

    return normalized


def get_cm(session_id: str) -> ContextManager:
    if session_id not in sessions:
        sessions[session_id] = ContextManager(max_context_size=MAX_CONTEXT_SIZE)

    if session_id not in session_usage:
        session_usage[session_id] = empty_usage()
    else:
        session_usage[session_id] = normalize_usage_counts(session_usage[session_id])

    return sessions[session_id]


def add_usage(session_id: str, category: str, amount: int) -> None:
    if session_id not in session_usage:
        session_usage[session_id] = empty_usage()
    else:
        session_usage[session_id] = normalize_usage_counts(session_usage[session_id])

    if category not in session_usage[session_id]:
        session_usage[session_id][category] = 0

    session_usage[session_id][category] += max(0, amount)


def get_stat_value(stats: dict, keys: List[str], default: int = 0) -> int:
    for key in keys:
        value = stats.get(key)
        if isinstance(value, (int, float)):
            return int(value)

    return default


def classify_message(msg: dict) -> str:
    role = str(msg.get("role", "")).lower()
    content = str(msg.get("content", ""))
    content_lower = content.lower()

    msg_type = str(msg.get("type", "")).lower()
    msg_kind = str(msg.get("kind", "")).lower()
    category = str(msg.get("category", "")).lower()

    marker = f"{msg_type} {msg_kind} {category} {content_lower}"

    if role == "tool":
        return "results"

    if role == "user":
        return "user"

    if role == "assistant":
        return "assistant"

    if role == "system":
        if "tool schema" in marker or '"parameters"' in marker or '"function"' in marker:
            return "tools"
        if "data index" in marker:
            return "index"
        if "fetched data" in marker or "retrieved data" in marker:
            return "data"
        return "master"

    if "tool_schema" in marker or "tool schema" in marker:
        return "tools"

    if "tool_result" in marker or "tool result" in marker:
        return "results"

    if "data_index" in marker or "data index" in marker:
        return "index"

    if "fetched_data" in marker or "fetched data" in marker or "retrieved data" in marker:
        return "data"

    return "assistant"


def estimate_usage_from_messages(messages: List[dict]) -> Dict[str, int]:
    counts = empty_usage()

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        content = msg.get("content", "")
        if content is None:
            content = ""

        category = classify_message(msg)
        counts[category] += len(str(content))

    return counts


def calculate_usage(session_id: str) -> dict:
    cm = get_cm(session_id)
    stats = cm.calculate_free_space()
    messages = cm.get_context()

    max_context = get_stat_value(
        stats,
        ["max", "maximum", "max_context_size", "total", "limit"],
        MAX_CONTEXT_SIZE,
    )

    stats_used = get_stat_value(stats, ["used", "usage", "consumed"], 0)

    tracked_counts = normalize_usage_counts(session_usage.get(session_id, empty_usage()))
    tracked_total = sum(tracked_counts.values())

    # If the sidecar ledger has no data, fall back to classifying assembled messages.
    if tracked_total == 0:
        tracked_counts = estimate_usage_from_messages(messages)
        tracked_total = sum(tracked_counts.values())

    # Preserve compatibility with ContextManager's own accounting.
    # If calculate_free_space() reports more used chars than the category ledger,
    # put the difference into assistant so the visual bar still fills correctly.
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


# --- UI ROUTES ---

@app.get("/")
async def master_dashboard(request: Request):
    """Directory of all active sessions."""
    session_cards = []

    for session_id in sessions.keys():
        usage = calculate_usage(session_id)
        session_cards.append({
            "session_id": session_id,
            "used": usage["used"],
            "max": usage["max"],
            "free": usage["free"],
        })

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "sessions": session_cards,
        },
    )


@app.get("/{session_id}")
async def individual_dashboard(request: Request, session_id: str):
    """Live-updating deep dive dashboard for a single session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "session_id": session_id,
            "session_id_json": json.dumps(session_id),
        },
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


@app.post("/api/context/tools")
async def add_tool(payload: ToolPayload):
    serialized_tool = json.dumps(payload.tool_schema, ensure_ascii=False)
    entry_id = get_cm(payload.session_id).add_tool(payload.tool_schema)
    add_usage(payload.session_id, "tools", len(serialized_tool))
    return {"status": "success", "id": entry_id}


@app.post("/api/context/message")
async def add_message(payload: MessagePayload):
    entry_id = get_cm(payload.session_id).add_message(payload.role, payload.content)

    usage_category = classify_message({
        "role": payload.role,
        "content": payload.content,
    })

    add_usage(payload.session_id, usage_category, len(payload.content))
    return {"status": "success", "id": entry_id}


@app.post("/api/context/tool_results")
async def add_tool_result(payload: ToolResultPayload):
    entry_id = get_cm(payload.session_id).add_tool_result(
        payload.tool_name,
        payload.result,
        payload.associated_id,
    )
    add_usage(payload.session_id, "results", len(payload.result))
    return {"status": "success", "id": entry_id}


@app.get("/api/context/stats/{session_id}")
async def get_stats(session_id: str):
    return get_cm(session_id).calculate_free_space()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7999)