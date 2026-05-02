import uuid
import time
import json
import os
import datetime
import threading
from typing import Dict, List, Any, Optional


class ContextManager:
    def __init__(self, max_context_size: int = 32768):
        self.max_context_size = max_context_size
        self._lock = threading.Lock()
        self.segments = {
            "master_prompt": [],
            "tools": [],
            "tool_results": [],
            "data_index": [],
            "fetched_data": [],
            "message_history": []
        }

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(os.getcwd(), "../", "../", "data", "contexts", timestamp)
        os.makedirs(self.session_dir, exist_ok=True)
        self.state_file = os.path.join(self.session_dir, "context.json")
        self.log_file = os.path.join(self.session_dir, "context.log")

    def _create_entry(self, content: Any, associated_id: Optional[str] = None) -> Dict:
        return {"id": str(uuid.uuid4()), "timestamp": time.time(), "content": content, "associated_id": associated_id}

    def _save_state(self):
        with self._lock:
            with open(self.state_file, "w") as f: json.dump(self.segments, f, indent=4)

    def add_master_prompt(self, text: str):
        entry = self._create_entry(text)
        self.segments["master_prompt"].append(entry)
        self._save_state()
        return entry["id"]

    def add_tool(self, tool_schema: dict):
        entry = self._create_entry(tool_schema)
        self.segments["tools"].append(entry)
        self._save_state()
        return entry["id"]

    def add_tool_result(self, tool_name: str, result: str, associated_id: Optional[str] = None):
        entry = self._create_entry({"tool_name": tool_name, "result": result}, associated_id)
        self.segments["tool_results"].append(entry)
        self._save_state()
        return entry["id"]

    def add_message(self, role: str, content: str):
        entry = self._create_entry({"role": role, "content": content})
        self.segments["message_history"].append(entry)
        self._save_state()
        return entry["id"]

    def get_context(self) -> List[Dict[str, Any]]:
        # This mirrors your block-compilation logic exactly
        messages = []
        sys_block = []
        if self.segments["master_prompt"]:
            sys_block.append("=== SYSTEM ===\n" + "\n".join([e["content"] for e in self.segments["master_prompt"]]))
        if self.segments["tools"]:
            sys_block.append("\n=== TOOLS ===\n" + json.dumps([e["content"] for e in self.segments["tools"]]))

        if sys_block:
            messages.append({"role": "system", "content": "\n".join(sys_block)})

        for msg in self.segments["message_history"]:
            messages.append(msg["content"])
            mid = msg["id"]
            related_tools = [t for t in self.segments["tool_results"] if t["associated_id"] == mid]
            if related_tools:
                t_content = "\n".join(
                    [f"Tool: {t['content']['tool_name']}\nOutput: {t['content']['result']}" for t in related_tools])
                messages.append({"role": "system", "content": f"--- TOOL RESULTS ---\n{t_content}"})
        return messages

    def calculate_free_space(self):
        used = len(json.dumps(self.get_context()))
        return {"used": used, "free": max(0, self.max_context_size - used), "max": self.max_context_size}