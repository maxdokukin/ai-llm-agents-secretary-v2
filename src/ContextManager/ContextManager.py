import uuid
import time
import json
import os
import datetime
import threading
from typing import Dict, List, Any, Optional


class ContextManager:
    def __init__(self, session_id: str = "default", max_context_size: int = 128000):
        self.session_id = session_id
        self.max_context_size = max_context_size
        self._lock = threading.RLock()

        # Primary state payload
        self.segments = {
            "master_prompt": [],
            "tools": [],
            "tool_results": [],
            "data_index": [],
            "fetched_data": [],
            "message_history": []
        }

        # Embedded schema tracking
        self.session_usage: Dict[str, Dict[str, int]] = {
            self.session_id: {
                "master": 0,
                "tools": 0,
                "results": 0,
                "index": 0,
                "data": 0,
                "user": 0,
                "assistant": 0,
            }
        }

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(os.getcwd(), "../", "../", "data", "contexts", f"{timestamp}_{self.session_id}")
        os.makedirs(self.session_dir, exist_ok=True)
        self.state_file = os.path.join(self.session_dir, "context.json")
        self.log_file = os.path.join(self.session_dir, "context.log")

        self._log_action("session_initialized", {"session_id": self.session_id, "max_context": self.max_context_size})

    def _log_action(self, action: str, details: dict):
        with self._lock:
            try:
                log_entry = {
                    "timestamp": time.time(),
                    "datetime": datetime.datetime.now().isoformat(),
                    "action": action,
                    "details": details
                }
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                print(f"Failed to write to context.log: {e}")

    def _create_entry(self, content: Any, associated_id: Optional[str] = None) -> Dict:
        return {"id": str(uuid.uuid4()), "timestamp": time.time(), "content": content, "associated_id": associated_id}

    def _save_state(self):
        with self._lock:
            state = {
                "session_usage": self.session_usage,
                "segments": self.segments
            }
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=4)

    def update_usage(self, category: str, amount: int):
        with self._lock:
            if category in self.session_usage[self.session_id]:
                self.session_usage[self.session_id][category] += max(0, amount)
        self._save_state()
        self._log_action("update_usage", {"category": category, "amount": amount})

    def add_master_prompt(self, text: str):
        entry = self._create_entry(text)
        with self._lock:
            self.segments["master_prompt"].append(entry)
        self._save_state()
        self._log_action("add_master_prompt", {"id": entry["id"], "length": len(text)})
        return entry["id"]

    def add_tool(self, tool_schema: dict):
        entry = self._create_entry(tool_schema)
        with self._lock:
            self.segments["tools"].append(entry)
        self._save_state()
        self._log_action("add_tool", {"id": entry["id"], "tool_name": tool_schema.get("name", "unknown")})
        return entry["id"]

    def add_tool_result(self, tool_name: str, result: str, associated_id: Optional[str] = None):
        entry = self._create_entry({"tool_name": tool_name, "result": result}, associated_id)
        with self._lock:
            self.segments["tool_results"].append(entry)
        self._save_state()
        self._log_action("add_tool_result", {"id": entry["id"], "tool_name": tool_name, "result_length": len(result)})
        return entry["id"]

    def add_data_index(self, index_dict: dict):
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "index_data": index_dict
        }
        with self._lock:
            self.segments["data_index"].append(entry)
        self._save_state()
        self._log_action("add_data_index", {"id": entry["id"], "keys": list(index_dict.keys())})
        return entry["id"]

    def add_fetched_data(self, tool_name: str, data: Any, associated_id: Optional[str] = None):
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "tool_name": tool_name,
            "associated_id": associated_id,
            "data": data
        }
        with self._lock:
            self.segments["fetched_data"].append(entry)
        self._save_state()
        self._log_action("add_fetched_data", {"id": entry["id"], "tool_name": tool_name})
        return entry["id"]

    def add_message(self, role: str, content: str):
        entry = self._create_entry({"role": role, "content": content})
        with self._lock:
            self.segments["message_history"].append(entry)
        self._save_state()
        self._log_action("add_message", {"id": entry["id"], "role": role, "content_length": len(content)})
        return entry["id"]

    def get_context(self) -> List[Dict[str, Any]]:
        with self._lock:
            messages = []
            sys_block = []

            if self.segments["master_prompt"]:
                sys_block.append("=== SYSTEM ===\n" + "\n".join([e["content"] for e in self.segments["master_prompt"]]))

            if self.segments["tools"]:
                sys_block.append("\n=== TOOLS ===\n" + json.dumps([e["content"] for e in self.segments["tools"]]))

            if self.segments["data_index"]:
                idx_lines = []
                for e in self.segments["data_index"]:
                    for k, v in e["index_data"].items():
                        idx_lines.append(f"- {k}: {v}")
                if idx_lines:
                    sys_block.append("\n=== DATA INDEX ===\n" + "\n".join(idx_lines))

            if self.segments["fetched_data"]:
                fd_blocks = []
                for e in self.segments["fetched_data"]:
                    fd_blocks.append(
                        f"Tool: {e['tool_name']}\nAssociated Message ID: {e['associated_id']}\n\n{e['data']}")
                if fd_blocks:
                    sys_block.append("\n=== FETCHED DATA ===\n" + "\n---\n".join(fd_blocks))

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