import uuid
import time
import json
import os
import datetime
import threading
from typing import Dict, List, Any, Optional


class ContextManager:
    def __init__(self, max_context_size: int = 32768):
        # max_context_size is treated as characters here. 32768 chars ≈ 8192 tokens.
        self.max_context_size = max_context_size
        self._lock = threading.Lock()

        # ---------------------------------------------------------
        # Storage segments perfectly mirroring the visual architecture
        # ---------------------------------------------------------
        self.segments = {
            "master_prompt": [],
            "tools": [],
            "tool_results": [],
            "data_index": [],
            "fetched_data": [],
            "message_history": []
        }

        # --- Persistence & Logging Setup Restored ---
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Ensure path matches your project structure
        self.session_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "contexts", timestamp)
        os.makedirs(self.session_dir, exist_ok=True)

        self.state_file = os.path.join(self.session_dir, "context.json")
        self.log_file = os.path.join(self.session_dir, "context.log")

        self._log_action("SERVER_START",
                         {"message": f"Session initialized at {timestamp}", "max_size_chars": self.max_context_size})
        self._save_state()

    # --- Persistence Helpers ---
    def _log_action(self, action: str, details: dict):
        log_entry = {
            "timestamp": time.time(),
            "action": action,
            "details": details
        }
        with self._lock:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

    def _save_state(self):
        state = {
            "config": {"max_context_size_chars": self.max_context_size},
            "segments": self.segments
        }
        with self._lock:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)

    # --- Internal Helpers ---
    def _create_entry(self, content: Any, associated_id: Optional[str] = None) -> Dict:
        """Generates the standard traceable wrapper for every block."""
        return {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "content": content,
            "associated_id": associated_id
        }

    def _remove_entry(self, segment: str, entry_id: str) -> bool:
        """Generic remove function for any segment."""
        initial_length = len(self.segments[segment])
        self.segments[segment] = [e for e in self.segments[segment] if e["id"] != entry_id]
        success = len(self.segments[segment]) < initial_length

        if success:
            self._log_action("REMOVE_ENTRY", {"segment": segment, "entry_id": entry_id})
            self._save_state()

        return success

    # ==========================================
    # 1. Master Prompt APIs
    # ==========================================
    def add_master_prompt(self, text: str) -> str:
        entry = self._create_entry(text)
        self.segments["master_prompt"].append(entry)
        self._log_action("ADD_MASTER_PROMPT", {"entry_id": entry["id"]})
        self._save_state()
        return entry["id"]

    def remove_master_prompt(self, entry_id: str) -> bool:
        return self._remove_entry("master_prompt", entry_id)

    # ==========================================
    # 2. Tools APIs
    # ==========================================
    def add_tool(self, tool_schema: dict) -> str:
        """Add a tool's JSON schema so the LLM knows it exists."""
        entry = self._create_entry(tool_schema)
        self.segments["tools"].append(entry)

        # Safely extract tool name for logging
        tool_name = "unknown"
        if isinstance(tool_schema, dict):
            tool_name = tool_schema.get("function", {}).get("name", "unknown")

        self._log_action("ADD_TOOL", {"entry_id": entry["id"], "tool_name": tool_name})
        self._save_state()
        return entry["id"]

    def remove_tool(self, entry_id: str) -> bool:
        return self._remove_entry("tools", entry_id)

    # ==========================================
    # 3. Tool Results APIs
    # ==========================================
    def add_tool_result(self, tool_name: str, result: str, associated_id: Optional[str] = None) -> str:
        """Add the output of an executed tool. Link to a user message ID if applicable."""
        content = {"tool_name": tool_name, "result": result}
        entry = self._create_entry(content, associated_id)
        self.segments["tool_results"].append(entry)
        self._log_action("ADD_TOOL_RESULT",
                         {"entry_id": entry["id"], "tool_name": tool_name, "associated_id": associated_id})
        self._save_state()
        return entry["id"]

    def remove_tool_result(self, entry_id: str) -> bool:
        return self._remove_entry("tool_results", entry_id)

    # ==========================================
    # 4. Data Index APIs
    # ==========================================
    def add_data_index(self, index_info: str) -> str:
        """Add summaries or pointers of data the LLM can ask to fetch."""
        entry = self._create_entry(index_info)
        self.segments["data_index"].append(entry)
        self._log_action("ADD_DATA_INDEX", {"entry_id": entry["id"]})
        self._save_state()
        return entry["id"]

    def remove_data_index(self, entry_id: str) -> bool:
        return self._remove_entry("data_index", entry_id)

    # ==========================================
    # 5. Fetched Data APIs
    # ==========================================
    def add_fetched_data(self, label: str, data: str, associated_id: Optional[str] = None) -> str:
        """Add raw data blobs retrieved from the index. Link to a user message ID if applicable."""
        content = {"label": label, "data": data}
        entry = self._create_entry(content, associated_id)
        self.segments["fetched_data"].append(entry)
        self._log_action("ADD_FETCHED_DATA", {"entry_id": entry["id"], "label": label, "associated_id": associated_id})
        self._save_state()
        return entry["id"]

    def remove_fetched_data(self, entry_id: str) -> bool:
        return self._remove_entry("fetched_data", entry_id)

    # ==========================================
    # 6. Message History APIs
    # ==========================================
    def add_message(self, role: str, content: str) -> str:
        """Add a conversational turn (user or assistant)."""
        msg_content = {"role": role, "content": content}
        entry = self._create_entry(msg_content)
        self.segments["message_history"].append(entry)
        self._log_action("ADD_MESSAGE", {"entry_id": entry["id"], "role": role})
        self._save_state()
        return entry["id"]

    def remove_message(self, entry_id: str) -> bool:
        return self._remove_entry("message_history", entry_id)

    # ==========================================
    # LLM Assembly
    # ==========================================
    def get_context(self) -> List[Dict[str, Any]]:
        """
        Builds the final payload to be sent to the LLM.
        It physically reconstructs the blocks from the internal state.
        """
        messages = []

        # --- BLOCK COMPILATION: System, Tools, and Index ---
        system_content = []

        if self.segments["master_prompt"]:
            system_content.append("=== SYSTEM INSTRUCTIONS ===")
            for entry in self.segments["master_prompt"]:
                system_content.append(entry["content"])

        if self.segments["tools"]:
            system_content.append("\n=== AVAILABLE TOOLS ===")
            system_content.append("You have access to the following tools. Output standard JSON to trigger them.")
            tools_list = [entry["content"] for entry in self.segments["tools"]]
            system_content.append(json.dumps(tools_list, indent=2))

        if self.segments["data_index"]:
            system_content.append("\n=== DATA INDEX ===")
            for entry in self.segments["data_index"]:
                system_content.append(entry["content"])

        # Create one unified System Message block
        if system_content:
            messages.append({"role": "system", "content": "\n".join(system_content)})

        # --- BLOCK COMPILATION: Conversation, Fetched Data, Tool Results ---
        for msg_entry in self.segments["message_history"]:
            # 1. Append the chat message
            messages.append({
                "role": msg_entry["content"]["role"],
                "content": msg_entry["content"]["content"]
            })

            # 2. Check for associative data linked to this specific message
            msg_id = msg_entry["id"]
            associated_data = [d for d in self.segments["fetched_data"] if d["associated_id"] == msg_id]
            associated_tools = [t for t in self.segments["tool_results"] if t["associated_id"] == msg_id]

            if associated_data or associated_tools:
                block_content = []
                if associated_tools:
                    block_content.append("--- TOOL EXECUTIONS ---")
                    for t in associated_tools:
                        block_content.append(f"Tool: {t['content']['tool_name']}\nResult: {t['content']['result']}")

                if associated_data:
                    block_content.append("--- FETCHED DATA ---")
                    for d in associated_data:
                        block_content.append(f"{d['content']['label']}:\n{d['content']['data']}")

                messages.append({
                    "role": "system",
                    "content": "\n".join(block_content)
                })

        # --- Fallback: Unassociated Data/Tools ---
        unassociated_data = [d for d in self.segments["fetched_data"] if not d["associated_id"]]
        unassociated_tools = [t for t in self.segments["tool_results"] if not t["associated_id"]]

        if unassociated_data or unassociated_tools:
            block_content = []
            if unassociated_tools:
                block_content.append("--- RECENT TOOL EXECUTIONS ---")
                for t in unassociated_tools:
                    block_content.append(f"Tool: {t['content']['tool_name']}\nResult: {t['content']['result']}")

            if unassociated_data:
                block_content.append("--- RECENT FETCHED DATA ---")
                for d in unassociated_data:
                    block_content.append(f"{d['content']['label']}:\n{d['content']['data']}")

            messages.append({"role": "system", "content": "\n".join(block_content)})

        return messages

    def calculate_free_space(self) -> dict:
        """Approximates the remaining space based on character counts."""
        total_chars = len(json.dumps(self.get_context()))
        return {
            "used": total_chars,
            "free": max(0, self.max_context_size - total_chars),
            "max": self.max_context_size
        }