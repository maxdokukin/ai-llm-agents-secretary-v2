import uuid
import time
import json
import os
import datetime


class ContextManager:
    def __init__(self):
        # Global Settings
        self.context_size = 4096  # Default char count
        self.compression_enabled = False

        # Master Prompt
        self.master_prompt = ""

        # Context Data
        self.context_data = []
        self.context_data_limit = 50

        # Chat History
        self.chat_history = []
        self.chat_history_limit = 50

        # --- Persistence & Logging Setup ---
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join("contexts", timestamp)
        os.makedirs(self.session_dir, exist_ok=True)

        self.state_file = os.path.join(self.session_dir, "context.json")
        self.log_file = os.path.join(self.session_dir, "context.log")

        self._log_action("SERVER_START", {"message": f"Session initialized at {timestamp}"})
        self._save_state()

    # --- Internal Persistence Helpers ---
    def _log_action(self, action: str, details: dict):
        log_entry = {
            "timestamp": time.time(),
            "action": action,
            "details": details
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    def _save_state(self):
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.get_full_state(), f, indent=4)

    def get_full_state(self) -> dict:
        return {
            "config": {
                "context_size": self.context_size,
                "context_data_limit": self.context_data_limit,
                "chat_history_limit": self.chat_history_limit,
                "compression_enabled": self.compression_enabled
            },
            "master_prompt": self.master_prompt,
            "context_data": self.context_data,
            "chat_history": self.chat_history
        }

    # --- Global Context Size ---
    def set_context_size(self, size: int):
        self.context_size = size
        self._log_action("SET_CONTEXT_SIZE", {"size": size})
        self._save_state()

    def get_context_size(self) -> int:
        return self.context_size

    # --- Master Prompt ---
    def set_master_prompt(self, prompt: str):
        self.master_prompt = prompt
        self._log_action("SET_MASTER_PROMPT", {"prompt": prompt})
        self._save_state()

    def get_master_prompt(self) -> str:
        return self.master_prompt

    # --- Context Data ---
    def add_context_data(self, label: str, data: str):
        entry = {
            "entry_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "label": label,
            "data": data
        }
        self.context_data.append(entry)
        if len(self.context_data) > self.context_data_limit:
            self.context_data = self.context_data[-self.context_data_limit:]

        self._log_action("ADD_CONTEXT_DATA", {"entry_id": entry["entry_id"], "label": label})
        self._save_state()
        return entry["entry_id"]

    def get_context_data(self):
        return self.context_data

    def set_context_data_limit(self, limit: int):
        self.context_data_limit = limit
        if len(self.context_data) > self.context_data_limit:
            self.context_data = self.context_data[-self.context_data_limit:]

        self._log_action("SET_DATA_LIMIT", {"limit": limit})
        self._save_state()

    # --- Chat History ---
    def _add_message(self, role: str, msg: str):
        entry = {
            "entry_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "role": role,
            "content": msg
        }
        self.chat_history.append(entry)
        if len(self.chat_history) > self.chat_history_limit:
            self.chat_history = self.chat_history[-self.chat_history_limit:]

        self._log_action("ADD_MESSAGE", {"role": role, "entry_id": entry["entry_id"]})
        self._save_state()
        return entry["entry_id"]

    def add_user_message(self, msg: str):
        return self._add_message("user", msg)

    def add_llm_message(self, msg: str):
        return self._add_message("llm", msg)

    def set_chat_history_limit(self, limit: int):
        self.chat_history_limit = limit
        if len(self.chat_history) > self.chat_history_limit:
            self.chat_history = self.chat_history[-self.chat_history_limit:]

        self._log_action("SET_CHAT_LIMIT", {"limit": limit})
        self._save_state()

    def get_message_history(self, author: str = "all"):
        if author == "all":
            return self.chat_history
        return [msg for msg in self.chat_history if msg["role"] == author]

    # --- Compression ---
    def set_context_compression(self, enabled: bool):
        self.compression_enabled = enabled
        self._log_action("SET_COMPRESSION", {"enabled": enabled})
        self._save_state()

    # --- Outputs & Visualizations ---
    def print_context(self) -> str:
        parts = []
        if self.master_prompt:
            parts.append(f"System: {self.master_prompt}")

        if self.context_data:
            data_str = json.dumps(self.context_data, indent=2)
            parts.append(f"Context Data:\n{data_str}")

        if self.chat_history:
            for msg in self.chat_history:
                parts.append(f"{msg['role'].capitalize()}: {msg['content']}")

        full_context = "\n\n".join(parts)

        if self.compression_enabled and len(full_context) > self.context_size:
            full_context = full_context[:self.context_size] + "\n...[TRUNCATED/COMPRESSED]"

        return full_context

    def get_usage_stats(self):
        prompt_len = len(self.master_prompt)
        data_len = sum(len(json.dumps(d)) for d in self.context_data)
        history_len = sum(len(m["content"]) for m in self.chat_history)
        total_used = prompt_len + data_len + history_len
        return prompt_len, data_len, history_len, total_used

    def get_cli_bar(self, width: int = 50) -> str:
        prompt_len, data_len, history_len, total_used = self.get_usage_stats()

        if self.context_size == 0:
            return "Context size limit is 0"

        prompt_chars = int((prompt_len / self.context_size) * width)
        data_chars = int((data_len / self.context_size) * width)
        history_chars = int((history_len / self.context_size) * width)

        total_chars = prompt_chars + data_chars + history_chars
        if total_chars > width:
            scale = width / total_chars
            prompt_chars = int(prompt_chars * scale)
            data_chars = int(data_chars * scale)
            history_chars = int(history_chars * scale)

        free_chars = width - (prompt_chars + data_chars + history_chars)

        bar = (
                "#" * prompt_chars +
                "x" * data_chars +
                "*" * history_chars +
                "." * max(0, free_chars)
        )

        return f"------------\n|{bar}|\n------------"