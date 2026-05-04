import uuid
import time
import json
import os
import datetime
import threading
from typing import Dict, List, Any, Optional

# --- ONE CONSTANT TO RULE THEM ALL ---
MAX_CONTEXT_SIZE = 128000


class ContextManager:
    def __init__(self, session_id: str = "default", max_context_size: int = MAX_CONTEXT_SIZE):
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

            # 1. Base System Prompts
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

            if sys_block:
                messages.append({"role": "system", "content": "\n".join(sys_block)})

            # 2. History & Inline Logs
            for msg in self.segments["message_history"]:
                messages.append(msg["content"])
                mid = msg["id"]

                # Retrieve tools and data mapped to this particular message
                related_tools = [t for t in self.segments["tool_results"] if t["associated_id"] == mid]
                related_data = [d for d in self.segments["fetched_data"] if d["associated_id"] == mid]

                # Append them linearly instead of throwing data into the header block
                if related_tools or related_data:
                    blocks = []
                    if related_tools:
                        t_content = "\n".join(
                            [f"Tool: {t['content']['tool_name']}\nOutput: {t['content']['result']}" for t in
                             related_tools])
                        blocks.append(f"--- TOOL RESULTS ---\n{t_content}")
                    if related_data:
                        d_content = "\n".join([f"Tool: {d['tool_name']}\nData:\n{d['data']}" for d in related_data])
                        blocks.append(f"--- FETCHED DATA ---\n{d_content}")

                    messages.append({"role": "system", "content": "\n\n".join(blocks)})

            return messages

    def calculate_free_space(self):
        used = len(json.dumps(self.get_context()))
        return {"used": used, "free": max(0, self.max_context_size - used), "max": self.max_context_size}

    def save_context_as_png(self, folder_path: str, usage_counts: Optional[Dict[str, int]] = None,
                            max_size: Optional[int] = None) -> str:
        """
        Renders and saves the current context utilization bar graph as a highly stylized PNG.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.ticker as ticker
        except ImportError:
            raise ImportError("matplotlib is required. Run `pip install matplotlib`.")

        with self._lock:
            counts = usage_counts if usage_counts is not None else self.session_usage.get(self.session_id, {})
            max_val = max_size if max_size is not None else self.max_context_size

            labels = ['master', 'tools', 'results', 'index', 'data', 'user', 'assistant']
            values = [counts.get(l, 0) for l in labels]

            # Refined, highly distinct color palette
            colors = ['#64748b', '#f59e0b', '#10b981', '#a855f7', '#ec4899', '#3b82f6', '#14b8a6']

            # Taller layout to prevent squishing
            fig, ax = plt.subplots(figsize=(12, 3.5))
            left = 0

            # Plot segments with borders for better contrast
            for label, value, color in zip(labels, values, colors):
                if value > 0:
                    ax.barh(['Context Utilization'], [value], left=left, color=color, label=label.capitalize(),
                            edgecolor='white', linewidth=1)
                    left += value

            free_space = max(0, max_val - left)
            if free_space > 0:
                ax.barh(['Context Utilization'], [free_space], left=left, color='#e2e8f0', label='Free',
                        edgecolor='white', linewidth=1)

            # Clean styling
            ax.set_xlim(0, max_val)
            ax.set_title(f"Session Context: {self.session_id}", pad=20, fontsize=14, fontweight='bold', color='#334155')
            ax.set_xlabel("Characters Consumed", labelpad=10, fontsize=12, color='#475569')

            # Remove y-axis and top/right spines
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#cbd5e1')

            # Add commas to x-axis
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x):,}"))
            ax.tick_params(axis='x', colors='#64748b', labelsize=11)

            # Nicer legend formatting (4 columns)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=4, frameon=False, fontsize=11)

            plt.tight_layout()

            os.makedirs(folder_path, exist_ok=True)
            filename = f"context_usage_{self.session_id}_{int(time.time())}.png"
            filepath = os.path.join(folder_path, filename)

            # Slightly off-white background to make bars pop
            plt.savefig(filepath, bbox_inches="tight", dpi=200, facecolor='#f8fafc')
            plt.close()

            self._log_action("save_png", {"filepath": filepath})
            return filepath