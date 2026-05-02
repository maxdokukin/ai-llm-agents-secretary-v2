import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Callable


class ToolManager:
    def __init__(self, toolbox_dir: str = "toolbox"):
        """
        Initializes the ToolManager by recursively scanning the toolbox directory.
        """
        # Resolve path robustly relative to the execution root
        self.toolbox_dir = Path(toolbox_dir).resolve()
        self.tools: Dict[str, Callable] = {}  # Maps tool names to their execute functions
        self.schemas: List[Dict[str, Any]] = []  # Stores the JSON schemas for the LLM

        self._register_all_tools()

    def _register_all_tools(self) -> None:
        if not self.toolbox_dir.exists():
            print(f"⚠️ Directory '{self.toolbox_dir}' does not exist. No tools loaded.")
            return

        # Recursively find all .py files (-r behavior)
        for py_file in self.toolbox_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            # Auto-infer name: e.g., toolbox/math/add.py -> math_add
            rel_path = py_file.relative_to(self.toolbox_dir)
            inferred_name = "_".join(rel_path.with_suffix('').parts)
            module_name = f"toolbox_{inferred_name}"

            # Dynamically load the module
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)

                    # Enforce the contract: must have 'tool_schema' and 'execute'
                    if hasattr(module, 'tool_schema') and hasattr(module, 'execute'):
                        schema = module.tool_schema

                        # 1. Ensure OpenAI-compatible wrapping
                        if "type" not in schema:
                            schema = {
                                "type": "function",
                                "function": schema.get("function", schema)
                                # Fallback if authored without "function" key
                            }

                        # 2. Inject the auto-inferred name
                        schema["function"]["name"] = inferred_name

                        self.schemas.append(schema)
                        self.tools[inferred_name] = module.execute
                        print(f"✅ Registered tool: {inferred_name} (from {rel_path})")
                    else:
                        print(f"⏭️ Skipping {rel_path}: Missing 'tool_schema' or 'execute()'.")

                except Exception as e:
                    print(f"❌ Error loading {rel_path}: {e}")

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Returns the list of tools formatted for the LLM client."""
        return self.schemas

    def execute_tool(self, tool_name: str, arguments: Any) -> str:
        """
        Executes a registered tool based on the LLM's request.
        Accepts arguments as either a JSON string or a parsed dictionary.
        """
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found."

        try:
            # Safely handle both stringified JSON and pre-parsed dictionaries
            if isinstance(arguments, str):
                args = json.loads(arguments) if arguments.strip() else {}
            elif isinstance(arguments, dict):
                args = arguments
            else:
                args = {}

            # Execute the mapped function
            result = self.tools[tool_name](**args)
            return str(result)

        except json.JSONDecodeError:
            return "Error: Invalid JSON arguments provided by the LLM."
        except Exception as e:
            return f"Error executing '{tool_name}': {str(e)}"