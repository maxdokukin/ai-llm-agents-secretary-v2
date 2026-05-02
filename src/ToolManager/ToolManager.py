import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Callable


class ToolManager:
    def __init__(self, toolbox_dir: str = "toolbox"):
        """
        Initializes the ToolManager by recursively scanning the toolbox directory.
        """
        self.toolbox_dir = Path(toolbox_dir).resolve()
        self.tools: Dict[str, Callable] = {}
        self.tool_returns_data: Dict[str, bool] = {}
        self.schemas: List[Dict[str, Any]] = []

        self._register_all_tools()

    def _is_data_tool(self, py_file: Path) -> bool:
        """
        Returns True when the tool file is located inside a directory named 'data'.

        Example:
            toolbox/data/get_users.py -> True
            toolbox/math/add.py       -> False
        """
        rel_path = py_file.relative_to(self.toolbox_dir)
        return "data" in rel_path.parts[:-1]

    def _register_all_tools(self) -> None:
        if not self.toolbox_dir.exists():
            print(f"⚠️ Directory '{self.toolbox_dir}' does not exist. No tools loaded.")
            return

        for py_file in self.toolbox_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            rel_path = py_file.relative_to(self.toolbox_dir)
            inferred_name = ".".join(rel_path.with_suffix("").parts)
            module_name = f"toolbox.{inferred_name}"

            returns_data = self._is_data_tool(py_file)

            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)

                try:
                    spec.loader.exec_module(module)

                    if hasattr(module, "tool_schema") and hasattr(module, "execute"):
                        raw_schema = module.tool_schema

                        # Ensure OpenAI-compatible wrapping
                        if "type" not in raw_schema:
                            schema = {
                                "type": "function",
                                "function": raw_schema.get("function", raw_schema),
                            }
                        else:
                            schema = dict(raw_schema)
                            schema["function"] = dict(schema.get("function", {}))

                        # Inject auto-inferred metadata
                        schema["function"]["name"] = inferred_name
                        schema["function"]["returns_data"] = returns_data

                        self.schemas.append(schema)
                        self.tools[inferred_name] = module.execute
                        self.tool_returns_data[inferred_name] = returns_data

                        print(
                            f"✅ Registered tool: {inferred_name} "
                            f"(from {rel_path}, returns_data={returns_data})"
                        )
                    else:
                        print(f"⏭️ Skipping {rel_path}: Missing 'tool_schema' or 'execute()'.")

                except Exception as e:
                    print(f"❌ Error loading {rel_path}: {e}")

    def get_schemas(self) -> List[Dict[str, Any]]:
        """
        Returns the list of tools formatted for the LLM client.
        Each schema includes function.returns_data.
        """
        return self.schemas

    def execute_tool(self, tool_name: str, arguments: Any) -> str:
        """
        Executes a registered tool based on the LLM's request.
        Accepts arguments as either a JSON string or a parsed dictionary.
        """
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found."

        try:
            if isinstance(arguments, str):
                args = json.loads(arguments) if arguments.strip() else {}
            elif isinstance(arguments, dict):
                args = arguments
            else:
                args = {}

            result = self.tools[tool_name](**args)
            return str(result)

        except json.JSONDecodeError:
            return "Error: Invalid JSON arguments provided by the LLM."
        except Exception as e:
            return f"Error executing '{tool_name}': {str(e)}"