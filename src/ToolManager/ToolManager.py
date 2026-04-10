import json
import importlib.util
from pathlib import Path


class ToolManager:
    def __init__(self, toolbox_dir: str = "toolbox"):
        """
        Initializes the ToolManager by recursively scanning the toolbox directory.
        """
        self.toolbox_dir = Path(__file__).parent / toolbox_dir
        self.tools = {}  # Maps tool names to their execute functions
        self.schemas = []  # Stores the JSON schemas for the LLM
        self._register_all_tools()

    def _register_all_tools(self):
        if not self.toolbox_dir.exists():
            print(f"Directory '{self.toolbox_dir}' does not exist.")
            return

        # Recursively find all .py files (-r behavior)
        for py_file in self.toolbox_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            rel_path = py_file.relative_to(self.toolbox_dir)
            module_name = f"toolbox_{rel_path.stem}"

            # Dynamically load the module
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            module = importlib.util.module_from_spec(spec)

            try:
                spec.loader.exec_module(module)

                # Enforce the contract: must have 'tool_schema' and 'execute'
                if hasattr(module, 'tool_schema') and hasattr(module, 'execute'):
                    schema = module.tool_schema
                    tool_name = schema['function']['name']

                    self.schemas.append(schema)
                    self.tools[tool_name] = module.execute
                    print(f"Registered tool: {tool_name} (from {rel_path})")
                else:
                    print(f"Skipping {rel_path}: Missing 'tool_schema' or 'execute()'.")
            except Exception as e:
                print(f"Error loading {rel_path}: {e}")

    def get_schemas(self) -> list:
        """Returns the list of tools formatted for the OpenAI client."""
        return self.schemas

    def execute_tool(self, tool_name: str, arguments_json: str) -> str:
        """
        Executes a registered tool based on the LLM's request.
        """
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found."

        try:
            # Parse the JSON string arguments provided by the LLM
            args = json.loads(arguments_json)

            # Execute the mapped function
            result = self.tools[tool_name](**args)
            return str(result)
        except json.JSONDecodeError:
            return "Error: Invalid JSON arguments provided."
        except Exception as e:
            return f"Error executing '{tool_name}': {str(e)}"