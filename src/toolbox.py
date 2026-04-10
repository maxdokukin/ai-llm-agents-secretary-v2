import json
from datetime import datetime


def get_time():
    """Return the current local time."""
    return {
        "iso": datetime.now().isoformat(),
        "readable": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def add(a, b):
    """Add two numbers."""
    return {"result": a + b}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current local time",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number",
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number",
                    },
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    },
]


def execute_tool(name, arguments):
    """
    Execute a tool by name.

    arguments may be a dict or a JSON string.
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}

    if arguments is None:
        arguments = {}

    if name == "get_time":
        return get_time()

    if name == "add":
        return add(
            a=arguments["a"],
            b=arguments["b"],
        )

    raise ValueError(f"Unknown tool: {name}")