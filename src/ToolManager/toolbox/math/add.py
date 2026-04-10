tool_schema = {
    "type": "function",
    "function": {
        "name": "math_add",
        "description": "Adds two numbers together.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "The first number."},
                "b": {"type": "number", "description": "The second number."}
            },
            "required": ["a", "b"]
        }
    }
}

def execute(a: float, b: float) -> float:
    return a + b