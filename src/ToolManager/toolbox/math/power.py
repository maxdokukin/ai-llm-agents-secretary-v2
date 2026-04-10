tool_schema = {
    "type": "function",
    "function": {
        "description": "Calculates the result of a base number raised to an exponent.",
        "parameters": {
            "type": "object",
            "properties": {
                "base": {"type": "number", "description": "The base number."},
                "exponent": {"type": "number", "description": "The power to raise the base to."}
            },
            "required": ["base", "exponent"]
        }
    }
}

def execute(base: float, exponent: float) -> float:
    return base ** exponent