import subprocess
import shlex
import os

tool_schema = {
    "type": "function",
    "function": {
        "description": "Executes allowed shell commands (ls, cd, pwd) on the local system and returns the standard output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string",
                            "description": "The command to execute. Only 'ls', 'cd', and 'pwd' are permitted."}
            },
            "required": ["command"]
        }
    }
}


def execute(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError as e:
        return f"Error parsing command: {str(e)}"

    if not parts:
        return "Error: Empty command."

    base_cmd = parts[0]
    allowed_commands = ["ls", "cd", "pwd"]

    if base_cmd not in allowed_commands:
        return f"Security Error: Command '{base_cmd}' is blocked. Only {allowed_commands} are permitted."

    if base_cmd == "cd":
        target_dir = parts[1] if len(parts) > 1 else os.path.expanduser("~")
        try:
            os.chdir(target_dir)
            return f"Success: Changed working directory to {os.getcwd()}"
        except Exception as e:
            return f"Error changing directory: {str(e)}"

    try:
        result = subprocess.run(
            parts,
            shell=False,
            capture_output=True,
            text=True,
            timeout=10
        )
        output = result.stdout if result.stdout else result.stderr
        return output.strip() if output else "Command executed successfully with no output."
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 10 seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"