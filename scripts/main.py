from openai import OpenAI
from src.ToolManager.ToolManager import ToolManager

# 1. Initialize Tool Manager
manager = ToolManager()

# 2. Setup OpenAI Client pointing to local llama.cpp server
# (llama.cpp default port is usually 8080)
client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-local")

# 3. Request completion with tools
response = client.chat.completions.create(
    model="local-model",  # llama.cpp ignores this field, but OpenAI requires it
    messages=[
        {"role": "user", "content": "fetch available educations data"}
    ],
    tools=manager.get_schemas(),
    tool_choice="auto"
)

message = response.choices[0].message

# 4. Handle the Tool Call
if message.tool_calls:
    for tool_call in message.tool_calls:
        print(f"Model wants to call: {tool_call.function.name}")

        # Execute dynamically
        result = manager.execute_tool(
            tool_name=tool_call.function.name,
            arguments_json=tool_call.function.arguments
        )

        print(f"Tool Result: {result}")