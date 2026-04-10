import sys
from openai import OpenAI
from src.ToolManager.ToolManager import ToolManager

# 1. Initialize Tool Manager
manager = ToolManager()

# 2. Setup OpenAI Client pointing to local llama.cpp server
client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-local")

messages = [
    {"role": "user", "content": "compare data you have in all tables you have db access to"}
]

print("Model: ", end="", flush=True)

# 3. Request completion with tools and stream=True
response = client.chat.completions.create(
    model="local-model",
    messages=messages,
    tools=manager.get_schemas(),
    tool_choice="auto",
    stream=True
)

# Accumulator dictionaries for the stream
assembled_tool_calls = {}

# 4. Iterate through the stream chunks
for chunk in response:
    delta = chunk.choices[0].delta

    # A. Stream Content/Reasoning directly to the console
    # (If your model uses <think> tags, they will stream out here naturally)
    if delta.content:
        sys.stdout.write(delta.content)
        sys.stdout.flush()

    # (Optional safeguard): Check if server uses separate reasoning field (e.g., DeepSeek R1 setups)
    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
        sys.stdout.write(delta.reasoning_content)
        sys.stdout.flush()

    # B. Accumulate fragmented Tool Calls
    if delta.tool_calls:
        for tc_chunk in delta.tool_calls:
            # Tools stream with an index to keep track of multiple parallel calls
            idx = tc_chunk.index

            # Initialize the dictionary for this index if we haven't seen it yet
            if idx not in assembled_tool_calls:
                assembled_tool_calls[idx] = {"name": "", "arguments": ""}

            # Append the name chunks
            if tc_chunk.function.name:
                assembled_tool_calls[idx]["name"] += tc_chunk.function.name

            # Append the JSON argument chunks
            if tc_chunk.function.arguments:
                assembled_tool_calls[idx]["arguments"] += tc_chunk.function.arguments

print("\n")  # Clean line break after the stream finishes

# 5. Handle the Assembled Tool Calls
if assembled_tool_calls:
    print("--- Executing Tools ---")
    for idx, tool_call in assembled_tool_calls.items():
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]

        print(f"Executing: {tool_name}")

        # Execute dynamically
        result = manager.execute_tool(
            tool_name=tool_name,
            arguments_json=tool_args
        )

        print(f"Tool Result: {result}\n")