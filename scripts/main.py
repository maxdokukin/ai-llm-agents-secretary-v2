import sys
import uuid
from openai import OpenAI
from src.ToolManager.ToolManager import ToolManager

# 1. Initialize Tool Manager and OpenAI Client
manager = ToolManager()
client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-local")

# 2. Maintain Conversation History
messages = [
    {"role": "user", "content": "compare data you have in all tables you have db access to"}
]

print("Starting Agent Loop...\n")

# 3. The Continuous Agent Loop
while True:
    print("\n[Model Responding...] ", end="", flush=True)

    import json

    # 1. Prepare your request arguments
    request_payload = {
        "model": "local-model",
        "messages": messages,
        "tools": manager.get_schemas(),
        "tool_choice": "auto",
        "stream": True
    }

    # 2. Print the raw JSON payload
    print("--- RAW REQUEST PAYLOAD ---")
    print(json.dumps(request_payload, indent=2))
    print("---------------------------\n")

    # 3. Send the request
    response = client.chat.completions.create(**request_payload)

    # Accumulators for this specific turn
    turn_content = ""
    assembled_tool_calls = {}

    # 4. Stream and Accumulate
    for chunk in response:
        # Check if choices array exists and is not empty (sometimes the final chunk is empty)
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        # A. Stream Reasoning Tokens (If model uses a dedicated reasoning field)
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
            sys.stdout.write(delta.reasoning_content)
            sys.stdout.flush()

        # B. Stream Standard Content (The actual answer, or embedded <think> tags)
        if hasattr(delta, 'content') and delta.content is not None:
            sys.stdout.write(delta.content)
            sys.stdout.flush()
            turn_content += delta.content

        # C. Accumulate Fragmented Tool Calls
        if hasattr(delta, 'tool_calls') and delta.tool_calls:
            for tc_chunk in delta.tool_calls:
                idx = tc_chunk.index

                # Initialize with an ID (llama.cpp sometimes misses streaming the ID, so we generate a fallback)
                if idx not in assembled_tool_calls:
                    tc_id = tc_chunk.id if tc_chunk.id else f"call_{uuid.uuid4().hex[:8]}"
                    assembled_tool_calls[idx] = {"id": tc_id, "name": "", "arguments": ""}

                if tc_chunk.function.name:
                    assembled_tool_calls[idx]["name"] += tc_chunk.function.name

                if tc_chunk.function.arguments:
                    assembled_tool_calls[idx]["arguments"] += tc_chunk.function.arguments

    print()  # Line break after stream finishes

    # 5. Format the Assistant's Message for History
    assistant_message = {
        "role": "assistant",
        "content": turn_content if turn_content else None
    }

    if not assembled_tool_calls:
        # Loop Exit Condition: The model didn't call any tools. It is done.
        messages.append(assistant_message)
        print("\n[Agent Finished]")
        break

    # If tools were called, attach them to the assistant message
    formatted_tool_calls = []
    for idx, tc in assembled_tool_calls.items():
        formatted_tool_calls.append({
            "id": tc["id"],
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": tc["arguments"]
            }
        })

    assistant_message["tool_calls"] = formatted_tool_calls
    messages.append(assistant_message)

    # 6. Execute Tools and Append Results to History
    print("\n--- Executing Tools ---")
    for tc in formatted_tool_calls:
        tool_name = tc["function"]["name"]
        tool_args = tc["function"]["arguments"]
        tool_id = tc["id"]

        print(f"-> Running: {tool_name}")

        # Execute dynamically
        result = manager.execute_tool(
            tool_name=tool_name,
            arguments_json=tool_args
        )

        # Truncate print output if it's massive, just to keep the terminal clean
        print(f"-> Result snippet: {result[:100]}...\n")

        # Append the tool's result to the conversation history
        messages.append({
            "role": "tool",
            "tool_call_id": tool_id,
            "content": result
        })

    # After appending the tool results, the 'while True' loop restarts,
    # sending the updated history back to the model!