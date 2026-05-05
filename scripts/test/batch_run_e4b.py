import csv
import uuid
import time
import asyncio
import httpx
from pathlib import Path
from openai import AsyncOpenAI

# --- Logic Configuration ---
from src.ToolManager.ToolManager import ToolManager
from src.data.supabase import fetch_db_index

CTX_SERVER = "http://localhost:7999/api/context"
LLM_SERVER = "http://localhost:8080/v1"
# LLM_SERVER = "http://10.0.0.43:8080/v1"

def tool_returns_data(t_manager: ToolManager, tool_name: str) -> bool:
    """Returns whether a registered tool is a data-returning tool."""
    if hasattr(t_manager, "tool_returns_data"):
        return bool(t_manager.tool_returns_data.get(tool_name, False))

    for schema in t_manager.get_schemas():
        function_schema = schema.get("function", {})
        if function_schema.get("name") == tool_name:
            return bool(function_schema.get("returns_data", False))

    return False


async def append_tool_result_to_context(
        http: httpx.AsyncClient,
        *,
        session_id: str,
        tool_name: str,
        result: str,
        associated_id: str,
        returns_data: bool,
) -> None:
    """Routes tool output into the correct context bucket."""
    endpoint = "fetched_data" if returns_data else "tool_results"
    payload = {
        "session_id": session_id,
        "tool_name": tool_name,
        "associated_id": associated_id
    }
    if returns_data:
        payload["data"] = result
    else:
        payload["result"] = result

    await http.post(f"{CTX_SERVER}/{endpoint}", json=payload)


async def initialize_session(t_manager: ToolManager) -> str:
    """Bootstraps a fresh session context on the Context Server."""
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    print(f"\n--- INITIALIZING CONTEXT SESSION: {session_id} ---")

    async with httpx.AsyncClient() as http:
        # Use .read() instead of .readlines() to get a single string
        with open("/Users/max/Codebase/github/ai-llm-agents-secretary-v2/llm/prompts/secretary_prompt.txt") as f:
            master_prompt = f.read()

        await http.post(
            f"{CTX_SERVER}/master_prompt",
            json={
                "session_id": session_id,
                "text": master_prompt
            }
        )

        # Data index
        await http.post(f"{CTX_SERVER}/data_index", json={
            "session_id": session_id,
            "index_data": fetch_db_index(),
        })

        # Tool schemas
        schemas = t_manager.get_schemas()
        for schema in schemas:
            await http.post(f"{CTX_SERVER}/tools", json={
                "session_id": session_id,
                "tool_schema": schema
            })

    return session_id


async def process_query(session_id: str, user_text: str, t_manager: ToolManager, llm_client: AsyncOpenAI) -> str:
    """Runs the ReAct loop for a single query and returns the final answer."""
    async with httpx.AsyncClient() as http:
        resp = await http.post(f"{CTX_SERVER}/message", json={
            "session_id": session_id,
            "role": "user",
            "content": user_text
        })
        user_msg_id = resp.json()["id"]

    final_answer = ""

    while True:
        async with httpx.AsyncClient() as http:
            ctx_resp = await http.get(f"{CTX_SERVER}/assemble/{session_id}")
            current_messages = ctx_resp.json()["messages"]

        if current_messages and current_messages[-1].get("role") != "user":
            current_messages.append({
                "role": "user",
                "content": "Action completed. Please evaluate the context and continue."
            })

        schemas = t_manager.get_schemas()
        response = await llm_client.chat.completions.create(
            model="local-model",
            messages=current_messages,
            tools=schemas if schemas else None,
            stream=True
        )

        turn_content = ""
        tool_calls = {}

        async for chunk in response:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                turn_content += delta.content

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index

                    if idx not in tool_calls:
                        tool_calls[idx] = {"id": tc.id, "name": "", "arguments": ""}

                    if tc.function.name:
                        tool_calls[idx]["name"] += tc.function.name

                    if tc.function.arguments:
                        tool_calls[idx]["arguments"] += tc.function.arguments

        assistant_log = turn_content
        if tool_calls:
            for tc in tool_calls.values():
                assistant_log += f"\n[Action: Executed tool '{tc['name']}' with args: {tc['arguments']}]"

        assistant_msg_id = None
        if assistant_log.strip():
            async with httpx.AsyncClient() as http:
                resp = await http.post(f"{CTX_SERVER}/message", json={
                    "session_id": session_id,
                    "role": "assistant",
                    "content": assistant_log.strip()
                })
                assistant_msg_id = resp.json()["id"]

        if not tool_calls:
            final_answer = turn_content
            break

        for tc in tool_calls.values():
            tool_name = tc["name"]
            print(f"  [EXE] Calling {tool_name}...")
            returns_data = tool_returns_data(t_manager, tool_name)

            result = str(t_manager.execute_tool(tool_name, tc["arguments"]))

            async with httpx.AsyncClient() as http:
                await append_tool_result_to_context(
                    http,
                    session_id=session_id,
                    tool_name=tool_name,
                    result=result,
                    associated_id=assistant_msg_id or user_msg_id,
                    returns_data=returns_data,
                )

    return final_answer


async def save_context_png(session_id: str):
    """Triggers the context server to render and save the context map PNG to its default location."""
    print(f"  📸 Generating context map PNG...")
    async with httpx.AsyncClient() as http:
        try:
            resp = await http.post(f"{CTX_SERVER}/save_png", json={
                "session_id": session_id
            })
            resp.raise_for_status()
            data = resp.json()
            print(f"  ✅ PNG saved to: {data.get('filepath')}")
        except Exception as e:
            print(f"  ❌ Failed to save PNG: {e}")


async def run_batch(input_csv_path: str, output_csv_path: str, query_column_name: str = "query"):
    """Reads a CSV, processes each query through the LLM, and writes the results out."""

    # 1. Init Local Tool Discovery & Inference Client
    t_manager = ToolManager(toolbox_dir="../src/ToolManager/toolbox")
    llm_client = AsyncOpenAI(base_url=LLM_SERVER, api_key="sk-local")

    print(f"🛠️  Registered {len(t_manager.get_schemas())} tools.")
    print(f"📂 Opening input file: {input_csv_path}\n")

    with open(input_csv_path, mode='r', encoding='utf-8') as infile, \
            open(output_csv_path, mode='w', encoding='utf-8', newline='') as outfile:

        reader = csv.DictReader(infile)

        if query_column_name not in reader.fieldnames:
            print(f"❌ Error: Could not find column '{query_column_name}' in the CSV.")
            return

        # Prepare output columns
        fieldnames = reader.fieldnames + ["Answer", "Time_Taken_s"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row_index, row in enumerate(reader, start=1):
            query = row.get(query_column_name, "").strip()
            if not query:
                continue

            print(f"\n[{row_index}] Processing Query: {query}")

            session_id = await initialize_session(t_manager)
            start_time = time.time()

            try:
                answer = await process_query(session_id, query, t_manager, llm_client)

                # --- NEW: Save PNG upon successful completion without passing paths ---
                await save_context_png(session_id)

            except Exception as e:
                print(f"❌ Error processing row {row_index}: {e}")
                answer = f"ERROR: {str(e)}"

            time_taken = round(time.time() - start_time, 2)

            # Write to output row
            row["Answer"] = answer
            row["Time_Taken_s"] = time_taken
            writer.writerow(row)

            print(f"✅ Completed in {time_taken} seconds.")

    print(f"\n🎉 Batch run complete! Results saved to {output_csv_path}")


if __name__ == "__main__":
    # Specify your CSV names here
    INPUT_CSV = "/Users/max/Codebase/github/ai-llm-agents-secretary-v2/data/test_queries.csv"
    # OUTPUT_CSV = "/Users/max/Codebase/github/ai-llm-agents-secretary-v2/data/test_queries_results_e4b.csv"
    OUTPUT_CSV = "/Users/max/Codebase/github/ai-llm-agents-secretary-v2/data/test_queries_results_e4b.csv"

    # Specify the name of the column that contains the prompt/query
    QUERY_COLUMN = "query"

    asyncio.run(run_batch(INPUT_CSV, OUTPUT_CSV, QUERY_COLUMN))