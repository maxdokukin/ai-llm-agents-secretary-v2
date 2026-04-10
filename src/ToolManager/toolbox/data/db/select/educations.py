import os
import json
from supabase import create_client, Client
from dotenv import load_dotenv

tool_schema = {
    "type": "function",
    "function": {
        "name": "data_db_select_educations",
        "description": "Fetches education records from the Supabase database.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "The maximum number of records to return. Defaults to 10."
                }
            },
            "required": []
        }
    }
}


def execute(limit: int = 10) -> str:
    try:
        load_dotenv()

        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")

        if not url or not key:
            return "Error: SUPABASE_URL or SUPABASE_KEY is missing from environment variables."

        supabase: Client = create_client(url, key)

        # Execute the query with the optional limit
        response = supabase.table('educations').select("*").limit(limit).execute()
        educations = response.data

        # Return the data as a JSON string for the LLM to process
        return json.dumps(educations)

    except Exception as e:
        return f"Error fetching education data: {str(e)}"