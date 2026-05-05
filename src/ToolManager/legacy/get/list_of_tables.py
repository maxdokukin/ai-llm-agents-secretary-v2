import os
import json
import urllib.request
from dotenv import load_dotenv

tool_schema = {
    "type": "function",
    "function": {
        "description": "Retrieves a list of all available user tables in the database.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}


def execute() -> str:
    try:
        load_dotenv()

        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")

        if not url or not key:
            return "Error: SUPABASE_URL or SUPABASE_KEY is missing from environment variables."

        # Supabase exposes its OpenAPI spec at the root of the REST API
        req_url = f"{url.rstrip('/')}/rest/v1/"

        req = urllib.request.Request(req_url, headers={
            "apikey": key,
            "Authorization": f"Bearer {key}"
        })

        with urllib.request.urlopen(req) as response:
            openapi_spec = json.loads(response.read().decode())

        # PostgREST uses Swagger 2.0; the tables and views are stored inside 'definitions'
        definitions = openapi_spec.get("definitions", {})

        if not definitions:
            return "No tables found or OpenAPI spec is empty."

        # Extract table names (ignoring potential standard PostgREST utility objects)
        table_names = list(definitions.keys())

        return json.dumps(table_names)

    except Exception as e:
        return f"Error fetching list of tables via OpenAPI spec: {str(e)}"