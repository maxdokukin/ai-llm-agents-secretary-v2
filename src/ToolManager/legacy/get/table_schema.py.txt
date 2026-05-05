import os
import json
import urllib.request
from dotenv import load_dotenv

tool_schema = {
    "type": "function",
    "function": {
        "description": "Retrieves the schema (list of columns and their data types) for a specified database table.",
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "The name of the database table to inspect."
                }
            },
            "required": ["table_name"]
        }
    }
}


def execute(table_name: str) -> str:
    try:
        load_dotenv()

        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")

        if not url or not key:
            return "Error: SUPABASE_URL or SUPABASE_KEY is missing from environment variables."

        # Request the OpenAPI spec
        req_url = f"{url.rstrip('/')}/rest/v1/"

        req = urllib.request.Request(req_url, headers={
            "apikey": key,
            "Authorization": f"Bearer {key}"
        })

        with urllib.request.urlopen(req) as response:
            openapi_spec = json.loads(response.read().decode())

        definitions = openapi_spec.get("definitions", {})

        # Ensure the requested table exists in the definitions
        if table_name not in definitions:
            return f"Error: Table '{table_name}' not found in the database schema."

        # Extract the column properties
        properties = definitions[table_name].get("properties", {})

        # Format the properties into a clean list of dictionaries for the LLM
        schema = []
        for col_name, col_props in properties.items():
            # Use format if available (e.g., 'uuid', 'timestamp'), otherwise fallback to base type
            data_type = col_props.get("format", col_props.get("type", "unknown"))
            description = col_props.get("description", "")

            schema_entry = {"column_name": col_name, "data_type": data_type}
            if description:
                schema_entry["description"] = description

            schema.append(schema_entry)

        return json.dumps(schema)

    except Exception as e:
        return f"Error fetching table schema via OpenAPI spec: {str(e)}"