tool_schema = {
    "type": "function",
    "function": {
        "name": "all_table_slugs",
        "description": "Fetches a complete index of all available slugs from a specified database table.",
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "The name of the database table to query."
                }
            },
            "required": ["table_name"]
        }
    }
}

import os
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv

def execute(table_name: str) -> str:
    """
    Fetches an index of all slugs from a specified table.
    Returns a JSON string array of slugs, or an error string if failed.
    """
    try:
        load_dotenv()

        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")

        if not url or not key:
            return "Error: SUPABASE_URL or SUPABASE_KEY is missing from environment variables."

        base_url = url.rstrip('/')

        # We use Supabase's select filter to only pull the slug column
        req_url = f"{base_url}/rest/v1/{table_name}?select=slug"

        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json"
        }

        req = urllib.request.Request(req_url, headers=headers)

        with urllib.request.urlopen(req) as response:
            raw_data = json.loads(response.read().decode())

        # Extract just the slug strings from the array of dictionaries
        # e.g., [{"slug": "foo"}, {"slug": "bar"}] -> ["foo", "bar"]
        slug_list = [item.get("slug") for item in raw_data if "slug" in item]

        return json.dumps(slug_list)

    except urllib.error.HTTPError as e:
        return f"Error: HTTP Error fetching slug index: {e.code} - {e.reason}"
    except Exception as e:
        return f"Error executing index fetch: {str(e)}"