import os
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv

tool_schema = {
    "type": "function",
    "function": {
        "description": "Fetches a single row from a specified database table using the slug as the unique identifier.",
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "The name of the database table to query."
                },
                "slug": {
                    "type": "string",
                    "description": "The slug identifier for the specific row to fetch."
                }
            },
            "required": ["table_name", "slug"]
        }
    }
}


def execute(table_name: str, slug: str) -> str:
    """
    Fetches a single row from a specified table using the slug as the identifier.
    Returns the row as a JSON string, or an error string if not found/failed.
    """
    try:
        load_dotenv()

        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")

        if not url or not key:
            return "Error: SUPABASE_URL or SUPABASE_KEY is missing from environment variables."

        base_url = url.rstrip('/')

        # We use Supabase's eq. filter to match the slug
        # We also set the Accept header to return a single JSON object rather than an array
        req_url = f"{base_url}/rest/v1/{table_name}?slug=eq.{slug}"

        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/vnd.pgrst.object+json"
        }

        req = urllib.request.Request(req_url, headers=headers)

        with urllib.request.urlopen(req) as response:
            row_data = json.loads(response.read().decode())

        return json.dumps(row_data)

    except urllib.error.HTTPError as e:
        # Supabase returns a 406 Not Acceptable if the Accept header is set to object+json
        # but 0 rows are found. We can catch this specifically to return a cleaner error.
        if e.code == 406:
            return f"Error: Entry with slug '{slug}' not found in table '{table_name}'."
        return f"Error: HTTP Error fetching row: {e.code} - {e.reason}"
    except Exception as e:
        return f"Error executing row fetch: {str(e)}"