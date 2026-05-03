import os
import json
import urllib.request
from dotenv import load_dotenv


def row(table_name: str, slug: str) -> dict:
    """
    Fetches a single row from a specified table using the slug as the identifier.
    Returns the row as a dictionary, or an error dictionary if not found/failed.
    """
    try:
        load_dotenv()

        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")

        if not url or not key:
            return {"error": "SUPABASE_URL or SUPABASE_KEY is missing from environment variables."}

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

        return row_data

    except urllib.error.HTTPError as e:
        # Supabase returns a 406 Not Acceptable if the Accept header is set to object+json 
        # but 0 rows are found. We can catch this specifically to return a cleaner error.
        if e.code == 406:
            return {"error": f"Entry with slug '{slug}' not found in table '{table_name}'."}
        return {"error": f"HTTP Error fetching row: {e.code} - {e.reason}"}
    except Exception as e:
        return {"error": f"Error executing row: {str(e)}"}


# Optional: To test the drop-in file locally
if __name__ == "__main__":
    # Example usage:
    # result = row("projects", "my-awesome-project")
    # print(json.dumps(result, indent=2))
    pass