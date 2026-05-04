import os
import json
import urllib.request
from dotenv import load_dotenv


def fetch_db_index() -> dict:
    """
    Fetches the schema and specific column indexes for classes, projects, works, and educations.
    Returns a dictionary containing 'table_schema' and 'table_index'.
    """
    try:
        load_dotenv()

        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")

        if not url or not key:
            return {"error": "SUPABASE_URL or SUPABASE_KEY is missing from environment variables."}

        base_url = url.rstrip('/')
        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}"
        }

        # 1. Fetch the OpenAPI spec for the schema definitions
        req_url = f"{base_url}/rest/v1/"
        req_spec = urllib.request.Request(req_url, headers=headers)

        with urllib.request.urlopen(req_spec) as response:
            openapi_spec = json.loads(response.read().decode())

        definitions = openapi_spec.get("definitions", {})

        # 2. Define the tables and the exact columns needed for the index
        tables_config = {
            "classes": ["name"],
            "projects": ["title", "short_description", "skills", "software_skills", "data_skills", "frameworks", "ranking", "slug"],
            "works": ["title", "employer", "short_description", "skills", "frameworks", "ranking", "slug"],
            "educations": ["location", "title", "subtitle"]
        }
        # tables_config = {
        #     "classes": ["slug"],
        #     "projects": ["title", "short_description", "skills", "ranking", "slug"],
        #     "works": ["title", "employer", "short_description", "skills", "frameworks", "ranking", "slug"],
        #     "educations": ["location", "title", "subtitle"]
        # }

        output = {
            "table_schema": {},
            "table_index": {}
        }

        # 3. Iterate through our required tables to populate schema and index
        for table_name, columns in tables_config.items():

            # --- Populate Table Schema ---
            if table_name in definitions:
                properties = definitions[table_name].get("properties", {})
                schema = []
                for col_name, col_props in properties.items():
                    data_type = col_props.get("format", col_props.get("type", "unknown"))
                    description = col_props.get("description", "")

                    schema_entry = {"column_name": col_name, "data_type": data_type}
                    if description:
                        schema_entry["description"] = description
                    schema.append(schema_entry)

                output["table_schema"][table_name] = schema
            else:
                output["table_schema"][table_name] = {"error": f"Table '{table_name}' not found in database schema."}

            # --- Populate Table Index (Data) ---
            # Construct a comma-separated list of columns for the Supabase 'select' parameter
            select_query = ",".join(columns)
            req_index = urllib.request.Request(
                f"{base_url}/rest/v1/{table_name}?select={select_query}",
                headers=headers
            )

            try:
                with urllib.request.urlopen(req_index) as response:
                    index_data = json.loads(response.read().decode())
                output["table_index"][table_name] = index_data
            except Exception as e:
                output["table_index"][table_name] = {"error": f"Failed to fetch data: {str(e)}"}

        return output

    except Exception as e:
        return {"error": f"Error executing get_db_index: {str(e)}"}

# Optional: To test the drop-in file