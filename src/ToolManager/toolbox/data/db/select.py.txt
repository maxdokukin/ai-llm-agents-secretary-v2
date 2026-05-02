import os
import json
import re
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

tool_schema = {
    "type": "function",
    "function": {
        "description": "Executes a read-only SQL SELECT query against the database.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The raw SQL SELECT query to execute."
                }
            },
            "required": ["query"]
        }
    }
}


def _is_query_safe(query: str) -> bool:
    """
    Validates that the query is a read-only SELECT statement and contains no mutations.
    """
    q_upper = query.strip().upper()

    # Must start with SELECT or WITH (for Common Table Expressions)
    if not (q_upper.startswith("SELECT") or q_upper.startswith("WITH")):
        return False

    # List of forbidden SQL mutation / DDL keywords
    forbidden_keywords = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
        "CREATE", "GRANT", "REVOKE", "MERGE", "EXEC", "EXECUTE",
        "CALL", "REPLACE", "UPSERT"
    ]

    # Use word boundaries (\b) so we don't accidentally block a column named "update_date"
    pattern = re.compile(r'\b(' + '|'.join(forbidden_keywords) + r')\b')

    if pattern.search(q_upper):
        return False

    return True


def execute(query: str) -> str:
    # 1. Enforce strict read-only safety checks
    if not _is_query_safe(query):
        return (
            "Security Error: Query blocked. Only safe, read-only SELECT queries are allowed. "
            "No data mutation or schema modification commands are permitted."
        )

    try:
        load_dotenv()

        # You will need your Supabase Postgres Connection String (Transaction mode)
        # e.g., postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres
        db_url = os.environ.get("SUPABASE_URL")

        if not db_url:
            return "Error: SUPABASE_URL is missing from environment variables."

        # 2. Connect to the database
        # RealDictCursor ensures the rows come back as standard Python dictionaries
        conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)

        # Enforce read-only mode at the connection level as a secondary safeguard
        conn.set_session(readonly=True)
        cursor = conn.cursor()

        # 3. Execute the query
        cursor.execute(query)
        records = cursor.fetchall()

        # Clean up connections
        cursor.close()
        conn.close()

        # 4. Return formatted JSON
        return json.dumps(records, default=str)  # default=str handles datetimes/UUIDs

    except psycopg2.Error as db_err:
        return f"Database Execution Error: {str(db_err)}"
    except Exception as e:
        return f"Error executing select tool: {str(e)}"