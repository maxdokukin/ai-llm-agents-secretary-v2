import os
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Sequence


DB_PATH = Path(__file__).resolve().parent.parent / "../" / "data" / "test" / "synthetic_data.sqlite"

def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def _get_table_schema(conn: sqlite3.Connection, table_name: str) -> List[Dict[str, Any]]:
    rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()

    schema = []
    for row in rows:
        schema_entry = {
            "column_name": row["name"],
            "data_type": row["type"] or "unknown",
        }

        if row["pk"]:
            schema_entry["description"] = "primary key"

        schema.append(schema_entry)

    return schema


def _get_table_index(
    conn: sqlite3.Connection,
    table_name: str,
    columns: Sequence[str],
) -> List[Dict[str, Any]]:
    available_columns = {
        row["name"]
        for row in conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    }

    missing_columns = [col for col in columns if col not in available_columns]
    if missing_columns:
        raise ValueError(
            f"Missing columns in table '{table_name}': {', '.join(missing_columns)}"
        )

    quoted_columns = ", ".join(f'"{col}"' for col in columns)

    order_clause = ""
    if "ranking" in available_columns:
        order_clause = ' ORDER BY "ranking"'
    elif "id" in available_columns:
        order_clause = ' ORDER BY "id"'

    rows = conn.execute(
        f'SELECT {quoted_columns} FROM "{table_name}"{order_clause}'
    ).fetchall()

    return [dict(row) for row in rows]


def fetch_db_index() -> dict:
    """
    Fetches the SQLite schema and specific column indexes for classes,
    projects, works, and educations.

    Returns a dictionary containing 'table_schema' and 'table_index'.
    """
    try:
        db_path = Path(os.environ.get("SQLITE_DB_PATH", DB_PATH)).expanduser().resolve()

        if not db_path.exists():
            return {"error": f"SQLite database not found at: {db_path}"}

        tables_config = {
            "classes": ["slug"],
            "projects": ["slug"],
            "works": ["slug"],
            "educations": ["slug"],
        }

        output = {
            "table_schema": {},
            "table_index": {},
        }

        with _connect(db_path) as conn:
            for table_name, columns in tables_config.items():
                if table_name == "sqlite_sequence":
                    continue

                if not _table_exists(conn, table_name):
                    output["table_schema"][table_name] = {
                        "error": f"Table '{table_name}' not found in database schema."
                    }
                else:
                    try:
                        output["table_schema"][table_name] = _get_table_schema(
                            conn,
                            table_name,
                        )
                    except Exception as e:
                        output["table_schema"][table_name] = {
                            "error": f"Failed to fetch schema: {str(e)}"
                        }

                try:
                    if not _table_exists(conn, table_name):
                        raise ValueError(f"Table '{table_name}' not found in database.")

                    output["table_index"][table_name] = _get_table_index(
                        conn,
                        table_name,
                        columns,
                    )
                except Exception as e:
                    output["table_index"][table_name] = {
                        "error": f"Failed to fetch data: {str(e)}"
                    }

        return output

    except Exception as e:
        return {"error": f"Error executing get_db_index: {str(e)}"}


if __name__ == "__main__":
    print(json.dumps(fetch_db_index(), indent=2))
