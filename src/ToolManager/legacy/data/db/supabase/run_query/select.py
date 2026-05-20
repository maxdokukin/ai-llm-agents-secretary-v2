import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from supabase import create_client, Client


tool_schema = {
    "type": "function",
    "function": {
        "name": "execute",
        "description": (
            "Executes a restricted, read-only PostgreSQL-style SELECT query "
            "through the Supabase client. Supports simple SELECT/FROM/WHERE/"
            "ORDER BY/LIMIT queries only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A restricted SELECT query. Supported shape: "
                        "SELECT columns FROM table "
                        "[WHERE col op value [AND col op value ...]] "
                        "[ORDER BY col ASC|DESC] [LIMIT n]"
                    ),
                }
            },
            "required": ["query"],
        },
    },
}


_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
_TABLE_IDENTIFIER = rf"(?:{_IDENTIFIER}\.)?{_IDENTIFIER}"

_FORBIDDEN_KEYWORDS = [
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "GRANT",
    "REVOKE",
    "CREATE",
    "REPLACE",
    "MERGE",
    "EXEC",
    "EXECUTE",
    "CALL",
    "COMMIT",
    "ROLLBACK",
    "UPSERT",
    "COPY",
    "VACUUM",
    "ANALYZE",
    "LOCK",
    "DO",
    "BEGIN",
    "END",
]


_SELECT_RE = re.compile(
    rf"""
    ^\s*
    SELECT\s+(?P<columns>.+?)
    \s+FROM\s+(?P<table>{_TABLE_IDENTIFIER})
    (?:\s+WHERE\s+(?P<where>.*?)(?=\s+ORDER\s+BY|\s+LIMIT|\s*;?\s*$))?
    (?:\s+ORDER\s+BY\s+(?P<order_col>{_IDENTIFIER})(?:\s+(?P<order_dir>ASC|DESC))?)?
    (?:\s+LIMIT\s+(?P<limit>\d+))?
    \s*;?\s*$
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


_CONDITION_RE = re.compile(
    rf"""
    ^\s*
    (?P<column>{_IDENTIFIER})
    \s*
    (?P<operator>=|!=|<>|>=|<=|>|<|ILIKE|LIKE|IS|IN)
    \s*
    (?P<value>.+?)
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


def _get_supabase_client() -> Client:
    load_dotenv()

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = (
        os.environ.get("SUPABASE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    )

    if not supabase_url:
        raise ValueError("SUPABASE_URL is missing from environment variables.")

    if not supabase_key:
        raise ValueError(
            "SUPABASE_KEY is missing from environment variables. "
            "You may also use SUPABASE_ANON_KEY or SUPABASE_SERVICE_ROLE_KEY."
        )

    if not supabase_url.startswith(("http://", "https://")):
        raise ValueError("SUPABASE_URL must be your Supabase HTTPS project URL.")

    return create_client(supabase_url, supabase_key)


def _reject_comments(query: str) -> bool:
    return "--" in query or "/*" in query or "*/" in query


def _has_stacked_queries(query: str) -> bool:
    stripped = query.strip()
    return ";" in stripped.rstrip(";")


def _contains_forbidden_keyword(query: str) -> bool:
    pattern = re.compile(
        r"\b(?:" + "|".join(map(re.escape, _FORBIDDEN_KEYWORDS)) + r")\b",
        re.IGNORECASE,
    )
    return bool(pattern.search(query))


def _is_query_safe(query: str) -> bool:
    clean_query = query.strip()

    if not clean_query:
        return False

    if not clean_query.upper().startswith("SELECT"):
        return False

    if _reject_comments(clean_query):
        return False

    if _has_stacked_queries(clean_query):
        return False

    if _contains_forbidden_keyword(clean_query):
        return False

    return True


def _parse_columns(columns: str) -> str:
    columns = columns.strip()

    if columns == "*":
        return "*"

    parts = [part.strip() for part in columns.split(",")]

    if not parts or any(not part for part in parts):
        raise ValueError("Invalid SELECT column list.")

    for part in parts:
        if not re.fullmatch(_IDENTIFIER, part):
            raise ValueError(
                "Only simple column names are supported in SELECT. "
                "Aliases, functions, casts, and expressions are not allowed."
            )

    return ",".join(parts)


def _split_outside_quotes_and_parens(value: str, separator: str) -> List[str]:
    items = []
    current = []
    quote: Optional[str] = None
    paren_depth = 0
    i = 0

    while i < len(value):
        ch = value[i]

        if quote:
            current.append(ch)

            if ch == quote:
                if i + 1 < len(value) and value[i + 1] == quote:
                    current.append(value[i + 1])
                    i += 1
                else:
                    quote = None

        else:
            if ch in ("'", '"'):
                quote = ch
                current.append(ch)
            elif ch == "(":
                paren_depth += 1
                current.append(ch)
            elif ch == ")":
                paren_depth -= 1
                if paren_depth < 0:
                    raise ValueError("Unbalanced parentheses.")
                current.append(ch)
            elif (
                paren_depth == 0
                and value[i : i + len(separator)].upper() == separator.upper()
            ):
                items.append("".join(current).strip())
                current = []
                i += len(separator) - 1
            else:
                current.append(ch)

        i += 1

    if quote:
        raise ValueError("Unclosed quoted string.")

    if paren_depth != 0:
        raise ValueError("Unbalanced parentheses.")

    items.append("".join(current).strip())
    return items


def _split_where_conditions(where: str) -> List[str]:
    if not where:
        return []

    return _split_outside_quotes_and_parens(where, " AND ")


def _parse_literal(raw: str) -> Any:
    value = raw.strip()

    if not value:
        raise ValueError("Empty value in WHERE condition.")

    upper = value.upper()

    if upper == "NULL":
        return None

    if upper == "TRUE":
        return True

    if upper == "FALSE":
        return False

    if (
        len(value) >= 2
        and value[0] == "'"
        and value[-1] == "'"
    ):
        return value[1:-1].replace("''", "'")

    if (
        len(value) >= 2
        and value[0] == '"'
        and value[-1] == '"'
    ):
        return value[1:-1].replace('""', '"')

    if re.fullmatch(r"-?\d+", value):
        return int(value)

    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)

    raise ValueError(
        f"Unsupported literal value: {raw!r}. "
        "Use quoted strings, numbers, true, false, or null."
    )


def _parse_in_list(raw: str) -> List[Any]:
    value = raw.strip()

    if not (value.startswith("(") and value.endswith(")")):
        raise ValueError("IN requires a parenthesized value list.")

    inner = value[1:-1].strip()

    if not inner:
        raise ValueError("IN list cannot be empty.")

    parts = _split_outside_quotes_and_parens(inner, ",")
    return [_parse_literal(part) for part in parts]


def _parse_query(query: str) -> Dict[str, Any]:
    match = _SELECT_RE.match(query)

    if not match:
        raise ValueError(
            "Unsupported query shape. Use: "
            "SELECT columns FROM table "
            "[WHERE col op value [AND col op value ...]] "
            "[ORDER BY col ASC|DESC] [LIMIT n]"
        )

    columns = _parse_columns(match.group("columns"))
    table = match.group("table")
    where = match.group("where")
    order_col = match.group("order_col")
    order_dir = match.group("order_dir") or "ASC"
    limit_raw = match.group("limit")

    conditions = []

    if where:
        for condition_text in _split_where_conditions(where):
            condition_match = _CONDITION_RE.match(condition_text)

            if not condition_match:
                raise ValueError(f"Unsupported WHERE condition: {condition_text!r}")

            column = condition_match.group("column")
            operator = condition_match.group("operator").upper()
            raw_value = condition_match.group("value")

            if operator == "IN":
                value = _parse_in_list(raw_value)
            else:
                value = _parse_literal(raw_value)

            conditions.append(
                {
                    "column": column,
                    "operator": operator,
                    "value": value,
                }
            )

    limit = None
    if limit_raw is not None:
        limit = int(limit_raw)
        if limit < 1 or limit > 1000:
            raise ValueError("LIMIT must be between 1 and 1000.")

    return {
        "columns": columns,
        "table": table,
        "conditions": conditions,
        "order_col": order_col,
        "order_dir": order_dir.upper(),
        "limit": limit,
    }


def _apply_condition(builder: Any, condition: Dict[str, Any]) -> Any:
    column = condition["column"]
    operator = condition["operator"]
    value = condition["value"]

    if operator == "=":
        return builder.eq(column, value)

    if operator in ("!=", "<>"):
        return builder.neq(column, value)

    if operator == ">":
        return builder.gt(column, value)

    if operator == ">=":
        return builder.gte(column, value)

    if operator == "<":
        return builder.lt(column, value)

    if operator == "<=":
        return builder.lte(column, value)

    if operator == "LIKE":
        return builder.like(column, value)

    if operator == "ILIKE":
        return builder.ilike(column, value)

    if operator == "IS":
        if value is None:
            return builder.is_(column, "null")
        if value is True:
            return builder.is_(column, "true")
        if value is False:
            return builder.is_(column, "false")
        raise ValueError("IS only supports NULL, TRUE, or FALSE.")

    if operator == "IN":
        return builder.in_(column, value)

    raise ValueError(f"Unsupported operator: {operator}")


def execute(query: str) -> str:
    """
    Executes a restricted, read-only Supabase SELECT query.

    Supported examples:
        SELECT * FROM todos;
        SELECT id, name FROM todos LIMIT 10;
        SELECT id, name FROM todos WHERE completed = false;
        SELECT id, name FROM todos WHERE user_id = 'abc' AND completed = false ORDER BY created_at DESC LIMIT 20;

    Returns:
        JSON string of rows, or an error string.
    """
    try:
        if not _is_query_safe(query):
            return (
                "Error: Security violation. Only standalone SELECT queries are "
                "permitted. Comments, stacked queries, and modification keywords "
                "are blocked."
            )

        parsed = _parse_query(query)
        supabase = _get_supabase_client()

        table_name = parsed["table"]

        if "." in table_name:
            schema_name, plain_table_name = table_name.split(".", 1)
            builder = supabase.schema(schema_name).table(plain_table_name)
        else:
            builder = supabase.table(table_name)

        request = builder.select(parsed["columns"])

        for condition in parsed["conditions"]:
            request = _apply_condition(request, condition)

        if parsed["order_col"]:
            request = request.order(
                parsed["order_col"],
                desc=parsed["order_dir"] == "DESC",
            )

        if parsed["limit"] is not None:
            request = request.limit(parsed["limit"])

        response = request.execute()

        return json.dumps(response.data, default=str)

    except Exception as e:
        return f"Error executing query: {str(e)}"