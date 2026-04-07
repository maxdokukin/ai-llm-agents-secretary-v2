#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="${CACHE_DIR:-$SCRIPT_DIR/cache}"
PID_FILE="${PID_FILE:-$CACHE_DIR/.llmserver.pid}"

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found: $PID_FILE"
    exit 0
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"

if [ -z "${PID:-}" ]; then
    echo "PID file is empty. Removing it."
    rm -f "$PID_FILE"
    exit 0
fi

kill_descendants() {
    local parent="$1"
    local children

    children="$(pgrep -P "$parent" || true)"
    if [ -n "$children" ]; then
        for child in $children; do
            kill_descendants "$child"
        done
    fi

    if kill -0 "$parent" 2>/dev/null; then
        kill "$parent" 2>/dev/null || true
    fi
}

force_kill_descendants() {
    local parent="$1"
    local children

    children="$(pgrep -P "$parent" || true)"
    if [ -n "$children" ]; then
        for child in $children; do
            force_kill_descendants "$child"
        done
    fi

    if kill -0 "$parent" 2>/dev/null; then
        kill -9 "$parent" 2>/dev/null || true
    fi
}

echo "Stopping LLMServer process tree rooted at PID $PID ..."
kill_descendants "$PID"

for _ in $(seq 1 10); do
    if kill -0 "$PID" 2>/dev/null; then
        sleep 1
    else
        break
    fi
done

if kill -0 "$PID" 2>/dev/null; then
    echo "Process tree still running, force killing..."
    force_kill_descendants "$PID"
fi

rm -f "$PID_FILE"
echo "LLMServer stopped."