#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CACHE_DIR="${CACHE_DIR:-$SCRIPT_DIR/cache}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
SERVER_SCRIPT="${SERVER_SCRIPT:-$ROOT_DIR/src/LLMServer.py}"
PID_FILE="${PID_FILE:-$CACHE_DIR/.llmserver.pid}"
LOG_FILE="${LOG_FILE:-$CACHE_DIR/.llmserver.log}"

MODEL="${MODEL:-1.7b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

mkdir -p "$CACHE_DIR"
mkdir -p "$CACHE_DIR/huggingface"
mkdir -p "$CACHE_DIR/xdg"

export HF_HOME="${HF_HOME:-$CACHE_DIR/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$CACHE_DIR/huggingface/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CACHE_DIR/huggingface/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_DIR/xdg}"

if [ -f "$PID_FILE" ]; then
    OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "LLMServer already running with PID $OLD_PID"
        echo "Host: $HOST"
        echo "Port: $PORT"
        echo "Cache: $CACHE_DIR"
        exit 0
    else
        rm -f "$PID_FILE"
    fi
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [ ! -f "$SERVER_SCRIPT" ]; then
    echo "Server script not found: $SERVER_SCRIPT" >&2
    exit 1
fi

echo "Starting LLMServer..."
echo "Python: $PYTHON_BIN"
echo "Script: $SERVER_SCRIPT"
echo "Model : $MODEL"
echo "Host  : $HOST"
echo "Port  : $PORT"
echo "Log   : $LOG_FILE"
echo "Cache : $CACHE_DIR"

nohup "$PYTHON_BIN" "$SERVER_SCRIPT" \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    >> "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

sleep 2

if kill -0 "$PID" 2>/dev/null; then
    echo "LLMServer started with PID $PID"
else
    echo "LLMServer failed to start. Check log: $LOG_FILE" >&2
    rm -f "$PID_FILE"
    exit 1
fi