#!/bin/bash
set -e

CONFIG=${1:-configs/baseline.yaml} # add "-" to say if empty, use configs/baseline.yaml
# 1 means it always takes the first argument as CONFIG. Fundamentally flawed bcuz if i want to only pass flag --lr 0.001, --lr becomes config
shift || true

if command -v python.exe >/dev/null 2>&1; then
    PYTHON=python.exe
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
else
    echo "Error: Python interpreter not found." >&2
    exit 1
fi

exec "$PYTHON" src/main.py --config "$CONFIG" "$@"
