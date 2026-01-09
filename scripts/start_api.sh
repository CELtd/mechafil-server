#!/usr/bin/env bash
set -euo pipefail

: "${USE_SHARED_CACHE:=true}"
: "${SHARED_CACHE_DIR:=./shared-cache}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"

export USE_SHARED_CACHE
export SHARED_CACHE_DIR
export HOST
export PORT

if command -v lsof >/dev/null 2>&1; then
  if lsof -iTCP:"${PORT}" -sTCP:LISTEN -n -P >/dev/null 2>&1; then
    echo "Port ${PORT} is already in use. Set PORT to a free port and retry." >&2
    exit 1
  fi
fi

poetry run uvicorn services.api.main:app --reload --host "${HOST}" --port "${PORT}"
