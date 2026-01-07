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

poetry run uvicorn services.api.main:app --reload --host "${HOST}" --port "${PORT}"
