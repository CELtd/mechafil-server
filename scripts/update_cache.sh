#!/usr/bin/env bash
set -euo pipefail

: "${USE_SHARED_CACHE:=true}"
: "${SHARED_CACHE_DIR:=./shared-cache}"

export USE_SHARED_CACHE
export SHARED_CACHE_DIR

poetry run python -m services.cache_updater.main --once
