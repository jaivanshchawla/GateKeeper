#!/bin/sh
# Render sets the PORT environment variable (default: 10000).
# Locally we default to 8000 for convenience.
PORT="${PORT:-8000}"
exec uvicorn api.main:app --host 0.0.0.0 --port "$PORT"
