#!/bin/bash
# Start the FastAPI application
# Works locally and on Render

# Use PORT environment variable if available (Render sets this)
# Otherwise default to 8001 for local development
PORT=${PORT:-8001}

uvicorn app:app --host 0.0.0.0 --port $PORT