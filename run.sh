#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Run the FastAPI server
uvicorn backend:app --host 0.0.0.0 --port 8001