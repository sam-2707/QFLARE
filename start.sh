#!/bin/bash

# QFLARE Server Start Script for Railway
# This script properly handles the PORT environment variable

echo "🚀 Starting QFLARE Server..."

# Get PORT from environment, default to 8000
PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}

echo "📋 Configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT"

# Start the server
exec uvicorn server.main:app --host $HOST --port $PORT 