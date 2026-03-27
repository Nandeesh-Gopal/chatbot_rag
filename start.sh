#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "    Starting Nova AI RAG Application    "
echo "========================================"

# Install backend dependencies
echo ""
echo "📦 Installing backend dependencies..."
pip install -r "$ROOT_DIR/backend/requirements.txt" -q

# Start FastAPI backend
echo ""
echo "🚀 Starting FastAPI backend on http://127.0.0.1:8000 ..."
cd "$ROOT_DIR/backend" && uvicorn api:app --reload --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Give backend a moment to start
sleep 2

# Start frontend dev server
echo ""
echo "🌐 Starting Vite frontend on http://localhost:5173 ..."
cd "$ROOT_DIR/frontend" && npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Both servers are running!"
echo "   Backend  → http://127.0.0.1:8000"
echo "   Frontend → http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers."

# Wait and handle Ctrl+C gracefully
trap "echo ''; echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
