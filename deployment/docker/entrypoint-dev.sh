#!/bin/bash
# Entrypoint script for development environment
# Runs with hot-reload enabled

set -e

echo "=========================================="
echo "Japanese OCR RAG System - Development"
echo "=========================================="

# Note: Docker Compose handles service dependencies via depends_on and health checks
# Services are ready when this script starts

# Run database migrations
echo "Running database migrations..."
cd /app && python -m alembic upgrade head 2>/dev/null || echo "Migration skipped (first run or no changes)"

# Start services
echo ""
echo "Starting development services..."
echo "  - FastAPI backend: http://localhost:8000 (with hot-reload)"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Streamlit frontend: http://localhost:8501"
echo "  - Flower (Celery): http://localhost:9100"
echo ""

# Create logs directory
mkdir -p /app/logs

# Start FastAPI with auto-reload (output to stdout/stderr)
cd /app
uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info \
    --access-log \
    2>&1 &
FASTAPI_PID=$!

# Start Celery worker with auto-reload (output to stdout/stderr)
celery -A backend.tasks.celery_app worker \
    --loglevel=info \
    --concurrency=2 \
    2>&1 &
CELERY_PID=$!

# Start Flower for Celery monitoring (output to stdout/stderr)
celery -A backend.tasks.celery_app flower \
    --port=9100 \
    --broker=redis://${REDIS_PASSWORD:-}@${REDIS_HOST:-redis}:${REDIS_PORT:-6379}/1 \
    2>&1 &
FLOWER_PID=$!

# Start Streamlit (output to stdout/stderr)
streamlit run /app/frontend/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --logger.level=info \
    2>&1 &
STREAMLIT_PID=$!

echo "All services started!"
echo "  FastAPI PID: $FASTAPI_PID"
echo "  Celery PID: $CELERY_PID"
echo "  Flower PID: $FLOWER_PID"
echo "  Streamlit PID: $STREAMLIT_PID"
echo ""

# Handle shutdown signals
trap 'echo "Shutting down..."; kill $FASTAPI_PID $CELERY_PID $FLOWER_PID $STREAMLIT_PID 2>/dev/null; wait; exit 0' SIGTERM SIGINT

# Wait for any process to exit
wait -n

# If one process exits, kill the others
echo "One service exited, shutting down others..."
kill $FASTAPI_PID $CELERY_PID $FLOWER_PID $STREAMLIT_PID 2>/dev/null
wait

exit 0
