#!/bin/bash
# Entrypoint script for development environment
# Runs with hot-reload enabled

set -e

echo "=========================================="
echo "Japanese OCR RAG System - Development"
echo "=========================================="

# Function to check if a service is ready
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    local max_attempts=${4:-30}
    local attempt=1

    echo "Waiting for $service ($host:$port)..."
    while [ $attempt -le $max_attempts ]; do
        if (echo > /dev/tcp/$host/$port) 2>/dev/null; then
            echo "$service is ready!"
            return 0
        fi
        echo "Attempt $attempt/$max_attempts: $service not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    echo "WARNING: $service did not become ready (continuing anyway)"
    return 0
}

# Wait for dependencies (with timeout, but continue if not ready)
wait_for_service ${POSTGRES_HOST:-postgres} ${POSTGRES_PORT:-5432} "PostgreSQL" 30 &
wait_for_service ${MILVUS_HOST:-milvus} ${MILVUS_PORT:-19530} "Milvus" 60 &
wait_for_service ${REDIS_HOST:-redis} ${REDIS_PORT:-6379} "Redis" 30 &
wait_for_service ${MINIO_ENDPOINT%%:*} 9000 "MinIO" 30 &
wait

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

# Start FastAPI with auto-reload
cd /app
uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level debug \
    --access-log \
    > /app/logs/fastapi.log 2>&1 &
FASTAPI_PID=$!

# Start Celery worker with auto-reload
celery -A backend.tasks.celery_app worker \
    --loglevel=info \
    --concurrency=2 \
    > /app/logs/celery.log 2>&1 &
CELERY_PID=$!

# Start Flower for Celery monitoring
celery -A backend.tasks.celery_app flower \
    --port=9100 \
    --broker=redis://${REDIS_PASSWORD:-}@${REDIS_HOST:-redis}:${REDIS_PORT:-6379}/1 \
    > /app/logs/flower.log 2>&1 &
FLOWER_PID=$!

# Start Streamlit
streamlit run /app/frontend/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --logger.level=debug \
    > /app/logs/streamlit.log 2>&1 &
STREAMLIT_PID=$!

echo "All services started!"
echo "  FastAPI PID: $FASTAPI_PID"
echo "  Celery PID: $CELERY_PID"
echo "  Flower PID: $FLOWER_PID"
echo "  Streamlit PID: $STREAMLIT_PID"
echo ""
echo "Logs available in /app/logs/"
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
