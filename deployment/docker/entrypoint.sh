#!/bin/bash
# Entrypoint script for production environment
# Starts both FastAPI backend and Streamlit frontend

set -e

echo "=========================================="
echo "Japanese OCR RAG System - Production"
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
    echo "ERROR: $service did not become ready in time"
    return 1
}

# Wait for dependencies
wait_for_service ${POSTGRES_HOST:-postgres} ${POSTGRES_PORT:-5432} "PostgreSQL" 30 &
wait_for_service ${MILVUS_HOST:-milvus} ${MILVUS_PORT:-19530} "Milvus" 60 &
wait_for_service ${REDIS_HOST:-redis} ${REDIS_PORT:-6379} "Redis" 30 &
wait_for_service ${MINIO_ENDPOINT%:*} ${MINIO_ENDPOINT##*:} "MinIO" 30 &
wait

# Initialize database if needed
echo "Checking database initialization..."
python -c "
import asyncio
from backend.db.session import init_db, check_schema
async def setup():
    if not await check_schema():
        print('Initializing database schema...')
        await init_db()
    else:
        print('Database schema exists')
asyncio.run(setup())
" 2>/dev/null || echo "Database initialization skipped or will be done on first run"

# Create Milvus collection if needed
echo "Checking Milvus collection..."
python -c "
import asyncio
from backend.db.vector.client import init_milvus
async def setup():
    await init_milvus()
asyncio.run(setup())
" 2>/dev/null || echo "Milvus initialization skipped"

# Create MinIO buckets if needed
echo "Checking MinIO buckets..."
python -c "
import asyncio
from backend.storage.client import create_buckets
async def setup():
    await create_buckets()
asyncio.run(setup())
" 2>/dev/null || echo "MinIO initialization skipped"

# Start services
echo ""
echo "Starting services..."
echo "  - FastAPI backend: http://0.0.0.0:8000"
echo "  - Streamlit frontend: http://0.0.0.0:8501"
echo ""

# Start FastAPI in background
cd /app
uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --loop uvloop \
    --log-level info \
    --access-log \
    --no-use-colors > /app/logs/fastapi.log 2>&1 &
FASTAPI_PID=$!

# Start Streamlit
streamlit run /app/frontend/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --logger.level=info > /app/logs/streamlit.log 2>&1 &
STREAMLIT_PID=$!

echo "Services started!"
echo "  FastAPI PID: $FASTAPI_PID"
echo "  Streamlit PID: $STREAMLIT_PID"
echo ""

# Handle shutdown signals
trap 'echo "Shutting down..."; kill $FASTAPI_PID $STREAMLIT_PID 2>/dev/null; wait; exit 0' SIGTERM SIGINT

# Wait for any process to exit
wait -n

# If one process exits, kill the other
echo "One service exited, shutting down..."
kill $FASTAPI_PID $STREAMLIT_PID 2>/dev/null
wait

exit 0
