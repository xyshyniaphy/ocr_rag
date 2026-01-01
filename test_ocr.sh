#!/bin/bash
# Test OCR Service in Docker
# This script runs the OCR service test inside the Docker container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}OCR Service Test Runner${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Detect which compose file to use
COMPOSE_FILE="docker-compose.dev.yml"
if [ ! -f "$COMPOSE_FILE" ]; then
    COMPOSE_FILE="docker-compose.yml"
fi

# Check if containers are running
if ! docker compose -f "$COMPOSE_FILE" ps 2>/dev/null | grep -q "app.*Up"; then
    echo -e "${YELLOW}Docker containers not running. Starting...${NC}"
    docker compose -f "$COMPOSE_FILE" up -d
    echo -e "${YELLOW}Waiting for services to be ready...${NC}"
    sleep 10
fi

echo -e "${GREEN}Using compose file: ${COMPOSE_FILE}${NC}"
echo ""

# Run the test inside the app container
echo -e "${GREEN}Running OCR service test in Docker...${NC}"
echo ""

# Try pytest first, fall back to direct python execution
docker compose -f "$COMPOSE_FILE" exec -T app python3 tests/test_ocr_service.py 2>&1

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo -e "${GREEN}======================================${NC}"
else
    echo -e "${RED}======================================${NC}"
    echo -e "${RED}✗ Tests failed with exit code $EXIT_CODE${NC}"
    echo -e "${RED}======================================${NC}"
fi

exit $EXIT_CODE
