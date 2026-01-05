#!/bin/bash
# dev.sh - Development environment management script
# Usage: ./dev.sh [command]
# Commands: up, down, restart, logs, shell, ps, rebuild, help

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project configuration
COMPOSE_FILE="docker-compose.dev.yml"
PROJECT_NAME="ocr-rag-dev"
BASE_IMAGE="ocr-rag:base"
APP_IMAGE="ocr-rag-app:dev"

# Print colored message
print_msg() {
    local color=$1
    shift
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $*"
}

# Print header
print_header() {
    echo ""
    print_msg "${BLUE}" "=========================================="
    print_msg "${BLUE}" "$*"
    print_msg "${BLUE}" "=========================================="
    echo ""
}

# Check if docker-compose file exists
check_compose_file() {
    if [ ! -f "$COMPOSE_FILE" ]; then
        print_msg "${RED}" "Error: $COMPOSE_FILE not found!"
        exit 1
    fi
}

# Check if base image exists, build if not
ensure_base_image() {
    if ! docker image inspect "$BASE_IMAGE" &>/dev/null; then
        print_msg "${YELLOW}" "Base image $BASE_IMAGE not found. Building..."
        docker build -f Dockerfile.base -t "$BASE_IMAGE" .
        print_msg "${GREEN}" "Base image built successfully!"
    fi
}

# Build base image
build_base() {
    print_header "Building Base Image"
    docker build -f Dockerfile.base -t "$BASE_IMAGE" .
    print_msg "${GREEN}" "Base image build complete!"
}

# Build app image
build_app() {
    print_header "Building Application Image"
    ensure_base_image
    docker build -f Dockerfile.app -t "$APP_IMAGE" .
    print_msg "${GREEN}" "Application image build complete!"
}

# Start development environment
dev_up() {
    print_header "Starting Development Environment"

    check_compose_file
    ensure_base_image

    # Create necessary directories
    mkdir -p data logs models config/prometheus config/grafana/{dashboards,datasources}

    # Start services
    print_msg "${BLUE}" "Starting Docker Compose services..."
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d

    # Wait for services to be healthy
    print_msg "${YELLOW}" "Waiting for services to be ready..."
    sleep 5

    # Show status
    dev_ps

    # Show access information
    print_header "Access Points"
    cat << 'EOF'
Service                URL
----------------------  ------------------------------------------
FastAPI Backend        http://localhost:8000
API Docs               http://localhost:8000/docs
Streamlit Admin UI     http://localhost:8501
WebSocket              ws://localhost:8000/api/v1/stream/ws
MinIO Console          http://localhost:9001
PostgreSQL             localhost:5432
Milvus                 localhost:19530
Redis                  localhost:6379
Ollama                 http://localhost:11434
Flower (Celery)        http://localhost:9100
Prometheus             http://localhost:9090 (with --profile monitoring)
Grafana                http://localhost:3000 (with --profile monitoring)

Default Credentials:
  Admin User: admin@example.com / admin123
  MinIO: minioadmin / minioadmin

Commands:
  ./dev.sh logs       - View logs
  ./dev.sh shell      - Open shell in app container
  ./dev.sh down       - Stop all services
EOF
}

# Stop development environment
dev_down() {
    print_header "Stopping Development Environment"

    # Stop all services with different profiles
    print_msg "${BLUE}" "Stopping main services..."
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down --remove-orphans 2>/dev/null || true

    # Stop test services if running
    print_msg "${BLUE}" "Stopping test services..."
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" --profile test down --remove-orphans 2>/dev/null || true

    # Stop monitoring services if running
    print_msg "${BLUE}" "Stopping monitoring services..."
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" --profile monitoring down --remove-orphans 2>/dev/null || true

    # Stop tools services if running
    print_msg "${BLUE}" "Stopping tools services..."
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" --profile tools down --remove-orphans 2>/dev/null || true

    # Force stop any remaining containers with project name
    print_msg "${YELLOW}" "Checking for any remaining containers..."
    remaining_containers=$(docker ps -a --filter "name=ocr-rag-" --format "{{.Names}}" 2>/dev/null || true)
    if [ -n "$remaining_containers" ]; then
        print_msg "${YELLOW}" "Force stopping remaining containers: $remaining_containers"
        echo "$remaining_containers" | xargs -r docker stop -t 5 2>/dev/null || true
        echo "$remaining_containers" | xargs -r docker rm 2>/dev/null || true
    fi

    # Remove network if it still exists
    print_msg "${BLUE}" "Cleaning up network..."
    docker network rm ocr-rag-net-dev 2>/dev/null || true

    print_msg "${GREEN}" "All services stopped!"
}

# Restart development environment
dev_restart() {
    print_header "Restarting Development Environment"
    dev_down
    sleep 2
    dev_up
}

# View logs
dev_logs() {
    local service=$1

    if [ -n "$service" ]; then
        print_msg "${BLUE}" "Showing logs for: $service"
        docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs -f "$service"
    else
        print_msg "${BLUE}" "Showing logs for all services (Ctrl+C to exit)"
        docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs -f
    fi
}

# Open shell in app container
dev_shell() {
    print_header "Opening Shell in App Container"
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" exec app /bin/bash
}

# Show process status
dev_ps() {
    print_header "Container Status"
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps
}

# Rebuild images
dev_rebuild() {
    local what=${1:-app}

    case "$what" in
        base)
            build_base
            ;;
        app)
            build_app
            ;;
        all)
            build_base
            build_app
            ;;
        *)
            print_msg "${RED}" "Unknown target: $what"
            print_msg "${YELLOW}" "Valid targets: base, app, all"
            exit 1
            ;;
    esac
}

# Pull latest images
dev_pull() {
    print_header "Pulling Latest Images"
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" pull
}

# Clean everything (volumes, images, containers)
dev_clean() {
    print_header "Cleaning Development Environment"
    print_msg "${YELLOW}" "This will remove all containers, volumes, and images!"
    read -p "Are you sure? (yes/no): " confirm

    if [ "$confirm" = "yes" ]; then
        docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down -v --remove-orphans
        docker rmi "$APP_IMAGE" "$BASE_IMAGE" 2>/dev/null || true
        print_msg "${GREEN}" "Cleanup complete!"
    else
        print_msg "${YELLOW}" "Cleanup cancelled."
    fi
}

# Show help
dev_help() {
    cat << 'EOF'
dev.sh - Development Environment Management Script

Usage: ./dev.sh [command] [options]

Commands:
  up                 Start development environment (default)
  down               Stop all services
  restart            Restart all services
  logs [service]     View logs (all services or specific service)
  shell              Open shell in app container
  ps                 Show container status
  rebuild [target]   Rebuild images (target: base, app, all)
  pull               Pull latest base images
  clean              Remove all containers, volumes, and images
  help               Show this help message

Services for logs command:
  app, postgres, milvus, etcd, minio, redis, ollama, worker, pgadmin,
  prometheus, grafana

Examples:
  ./dev.sh                    # Start development environment
  ./dev.sh logs app           # View app logs
  ./dev.sh shell              # Open shell in app container
  ./dev.sh rebuild app        # Rebuild application image
  ./dev.sh down               # Stop all services

Options:
  --profile [name]            Enable profile (monitoring, tools)
  --build                     Build images before starting

Environment Variables:
  COMPOSE_FILE                Docker Compose file (default: docker-compose.dev.yml)
  PROJECT_NAME                Project name (default: ocr-rag-dev)
EOF
}

# Main script logic
main() {
    local command=${1:-up}

    case "$command" in
        up)
            shift
            docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up "$@"
            ;;
        down)
            dev_down
            ;;
        restart)
            dev_restart
            ;;
        logs)
            shift
            dev_logs "$@"
            ;;
        shell)
            dev_shell
            ;;
        ps|status)
            dev_ps
            ;;
        rebuild)
            shift
            dev_rebuild "$@"
            ;;
        pull)
            dev_pull
            ;;
        clean)
            dev_clean
            ;;
        help|--help|-h)
            dev_help
            ;;
        *)
            print_msg "${RED}" "Unknown command: $command"
            echo ""
            dev_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
