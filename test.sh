#!/bin/bash
# ============================================
# Test Runner Script for Docker Environment
# Runs all tests inside the Docker container
# ============================================

set -e

# ============================================
# CONFIGURATION
# ============================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
TEST_IMAGE="ocr-rag-test:dev"
TEST_CONTAINER="ocr-rag-test-dev"
APP_CONTAINER="ocr-rag-app-dev"
COMPOSE_FILE="docker-compose.dev.yml"
TEST_RESULTS_DIR="test-results"
COVERAGE_DIR="htmlcov"

# Test type defaults
TEST_TYPE="all"
COVERAGE=false
VERBOSE=false
PARALLEL=false
MARKER=""
WATCH=false
KEEP_CONTAINERS=false
BUILD=false

# ============================================
# FUNCTIONS
# ============================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_test() {
    echo -e "${CYAN}[TEST]${NC} $1"
}

log_header() {
    echo ""
    echo -e "${MAGENTA}============================================${NC}"
    echo -e "${MAGENTA}$1${NC}"
    echo -e "${MAGENTA}============================================${NC}"
}

# Show usage
show_usage() {
    cat << EOF
${CYAN}Usage:${NC} $0 [OPTIONS] [TEST_TYPE]

${CYAN}Test Types:${NC}
  unit              Run unit tests only (fast)
  integration       Run integration tests (medium)
  e2e              Run end-to-end tests (slow)
  performance      Run performance benchmarks
  all              Run all tests (default)

${CYAN}Options:${NC}
  -c, --coverage    Generate coverage report
  -v, --verbose     Verbose output
  -p, --parallel    Run tests in parallel
  -m, --marker MARK Run tests with marker (e.g., "not slow")
  -w, --watch       Watch mode (re-run on changes)
  -b, --build       Rebuild containers before testing
  -k, --keep        Keep containers running after tests
  -h, --help        Show this help message

${CYAN}Examples:${NC}
  $0 unit                    # Run unit tests
  $0 integration --coverage   # Run integration tests with coverage
  $0 -m "not slow"            # Run fast tests only
  $0 all --coverage --parallel # Run all tests with coverage in parallel
  $0 -b                       # Rebuild and run all tests

${CYAN}Markers:${NC}
  unit              Unit tests (fast, isolated)
  integration       Integration tests (medium speed)
  e2e              End-to-end tests (slow)
  slow             Slow running tests
  gpu              Tests requiring GPU
  external         Tests requiring external services

${CYAN}Environment:${NC}
  Tests run inside Docker container: ${APP_CONTAINER}
  Requires docker-compose services to be running

EOF
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            unit|integration|e2e|performance|all)
                TEST_TYPE="$1"
                shift
                ;;
            -c|--coverage)
                COVERAGE=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -p|--parallel)
                PARALLEL=true
                shift
                ;;
            -m|--marker)
                if [ -n "$2" ] && [[ ! $2 =~ ^- ]]; then
                    MARKER="$2"
                    shift 2
                else
                    log_error "Marker requires a value"
                    exit 1
                fi
                ;;
            -w|--watch)
                WATCH=true
                shift
                ;;
            -b|--build)
                BUILD=true
                shift
                ;;
            -k|--keep)
                KEEP_CONTAINERS=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Check if container is running
check_container() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${TEST_CONTAINER}$"; then
        return 1
    fi
    return 0
}

# Start containers
start_containers() {
    log_info "Starting Docker containers..."
    docker-compose -f "$COMPOSE_FILE" up -d

    log_info "Waiting for services to be healthy..."
    sleep 10

    if ! check_container; then
        log_error "Failed to start ${APP_CONTAINER}"
        exit 1
    fi

    log_success "Containers started successfully"
}

# Stop containers
stop_containers() {
    if [ "$KEEP_CONTAINERS" = false ]; then
        log_info "Stopping test container..."
        docker-compose -f "$COMPOSE_FILE" stop test
        log_success "Test container stopped"
        log_info "Other containers are still running (use ./dev.sh down to stop all)"
    else
        log_info "Keeping containers running"
    fi
}

# Build containers
build_containers() {
    log_info "Building containers..."

    # Build test image specifically
    docker build -f Dockerfile.test --target test -t "$TEST_IMAGE" .

    log_success "Containers built successfully"
}

# Build test image if needed
build_test_image_if_needed() {
    # Check if test image exists
    if ! docker image inspect "$TEST_IMAGE" &> /dev/null; then
        log_info "Test image not found, building..."
        build_containers
    else
        # Check if base image or Dockerfile.test is newer
        base_image_mtime=$(docker inspect ocr-rag:base --format '{{.Created}}' 2>/dev/null || echo "")
        test_image_mtime=$(docker inspect "$TEST_IMAGE" --format '{{.Created}}' 2>/dev/null || echo "")
        dockerfile_mtime=$(stat -c %Y Dockerfile.test 2>/dev/null || stat -f %m Dockerfile.test 2>/dev/null || echo "0")

        # Simple heuristic: rebuild if Dockerfile.test is newer than test image
        if [ "$dockerfile_mtime" -gt 0 ]; then
            log_info "Checking if test image needs rebuild..."
            # For simplicity, we'll skip complex timestamp comparison
            # Just ensure the image exists
        fi
    fi
}

# Create test results directory
create_results_dir() {
    mkdir -p "$TEST_RESULTS_DIR"
    mkdir -p "$COVERAGE_DIR"
}

# Build pytest command
build_pytest_command() {
    local cmd="pytest"

    # Add test path based on type
    case $TEST_TYPE in
        unit)
            cmd="$cmd tests/unit"
            ;;
        integration)
            cmd="$cmd tests/integration"
            ;;
        e2e)
            cmd="$cmd tests/e2e"
            ;;
        performance)
            cmd="$cmd tests/performance --benchmark-only"
            ;;
        all)
            cmd="$cmd tests"
            ;;
    esac

    # Add options
    if [ "$VERBOSE" = true ]; then
        cmd="$cmd -vv"
    else
        cmd="$cmd -v"
    fi

    if [ "$PARALLEL" = true ]; then
        cmd="$cmd -n auto"
    fi

    if [ "$WATCH" = true ]; then
        cmd="$cmd -f"
    fi

    if [ -n "$MARKER" ]; then
        cmd="$cmd -m \"$MARKER\""
    fi

    if [ "$COVERAGE" = true ]; then
        cmd="$cmd --cov=backend --cov-report=term-missing --cov-report=html:/app/htmlcov --cov-report=xml:/app/test-results/coverage.xml --cov-report=json:/app/test-results/coverage.json"
    fi

    # Add junitxml report
    cmd="$cmd --junitxml=/app/test-results/junit.xml"

    # Add html report
    cmd="$cmd --html=/app/test-results/report.html --self-contained-html"

    echo "$cmd"
}

# Run tests in container
run_tests() {
    local pytest_cmd=$(build_pytest_command)

    log_header "Running Tests in Docker"
    log_info "Container: ${TEST_CONTAINER}"
    log_info "Test Type: ${TEST_TYPE}"
    log_info "Command: ${pytest_cmd}"
    echo ""

    # Run tests in container (no -it flag for non-interactive)
    docker exec \
        -e PYTEST_ADDOPTS="-v" \
        -e PYTHONPATH=/app \
        "$TEST_CONTAINER" \
        bash -c "cd /app && $pytest_cmd"

    local exit_code=$?

    # Copy results from container
    if [ -d "$TEST_RESULTS_DIR" ] || [ "$COVERAGE" = true ]; then
        log_info "Copying test results from container..."
        docker cp "${TEST_CONTAINER}:/app/test-results/." "$TEST_RESULTS_DIR/" 2>/dev/null || true
        docker cp "${TEST_CONTAINER}:/app/htmlcov/." "$COVERAGE_DIR/" 2>/dev/null || true
    fi

    return $exit_code
}

# Show test results
show_results() {
    local exit_code=$1

    echo ""
    log_header "Test Results"

    if [ $exit_code -eq 0 ]; then
        log_success "All tests passed!"

        if [ "$COVERAGE" = true ]; then
            echo ""
            log_info "Coverage Report: ${COVERAGE_DIR}/index.html"
            log_info "Coverage JSON: ${TEST_RESULTS_DIR}/coverage.json"
        fi

        echo ""
        log_info "HTML Report: ${TEST_RESULTS_DIR}/report.html"
        log_info "JUnit XML: ${TEST_RESULTS_DIR}/junit.xml"
    else
        log_error "Some tests failed"
        echo ""
        log_info "Check the test output above for details"
    fi

    return $exit_code
}

# ============================================
# MAIN SCRIPT
# ============================================

main() {
    log_header "OCR RAG System - Docker Test Runner"

    # Parse arguments
    parse_args "$@"

    # Build test image if requested or if it doesn't exist
    if [ "$BUILD" = true ]; then
        build_containers
    else
        build_test_image_if_needed
    fi

    # Check if test container is running, if not start it
    if ! check_container; then
        log_info "Test container not running, starting..."
        docker-compose -f "$COMPOSE_FILE" up -d test
        sleep 5
    fi

    # Create results directory
    create_results_dir

    # Run tests
    run_tests
    local exit_code=$?

    # Show results
    show_results $exit_code

    # Stop containers if not keeping them
    if [ "$KEEP_CONTAINERS" = false ]; then
        echo ""
        log_info "Containers are still running (use ./dev.sh down to stop)"
    fi

    exit $exit_code
}

# Run main function
main "$@"
