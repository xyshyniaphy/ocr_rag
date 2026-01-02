#!/bin/bash
# ============================================
# Test Runner Script
# Convenience script for running tests
# ============================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
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

# Default values
TEST_TYPE="all"
COVERAGE=false
VERBOSE=false
PARALLEL=false
MARKERS=""
WATCH=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        unit|integration|e2e|performance)
            TEST_TYPE="$1"
            shift
            ;;
        --coverage|-c)
            COVERAGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --parallel|-p)
            PARALLEL=true
            shift
            ;;
        --marker|-m)
            MARKERS="-m $2"
            shift 2
            ;;
        --watch|-w)
            WATCH=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [unit|integration|e2e|performance] [options]"
            echo ""
            echo "Options:"
            echo "  --coverage, -c     Generate coverage report"
            echo "  --verbose, -v      Verbose output"
            echo "  --parallel, -p     Run tests in parallel"
            echo "  --marker, -m MARK  Run tests with marker"
            echo "  --watch, -w        Watch mode (re-run on changes)"
            echo "  --help, -h         Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 unit                    # Run unit tests"
            echo "  $0 integration --coverage  # Run integration tests with coverage"
            echo "  $0 -m 'not slow'           # Run fast tests only"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest"

# Add test path based on type
case $TEST_TYPE in
    unit)
        PYTEST_CMD="$PYTEST_CMD tests/unit"
        ;;
    integration)
        PYTEST_CMD="$PYTEST_CMD tests/integration"
        ;;
    e2e)
        PYTEST_CMD="$PYTEST_CMD tests/e2e"
        ;;
    performance)
        PYTEST_CMD="$PYTEST_CMD tests/performance --benchmark-only"
        ;;
    all)
        PYTEST_CMD="$PYTEST_CMD tests"
        ;;
esac

# Add options
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
else
    PYTEST_CMD="$PYTEST_CMD -q"
fi

if [ "$PARALLEL" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -n auto"
fi

if [ "$WATCH" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -f"
fi

if [ -n "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD $MARKERS"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=backend --cov-report=term-missing --cov-report=html"
fi

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    log_error "pytest is not installed"
    log_info "Install with: pip install -r requirements-test.txt"
    exit 1
fi

# Run tests
log_info "Running tests..."
log_info "Command: $PYTEST_CMD"
echo ""

# Run pytest
if eval $PYTEST_CMD; then
    echo ""
    log_success "All tests passed!"
    EXIT_CODE=0
else
    echo ""
    log_error "Some tests failed"
    EXIT_CODE=1
fi

# Show coverage report if requested
if [ "$COVERAGE" = true ]; then
    echo ""
    log_info "Coverage report generated: htmlcov/index.html"
fi

exit $EXIT_CODE
