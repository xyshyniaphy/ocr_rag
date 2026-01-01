# Makefile for Japanese OCR RAG System

.PHONY: help build dev prod up down logs shell test lint format clean

# Default target
help:
	@echo "Japanese OCR RAG System - Makefile Commands"
	@echo ""
	@echo "Development:"
	@echo "  make build        - Build Docker images"
	@echo "  make dev          - Start development environment"
	@echo "  make prod         - Start production environment"
	@echo "  make up           - Start services (dev)"
	@echo "  make down         - Stop all services"
	@echo "  make logs         - Show logs from all services"
	@echo "  make shell        - Open shell in app container"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linting"
	@echo "  make format       - Format code"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        - Remove generated files"
	@echo "  make rebuild      - Rebuild Docker images"

# Build Docker images
build:
	@echo "Building Docker images..."
	docker-compose -f docker-compose.dev.yml build

# Build production images
build-prod:
	@echo "Building production Docker images..."
	docker build -f Dockerfile.base -t ocr-rag-base:latest .
	docker build -f Dockerfile.app -t ocr-rag/app:latest .

# Development environment
dev:
	@echo "Starting development environment..."
	docker-compose -f docker-compose.dev.yml --profile monitoring up -d

# Production environment
prod:
	@echo "Starting production environment..."
	docker-compose -f docker-compose.prd.yml up -d

# Start services
up:
	docker-compose -f docker-compose.dev.yml up -d

# Stop services
down:
	docker-compose -f docker-compose.dev.yml down

# Show logs
logs:
	docker-compose -f docker-compose.dev.yml logs -f

# Open shell
shell:
	docker-compose -f docker-compose.dev.yml exec app bash

# Run tests
test:
	docker-compose -f docker-compose.dev.yml exec app pytest tests/ -v

# Lint code
lint:
	docker-compose -f docker-compose.dev.yml exec app ruff check app/

# Format code
format:
	docker-compose -f docker-compose.dev.yml exec app ruff format app/

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	find . -type f -name '*.pyo' -delete 2>/dev/null || true
	find . -type f -name '*.coverage' -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov

# Rebuild
rebuild: clean build
	@echo "Rebuilt Docker images"
