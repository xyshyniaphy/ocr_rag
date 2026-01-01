#!/bin/bash
# Ollama container entrypoint
# Pre-pulls qwen3:4b model on startup, then serves

set -e

echo "=========================================="
echo "Starting Ollama with model pre-pull"
echo "=========================================="

# Pull qwen3:4b model if not already present
echo "Checking for qwen3:4b model..."
if ! ollama list | grep -q "qwen3:4b"; then
    echo "Pulling qwen3:4b model... (this may take a while on first run)"
    ollama pull qwen3:4b
    echo "✓ Model qwen3:4b pulled successfully"
else
    echo "✓ Model qwen3:4b already available"
fi

echo "=========================================="
echo "Starting Ollama server..."
echo "=========================================="

# Start Ollama server
exec ollama serve
