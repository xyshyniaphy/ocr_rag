"""
LLM Service
Large Language Model generation using Qwen models via Ollama

This service provides:
- Chat completions with conversation history
- Text completions for simple prompts
- RAG-augmented generation with retrieved contexts
- Streaming responses
- Model management and health checks

Model: Qwen3 (4B parameters via Ollama)
"""

from backend.services.llm.service import LLMService, get_llm_service
from backend.services.llm.models import (
    Message,
    LLMOptions,
    LLMChatRequest,
    LLMCompletionRequest,
    LLMResponse,
    RAGRequest,
    RAGResponse,
    RAGContext,
    LLMValidationError,
    LLMProcessingError,
    LLMConnectionError,
    LLMModelNotFoundError,
)

__all__ = [
    # Service
    "LLMService",
    "get_llm_service",
    # Models
    "Message",
    "LLMOptions",
    "LLMChatRequest",
    "LLMCompletionRequest",
    "LLMResponse",
    "RAGRequest",
    "RAGResponse",
    "RAGContext",
    # Exceptions
    "LLMValidationError",
    "LLMProcessingError",
    "LLMConnectionError",
    "LLMModelNotFoundError",
]
