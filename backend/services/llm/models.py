"""
LLM Service Models
Pydantic models for LLM requests, responses, and configuration
"""

from typing import List, Optional, Dict, Any, Sequence
from pydantic import BaseModel, Field, field_validator
import time


# ============================================================================
# Configuration Models
# ============================================================================

class LLMOptions(BaseModel):
    """
    LLM generation options

    Attributes:
        temperature: Sampling temperature (0.0-1.0). Lower = more deterministic
        top_p: Nucleus sampling parameter (0.0-1.0)
        top_k: Top-k sampling parameter
        num_ctx: Context window size (tokens)
        num_predict: Maximum tokens to generate
        repeat_penalty: Penalty for repeating tokens (1.0 = no penalty)
        seed: Random seed for reproducible generation
        stop: Stop sequences (list of strings)
    """

    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling")
    top_k: int = Field(default=40, ge=0, description="Top-k sampling")
    num_ctx: int = Field(default=32768, ge=1, description="Context window size")
    num_predict: int = Field(default=2048, ge=0, description="Max tokens to generate")
    repeat_penalty: float = Field(default=1.1, ge=0.0, le=2.0, description="Repeat penalty")
    seed: Optional[int] = Field(default=None, ge=0, description="Random seed")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")

    # Streaming options
    stream: bool = Field(default=False, description="Enable streaming responses")

    def to_ollama_format(self) -> Dict[str, Any]:
        """Convert to Ollama API format"""
        options = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "repeat_penalty": self.repeat_penalty,
        }
        if self.seed is not None:
            options["seed"] = self.seed
        if self.stop:
            options["stop"] = self.stop
        return options


class Message(BaseModel):
    """
    Chat message

    Attributes:
        role: Message role (system, user, assistant)
        content: Message content
    """

    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., min_length=1, description="Message content")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        valid_roles = ["system", "user", "assistant"]
        if v not in valid_roles:
            raise ValueError(f"Role must be one of {valid_roles}, got '{v}'")
        return v

    def to_ollama_format(self) -> Dict[str, str]:
        """Convert to Ollama API format"""
        return {"role": self.role, "content": self.content}


# ============================================================================
# Request Models
# ============================================================================

class LLMChatRequest(BaseModel):
    """
    LLM chat generation request

    Attributes:
        messages: List of chat messages
        options: Generation options (uses default if None)
        model: Model name (uses default if None)
    """

    messages: Sequence[Message] = Field(..., min_length=1, description="Chat messages")
    options: Optional[LLMOptions] = Field(default=None, description="Generation options")
    model: Optional[str] = Field(default=None, description="Model name")

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: Sequence[Message]) -> Sequence[Message]:
        if not v:
            raise ValueError("Messages list cannot be empty")
        return v


class LLMCompletionRequest(BaseModel):
    """
    LLM text completion request (simpler API for single prompt)

    Attributes:
        prompt: Text prompt
        system_prompt: Optional system prompt
        options: Generation options (uses default if None)
        model: Model name (uses default if None)
    """

    prompt: str = Field(..., min_length=1, description="Text prompt")
    system_prompt: Optional[str] = Field(default=None, description="System prompt")
    options: Optional[LLMOptions] = Field(default=None, description="Generation options")
    model: Optional[str] = Field(default=None, description="Model name")


# ============================================================================
# Response Models
# ============================================================================

class LLMMessage(BaseModel):
    """Single message in response"""

    role: str = Field(..., description="Message role")
    content: str = Field(..., description="Message content")


class LLMResponse(BaseModel):
    """
    LLM generation response

    Attributes:
        content: Generated text content
        model: Model name used
        finish_reason: Reason for completion (stop, length, etc.)
        prompt_tokens: Number of tokens in prompt
        completion_tokens: Number of tokens generated
        total_tokens: Total tokens used
        processing_time_ms: Processing time in milliseconds
    """

    content: str = Field(..., description="Generated text")
    model: str = Field(..., description="Model name used")
    finish_reason: str = Field(..., description="Completion reason")
    prompt_tokens: int = Field(default=0, ge=0, description="Prompt tokens")
    completion_tokens: int = Field(default=0, ge=0, description="Generated tokens")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens")
    processing_time_ms: float = Field(..., ge=0, description="Processing time")


class LLMStreamingChunk(BaseModel):
    """
    Streaming response chunk

    Attributes:
        content: Partial text content
        finish_reason: Completion reason (None if not finished)
        done: Whether generation is complete
    """

    content: str = Field(default="", description="Partial content")
    finish_reason: Optional[str] = Field(default=None, description="Finish reason")
    done: bool = Field(default=False, description="Is complete")


# ============================================================================
# RAG-Specific Models
# ============================================================================

class RAGContext(BaseModel):
    """
    RAG context document

    Attributes:
        text: Document text
        doc_id: Document ID
        score: Relevance score
        metadata: Additional metadata
    """

    text: str = Field(..., min_length=1, description="Document text")
    doc_id: Optional[str] = Field(default=None, description="Document ID")
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RAGRequest(BaseModel):
    """
    RAG-augmented generation request

    Attributes:
        query: User query
        contexts: Retrieved context documents
        system_prompt: Optional system prompt
        options: Generation options
        model: Model name
    """

    query: str = Field(..., min_length=1, description="User query")
    contexts: Sequence[RAGContext] = Field(..., min_length=1, description="Retrieved contexts")
    system_prompt: Optional[str] = Field(default=None, description="System prompt")
    options: Optional[LLMOptions] = Field(default=None, description="Generation options")
    model: Optional[str] = Field(default=None, description="Model name")


class RAGResponse(BaseModel):
    """
    RAG-augmented generation response

    Attributes:
        query: Original query
        answer: Generated answer
        sources: Source information
        model: Model used
        processing_time_ms: Processing time
    """

    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source information")
    model: str = Field(..., description="Model used")
    processing_time_ms: float = Field(..., ge=0, description="Processing time")


# ============================================================================
# Exceptions
# ============================================================================

class LLMValidationError(Exception):
    """Raised when input validation fails"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class LLMProcessingError(Exception):
    """Raised when LLM generation fails"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class LLMConnectionError(Exception):
    """Raised when connection to Ollama fails"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class LLMModelNotFoundError(Exception):
    """Raised when requested model is not available"""

    def __init__(self, model: str, details: Optional[Dict[str, Any]] = None):
        self.model = model
        self.message = f"Model '{model}' not found or not loaded"
        self.details = details or {}
        super().__init__(self.message)
