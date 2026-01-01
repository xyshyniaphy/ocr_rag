"""
Ollama Client
HTTP client for Ollama API (Qwen models via Ollama)

Ollama provides a simple HTTP API for running LLMs locally.
This client handles:
- Model listing and availability checks
- Chat completions
- Text completions
- Streaming responses
- Error handling and retries

API Reference: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
import aiohttp

from backend.core.logging import get_logger
from backend.core.config import settings
from backend.services.llm.models import (
    Message,
    LLMOptions,
    LLMResponse,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMProcessingError,
)

logger = get_logger(__name__)


class OllamaClient:
    """
    Ollama HTTP client for LLM inference

    This client connects to Ollama server and provides methods for:
    - Chat completions
    - Text completions
    - Model listing
    - Health checks

    Example:
        ```python
        client = OllamaClient(host="http://localhost:11434")

        # Chat completion
        response = await client.chat(
            messages=[
                {"role": "user", "content": "Hello!"}
            ],
            model="qwen3:4b"
        )

        # Text completion
        response = await client.generate(
            prompt="Once upon a time...",
            model="qwen3:4b"
        )
        ```
    """

    def __init__(
        self,
        host: Optional[str] = None,
        timeout: int = 120,
        connect_timeout: int = 10,
    ):
        """
        Initialize Ollama client

        Args:
            host: Ollama server host (uses config default if None)
            timeout: Request timeout in seconds
            connect_timeout: Connection timeout in seconds
        """
        # Ensure host has http:// prefix
        host_str = host or settings.OLLAMA_HOST
        if not host_str.startswith(("http://", "https://")):
            host_str = f"http://{host_str}"
        self.host = host_str.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout, connect=connect_timeout)
        self._session: Optional[aiohttp.ClientSession] = None
        self._available_models: Optional[List[str]] = None

        logger.info(f"OllamaClient initialized (host={self.host}, timeout={timeout}s)")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self) -> None:
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("OllamaClient session closed")

    # ========================================================================
    # Model Management
    # ========================================================================

    async def list_models(self, force_refresh: bool = False) -> List[str]:
        """
        List available models

        Args:
            force_refresh: Force refresh of cached model list

        Returns:
            List of model names
        """
        if self._available_models and not force_refresh:
            return self._available_models

        try:
            session = await self._get_session()
            url = f"{self.host}/api/tags"

            async with session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMConnectionError(
                        f"Failed to list models: HTTP {response.status}",
                        details={"status": response.status, "error": error_text},
                    )

                data = await response.json()
                models = [m["name"] for m in data.get("models", [])]
                self._available_models = models

                logger.debug(f"Available models: {models}")
                return models

        except aiohttp.ClientError as e:
            raise LLMConnectionError(
                f"Failed to connect to Ollama: {str(e)}",
                details={"host": self.host, "error_type": type(e).__name__},
            )

    async def check_model(self, model: str) -> bool:
        """
        Check if a model is available

        Args:
            model: Model name

        Returns:
            True if model is available, False otherwise
        """
        try:
            models = await self.list_models()
            # Handle both exact match and partial match (e.g., "qwen3:4b" vs "qwen3:4b-q4_K_M")
            return any(m == model or m.startswith(model + ":") or model in m for m in models)
        except Exception as e:
            logger.warning(f"Failed to check model availability: {e}")
            return False

    async def pull_model(self, model: str) -> None:
        """
        Pull a model from Ollama registry

        Args:
            model: Model name to pull
        """
        logger.info(f"Pulling model: {model}")

        try:
            session = await self._get_session()
            url = f"{self.host}/api/pull"

            payload = {"name": model, "stream": False}

            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMProcessingError(
                        f"Failed to pull model: HTTP {response.status}",
                        details={"model": model, "status": response.status, "error": error_text},
                    )

                logger.info(f"Model '{model}' pulled successfully")

        except aiohttp.ClientError as e:
            raise LLMConnectionError(
                f"Failed to pull model: {str(e)}",
                details={"model": model, "error_type": type(e).__name__},
            )

    # ========================================================================
    # Chat Completions
    # ========================================================================

    async def chat(
        self,
        messages: List[Message],
        model: str,
        options: Optional[LLMOptions] = None,
        stream: bool = False,
    ) -> LLMResponse:
        """
        Chat completion

        Args:
            messages: List of chat messages
            model: Model name
            options: Generation options
            stream: Enable streaming (not yet implemented)

        Returns:
            LLMResponse with generated content

        Raises:
            LLMValidationError: If input validation fails
            LLMModelNotFoundError: If model not available
            LLMConnectionError: If connection fails
            LLMProcessingError: If generation fails
        """
        if not messages:
            raise LLMValidationError("Messages list cannot be empty")

        # Check model availability
        if not await self.check_model(model):
            raise LLMModelNotFoundError(model, details={"available_models": await self.list_models()})

        start_time = time.time()

        try:
            session = await self._get_session()
            url = f"{self.host}/api/chat"

            # Build request payload
            payload = {
                "model": model,
                "messages": [msg.to_ollama_format() for msg in messages],
                "stream": stream,
            }

            if options:
                payload["options"] = options.to_ollama_format()

            logger.debug(f"Chat request: model={model}, messages={len(messages)}")

            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMProcessingError(
                        f"Chat failed: HTTP {response.status}",
                        details={
                            "model": model,
                            "status": response.status,
                            "error": error_text,
                        },
                    )

                data = await response.json()

                # Parse response
                content = data.get("message", {}).get("content", "")
                finish_reason = data.get("done_reason", "stop")

                # Token counts (Ollama provides these in the response)
                prompt_tokens = data.get("prompt_eval_count", 0)
                completion_tokens = data.get("eval_count", 0)
                total_tokens = prompt_tokens + completion_tokens

                processing_time = (time.time() - start_time) * 1000

                logger.debug(
                    f"Chat response: model={model}, "
                    f"tokens={total_tokens}, time={processing_time:.0f}ms"
                )

                return LLMResponse(
                    content=content,
                    model=model,
                    finish_reason=finish_reason,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    processing_time_ms=round(processing_time, 2),
                )

        except LLMProcessingError:
            raise
        except LLMModelNotFoundError:
            raise
        except aiohttp.ClientError as e:
            raise LLMConnectionError(
                f"Failed to connect to Ollama: {str(e)}",
                details={"host": self.host, "model": model, "error_type": type(e).__name__},
            )
        except Exception as e:
            raise LLMProcessingError(
                f"Chat generation failed: {str(e)}",
                details={"model": model, "error_type": type(e).__name__},
            )

    async def chat_stream(
        self,
        messages: List[Message],
        model: str,
        options: Optional[LLMOptions] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming chat completion

        Args:
            messages: List of chat messages
            model: Model name
            options: Generation options

        Yields:
            Text chunks as they are generated
        """
        if not messages:
            raise LLMValidationError("Messages list cannot be empty")

        if not await self.check_model(model):
            raise LLMModelNotFoundError(model, details={"available_models": await self.list_models()})

        try:
            session = await self._get_session()
            url = f"{self.host}/api/chat"

            payload = {
                "model": model,
                "messages": [msg.to_ollama_format() for msg in messages],
                "stream": True,
            }

            if options:
                payload["options"] = options.to_ollama_format()

            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMProcessingError(
                        f"Streaming chat failed: HTTP {response.status}",
                        details={"model": model, "status": response.status, "error": error_text},
                    )

                # Read streaming response line by line
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue

                    try:
                        import json
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content

                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse streaming chunk: {line}")

        except (LLMProcessingError, LLMModelNotFoundError):
            raise
        except aiohttp.ClientError as e:
            raise LLMConnectionError(
                f"Failed to connect to Ollama: {str(e)}",
                details={"host": self.host, "model": model, "error_type": type(e).__name__},
            )
        except Exception as e:
            raise LLMProcessingError(
                f"Streaming chat failed: {str(e)}",
                details={"model": model, "error_type": type(e).__name__},
            )

    # ========================================================================
    # Text Completions
    # ========================================================================

    async def generate(
        self,
        prompt: str,
        model: str,
        options: Optional[LLMOptions] = None,
        system: Optional[str] = None,
    ) -> LLMResponse:
        """
        Text completion (simpler API for single prompt)

        Args:
            prompt: Text prompt
            model: Model name
            options: Generation options
            system: Optional system prompt

        Returns:
            LLMResponse with generated content
        """
        if not prompt:
            raise LLMValidationError("Prompt cannot be empty")

        # Convert to chat format
        messages = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))

        return await self.chat(messages=messages, model=model, options=options)

    # ========================================================================
    # Health Check
    # ========================================================================

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Ollama service

        Returns:
            Dictionary with health status
        """
        health = {
            "status": "unknown",
            "host": self.host,
            "models_available": [],
            "errors": [],
        }

        try:
            # Check connection
            session = await self._get_session()
            url = f"{self.host}/api/tags"

            async with session.get(url) as response:
                if response.status == 200:
                    health["status"] = "healthy"
                    health["models_available"] = await self.list_models()
                else:
                    health["status"] = "error"
                    health["errors"].append(f"HTTP {response.status}")

        except Exception as e:
            health["status"] = "error"
            health["errors"].append(str(e))

        return health

    # ========================================================================
    # Context Manager
    # ========================================================================

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
