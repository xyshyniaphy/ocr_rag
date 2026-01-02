"""
GLM Client
GLM-4.5 Air client using OpenAI-compatible API

This client uses the standard OpenAI Python SDK to communicate with
GLM models via the international API endpoint (https://api.z.ai/).

Supported models:
- GLM-4.5-Air (recommended, fast and cost-effective)
- GLM-4.5 (full model)
- GLM-4.7 (latest)
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Sequence

from openai import AsyncOpenAI
from openai import Stream as AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from backend.core.logging import get_logger
from backend.core.config import settings
from backend.services.llm.models import (
    LLMOptions,
    LLMResponse,
    LLMValidationError,
    LLMProcessingError,
    LLMConnectionError,
    LLMModelNotFoundError,
)

logger = get_logger(__name__)


class GLMClient:
    """
    GLM client using OpenAI-compatible API

    This client provides async chat completions using GLM models via
    the international Z.ai platform (https://api.z.ai/).

    Example:
        ```python
        from backend.services.llm.glm_client import GLMClient

        client = GLMClient()
        await client.initialize()

        response = await client.chat(
            messages=[{"role": "user", "content": "Hello!"}],
            model="GLM-4.5-Air"
        )

        print(response.content)
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize GLM client

        Args:
            api_key: GLM API key (uses settings.GLM_API_KEY if None)
            base_url: GLM API base URL (uses settings.GLM_BASE_URL if None)
            model: Default model name (uses settings.GLM_MODEL if None)
        """
        self.api_key = api_key or settings.GLM_API_KEY
        self.base_url = base_url or settings.GLM_BASE_URL
        self.model = model or settings.GLM_MODEL
        self._client: Optional[AsyncOpenAI] = None
        self._is_initialized = False

        logger.debug(
            f"GLM client created: model={self.model}, base_url={self.base_url}"
        )

    async def initialize(self) -> None:
        """
        Initialize the GLM client and validate connection

        Raises:
            LLMConnectionError: If connection fails
            LLMModelNotFoundError: If model is not available
        """
        if self._is_initialized:
            logger.debug("GLM client already initialized")
            return

        try:
            logger.info("Initializing GLM client...")

            # Create OpenAI client for GLM
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

            # Test connection with a simple request
            test_start = time.time()
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=10,
            )
            test_time = (time.time() - test_start) * 1000

            logger.info(
                f"GLM client initialized successfully: "
                f"model={self.model}, "
                f"response_time={test_time:.0f}ms"
            )

            self._is_initialized = True

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to initialize GLM client: {error_msg}")

            # Parse error type
            if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                raise LLMConnectionError(
                    f"GLM API authentication failed: {error_msg}",
                    details={"base_url": self.base_url, "error_type": "auth"},
                )
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                raise LLMModelNotFoundError(
                    self.model,
                    details={"base_url": self.base_url, "error": error_msg},
                )
            else:
                raise LLMConnectionError(
                    f"Failed to connect to GLM API: {error_msg}",
                    details={"base_url": self.base_url, "error_type": type(e).__name__},
                )

    async def shutdown(self) -> None:
        """Shutdown the GLM client"""
        if self._client:
            await self._client.close()
            self._client = None

        self._is_initialized = False
        logger.info("GLM client shutdown successfully")

    async def chat(
        self,
        messages: Sequence[Dict[str, str]],
        model: Optional[str] = None,
        options: Optional[LLMOptions] = None,
    ) -> LLMResponse:
        """
        Chat completion using GLM

        Args:
            messages: List of chat messages with role and content
            model: Model name (uses default if None)
            options: Generation options

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            LLMValidationError: If input validation fails
            LLMConnectionError: If connection fails
            LLMModelNotFoundError: If model not found
            LLMProcessingError: If generation fails
        """
        if not self._is_initialized:
            await self.initialize()

        if not messages:
            raise LLMValidationError("Messages list cannot be empty")

        model = model or self.model
        options = options or LLMOptions()

        # Debug: Log message types
        logger.info(f"GLM chat: received {len(messages)} messages of type {type(messages[0])}")
        for i, msg in enumerate(messages):
            logger.info(f"  Message {i}: type={type(msg)}, has_role={hasattr(msg, 'role')}, has_content={hasattr(msg, 'content')}")

        # Build request parameters
        request_params = {
            "model": model,
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ],
        }

        # Add optional parameters (GLM/OpenAI-compatible API doesn't support top_k)
        if options.temperature is not None:
            request_params["temperature"] = options.temperature
        if options.top_p is not None:
            request_params["top_p"] = options.top_p
        # Note: top_k is not supported by OpenAI-compatible API, skip it
        if options.num_predict is not None:
            request_params["max_tokens"] = options.num_predict
        if options.stop is not None:
            request_params["stop"] = options.stop

        logger.debug(
            f"GLM chat request: model={model}, "
            f"messages={len(messages)}, "
            f"temperature={options.temperature}"
        )

        try:
            start_time = time.time()

            # Make API call
            response: ChatCompletion = await self._client.chat.completions.create(
                **request_params
            )

            processing_time = (time.time() - start_time) * 1000

            # Extract response data
            choice = response.choices[0]
            content = choice.message.content or ""
            finish_reason = choice.finish_reason or "stop"

            # Get token usage
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0

            logger.info(
                f"GLM chat response: model={model}, "
                f"tokens={total_tokens}, "
                f"time={processing_time:.0f}ms"
            )

            return LLMResponse(
                content=content,
                model=model,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"GLM chat failed: {error_msg}")

            # Parse error type
            if "model" in error_msg.lower() and "not found" in error_msg.lower():
                raise LLMModelNotFoundError(
                    model,
                    details={"error": error_msg},
                )
            elif "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                raise LLMConnectionError(
                    f"GLM API authentication failed: {error_msg}",
                    details={"error_type": "auth"},
                )
            else:
                raise LLMProcessingError(
                    f"GLM chat completion failed: {error_msg}",
                    details={"model": model, "error_type": type(e).__name__},
                )

    async def chat_stream(
        self,
        messages: Sequence[Dict[str, str]],
        model: Optional[str] = None,
        options: Optional[LLMOptions] = None,
    ):
        """
        Streaming chat completion (not yet implemented)

        Args:
            messages: List of chat messages
            model: Model name
            options: Generation options

        Yields:
            StreamingResponse chunks

        Raises:
            LLMProcessingError: Streaming not yet implemented
        """
        raise LLMProcessingError(
            "Streaming not yet implemented for GLM client",
            details={"feature": "streaming", "status": "not_implemented"},
        )

    async def health_check(self) -> bool:
        """
        Check if GLM API is accessible

        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self._is_initialized:
                await self.initialize()

            # Quick test request
            await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return True

        except Exception as e:
            logger.error(f"GLM health check failed: {e}")
            return False
