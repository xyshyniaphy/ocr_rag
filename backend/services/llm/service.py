"""
LLM Service
Main service for LLM generation using GLM or Ollama

This service provides:
- Chat completions with conversation history
- Text completions for simple prompts
- RAG-augmented generation with retrieved contexts
- Streaming responses
- Model management and health checks

Supported LLM providers:
- GLM-4.5 Air via Z.ai international API (default, recommended)
- Ollama (local models, fallback)
"""

import time
from typing import List, Optional, Dict, Any, Sequence, Union

from backend.core.logging import get_logger
from backend.core.config import settings
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
from backend.services.llm.ollama_client import OllamaClient
from backend.services.llm.glm_client import GLMClient

logger = get_logger(__name__)

# Default RAG prompt template
DEFAULT_RAG_SYSTEM_PROMPT = """あなたは日本語のドキュメントに基づいて質問に答えるAIアシスタントです。

以下のガイドラインに従って回答してください：
1. 提供されたコンテキスト情報のみに基づいて回答してください
2. コンテキストに情報がない場合は「情報が不足しているため回答できません」と明確に述べてください
3. 答えを知らない場合は、決して作り話をしないでください
4. 回答は簡潔かつ正確に、日本語で提供してください
5. 必要に応じて、参照したソースを明示してください

コンテキスト情報：
{context}

ユーザーの質問に適切に答えてください。"""


class LLMService:
    """
    Main LLM service for text generation

    This service manages both GLM and Ollama clients and provides a
    unified API for LLM generation tasks including chat, completion, and RAG.

    Provider Selection:
    - Uses GLM-4.5 Air by default (fast, cost-effective)
    - Falls back to Ollama if GLM is unavailable
    - Provider can be specified via LLM_PROVIDER env variable

    Example:
        ```python
        from backend.services.llm.service import get_llm_service

        # Get service instance
        service = await get_llm_service()

        # Chat completion
        response = await service.chat(
            messages=[
                {"role": "user", "content": "こんにちは！"}
            ]
        )

        print(response.content)
        ```
    """

    def __init__(
        self,
        client: Optional[Union[GLMClient, OllamaClient]] = None,
        model: Optional[str] = None,
        default_options: Optional[LLMOptions] = None,
        provider: Optional[str] = None,
    ):
        """
        Initialize the LLM service

        Args:
            client: GLMClient or OllamaClient instance (created automatically if None)
            model: Default model name (uses config default if None)
            default_options: Default generation options
            provider: LLM provider - "glm" or "ollama" (uses config default if None)
        """
        provider = provider or settings.LLM_PROVIDER

        # Create client based on provider
        if client is None:
            if provider == "glm":
                self.client = GLMClient()
                self.model = model or settings.GLM_MODEL
            elif provider == "ollama":
                self.client = OllamaClient()
                self.model = model or settings.OLLAMA_MODEL
            else:
                logger.warning(f"Unknown provider '{provider}', defaulting to GLM")
                self.client = GLMClient()
                self.model = model or settings.GLM_MODEL
        else:
            self.client = client
            self.model = model or settings.GLM_MODEL if provider == "glm" else settings.OLLAMA_MODEL

        # Set default options based on provider
        if default_options:
            self.default_options = default_options
        elif provider == "glm":
            self.default_options = LLMOptions(
                temperature=settings.GLM_TEMPERATURE if hasattr(settings, 'GLM_TEMPERATURE') else 0.1,
                stream=False,
            )
        else:  # ollama
            self.default_options = LLMOptions(
                temperature=settings.OLLAMA_TEMPERATURE,
                num_ctx=settings.OLLAMA_NUM_CTX,
                stream=False,
            )

        self.provider = provider
        self._is_initialized = False

        logger.info(
            f"LLMService initialized (provider={provider}, model={self.model}, "
            f"temperature={self.default_options.temperature})"
        )

    async def initialize(self) -> None:
        """
        Initialize the LLM service

        Checks model availability and connection to Ollama.
        This should be called before using the service, or it will be
        called automatically on the first request.
        """
        if self._is_initialized:
            logger.debug("LLMService already initialized")
            return

        try:
            start_time = time.time()

            # Check connection
            health = await self.client.health_check()

            # Handle different health check return types based on provider
            if self.provider == "glm":
                # GLM returns bool
                if not health:
                    raise LLMConnectionError(
                        "GLM API is not accessible",
                        details={"health_check_result": health},
                    )
            else:  # ollama
                # Ollama returns dict with "status" key
                if health["status"] != "healthy":
                    raise LLMConnectionError(
                        "Ollama service is not healthy",
                        details=health,
                    )

            # Check model availability (Ollama only - GLM doesn't need this)
            if self.provider != "glm":
                if not await self.client.check_model(self.model):
                    logger.warning(f"Model '{self.model}' not available, attempting to pull...")
                    await self.client.pull_model(self.model)

            self._is_initialized = True

            init_time = int((time.time() - start_time) * 1000)
            logger.info(f"LLMService initialized in {init_time}ms (model={self.model})")

        except LLMConnectionError:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize LLMService: {e}")
            raise LLMProcessingError(
                f"Failed to initialize LLM service: {str(e)}",
                details={"error_type": type(e).__name__},
            )

    async def chat(
        self,
        messages: Sequence[Message],
        options: Optional[LLMOptions] = None,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """
        Chat completion with conversation history

        Args:
            messages: List of chat messages
            options: Generation options (uses service default if None)
            model: Model name (uses service default if None)

        Returns:
            LLMResponse with generated content

        Raises:
            LLMValidationError: If input validation fails
            LLMProcessingError: If generation fails
        """
        if not self._is_initialized:
            await self.initialize()

        model = model or self.model
        options = options or self.default_options

        # Validate messages
        message_list = list(messages)
        if not message_list:
            raise LLMValidationError("Messages list cannot be empty")

        # Convert to Message objects if needed
        validated_messages = []
        for msg in message_list:
            if isinstance(msg, dict):
                validated_messages.append(Message(**msg))
            elif isinstance(msg, Message):
                validated_messages.append(msg)
            else:
                raise LLMValidationError(f"Invalid message type: {type(msg)}")

        logger.info(f"Chat request: model={model}, messages={len(validated_messages)}")

        try:
            response = await self.client.chat(
                messages=validated_messages,
                model=model,
                options=options,
            )

            logger.info(
                f"Chat response: model={model}, "
                f"tokens={response.total_tokens}, "
                f"time={response.processing_time_ms:.0f}ms"
            )

            return response

        except (LLMConnectionError, LLMModelNotFoundError, LLMProcessingError):
            raise
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            raise LLMProcessingError(
                f"Chat generation failed: {str(e)}",
                details={"model": model, "error_type": type(e).__name__},
            )

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        options: Optional[LLMOptions] = None,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """
        Text completion (simpler API for single prompt)

        Args:
            prompt: Text prompt
            system_prompt: Optional system prompt
            options: Generation options (uses service default if None)
            model: Model name (uses service default if None)

        Returns:
            LLMResponse with generated content
        """
        if not self._is_initialized:
            await self.initialize()

        model = model or self.model
        options = options or self.default_options

        logger.info(f"Completion request: model={model}, prompt_length={len(prompt)}")

        try:
            response = await self.client.generate(
                prompt=prompt,
                model=model,
                options=options,
                system=system_prompt,
            )

            logger.info(
                f"Completion response: model={model}, "
                f"tokens={response.total_tokens}, "
                f"time={response.processing_time_ms:.0f}ms"
            )

            return response

        except (LLMConnectionError, LLMModelNotFoundError, LLMProcessingError):
            raise
        except Exception as e:
            logger.error(f"Completion failed: {e}")
            raise LLMProcessingError(
                f"Text completion failed: {str(e)}",
                details={"model": model, "error_type": type(e).__name__},
            )

    async def generate_rag(
        self,
        query: str,
        contexts: Sequence[RAGContext],
        system_prompt: Optional[str] = None,
        options: Optional[LLMOptions] = None,
        model: Optional[str] = None,
    ) -> RAGResponse:
        """
        RAG-augmented generation with retrieved contexts

        Args:
            query: User query
            contexts: Retrieved context documents
            system_prompt: Optional system prompt (uses default RAG prompt if None)
            options: Generation options (uses service default if None)
            model: Model name (uses service default if None)

        Returns:
            RAGResponse with answer and source information
        """
        if not self._is_initialized:
            await self.initialize()

        if not query:
            raise LLMValidationError("Query cannot be empty")

        if not contexts:
            raise LLMValidationError("Contexts list cannot be empty")

        start_time = time.time()
        model = model or self.model
        options = options or self.default_options

        # Build context from retrieved documents
        context_texts = []
        sources = []

        for i, ctx in enumerate(contexts):
            # Convert to RAGContext if needed
            if isinstance(ctx, dict):
                ctx = RAGContext(**ctx)

            context_texts.append(f"[ドキュメント {i+1}] {ctx.text}")
            sources.append({
                "doc_id": ctx.doc_id,
                "score": ctx.score,
                "text": ctx.text[:200] + "..." if len(ctx.text) > 200 else ctx.text,
                "metadata": ctx.metadata,
            })

        # Sort sources by relevance score
        sources.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Build context block
        context_block = "\n\n".join(context_texts)

        # Use default or custom system prompt
        if system_prompt is None:
            system_prompt = DEFAULT_RAG_SYSTEM_PROMPT.format(context=context_block)
        else:
            # Replace {context} placeholder if present
            system_prompt = system_prompt.format(context=context_block)

        # Build messages
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=query),
        ]

        # Debug: Log messages
        logger.info(f"Built {len(messages)} messages for RAG generation")
        for i, msg in enumerate(messages):
            logger.info(f"  Message {i}: type={type(msg).__name__}, role={msg.role if hasattr(msg, 'role') else 'N/A'}")

        logger.info(
            f"RAG generation: model={model}, "
            f"contexts={len(contexts)}, query_length={len(query)}"
        )

        try:
            # Generate response
            logger.info(f"About to call self.client.chat(), client type: {type(self.client)}, provider: {self.provider}")
            llm_response = await self.client.chat(
                messages=messages,
                model=model,
                options=options,
            )

            processing_time = (time.time() - start_time) * 1000

            logger.info(
                f"RAG response: model={model}, "
                f"tokens={llm_response.total_tokens}, "
                f"time={processing_time:.0f}ms"
            )

            return RAGResponse(
                query=query,
                answer=llm_response.content,
                sources=sources[:5],  # Top 5 sources
                model=model,
                processing_time_ms=round(processing_time, 2),
            )

        except (LLMConnectionError, LLMModelNotFoundError, LLMProcessingError):
            raise
        except Exception as e:
            logger.error(f"RAG generation failed: {e}")
            raise LLMProcessingError(
                f"RAG generation failed: {str(e)}",
                details={
                    "model": model,
                    "num_contexts": len(contexts),
                    "error_type": type(e).__name__,
                },
            )

    async def chat_stream(
        self,
        messages: Sequence[Message],
        options: Optional[LLMOptions] = None,
        model: Optional[str] = None,
    ):
        """
        Streaming chat completion

        Args:
            messages: List of chat messages
            options: Generation options
            model: Model name

        Yields:
            Text chunks as they are generated
        """
        if not self._is_initialized:
            await self.initialize()

        model = model or self.model
        options = options or self.default_options

        # Validate and convert messages
        validated_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                validated_messages.append(Message(**msg))
            elif isinstance(msg, Message):
                validated_messages.append(msg)
            else:
                raise LLMValidationError(f"Invalid message type: {type(msg)}")

        async for chunk in self.client.chat_stream(
            messages=validated_messages,
            model=model,
            options=options,
        ):
            yield chunk

    @property
    def is_initialized(self) -> bool:
        """Check if the service is initialized"""
        return self._is_initialized

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the LLM service

        Returns:
            Dictionary with health status and metrics
        """
        health = {
            "status": "healthy" if self._is_initialized else "uninitialized",
            "model": self.model,
            "default_options": self.default_options.model_dump(),
        }

        if self._is_initialized:
            # Check Ollama connection
            try:
                ollama_health = await self.client.health_check()
                health["ollama_status"] = ollama_health["status"]
                health["available_models"] = ollama_health.get("models_available", [])

                # Test generation
                start_time = time.time()
                test_response = await self.chat(
                    messages=[Message(role="user", content="テスト")],
                    options=LLMOptions(temperature=0.0, num_predict=10),
                )
                health["test_generation_time_ms"] = round((time.time() - start_time) * 1000, 2)
                health["test_response_length"] = len(test_response.content)

            except Exception as e:
                health["status"] = "error"
                health["error"] = str(e)

        return health

    async def shutdown(self) -> None:
        """
        Shutdown the LLM service and free resources
        """
        if self._is_initialized:
            await self.client.close()
            self._is_initialized = False
            logger.info("LLMService shutdown complete")


# Global singleton instance
_llm_service: Optional[LLMService] = None


async def get_llm_service() -> LLMService:
    """
    Get or create the global LLM service instance

    Returns:
        LLMService singleton
    """
    global _llm_service

    if _llm_service is None:
        default_options = LLMOptions(
            temperature=settings.OLLAMA_TEMPERATURE,
            top_p=0.9,
            top_k=40,
            num_ctx=settings.OLLAMA_NUM_CTX,
            num_predict=2048,
            repeat_penalty=1.1,
            stream=False,
        )
        _llm_service = LLMService(default_options=default_options)
        await _llm_service.initialize()

    return _llm_service


# Convenience export
llm_service: LLMService = None  # Will be initialized by get_llm_service()
