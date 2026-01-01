"""
Llama-3.2-NV Reranker Model
NVIDIA's Llama-3.2-NV-RerankQA-1B-v2 cross-encoder for document reranking
https://huggingface.co/nvidia/Llama-3.2-NV-RerankQA-1B-v2

Llama-3.2-NV-RerankQA-1B-v2 is optimized for:
- Question-answering retrieval reranking
- Japanese and multilingual support
- GPU-accelerated inference (optimized for NVIDIA GPUs)
- 1B parameter model for fast inference
"""

import asyncio
import os
import time
from typing import List, Optional, Tuple

from backend.core.logging import get_logger
from backend.core.config import settings
from backend.services.reranker.models import (
    RerankerOptions,
    RerankingProcessingError,
)

logger = get_logger(__name__)


class LlamaNVRerankerModel:
    """
    Llama-3.2-NV Reranker Model implementation

    NVIDIA's Llama-3.2-NV-RerankQA-1B-v2 is a cross-encoder model
    specifically designed for reranking retrieval results.

    Model details:
    - Model: nvidia/Llama-3.2-NV-RerankQA-1B-v2
    - Parameters: 1B
    - Architecture: Cross-encoder (BERT-like)
    - Max Length: 512 tokens per query-document pair
    - Languages: Multilingual (including Japanese)
    """

    # Model configuration
    MODEL_NAME = "nvidia/Llama-3.2-NV-RerankQA-1B-v2"
    MAX_LENGTH = 512
    BATCH_SIZE = 32

    # Local model path (cached via volume mount, NOT in base image)
    MODEL_PATH = "/app/reranker_models/llama-nv-reranker"
    CACHE_DIR = "/app/reranker_models/huggingface_cache"

    def __init__(
        self,
        options: Optional[RerankerOptions] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize Llama-3.2-NV Reranker model

        Args:
            options: Reranker options
            device: Device to use (cuda:0, cpu, etc.). Uses config default if None
        """
        self.options = options or RerankerOptions()
        self.device = device or settings.RERANKER_DEVICE
        self._model = None
        self._tokenizer = None
        self._is_loaded = False

        logger.info(
            f"LlamaNVRerankerModel initialized with device={self.device}, "
            f"batch_size={self.options.batch_size}, "
            f"threshold={self.options.threshold}"
        )

    @property
    def model_name(self) -> str:
        """Return the model name"""
        return self.MODEL_NAME

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded"""
        return self._is_loaded

    async def load_model(self) -> None:
        """
        Load the Llama-3.2-NV Reranker model into memory

        CRITICAL: Model MUST be pre-downloaded in Docker base image.
        NO HuggingFace Hub fallback allowed.
        """
        if self._is_loaded:
            logger.debug("Llama-3.2-NV Reranker model already loaded")
            return

        try:
            # Set environment variable for protobuf compatibility
            os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            start_time = time.time()

            # CRITICAL: Use local path ONLY - NO HuggingFace Hub fallback
            # All models MUST be pre-downloaded in Docker base image
            model_path = self.MODEL_PATH

            # Check if local model exists
            if not os.path.exists(model_path) or not os.listdir(model_path):
                raise RerankingProcessingError(
                    f"Local model not found at {model_path}. "
                    f"All models MUST be pre-downloaded in Docker base image. "
                    f"Rebuild the base image: ./dev.sh rebuild base",
                    details={
                        "model_path": model_path,
                        "expected_location": "Docker base image",
                        "fix": "Rebuild base image with model pre-downloaded",
                    },
                )

            # Load tokenizer and model from local path ONLY
            # No cache_dir needed - model is already in base image
            logger.info(f"Loading Llama-3.2-NV Reranker from local path: {model_path}...")

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=None,  # Disable cache - use local files only
                trust_remote_code=True,
                local_files_only=True,  # CRITICAL: Enforce local files only
            )

            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                cache_dir=None,  # Disable cache - use local files only
                trust_remote_code=True,
                local_files_only=True,  # CRITICAL: Enforce local files only
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            )

            # Move model to device
            self._model.to(self.device)
            self._model.eval()

            self._is_loaded = True

            load_time = int((time.time() - start_time) * 1000)
            logger.info(
                f"Llama-3.2-NV Reranker model loaded successfully in {load_time}ms "
                f"(device={self.device})"
            )

        except ImportError as e:
            raise RerankingProcessingError(
                "transformers library not installed",
                details={"install": "pip install transformers"},
            )
        except Exception as e:
            raise RerankingProcessingError(
                f"Failed to load Llama-3.2-NV Reranker model: {str(e)}",
                details={
                    "model_path": self.MODEL_PATH,
                    "device": self.device,
                    "error_type": type(e).__name__,
                },
            )

    async def rerank(
        self,
        query: str,
        documents: List[str],
        threshold: Optional[float] = None,
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents for a query

        Args:
            query: Query text
            documents: List of document texts
            threshold: Minimum relevance score (uses options default if None)

        Returns:
            List of (index, score) tuples sorted by relevance (descending)

        Raises:
            RerankingProcessingError: If reranking fails
        """
        if not self._is_loaded:
            await self.load_model()

        if not documents:
            return []

        try:
            import torch

            threshold = threshold if threshold is not None else self.options.threshold
            batch_size = self.options.batch_size

            # Prepare query-document pairs
            pairs = [[query, doc] for doc in documents]

            # Compute scores in batches
            all_scores = []

            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]

                # Tokenize
                inputs = self._tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.MAX_LENGTH,
                    return_tensors="pt",
                )

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Compute scores
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    # Use sigmoid to get probabilities [0, 1]
                    scores = torch.sigmoid(outputs.logits.squeeze(-1))

                all_scores.extend(scores.cpu().tolist())

            # Filter by threshold and sort by score
            filtered_indices_scores = [
                (idx, score)
                for idx, score in enumerate(all_scores)
                if score >= threshold
            ]

            # Sort by score descending
            filtered_indices_scores.sort(key=lambda x: x[1], reverse=True)

            logger.debug(
                f"Reranked {len(documents)} documents, "
                f"{len(filtered_indices_scores)} passed threshold {threshold}"
            )

            return filtered_indices_scores

        except Exception as e:
            logger.error(f"Llama-3.2-NV reranking failed: {e}")
            raise RerankingProcessingError(
                f"Failed to rerank documents: {str(e)}",
                details={
                    "num_documents": len(documents),
                    "batch_size": batch_size,
                    "error_type": type(e).__name__,
                },
            )

    async def unload(self) -> None:
        """
        Unload the model and free GPU memory
        """
        if self._model is not None:
            try:
                import torch

                # Clear CUDA cache if using GPU
                if "cuda" in self.device:
                    del self._model
                    del self._tokenizer
                    torch.cuda.empty_cache()
                    logger.debug(f"Cleared CUDA cache for {self.device}")
            except Exception as e:
                logger.warning(f"Error clearing CUDA cache: {e}")

        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        logger.info("Llama-3.2-NV Reranker model unloaded")

    @property
    def version(self) -> str:
        """Return the model version"""
        try:
            from transformers import __version__ as transformers_version
            return f"transformers-{transformers_version}"
        except (ImportError, AttributeError):
            return "unknown"
