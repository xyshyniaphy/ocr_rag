"""
Sarashina Embedding Model
Japanese text embedding model from SBI Intuitions
https://huggingface.co/sbintuitions/sarashina-embedding-v1-1b

Sarashina is a Japanese text embedding model optimized for:
- Semantic similarity search
- Document retrieval
- RAG applications
- 768-dimensional embeddings
- Max 512 token length
"""

import time
from typing import List, Optional

from backend.core.logging import get_logger
from backend.core.config import settings
from backend.services.embedding.base import BaseEmbeddingModel
from backend.services.embedding.models import (
    EmbeddingOptions,
    EmbeddingProcessingError,
)

logger = get_logger(__name__)


class SarashinaEmbeddingModel(BaseEmbeddingModel):
    """
    Sarashina Embedding Model implementation

    Sarashina-Embedding-v1-1B is a Japanese text embedding model
    from SBI Intuitions based on the BERT architecture.

    Model details:
    - Model: sbintuitions/sarashina-embedding-v1-1b
    - Dimension: 768
    - Max Length: 512 tokens
    - Language: Japanese optimized
    """

    # Model configuration
    MODEL_NAME = "sbintuitions/sarashina-embedding-v1-1b"
    DIMENSION = 768
    MAX_LENGTH = 512

    # Local model path (from Docker base image)
    MODEL_PATH = "/app/models/sarashina"

    def __init__(
        self,
        options: Optional[EmbeddingOptions] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize Sarashina embedding model

        Args:
            options: Embedding generation options
            device: Device to use (cuda:0, cpu, etc.). Uses config default if None
        """
        super().__init__(options or EmbeddingOptions())

        self.device = device or settings.EMBEDDING_DEVICE
        self._model = None
        self._tokenizer = None
        self._is_loaded = False

        logger.info(
            f"SarashinaEmbeddingModel initialized with device={self.device}, "
            f"batch_size={self.options.batch_size}"
        )

    @property
    def model_name(self) -> str:
        """Return the model name"""
        return self.MODEL_NAME

    @property
    def dimension(self) -> int:
        """Return the embedding dimension"""
        return self.DIMENSION

    @property
    def max_length(self) -> int:
        """Return the maximum token length"""
        return self.MAX_LENGTH

    async def load_model(self) -> None:
        """
        Load the Sarashina model into memory

        The model files are expected to be in /app/models/sarashina/
        from the Docker base image.
        """
        if self._is_loaded:
            logger.debug("Sarashina model already loaded")
            return

        try:
            from sentence_transformers import SentenceTransformer
            import torch

            start_time = time.time()

            # Try local path first, then HuggingFace Hub
            model_path = self.MODEL_PATH

            # Check if local model exists
            import os
            if not os.path.exists(model_path) or not os.listdir(model_path):
                logger.info(
                    f"Local model not found at {model_path}, "
                    f"using HuggingFace Hub: {self.MODEL_NAME}"
                )
                model_path = self.MODEL_NAME

            # Initialize the model
            self._model = SentenceTransformer(
                model_path,
                device=self.device,
            )

            # Verify model dimension
            test_embedding = self._model.encode("テスト", convert_to_numpy=True)
            if len(test_embedding) != self.DIMENSION:
                raise EmbeddingProcessingError(
                    f"Model output dimension {len(test_embedding)} "
                    f"does not match expected {self.DIMENSION}",
                    details={
                        "model_path": model_path,
                        "expected_dim": self.DIMENSION,
                        "actual_dim": len(test_embedding),
                    },
                )

            self._is_loaded = True

            load_time = int((time.time() - start_time) * 1000)
            logger.info(
                f"Sarashina model loaded successfully in {load_time}ms "
                f"(device={self.device}, dim={self.DIMENSION})"
            )

        except ImportError as e:
            raise EmbeddingProcessingError(
                "sentence-transformers library not installed",
                details={"install": "pip install sentence-transformers"},
            )
        except Exception as e:
            raise EmbeddingProcessingError(
                f"Failed to load Sarashina model: {str(e)}",
                details={
                    "model_path": self.MODEL_PATH,
                    "device": self.device,
                    "error_type": type(e).__name__,
                },
            )

    async def _encode_batch(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> List[List[float]]:
        """
        Encode a batch of texts to embeddings

        Args:
            texts: List of input texts
            normalize: Whether to L2-normalize embeddings

        Returns:
            List of embedding vectors (each as list of floats)

        Raises:
            EmbeddingProcessingError: If encoding fails
        """
        if not self._is_loaded:
            await self.load_model()

        try:
            import numpy as np

            # Encode batch
            # convert_to_numpy=True returns numpy array (more efficient for batching)
            # convert_to_tensor=False to avoid CUDA memory issues
            embeddings = self._model.encode(
                texts,
                batch_size=self.options.batch_size,
                show_progress_bar=self.options.show_progress,
                convert_to_numpy=True,
                convert_to_tensor=False,
                normalize_embeddings=normalize,
            )

            # Convert to list of lists
            if isinstance(embeddings, np.ndarray):
                if embeddings.ndim == 1:
                    # Single embedding
                    return [embeddings.tolist()]
                else:
                    # Batch of embeddings
                    return embeddings.tolist()
            else:
                # Already a list
                return embeddings

        except Exception as e:
            logger.error(f"Sarashina encoding failed: {e}")
            raise EmbeddingProcessingError(
                f"Failed to encode texts: {str(e)}",
                details={
                    "num_texts": len(texts),
                    "batch_size": self.options.batch_size,
                    "error_type": type(e).__name__,
                },
            )

    async def unload(self) -> None:
        """
        Unload the model and free GPU memory

        Sarashina-specific cleanup to ensure GPU memory is freed
        """
        if self._model is not None:
            try:
                import torch

                # Clear CUDA cache if using GPU
                if "cuda" in self.device:
                    torch.cuda.empty_cache()
                    logger.debug(f"Cleared CUDA cache for {self.device}")
            except Exception as e:
                logger.warning(f"Error clearing CUDA cache: {e}")

        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        logger.info("Sarashina model unloaded")

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for Japanese text

        Sarashina uses a Japanese-specific tokenizer, so we estimate
        based on character patterns typical for Japanese BERT models.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Japanese BERT tokenization patterns:
        # - Kanji: ~1 char per token
        # - Hiragana/Katakana: ~2-3 chars per token
        # - ASCII: ~4 chars per token

        kanji = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)
        hiragana = sum(1 for c in text if 0x3040 <= ord(c) <= 0x309F)
        katakana = sum(1 for c in text if 0x30A0 <= ord(c) <= 0x30FF)
        ascii_chars = sum(1 for c in text if ord(c) < 0x80)

        # Rough estimates based on BERT tokenization
        kanji_tokens = kanji * 1.0
        kana_tokens = (hiragana + katakana) * 0.4
        ascii_tokens = ascii_chars * 0.25

        return int(kanji_tokens + kana_tokens + ascii_tokens)

    @property
    def version(self) -> str:
        """Return the model version"""
        try:
            from sentence_transformers import __version__ as st_version
            return f"sentence-transformers-{st_version}"
        except (ImportError, AttributeError):
            return "unknown"
