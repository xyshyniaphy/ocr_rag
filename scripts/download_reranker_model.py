#!/usr/bin/env python3
"""
Download Reranker Model Script
Downloads the Llama-3.2-NV-RerankQA-1B-v2 model to the local cache

Usage:
    python scripts/download_reranker_model.py
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from backend.core.logging import get_logger

logger = get_logger(__name__)


async def download_reranker_model():
    """Download the reranker model"""

    # Model configuration
    MODEL_NAME = "nvidia/Llama-3.2-NV-RerankQA-1B-v2"
    CACHE_DIR = "/app/reranker_models/huggingface_cache"
    MODEL_PATH = "/app/reranker_models/llama-nv-reranker"

    logger.info(f"Downloading reranker model: {MODEL_NAME}")
    logger.info(f"Cache directory: {CACHE_DIR}")
    logger.info(f"Model path: {MODEL_PATH}")

    # Create directories
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        logger.info("Starting download... (this may take a while)")

        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )

        # Download model
        logger.info("Downloading model weights...")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # Save to local path for faster loading
        logger.info(f"Saving model to {MODEL_PATH}...")
        tokenizer.save_pretrained(MODEL_PATH)
        model.save_pretrained(MODEL_PATH)

        logger.info("✓ Model downloaded and saved successfully!")
        logger.info(f"Model location: {MODEL_PATH}")
        logger.info(f"Cache location: {CACHE_DIR}")

        # Show disk usage
        import shutil
        size_mb = sum(
            f.stat().st_size for f in Path(MODEL_PATH).rglob('*') if f.is_file()
        ) / (1024 * 1024)
        logger.info(f"Model size: {size_mb:.1f} MB")

    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.info("Install with: pip install transformers torch")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(download_reranker_model())
