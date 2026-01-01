#!/usr/bin/env python3
"""
Download Reranker Model Script

Pre-downloads the Llama-3.2-NV-RerankQA-1B-v2 model from HuggingFace Hub
to the local volume. This avoids the 3.7 minute download delay on first use.

Usage:
    python scripts/download_reranker_model.py

The model will be downloaded to:
    /app/reranker_models/llama-nv-reranker/

This script can be run from within the app container:
    docker exec ocr-rag-app-dev python /app/scripts/download_reranker_model.py
"""

import os
import sys
import time
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, "/app")


def download_reranker_model():
    """Download the reranker model to local volume"""
    try:
        from huggingface_hub import snapshot_download
        from backend.core.logging import get_logger

        logger = get_logger(__name__)

        model_name = "nvidia/Llama-3.2-NV-RerankQA-1B-v2"
        local_path = "/app/reranker_models/llama-nv-reranker"

        logger.info(f"Starting download of {model_name}...")
        logger.info(f"Target directory: {local_path}")

        # Check if model already exists
        if os.path.exists(local_path) and os.listdir(local_path):
            logger.info(
                f"Model already exists at {local_path}. "
                f"Skipping download."
            )
            return True

        # Create directory if it doesn't exist
        os.makedirs(local_path, exist_ok=True)

        # Download model
        start_time = time.time()

        snapshot_download(
            repo_id=model_name,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            # Use cache directory
            cache_dir="/app/reranker_models/huggingface_cache",
        )

        download_time = time.time() - start_time

        # Verify download
        if not os.path.exists(local_path) or not os.listdir(local_path):
            logger.error(f"Download failed: {local_path} is empty")
            return False

        # Calculate size
        total_size = sum(
            f.stat().st_size
            for f in Path(local_path).rglob('*')
            if f.is_file()
        )
        size_mb = total_size / (1024 * 1024)

        logger.info(
            f"✓ Download complete: {model_name}\n"
            f"  Location: {local_path}\n"
            f"  Size: {size_mb:.1f} MB\n"
            f"  Time: {download_time:.1f} seconds\n"
            f"  Files: {len(list(Path(local_path).rglob('*')))}"
        )

        return True

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def main():
    """Main entry point"""
    print("=" * 80)
    print("Reranker Model Download Script")
    print("=" * 80)
    print()

    success = download_reranker_model()

    print()
    if success:
        print("✓ SUCCESS: Reranker model downloaded successfully")
        print()
        print("Next steps:")
        print("  - Restart the app container: ./dev.sh restart app")
        print("  - Reranker will load from local path (~6 seconds instead of ~3.7 minutes)")
    else:
        print("✗ FAILED: Could not download reranker model")
        print()
        print("Troubleshooting:")
        print("  - Check internet connection")
        print("  - Verify HuggingFace Hub is accessible")
        print("  - Check disk space (requires ~1GB)")

    print("=" * 80)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
