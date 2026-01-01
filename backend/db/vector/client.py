"""
Milvus Vector Database Client
Vector database operations for embeddings
"""

from typing import Any, Dict, List, Optional

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from pymilvus.exceptions import MilvusException

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.core.exceptions import EmbeddingException

logger = get_logger(__name__)

# Global Milvus client
_client: Optional[connections] = None


def get_milvus_client() -> Optional[connections]:
    """Get Milvus client connection"""
    return _client


async def init_milvus() -> None:
    """Initialize Milvus connection and create collection"""
    global _client

    try:
        logger.info(f"Connecting to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")

        # Connect to Milvus
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
        )
        _client = connections

        # Create collection if it doesn't exist
        if not utility.has_collection(settings.MILVUS_COLLECTION):
            logger.info(f"Creating Milvus collection: {settings.MILVUS_COLLECTION}")
            _create_collection()
        else:
            logger.info(f"Milvus collection exists: {settings.MILVUS_COLLECTION}")

        logger.info("Milvus initialized successfully")

    except MilvusException as e:
        logger.error(f"Failed to initialize Milvus: {e}")
        raise EmbeddingException(
            message="Failed to initialize vector database",
            details={"error": str(e)},
        )


async def close_milvus() -> None:
    """Close Milvus connection"""
    global _client

    try:
        if _client:
            connections.disconnect("default")
            _client = None
            logger.info("Milvus connection closed")
    except Exception as e:
        logger.error(f"Error closing Milvus: {e}")


def _create_collection() -> Collection:
    """Create the document chunks collection"""

    # Define fields
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIMENSION),
        FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="page_number", dtype=DataType.INT32),
        FieldSchema(name="chunk_index", dtype=DataType.INT32),
        FieldSchema(name="metadata", dtype=DataType.JSON),  # Use JSON instead of JSONB for Milvus
    ]

    # Create schema
    schema = CollectionSchema(
        fields=fields,
        description="Document chunks with embeddings for semantic search",
        enable_dynamic_field=True,
    )

    # Create collection
    collection = Collection(
        name=settings.MILVUS_COLLECTION,
        schema=schema,
    )

    # Create index
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",  # Inner Product for L2-normalized vectors
        "params": {"nlist": 1024},
    }

    collection.create_index(
        field_name="embedding",
        index_params=index_params,
    )

    logger.info(f"Collection {settings.MILVUS_COLLECTION} created with index")

    return collection


def get_collection() -> Collection:
    """Get the document chunks collection"""
    if not utility.has_collection(settings.MILVUS_COLLECTION):
        raise EmbeddingException(
            message=f"Collection {settings.MILVUS_COLLECTION} does not exist"
        )

    collection = Collection(settings.MILVUS_COLLECTION)
    collection.load()
    return collection
