"""
Milvus Vector Database Client
Vector database operations for embeddings
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from pymilvus.exceptions import MilvusException

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.core.exceptions import EmbeddingException, RetrievalException

logger = get_logger(__name__)

# Global Milvus client
_client: Optional[connections] = None

# Milvus index configuration
INDEX_TYPE = "HNSW"  # Hierarchical Navigable Small World - best for semantic search
METRIC_TYPE = "COSINE"  # Cosine similarity for normalized embeddings
INDEX_PARAMS = {
    "M": 16,  # Number of bi-directional links for each node
    "efConstruction": 256,  # Size of dynamic candidate list for construction
}
SEARCH_PARAMS = {
    "metric_type": METRIC_TYPE,
    "params": {"ef": 64},  # Size of dynamic candidate list for search
}


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
            await _create_collection()
        else:
            logger.info(f"Milvus collection exists: {settings.MILVUS_COLLECTION}")
            # Ensure index exists
            await _ensure_index()

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


async def _create_collection() -> Collection:
    """Create the document chunks collection with HNSW index"""
    loop = asyncio.get_event_loop()

    def _create():
        # Define fields
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIMENSION),
            FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="page_number", dtype=DataType.INT32),
            FieldSchema(name="chunk_index", dtype=DataType.INT32),
            FieldSchema(name="metadata", dtype=DataType.JSON),
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

        # Create HNSW index for fast approximate search
        index_params = {
            "index_type": INDEX_TYPE,
            "metric_type": METRIC_TYPE,
            "params": INDEX_PARAMS,
        }

        collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )

        logger.info(f"Collection {settings.MILVUS_COLLECTION} created with HNSW index")
        return collection

    return await loop.run_in_executor(None, _create)


async def _ensure_index() -> None:
    """Ensure HNSW index exists on collection"""
    loop = asyncio.get_event_loop()

    def _check():
        try:
            collection = Collection(settings.MILVUS_COLLECTION)
            indexes = collection.indexes

            if not indexes:
                logger.info("No index found, creating HNSW index...")
                index_params = {
                    "index_type": INDEX_TYPE,
                    "metric_type": METRIC_TYPE,
                    "params": INDEX_PARAMS,
                }
                collection.create_index(
                    field_name="embedding",
                    index_params=index_params,
                )
                logger.info("HNSW index created successfully")
            else:
                logger.debug(f"Index exists: {indexes[0].params}")
        except Exception as e:
            logger.warning(f"Error checking index: {e}")

    await loop.run_in_executor(None, _check)


def get_collection() -> Collection:
    """Get the document chunks collection"""
    if not utility.has_collection(settings.MILVUS_COLLECTION):
        raise RetrievalException(
            message=f"Collection {settings.MILVUS_COLLECTION} does not exist"
        )

    collection = Collection(settings.MILVUS_COLLECTION)
    collection.load()
    return collection


async def insert_chunks(
    chunk_ids: List[str],
    embeddings: List[List[float]],
    texts: List[str],
    document_ids: List[str],
    page_numbers: List[int],
    chunk_indices: List[int],
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """
    Insert chunks with embeddings into Milvus

    Args:
        chunk_ids: List of chunk IDs
        embeddings: List of embedding vectors
        texts: List of text content
        document_ids: List of document IDs
        page_numbers: List of page numbers
        chunk_indices: List of chunk indices
        metadata: Optional list of metadata dictionaries

    Returns:
        Number of chunks inserted
    """
    loop = asyncio.get_event_loop()

    def _insert():
        try:
            collection = get_collection()

            # Prepare data
            data = [
                chunk_ids,
                embeddings,
                texts,
                document_ids,
                page_numbers,
                chunk_indices,
                metadata or [{}] * len(chunk_ids),
            ]

            # Insert data
            collection.insert(data)

            # Flush to ensure data is persisted
            collection.flush()

            logger.info(f"Inserted {len(chunk_ids)} chunks into Milvus")
            return len(chunk_ids)

        except MilvusException as e:
            logger.error(f"Failed to insert chunks: {e}")
            raise RetrievalException(
                message="Failed to insert chunks into vector database",
                details={"error": str(e), "count": len(chunk_ids)},
            )

    return await loop.run_in_executor(None, _insert)


async def search_similar(
    query_embedding: List[float],
    top_k: int = settings.SEARCH_TOP_K,
    document_ids: Optional[List[str]] = None,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Search for similar chunks by vector similarity

    Args:
        query_embedding: Query embedding vector
        top_k: Number of results to return
        document_ids: Optional list of document IDs to filter by
        min_score: Minimum similarity score threshold

    Returns:
        List of similar chunks with scores
    """
    loop = asyncio.get_event_loop()

    def _search():
        try:
            collection = get_collection()

            # Build search expression
            expr = None
            if document_ids:
                # Filter by document IDs
                doc_ids_str = ", ".join([f'"{did}"' for did in document_ids])
                expr = f"document_id in [{doc_ids_str}]"

            # Search
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=SEARCH_PARAMS,
                limit=top_k,
                expr=expr,
                output_fields=["chunk_id", "text_content", "document_id", "page_number", "chunk_index", "metadata"],
            )

            # Format results
            chunks = []
            for hit in results[0]:
                if hit.score >= min_score:
                    # Safely extract entity fields (pymilvus Hit.get() doesn't support default value)
                    def get_field(field_name, default=None):
                        try:
                            return hit.entity.get(field_name)
                        except (AttributeError, KeyError):
                            return default

                    chunks.append({
                        "chunk_id": get_field("chunk_id"),
                        "text": get_field("text_content"),
                        "document_id": get_field("document_id"),
                        "page_number": get_field("page_number"),
                        "chunk_index": get_field("chunk_index"),
                        "metadata": get_field("metadata") or {},
                        "score": float(hit.score),
                    })

            logger.debug(f"Found {len(chunks)} chunks (min_score: {min_score})")
            return chunks

        except MilvusException as e:
            logger.error(f"Failed to search Milvus: {e}")
            raise RetrievalException(
                message="Failed to search vector database",
                details={"error": str(e)},
            )

    return await loop.run_in_executor(None, _search)


async def delete_chunks(chunk_ids: List[str]) -> int:
    """
    Delete chunks from Milvus

    Args:
        chunk_ids: List of chunk IDs to delete

    Returns:
        Number of chunks deleted
    """
    loop = asyncio.get_event_loop()

    def _delete():
        try:
            collection = get_collection()

            # Build expression for deletion
            ids_str = ", ".join([f'"{cid}"' for cid in chunk_ids])
            expr = f"chunk_id in [{ids_str}]"

            # Delete
            collection.delete(expr)

            # Flush to ensure deletion is persisted
            collection.flush()

            logger.info(f"Deleted {len(chunk_ids)} chunks from Milvus")
            return len(chunk_ids)

        except MilvusException as e:
            logger.error(f"Failed to delete chunks: {e}")
            raise RetrievalException(
                message="Failed to delete chunks from vector database",
                details={"error": str(e), "count": len(chunk_ids)},
            )

    return await loop.run_in_executor(None, _delete)


async def delete_by_document(document_id: str) -> int:
    """
    Delete all chunks for a document

    Args:
        document_id: Document ID

    Returns:
        Number of chunks deleted
    """
    loop = asyncio.get_event_loop()

    def _delete():
        try:
            collection = get_collection()

            # Build expression for deletion
            expr = f'document_id == "{document_id}"'

            # Delete
            collection.delete(expr)

            # Flush to ensure deletion is persisted
            collection.flush()

            logger.info(f"Deleted all chunks for document {document_id}")
            return 0  # Milvus doesn't return count

        except MilvusException as e:
            logger.error(f"Failed to delete document chunks: {e}")
            raise RetrievalException(
                message="Failed to delete document chunks from vector database",
                details={"error": str(e), "document_id": document_id},
            )

    return await loop.run_in_executor(None, _delete)


async def get_collection_stats() -> Dict[str, Any]:
    """
    Get collection statistics

    Returns:
        Dictionary with collection stats
    """
    loop = asyncio.get_event_loop()

    def _stats():
        try:
            collection = get_collection()

            # Get collection info
            stats = {
                "name": collection.name,
                "count": collection.num_entities,
                "index_type": INDEX_TYPE,
                "metric_type": METRIC_TYPE,
                "dimension": settings.EMBEDDING_DIMENSION,
            }

            return stats

        except MilvusException as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

    return await loop.run_in_executor(None, _stats)


async def health_check() -> bool:
    """
    Check Milvus connection health

    Returns:
        True if healthy, False otherwise
    """
    loop = asyncio.get_event_loop()

    def _check():
        try:
            if not utility.has_collection(settings.MILVUS_COLLECTION):
                return False

            # Try to get collection info
            collection = Collection(settings.MILVUS_COLLECTION)
            collection.load()
            return True

        except Exception as e:
            logger.error(f"Milvus health check failed: {e}")
            return False

    return await loop.run_in_executor(None, _check)
