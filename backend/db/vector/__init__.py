"""
Milvus Vector Database Module
Provides vector database operations for embeddings
"""

from backend.db.vector.client import (
    get_milvus_client,
    init_milvus,
    close_milvus,
    get_collection,
    insert_chunks,
    search_similar,
    delete_chunks,
    delete_by_document,
    get_collection_stats,
    health_check,
    INDEX_TYPE,
    METRIC_TYPE,
    INDEX_PARAMS,
    SEARCH_PARAMS,
)

__all__ = [
    "get_milvus_client",
    "init_milvus",
    "close_milvus",
    "get_collection",
    "insert_chunks",
    "search_similar",
    "delete_chunks",
    "delete_by_document",
    "get_collection_stats",
    "health_check",
    "INDEX_TYPE",
    "METRIC_TYPE",
    "INDEX_PARAMS",
    "SEARCH_PARAMS",
]
