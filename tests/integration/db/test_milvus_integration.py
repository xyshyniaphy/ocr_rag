#!/usr/bin/env python3
"""
Integration Tests for Milvus Vector Database
Tests for Milvus integration
"""

import pytest
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType


@pytest.mark.integration
@pytest.mark.external
class TestMilvusConnection:
    """Test Milvus database connection"""

    def test_connect_to_milvus(self):
        """Test connecting to Milvus"""
        from backend.core.config import settings

        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT
        )

        # Check if connected
        assert connections.has_connection("default") is True

        # Disconnect
        connections.disconnect("default")

    def test_milvus_server_info(self):
        """Test getting Milvus server info"""
        from backend.core.config import settings

        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT
        )

        # Get server version
        from pymilvus import utility
        version = utility.get_server_version()
        assert version is not None

        connections.disconnect("default")


@pytest.mark.integration
@pytest.mark.external
class TestMilvusCollection:
    """Test Milvus collection operations"""

    @pytest.fixture
    def test_collection(self):
        """Create a test collection"""
        from backend.core.config import settings

        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT
        )

        # Define collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1792),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        ]

        schema = CollectionSchema(fields, description="Test collection")
        collection = Collection(name="test_collection", schema=schema)

        yield collection

        # Cleanup: drop collection
        collection.drop()
        connections.disconnect("default")

    def test_create_collection(self, test_collection):
        """Test creating a collection"""
        assert test_collection is not None
        assert test_collection.name == "test_collection"

    def test_insert_vectors(self, test_collection):
        """Test inserting vectors into collection"""
        # Generate test data
        vectors = np.random.rand(10, 1792).tolist()
        ids = [f"test_id_{i}" for i in range(10)]
        texts = [f"Test text {i}" for i in range(10)]

        # Insert
        data = [ids, vectors, texts]
        test_collection.insert(data)

        # Flush to ensure data is persisted
        test_collection.flush()

        # Check collection has data
        test_collection.load()
        num_entities = test_collection.num_entities
        assert num_entities == 10

    def test_search_vectors(self, test_collection):
        """Test searching vectors"""
        # Insert test data
        vectors = np.random.rand(10, 1792).tolist()
        ids = [f"test_id_{i}" for i in range(10)]
        texts = [f"Test text {i}" for i in range(10)]

        test_collection.insert([ids, vectors, texts])
        test_collection.flush()
        test_collection.load()

        # Search with query vector
        query_vector = np.random.rand(1792).tolist()
        results = test_collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=5,
            output_fields=["text"]
        )

        # Should return results
        assert len(results) == 1  # One query vector
        assert len(results[0]) <= 5  # At most 5 results

    def test_delete_vectors(self, test_collection):
        """Test deleting vectors from collection"""
        # Insert test data
        vectors = np.random.rand(5, 1792).tolist()
        ids = [f"test_id_{i}" for i in range(5)]
        texts = [f"Test text {i}" for i in range(5)]

        test_collection.insert([ids, vectors, texts])
        test_collection.flush()
        test_collection.load()

        # Delete one entity
        test_collection.delete(f"id in ['{ids[0]}']")
        test_collection.flush()

        # Verify deletion
        test_collection.load()
        num_entities = test_collection.num_entities
        assert num_entities == 4  # 5 - 1 = 4


@pytest.mark.integration
@pytest.mark.external
class TestMilvusVectorOperations:
    """Test Milvus vector operations"""

    @pytest.fixture
    def collection_with_index(self):
        """Create collection with index"""
        from backend.core.config import settings

        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT
        )

        # Create collection
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1792),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
        ]

        schema = CollectionSchema(fields, description="Test collection with index")
        collection = Collection(name="test_index_collection", schema=schema)

        # Create index
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="vector", index_params=index_params)

        yield collection

        collection.drop()
        connections.disconnect("default")

    def test_create_index(self, collection_with_index):
        """Test creating index on vector field"""
        indexes = collection_with_index.indexes
        assert len(indexes) > 0
        assert indexes[0].field_name == "vector"

    def test_search_with_index(self, collection_with_index):
        """Test search with index"""
        # Insert data
        vectors = np.random.rand(100, 1792).tolist()
        ids = [f"id_{i}" for i in range(100)]
        metadata = [f"meta_{i}" for i in range(100)]

        collection_with_index.insert([ids, vectors, metadata])
        collection_with_index.flush()
        collection_with_index.load()

        # Search
        query_vector = np.random.rand(1792).tolist()
        results = collection_with_index.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=10,
            output_fields=["metadata"]
        )

        assert len(results[0]) <= 10


@pytest.mark.integration
@pytest.mark.external
class TestMilvusClient:
    """Test Milvus client wrapper"""

    @pytest.mark.asyncio
    async def test_milvus_client_initialization(self):
        """Test Milvus client initialization"""
        from backend.db.vector.milvus_client import get_milvus_client

        client = await get_milvus_client()
        assert client is not None
        assert client.collection_name is not None

    @pytest.mark.asyncio
    async def test_milvus_insert_embeddings(self):
        """Test inserting embeddings via client"""
        from backend.db.vector.milvus_client import get_milvus_client

        client = await get_milvus_client()

        # Prepare test data
        embeddings = np.random.rand(5, 1792).tolist()
        chunk_ids = [f"chunk_{i}" for i in range(5)]
        texts = [f"Text {i}" for i in range(5)]

        # Insert
        await client.insert_embeddings(chunk_ids, embeddings, texts)

    @pytest.mark.asyncio
    async def test_milvus_search(self):
        """Test searching via client"""
        from backend.db.vector.milvus_client import get_milvus_client

        client = await get_milvus_client()

        # Insert test data
        embeddings = np.random.rand(10, 1792).tolist()
        chunk_ids = [f"chunk_{i}" for i in range(10)]
        texts = [f"Text {i}" for i in range(10)]

        await client.insert_embeddings(chunk_ids, embeddings, texts)

        # Search
        query_embedding = np.random.rand(1792).tolist()
        results = await client.search(query_embedding, top_k=5)

        assert len(results) <= 5


@pytest.mark.integration
@pytest.mark.external
class TestMilvusPerformance:
    """Test Milvus performance"""

    @pytest.mark.asyncio
    async def test_bulk_insert_performance(self):
        """Test bulk insert performance"""
        import time
        from backend.db.vector.milvus_client import get_milvus_client

        client = await get_milvus_client()

        # Generate large batch
        num_vectors = 1000
        embeddings = np.random.rand(num_vectors, 1792).tolist()
        chunk_ids = [f"chunk_{i}" for i in range(num_vectors)]
        texts = [f"Text {i}" for i in range(num_vectors)]

        # Measure insert time
        start = time.time()
        await client.insert_embeddings(chunk_ids, embeddings, texts)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 10  # Less than 10 seconds for 1000 vectors

    @pytest.mark.asyncio
    async def test_search_latency(self):
        """Test search latency"""
        import time
        from backend.db.vector.milvus_client import get_milvus_client

        client = await get_milvus_client()

        # Ensure some data exists
        embeddings = np.random.rand(100, 1792).tolist()
        chunk_ids = [f"chunk_{i}" for i in range(100)]
        texts = [f"Text {i}" for i in range(100)]
        await client.insert_embeddings(chunk_ids, embeddings, texts)

        # Measure search latency
        query_embedding = np.random.rand(1792).tolist()

        latencies = []
        for _ in range(10):
            start = time.time()
            await client.search(query_embedding, top_k=10)
            latencies.append(time.time() - start)

        avg_latency = sum(latencies) / len(latencies)

        # Average search should be fast
        assert avg_latency < 0.5  # Less than 500ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
