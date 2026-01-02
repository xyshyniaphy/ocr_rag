"""
Cleanup Orphaned Chunks from Milvus

This script removes chunks from Milvus that reference soft-deleted documents
in PostgreSQL. This ensures data consistency between the two databases.

Run this script after using "Clear All Documents" or when documents
are soft-deleted but their chunks remain in the vector database.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.core.logging import get_logger
from backend.db.session import async_session_maker
from backend.db.vector.client import get_milvus_client
from sqlalchemy import select
from backend.db.models import Document as DocumentModel
from pymilvus import MilvusException

logger = get_logger(__name__)


async def get_active_document_ids():
    """Get list of all active (non-deleted) document IDs from PostgreSQL"""
    from backend.db.session import init_db

    # Initialize database connection
    await init_db()
    from backend.db.session import async_session_maker

    async with async_session_maker() as session:
        result = await session.execute(
            select(DocumentModel.id).where(DocumentModel.deleted_at.is_(None))
        )
        active_ids = [row[0] for row in result.fetchall()]
        return set(str(doc_id) for doc_id in active_ids)


async def cleanup_orphaned_chunks():
    """Remove chunks that reference deleted documents"""
    logger.info("Starting cleanup of orphaned chunks...")

    # Get active document IDs from PostgreSQL
    logger.info("Fetching active document IDs from PostgreSQL...")
    active_doc_ids = await get_active_document_ids()
    logger.info(f"Found {len(active_doc_ids)} active documents in PostgreSQL")

    # Get all document IDs from Milvus
    logger.info("Fetching document IDs from Milvus...")
    from pymilvus import Collection, connections

    # Connect to Milvus
    connections.connect("default", host="milvus", port="19530")

    collection = Collection("document_chunks")
    collection.load()

    # Query all chunks to get document IDs
    # Milvus has a max query limit of 16384, so we query in batches
    all_doc_ids = set()
    offset = 0
    batch_size = 16384  # Maximum allowed by Milvus

    while True:
        result = collection.query(
            expr="",
            output_fields=["document_id"],
            limit=batch_size,
            offset=offset,
        )

        if not result:
            break

        batch_doc_ids = set(chunk["document_id"] for chunk in result)
        all_doc_ids.update(batch_doc_ids)

        # If we got fewer results than batch size, we're done
        if len(result) < batch_size:
            break

        offset += batch_size

    milvus_doc_ids = all_doc_ids
    logger.info(f"Found {len(milvus_doc_ids)} unique documents in Milvus")

    # Find orphaned document IDs (in Milvus but not in PostgreSQL)
    orphaned_doc_ids = milvus_doc_ids - active_doc_ids

    if not orphaned_doc_ids:
        logger.info("✅ No orphaned chunks found. Database is consistent.")
        return 0

    logger.warning(f"Found {len(orphaned_doc_ids)} orphaned documents in Milvus")
    logger.info(f"Orphaned document IDs: {list(orphaned_doc_ids)[:10]}...")

    # Delete orphaned chunks from Milvus
    logger.info("Deleting orphaned chunks from Milvus...")

    # Build deletion expression for each orphaned document
    total_deleted = 0
    for doc_id in orphaned_doc_ids:
        try:
            # Delete all chunks for this document
            expr = f'document_id == "{doc_id}"'
            collection.delete(expr)
            total_deleted += 1
            logger.info(f"Deleted chunks for document: {doc_id}")
        except MilvusException as e:
            logger.error(f"Error deleting chunks for document {doc_id}: {e}")

    # Flush to ensure deletion is persisted
    collection.flush()
    collection.load()  # Reload after deletion

    logger.info(f"✅ Cleanup complete. Deleted chunks for {total_deleted} orphaned documents.")
    return total_deleted


async def verify_cleanup():
    """Verify that cleanup was successful"""
    logger.info("Verifying cleanup...")

    # Get counts
    active_doc_ids = await get_active_document_ids()

    from pymilvus import Collection, connections

    # Ensure connection
    try:
        connections.connect("default", host="milvus", port="19530")
    except:
        pass  # Already connected

    collection = Collection("document_chunks")
    collection.load()

    # Query in batches
    all_doc_ids = set()
    offset = 0
    batch_size = 16384

    while True:
        result = collection.query(
            expr="",
            output_fields=["document_id"],
            limit=batch_size,
            offset=offset,
        )

        if not result:
            break

        batch_doc_ids = set(chunk["document_id"] for chunk in result)
        all_doc_ids.update(batch_doc_ids)

        if len(result) < batch_size:
            break

        offset += batch_size

    milvus_doc_ids = all_doc_ids
    orphaned_doc_ids = milvus_doc_ids - active_doc_ids

    if orphaned_doc_ids:
        logger.warning(f"⚠️ Still found {len(orphaned_doc_ids)} orphaned documents")
        return False
    else:
        logger.info("✅ Verification passed. No orphaned chunks remain.")
        return True


async def main():
    """Main cleanup function"""
    print("\n" + "="*60)
    print("Milvus Orphaned Chunks Cleanup")
    print("="*60 + "\n")

    try:
        # Run cleanup
        deleted_count = await cleanup_orphaned_chunks()

        # Verify
        success = await verify_cleanup()

        print("\n" + "="*60)
        if success:
            print("✅ Cleanup completed successfully!")
            print(f"   Removed chunks for {deleted_count} deleted documents")
        else:
            print("⚠️ Cleanup completed with warnings")
        print("="*60 + "\n")

    except Exception as e:
        logger.error(f"Cleanup failed: {e}", exc_info=True)
        print("\n" + "="*60)
        print(f"❌ Cleanup failed: {e}")
        print("="*60 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
