"""
Document Processing Tasks
Background tasks for document ingestion pipeline
"""

import uuid
from pathlib import Path
from typing import Dict, Any

from backend.tasks.celery_app import celery_app
from backend.core.logging import get_logger

logger = get_logger(__name__)


@celery_app.task(bind=True)
def process_document(self, document_id: str) -> Dict[str, Any]:
    """
    Process document through the OCR pipeline

    Stages:
    1. OCR Processing (YomiToku)
    2. Text Chunking
    3. Embedding Generation
    4. Vector DB Insertion
    """
    import asyncio
    from datetime import datetime
    import backend.db.session as session_module
    from backend.db.models import Document as DocumentModel
    from sqlalchemy import select

    logger.info(f"Processing document: {document_id}")

    async def process():
        # Ensure database is initialized (for Celery worker process)
        if session_module.async_session_maker is None:
            logger.info("Initializing database connection for Celery worker")
            await session_module.init_db()

        # Get session maker after potential init
        session_maker = session_module.async_session_maker

        # Get database session directly
        async with session_maker() as db:
            try:
                logger.info(f"Database session acquired for document: {document_id}")

                # Get document
                result = await db.execute(
                    select(DocumentModel).where(DocumentModel.id == uuid.UUID(document_id))
                )
                document = result.scalar_one_or_none()

                if not document:
                    logger.error(f"Document not found: {document_id}")
                    return {"error": "Document not found"}

                logger.info(f"Document found: {document_id}, current status: {document.status}")

                # Update status to processing
                document.status = "processing"
                await db.commit()
                logger.info(f"Document status updated to: processing")

                # TODO: Implement actual processing pipeline
                # For now, simulate processing with placeholder data

                # Simulate OCR processing
                await asyncio.sleep(1)

                # Update document with placeholder results
                document.status = "completed"
                document.page_count = 1
                document.chunk_count = 1
                document.ocr_confidence = 0.95
                document.processing_completed_at = datetime.utcnow()

                await db.commit()
                logger.info(f"Document processing complete: {document_id}")

                return {
                    "document_id": document_id,
                    "status": "completed",
                    "pages_processed": document.page_count,
                    "chunks_created": document.chunk_count,
                    "confidence": document.ocr_confidence,
                }

            except Exception as e:
                logger.error(f"Document processing failed: {document_id} - {e}", exc_info=True)
                # Update status to failed
                try:
                    if 'document' in locals() and document:
                        document.status = "failed"
                        await db.commit()
                except:
                    pass
                raise

    try:
        # Run async function with proper event loop management
        result = asyncio.run(process())
        return result

    except Exception as e:
        logger.error(f"Document processing failed: {document_id} - {e}", exc_info=True)
        # Don't use update_state as it may not be available in all contexts
        raise


@celery_app.task
def batch_process_documents(document_ids: list) -> Dict[str, Any]:
    """Process multiple documents in batch"""
    logger.info(f"Batch processing {len(document_ids)} documents")

    results = []
    for doc_id in document_ids:
        try:
            result = process_document.delay(doc_id)
            results.append({"document_id": doc_id, "task_id": result.id})
        except Exception as e:
            logger.error(f"Failed to queue document {doc_id}: {e}")
            results.append({"document_id": doc_id, "error": str(e)})

    return {
        "queued": len(results),
        "results": results,
    }


@celery_app.task
def reprocess_document(document_id: str) -> Dict[str, Any]:
    """Reprocess a document (force re-OCR)"""
    logger.info(f"Reprocessing document: {document_id}")

    # TODO: Implement reprocessing logic
    # This should:
    # 1. Delete existing chunks
    # 2. Reset document status
    # 3. Trigger full reprocessing

    return {
        "document_id": document_id,
        "status": "queued",
    }


@celery_app.task
def cleanup_old_documents(days: int = 90) -> Dict[str, Any]:
    """Clean up documents older than specified days"""
    logger.info(f"Cleaning up documents older than {days} days")

    # TODO: Implement cleanup logic
    # This should:
    # 1. Find documents not accessed in X days
    # 2. Soft delete or archive
    # 3. Clean up MinIO storage

    return {
        "cleaned": 0,
        "archived": 0,
    }
