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

                # ===== IMPLEMENTATION: Document Processing Pipeline =====
                from backend.storage.client import download_file, BUCKET_RAW_PDFS, init_minio
                from backend.db.vector.client import init_milvus

                # Initialize storage and vector DB clients
                await init_minio()
                await init_milvus()
                from backend.services.ocr.yomitoku import YomiTokuOCREngine
                from backend.services.ocr.models import OCROptions
                from backend.services.processing.chunking.strategies import JapaneseRecursiveSplitter
                from backend.services.processing.chunking.models import ChunkingOptions
                from backend.services.embedding.service import EmbeddingService
                from backend.models.chunk import Chunk as ChunkModel
                from pymilvus import Collection, connections
                import numpy as np
                from datetime import datetime

                # 1. Download PDF from MinIO
                logger.info(f"Downloading PDF from MinIO: {document.storage_path}")
                try:
                    # storage_path format: "raw-pdfs/{document_id}/{filename}"
                    # Need to extract object_name
                    object_name = document.storage_path.split("/", 1)[1]
                    pdf_bytes = await download_file(BUCKET_RAW_PDFS, object_name)
                    logger.info(f"PDF downloaded successfully: {len(pdf_bytes)} bytes")
                except Exception as e:
                    logger.error(f"Failed to download PDF from MinIO: {e}")
                    raise

                # 2. Run OCR using YomiToku
                logger.info("Starting OCR processing...")
                ocr_engine = YomiTokuOCREngine(
                    options=OCROptions(
                        engine="yomitoku",
                        confidence_threshold=0.85,
                    )
                )

                # Load OCR model
                await ocr_engine.load_model()

                # Convert PDF to images and run OCR
                # For simplicity, we'll use pdf2image to convert PDF pages to images
                import pdf2image
                import io

                # Convert PDF bytes to PIL images
                pdf_images = pdf2image.convert_from_bytes(
                    pdf_bytes,
                    dpi=200,
                    fmt='png'
                )

                logger.info(f"PDF has {len(pdf_images)} pages")

                # Process each page with OCR
                ocr_pages = []
                total_confidence = 0.0

                for page_num, image in enumerate(pdf_images, start=1):
                    logger.info(f"Processing page {page_num}/{len(pdf_images)}...")

                    # Convert PIL image to bytes
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format='PNG')
                    img_bytes = img_bytes.getvalue()

                    # Run OCR
                    ocr_page = await ocr_engine.process_page(
                        img_bytes,
                        page_number=page_num,
                    )

                    ocr_pages.append(ocr_page)
                    total_confidence += ocr_page.confidence
                    logger.info(f"Page {page_num} OCR complete: {len(ocr_page.text)} chars, confidence={ocr_page.confidence:.2f}")

                avg_confidence = total_confidence / len(ocr_pages) if ocr_pages else 0.0
                logger.info(f"OCR complete: {len(ocr_pages)} pages, avg confidence={avg_confidence:.2f}")

                # 3. Chunk text using Japanese-aware splitter
                logger.info("Starting text chunking...")
                chunking_options = ChunkingOptions(
                    chunk_size=512,
                    chunk_overlap=50,
                    min_chunk_size=20,
                )
                chunker = JapaneseRecursiveSplitter(chunking_options)

                all_chunks = []
                chunk_index = 0

                for ocr_page in ocr_pages:
                    chunks = await chunker.chunk_page(
                        ocr_page,
                        document_id=str(document.id),
                        chunk_start_index=chunk_index,
                    )
                    all_chunks.extend(chunks)
                    chunk_index += len(chunks)

                logger.info(f"Chunking complete: {len(all_chunks)} chunks created")

                # 4. Generate embeddings using Sarashina
                logger.info("Generating embeddings...")
                embedding_service = EmbeddingService()
                await embedding_service.initialize()

                # Batch embed chunks
                chunk_texts = [chunk.text for chunk in all_chunks]
                embedding_result = await embedding_service.embed_texts(chunk_texts)

                logger.info(f"Embeddings generated: {embedding_result.total_embeddings} vectors")

                # 5. Insert into Milvus (vector DB)
                logger.info("Inserting chunks into Milvus...")
                connections.connect("default", host="milvus", port="19530")
                collection = Collection("document_chunks")

                # Prepare data for Milvus insertion
                milvus_data = []
                chunk_records = []

                for i, (chunk, text_embedding) in enumerate(zip(all_chunks, embedding_result.embeddings)):
                    # Create PostgreSQL chunk record
                    milvus_id = f"{document.id}_chunk_{i}"

                    chunk_record = ChunkModel(
                        id=uuid.UUID(chunk.chunk_id),
                        document_id=document.id,
                        milvus_id=milvus_id,
                        page_number=chunk.metadata.page_number,
                        chunk_index=chunk.metadata.chunk_index,
                        text_content=chunk.text,
                        token_count=chunk.token_count,
                        chunk_type=chunk.metadata.chunk_type,
                        section_header=chunk.metadata.section_header,
                        confidence=chunk.metadata.confidence,
                        embedding_model="sbintuitions/sarashina-embedding-v1-1b",
                        embedding_dimension=1792,
                        embedding_created_at=datetime.utcnow(),
                    )

                    chunk_records.append(chunk_record)

                    # Prepare Milvus data
                    milvus_data.append({
                        "chunk_id": chunk.chunk_id,
                        "embedding": text_embedding.embedding.vector,
                        "text_content": chunk.text,
                        "document_id": str(document.id),
                        "page_number": chunk.metadata.page_number,
                        "chunk_index": chunk.metadata.chunk_index,
                        "metadata": {
                            "token_count": chunk.token_count,
                            "chunk_type": chunk.metadata.chunk_type,
                            "confidence": chunk.metadata.confidence,
                            "created_at": datetime.utcnow().isoformat(),
                            "filename": document.filename,
                            "document_title": document.title,  # Add document title for query results
                        }
                    })

                # Bulk insert PostgreSQL chunks
                db.add_all(chunk_records)
                await db.commit()
                logger.info(f"Inserted {len(chunk_records)} chunk records into PostgreSQL")

                # Bulk insert Milvus vectors
                collection.insert(milvus_data)
                collection.flush()
                logger.info(f"Inserted {len(milvus_data)} vectors into Milvus")

                # 6. Update document status
                document.status = "completed"
                document.page_count = len(ocr_pages)
                document.chunk_count = len(all_chunks)
                document.ocr_confidence = avg_confidence
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
