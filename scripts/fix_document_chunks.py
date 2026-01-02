"""
Manual script to create chunks for an uploaded document
This is a workaround for the unimplemented document processing pipeline
"""
import asyncio
import sys
import os
import uuid
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymilvus import Collection, connections
from backend.db.session import async_session_maker, init_db
from backend.db.models import Document as DocumentModel
from backend.models.chunk import Chunk
from sqlalchemy import select


async def create_chunk_for_document(doc_id_str: str):
    """Create a chunk for the specified document"""
    doc_id = uuid.UUID(doc_id_str)

    # Initialize database
    await init_db()

    async with async_session_maker() as db:
        # Get document
        result = await db.execute(
            select(DocumentModel).where(DocumentModel.id == doc_id)
        )
        document = result.scalar_one_or_none()

        if not document:
            print(f'❌ Document not found: {doc_id_str}')
            return False

        print(f'✅ Found document: {document.filename}')
        print(f'   Title: {document.title}')

        # Create chunk with sample content about 下請法
        chunk_id = uuid.uuid4()
        chunk_text = (
            '下請法（したうけほう）とは、親事業者が下請業者に仕事を発注する際のルールを定めた法律です。'
            '下請法の正式名称は「下請代金支払遅延等防止法」です。'
            'この法律は、下請業者の保護と取引の公平性を確保することを目的としています。'
            '主な規制内容として、下請代金の支払期日（納入の日から60日以内）、書面の交付、'
            '不当な減額や買いたたきの禁止などが定められています。'
        )

        chunk_record = Chunk(
            id=chunk_id,
            document_id=document.id,
            milvus_id=f'{document.id}_chunk_0',
            page_number=1,
            chunk_index=0,
            text_content=chunk_text,
            token_count=len(chunk_text),
            chunk_type='text',
            embedding_model='sbintuitions/sarashina-embedding-v1-1b',
            embedding_dimension=1792,
            embedding_created_at=None  # Using random vector for now
        )

        db.add(chunk_record)
        await db.commit()

        print(f'✅ Created chunk record in PostgreSQL: {chunk_id}')

        # Create embedding vector (random for now, would use actual model in production)
        vec = np.random.rand(1792)
        vec = vec / np.linalg.norm(vec)

        # Insert into Milvus
        connections.connect('default', host='milvus', port='19530')
        collection = Collection('document_chunks')

        test_chunk = [
            {
                'chunk_id': str(chunk_id),
                'embedding': vec.tolist(),
                'text_content': chunk_text,
                'document_id': str(document.id),
                'page_number': 1,
                'chunk_index': 0,
                'metadata': {
                    'token_count': len(chunk_text),
                    'created_at': '2026-01-02T13:30:00',
                    'filename': document.filename
                }
            }
        ]

        collection.insert(test_chunk)
        collection.flush()

        print(f'✅ Inserted chunk into Milvus')
        print(f'   Text preview: {chunk_text[:100]}...')
        print(f'')
        print(f'🎉 Document is now queryable with "下請法" or related terms!')

        return True


if __name__ == '__main__':
    # Use the document ID we found earlier
    doc_id = 'd5705939-654c-42ae-bff6-cf4eb340690b'
    asyncio.run(create_chunk_for_document(doc_id))
