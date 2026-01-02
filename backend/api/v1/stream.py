"""
WebSocket API Routes
Real-time streaming endpoints
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import uuid

from backend.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Active WebSocket connections
active_connections: Dict[str, Set[WebSocket]] = {}


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time queries

    Connect with: ws://localhost:8000/api/v1/stream/ws

    Send JSON:
    {
        "type": "query",
        "query": "your query here",
        "top_k": 5
    }

    Receive JSON:
    {
        "type": "token" | "sources" | "complete" | "error",
        "content": "...",
        ...
    }
    """
    await websocket.accept()
    connection_id = str(uuid.uuid4())

    if connection_id not in active_connections:
        active_connections[connection_id] = set()
    active_connections[connection_id].add(websocket)

    logger.info(f"WebSocket connected: {connection_id}")

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)

            logger.debug(f"WebSocket message from {connection_id}: {message.get('type')}")

            # Handle message types
            if message.get("type") == "query":
                await handle_query(websocket, message, connection_id)
            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            else:
                await websocket.send_json({
                    "type": "error",
                    "code": "unknown_message_type",
                    "message": f"Unknown message type: {message.get('type')}"
                })

    except WebSocketDisconnect:
        active_connections[connection_id].discard(websocket)
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "code": "internal_error",
            "message": str(e)
        })


async def handle_query(websocket: WebSocket, message: dict, connection_id: str):
    """Handle query message using real RAG pipeline"""
    import time
    from backend.services.rag import get_rag_service, RAGQueryOptions
    from backend.core.exceptions import RAGValidationError, RAGProcessingError

    query_text = message.get("query", "")
    top_k = message.get("top_k", 5)
    rerank = message.get("rerank", True)
    document_ids = message.get("document_ids")

    # Validate query
    if not query_text or not query_text.strip():
        await websocket.send_json({
            "type": "error",
            "code": "invalid_query",
            "message": "Query cannot be empty"
        })
        return

    # Send metadata
    query_id = str(uuid.uuid4())
    await websocket.send_json({
        "type": "metadata",
        "query_id": query_id,
        "timestamp": time.time(),
    })

    try:
        # Build RAG options
        options = RAGQueryOptions(
            top_k=top_k,
            retrieval_top_k=min(top_k * 2, 50),
            rerank_top_k=top_k,
            rerank=rerank,
            retrieval_method="hybrid",
            document_ids=document_ids,
            include_sources=True,
            use_cache=True,
            language="ja",
        )

        # Get RAG service and process query
        rag_service = get_rag_service()
        rag_result = await rag_service.query(
            query=query_text,
            options=options,
        )

        # Send sources
        sources = [
            {
                "chunk_id": source.chunk_id,
                "document_id": str(source.document_id),
                "document_title": source.document_title or "Unknown Document",
                "page_number": source.page_number,
                "chunk_index": source.chunk_index,
                "text": source.text[:200] + "..." if len(source.text) > 200 else source.text,
                "relevance_score": source.score,
                "rerank_score": source.rerank_score,
            }
            for source in rag_result.sources
        ]

        await websocket.send_json({
            "type": "sources",
            "sources": sources,
        })

        # Stream response tokens
        response = rag_result.answer
        for word in response.split():
            await websocket.send_json({
                "type": "token",
                "content": word + " ",
            })
            await asyncio.sleep(0.01)  # Small delay for streaming effect

        # Send completion
        await websocket.send_json({
            "type": "complete",
            "query_id": query_id,
            "processing_time_ms": rag_result.processing_time_ms,
            "confidence": rag_result.confidence,
            "llm_model": rag_result.llm_model,
            "embedding_model": rag_result.embedding_model,
        })

    except RAGValidationError as e:
        await websocket.send_json({
            "type": "error",
            "code": "validation_error",
            "message": e.message,
            "details": e.details,
        })
    except RAGProcessingError as e:
        logger.error(f"RAG processing error: {e.message} (stage={e.stage})")
        await websocket.send_json({
            "type": "error",
            "code": "processing_error",
            "message": e.message,
            "stage": e.stage,
            "details": e.details,
        })
    except Exception as e:
        logger.error(f"WebSocket query error: {e}")
        await websocket.send_json({
            "type": "error",
            "code": "internal_error",
            "message": str(e),
        })


# Import asyncio for the sleep
import asyncio
