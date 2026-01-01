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
    """Handle query message"""
    import time

    query_text = message.get("query", "")
    top_k = message.get("top_k", 5)

    # Send metadata
    query_id = str(uuid.uuid4())
    await websocket.send_json({
        "type": "metadata",
        "query_id": query_id,
        "timestamp": time.time(),
    })

    # Mock query processing
    await websocket.send_json({
        "type": "sources",
        "sources": [
            {
                "document_id": "00000000-0000-0000-0000-000000000000",
                "document_title": "Sample Document",
                "page_number": 1,
                "relevance_score": 0.9,
            }
        ]
    })

    # Stream response tokens
    response = "This is a placeholder response. The RAG pipeline is not yet implemented."
    for word in response.split():
        await websocket.send_json({
            "type": "token",
            "content": word + " ",
        })
        await asyncio.sleep(0.05)  # Simulate streaming

    # Send completion
    await websocket.send_json({
        "type": "complete",
        "query_id": query_id,
        "processing_time_ms": 500,
        "confidence": 0.8,
    })


# Import asyncio for the sleep
import asyncio
