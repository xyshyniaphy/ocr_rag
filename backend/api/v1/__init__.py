# API v1 routes
from fastapi import APIRouter

from backend.api.v1 import auth, documents, query, admin, stream

router = APIRouter()

router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(documents.router, prefix="/documents", tags=["documents"])
router.include_router(query.router, prefix="/query", tags=["query"])
router.include_router(admin.router, prefix="/admin", tags=["admin"])
router.include_router(stream.router, prefix="/stream", tags=["stream"])
