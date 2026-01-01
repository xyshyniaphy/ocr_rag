# API v1 routes
from fastapi import APIRouter

from backend.api.v1 import auth, documents, query, admin, stream

router = APIRouter()

router.include_router(auth.router)
router.include_router(documents.router)
router.include_router(query.router)
router.include_router(admin.router)
router.include_router(stream.router)
