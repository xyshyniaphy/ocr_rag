"""
FastAPI Application Entry Point
Main application with all routes and middleware
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.core.config import settings
from backend.core.logging import setup_logging, get_logger
from backend.core.exceptions import AppException
from backend.api.v1 import auth, documents, query, admin, stream

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan management"""
    logger.info("Starting Japanese OCR RAG System...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug: {settings.DEBUG}")

    # Startup
    try:
        from backend.db.session import init_db, close_db
        from backend.db.vector.client import init_milvus, close_milvus
        from backend.storage.client import init_minio

        # Initialize databases
        await init_db()
        await init_milvus()
        await init_minio()

        logger.info("All services initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        # Don't fail startup, allow individual services to handle errors

    yield

    # Shutdown
    logger.info("Shutting down...")
    try:
        from backend.db.session import close_db
        from backend.db.vector.client import close_milvus

        await close_db()
        await close_milvus()

        logger.info("Shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Production-grade RAG system for Japanese PDF document processing",
    version=settings.APP_VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan
)


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Exception Handlers
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle application-specific exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
                "timestamp": exc.timestamp,
            }
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle HTTP exceptions"""
    # Handle both string and dict details
    if isinstance(exc.detail, dict):
        # Detail is already a dict (e.g., {"message": "...", "details": {...}})
        error_code = exc.detail.get("code", "http_error")
        error_message = exc.detail.get("message", str(exc.detail))
        error_details = {k: v for k, v in exc.detail.items() if k not in ("code", "message")}
    else:
        # Detail is a string
        error_code = str(exc.detail).lower().replace(" ", "_")
        error_message = str(exc.detail)
        error_details = None

    content = {
        "error": {
            "code": error_code,
            "message": error_message,
            "timestamp": None,
        }
    }

    if error_details:
        content["error"]["details"] = error_details

    return JSONResponse(
        status_code=exc.status_code,
        content=content,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": {
                "code": "validation_error",
                "message": "Invalid request parameters",
                "details": {"errors": exc.errors()},
                "timestamp": None,
            }
        },
    )


# Include routers
app.include_router(
    auth.router,
    prefix="/api/v1/auth",
    tags=["Authentication"]
)

app.include_router(
    documents.router,
    prefix="/api/v1/documents",
    tags=["Documents"]
)

app.include_router(
    query.router,
    prefix="/api/v1",
    tags=["Query"]
)

app.include_router(
    admin.router,
    prefix="/api/v1/admin",
    tags=["Administration"]
)

app.include_router(
    stream.router,
    prefix="/api/v1/stream",
    tags=["WebSocket"]
)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "docs_url": "/docs" if settings.DEBUG else None,
    }


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "timestamp": None,
    }

    # Check dependencies (optional, can be slow)
    try:
        from backend.db.session import get_db_session
        from sqlalchemy import text
        async for session in get_db_session():
            await session.execute(text("SELECT 1"))
            health_status["services"] = health_status.get("services", {})
            health_status["services"]["postgres"] = "healthy"
            break
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["services"] = health_status.get("services", {})
        health_status["services"]["postgres"] = f"unhealthy: {str(e)}"

    try:
        from backend.db.vector.client import get_milvus_client
        client = get_milvus_client()
        if client:
            health_status["services"] = health_status.get("services", {})
            health_status["services"]["milvus"] = "healthy"
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["services"] = health_status.get("services", {})
        health_status["services"]["milvus"] = f"unhealthy: {str(e)}"

    return health_status


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
