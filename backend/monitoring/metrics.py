"""
Prometheus metrics for the Japanese OCR RAG System
"""

import time
from functools import wraps
from typing import Callable, Optional
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from prometheus_client.exposition import generate_latest

# Create a custom registry
metrics_registry = CollectorRegistry()

# Request metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
    registry=metrics_registry
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=(.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0, 7.5, 10.0),
    registry=metrics_registry
)

# Query metrics
rag_queries_total = Counter(
    "rag_queries_total",
    "Total RAG queries processed",
    ["status"],
    registry=metrics_registry
)

rag_query_duration_seconds = Histogram(
    "rag_query_duration_seconds",
    "RAG query processing time",
    ["llm_provider"],
    buckets=(.1, .25, .5, .75, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0),
    registry=metrics_registry
)

rag_query_stage_duration_seconds = Histogram(
    "rag_query_stage_duration_seconds",
    "RAG query stage processing time",
    ["stage"],
    buckets=(.01, .025, .05, .075, .1, .25, .5, .75, 1.0),
    registry=metrics_registry
)

# OCR metrics
ocr_documents_total = Counter(
    "ocr_documents_total",
    "Total documents processed by OCR",
    ["status"],
    registry=metrics_registry
)

ocr_document_duration_seconds = Histogram(
    "ocr_document_duration_seconds",
    "OCR document processing time",
    ["status"],
    buckets=(1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 60.0),
    registry=metrics_registry
)

ocr_page_duration_seconds = Histogram(
    "ocr_page_duration_seconds",
    "OCR page processing time",
    buckets=(.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 7.5, 10.0),
    registry=metrics_registry
)

ocr_confidence_score = Gauge(
    "ocr_confidence_score",
    "OCR confidence score",
    ["document_id"],
    registry=metrics_registry
)

# Embedding metrics
embedding_chunks_total = Counter(
    "embedding_chunks_total",
    "Total chunks embedded",
    ["status"],
    registry=metrics_registry
)

embedding_duration_seconds = Histogram(
    "embedding_duration_seconds",
    "Embedding generation time",
    ["batch_size"],
    buckets=(.01, .025, .05, .075, .1, .25, .5, .75, 1.0),
    registry=metrics_registry
)

# Document metrics
documents_total = Gauge(
    "documents_total",
    "Total number of documents",
    ["status"],
    registry=metrics_registry
)

documents_processing_duration_seconds = Histogram(
    "documents_processing_duration_seconds",
    "Document end-to-end processing time",
    ["status"],
    buckets=(5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 120.0),
    registry=metrics_registry
)

# Error metrics
errors_total = Counter(
    "errors_total",
    "Total errors",
    ["error_type", "endpoint"],
    registry=metrics_registry
)

# User metrics
users_total = Gauge(
    "users_total",
    "Total number of users",
    ["role", "status"],
    registry=metrics_registry
)


def track_request(method: str, endpoint: str):
    """Decorator to track HTTP requests"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                errors_total.labels(
                    error_type=type(e).__name__,
                    endpoint=endpoint
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status=status
                ).inc()
                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
        return wrapper
    return decorator


def track_query_latency(llm_provider: str = "glm"):
    """Decorator to track RAG query latency"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            try:
                result = await func(*args, **kwargs)
                # If result has stage_timings, track them
                if hasattr(result, 'stage_timings') and result.stage_timings:
                    for stage_metric in result.stage_timings:
                        if hasattr(stage_metric, 'stage_name') and hasattr(stage_metric, 'duration_ms'):
                            rag_query_stage_duration_seconds.labels(
                                stage=stage_metric.stage_name
                            ).observe(stage_metric.duration_ms / 1000.0)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                rag_queries_total.labels(status=status).inc()
                rag_query_duration_seconds.labels(
                    llm_provider=llm_provider
                ).observe(duration)
        return wrapper
    return decorator


def track_ocr_processing(document_id: Optional[str] = None):
    """Decorator/context manager to track OCR processing"""
    class OCRAuditContext:
        def __init__(self, doc_id: Optional[str] = None):
            self.doc_id = doc_id
            self.start_time = None
            self.status = "success"

        async def __aenter__(self):
            self.start_time = time.time()
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - self.start_time
            if exc_type is not None:
                self.status = "error"
            ocr_document_duration_seconds.labels(status=self.status).observe(duration)
            if self.doc_id:
                ocr_documents_total.labels(status=self.status).inc()
            return False

    return OCRAuditContext(document_id)


def track_page_processing():
    """Decorator/context manager to track page-level OCR processing"""
    class PageOCRContext:
        def __init__(self):
            self.start_time = None

        async def __aenter__(self):
            self.start_time = time.time()
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - self.start_time
            ocr_page_duration_seconds.observe(duration)
            return False

    return PageOCRContext()


def get_metrics() -> bytes:
    """Generate Prometheus metrics exposition format"""
    return generate_latest(metrics_registry)
