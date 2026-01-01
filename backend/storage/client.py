"""
MinIO Object Storage Client
File storage for PDFs, OCR outputs, and thumbnails
"""

from typing import Optional
from pathlib import Path
import io

from minio import Minio
from minio.error import MinioException

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.core.exceptions import AppException

logger = get_logger(__name__)

# Global MinIO client
_client: Optional[Minio] = None

# Bucket names
BUCKET_RAW_PDFS = "raw-pdfs"
BUCKET_OCR_OUTPUTS = "ocr-outputs"
BUCKET_THUMBNAILS = "thumbnails"


def get_minio_client() -> Minio:
    """Get MinIO client"""
    if _client is None:
        raise AppException("MinIO client not initialized")
    return _client


async def init_minio() -> None:
    """Initialize MinIO client and create buckets"""
    global _client

    try:
        logger.info(f"Connecting to MinIO at {settings.MINIO_ENDPOINT}")

        # Parse endpoint
        endpoint = settings.MINIO_ENDPOINT
        if "://" in endpoint:
            endpoint = endpoint.split("://")[1]

        # Create client
        _client = Minio(
            endpoint,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_USE_SSL,
        )

        # Check connection
        _client.list_buckets()

        # Create buckets
        await create_buckets()

        logger.info("MinIO initialized successfully")

    except MinioException as e:
        logger.error(f"Failed to initialize MinIO: {e}")
        raise AppException(
            message="Failed to initialize object storage",
            details={"error": str(e)},
        )


async def create_buckets() -> None:
    """Create all required buckets if they don't exist"""

    if _client is None:
        raise AppException("MinIO client not initialized")

    buckets = [BUCKET_RAW_PDFS, BUCKET_OCR_OUTPUTS, BUCKET_THUMBNAILS]

    for bucket in buckets:
        try:
            if not _client.bucket_exists(bucket):
                _client.make_bucket(bucket)
                logger.info(f"Created bucket: {bucket}")
            else:
                logger.debug(f"Bucket exists: {bucket}")
        except MinioException as e:
            logger.error(f"Failed to create bucket {bucket}: {e}")


async def upload_file(
    bucket: str,
    object_name: str,
    data: bytes,
    content_type: str = "application/octet-stream",
) -> str:
    """Upload a file to MinIO"""

    if _client is None:
        raise AppException("MinIO client not initialized")

    try:
        data_stream = io.BytesIO(data)
        _client.put_object(
            bucket,
            object_name,
            data_stream,
            length=len(data),
            content_type=content_type,
        )
        logger.debug(f"Uploaded file: {bucket}/{object_name}")
        return object_name

    except MinioException as e:
        logger.error(f"Failed to upload file {bucket}/{object_name}: {e}")
        raise AppException(
            message="Failed to upload file",
            details={"bucket": bucket, "object_name": object_name},
        )


async def download_file(bucket: str, object_name: str) -> bytes:
    """Download a file from MinIO"""

    if _client is None:
        raise AppException("MinIO client not initialized")

    try:
        response = _client.get_object(bucket, object_name)
        data = response.read()
        logger.debug(f"Downloaded file: {bucket}/{object_name}")
        return data

    except MinioException as e:
        logger.error(f"Failed to download file {bucket}/{object_name}: {e}")
        raise AppException(
            message="Failed to download file",
            details={"bucket": bucket, "object_name": object_name},
        )


async def delete_file(bucket: str, object_name: str) -> None:
    """Delete a file from MinIO"""

    if _client is None:
        raise AppException("MinIO client not initialized")

    try:
        _client.remove_object(bucket, object_name)
        logger.debug(f"Deleted file: {bucket}/{object_name}")

    except MinioException as e:
        logger.error(f"Failed to delete file {bucket}/{object_name}: {e}")
        raise AppException(
            message="Failed to delete file",
            details={"bucket": bucket, "object_name": object_name},
        )


async def get_presigned_url(
    bucket: str,
    object_name: str,
    expires: int = 3600,
) -> str:
    """Generate a presigned URL for download"""

    if _client is None:
        raise AppException("MinIO client not initialized")

    try:
        url = _client.presigned_get_object(
            bucket,
            object_name,
            expires=expires,
        )
        return url

    except MinioException as e:
        logger.error(f"Failed to generate presigned URL for {bucket}/{object_name}: {e}")
        raise AppException(
            message="Failed to generate presigned URL",
            details={"bucket": bucket, "object_name": object_name},
        )
