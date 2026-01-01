# ========================================
# Stage 1: App Builder
# Installs application dependencies into virtual environment
# ========================================
FROM ocr-rag:base AS app-builder

# Switch to root for installation
USER root

# Set working directory
WORKDIR /app

# Install uv (fast Python package installer) in app-builder
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

# Copy application requirements
COPY requirements-app.txt /app/

# Uninstall old SQLAlchemy to clear Cython cache, then install all dependencies
RUN uv pip uninstall -y sqlalchemy || true && \
    uv pip install --no-cache-dir --reinstall -r requirements-app.txt


# ========================================
# Stage 2: App
# Final runtime image with venv and application code
# ========================================
FROM ocr-rag:base AS app

# Switch to root for setup
USER root

# Set working directory
WORKDIR /app

# Copy virtual environment with app dependencies from app-builder
COPY --from=app-builder /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser backend/ /app/backend/
COPY --chown=appuser:appuser frontend/ /app/frontend/
COPY --chown=appuser:appuser scripts/ /app/scripts/

# Copy and set up entrypoint directly in the image
COPY --chown=appuser:appuser deployment/docker/entrypoint-dev.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/data /app/tmp && \
    chown -R appuser:appuser /app/logs /app/data /app/tmp

# Switch to non-root user
USER appuser

# Set environment variables (venv already in PATH from base image)
ENV PYTHONPATH="/app:$PYTHONPATH"

# Expose application ports
# 8000 - FastAPI backend
# 8501 - Streamlit frontend
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - runs both FastAPI and Streamlit
CMD ["/app/entrypoint.sh"]
