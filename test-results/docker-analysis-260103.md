# Docker Configuration Analysis Report

**Date:** 2026-01-03 14:20 UTC
**Analyzer:** Automated (Code Improvement Analysis)
**Scope:** All Dockerfiles and Docker Compose configurations

---

## Executive Summary

**Status:** ⚠️ **Multiple Issues Found**

| Category | Issues Found | Severity |
|----------|--------------|----------|
| **Security** | 5 | High |
| **Performance** | 3 | Medium |
| **Configuration** | 4 | Medium |
| **GPU Management** | 1 | High |

---

## Critical Issues

### 1. ❌ SEVERE: API Key Exposure in docker-compose.dev.yml

**File:** `docker-compose.dev.yml:61`

**Issue:**
```yaml
- GLM_API_KEY=59e293a0619b4844b1bd3e6d03291894.Hwq4vGCGaqcSVqVO
```

**Problem:**
- API key hardcoded in version control
- Key is visible to anyone with repository access
- Security breach if repository is public

**Impact:** CRITICAL - API key compromised

**Recommendation:**
```yaml
# Use .env file instead
- GLM_API_KEY=${GLM_API_KEY}
```

Create `.env` file (add to `.gitignore`):
```
GLM_API_KEY=59e293a0619b4844b1bd3e6d03291894.Hwq4vGCGaqcSVqVO
```

---

### 2. ❌ SEVERE: Weak Secret Keys in Development

**Files:** `docker-compose.dev.yml:25, 274, 443`

**Issue:**
```yaml
- SECRET_KEY=dev-secret-key-change-in-production-12345
- SECRET_KEY=test-secret-key-for-testing-only-12345
```

**Problem:**
- Predictable secret keys
- Same keys used across environments
- Allows JWT token forgery

**Impact:** HIGH - Authentication bypass possible

**Recommendation:**
```yaml
# Generate secure random keys
- SECRET_KEY=${SECRET_KEY}
```

Generate secure key:
```bash
openssl rand -hex 32
```

---

### 3. ⚠️ HIGH: Exposed MinIO Console in Development

**File:** `docker-compose.dev.yml:200`

**Issue:**
```yaml
ports:
  - "9001:9001"  # Console (dev debugging)
```

**Problem:**
- MinIO console exposed on all interfaces
- Default credentials: `minioadmin/minioadmin`
- Accessible from network

**Impact:** MEDIUM - Unauthorized object storage access

**Recommendation:**
```yaml
# Bind to localhost only or remove entirely
ports:
  - "127.0.0.1:9001:9001"
```

---

### 4. ⚠️ HIGH: Unhealthy Containers Running

**Container Status:**
```
ocr-rag-worker-dev: unhealthy (Up 26 hours)
ocr-rag-test-dev: unhealthy (Up 40 hours)
```

**Problem:**
- Worker container health check failing
- Test container unhealthy (should be stopped)
- May indicate resource exhaustion or configuration issues

**Impact:** MEDIUM - Background tasks may be failing

**Recommendation:**
```bash
# Check worker logs
docker logs ocr-rag-worker-dev --tail 100

# Restart unhealthy containers
docker restart ocr-rag-worker-dev

# Stop unused test container
docker stop ocr-rag-test-dev
```

---

## Performance Issues

### 5. ⚠️ MEDIUM: Single GPU Oversubscription

**Current GPU Allocation:**
```
Services requiring GPU: 4 (app, worker, milvus, ollama, test)
GPU Memory: 8GB total (RTX 3080 Laptop)
Used: 4001MB / 8192MB (~49%)
```

**Problem:**
- 4-5 containers compete for single GPU
- No GPU memory limits set
- Potential for OOM (Out of Memory) errors
- Worker and app both reserve entire GPU (inefficient)

**Impact:** MEDIUM - Performance degradation, potential crashes

**Recommendation:**

**Option A: Use MIG (Multi-Instance GPU) if supported**
```yaml
app:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ['0']
            capabilities: [gpu, compute, utility]
```

**Option B: Disable GPU for non-critical services**
```yaml
# Worker can use CPU for some tasks
worker:
  environment:
    - EMBEDDING_DEVICE=cpu  # Fallback to CPU
```

**Option C: Use Fractional GPU Reservation**
```yaml
# Requires NVIDIA GPU with MIG support
app:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all  # Share GPU
            capabilities: [gpu]
```

---

### 6. ⚠️ MEDIUM: Test Container Left Running

**File:** `docker-compose.dev.yml:434`

**Issue:**
```yaml
test:
  restart: "no"
  command: ["sleep", "infinity"]
```

**Problem:**
- Test container running for 40+ hours
- Consuming GPU resources unnecessarily
- Should be ephemeral

**Impact:** LOW - Wasted resources

**Recommendation:**
```yaml
test:
  restart: "no"  # Current is correct
  # Stop after test completion
  # Use: docker compose --profile test run --rm test
```

Stop the container:
```bash
docker stop ocr-rag-test-dev
docker rm ocr-rag-test-dev
```

---

### 7. ⚠️ MEDIUM: No Build Cache Optimization

**Files:** `Dockerfile.base, Dockerfile.app`

**Issue:**
```dockerfile
# No layer caching optimization
COPY requirements-base.txt /app/
RUN uv pip install --no-cache-dir -r requirements-base.txt
```

**Problem:**
- Requirements install on every code change
- Long rebuild times
- Inefficient cache utilization

**Impact:** LOW - Slower development iteration

**Recommendation:**
```dockerfile
# Dockerfile.base - Already optimal (separate builder stage)

# Dockerfile.app - Improve layer order
COPY requirements-app.txt /app/  # Copy only requirements first
RUN uv pip install --no-cache-dir -r requirements-app.txt  # Cache this layer
COPY --chown=appuser:appuser backend/ /app/backend/  # Code changes don't invalidate requirements
COPY --chown=appuser:appuser frontend/ /app/frontend/
```

---

## Configuration Issues

### 8. ⚠️ MEDIUM: Missing Entrypoint in Dockerfile.test

**File:** `Dockerfile.test:42`

**Issue:**
```dockerfile
CMD ["pytest", "tests/", "-v", "--tb=short"]
```

**Problem:**
- Default command overrides test.sh
- No integration with test profiles
- Missing environment setup

**Impact:** LOW - Inconsistent test execution

**Recommendation:**
```dockerfile
# Don't set CMD, let compose override
# Or use entrypoint script
COPY --chown=appuser:appuser deployment/docker/entrypoint-test.sh /app/entrypoint-test.sh
RUN chmod +x /app/entrypoint-test.sh

ENTRYPOINT ["/app/entrypoint-test.sh"]
```

---

### 9. ⚠️ LOW: Inconsistent Health Check Start Periods

**Files:** Multiple docker-compose files

**Issue:**
```yaml
# Base image (Dockerfile.base)
HEALTHCHECK --start-period=5s

# docker-compose.dev.yml
start_period: 30s  # app service

# docker-compose.prd.yml
start_period: 60s  # app service
```

**Problem:**
- Base image expects 5s startup
- Compose overrides to 30s/60s
- Inconsistent expectations

**Impact:** LOW - False positive health failures

**Recommendation:**
```dockerfile
# Dockerfile.base - Remove healthcheck from base
# Let compose files define service-specific checks

# docker-compose.dev.yml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s  # Allow for model loading
```

---

### 10. ⚠️ LOW: Missing Volume Permissions

**Files:** `docker-compose.dev.yml, docker-compose.prd.yml`

**Issue:**
```yaml
volumes:
  - ./data:/app/data
  - ./logs:/app/logs
```

**Problem:**
- No uid/gid specification
- May cause permission issues
- Files created as root on Linux hosts

**Impact:** LOW - Permission errors on Linux

**Recommendation:**
```yaml
services:
  app:
    user: "${UID:-1001}:${GID:-1001}"
    # Or use named volumes with proper permissions
```

---

## Minor Issues

### 11. ℹ️ INFO: Hardcoded Ollama Model Version

**Files:** `docker-compose.dev.yml:57, docker-compose.prd.yml:52`

**Issue:**
```yaml
- OLLAMA_MODEL=qwen3:4b  # dev
- OLLAMA_MODEL=qwen2.5:14b-instruct-q4_K_M  # prd
```

**Problem:**
- Version mismatch between environments
- No environment variable override

**Impact:** INFO - Inconsistent behavior

**Recommendation:**
```yaml
- OLLAMA_MODEL=${OLLAMA_MODEL:-qwen2.5:14b-instruct-q4_K_M}
```

---

### 12. ℹ️ INFO: Missing Resource Limits in Development

**File:** `docker-compose.dev.yml`

**Issue:**
```yaml
app:
  deploy:
    resources:
      reservations:
        devices: [gpu]
      # No limits defined
```

**Problem:**
- No memory limits
- No CPU limits
- Single service can exhaust host resources

**Impact:** INFO - Development stability

**Recommendation:**
```yaml
app:
  deploy:
    resources:
      limits:
        memory: 16G
        cpus: '8'
      reservations:
        memory: 8G
        cpus: '4'
```

---

## GPU-Specific Issues

### 13. ⚠️ HIGH: Single GPU with Multiple GPU-Intensive Services

**Analysis:**
```
GPU: NVIDIA GeForce RTX 3080 Laptop GPU (8GB)
Services requiring GPU:
  - app: Embedding (Sarashina)
  - worker: Embedding + OCR (YomiToku)
  - milvus: Vector search (GPU-accelerated)
  - ollama: LLM inference
  - test: Testing
```

**Problem:**
- All services compete for same GPU
- No GPU partitioning strategy
- Worker and app duplicate model loading

**Impact:** HIGH - Performance bottleneck, OOM risk

**Recommendations:**

**Immediate (No Code Changes):**
```bash
# Stop unused GPU services
docker stop ocr-rag-test-dev

# Reduce worker concurrency
# docker-compose.dev.yml:317
- CELERY_WORKER_CONCURRENCY=1  # Reduce from 2

# Disable GPU for test container
test:
  environment:
    - EMBEDDING_DEVICE=cpu  # Use CPU for tests
```

**Long-term (Architecture):**
1. **Use separate GPU workers** for different services
2. **Implement GPU scheduling** (e.g., only worker uses GPU for OCR)
3. **CPU fallback** for non-critical tasks
4. **Consider multi-GPU setup** for production

---

## Improvement Recommendations

### Priority 1 (Fix Immediately)

1. **Remove API keys from docker-compose.dev.yml**
   ```bash
   # Create .env file
   echo "GLM_API_KEY=your_key_here" > .env
   echo ".env" >> .gitignore
   ```

2. **Generate secure SECRET_KEY**
   ```bash
   # Add to .env
   echo "SECRET_KEY=$(openssl rand -hex 32)" >> .env
   ```

3. **Stop unhealthy test container**
   ```bash
   docker stop ocr-rag-test-dev && docker rm ocr-rag-test-dev
   ```

4. **Check worker health**
   ```bash
   docker logs ocr-rag-worker-dev --tail 50
   ```

### Priority 2 (Fix Soon)

5. **Add memory limits to dev environment**
6. **Implement CPU fallback for worker**
7. **Fix health check consistency**
8. **Minimize GPU over-subscription**

### Priority 3 (Nice to Have)

9. **Optimize Docker layer caching**
10. **Add volume permissions handling**
11. **Unify model version configuration**
12. **Add container resource monitoring**

---

## Quick Fixes (Ready to Apply)

### Fix 1: Create .env.example Template

**Create:** `.env.example`
```bash
# Security
SECRET_KEY=generate-with-openssl-rand-hex-32
GLM_API_KEY=your-glm-api-key-from-z-ai
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# Database
POSTGRES_USER=raguser
POSTGRES_PASSWORD=secure_password_here

# Redis
REDIS_PASSWORD=secure_redis_password

# MinIO Production
MINIO_ACCESS_KEY=production_key
MINIO_SECRET_KEY=production_secret

# Ollama
OLLAMA_MODEL=qwen2.5:14b-instruct-q4_K_M

# User/Group for Linux
UID=1001
GID=1001
```

### Fix 2: Update docker-compose.dev.yml

```yaml
environment:
  # Change from hardcoded to:
  - SECRET_KEY=${SECRET_KEY:-dev-secret-key-change-in-production}
  - GLM_API_KEY=${GLM_API_KEY}
  - MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY:-minioadmin}
  - MINIO_SECRET_KEY=${MINIO_SECRET_KEY:-minioadmin}

ports:
  # Change MinIO console to localhost only:
  - "127.0.0.1:9001:9001"
```

### Fix 3: Reduce Worker Concurrency

```yaml
worker:
  environment:
    - CELERY_WORKER_CONCURRENCY=1  # Reduce from 2
    - EMBEDDING_DEVICE=cuda:0  # Explicit GPU device
```

---

## Testing Checklist

After applying fixes, verify:

- [ ] All containers start successfully: `docker compose up -d`
- [ ] All health checks pass: `docker compose ps`
- [ ] No API keys in repository: `git grep -i "api_key\|secret"`
- [ ] GPU memory usage reasonable: `nvidia-smi`
- [ ] Worker processing documents: Check logs
- [ ] Test cleanup works: `docker compose --profile test run --rm test pytest tests/`

---

## Conclusion

The Docker configuration has **13 issues** across security, performance, and configuration:

- **5 Critical/High** severity (fix immediately)
- **6 Medium** severity (fix soon)
- **2 Low** severity (improvements)

**Most Critical:** API key exposure and weak secrets pose immediate security risks.

**Performance Bottleneck:** Single GPU over-subscription will cause issues under load.

**Next Steps:**
1. Fix security issues (Priority 1)
2. Optimize GPU usage (Priority 2)
3. Apply remaining improvements (Priority 3)

---

**End of Report**
