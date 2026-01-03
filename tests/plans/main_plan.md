# Main E2E Test Plan - OCR RAG System

**Test Plan ID:** TP-001
**Version:** 1.0
**Created:** 2026-01-03
**Status:** Active

## Overview

This test plan provides comprehensive end-to-end testing for the Japanese OCR RAG System using Chrome MCP for automated browser testing. The plan covers document upload, processing, and query functionality with various parameter combinations.

## Test Environment

- **Base URL:** http://localhost:8501/
- **Test Data:** tests/testdata/test.pdf
- **Browser:** Chrome (via MCP)
- **Test Type:** End-to-End (E2E)

## Test Document Analysis

**Document:** test.pdf (Japanese OCR Technical Document)

**Content Summary:**
- OCR technology overview and principles
- OCR engine types and features (YomiToku, Tesseract, etc.)
- Use cases and applications
- Accuracy evaluation and challenges
- Future trends in OCR technology

**Key Topics for Question Generation:**
1. OCR technology principles
2. OCR engine comparisons
3. OCR use cases and applications
4. OCR accuracy and challenges

## Test Prerequisites

1. **Services Running:**
   ```bash
   ./dev.sh up
   ```

2. **Verify Services:**
   - Streamlit UI: http://localhost:8501/
   - Backend API: http://localhost:8000/docs
   - All containers healthy

3. **Test Data:**
   - File exists: `tests/testdata/test.pdf`
   - File size: ~557 KB
   - File type: PDF

4. **Chrome MCP:**
   - Chrome DevTools Protocol enabled
   - MCP server running

## Test Scenarios

### Phase 1: Setup and Cleanup (Pre-Test)

**Test Case 1.1: Navigate to Application**
```
Given the browser is opened
When I navigate to http://localhost:8501/
Then the Streamlit application should load
And the page title should be visible
```

**Test Case 1.2: Clear All Documents**
```
Given I am on the Streamlit application
When I navigate to "Settings" page
And I click "Clear All Documents" button
And I confirm the action
Then all documents should be removed
And success message should be displayed
```

### Phase 2: Document Upload

**IMPORTANT: Use API for file upload, NOT browser UI**

Browser file upload dialogs block test execution. Always use the backend API with authentication tokens.

**Test Case 2.1: Upload Test Document via API**
```
Given I have an authentication token
When I POST to /api/v1/documents/upload
And I include the PDF file in multipart/form-data
And I set proper headers (Authorization: Bearer <token>)
Then the document should be uploaded
And I receive a document_id
And status should be "pending"
And processing starts automatically
```

**API Upload Steps:**
1. Login via POST /api/v1/auth/login to get access_token
2. Read file: tests/testdata/test.pdf
3. POST to /api/v1/documents/upload with:
   - Headers: Authorization: Bearer <token>
   - Files: file=(filename, file_bytes, "application/pdf")
4. Receive response with document_id
5. Poll /api/v1/documents/{document_id}/status until "completed"
6. Verify document metadata via /api/v1/documents/{document_id}

**Expected Document Metadata:**
- Filename: test.pdf
- File Size: ~557 KB
- Page Count: > 0
- Chunk Count: > 0
- OCR Confidence: > 0.5

### Phase 3: Query Testing

**Dynamic Question Generation**

Based on the PDF content about OCR technology, the following test questions will be used:

**Question 1 (Basic - OCR Definition):**
- Japanese: 「OCRとは何ですか？」
- English: "What is OCR?"
- **Expected Answer:** Explanation of OCR (Optical Character Recognition) technology

**Question 2 (Intermediate - OCR Engines):**
- Japanese: 「OCRエンジンの種類と特徴を教えてください」
- English: "What are the types and features of OCR engines?"
- **Expected Answer:** Information about various OCR engines like YomiToku, Tesseract, etc.

**Question 3 (Advanced - OCR Use Cases):**
- Japanese: 「OCRの活用事例と応用分野について教えてください」
- English: "What are the use cases and application fields of OCR?"
- **Expected Answer:** Examples of OCR applications in different industries

#### Test Matrix for Query Testing

**Parameters:**
- **Reranker:** [On, Off]
- **Number of Sources:** [1, 5, 10, 20]
- **Language:** [Japanese, English]

**Total Combinations:** 2 × 4 × 2 = 16 scenarios per question

**Test Case 3.1: Query with Reranker On, 1 Source, Japanese**
```
Given the document is processed and status is "completed"
When I navigate to "Query" page
And I enter question: 「OCRとは何ですか？」
And I set "Reranker" to "On"
And I set "Number of Sources" to 1
And I select "Language" as "Japanese"
And I click "Submit Query"
Then the query should be submitted
And results should be displayed within 30 seconds
And at least 1 source should be returned
And the answer should be in Japanese
And the source should show document title (not "Unknown Document")
```

**Test Case 3.2: Query with Reranker Off, 5 Sources, Japanese**
```
Given the document is processed
When I navigate to "Query" page
And I enter question: 「OCRエンジンの種類と特徴を教えてください」
And I set "Reranker" to "Off"
And I set "Number of Sources" to 5
And I select "Language" as "Japanese"
And I click "Submit Query"
Then the query should be submitted
And results should be displayed
And exactly 5 sources should be returned
And all sources should have valid document titles
```

**Test Case 3.3: Query with Reranker On, 10 Sources, English**
```
Given the document is processed
When I navigate to "Query" page
And I enter question: "What is OCR?"
And I set "Reranker" to "On"
And I set "Number of Sources" to 10
And I select "Language" as "English"
And I click "Submit Query"
Then the query should be displayed
And up to 10 sources should be returned
And the answer should be in English
And source scores should be displayed
```

**Test Case 3.4: Query with Reranker Off, 20 Sources, English**
```
Given the document is processed
When I navigate to "Query" page
And I enter question: "What are the use cases and application fields of OCR?"
And I set "Reranker" to "Off"
And I set "Number of Sources" to 20
And I select "Language" as "English"
And I click "Submit Query"
Then the query should be submitted
And up to 20 sources should be returned
And results should not contain "Unknown Document"
```

### Phase 4: Parameter Variation Tests

**Test Case 4.1: All Source Count Variations**
```
Given the document is processed
When I test with sources = [1, 5, 10, 20]
Then each test should return the expected number of sources
And all sources should have valid metadata
```

**Test Case 4.2: Reranker Toggle Tests**
```
Given the document is processed
When I test with reranker = [On, Off]
Then reranker On should show rerank scores
And reranker Off should not show rerank scores
And both should return valid results
```

**Test Case 4.3: Language Switch Tests**
```
Given the document is processed
When I test with language = [Japanese, English]
Then Japanese queries should return Japanese answers
And English queries should return English answers
And language should be consistent throughout
```

### Phase 5: Error Handling and Edge Cases

**Test Case 5.1: Query Before Document Processing Complete**
```
Given a document is uploaded but still processing
When I try to submit a query
Then I should see a "Processing in progress" message
Or the query should wait for processing to complete
```

**Test Case 5.2: Empty Query**
```
Given the document is processed
When I submit an empty query
Then I should see a validation error
```

**Test Case 5.3: Invalid Source Count**
```
Given the document is processed
When I set source count to 0 or negative value
Then the input should be rejected or clamped to valid range
```

**Test Case 5.4: Network Error During Query**
```
Given the document is processed
When I submit a query and the backend is unavailable
Then I should see an appropriate error message
And the UI should remain responsive
```

### Phase 6: Validation Tests

**Test Case 6.1: Document Title Display Validation**
```
Given a query has been executed
When I examine the sources
Then NO source should display "Unknown Document"
And all sources should show the actual document title
```

**Test Case 6.2: Source Metadata Validation**
```
Given a query has returned sources
When I examine each source
Then each source should have:
- Document ID (valid UUID)
- Document Title (not "Unknown Document")
- Page Number (integer >= 1)
- Chunk Index (integer >= 0)
- Chunk Text (non-empty string)
- Score (float between 0 and 1)
- Rerank Score (if reranker enabled)
```

**Test Case 6.3: Answer Quality Validation**
```
Given a query has been executed
When I examine the answer
Then the answer should:
- Be relevant to the question
- Be in the correct language
- Not be empty or just placeholders
- Be based on the retrieved sources
```

## Test Execution Steps

### Step 1: Environment Setup
```bash
# Start services
./dev.sh up

# Verify all services are running
docker compose ps

# Check logs if needed
./dev.sh logs
```

### Step 2: Initialize Chrome MCP
```python
# Connect to Chrome DevTools
# Navigate to base URL
```

### Step 3: Execute Phase 1 (Setup and Cleanup)
1. Open http://localhost:8501/
2. Take screenshot of home page
3. Navigate to Settings
4. Clear all documents
5. Verify success message

### Step 4: Execute Phase 2 (Document Upload via API)
1. Get auth token: POST /api/v1/auth/login
2. Upload document: POST /api/v1/documents/upload with file
3. Extract document_id from response
4. Poll status: GET /api/v1/documents/{document_id}/status
5. Wait for status to change: pending → processing → completed (max 60s)
6. Verify metadata: GET /api/v1/documents/{document_id}
7. Record processing time and document stats

### Step 5: Execute Phase 3 (Query Testing)
For each test case in the test matrix:

1. Navigate to Query page
2. Enter test question
3. Set parameters (reranker, sources, language)
4. Submit query
5. Wait for results (max 30 seconds)
6. Record response time
7. Validate results:
   - Answer is present
   - Correct number of sources
   - No "Unknown Document" errors
   - Correct language
   - Valid scores
8. Take screenshot
9. Log results

### Step 6: Execute Phase 4 (Parameter Variations)
1. Systematically test all parameter combinations
2. Record results for each combination
3. Identify any failures or inconsistencies

### Step 7: Execute Phase 5 (Error Handling)
1. Test edge cases
2. Verify error messages
3. Check UI responsiveness

### Step 8: Execute Phase 6 (Validation)
1. Validate document titles
2. Validate source metadata
3. Validate answer quality
4. Record any issues

## Expected Results

### Success Criteria

1. **Document Upload:**
   - ✅ Upload completes successfully
   - ✅ Status transitions: pending → processing → completed
   - ✅ Processing time < 60 seconds
   - ✅ Document metadata is correct
   - ✅ No errors in logs

2. **Query Execution:**
   - ✅ All 3 questions execute successfully
   - ✅ All parameter combinations work (16 per question = 48 total)
   - ✅ Response time < 30 seconds per query
   - ✅ Correct number of sources returned
   - ✅ No "Unknown Document" errors

3. **Reranker Functionality:**
   - ✅ Reranker On: Shows rerank scores
   - ✅ Reranker Off: No rerank scores
   - ✅ Results differ between on/off (expected)

4. **Language Support:**
   - ✅ Japanese queries return Japanese answers
   - ✅ English queries return English answers
   - ✅ Language is consistent

5. **Source Validation:**
   - ✅ All sources have valid document titles
   - ✅ All sources have valid metadata
   - ✅ Scores are in range [0, 1]
   - ✅ Page numbers are valid
   - ✅ Chunk indices are valid

6. **Error Handling:**
   - ✅ Appropriate error messages for edge cases
   - ✅ UI remains responsive
   - ✅ No crashes or freezes

### Failure Criteria

1. **Critical Failures:**
   - ❌ Document upload fails
   - ❌ Document processing fails
   - ❌ Query submission fails
   - ❌ "Unknown Document" errors appear
   - ❌ Application crashes or freezes
   - ❌ Backend errors (500, 503, etc.)

2. **Non-Critical Failures:**
   - ⚠️ Slow response times (> 30s)
   - ⚠️ Minor UI issues
   - ⚠️ Inconsistent metadata
   - ⚠️ Wrong language in answer

## Test Data

### Document: test.pdf

**File Information:**
- Path: `tests/testdata/test.pdf`
- Size: ~557 KB
- Type: PDF document
- Language: Japanese
- Content: OCR technical documentation

**Content Overview:**
- OCR technology principles
- OCR engine comparisons
- OCR use cases and applications
- OCR accuracy and evaluation
- Future trends

### Test Questions

**Question Set 1 (Japanese):**
1. 「OCRとは何ですか？」
2. 「OCRエンジンの種類と特徴を教えてください」
3. 「OCRの活用事例と応用分野について教えてください」

**Question Set 2 (English):**
1. "What is OCR?"
2. "What are the types and features of OCR engines?"
3. "What are the use cases and application fields of OCR?"

### Parameter Combinations

**Reranker:** [On, Off] (2 options)

**Number of Sources:** [1, 5, 10, 20] (4 options)

**Languages:** [Japanese, English] (2 options)

**Total Test Combinations:** 2 × 4 × 2 × 3 questions = **48 test cases**

## Test Automation Script (Chrome MCP)

### Prerequisites

```python
from mcp chrome_devtools import *
import asyncio
import time
from typing import List, Dict
```

### Test Suite Structure

```python
class MainTestSuite:
    """Main E2E Test Suite for OCR RAG System"""

    def __init__(self):
        self.api_base = "http://localhost:8000/api/v1"
        self.test_pdf = "tests/testdata/test.pdf"
        self.results = []
        self.auth_token = None

    def setup(self):
        """Initialize test environment"""
        # Get auth token
        login_response = requests.post(
            f"{self.api_base}/auth/login",
            json={"email": "admin@example.com", "password": "admin123"}
        )
        self.auth_token = login_response.json()["access_token"]

    def test_clear_documents(self):
        """Clear all documents via API"""
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        response = requests.delete(
            f"{self.api_base}/documents/all",
            headers=headers
        )
        return response.status_code == 200

    def test_upload_document(self):
        """Upload document via API"""
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        with open(self.test_pdf, "rb") as f:
            files = {"file": (os.path.basename(self.test_pdf), f, "application/pdf")}
            response = requests.post(
                f"{self.api_base}/documents/upload",
                headers=headers,
                files=files
            )
        return response.json()

    def wait_for_processing(self, document_id, timeout=60):
        """Wait for document processing to complete"""
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        start_time = time.time()

        while time.time() - start_time < timeout:
            response = requests.get(
                f"{self.api_base}/documents/{document_id}/status",
                headers=headers
            )
            status = response.json()

            if status["status"] == "completed":
                return True
            elif status["status"] == "failed":
                return False

            time.sleep(2)

        return False

    async def test_query_all_combinations(self):
        """Test all query parameter combinations"""
        questions = [
            ("OCRとは何ですか？", "ja"),
            ("OCRエンジンの種類と特徴を教えてください", "ja"),
            ("OCRの活用事例と応用分野について教えてください", "ja"),
            ("What is OCR?", "en"),
            ("What are the types and features of OCR engines?", "en"),
            ("What are the use cases and application fields of OCR?", "en"),
        ]

        for reranker in [True, False]:
            for sources in [1, 5, 10, 20]:
                for question, lang in questions:
                    await self.test_query(
                        question=question,
                        reranker=reranker,
                        sources=sources,
                        language=lang
                    )

    async def test_query(self, question: str, reranker: bool,
                        sources: int, language: str):
        """Execute single query test"""
        # Navigate to query page
        # Enter question
        # Set parameters
        # Submit query
        # Wait for results
        # Validate results
        # Record outcome
        pass

    async def validate_results(self):
        """Validate all test results"""
        # Check for "Unknown Document" errors
        # Validate source metadata
        # Check answer quality
        pass

    async def teardown(self):
        """Cleanup test environment"""
        await self.clear_all_documents()
```

## Test Reporting

### Test Execution Report Template

```markdown
# Test Execution Report

**Date:** YYYY-MM-DD HH:MM:SS
**Test Plan:** main_plan.md
**Tester:** [Automated/Manual]
**Environment:** Development

## Summary
- Total Tests: 48
- Passed: XX
- Failed: XX
- Skipped: XX
- Pass Rate: XX%

## Results by Phase

### Phase 1: Setup and Cleanup
- Status: [Passed/Failed]
- Duration: XX seconds
- Notes: [...]

### Phase 2: Document Upload
- Status: [Passed/Failed]
- Upload Duration: XX seconds
- Processing Duration: XX seconds
- Document Metadata: [...]
- Screenshot: [path]

### Phase 3: Query Testing
- Total Queries: 48
- Passed: XX
- Failed: XX
- Average Response Time: XX seconds
- Min Response Time: XX seconds
- Max Response Time: XX seconds

#### Query Results Table
| Question | Reranker | Sources | Language | Status | Response Time | Notes |
|----------|----------|---------|----------|--------|---------------|-------|
| ... | ... | ... | ... | ... | ... | ... |

### Phase 4: Parameter Variations
- All combinations tested: [Yes/No]
- Issues found: [...]

### Phase 5: Error Handling
- Edge cases tested: XX
- Error messages validated: [Yes/No]

### Phase 6: Validation
- "Unknown Document" errors: 0
- Invalid metadata: 0
- Answer quality issues: 0

## Issues Found

### Critical Issues
1. [Description]
   - Severity: Critical
   - Steps to reproduce: [...]
   - Expected: [...]
   - Actual: [...]

### Non-Critical Issues
1. [Description]
   - Severity: [Medium/Low]
   - Steps to reproduce: [...]
   - Expected: [...]
   - Actual: [...]

## Recommendations

1. [Fix for critical issue]
2. [Improvement suggestion]
3. [Test enhancement]

## Screenshots
- [List of screenshot paths]
```

## Error Handling and Debugging

### Common Issues and Solutions

**Issue 1: Document Upload Fails**
- **Symptoms:** Upload button doesn't work or returns error
- **Possible Causes:**
  - File not found
  - Invalid file format
  - File size exceeds limit
  - Backend service down
- **Debug Steps:**
  1. Check file exists: `ls -la tests/testdata/test.pdf`
  2. Check backend logs: `./dev.sh logs app`
  3. Verify MinIO is running: `docker compose ps minio`
  4. Check network connectivity

**Issue 2: Document Processing Stalls**
- **Symptoms:** Status stuck on "processing"
- **Possible Causes:**
  - OCR service not responding
  - GPU not available
  - Memory issues
  - Celery worker not running
- **Debug Steps:**
  1. Check Celery logs: `./dev.sh logs celery`
  2. Check GPU: `nvidia-smi`
  3. Check memory: `docker stats`
  4. Verify OCR model is loaded

**Issue 3: Query Returns "Unknown Document"**
- **Symptoms:** Sources show "Unknown Document (Page X)"
- **Possible Causes:**
  - Bug in metadata propagation
  - Milvus metadata missing
  - Document not fully processed
- **Debug Steps:**
  1. Check document status is "completed"
  2. Verify Milvus metadata
  3. Check backend logs
  4. Re-upload document if needed

**Issue 4: Query Timeout**
- **Symptoms:** Query takes > 30 seconds or times out
- **Possible Causes:**
  - Backend overloaded
  - Database issues
  - Network issues
  - Reranker slow
- **Debug Steps:**
  1. Check backend logs: `./dev.sh logs app`
  2. Check database: `./dev.sh logs postgres`
  3. Check reranker service
  4. Reduce parameter values (sources, reranker)

**Issue 5: Chrome MCP Connection Issues**
- **Symptoms:** Cannot connect to Chrome or control page
- **Possible Causes:**
  - Chrome not running
  - Port not accessible
  - MCP server not started
- **Debug Steps:**
  1. Start Chrome with remote debugging
  2. Verify MCP server is running
  3. Check network connectivity
  4. Restart Chrome MCP

## Test Metrics

### Key Performance Indicators (KPIs)

1. **Document Upload Time**
   - Target: < 5 seconds
   - Maximum: 10 seconds

2. **Document Processing Time**
   - Target: < 30 seconds
   - Maximum: 60 seconds

3. **Query Response Time**
   - Target: < 5 seconds
   - Maximum: 30 seconds (95th percentile)

4. **System Availability**
   - Target: 99.9%
   - Measurement: Uptime during tests

5. **Error Rate**
   - Target: < 1%
   - Measurement: Failed queries / Total queries

### Performance Benchmarks

| Operation | Target | Acceptable | Maximum |
|-----------|--------|------------|---------|
| Document Upload | < 5s | < 10s | 30s |
| OCR Processing | < 20s | < 40s | 60s |
| Embedding Generation | < 5s | < 10s | 20s |
| Vector Search | < 1s | < 2s | 5s |
| Reranking | < 2s | < 5s | 10s |
| LLM Generation | < 3s | < 10s | 20s |
| Total Query Time | < 5s | < 15s | 30s |

## Test Maintenance

### Update Triggers

This test plan should be updated when:
1. New features are added to the application
2. UI changes are made
3. New parameter options are added
4. Bug fixes are implemented
5. Performance requirements change

### Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-01-03 | Initial test plan creation | Tech Team |

## Appendix

### Appendix A: Test Questions Generation

The test questions are dynamically generated based on the PDF content:

1. **Basic Question:** Extracts fundamental definition
2. **Intermediate Question:** Explores types and features
3. **Advanced Question:** Discusses applications and use cases

### Appendix B: Parameter Matrix

Full test matrix showing all 48 combinations:

| # | Question | Reranker | Sources | Language | Test Case ID |
|---|----------|----------|---------|----------|--------------|
| 1 | Q1-JA | On | 1 | ja | TC-3.1.1 |
| 2 | Q1-JA | On | 5 | ja | TC-3.1.2 |
| 3 | Q1-JA | On | 10 | ja | TC-3.1.3 |
| 4 | Q1-JA | On | 20 | ja | TC-3.1.4 |
| 5 | Q1-JA | Off | 1 | ja | TC-3.1.5 |
| 6 | Q1-JA | Off | 5 | ja | TC-3.1.6 |
| 7 | Q1-JA | Off | 10 | ja | TC-3.1.7 |
| 8 | Q1-JA | Off | 20 | ja | TC-3.1.8 |
| 9 | Q2-JA | On | 1 | ja | TC-3.2.1 |
| 10 | Q2-JA | On | 5 | ja | TC-3.2.2 |
| ... | ... | ... | ... | ... | ... |
| 41 | Q3-EN | Off | 5 | en | TC-3.4.6 |
| 42 | Q3-EN | Off | 10 | en | TC-3.4.7 |
| 43 | Q3-EN | Off | 20 | en | TC-3.4.8 |
| 44 | Q1-EN | On | 1 | en | TC-3.3.1 |
| 45 | Q1-EN | On | 5 | en | TC-3.3.2 |
| 46 | Q1-EN | On | 10 | en | TC-3.3.3 |
| 47 | Q1-EN | On | 20 | en | TC-3.3.4 |
| 48 | Q1-EN | Off | 1 | en | TC-3.3.5 |

### Appendix C: Chrome MCP Commands Reference

**Navigation Commands:**
```python
await navigate_page(url="http://localhost:8501/")
await take_snapshot()
await take_screenshot(path="screenshot.png")
```

**Interaction Commands:**
```python
await click(uid="element_uid")
await fill(uid="input_uid", value="text")
await hover(uid="element_uid")
```

**Query Commands:**
```python
await wait_for(text="Expected Text", timeout=30000)
await evaluate_script(function="() => document.title")
```

### Appendix D: Validation Criteria

**Document Title Validation:**
- ✅ Must not be "Unknown Document"
- ✅ Must match document.title in database
- ✅ Must be non-empty string

**Source Metadata Validation:**
- ✅ document_id: Valid UUID format
- ✅ document_title: Non-empty, not "Unknown Document"
- ✅ page_number: Integer >= 1
- ✅ chunk_index: Integer >= 0
- ✅ chunk_text: Non-empty string
- ✅ score: Float in range [0.0, 1.0]
- ✅ rerank_score: Float in range [0.0, 1.0] (if reranker enabled)

**Answer Quality Validation:**
- ✅ Relevant to question
- ✅ In correct language
- ✅ Not empty or placeholder
- ✅ Based on retrieved sources
- ✅ Coherent and readable

---

**End of Test Plan**

For questions or issues, refer to:
- CLAUDE.md - Developer guide
- README.md - System documentation
- Project issues tracker
