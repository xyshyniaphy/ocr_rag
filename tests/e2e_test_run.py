#!/usr/bin/env python3
"""
Complete E2E Test Execution for Japanese OCR RAG System
Based on test plan: tests/plans/main_plan.md
"""

import requests
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
API_BASE = "http://localhost:8000/api/v1"
TEST_PDF = "/app/tests/testdata/test.pdf"
ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "admin123"

# Test questions (6 total - 3 Japanese, 3 English)
QUESTIONS = [
    ("OCRとは何ですか？", "ja"),
    ("OCRエンジンの種類と特徴を教えてください", "ja"),
    ("OCRの活用事例と応用分野について教えてください", "ja"),
    ("What is OCR?", "en"),
    ("What are the types and features of OCR engines?", "en"),
    ("What are the use cases and application fields of OCR?", "en"),
]

# Test configurations
RERANKER_OPTIONS = [True, False]
SOURCE_COUNTS = [1, 5, 10, 20]


class E2ETester:
    def __init__(self):
        self.token = None
        self.document_id = None
        self.results = []
        self.start_time = None

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

    def phase1_setup_cleanup(self):
        """Phase 1: Login and clear all documents"""
        self.log("=== PHASE 1: Setup and Cleanup ===")

        # Login
        self.log(f"Logging in as {ADMIN_EMAIL}...")
        response = requests.post(
            f"{API_BASE}/auth/login",
            json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
        )
        response.raise_for_status()
        self.token = response.json()["access_token"]
        self.log("✅ Login successful")

        # Clear all documents
        self.log("Clearing all documents via API...")
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.delete(
            f"{API_BASE}/documents/all",
            headers=headers
        )
        response.raise_for_status()
        self.log(f"✅ Cleared: {response.json().get('deleted_count', 0)} documents")

        return True

    def phase2_upload_document(self):
        """Phase 2: Upload test document and wait for processing"""
        self.log("=== PHASE 2: Document Upload ===")

        # Read PDF file
        self.log(f"Reading test PDF: {TEST_PDF}")
        with open(TEST_PDF, "rb") as f:
            pdf_bytes = f.read()

        file_size = len(pdf_bytes)
        self.log(f"File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")

        # Upload
        self.log("Uploading document...")
        headers = {"Authorization": f"Bearer {self.token}"}
        files = {
            "file": ("test.pdf", pdf_bytes, "application/pdf")
        }
        response = requests.post(
            f"{API_BASE}/documents/upload",
            headers=headers,
            files=files
        )
        response.raise_for_status()
        data = response.json()
        self.document_id = data["id"]
        self.log(f"✅ Upload successful - Document ID: {self.document_id}")

        # Poll for completion
        self.log("Waiting for processing to complete...")
        max_wait = 180  # 3 minutes max
        start = time.time()

        while True:
            elapsed = time.time() - start
            if elapsed > max_wait:
                raise TimeoutError(f"Processing timeout after {max_wait}s")

            response = requests.get(
                f"{API_BASE}/documents/{self.document_id}/status",
                headers=headers
            )
            response.raise_for_status()
            status_data = response.json()
            status = status_data["status"]
            progress = status_data.get("progress", 0)

            if status == "completed":
                self.log(f"✅ Processing completed in {elapsed:.1f}s (100%)")
                break
            elif status == "failed":
                error = status_data.get("error", "Unknown error")
                raise Exception(f"Processing failed: {error}")
            else:
                self.log(f"  Status: {status} ({progress}%) - {elapsed:.1f}s elapsed")

            time.sleep(2)

        return True

    def phase3_query_testing(self):
        """Phase 3: Execute all query tests (48 combinations)"""
        self.log("=== PHASE 3: Query Testing ===")
        self.log(f"Test matrix: {len(QUESTIONS)} questions × {len(RERANKER_OPTIONS)} reranker × {len(SOURCE_COUNTS)} source_counts = {len(QUESTIONS) * len(RERANKER_OPTIONS) * len(SOURCE_COUNTS)} tests")

        headers = {"Authorization": f"Bearer {self.token}"}
        test_count = 0
        total_tests = len(QUESTIONS) * len(RERANKER_OPTIONS) * len(SOURCE_COUNTS)

        for question, language in QUESTIONS:
            for reranker in RERANKER_OPTIONS:
                for source_count in SOURCE_COUNTS:
                    test_count += 1
                    test_id = f"Q{QUESTIONS.index((question, language)) + 1}-{'JA' if language == 'ja' else 'EN'}_R{reranker}_S{source_count}"

                    self.log(f"  Test {test_count}/{total_tests}: {test_id} - {language.upper()} - Reranker={reranker} - Sources={source_count}")

                    # Execute query
                    query_start = time.time()
                    response = requests.post(
                        f"{API_BASE}/query",
                        headers=headers,
                        json={
                            "query": question,
                            "reranker": reranker,
                            "top_k": source_count,
                            "language": language
                        }
                    )
                    query_time = time.time() - query_start

                    if response.status_code != 200:
                        self.log(f"    ❌ ERROR: HTTP {response.status_code}")
                        self.results.append({
                            "test_id": test_id,
                            "question": question,
                            "language": language,
                            "reranker": reranker,
                            "source_count": source_count,
                            "status": "error",
                            "error": f"HTTP {response.status_code}",
                            "response_time": query_time
                        })
                        continue

                    data = response.json()

                    # Count "Unknown Document" errors
                    sources = data.get("sources", [])
                    unknown_count = sum(1 for s in sources if s.get("document_title") == "Unknown Document")

                    # Determine test status
                    status = "PASS" if unknown_count == 0 else "FAIL"

                    self.results.append({
                        "test_id": test_id,
                        "question": question,
                        "language": language,
                        "reranker": reranker,
                        "source_count": source_count,
                        "status": status,
                        "sources_returned": len(sources),
                        "unknown_document_errors": unknown_count,
                        "response_time": query_time,
                        "answer": data.get("answer", "")[:100] + "..." if len(data.get("answer", "")) > 100 else data.get("answer", "")
                    })

                    self.log(f"    Result: {status} - {len(sources)} sources - {unknown_count} unknown docs - {query_time:.2f}s")

        return True

    def phase4_validation(self):
        """Phase 4: Validate results"""
        self.log("=== PHASE 4: Validation ===")

        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        total_unknown = sum(r["unknown_document_errors"] for r in self.results)

        self.log(f"Tests Passed: {passed}/{len(self.results)} ({100 * passed / len(self.results):.1f}%)")
        self.log(f"Tests Failed: {failed}/{len(self.results)} ({100 * failed / len(self.results):.1f}%)")
        self.log(f"Total 'Unknown Document' Errors: {total_unknown}")

        return {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "unknown_errors": total_unknown
        }

    def generate_report(self):
        """Generate timestamped markdown report"""
        self.log("=== Generating Test Report ===")

        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        report_path = f"/app/test-results/t_{timestamp}.md"

        # Calculate statistics
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        total_unknown = sum(r["unknown_document_errors"] for r in self.results)
        avg_time = sum(r["response_time"] for r in self.results) / len(self.results)

        # Sort results by configuration
        results_by_reranker = {
            True: [r for r in self.results if r["reranker"]],
            False: [r for r in self.results if not r["reranker"]]
        }
        results_by_language = {
            "ja": [r for r in self.results if r["language"] == "ja"],
            "en": [r for r in self.results if r["language"] == "en"]
        }

        # Generate markdown
        report_lines = [
            "# E2E Test Execution Report",
            "",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC",
            f"**Test Plan:** tests/plans/main_plan.md (TP-001 v1.0)",
            f"**Tester:** Automated (Python API)",
            f"**Environment:** Development (Docker Compose)",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            f"**Status:** {'✅ PASSED' if passed == len(self.results) else '⚠️ PARTIAL SUCCESS' if passed > 0 else '❌ FAILED'}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Total Tests** | {len(self.results)} |",
            f"| **Passed** | {passed} ({100 * passed / len(self.results):.1f}%) |",
            f"| **Failed** | {failed} ({100 * failed / len(self.results):.1f}%) |",
            f"| **'Unknown Document' Errors** | {total_unknown} |",
            f"| **Avg Response Time** | {avg_time:.2f}s |",
            "",
            "---",
            "",
            "## Test Execution Summary",
            "",
            "### Phase 1: Setup and Cleanup ✅ PASSED",
            "",
            "**Status:** COMPLETED",
            "",
            "**Steps Completed:**",
            f"1. ✅ Authenticated as {ADMIN_EMAIL}",
            f"2. ✅ Cleared all documents via API",
            "3. ✅ System ready for testing",
            "",
            "---",
            "",
            "### Phase 2: Document Upload ✅ PASSED",
            "",
            "**Status:** COMPLETED",
            "",
            f"**Document ID:** {self.document_id}",
            "**Filename:** test.pdf",
            "",
            "---",
            "",
            "### Phase 3: Query Testing",
            "",
            f"**Status:** {'✅ PASSED' if passed == len(self.results) else '⚠️ PARTIAL' if passed > 0 else '❌ FAILED'}",
            "",
            "**Test Results:**",
            f"- **Tests Executed:** {len(self.results)}",
            f"- **Passed:** {passed} ({100 * passed / len(self.results):.1f}%)",
            f"- **Failed:** {failed} ({100 * failed / len(self.results):.1f}%)",
            f"- **'Unknown Document' Errors:** {total_unknown}",
            "",
            "---",
            "",
            "## Detailed Results",
            "",
            "### By Reranker Setting",
            "",
            "| Reranker | Total | Passed | Failed | Avg Time |",
            "|----------|-------|--------|--------|----------|",
        ]

        for reranker in [True, False]:
            results = results_by_reranker[reranker]
            p = sum(1 for r in results if r["status"] == "PASS")
            t = len(results)
            avg = sum(r["response_time"] for r in results) / t if t > 0 else 0
            report_lines.append(
                f"| {'On' if reranker else 'Off'} | {t} | {p} | {t - p} | {avg:.2f}s |"
            )

        report_lines.extend([
            "",
            "### By Language",
            "",
            "| Language | Total | Passed | Failed | Avg Time |",
            "|----------|-------|--------|--------|----------|",
        ])

        for lang in ["ja", "en"]:
            results = results_by_language[lang]
            p = sum(1 for r in results if r["status"] == "PASS")
            t = len(results)
            avg = sum(r["response_time"] for r in results) / t if t > 0 else 0
            report_lines.append(
                f"| {lang.upper()} | {t} | {p} | {t - p} | {avg:.2f}s |"
            )

        report_lines.extend([
            "",
            "## Sign-Off",
            "",
            f"**Test Execution:** Automated (Python API)",
            f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC",
            f"**Report File:** test-results/t_{timestamp}.md",
            f"**Test Plan Version:** TP-001 v1.0",
            f"**Status:** {'✅ PASSED' if passed == len(self.results) else '⚠️ PARTIAL SUCCESS' if passed > 0 else '❌ FAILED'}",
            "",
            "---",
            "",
            "**End of Report**"
        ])

        # Write report
        report_content = "\n".join(report_lines)
        Path(report_path).write_text(report_content)
        self.log(f"✅ Report saved: {report_path}")

        return report_path

    def run(self):
        """Execute complete test plan"""
        self.start_time = time.time()
        self.log("Starting E2E Test Execution")
        self.log("=" * 60)

        try:
            # Phase 1: Setup
            self.phase1_setup_cleanup()

            # Phase 2: Upload
            self.phase2_upload_document()

            # Phase 3: Query Testing
            self.phase3_query_testing()

            # Phase 4: Validation
            validation = self.phase4_validation()

            # Generate Report
            report_path = self.generate_report()

            total_time = time.time() - self.start_time
            self.log("=" * 60)
            self.log(f"Test execution completed in {total_time:.1f}s")
            self.log(f"Report: {report_path}")

            return validation["passed"] == len(self.results)

        except Exception as e:
            self.log(f"❌ Test execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    tester = E2ETester()
    success = tester.run()
    exit(0 if success else 1)
