"""
Streamlit Frontend Application
Admin UI for Japanese OCR RAG System
"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import List, Dict, Any

# Configuration
st.set_page_config(
    page_title="Japanese OCR RAG System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"


# Session state initialization
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = None
if "user" not in st.session_state:
    st.session_state.user = None


def api_request(
    method: str,
    endpoint: str,
    data: Dict[str, Any] = None,
    params: Dict[str, Any] = None,
    files: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Make API request"""
    url = f"{API_BASE_URL}{endpoint}"
    headers = {}
    if st.session_state.access_token:
        headers["Authorization"] = f"Bearer {st.session_state.access_token}"

    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, params=params, timeout=30)
        elif method.upper() == "POST":
            if files:
                response = requests.post(
                    url, headers=headers, data=data, files=files, timeout=60
                )
            else:
                response = requests.post(url, headers=headers, json=data, timeout=30)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return {}


def login(email: str, password: str) -> bool:
    """Login user"""
    data = api_request("POST", "/auth/login", {"email": email, "password": password})
    if "access_token" in data:
        st.session_state.access_token = data["access_token"]
        st.session_state.refresh_token = data["refresh_token"]
        st.session_state.user = data["user"]
        st.session_state.authenticated = True
        return True
    return False


def logout():
    """Logout user"""
    st.session_state.authenticated = False
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.session_state.user = None
    st.rerun()


# ============================================
# SIDEBAR
# ============================================


def render_sidebar():
    """Render sidebar navigation"""
    with st.sidebar:
        st.title("📄 Japanese OCR RAG")
        st.markdown("---")

        if st.session_state.authenticated:
            st.success(f"Logged in as: {st.session_state.user.get('full_name', 'User')}")
            st.caption(f"Role: {st.session_state.user.get('role', 'user')}")

            st.markdown("---")

            page = st.radio(
                "Navigation",
                [
                    "🏠 Dashboard",
                    "📚 Documents",
                    "🔍 Query",
                    "⚙️ Settings",
                ],
                label_visibility="collapsed",
            )

            st.markdown("---")
            if st.button("Logout"):
                logout()

        else:
            page = "login"

        return page


# ============================================
# LOGIN PAGE
# ============================================


def render_login():
    """Render login page"""
    st.markdown(
        """
        <div style='text-align: center; padding: 50px;'>
            <h1>📄 Japanese OCR RAG System</h1>
            <p>Production-grade RAG system for Japanese PDF document processing</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.subheader("Login")
            email = st.text_input("Email", placeholder="user@example.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

            if submitted:
                if email and password:
                    if login(email, password):
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password")
                else:
                    st.warning("Please enter email and password")


# ============================================
# DASHBOARD PAGE
# ============================================


def render_dashboard():
    """Render dashboard page"""
    st.title("🏠 Dashboard")
    st.markdown("---")

    # System stats
    stats = api_request("GET", "/admin/stats")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Documents", stats.get("documents", {}).get("total", 0))
    with col2:
        st.metric("Total Queries", stats.get("queries", {}).get("total", 0))
    with col3:
        st.metric("Total Users", stats.get("users", {}).get("total", 0))
    with col4:
        st.metric("Storage (GB)", stats.get("storage", {}).get("object_storage_size_gb", 0))

    st.markdown("---")

    # Recent documents
    st.subheader("Recent Documents")
    documents = api_request("GET", "/documents", params={"limit": 5})

    if documents.get("results"):
        for doc in documents["results"]:
            with st.expander(f"📄 {doc.get('filename', 'Unknown')}"):
                col1, col2, col3 = st.columns(3)
                col1.write(f"**Status:** {doc.get('status', 'unknown')}")
                col2.write(f"**Pages:** {doc.get('page_count', 0)}")
                col3.write(f"**Chunks:** {doc.get('chunk_count', 0)}")
    else:
        st.info("No documents found. Upload a PDF to get started.")


# ============================================
# DOCUMENTS PAGE
# ============================================


def render_documents():
    """Render documents page"""
    st.title("📚 Documents")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            label_visibility="collapsed",
        )
    with col2:
        st.write("")
        st.write("")
        title = st.text_input("Title (optional)")
        author = st.text_input("Author (optional)")
        keywords = st.text_input("Keywords (comma-separated)")

    if uploaded_file:
        metadata = {}
        if title:
            metadata["title"] = title
        if author:
            metadata["author"] = author
        if keywords:
            metadata["keywords"] = [k.strip() for k in keywords.split(",")]

        if st.button("Upload", use_container_width=True, type="primary"):
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            data = {"metadata": json.dumps(metadata)} if metadata else None

            result = api_request("POST", "/documents/upload", data=data, files=files)
            if "document_id" in result:
                st.success(f"Document uploaded! ID: {result['document_id']}")
                st.rerun()

    st.markdown("---")
    st.subheader("Document List")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox("Status", ["All", "pending", "processing", "completed", "failed"])
    with col2:
        search = st.text_input("Search")
    with col3:
        limit = st.number_input("PerPage", min_value=5, max_value=100, value=20)

    # Fetch documents
    params = {"limit": limit}
    if status_filter != "All":
        params["status"] = status_filter
    if search:
        params["search"] = search

    documents = api_request("GET", "/documents", params=params)

    if documents.get("results"):
        for doc in documents["results"]:
            with st.expander(f"📄 {doc.get('filename', 'Unknown')} - {doc.get('status', 'unknown').upper()}"):
                col1, col2, col3, col4 = st.columns(4)
                col1.write(f"**Title:** {doc.get('title', 'N/A')}")
                col2.write(f"**Pages:** {doc.get('page_count', 0)}")
                col3.write(f"**Chunks:** {doc.get('chunk_count', 0)}")
                col4.write(f"**OCR:** {doc.get('ocr_confidence', 0):.2%}" if doc.get('ocr_confidence') else "**OCR:** N/A")

                if st.button(f"View Details", key=f"view_{doc.get('document_id')}"):
                    st.json(doc)
    else:
        st.info("No documents found")


# ============================================
# QUERY PAGE
# ============================================


def render_query():
    """Render query page"""
    st.title("🔍 Query")
    st.markdown("---")

    # Query input
    st.subheader("Ask a Question")
    query = st.text_area(
        "Enter your question about the documents",
        placeholder="例：2025年第4四半期の営業利益は？",
        height=100,
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.slider("Number of sources", 1, 20, 5)
    with col2:
        rerank = st.checkbox("Use Reranker", value=True)
    with col3:
        language = st.selectbox("Language", ["ja", "en"])

    if st.button("Submit Query", use_container_width=True, type="primary"):
        if query:
            with st.spinner("Processing..."):
                result = api_request(
                    "POST",
                    "/query",
                    data={
                        "query": query,
                        "top_k": top_k,
                        "rerank": rerank,
                        "language": language,
                    },
                )

            if "answer" in result:
                st.subheader("Answer")
                st.write(result["answer"])

                st.caption(f"Processing time: {result.get('processing_time_ms', 0)}ms | Confidence: {result.get('confidence', 0):.2%}")

                if result.get("sources"):
                    st.subheader("Sources")
                    for i, source in enumerate(result["sources"], 1):
                        with st.expander(f"Source {i}: {source.get('document_title', 'Unknown')} (Page {source.get('page_number', 0)})"):
                            st.write(f"**Relevance:** {source.get('relevance_score', 0):.2%}")
                            if source.get("chunk_text"):
                                st.write(f"**Text:** {source['chunk_text'][:200]}...")
        else:
            st.warning("Please enter a question")


# ============================================
# SETTINGS PAGE
# ============================================


def render_settings():
    """Render settings page"""
    st.title("⚙️ Settings")
    st.markdown("---")

    st.subheader("API Configuration")
    api_url = st.text_input("API URL", value=API_BASE_URL)

    st.subheader("OCR Settings")
    ocr_engine = st.selectbox("OCR Engine", ["yomitoku", "paddleocr"])
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.85)

    st.subheader("Embedding Settings")
    embedding_model = st.selectbox(
        "Embedding Model",
        ["sbintuitions/sarashina-embedding-v1-1b"],
    )
    batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=64)

    if st.button("Save Settings"):
        st.success("Settings saved!")


# ============================================
# MAIN APP
# ============================================


def main():
    """Main application"""
    page = render_sidebar()

    if page == "login":
        render_login()
    elif page == "🏠 Dashboard":
        render_dashboard()
    elif page == "📚 Documents":
        render_documents()
    elif page == "🔍 Query":
        render_query()
    elif page == "⚙️ Settings":
        render_settings()


if __name__ == "__main__":
    main()
