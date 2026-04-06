"""Atlas RAG – Streamlit frontend."""
import streamlit as st
import requests
from pathlib import Path

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Atlas – Engineering Knowledge Assistant",
    page_icon="telescope",
    layout="wide",
)

def api_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json()
    except Exception:
        return {}


def api_ingest_files(files) -> dict:
    file_tuples = [("files", (f.name, f.getvalue(), f.type)) for f in files]
    r = requests.post(f"{API_BASE}/ingest/upload", files=file_tuples, timeout=120)
    r.raise_for_status()
    return r.json()


def api_ingest_directory() -> dict:
    r = requests.post(f"{API_BASE}/ingest/directory", timeout=120)
    r.raise_for_status()
    return r.json()


def api_ask(question: str, top_k: int) -> dict:
    payload = {"question": question, "top_k": top_k}
    r = requests.post(f"{API_BASE}/query/ask", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


with st.sidebar:
    st.image("https://raw.githubusercontent.com/google/material-design-icons/master/png/action/explore/materialicons/48dp/1x/baseline_explore_black_48dp.png", width=48)
    st.title("Atlas")
    st.caption("Engineering Knowledge Assistant")
    st.divider()

    health = api_health()
    if health:
        st.success("API connected")
        st.metric("Indexed chunks", health.get("index_size", 0))
        st.caption(f"Model: `{health.get('embedding_model', '—')}`")
        st.caption(f"Generation: `{health.get('generation_mode', '—')}`")
    else:
        st.error("API offline — start the backend first")
        st.code("python -m uvicorn backend.main:app --reload", language="bash")

    st.divider()
    top_k = st.slider("Retrieved chunks (top-k)", 1, 10, 5)

tab_ask, tab_ingest, tab_about = st.tabs(["Ask", "Ingest Documents", "About"])

with tab_ask:
    st.header("Ask a Question")
    st.caption("Atlas answers strictly from your ingested documents and cites its sources.")

    example_questions = [
        "What is the maximum operating temperature of the heat exchanger?",
        "Describe the failover procedure for the primary database.",
        "What are the load-bearing requirements for the steel frame?",
        "How does the pressure relief valve work?",
    ]

    with st.expander("Example questions"):
        for q in example_questions:
            if st.button(q, key=f"eq_{q[:20]}"):
                st.session_state["question_input"] = q

    question = st.text_area(
        "Your question",
        value=st.session_state.get("question_input", ""),
        placeholder="Ask anything about your ingested documents…",
        height=100,
        key="question_input",
    )

    if st.button("Ask Atlas", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving and generating answer…"):
                try:
                    result = api_ask(question.strip(), top_k)
                    st.success("Answer")
                    st.markdown(f"**{result['answer']}**")
                    st.divider()
                    st.subheader("Sources")
                    for i, src in enumerate(result["sources"], 1):
                        with st.expander(f"Source {i}: `{src['source']}` — page {src['page']}"):
                            st.markdown(f"*Chunk #{src['chunk_index']}*")
                            st.text(src["snippet"] + "…")
                    st.caption(f"Generation mode: `{result['mode']}`")
                except requests.HTTPError as e:
                    st.error(f"API error: {e.response.json().get('detail', str(e))}")
                except Exception as e:
                    st.error(f"Error: {e}")

with tab_ingest:
    st.header("Ingest Documents")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload files")
        uploaded = st.file_uploader(
            "Upload PDF, DOCX, TXT, or MD files",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
        )
        if st.button("Upload & Ingest", disabled=not uploaded, use_container_width=True):
            with st.spinner("Ingesting…"):
                try:
                    result = api_ingest_files(uploaded)
                    st.success(
                        f"Ingested {len(result['documents'])} document(s) "
                        f"-> {result['total_chunks']} chunks. "
                        f"Index size: {result['index_size']}"
                    )
                    for doc in result["documents"]:
                        st.write(f"• `{doc['source']}` — {doc['num_chunks']} chunks")
                except requests.HTTPError as e:
                    st.error(f"Error: {e.response.json().get('detail', str(e))}")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        st.subheader("Ingest sample_data directory")
        st.caption("Indexes all documents in the `sample_data/` folder (pre-loaded with engineering docs).")
        if st.button("Ingest Sample Data", use_container_width=True):
            with st.spinner("Ingesting sample_data…"):
                try:
                    result = api_ingest_directory()
                    st.success(
                        f"{len(result['documents'])} document(s) -> {result['total_chunks']} chunks"
                    )
                except requests.HTTPError as e:
                    st.error(f"Error: {e.response.json().get('detail', str(e))}")
                except Exception as e:
                    st.error(f"Error: {e}")

with tab_about:
    st.header("About Atlas")
    st.markdown("""
**Atlas** is a production-grade Retrieval-Augmented Generation (RAG) system built for engineering knowledge management.

### Architecture
```
Documents → Ingestion → Chunking → Embeddings (sentence-transformers)
                                          ↓
                                   FAISS Vector Index
                                          ↓
User Question → Query Embedding → Semantic Search → Top-K Chunks
                                                          ↓
                                               Answer Generation
                                            (Local / OpenAI / Anthropic)
                                                          ↓
                                           Answer + Source Citations
```

### Tech Stack
| Component | Technology |
|---|---|
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector DB | FAISS (CPU) |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Config | Pydantic Settings + .env |
| Logging | Loguru |

### Generation Modes
- **local** – extractive, no API key needed (default)
- **openai** – GPT-3.5/4 via `OPENAI_API_KEY`
- **anthropic** – Claude via `ANTHROPIC_API_KEY`
    """)
