# Atlas: Engineering Knowledge Assistant

Atlas is a Retrieval-Augmented Generation (RAG) system that enables question answering over engineering documents with source citations. The system includes a FastAPI backend, a Streamlit frontend, FAISS vector search, and an evaluation pipeline.

---

## Architecture

### Ingestion Pipeline

* Load documents
* Clean and normalize text
* Split into overlapping chunks
* Generate embeddings using sentence-transformers
* Store embeddings in FAISS index

### Query Pipeline

* Embed user query
* Retrieve top-K similar chunks from FAISS
* Generate answer from retrieved context
* Return answer with source citations

---

## Features

* Document ingestion (PDF, DOCX, TXT, Markdown)
* Semantic search using FAISS
* Source citations for all answers
* Local extractive mode (no API key required)
* Optional OpenAI or Anthropic generation
* FastAPI REST API
* Streamlit user interface
* Evaluation pipeline (context relevance, answer faithfulness, retrieval hit)
* Logging and test suite

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/sanjitchitturi/atlas-rag.git
cd atlas-rag
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create environment file

```bash
cp .env.example .env
```

### 5. Create required directories

```bash
mkdir logs
mkdir -p data/documents data/index
```

### 6. Ingest sample documents

```bash
python scripts/ingest_sample_data.py
```

---

## Running the Application

Start the backend:

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

Start the frontend:

```bash
streamlit run frontend/app.py
```

API documentation:

```
http://localhost:8000/docs
```

Streamlit interface:

```
http://localhost:8501
```

---

## API Example

Ask a question:

```bash
curl -X POST http://localhost:8000/query/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the maximum operating temperature?", "top_k": 5}'
```

---

## Evaluation

Run the evaluation pipeline:

```bash
python scripts/run_evaluation.py
```

### Metrics

* Context Relevance (CR)
* Answer Faithfulness (AF)
* Retrieval Hit (RH)
