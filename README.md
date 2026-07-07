# Simple RAG API

A lightweight RAG (Retrieval-Augmented Generation) API built with FastAPI, ChromaDB, and Gemini. Upload a document, ask questions about it, and get answers grounded in your actual content — not hallucinated.

## How it works

1. Document is loaded and split into overlapping chunks
2. Chunks are stored as vectors in ChromaDB
3. On each question, the most relevant chunks are retrieved via semantic search
4. Retrieved context is passed to Gemini to generate a grounded answer

## Tech stack

- **FastAPI** — API framework
- **ChromaDB** — local vector database
- **Google Gemini** — LLM for generation
- **Python** — everything else

## Setup

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your Gemini API key:
GEMINI_API_KEY=your_key_here
4. Add your document as `sample.txt` in the project root
5. Run: `python index.py`
6. Visit `http://localhost:8000/docs` to test the API

## Endpoints

- `GET /` — health check
- `POST /ask` — ask a question about your document with retrieved context
