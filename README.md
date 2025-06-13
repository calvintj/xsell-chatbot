# Jenius FX Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for Jenius by Bank BTPN, focused on Foreign Currency (FCY) services. Supports both English and Bahasa Indonesia, with CLI and API interfaces, and vector search powered by Pinecone and OpenAI embeddings.

## Features

- Language-aware chatbot (English & Bahasa Indonesia)
- Retrieval-augmented answers using Pinecone vector store
- FastAPI-based streaming API
- CLI terminal interface
- Modular scripts for data ingestion and vector building

## Directory Structure

```
src/
  app/         # Main app (CLI, API, routers)
  core/        # Core settings, logging, errors
  data/        # Data files (raw, processed)
  prompts/     # Prompt templates
  retrieval/   # Vector store, chunking, embeddings
  scripts/     # Data ingestion and utility scripts
  services/    # RAG, LLM, and business logic
  tests/       # Unit and integration tests
  utils/       # Utilities
```

## Setup

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Create a `.env` file** (see `.env.example` for required variables)

## Usage

### CLI Chatbot

```bash
PYTHONPATH=src python3 -m app.main
```

### FastAPI Server

```bash
PYTHONPATH=src uvicorn app.api:app --reload
```

### Ingest Data Scripts

Example:

```bash
PYTHONPATH=. python3 -m src.scripts.ingest_pdf_faq_en
```

## Running Tests

```bash
pytest
```

## Environment Variables

See `.env.example` for all required variables (OpenAI, Pinecone, etc).

## License

MIT (or specify your license)
