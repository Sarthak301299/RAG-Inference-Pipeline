# RAG Inference Pipeline

A self-hosted Retrieval-Augmented Generation (RAG) service that ingests documents (including source code), indexes them in a vector database, and serves grounded, structured answers through a FastAPI backend powered by vLLM.

## What it does

1. **Ingests** documents from a local directory — PDFs, text, Markdown, and source code (`.py`, `.c`, `.cpp`) are all supported via type-aware loading.
2. **Chunks** each document using a strategy suited to its type — recursive/character splitting for prose, language-aware splitting for code, and header-aware splitting for Markdown.
3. **Embeds** chunks using any Hugging Face embedding model and indexes them into a vector store.
4. **Retrieves** the top candidate chunks for a query, **reranks** them with a cross-encoder, and passes the best matches to the generator.
5. **Generates** a structured, grounded answer via a self-hosted vLLM engine, constrained to a JSON schema (`thought_process`, `answer`, `sources`) so responses stay parseable and citation-aware.
6. **Serves** all of this through a FastAPI application with health-check endpoints and a terminal-based chat client.

## Architecture

```
Documents (data/)
      │
      ▼
   Loader ──► Chunker ──► Embedder ──► Indexer ──► Vector Store
                                                   (Qdrant / Chroma / pgvector)
                                                          │
User Query ─────────────────────────────────────────────► │
                                                          ▼
                                                     Retriever
                                                          │
                                                          ▼
                                                     Reranker
                                                          │
                                                          ▼
                                              Prompt Builder (schemas)
                                                          │
                                                          ▼
                                              Generator (vLLM, async)
                                                          │
                                                          ▼
                                            Structured JSON Response
                                        (thought_process, answer, sources)
```

The vector store, embedding model, reranker model, and generation model are all swappable via a single config file — no application code needs to change to switch between them.

## Repository structure

```
config/
  config.yml          # single source of configuration for the whole pipeline
src/
  ingestion/
    loader.py          # dynamic directory loader (Unstructured-based)
    chunker.py          # type-aware chunking (recursive / fixed / language-aware / markdown)
    embedder.py          # Hugging Face embedding wrapper
    indexer.py            # multi-backend vector store abstraction (Qdrant / Chroma / pgvector)
  retrieval/
    retriever.py         # top-k retrieval from the vector store
    reranker.py           # cross-encoder reranking
  generation/
    backend.py             # abstract Generator interface
    vllm.py                 # vLLM-based async implementation
    schemas.py               # structured output schema (Pydantic + vLLM structured outputs)
  pipeline/
    rag_pipeline.py           # orchestrates ingestion + retrieval + generation
    schemas.py                 # prompt construction from query + retrieved chunks
  api/
    server.py                    # FastAPI app: health checks + /generate endpoint
    client.py                      # terminal chat client (Rich-based)
    schemas.py                      # request/response models
tests/
  test_ingestion/, test_retrieval/, test_generation/, test_pipeline/, test_api/
```

Every module in `ingestion`, `retrieval`, `generation`, and `api` has a corresponding test module under `tests/`.

## Configuration

Everything is driven by `config/config.yml`, covering:
- **Vector store**: backend choice (`qdrant` / `chroma` / `pgvector`), distance metric, retrieval-to-rerank ratio, final chunk count, search type
- **Ingestion**: source directory, chunking strategy and size, embedding model and batch size
- **Retrieval**: reranker model and batch size
- **Generation**: backend (`vllm`), model name, sampling parameters, and structured-output length limits

## Running the API

```bash
# install dependencies
uv sync

# run the server
python -m src.api.server

# in another terminal, run the interactive client
python -m src.api.client
```

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /generate` | Submit a query, get a grounded response |
| `POST /invocations` | SageMaker-compatible alias for `/generate` |
| `GET /health/live` | Liveness check |
| `GET /health/ready` | Readiness check (pipeline initialized + context ingested) |
| `GET /health/startup` | Startup check |
| `GET /ping` | Alias for readiness check |

On startup, the API initializes the full pipeline (loader, embedder, indexer, retriever, reranker, generator) and ingests documents from the configured source directory before accepting traffic.

## Response format

Every answer is returned as structured JSON, enforced via vLLM's structured outputs:

```json
{
  "thought_process": "Brief internal reasoning before answering.",
  "answer": "The grounded answer, based only on retrieved context.",
  "sources": ["doc1.md", "doc2.py"]
}
```

The system prompt explicitly constrains the model to answer only from retrieved context, avoid fabricated citations, and flag conflicting information across sources.

## Tech stack

- **Ingestion/parsing**: `langchain-unstructured`, `langchain-text-splitters`
- **Embeddings**: `sentence-transformers`, `langchain-huggingface`
- **Vector stores**: `qdrant-client`/`langchain-qdrant`, `chromadb`/`langchain-chroma`, `langchain-postgres` + `pgembed`
- **Reranking**: `sentence-transformers` cross-encoders
- **Generation**: `vllm` (async engine, structured outputs)
- **API**: `FastAPI`, `uvicorn`
- **CLI client**: `rich`
- **Tooling**: `uv`, `ruff`, `black`, `pyright`, `pytest` + `pytest-asyncio` + `pytest-cov`, `pre-commit`, `locust` (load testing)

## Status

Core pipeline (ingestion → retrieval → rerank → generation → API) is implemented and under active testing. Load testing, evaluation harness, and observability instrumentation are not yet part of this codebase.
