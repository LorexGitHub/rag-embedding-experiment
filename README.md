# Retrieval Arena

A framework for benchmarking and comparing embedding models in a retrieval-augmented generation (RAG) context. Access results through a Streamlit dashboard, a REST API, or an MCP server for LLM integration.

### Demo


https://github.com/user-attachments/assets/93e52b5e-fd6c-4495-9d49-ebdb948d0e4c



### The Problem

Choosing an embedding model for a retrieval pipeline involves trade-offs between speed, memory, and retrieval quality. Models vary wildly in capability — from lightweight 33M-parameter models to high-dimensional 600M-parameter ones — and the best choice depends on your specific domain and data. Without a structured comparison, teams rely on intuition rather than empirical evidence.

### The Solution

A containerized evaluation platform that runs a query against multiple embedding models side-by-side and scores each result using exact match, ROUGE-L, and semantic similarity. Three access paths:

- **Streamlit UI** — Interactive dashboard for single, comparison, and batch runs
- **FastAPI** — REST + SSE streaming for programmatic access
- **MCP Server** — Exposes RAG tools over the Model Context Protocol, allowing any MCP-compatible LLM to query and evaluate models externally

### Supported Embedding Models

**Fast (< 100M params):**
- **MiniLM-L12** — `sentence-transformers/all-MiniLM-L12-v2` (33M)
- **BGE-Small** — `BAAI/bge-small-en-v1.5` (33M)
- **GTE-Small** — `thenlper/gte-small` (33M)
- **Granite** — `ibm-granite/granite-embedding-small-english-r2` (102M)
- **Harrier** — `microsoft/harrier-oss-v1-270m` (270M)

**Medium (100–150M params):**
- **BGE-Base** — `BAAI/bge-base-en-v1.5` (110M)
- **MPNet** — `sentence-transformers/all-mpnet-base-v2` (110M)

**High-quality (300M+ params):**
- **Qwen3** — `Qwen/Qwen3-Embedding-0.6B` (600M)
- **Jina** — `jinaai/jina-embeddings-v5-text-small` (580M)
- **BGE-Large** — `BAAI/bge-large-en-v1.5` (335M)

### Project Structure

```
├── src/
│   ├── rag/              # RAG pipeline package
│   │   ├── config.py     # Environment-based configuration + model registry
│   │   ├── schemas.py    # Pydantic request/response models
│   │   ├── retriever.py  # In-process model caching with subprocess fallback
│   │   ├── generator.py  # LLM/template answer generation
│   │   ├── evaluator.py  # Evaluation metrics (EM, ROUGE, semantic sim, LLM judge)
│   │   ├── pipeline.py   # Orchestrates retrieval + generation + evaluation
│   │   └── experiment.py # Batch experiment runner
│   ├── api/
│   │   └── rag_api.py    # FastAPI with SSE streaming
│   ├── mcp/
│   │   ├── mcp_server.py # MCP SSE server (exposes tools over HTTP/SSE)
│   │   ├── stdio_server.py # Stdio entry point for Claude Desktop
│   │   └── tasks.py      # Celery async tasks
│   └── ui/
│       └── rag_ui.py     # Streamlit RAG experiment dashboard
├── data/
│   ├── datasets.json     # 11 category datasets (cars, cuisines, programming, etc.)
│   └── rag_queries.json  # 20 evaluation queries with ground truth
├── infra/
│   ├── main.tf           # Terraform for Hetzner CX23
│   └── nginx.conf        # Reverse proxy (MCP + Streamlit on port 80)
├── Dockerfile            # Python 3.12, CPU-based PyTorch
├── docker-compose.yaml   # 6 services (redis, rag-api, celery-worker, mcp-sse, nginx, rag-ui)
└── requirements.txt
```

### Tech Stack

- **Language**: Python 3.12
- **API**: FastAPI + Uvicorn with SSE streaming
- **ML**: Sentence-Transformers, PyTorch (CPU)
- **Retrieval**: Subprocess isolation (`multiprocessing.spawn`) per model, with in-process fallback for Celery workers
- **Async tasks**: Celery + Redis broker/backend
- **Frontend**: Streamlit with custom dark theme
- **MCP**: Model Context Protocol server for Claude Desktop integration
- **Infrastructure**: Docker Compose (local), Terraform + Hetzner (cloud)
- **Generator backends**: Local HF, OpenAI-compatible, Ollama, template fallback

### Architecture

```
                     ┌─────────────────┐
                     │  Claude Desktop  │
                     │  (MCP stdio)     │
                     └────────┬────────┘
                              │ docker exec -i
                     ┌────────▼────────┐
                     │    mcp-sse      │
                     │  (HTTP/SSE)     │
                     │  port 5100      │
                     └────────┬────────┘
                              │ Celery task
                     ┌────────▼────────┐
                     │  celery-worker  │
                     │  (async tasks)  │
                     │  ┌────────────┐ │
                     │  │ RAGPipeline│ │
                     │  └────────────┘ │
                     └────────┬────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
     ┌────────▼──────┐ ┌─────▼──────┐ ┌──────▼─────┐
     │    Redis      │ │  HF Hub    │ │  /data/    │
     │  (broker +    │ │ (downloads)│ │ (results)  │
     │   cache)      │ │           │ │            │
     └───────────────┘ └────────────┘ └────────────┘

  Browser ──► nginx:80 ──┬──► rag-ui:8501 (Streamlit)
                         └──► mcp-sse:5100 (/mcp/)

  Streamlit UI ──► rag-api:8002 ──► (direct RAG, no Celery)
```

### Getting Started (Local)

**Prerequisites**: Docker Desktop

1. Clone and start:
   ```bash
   docker compose up --build -d
   ```

2. Access the services:
   | Service        | URL                             |
   |----------------|---------------------------------|
   | Streamlit UI   | http://localhost:8766           |
   | FastAPI        | http://localhost:8765           |
   | MCP SSE        | http://localhost:5100/mcp/      |

3. Stop:
   ```bash
   docker compose down
   ```

### Connecting Claude Desktop

Add this to `claude_desktop_config.json` (located at `%LOCALAPPDATA%\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Roaming\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "rag": {
      "command": "docker",
      "args": ["exec", "-i", "retrieval-arena-mcp-sse-1", "python", "/app/src/mcp/stdio_server.py"]
    }
  }
}
```

Restart Claude Desktop — you'll see a hammer icon with 4 tools:

| Tool | Purpose |
|---|---|
| `submit_rag_job` | Submit async RAG query (returns job_id) |
| `check_job_status` | Poll job result by ID |
| `list_cached_results` | Browse recent results |
| `get_cached_result` | Fetch specific result by job_id |

The `submit_rag_job` tool also accepts an optional `ground_truth` parameter for evaluation. If omitted, evaluation metrics (EM, ROUGE-L, semantic similarity) will not be meaningful.

Example prompt: *"Submit a RAG job to find which tech company created the iPhone using minilm-l12 on the tech_companies dataset"*

### Generator Configuration

By default the app returns the top retrieved document as the "answer" (template mode). To use a real LLM, set environment variables on the `rag-api` service in `docker-compose.yaml`:

| Env var                          | Behavior                                                             |
| -------------------------------- | -------------------------------------------------------------------- |
| (none set)                       | Returns top retrieved document                                       |
| `LLM_BASE_URL`                   | OpenAI-compatible API (vLLM, OpenAI, etc.); set `LLM_API_KEY` if needed |
| `LLM_MODEL`                      | Local HuggingFace model (key or full model ID)                       |
| `LLM_USE_OLLAMA=1`               | Ollama (uses `OLLAMA_BASE_URL`, default `http://localhost:11434`)    |

### API Endpoints (FastAPI)

- `GET /models` — List available embedding models
- `GET /datasets` — List available datasets
- `GET /datasets/{name}` — Get categories for a dataset
- `POST /datasets/{name}` — Update categories
- `GET /queries` — List evaluation queries
- `POST /run` — Single-model RAG with SSE streaming
- `POST /compare` — All-model comparison with per-model SSE progress
- `POST /run-batch` — Batch experiment across 20 queries

### Cloud Deployment (Hetzner)

1. Set your API token:
   ```bash
   export TF_VAR_hcloud_token="your-hcloud-api-token"
   ```

2. Provision:
   ```bash
   cd infra
   terraform init
   terraform apply
   ```

   This creates a CX23 (2 vCPU, 4 GB RAM) with Docker + Docker Compose installed via cloud-init.

### Evaluation Metrics

- **Substring Match**: Binary — is the ground truth found within the generated answer?
- **ROUGE-L F1**: Measures longest common subsequence overlap
- **Semantic Similarity**: Cosine similarity between answer and ground truth embeddings
- **LLM Quality Score**: 1–5 rating from a judge LLM (requires generator LLM)
