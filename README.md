# Category-to-Vector: RAG Embedding Experiment

Benchmark how different embedding models affect retrieval-augmented generation (RAG) output quality. Run locally with Docker Compose or deploy to a Hetzner CX23 cloud server.

### The Problem

Different embedding models capture different semantic nuances. A query like *"A programming language named after a snake"* might retrieve different documents depending on whether you use MiniLM, BGE, GTE, Granite, Harrier, MPNet, Qwen3, or Jina. Which model works best for your domain and dataset?

### The Solution

A containerized RAG pipeline that runs a query through multiple embedding models, retrieves the top-K documents, generates an answer (via local LLM, OpenAI-compatible API, Ollama, or template fallback), and evaluates the result with Substring Match, ROUGE-L, Semantic Similarity, and LLM quality scoring.

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
│   │   ├── retriever.py  # Subprocess-isolated embedding retrieval
│   │   ├── generator.py  # LLM/template answer generation
│   │   ├── evaluator.py  # Evaluation metrics (EM, ROUGE, similarity, LLM)
│   │   ├── pipeline.py   # Orchestrates retrieval + generation + evaluation
│   │   └── experiment.py # Batch experiment runner
│   ├── api/
│   │   └── rag_api.py    # FastAPI with SSE streaming
│   └── ui/
│       ├── rag_ui.py     # Streamlit RAG experiment dashboard
│       └── ui.py         # Legacy dataset category manager
├── data/
│   ├── datasets.json     # 10 category datasets (cars, cuisines, tech, etc.)
│   └── rag_queries.json  # 20 evaluation queries with ground truth
├── infra/
│   └── main.tf           # Terraform for Hetzner CX23
├── Dockerfile            # Python 3.12, CPU-based torch
├── docker-compose.yaml   # rag-api + rag-ui services
└── requirements.txt
```

### Tech Stack

- **Language**: Python 3.12
- **API**: FastAPI + Uvicorn with SSE streaming
- **ML**: Sentence-Transformers, PyTorch (CPU)
- **Retrieval**: Subprocess isolation (`multiprocessing.spawn`) per model
- **Frontend**: Streamlit with custom dark theme
- **Infrastructure**: Docker Compose (local), Terraform + Hetzner (cloud)
- **Generator backends**: Local HF, OpenAI-compatible, Ollama, template fallback

### Getting Started (Local)

**Prerequisites**: Docker Desktop, Python 3.12+ (optional, for UI outside Docker)

1. Clone and start:
   ```
   docker compose up --build -d
   ```

2. Open the UI at [http://localhost:8501](http://localhost:8501)

   The API is at port 8002. Both services start automatically.

3. Compare models:
   - Select a dataset in the sidebar
   - Choose a sample query or type your own
   - Run a single model, compare all, or run a batch experiment

4. Stop:
   ```
   docker compose down
   ```

### Generator Configuration

Generator selection is explicit via environment variables (no auto-detection):

| Env var | Behavior |
|---|---|
| (none set) | `_TemplateGenerator` — returns top retrieved document |
| `LLM_BASE_URL` | OpenAI-compatible API (e.g., vLLM, OpenAI); set `LLM_API_KEY` if needed |
| `LLM_MODEL` or `LOCAL_LLM_MODEL` | Local HuggingFace model |
| `LLM_USE_OLLAMA=1` | Ollama (uses `OLLAMA_BASE_URL`, default `http://localhost:11434`) |

Set these in `docker-compose.yaml` under the `rag-api` service environment.

### API Endpoints

- `GET /models` — List available embedding models
- `GET /datasets` — List available datasets
- `GET /datasets/{name}` — Get categories for a dataset
- `POST /datasets/{name}` — Update categories
- `GET /queries` — List evaluation queries
- `POST /run` — Single-model RAG with SSE streaming (stages: loading → retrieval → generation → evaluation)
- `POST /compare` — All-model comparison with per-model SSE progress
- `POST /run-batch` — Batch experiment across 20 queries

### Cloud Deployment (Hetzner)

1. Set your API token:
   ```
   export TF_VAR_hcloud_token="your-hcloud-api-token"
   ```

2. Provision:
   ```
   cd infra
   terraform init
   terraform apply
   ```

   This creates a CX23 (2 vCPU, 4 GB RAM) with Docker + Docker Compose installed via cloud-init. The services start automatically on boot.

### Memory Management

Models are loaded one at a time in subprocess isolation (`multiprocessing.spawn`). Each model is loaded in a fresh child process, and all memory is fully freed when the process exits. This prevents page-file exhaustion on Windows and allows running all models sequentially even on machines with limited RAM.

### Evaluation Metrics

- **Substring Match**: Binary — is the ground truth found within the generated answer?
- **ROUGE-L F1**: Measures longest common subsequence overlap
- **Semantic Similarity**: Cosine similarity between answer and ground truth embeddings
- **LLM Quality Score**: 1–5 rating from a judge LLM (requires generator LLM)
