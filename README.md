# retrieval-arena


Evaluate and compare embedding models for retrieval-augmented generation across 10 metrics (4 retrieval + 6 generation). Results are accessible via a React dashboard, REST API, or an MCP server for LLM tool integration.

## Quick Start

```bash
docker compose up --build -d
```

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:8000 |
| API | http://localhost:8765 |

## Tech Stack

- **Frontend**: React 19, JavaScript, Tailwind CSS v4, shadcn/ui, Vite
- **Backend**: Python 3.12, FastAPI, Sentence-Transformers, Celery + Redis
- **Database**: PostgreSQL 16, local file-based vector cache
- **Infrastructure**: Docker Compose

## Project Structure

```
├── backend/
│   ├── rag/              # RAG pipeline (retrieval, generation, evaluation)
│   ├── api/              # FastAPI server with SSE streaming
│   ├── mcp/              # MCP server for LLM tool integration
│   └── combined_app.py   # Routes all traffic in single container
├── frontend/             # React SPA
│   ├── src/components/   # 18 components (layout, benchmark, UI primitives)
│   ├── src/hooks/        # Custom React hooks
│   └── src/lib/          # API client and utilities
├── data/                 # Datasets, queries, vector cache
├── Dockerfile            # Multi-stage build
└── docker-compose.yaml   # Full stack orchestration
```

## Features

- **10 evaluation metrics**: Hit Rate, MRR, Precision, NDCG, Exact Match, ROUGE-L, Semantic Similarity, Faithfulness, Answer Relevancy, LLM Quality Score
- **10 embedding models**: From 33M-param MiniLM to 600M-param Qwen3, all running locally via Sentence-Transformers
- **Optional LLM integration**: Uses HuggingFace models for answer generation (threaded subprocess, freed after each call)
- **Real-time progress**: SSE streaming from backend to frontend during evaluations
- **Dataset management**: Create, edit, and delete datasets through the sidebar
- **Multi-query evaluation**: Run multiple queries across multiple models in a single stream
- **MCP protocol**: Exposes all RAG tools for Claude Desktop integration

## Frontend Development

```bash
cd frontend
npm install
npm run dev
```

Opens at `http://localhost:5173` — proxies `/api/*` to `http://localhost:8765`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/models` | List embedding models |
| GET | `/api/datasets` | List datasets |
| GET | `/api/datasets/{name}/documents` | Get dataset documents |
| PUT | `/api/datasets/{name}` | Create or overwrite a dataset |
| DELETE | `/api/datasets/{name}` | Delete a dataset |
| GET | `/api/queries` | List evaluation queries |
| POST | `/api/run` | Single-model RAG with SSE |
| POST | `/api/compare` | Multi-model comparison with SSE |
| POST | `/api/compare-multi` | Multi-query × multi-model with SSE |
| POST | `/api/ingest-url` | Fetch URL, chunk, create dataset |

## Batch Experiment CLI (Thesis Runs)

```bash
# Full sweep: all 10 models × 7 datasets × top-k=3,5,10
python -m backend.cli.experiment --all --top-k 3,5,10 --name thesis --yes

# Dry-run to preview plan
python -m backend.cli.experiment --all --top-k 5 --dry-run

# Custom subset
python -m backend.cli.experiment --models minilm-l12,bge-small --top-k 5 --datasets cars,cuisines --name quick_test
```

Outputs per `top_k` saved to `data/runs/`:
- `{name}_k{top_k}.csv` — flattened results (query × model)
- `{name}_k{top_k}_aggregate.csv` — mean/std/min/max per model per metric
- Meta JSON with run configuration

## MCP Tools (Claude Desktop)

Configure in `claude_desktop_config.json`:

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

9 tools available: dataset CRUD, query listing, async RAG jobs with polling, and result caching.
