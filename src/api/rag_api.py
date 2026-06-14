from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
from pathlib import Path
import threading
import queue

from src.rag.experiment import run_experiment, run_batch, load_queries, load_dataset, DATA_DIR
from src.rag.schemas import ExperimentConfig
from src.rag.config import EMBEDDING_MODELS
from src.rag.pipeline import RAGPipeline

app = FastAPI(title="RAG Evaluation API")


class SingleRunRequest(BaseModel):
    query: str
    ground_truth: str
    dataset_name: str
    embedding_model: str
    top_k: int = 5


class CompareRequest(BaseModel):
    query: str
    ground_truth: str
    dataset_name: str
    embedding_models: Optional[list[str]] = None
    top_k: int = 5


class BatchRunRequest(BaseModel):
    embedding_models: Optional[list[str]] = None
    top_k: int = 5


@app.get("/models")
def list_models():
    return {"available_models": list(EMBEDDING_MODELS.keys())}


@app.get("/datasets")
def list_datasets():
    datasets_path = DATA_DIR / "datasets.json"
    with open(datasets_path) as f:
        datasets = json.load(f)
    return {"available_datasets": list(datasets.keys())}


@app.get("/datasets/{dataset_name}")
def get_dataset_categories(dataset_name: str):
    datasets_path = DATA_DIR / "datasets.json"
    with open(datasets_path) as f:
        datasets = json.load(f)
    categories = datasets.get(dataset_name)
    if categories is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found.")
    return {"dataset_name": dataset_name, "categories": categories}


class UpdateDatasetRequest(BaseModel):
    categories: list[str]


@app.post("/datasets/{dataset_name}")
def update_dataset(dataset_name: str, req: UpdateDatasetRequest):
    datasets_path = DATA_DIR / "datasets.json"
    with open(datasets_path) as f:
        datasets = json.load(f)
    datasets[dataset_name] = req.categories
    try:
        with open(datasets_path, "w") as f:
            json.dump(datasets, f, indent=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save: {e}")
    return {"message": f"Dataset '{dataset_name}' updated", "categories": req.categories}


@app.get("/queries")
def list_queries():
    return {"queries": load_queries()}


@app.post("/run")
def run_single(req: SingleRunRequest):
    documents = load_dataset(req.dataset_name)
    msg_queue = queue.Queue()

    def on_stage(stage: str):
        msg_queue.put(("stage", stage))

    def run_pipeline():
        try:
            pipeline = RAGPipeline()
            result = pipeline.run(
                query=req.query,
                documents=documents,
                ground_truth=req.ground_truth,
                dataset_name=req.dataset_name,
                embedding_model=req.embedding_model,
                top_k=req.top_k,
                on_stage=on_stage,
            )
            msg_queue.put(("result", result))
        except Exception as e:
            msg_queue.put(("error", str(e)))

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    def event_stream():
        while True:
            msg_type, payload = msg_queue.get()
            if msg_type == "stage":
                yield f"data: {json.dumps({'type': 'stage', 'message': payload})}\n\n"
            elif msg_type == "result":
                yield f"data: {json.dumps({'type': 'result', 'result': payload.model_dump()})}\n\n"
                break
            elif msg_type == "error":
                yield f"data: {json.dumps({'type': 'error', 'message': payload})}\n\n"
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/compare")
def compare_models(req: CompareRequest):
    documents = load_dataset(req.dataset_name)
    models = req.embedding_models or list(EMBEDDING_MODELS.keys())
    msg_queue = queue.Queue()

    def run_compare():
        try:
            results = {}
            for model_name in models:
                msg_queue.put(("stage", f"Processing: {model_name}"))
                pipeline = RAGPipeline()
                result = pipeline.run(
                    query=req.query,
                    documents=documents,
                    ground_truth=req.ground_truth,
                    dataset_name=req.dataset_name,
                    embedding_model=model_name,
                    top_k=req.top_k,
                )
                results[model_name] = result.model_dump()
            from src.rag.experiment import _pick_best
            report = {
                "query": req.query,
                "ground_truth": req.ground_truth,
                "dataset": req.dataset_name,
                "results": results,
            }
            _pick_best_report(report)
            msg_queue.put(("result", report))
        except Exception as e:
            msg_queue.put(("error", str(e)))

    thread = threading.Thread(target=run_compare, daemon=True)
    thread.start()

    def event_stream():
        while True:
            msg_type, payload = msg_queue.get()
            if msg_type == "stage":
                yield f"data: {json.dumps({'type': 'stage', 'message': payload})}\n\n"
            elif msg_type == "result":
                yield f"data: {json.dumps({'type': 'result', 'result': payload})}\n\n"
                break
            elif msg_type == "error":
                yield f"data: {json.dumps({'type': 'error', 'message': payload})}\n\n"
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _pick_best_report(report: dict):
    scored = []
    for name, result in report["results"].items():
        if "error" in result:
            continue
        ev = result["evaluation"]
        exact_bonus = 1.0 if ev.get("exact_match") else 0.0
        composite = (
            exact_bonus * 50.0
            + ev.get("semantic_similarity", 0.0) * 25.0
            + ev.get("rouge_l_f1", 0.0) * 15.0
            + ((ev.get("llm_quality_score") or 0) / 5.0) * 10.0
        )
        scored.append((composite, name))
    if scored:
        scored.sort(key=lambda x: (-x[0], x[1]))
        best = scored[0][0]
        winners = [name for score, name in scored if score == best]
        report["best_model"] = ", ".join(sorted(winners)) if len(winners) > 1 else winners[0]


@app.post("/run-batch")
def run_batch_endpoint(req: BatchRunRequest):
    queries = load_queries()
    reports = run_batch(queries, req.embedding_models, req.top_k)
    return {"reports": reports, "total_queries": len(reports)}
