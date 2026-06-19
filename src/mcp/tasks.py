"""Async Celery tasks for RAG pipeline"""
import os
from datetime import datetime
from pathlib import Path
from celery import Celery, Task
from celery.exceptions import SoftTimeLimitExceeded

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
celery_app = Celery("rag", broker=redis_url, backend=redis_url)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_soft_time_limit=600,  # 10 min
    task_time_limit=660,  # 11 min hard limit
    result_expires=86400,  # 1 day
)

RESULTS_FILE = Path(os.getenv("RESULTS_DIR", "/data")) / "results.jsonl"
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)


class RAGTask(Task):
    """Base task with error handling"""
    autoretry_for = (Exception,)
    max_retries = 2
    default_retry_delay = 60


@celery_app.task(base=RAGTask, bind=True)
def run_rag_task(self, job_id: str, query: str, model: str, dataset: str):
    """Run RAG pipeline async"""
    try:
        from rag.pipeline import RAGPipeline
        from rag.config import EMBEDDING_MODELS
        import json
        
        pipeline = RAGPipeline()
        
        # Load dataset
        from rag.experiment import load_dataset
        documents = load_dataset(dataset)
        ground_truth = "placeholder"  # Get from query config
        
        models_to_run = (
            list(EMBEDDING_MODELS.keys())
            if model == "all"
            else [model]
        )
        
        results = []
        for m in models_to_run:
            try:
                result = pipeline.run(
                    query=query,
                    documents=documents,
                    ground_truth=ground_truth,
                    dataset_name=dataset,
                    embedding_model=m,
                )
                results.append(result.model_dump())
            except Exception as e:
                results.append({
                    "error": str(e),
                    "model": m,
                    "query": query,
                })
        
        # Write results
        output = {
            "job_id": job_id,
            "query": query,
            "dataset": dataset,
            "models_run": models_to_run,
            "results": results,
            "completed_at": datetime.utcnow().isoformat(),
        }
        
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(output) + "\n")
        
        # Cache in Redis
        import redis
        r = redis.from_url(redis_url)
        r.setex(f"rag:result:{job_id}", 86400, json.dumps(output))
        
        return output
    
    except SoftTimeLimitExceeded:
        return {"error": "Task timeout (10 min)", "job_id": job_id}
    except Exception:
        raise