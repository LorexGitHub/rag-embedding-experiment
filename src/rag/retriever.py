import multiprocessing as mp

from .config import EMBEDDING_MODELS, RETRIEVAL_TOP_K
from .schemas import RetrievalResult

# Config keys that should NOT be forwarded to SentenceTransformer()
_MODEL_KWARGS_SKIP = {"model_name", "default_task", "encode_kwargs", "size", "memory", "speed"}


def _retrieve_worker(
    query: str,
    documents: list[str],
    model_id: str,
    model_kwargs: dict,
    encode_kwargs: dict,
    top_k: int,
    result_queue: mp.Queue,
):
    """Run inside a child process so all model memory is freed on exit."""
    try:
        from sentence_transformers import SentenceTransformer, util

        model = SentenceTransformer(model_id, **model_kwargs)
        doc_embs = model.encode(documents, convert_to_tensor=True, **encode_kwargs)
        query_emb = model.encode(query, convert_to_tensor=True, **encode_kwargs)
        scores = util.cos_sim(query_emb, doc_embs)[0]
        top_indices = scores.argsort(descending=True)[:top_k].tolist()

        result_queue.put({
            "documents": [documents[i] for i in top_indices],
            "scores": [float(scores[i]) for i in top_indices],
        })
    except Exception as e:
        result_queue.put({"error": str(e)})


class Retriever:
    @classmethod
    def retrieve(
        cls,
        query: str,
        documents: list[str],
        model_name: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> RetrievalResult:
        model_cfg = EMBEDDING_MODELS[model_name]
        model_id = model_cfg["model_name"]

        model_kwargs = {k: v for k, v in model_cfg.items() if k not in _MODEL_KWARGS_SKIP}

        encode_kwargs = {}
        task = model_cfg.get("default_task")
        if task:
            encode_kwargs["task"] = task
        extra_encode = model_cfg.get("encode_kwargs")
        if extra_encode:
            encode_kwargs.update(extra_encode)

        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(
            target=_retrieve_worker,
            args=(query, documents, model_id, model_kwargs, encode_kwargs, top_k, q),
        )
        p.start()
        try:
            data = q.get(timeout=300)
        except Exception:
            p.terminate()
            p.join(timeout=10)
            raise RuntimeError("Retrieval subprocess timed out or failed")
        p.join(timeout=10)

        if "error" in data:
            raise RuntimeError(data["error"])

        return RetrievalResult(
            documents=data["documents"],
            scores=data["scores"],
            model_name=model_name,
            top_k=top_k,
        )
