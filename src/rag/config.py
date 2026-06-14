import os

# Each model entry can include:
#   model_name       — HuggingFace model ID (required)
#   size             — human-readable param count for UI
#   memory           — estimated RAM usage for UI
#   speed            — relative speed category (fast / medium / slow)
#   trust_remote_code  — passed to SentenceTransformer
#   default_task     — passed to model.encode(task=...)
#   encode_kwargs    — extra kwargs passed to model.encode()
EMBEDDING_MODELS = {
    # --- Ultra-fast baselines (< 50M params) ---
    "minilm-l12": {
        "model_name": "sentence-transformers/all-MiniLM-L12-v2",
        "size": "33M params",
        "memory": "~90 MB",
        "speed": "fast",
    },
    "bge-small": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "size": "33M params",
        "memory": "~90 MB",
        "speed": "fast",
        "encode_kwargs": {"normalize_embeddings": True},
    },
    "gte-small": {
        "model_name": "thenlper/gte-small",
        "size": "33M params",
        "memory": "~100 MB",
        "speed": "fast",
    },
    "granite": {
        "model_name": "ibm-granite/granite-embedding-small-english-r2",
        "size": "102M params",
        "memory": "~200 MB",
        "speed": "fast",
    },
    "harrier": {
        "model_name": "microsoft/harrier-oss-v1-270m",
        "size": "270M params",
        "memory": "~550 MB",
        "speed": "fast",
    },
    # --- Balanced models (100–150M params) ---
    "bge-base": {
        "model_name": "BAAI/bge-base-en-v1.5",
        "size": "110M params",
        "memory": "~250 MB",
        "speed": "medium",
        "encode_kwargs": {"normalize_embeddings": True},
    },
    "mpnet": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "size": "110M params",
        "memory": "~250 MB",
        "speed": "medium",
    },
    # --- Higher-quality models (300M+ params) ---
    "qwen3": {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "size": "600M params",
        "memory": "~1.2 GB",
        "speed": "medium",
    },
    "jina": {
        "model_name": "jinaai/jina-embeddings-v5-text-small",
        "size": "580M params",
        "memory": "~1.2 GB",
        "speed": "medium",
        "trust_remote_code": True,
        "default_task": "retrieval",
    },
    "bge-large": {
        "model_name": "BAAI/bge-large-en-v1.5",
        "size": "335M params",
        "memory": "~700 MB",
        "speed": "slow",
        "encode_kwargs": {"normalize_embeddings": True},
    },
}

LLM_MODELS = {
    # --- Tiny models (< 1 GB RAM) ---
    "smol":        "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smol360":     "HuggingFaceTB/SmolLM2-360M-Instruct",

    # --- Small models (1-2 GB, already cached) ---
    "qwen2":       "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5":     "Qwen/Qwen2.5-1.5B-Instruct",

    # --- Medium models (2-4 GB) ---
    "tinyllama":   "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "glm-edge":    "THUDM/glm-edge-1.5b-chat",
    "glm4":        "THUDM/glm-4-9b-chat",
}

DEFAULT_LLM = os.getenv("LLM_MODEL", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

GENERATOR = {
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "256")),
}

RETRIEVAL_TOP_K = 5

EVAL_SEMANTIC_MODEL = "all-MiniLM-L6-v2"
