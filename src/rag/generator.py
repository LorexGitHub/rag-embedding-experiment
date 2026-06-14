import os
import json
import urllib.request
import urllib.error

from .config import LLM_MODELS, DEFAULT_LLM, LLM_BASE_URL, LLM_API_KEY, GENERATOR

_LLM_INSTANCE = None


def get_generator():
    global _LLM_INSTANCE
    if _LLM_INSTANCE is not None:
        return _LLM_INSTANCE

    model_key_or_path = DEFAULT_LLM or os.getenv("LOCAL_LLM_MODEL", "")
    base_url = LLM_BASE_URL

    if base_url:
        model = model_key_or_path or "local-model"
        _LLM_INSTANCE = _OpenAIGenerator(base_url, model, LLM_API_KEY)
    elif model_key_or_path:
        model_id = LLM_MODELS.get(model_key_or_path, model_key_or_path)
        _LLM_INSTANCE = _LocalLLM(model_id)
    elif os.getenv("LLM_USE_OLLAMA") == "1":
        _LLM_INSTANCE = _OllamaGenerator()
    else:
        _LLM_INSTANCE = _TemplateGenerator()
    return _LLM_INSTANCE


class Generator:
    def generate(self, query: str, context: list[str]) -> str:
        raise NotImplementedError


class _TemplateGenerator(Generator):
    def generate(self, query: str, context: list[str]) -> str:
        if not context:
            return "No relevant information found."
        return context[0]


class _OpenAIGenerator(Generator):
    """Connect to any OpenAI-compatible API (OpenAI, LM Studio, Ollama, etc.)."""

    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def generate(self, query: str, context: list[str]) -> str:
        context_str = ", ".join(context)
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": f"Answer based only on this context: {context_str}"},
                {"role": "user", "content": query},
            ],
            "temperature": GENERATOR["temperature"],
            "max_tokens": GENERATOR["max_tokens"],
            "stream": False,
        }).encode()

        base = self.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        url = f"{base}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            url,
            data=payload,
            headers=headers,
        )
        try:
            resp = urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            raise RuntimeError(f"API error {e.code}: {body}")
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()


class _LocalLLM(Generator):
    """Load any HuggingFace causal LM via transformers."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        from transformers import pipeline as hf_pipeline
        try:
            self._pipe = hf_pipeline(
                "text-generation",
                model=self.model_id,
                device_map="auto",
                dtype="auto",
                model_kwargs={"low_cpu_mem_usage": True},
            )
        except Exception:
            self._pipe = hf_pipeline(
                "text-generation",
                model="HuggingFaceTB/SmolLM2-360M-Instruct",
                device_map="auto",
                dtype="auto",
                model_kwargs={"low_cpu_mem_usage": True},
            )

    def generate(self, query: str, context: list[str]) -> str:
        self._load()
        context_str = ", ".join(context)

        if self._pipe.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": f"You have access to this context: {context_str}. Answer concisely."},
                {"role": "user", "content": query},
            ]
            prompt = self._pipe.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = (
                f"Context: {context_str}\n"
                f"Question: {query}\n"
                f"Answer:"
            )

        result = self._pipe(
            prompt,
            max_new_tokens=64,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.2,
            eos_token_id=self._pipe.tokenizer.eos_token_id,
            pad_token_id=self._pipe.tokenizer.pad_token_id or self._pipe.tokenizer.eos_token_id,
        )
        text = result[0]["generated_text"][len(prompt):].strip()
        return text.split("\n")[0][:200]


class _OllamaGenerator(Generator):
    def __init__(self):
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            resp = urllib.request.urlopen(req)
            models = json.loads(resp.read()).get("models", [])
            if not models:
                raise RuntimeError("Ollama is running but no models installed.")
            self.model = models[0]["name"]
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Ollama: {e}")

    def generate(self, query: str, context: list[str]) -> str:
        prompt = (
            f"Context: {', '.join(context)}\n\n"
            f"Question: {query}\n\n"
            f"Answer concisely based only on the context above:"
        )
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": GENERATOR["temperature"],
                "num_predict": GENERATOR["max_tokens"],
            },
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read())
        return data.get("response", "").strip()
