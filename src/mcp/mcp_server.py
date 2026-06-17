"""MCP Server for RAG Job Management"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import CallToolResult, TextContent, Tool
from starlette.responses import Response

logger = logging.getLogger(__name__)
server = Server("rag-mcp")
RESULTS_FILE = Path(os.getenv("RESULTS_DIR", "/data")) / "results.jsonl"
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)


class JobManager:
    @staticmethod
    def submit_job(query: str, model: str = "all", dataset: str = "tech") -> dict:
        from src.mcp.tasks import run_rag_task
        
        job_id = str(uuid4())
        task = run_rag_task.delay(job_id, query, model, dataset)
        
        return {
            "job_id": job_id,
            "celery_task_id": task.id,
            "status": "queued",
            "query": query,
            "model": model,
            "dataset": dataset,
            "submitted_at": datetime.utcnow().isoformat(),
        }
    
    @staticmethod
    def check_status(job_id: str) -> dict:
        from src.mcp.tasks import celery_app
        
        result = celery_app.backend.get(f"rag:result:{job_id}")
        if result:
            return json.loads(result)
        
        return {"job_id": job_id, "status": "not_found"}
    
    @staticmethod
    def list_results(limit: int = 50, filter_model: str = None) -> list:
        if not RESULTS_FILE.exists():
            return []
        
        results = []
        with open(RESULTS_FILE, "r") as f:
            for line in f:
                try:
                    result = json.loads(line)
                    if filter_model and result.get("model_name") != filter_model:
                        continue
                    results.append(result)
                except json.JSONDecodeError:
                    pass
        
        return results[:limit]
    
    @staticmethod
    def get_result(job_id: str) -> dict:
        if not RESULTS_FILE.exists():
            return None
        
        with open(RESULTS_FILE, "r") as f:
            for line in f:
                try:
                    result = json.loads(line)
                    if result.get("job_id") == job_id:
                        return result
                except json.JSONDecodeError:
                    pass
        return None


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="submit_rag_job",
            description="Submit async RAG query (non-blocking, returns job_id)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "User query"},
                    "model": {"type": "string", "description": "Embedding model", "default": "all"},
                    "dataset": {"type": "string", "description": "Dataset", "default": "tech"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="check_job_status",
            description="Check job progress and result",
            inputSchema={
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        ),
        Tool(
            name="list_cached_results",
            description="Browse cached results (instant)",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 50},
                    "model": {"type": "string"},
                },
            },
        ),
        Tool(
            name="get_cached_result",
            description="Get result by job_id",
            inputSchema={
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    try:
        if name == "submit_rag_job":
            result = JobManager.submit_job(
                arguments["query"],
                arguments.get("model", "all"),
                arguments.get("dataset", "tech"),
            )
        elif name == "check_job_status":
            result = JobManager.check_status(arguments["job_id"])
        elif name == "list_cached_results":
            results = JobManager.list_results(
                arguments.get("limit", 50),
                arguments.get("model"),
            )
            result = {"count": len(results), "results": results}
        elif name == "get_cached_result":
            result = JobManager.get_result(arguments["job_id"]) or {"error": "Not found"}
        else:
            result = {"error": f"Unknown: {name}"}
    except Exception as e:
        logger.exception("Tool call failed")
        result = {"error": str(e)}
    
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(result))])


sse = SseServerTransport("/mcp/messages/")


async def app(scope, receive, send):
    if scope["type"] != "http":
        response = Response(status_code=405)
        await response(scope, receive, send)
        return

    path = scope["path"]
    method = scope["method"]

    if path == "/mcp/" and method == "GET":
        async with sse.connect_sse(scope, receive, send) as streams:
            init_opts = server.create_initialization_options()
            await server.run(streams[0], streams[1], init_opts)
    elif path.startswith("/mcp/messages/") and method == "POST":
        await sse.handle_post_message(scope, receive, send)
    else:
        response = Response(status_code=404)
        await response(scope, receive, send)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5100)
