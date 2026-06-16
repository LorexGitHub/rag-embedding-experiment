"""HTTP wrapper for MCP server"""
import os
import logging
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from .mcp_server import JobManager

logger = logging.getLogger(__name__)
API_KEY = os.getenv("MCP_API_KEY", "changeme")
app = FastAPI(title="RAG MCP Server")

def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key is None:
        return None
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/tools/submit_rag_job")
async def submit_rag_job(request: dict, _: str = Depends(verify_api_key)):
    try:
        query = request.get("query")
        model = request.get("model", "all")
        dataset = request.get("dataset", "tech")
        if not query:
            raise HTTPException(status_code=400, detail="query required")
        result = JobManager.submit_job(query, model, dataset)
        return JSONResponse(result)
    except Exception as e:
        logger.exception("Error")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/tools/check_job_status/{job_id}")
async def check_job_status(job_id: str, _: str = Depends(verify_api_key)):
    try:
        result = JobManager.check_status(job_id)
        return JSONResponse(result)
    except Exception as e:
        logger.exception("Error")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/tools/list_cached_results")
async def list_cached_results(limit: int = 50, model: str = None, _: str = Depends(verify_api_key)):
    try:
        results = JobManager.list_results(limit, model)
        return JSONResponse({"count": len(results), "results": results[-10:] if results else []})
    except Exception as e:
        logger.exception("Error")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/tools/get_cached_result/{job_id}")
async def get_cached_result(job_id: str, _: str = Depends(verify_api_key)):
    try:
        result = JobManager.get_result(job_id)
        if result:
            return JSONResponse(result)
        return JSONResponse({"error": "Not found"}, status_code=404)
    except Exception as e:
        logger.exception("Error")
        return JSONResponse({"error": str(e)}, status_code=500)
