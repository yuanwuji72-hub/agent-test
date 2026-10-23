import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict,Any
from dynamic_memory import DynamicMemory

os.environ["MEM0_TELEMETRY"] = "False"
memory = None

OLLAMA_URL = "http://127.0.0.1:11434/v1"
EMBEDDINGS_URL = "http://127.0.0.1:11434/v1"
EMBED_MODEL = "bge-zh1.5:latest"
LLM_MODEL = "qwen2.5:7b"


def get_memory():
    global memory
    if memory is None:
        memory = DynamicMemory(
            llm_model=LLM_MODEL,
            llm_apikey="ollama",
            llm_url=OLLAMA_URL,
            embed_model=EMBED_MODEL,
            embed_apikey="ollama",
            embed_url=EMBEDDINGS_URL,
            max_tokens=2048,
            qdrant_url="http://qdrant:6333"

        )
    return memory

# ==================== 请求/响应模型 ====================

class AddRequest(BaseModel):
    messages: str | List[Dict[str, Any]]
    user_id: str = "default_user"

class SearchRequest(BaseModel):
    query: str
    user_id: str = "default_user"
    top_k: int = 3

class GetAllRequest(BaseModel):
    user_id: str = "default_user"
    top_k: int = 100

class DeleteRequest(BaseModel):
    memory_id: str

class DeleteAllRequest(BaseModel):
    user_id: str = "default_user"

# ==================== FastAPI 应用 ====================

app = FastAPI(title="DynamicMemory API", version="1.0.0")

@app.post("/memory/add")
def add_memory(req: AddRequest):
    try:
        result = get_memory().add(req.messages, user_id=req.user_id)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/search")
def search_memory(req: SearchRequest):
    try:
        result = get_memory().search(req.query, user_id=req.user_id, top_k=req.top_k)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/get_all")
def get_all_memory(req: GetAllRequest):
    try:
        result = get_memory().get_all(user_id=req.user_id, top_k=req.top_k)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/delete")
def delete_memory(req: DeleteRequest):
    try:
        get_memory().delete(req.memory_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/delete_all")
def delete_all_memory(req: DeleteAllRequest):
    try:
        get_memory().delete_all(user_id=req.user_id, confirm=True)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)