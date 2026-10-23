import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Mini RAG API")


class SearchReq(BaseModel):
    query: str
    top_k: int = 3


class IngestReq(BaseModel):
    texts: list[str]


@app.post("/ingest")
def ingest(req: IngestReq):
    """
    接收文本列表，后续在这里接入 embedding + 向量入库。
    当前示例只打印接收到的文本数量。
    """
    print(f"[INGEST] 收到 {len(req.texts)} 条文本")
    for i, text in enumerate(req.texts):
        print(f"  [{i}] {text}")
    return {"ok": True, "count": len(req.texts)}


@app.post("/search")
def search(req: SearchReq):
    """
    接收查询请求，后续在这里接入 embedding + 向量检索 + rerank。
    当前示例返回模拟结果。
    """
    print(f"[SEARCH] query={req.query}, top_k={req.top_k}")
    # TODO: 接入 Qdrant 检索 + BGE rerank
    mock_results = [
        {"text": f"模拟结果 {i}: {req.query}", "score": round(0.9 - i * 0.1, 2)}
        for i in range(req.top_k)
    ]
    return {"results": mock_results}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8000, reload=False)