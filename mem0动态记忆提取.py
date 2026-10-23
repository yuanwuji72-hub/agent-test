import os,requests
os.environ["MEM0_TELEMETRY"] = "False"
from mem0 import Memory
import atexit


try:
    from qdrant_client import QdrantClient
    QdrantClient.__del__ = lambda self: None
except Exception:
    pass


class DynamicMemory:
    """
        llm系列为提取关键信息模型的参数，请传入相关model，apikey，url。
        embed系列为向量语义模型的参数，请传入相关model，apikey，url。
        max_tokens为llm模型单次最大输出长度的限制。
        vector_store_provider为数据库的名称。
        数据库默认为连接本机6333端口，如需连接外部数据库填入qdrant_url和qdrant_api_key。
        collection_name为表名
    """
    def __init__(self,
             llm_model: str,
             max_tokens: int,
             llm_apikey: str,
             llm_url: str ,
             embed_apikey: str ,
             embed_model: str ,
             embed_url: str ,
             vector_store_provider: str = "qdrant",
             qdrant_url: str = None,
             qdrant_host: str = "127.0.0.1",
             qdrant_port: int = 6333,
             qdrant_api_key: str = None,
             collection_name: str = "mem0_local",
    ):
        self.llm_model = llm_model
        self.llm_apikey = llm_apikey
        self.llm_url = llm_url
        self.max_tokens = max_tokens
        self.embed_model = embed_model
        self.embed_apikey = embed_apikey
        self.embed_url = embed_url
        self.vector_store_provider = vector_store_provider

        emb_payload = {
            "model": embed_model,
            "input": "测试文本"
        }
        base = self.embed_url.rstrip("/")
        resp = requests.post(base.removesuffix("/v1") + "/api/embed", json=emb_payload)
        result = resp.json()
        vec = result["embeddings"][0]
        self.dims = len(vec)

        vs_config = {
            "collection_name": collection_name,
            "embedding_model_dims": self.dims,
        }
        if qdrant_url:
            vs_config["url"] = qdrant_url
            if qdrant_api_key:
                vs_config["api_key"] = qdrant_api_key
        else:
            vs_config["host"] = qdrant_host
            vs_config["port"] = qdrant_port
        config = {
            "version": "v1.1",
            "custom_instructions": (
                "从用户消息中提取事实、偏好和重要细节。"
                "每条事实用中文简短表述。"
            ),
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self.llm_model,
                    "temperature": 0,
                    "max_tokens": self.max_tokens,
                    "api_key": self.llm_apikey,
                    "openai_base_url": self.llm_url,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": self.embed_model,
                    "embedding_dims": self.dims,
                    "api_key": self.embed_apikey,
                    "openai_base_url": self.embed_url,
                },
            },
             "vector_store": {
                "provider": self.vector_store_provider,
                "config": vs_config,
            },
        }

        self.memory = Memory.from_config(config)
        self._closed = False
        atexit.register(self.close)

    def add(self, messages: str | list[dict], user_id: str = "default_user", **kwargs) -> dict:
        """添加新记忆。"""
        if not messages:
            return {"message": "No messages provided", "results": []}
        return self.memory.add(messages, user_id=user_id, **kwargs)

    def search(self,query: str,user_id: str = "default_user",top_k: int = 2,filters: dict | None = None,**kwargs) -> dict:
        """语义检索记忆。"""
        merged_filters = {"user_id": user_id}
        if filters:
            merged_filters.update(filters)
        return self.memory.search(
            query=query,
            filters=merged_filters,
            top_k=top_k,
            **kwargs,
        )

    def get_all(self,user_id: str = "default_user",top_k: int = 100,filters: dict | None = None,
    ) -> dict:
        """获取指定用户的记忆。"""
        merged_filters = {"user_id": user_id}
        if filters:
            merged_filters.update(filters)
        return self.memory.get_all(
            filters=merged_filters,
            top_k=top_k,
        )

    def delete(self, memory_id: str) -> None:
        """根据记忆 ID 删除单条记忆。"""
        if not memory_id:
            raise ValueError("memory_id 不能为空")
        self.memory.delete(memory_id)

    def delete_all(self, user_id: str = "default_user", confirm: bool = False) -> None:
        """清空指定用户的所有记忆。需要显式确认。"""
        if not confirm:
            raise ValueError("删除全部记忆需要传入 confirm=True 进行二次确认")
        self.memory.delete_all(user_id=user_id)

    def close(self):
        if self._closed:
            return

        self._closed = True

        try:
            if hasattr(self.memory, "vector_store"):
                store = self.memory.vector_store
                client = getattr(store, "client", None)
                if client is not None and hasattr(client, "close"):
                    client.close()
                if client is not None:
                    store.client = None  # 防止 __del__ 再次触发
        except Exception:
            pass

        try:
            if hasattr(self.memory, "close"):
                self.memory.close()
        except Exception:
            pass





if __name__ == "__main__":
    OLLAMA_URL = "http://127.0.0.1:11434/v1"
    EMBEDDINGS_URL = "http://127.0.0.1:11434/v1"
    EMBED_MODEL = "bge-zh1.5:latest"
    LLM_MODEL = "qwen2.5:7b"

    memory = DynamicMemory(
        llm_model=LLM_MODEL,
        llm_apikey="ollama",
        llm_url=OLLAMA_URL,
        embed_model=EMBED_MODEL,
        embed_apikey="ollama",
        embed_url=EMBEDDINGS_URL,
        max_tokens=2048
    )



    result = memory.add([{"role": "user", "content": "我非常讨厌吃甜的，吃多了牙疼"},],user_id="test")
    print("add 结果:", result)


    result = memory.search("讨厌什么",user_id="test")
    print("search 结果:", result)


    result = memory.get_all(user_id="test")
    print("get_all 结果:", result)

    memory.close()
