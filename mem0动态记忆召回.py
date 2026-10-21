import os
import atexit
import weakref
from mem0 import Memory
from qdrant_client import QdrantClient


_original_qdrant_del = QdrantClient.__del__

def _safe_qdrant_del(self):
    try:
        _original_qdrant_del(self)
    except (ImportError, AttributeError):
        # Python 解释器退出时 sys.meta_path 已被销毁，静默忽略
        pass

QdrantClient.__del__ = _safe_qdrant_del


os.environ["MEM0_TELEMETRY"] = "false"

class DynamicMemory:
    """
        llm系列为提取关键信息模型的参数，请传入相关model，apikey，url。
        embed系列为向量语义模型的参数，请传入相关model，apikey，url。
        provider为供应商，例如openai或者ollama，若为ollama则不需要apikey。
        max_tokens为llm模型单次最大输出长度的限制。
        embed_dims为embedding模型的维度信息，请验证后填入
        vector_store_provider为数据库的类型
        on_disk为是否存储在磁盘，False为内存存储
    """
    def __init__(self,
             llm_model: str,
             max_tokens: int,
             embed_dims: int,*,
             llm_apikey: str | None = None,
             llm_url: str = "",
             llm_provider: str = "ollama",
             embed_apikey: str | None = None,
             embed_model: str = "",
             embed_url: str = "",
             emb_provider: str = "ollama",
             vector_store_provider: str = "qdrant",
             on_disk: bool = True):
        self.llm_model = llm_model
        self.llm_apikey = llm_apikey
        self.llm_url = llm_url
        self.llm_provider = llm_provider
        self.max_tokens = max_tokens
        self.embed_model = embed_model
        self.embed_apikey = embed_apikey
        self.embed_url = embed_url
        self.emb_provider = emb_provider
        self.embed_dims = embed_dims
        self.vector_store_provider = vector_store_provider
        self.on_disk = on_disk

        llm_config = {
            "model": self.llm_model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }

        if self.llm_provider == "ollama":
            llm_config["ollama_base_url"] = self.llm_url
        else:
            llm_config["api_key"] = self.llm_apikey
            if self.llm_url:
                llm_config["base_url"] = self.llm_url
        embed_config = {
            "model": self.embed_model,
            "embedding_dims": self.embed_dims,
        }
        if self.emb_provider == "ollama":
            embed_config["ollama_base_url"] = self.embed_url
        else:
            embed_config["api_key"] = self.embed_apikey
            if self.embed_url:
                embed_config["base_url"] = self.embed_url

        config = {
            "version": "v1.1",
            "custom_instructions": (
                "你是一名个人信息整理专家。请仅从用户消息中提取事实、偏好、计划和重要细节，"
                "并输出为中文短句。每条记忆以'用户'或'助手'开头。"
                "不要添加解释，不要翻译原文，不要输出英文。"
                "如果无可提取内容，返回空列表。"
            ),
            "llm": {
                "provider": self.llm_provider,
                "config": llm_config,
            },
            "embedder": {
                "provider": self.emb_provider,
                "config": embed_config,
            },
            "vector_store": {
                "provider": self.vector_store_provider,
                "config": {
                    "collection_name": "mem0_local",
                    "path": f"./{self.vector_store_provider}_local_storage",
                    "embedding_model_dims": self.embed_dims,
                    "on_disk": self.on_disk,
                },
            },
        }
        self._validate_config()
        self.memory = Memory.from_config(config)
        self._closed = False
        self_ref = weakref.ref(self)

        def _safe_cleanup():
            obj = self_ref()
            if obj and not obj._closed:
                try:
                    obj.close()
                except Exception:
                    pass

        atexit.register(_safe_cleanup)

    # ==================== CRUD 方法 ====================

    def add(self, messages: str | list[dict], user_id: str = "default_user", **kwargs) -> dict:
        """
        添加新记忆。
        :param messages: 字符串或 OpenAI 格式的对话列表 [{"role": "user", "content": "..."}]
        :param user_id: 用户唯一标识
        :return: mem0 添加结果 (包含新增/更新/删除的记忆条目)
        """
        if not messages:
            return {"message": "No messages provided", "results": []}
        return self.memory.add(messages, user_id=user_id, **kwargs)

    def search(
            self,
            query: str,
            user_id: str = "default_user",
            limit: int = 5,
            filters: dict | None = None,
            **kwargs,
    )-> dict:
        """语义检索记忆。"""
        if filters is None:
            filters = {"user_id": user_id}
        return self.memory.search(query=query, filters=filters, limit=limit, **kwargs)

    def get_all(
            self,
            user_id: str = "default_user",
            limit: int = 100,
            filters: dict | None = None,
    ) -> dict:
        """获取指定用户的记忆。"""
        if filters is None:
            filters = {"user_id": user_id}
        return self.memory.get_all(filters=filters, limit=limit)

    def delete(self, memory_id: str) -> None:
        """根据记忆 ID 删除单条记忆。"""
        self.memory.delete(memory_id)

    def delete_all(self, user_id: str = "default_user", confirm: bool = False) -> None:
        """清空指定用户的所有记忆。需要显式确认。"""
        if not confirm:
            raise ValueError("删除全部记忆需要传入 confirm=True 进行二次确认")
        self.memory.delete_all(filters={"user_id": user_id})

    # ==================== 安全与运维方法 ====================

    def _validate_config(self):
        """
        初始化后的配置自检。
        防止因拼写错误或类型错误导致运行时才暴露问题。
        """
        assert isinstance(self.llm_model, str) and self.llm_model, "llm_model 不能为空"
        assert isinstance(self.embed_dims, int) and self.embed_dims > 0, "embed_dims 必须为正整数"
        assert self.llm_provider in ("ollama", "openai", "qwen", "anthropic"), \
            f"不支持的 llm_provider: {self.llm_provider}"

        # 非 Ollama 模式下强制校验 API Key
        if self.llm_provider != "ollama" and not self.llm_apikey:
            raise ValueError(f"使用 {self.llm_provider} 时，llm_apikey 不能为 None 或空字符串")

    def health_check(self) -> bool:
        """
        轻量级健康检查：验证 LLM 和 Embedder 是否可达。
        建议在服务启动时调用一次。
        """
        try:
            self.search("health check ping", limit=1)
            return True
        except Exception as e:
            print(f"[DynamicMemory] Health check failed: {e}")
            return False

    def __repr__(self) -> str:
        return (
            f"DynamicMemory(llm={self.llm_provider}/{self.llm_model}, "
            f"embed={self.emb_provider}/{self.embed_model}, "
            f"store={self.vector_store_provider}, dims={self.embed_dims})"
        )

    def close(self):
        """显式释放所有底层资源。"""
        if not self._closed:
            try:
                if hasattr(self.memory, 'vector_store'):
                    store = self.memory.vector_store
                    if hasattr(store, 'client'):
                        client = store.client
                        if client is not None and hasattr(client, 'close'):
                            client.close()
                        # 切断引用，防止 __del__ 再次被触发
                        store.client = None
            except Exception:
                pass

            # 调用 mem0 层的 close（兜底）
            if hasattr(self.memory, 'close'):
                try:
                    self.memory.close()
                except Exception:
                    pass

            # 同时关闭 llm 和 embedder 的客户端（如果有）
            for attr in ('llm', 'embedder'):
                try:
                    obj = getattr(self.memory, attr, None)
                    if obj is not None and hasattr(obj, 'client') and obj.client is not None:
                        obj.client.close()
                        obj.client = None
                except Exception:
                    pass

            self._closed = True

    def __del__(self):
        pass



if __name__ == "__main__":
    OLLAMA_URL = "http://127.0.0.1:11434"
    EMBED_MODEL = "bge-zh1.5:latest"
    EMBED_DIMS = 768
    QWEN_MODEL = "qwen3:8b"

    memory = DynamicMemory(
        llm_provider="ollama",
        llm_model=QWEN_MODEL,
        llm_url=OLLAMA_URL,
        emb_provider="ollama",
        embed_model=EMBED_MODEL,
        embed_url=OLLAMA_URL,
        embed_dims=EMBED_DIMS,
        max_tokens=512
    )

    try:
        # 写入测试
        result = memory.add([{"role": "user", "content": "我喜欢养狗，家里有一只柯基"}])
        print("add 结果:", result)

        # 检索测试
        result = memory.search("我喜欢什么")
        print("search 结果:", result)

        # 全量查看
        result = memory.get_all()
        print("get_all 结果:", result)
    finally:
        memory.close()