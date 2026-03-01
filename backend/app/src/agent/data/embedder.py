"""
TCM Text Embedder
中医文本向量嵌入器

支持多种嵌入服务
"""

import os
from typing import Optional
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """嵌入器基类"""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """生成单个文本的向量嵌入"""
        pass

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量生成文本的向量嵌入"""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度"""
        pass


class DashScopeEmbedder(BaseEmbedder):
    """通义千问嵌入器"""

    def __init__(self, model: str = "text-embedding-v3"):
        self.model = model
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self._dimension = 1024  # text-embedding-v3 默认维度

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_text(self, text: str) -> list[float]:
        """生成单个文本的向量嵌入"""
        import httpx

        url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "input": {"texts": [text]},
            "parameters": {"text_type": "document"}
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

        embeddings = data.get("output", {}).get("embeddings", [])
        if embeddings:
            return embeddings[0].get("embedding", [])
        return []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量生成文本的向量嵌入"""
        import httpx

        url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 分批处理，每批最多25条
        batch_size = 25
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            payload = {
                "model": self.model,
                "input": {"texts": batch},
                "parameters": {"text_type": "document"}
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=headers, timeout=60)
                response.raise_for_status()
                data = response.json()

            embeddings = data.get("output", {}).get("embeddings", [])
            for emb in embeddings:
                all_embeddings.append(emb.get("embedding", []))

        return all_embeddings


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI嵌入器"""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self._dimension = 1536 if "3-small" in model else 3072

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_text(self, text: str) -> list[float]:
        """生成单个文本的向量嵌入"""
        import httpx

        url = f"{self.base_url}/embeddings"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "input": text
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

        embeddings = data.get("data", [])
        if embeddings:
            return embeddings[0].get("embedding", [])
        return []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量生成文本的向量嵌入"""
        import httpx

        url = f"{self.base_url}/embeddings"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # OpenAI支持批量输入
        payload = {
            "model": self.model,
            "input": texts
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            data = response.json()

        embeddings = data.get("data", [])
        # 按index排序
        sorted_embeddings = sorted(embeddings, key=lambda x: x.get("index", 0))
        return [emb.get("embedding", []) for emb in sorted_embeddings]


class OllamaEmbedder(BaseEmbedder):
    """Ollama本地嵌入器"""

    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._dimension = 768  # nomic-embed-text 默认维度

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_text(self, text: str) -> list[float]:
        """生成单个文本的向量嵌入"""
        import httpx

        url = f"{self.base_url}/api/embeddings"

        payload = {
            "model": self.model,
            "prompt": text
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

        return data.get("embedding", [])

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量生成文本的向量嵌入"""
        # Ollama不支持批量，逐个处理
        embeddings = []
        for text in texts:
            emb = await self.embed_text(text)
            embeddings.append(emb)
        return embeddings


def get_embedder(service: str = None) -> BaseEmbedder:
    """
    获取嵌入器实例

    Args:
        service: 服务类型 (DASHSCOPE, OPENAI, OLLAMA)

    Returns:
        BaseEmbedder: 嵌入器实例
    """
    service = service or os.getenv("EMBEDDING_SERVICE", "DASHSCOPE")

    if service == "DASHSCOPE":
        return DashScopeEmbedder()
    elif service == "OPENAI":
        return OpenAIEmbedder()
    elif service == "OLLAMA":
        return OllamaEmbedder()
    else:
        # 默认使用DashScope
        return DashScopeEmbedder()
