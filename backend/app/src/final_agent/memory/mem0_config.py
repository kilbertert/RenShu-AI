"""
Mem0 配置模块

根据部署环境选择不同的存储后端
"""

import os
from typing import Optional
from pydantic import BaseModel, Field


class Mem0Config(BaseModel):
    """Mem0 配置类"""

    # LLM 配置（用于记忆提取）
    llm_provider: str = Field(default="deepseek", description="LLM提供商")
    llm_model: str = Field(default="deepseek-chat", description="LLM模型")
    llm_api_key: Optional[str] = Field(default=None, description="LLM API密钥")

    # Embedding 配置
    embedder_provider: str = Field(default="openai", description="Embedding提供商")
    embedder_model: str = Field(default="text-embedding-3-small", description="Embedding模型")
    embedding_dims: int = Field(default=1536, description="Embedding维度")

    # 向量存储配置
    vector_store_provider: str = Field(default="milvus", description="向量存储提供商")
    vector_store_config: dict = Field(default_factory=dict, description="向量存储配置")

    # 图数据库配置
    graph_store_provider: str = Field(default="neo4j", description="图存储提供商")
    graph_store_config: dict = Field(default_factory=dict, description="图存储配置")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @classmethod
    def from_env(cls) -> "Mem0Config":
        """从环境变量加载配置"""
        config = cls()

        # LLM配置
        config.llm_provider = os.getenv("LLM_PROVIDER", "deepseek")
        config.llm_model = os.getenv("LLM_MODEL", "deepseek-chat")
        config.llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")

        # Embedding配置
        config.embedder_provider = os.getenv("EMBEDDER_PROVIDER", "openai")
        config.embedder_model = os.getenv("EMBEDDER_MODEL", "text-embedding-3-small")
        config.embedding_dims = int(os.getenv("EMBEDDING_DIMS", "1536"))

        # 向量存储配置
        config.vector_store_provider = os.getenv("VECTOR_STORE", "milvus").lower()
        if config.vector_store_provider == "milvus":
            config.vector_store_config = {
                "url": os.getenv("MILVUS_URL", os.getenv("MILVUS_URI", "http://localhost:19530")),
                "token": os.getenv("MILVUS_TOKEN") or None,
                "collection_name": os.getenv("MILVUS_COLLECTION", "tcm_memories"),
                "embedding_model_dims": config.embedding_dims,
                "metric_type": os.getenv("MILVUS_METRIC_TYPE", "L2"),
                "db_name": os.getenv("MILVUS_DB_NAME", ""),
            }
        elif config.vector_store_provider == "qdrant":
            config.vector_store_config = {
                "host": os.getenv("QDRANT_HOST", "localhost"),
                "port": int(os.getenv("QDRANT_PORT", "6333")),
                "collection_name": os.getenv("QDRANT_COLLECTION", "tcm_memories"),
                "embedding_model_dims": config.embedding_dims,
                "on_disk": True  # 持久化到磁盘
            }

        # 图数据库配置
        config.graph_store_provider = os.getenv("GRAPH_STORE", "neo4j")
        if config.graph_store_provider == "neo4j":
            config.graph_store_config = {
                "url": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                "username": os.getenv("NEO4J_USER", "neo4j"),
                "password": os.getenv("NEO4J_PASSWORD", "tcm_graph_2026")
            }

        return config

    def to_mem0_config(self) -> dict:
        """
        转换为 Mem0 MemoryConfig 格式

        Returns:
            Mem0 配置字典
        """
        config_dict = {}

        # LLM 配置
        if self.llm_provider == "openai":
            config_dict["llm"] = {
                "provider": "openai",
                "config": {
                    "model": self.llm_model,
                    "api_key": self.llm_api_key
                }
            }
        elif self.llm_provider == "deepseek":
            config_dict["llm"] = {
                "provider": "openai_compatible",
                "config": {
                    "model": self.llm_model,
                    "base_url": "https://api.deepseek.com",
                    "api_key": self.llm_api_key
                }
            }

        # Embedding 配置
        if self.embedder_provider == "openai":
            config_dict["embedder"] = {
                "provider": "openai",
                "config": {
                    "model": self.embedder_model,
                    "api_key": self.llm_api_key
                }
            }

        # 向量存储配置
        if self.vector_store_provider and self.vector_store_config:
            vector_store_config = dict(self.vector_store_config)
            if self.vector_store_provider == "milvus" and not vector_store_config.get("url"):
                vector_store_config["url"] = "http://localhost:19530"
            if "embedding_model_dims" not in vector_store_config:
                vector_store_config["embedding_model_dims"] = self.embedding_dims

            config_dict["vector_store"] = {
                "provider": self.vector_store_provider,
                "config": vector_store_config,
            }

        # 图存储配置
        if self.graph_store_provider and self.graph_store_config:
            config_dict["graph_store"] = {
                "provider": self.graph_store_provider,
                "config": dict(self.graph_store_config),
            }

        return config_dict
