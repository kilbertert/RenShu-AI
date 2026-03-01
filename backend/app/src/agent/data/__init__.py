"""
TCM Data Module
中医数据模块

包含Neo4j初始化脚本、数据导入器和示例数据
"""

from .models import (
    ClassicRecord,
    CaseRecord,
    IngestResult,
    SyndromeNode,
    PrescriptionNode,
    HerbNode,
)
from .embedder import (
    BaseEmbedder,
    DashScopeEmbedder,
    OpenAIEmbedder,
    OllamaEmbedder,
    get_embedder,
)
from .schema_initializer import (
    init_schema,
    create_constraints,
    create_fulltext_indexes,
    create_vector_indexes,
    get_schema_info,
)
from .classic_ingestor import ClassicIngestor
from .case_ingestor import CaseIngestor

__all__ = [
    # 数据模型
    "ClassicRecord",
    "CaseRecord",
    "IngestResult",
    "SyndromeNode",
    "PrescriptionNode",
    "HerbNode",
    # 嵌入器
    "BaseEmbedder",
    "DashScopeEmbedder",
    "OpenAIEmbedder",
    "OllamaEmbedder",
    "get_embedder",
    # Schema初始化
    "init_schema",
    "create_constraints",
    "create_fulltext_indexes",
    "create_vector_indexes",
    "get_schema_info",
    # 数据导入器
    "ClassicIngestor",
    "CaseIngestor",
]
