"""
DeepSearch Agent 工具模块

提供数据查询工具（无模型调用）：
- kg_syndrome_search: 知识图谱证型查询
- case_vector_search: 医案向量检索
- classics_search: 古籍检索
- web_search: 网络搜索
"""

from .kg_tools import kg_syndrome_search, kg_organ_query
from .vector_tools import case_vector_search
from .classics_tools import classics_search
from .web_tools import web_search, medical_research_search

__all__ = [
    "kg_syndrome_search",
    "kg_organ_query",
    "case_vector_search",
    "classics_search",
    "web_search",
    "medical_research_search",
]
