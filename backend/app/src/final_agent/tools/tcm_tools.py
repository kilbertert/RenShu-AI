"""
合并后的 TCM 工具函数

从以下 4 个文件合并所有 @tool 函数：
- components/diagnose/nodes/complex/tools/kg_tools.py → kg_syndrome_search, kg_organ_query
- components/diagnose/nodes/complex/tools/vector_tools.py → case_vector_search
- components/diagnose/nodes/complex/tools/classics_tools.py → classics_search
- components/diagnose/nodes/complex/tools/web_tools.py → web_search, medical_research_search

原始文件通过 re-export 保持向后兼容。
"""

# 直接从原位置导入，避免代码复制
# 这样修改只需改一处，所有引用自动更新
from app.src.agent.components.diagnose.nodes.complex.tools.kg_tools import (
    kg_syndrome_search,
    kg_organ_query,
)
from app.src.agent.components.diagnose.nodes.complex.tools.vector_tools import (
    case_vector_search,
)
from app.src.agent.components.diagnose.nodes.complex.tools.classics_tools import (
    classics_search,
)
from app.src.agent.components.diagnose.nodes.complex.tools.web_tools import (
    web_search,
    medical_research_search,
)

__all__ = [
    "kg_syndrome_search",
    "kg_organ_query",
    "case_vector_search",
    "classics_search",
    "web_search",
    "medical_research_search",
]
