"""
复杂诊断模块

基于 LangChain DeepAgents 框架的深度辨证分析系统

架构特点：
1. 使用 SubAgentMiddleware 实现 5 个专家子 Agent 的并行调度
2. 使用 TodoListMiddleware 进行任务规划与跟踪
3. 使用 FilesystemMiddleware 管理大结果的自动驱逐
4. 使用 SummarizationMiddleware 压缩对话历史

专家子 Agent：
- 鉴别诊断专家 (differential_diagnosis_expert)
- 治则治法专家 (treatment_principle_expert)
- 方药推荐专家 (prescription_recommendation_expert)
- 预后评估专家 (prognosis_evaluation_expert)
- 质疑验证专家 (verification_expert)

数据查询工具：
- kg_syndrome_search: 知识图谱证型查询
- case_vector_search: 医案向量检索
- classics_search: 古籍检索
- web_search: 网络搜索
"""

from .complex_diagnosis import complex_diagnosis, run_complex_diagnosis
from .deep_search_agent import create_deep_search_agent, run_deep_search_diagnosis

__all__ = [
    "complex_diagnosis",
    "run_complex_diagnosis",
    "create_deep_search_agent",
    "run_deep_search_diagnosis",
]
