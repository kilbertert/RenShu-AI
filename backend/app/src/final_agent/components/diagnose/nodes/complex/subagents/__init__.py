"""
复杂诊断子 Agent 模块

包含多个专家 SubAgent（基于 DeepAgents 框架）：
- differential_diagnosis_expert: 鉴别诊断专家
- treatment_principle_expert: 治则治法专家
- prescription_expert: 方药推荐专家
- prognosis_expert: 预后评估专家
- verification_expert: 质疑验证专家

使用 deepsearch_prompts.py 中的专业提示词
"""

from .differential_diagnosis_subagent import create_differential_expert
from .treatment_principle_subagent import create_treatment_expert
from .prescription_subagent import create_prescription_expert
from .prognosis_subagent import create_prognosis_expert
from .verification_subagent import create_verification_expert

__all__ = [
    "create_differential_expert",
    "create_treatment_expert",
    "create_prescription_expert",
    "create_prognosis_expert",
    "create_verification_expert",
]
