"""
诊断子图提示词模块

包含：
- diagnosis_prompts: 主要诊断流程提示词
- deepsearch_prompts: DeepSearch 多专家提示词
- multimodal_prompts: 多模态分析提示词
"""

# ============================================================
# 从 diagnosis_prompts 导入
# ============================================================
from .diagnosis_prompts import (
    # 信息收集阶段
    SYMPTOM_EXTRACTION_PROMPT,
    FOLLOW_UP_DECISION_PROMPT,
    TONGUE_REQUEST_PROMPT,

    # 复杂度评估
    COMPLEXITY_ASSESSMENT_PROMPT,

    # 简单辨证
    SIMPLE_DIAGNOSIS_PROMPT,

    # 中等辨证（RAG辅助）
    MODERATE_DIAGNOSIS_PROMPT,

    # 结果生成
    DIAGNOSIS_RESULT_GENERATION_PROMPT,
    SEEK_DOCTOR_CRITERIA,
)

# ============================================================
# 从 deepsearch_prompts 导入
# ============================================================
from .deepsearch_prompts import (
    # DeepSearch 主控
    DEEPSEARCH_ORCHESTRATOR_PROMPT,

    # 六大专家
    SYMPTOM_ANALYSIS_EXPERT_PROMPT,      # 症状分析专家
    BA_GANG_EXPERT_PROMPT,               # 八纲辨证专家
    ORGAN_ANALYSIS_EXPERT_PROMPT,        # 脏腑辨证专家
    ETIOLOGY_PATHOGENESIS_EXPERT_PROMPT, # 病因病机专家
    YUNQI_ANALYSIS_EXPERT_PROMPT,        # 五运六气专家
    COMPREHENSIVE_EVALUATION_EXPERT_PROMPT,  # 综合评估专家

    # 辅助专家
    DIFFERENTIAL_DIAGNOSIS_EXPERT_PROMPT,    # 鉴别诊断专家
    TREATMENT_PRINCIPLE_EXPERT_PROMPT,       # 治则治法专家
    PRESCRIPTION_RECOMMENDATION_EXPERT_PROMPT,  # 方药推荐专家
    PROGNOSIS_EVALUATION_EXPERT_PROMPT,      # 预后评估专家
    VERIFICATION_EXPERT_PROMPT,              # 质疑验证专家

    # 迭代优化
    DEEPSEARCH_ITERATION_PROMPT,
)

# ============================================================
# 从 multimodal_prompts 导入
# ============================================================
from .multimodal_prompts import (
    TONGUE_ANALYSIS_PROMPT,
    REPORT_ANALYSIS_PROMPT,
    MULTIMODAL_FUSION_PROMPT,
)

# ============================================================
# 导出列表
# ============================================================
__all__ = [
    # === 信息收集 ===
    "SYMPTOM_EXTRACTION_PROMPT",
    "FOLLOW_UP_DECISION_PROMPT",
    "TONGUE_REQUEST_PROMPT",

    # === 复杂度评估 ===
    "COMPLEXITY_ASSESSMENT_PROMPT",

    # === 辨证 ===
    "SIMPLE_DIAGNOSIS_PROMPT",
    "MODERATE_DIAGNOSIS_PROMPT",

    # === 结果生成 ===
    "DIAGNOSIS_RESULT_GENERATION_PROMPT",
    "SEEK_DOCTOR_CRITERIA",

    # === DeepSearch 主控 ===
    "DEEPSEARCH_ORCHESTRATOR_PROMPT",

    # === DeepSearch 六大专家 ===
    "SYMPTOM_ANALYSIS_EXPERT_PROMPT",
    "BA_GANG_EXPERT_PROMPT",
    "ORGAN_ANALYSIS_EXPERT_PROMPT",
    "ETIOLOGY_PATHOGENESIS_EXPERT_PROMPT",
    "YUNQI_ANALYSIS_EXPERT_PROMPT",
    "COMPREHENSIVE_EVALUATION_EXPERT_PROMPT",

    # === DeepSearch 辅助专家 ===
    "DIFFERENTIAL_DIAGNOSIS_EXPERT_PROMPT",
    "TREATMENT_PRINCIPLE_EXPERT_PROMPT",
    "PRESCRIPTION_RECOMMENDATION_EXPERT_PROMPT",
    "PROGNOSIS_EVALUATION_EXPERT_PROMPT",
    "VERIFICATION_EXPERT_PROMPT",

    # === DeepSearch 迭代 ===
    "DEEPSEARCH_ITERATION_PROMPT",

    # === 多模态 ===
    "TONGUE_ANALYSIS_PROMPT",
    "REPORT_ANALYSIS_PROMPT",
    "MULTIMODAL_FUSION_PROMPT",
]


# ============================================================
# 提示词配置
# ============================================================

class DiagnosePromptConfig:
    """诊断提示词配置"""

    # 追问相关
    MAX_FOLLOW_UP_ROUNDS = 5
    MIN_REQUIRED_CATEGORIES = 4

    # DeepSearch 相关
    DEEPSEARCH_MAX_ITERATIONS = 3

    # 专家列表（按执行顺序）
    DEEPSEARCH_EXPERT_SEQUENCE = [
        "symptom_analysis",
        "ba_gang",
        "organ_analysis",
        "etiology_pathogenesis",
        "yunqi_analysis",
        "comprehensive_evaluation",
    ]

    # 验证专家（最后执行）
    VERIFICATION_EXPERTS = [
        "differential_diagnosis",
        "verification",
    ]

    # 治疗相关专家（辨证完成后执行）
    TREATMENT_EXPERTS = [
        "treatment_principle",
        "prescription_recommendation",
        "prognosis_evaluation",
    ]


# ============================================================
# 提示词模板工具函数
# ============================================================

def format_diagnosis_prompt(
    prompt_template: str,
    **kwargs
) -> str:
    """
    格式化诊断提示词模板

    Args:
        prompt_template: 提示词模板
        **kwargs: 模板变量

    Returns:
        str: 格式化后的提示词
    """
    try:
        return prompt_template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing required template variable: {e}")


def get_expert_prompt(expert_name: str) -> str:
    """
    根据专家名称获取对应的提示词

    Args:
        expert_name: 专家名称

    Returns:
        str: 对应的提示词模板
    """
    expert_prompts = {
        "symptom_analysis": SYMPTOM_ANALYSIS_EXPERT_PROMPT,
        "ba_gang": BA_GANG_EXPERT_PROMPT,
        "organ_analysis": ORGAN_ANALYSIS_EXPERT_PROMPT,
        "etiology_pathogenesis": ETIOLOGY_PATHOGENESIS_EXPERT_PROMPT,
        "yunqi_analysis": YUNQI_ANALYSIS_EXPERT_PROMPT,
        "comprehensive_evaluation": COMPREHENSIVE_EVALUATION_EXPERT_PROMPT,
        "differential_diagnosis": DIFFERENTIAL_DIAGNOSIS_EXPERT_PROMPT,
        "treatment_principle": TREATMENT_PRINCIPLE_EXPERT_PROMPT,
        "prescription_recommendation": PRESCRIPTION_RECOMMENDATION_EXPERT_PROMPT,
        "prognosis_evaluation": PROGNOSIS_EVALUATION_EXPERT_PROMPT,
        "verification": VERIFICATION_EXPERT_PROMPT,
    }

    if expert_name not in expert_prompts:
        raise ValueError(f"Unknown expert: {expert_name}")

    return expert_prompts[expert_name]
