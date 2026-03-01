"""
复杂诊断状态定义

用于 DeepSearch Agent 的状态管理
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from operator import add


class ComplexDiagnosisState(TypedDict):
    """复杂诊断状态"""

    # 输入信息
    collected_info: Dict[str, Any]  # 收集的患者信息
    preliminary_diagnosis: Optional[str]  # 初步诊断

    # 初步分析结果
    symptom_analysis: Optional[str]  # 症状分析
    ba_gang_analysis: Optional[str]  # 八纲辨证
    organ_analysis: Optional[str]  # 脏腑辨证
    etiology_analysis: Optional[str]  # 病因病机

    # 专家分析结果
    differential_diagnosis_result: Optional[Dict]  # 鉴别诊断结果
    treatment_principle_result: Optional[Dict]  # 治则治法结果
    prescription_result: Optional[Dict]  # 方药推荐结果
    prognosis_result: Optional[Dict]  # 预后评估结果

    # 验证结果
    verification_result: Optional[Dict]  # 验证结果

    # 迭代控制
    iteration_count: int  # 当前迭代次数
    max_iterations: int  # 最大迭代次数
    should_continue_iteration: bool  # 是否继续迭代

    # 最终输出
    final_diagnosis: Optional[Dict]  # 最终诊断结果
    confidence: float  # 置信度

    # 执行步骤记录
    steps: Annotated[List[str], add]  # 执行步骤

    # 错误信息
    error: Optional[str]  # 错误信息
