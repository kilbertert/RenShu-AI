"""
诊断子图路由逻辑
"""

from .states import DiagnoseOverallState, DiagnoseOutputState
from .models import ComplexityLevel
from langgraph.graph import END


def route_collection(state: DiagnoseOverallState) -> str:
    """
    信息收集阶段的路由

    根据 next_action 决定下一步：
    - "ask_symptom": 继续收集信息
    - "request_tongue": 等待舌像（暂时跳过，继续评估）
    - "request_report": 等待报告（暂时跳过，继续评估）
    - "assess_complexity": 进入复杂度评估
    - "intent_switch": 退出子图

    Args:
        state: 当前状态

    Returns:
        str: 下一个节点名称
    """
    next_action = state.get("next_action", "")

    if next_action == "ask_symptom":
        # 继续收集信息（回到 collect_info）
        return "collect_info"
    elif next_action == "request_tongue":
        # TODO: 实现舌像上传等待逻辑
        # 暂时跳过，直接进入评估
        return "assess_complexity"
    elif next_action == "request_report":
        # TODO: 实现报告上传等待逻辑
        # 暂时跳过，直接进入评估
        return "assess_complexity"
    elif next_action == "assess_complexity":
        # 进入复杂度评估
        return "assess_complexity"
    elif next_action == "intent_switch":
        # 用户切换了意图，退出子图
        return END
    else:
        # 默认进入评估
        return "assess_complexity"


def route_by_complexity(state: DiagnoseOverallState) -> str:
    """
    根据复杂度路由到不同的辨证节点

    Args:
        state: 当前状态

    Returns:
        str: 下一个节点名称
    """
    complexity_dict = state.get("complexity")

    if not complexity_dict:
        # 如果没有复杂度评估结果，默认简单
        return "simple_diagnosis"

    level = complexity_dict.get("level", ComplexityLevel.SIMPLE.value)

    if level == ComplexityLevel.SIMPLE.value:
        return "simple_diagnosis"
    elif level == ComplexityLevel.MODERATE.value:
        return "moderate_diagnosis"
    elif level == ComplexityLevel.COMPLEX.value:
        # TODO: 实现 DeepSearch Agent
        # 暂时降级到中等辨证
        return "moderate_diagnosis"
    else:
        # 默认简单辨证
        return "simple_diagnosis"
