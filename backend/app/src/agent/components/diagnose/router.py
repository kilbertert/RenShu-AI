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
        # 条件边映射使用的是业务键 "intent_switch"；直接返回 END 会被
        # LangGraph 当作未知分支 "__end__"，继而触发诊断 LLM 兜底。
        return "intent_switch"
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
        # 进入显式的复杂病例降级节点。该节点复用已去 mock 的 moderate RAG，
        # 不启用仍含模拟检索的 DeepSearch 工具链。
        return "complex_diagnosis"
    else:
        # 默认简单辨证
        return "simple_diagnosis"
