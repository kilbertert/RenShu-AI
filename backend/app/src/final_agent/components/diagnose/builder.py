"""
诊断子图构建器

构建完整的诊断子图工作流
"""

from langgraph.graph import StateGraph, START, END

from .states import DiagnoseInputState, DiagnoseOverallState, DiagnoseOutputState
from .router import route_collection, route_by_complexity
from .nodes import (
    collect_info,
    analyze_and_follow_up,
    assess_complexity,
    simple_diagnosis,
    moderate_diagnosis,
)
from .nodes.complex import complex_diagnosis


def create_diagnose_graph():
    """
    创建诊断子图

    流程：
    ┌─────────────────────────────────────────────────────────────┐
    │                     诊断子图 (Diagnose Subgraph)             │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │              信息收集循环 (Collection Loop)          │   │
    │  │                                                     │   │
    │  │   [collect_info] ←──────────────────────┐          │   │
    │  │        │                                │          │   │
    │  │        ▼                                │          │   │
    │  │   [analyze_and_follow_up]               │          │   │
    │  │        │                                │          │   │
    │  │   ┌────┴────┬─────────┬────────┐       │          │   │
    │  │   ▼         ▼         ▼        ▼       │          │   │
    │  │ [追问]  [请求舌像]  [请求报告]  [完成] ──┘          │   │
    │  └──────────────┼─────────────────────────────────────┘   │
    │                 ▼                                         │
    │        [assess_complexity] ─── 复杂度评估                  │
    │                 │                                         │
    │      ┌──────────┼──────────┐                             │
    │      ▼          ▼          ▼                             │
    │  [simple]   [moderate]  [complex]                        │
    │      │          │          │                             │
    │      └──────────┴──────────┘                             │
    │                 │                                         │
    │                 ▼                                         │
    │                END                                        │
    └─────────────────────────────────────────────────────────────┘

    Returns:
        CompiledGraph: 编译后的诊断子图
    """
    # 创建状态图
    workflow = StateGraph(
        DiagnoseOverallState,
        input=DiagnoseInputState,
        output=DiagnoseOutputState
    )

    # ============== 添加节点 ==============

    # 信息收集循环
    workflow.add_node("collect_info", collect_info)
    workflow.add_node("analyze_follow_up", analyze_and_follow_up)

    # 复杂度评估
    workflow.add_node("assess_complexity", assess_complexity)

    # 辨证节点
    workflow.add_node("simple_diagnosis", simple_diagnosis)
    workflow.add_node("moderate_diagnosis", moderate_diagnosis)
    workflow.add_node("complex_diagnosis", complex_diagnosis)  # DeepSearch Agent

    # ============== 添加边 ==============

    # 1. 入口 → 信息收集
    workflow.add_edge(START, "collect_info")

    # 2. 信息收集 → 分析追问
    workflow.add_edge("collect_info", "analyze_follow_up")

    # 3. 分析追问 → 条件路由（收集循环或进入评估）
    workflow.add_conditional_edges(
        "analyze_follow_up",
        route_collection,
        {
            "collect_info": "collect_info",           # 继续收集
            "assess_complexity": "assess_complexity",  # 进入评估
            "intent_switch": END,                      # 直接结束（意图切换等）
        }
    )

    # 4. 复杂度评估 → 条件路由（不同辨证策略）
    workflow.add_conditional_edges(
        "assess_complexity",
        route_by_complexity,
        {
            "simple_diagnosis": "simple_diagnosis",
            "moderate_diagnosis": "moderate_diagnosis",
            "complex_diagnosis": "complex_diagnosis",  # DeepSearch Agent
        }
    )

    # 5. 各辨证节点 → 结束
    workflow.add_edge("simple_diagnosis", END)
    workflow.add_edge("moderate_diagnosis", END)
    workflow.add_edge("complex_diagnosis", END)  # DeepSearch Agent

    # 编译子图
    return workflow.compile()


# 创建全局子图实例
_diagnose_graph = None


def get_diagnose_graph():
    """获取诊断子图实例（单例）"""
    global _diagnose_graph
    if _diagnose_graph is None:
        _diagnose_graph = create_diagnose_graph()
    return _diagnose_graph
