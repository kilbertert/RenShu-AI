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


async def complex_diagnosis(state, config):
    """复杂病例的安全降级路径。

    DeepSearch 的向量、古籍和网页工具尚未完成真实数据接入，因此复杂病例明确
    复用 moderate 的可审计 Neo4j GraphRAG，不调用任何 mock 工具。
    """
    result = await moderate_diagnosis(state, config)
    diagnosis_payload = result.get("diagnosis_result")
    if isinstance(diagnosis_payload, dict):
        from .models import CollectedDiagnoseInfo, DiagnosisResult
        from .structured_diagnosis import apply_clinical_safety_bounds

        diagnosis = DiagnosisResult.model_validate(diagnosis_payload)
        collected_info = CollectedDiagnoseInfo(**(state.get("collected_info") or {}))
        apply_clinical_safety_bounds(
            diagnosis,
            collected_info,
            complexity_level="complex",
            report_analysis=state.get("report_analysis"),
        )
        result["diagnosis_result"] = diagnosis.model_dump()
        result["answer"] = diagnosis.to_display()
    result["steps"] = [
        "复杂辨证: 使用可审计 GraphRAG 安全路径；多专家 DeepSearch 尚未启用"
    ] + result.get("steps", [])
    return result


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

    # 5. 各辨证节点 → 病例落库 → 结束
    # 落库节点内部吞掉异常，不会阻断诊断结果返回。
    from .nodes.save_case import save_case_node  # noqa: E402
    workflow.add_node("save_case", save_case_node)
    workflow.add_edge("simple_diagnosis", "save_case")
    workflow.add_edge("moderate_diagnosis", "save_case")
    workflow.add_edge("complex_diagnosis", "save_case")
    workflow.add_edge("save_case", END)

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
