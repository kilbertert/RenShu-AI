"""
中等辨证 Map-Reduce 子图构建器

使用 LangGraph Send() 实现并行查询的子图
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .moderate_diagnosis_map_reduce import (
    ModerateState,
    plan_queries,
    map_queries_to_executors,
    execute_query,
    synthesize_diagnosis,
)
from app.src.utils import get_logger

logger = get_logger("moderate_map_reduce_builder")


def create_moderate_map_reduce_graph():
    """
    创建中等辨证的 Map-Reduce 子图

    流程：
    ┌─────────────────────────────────────────────────────────────┐
    │           中等辨证 Map-Reduce 子图                           │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  START                                                      │
    │    │                                                        │
    │    ▼                                                        │
    │  [plan_queries] ─── 任务分解                                │
    │    │                                                        │
    │    │ map_queries_to_executors (返回 Send() 列表)            │
    │    ├──────────────┬──────────────┬────────────────┐        │
    │    ▼              ▼              ▼                ▼        │
    │  [execute_query] [execute_query] [execute_query]           │
    │  (syndrome)      (case)          (prescription)            │
    │    │              │              │                │        │
    │    └──────────────┴──────────────┴────────────────┘        │
    │                   │ (自动 Reduce)                           │
    │                   ▼                                         │
    │            [synthesize_diagnosis] ─── 综合分析              │
    │                   │                                         │
    │                   ▼                                         │
    │                  END                                        │
    └─────────────────────────────────────────────────────────────┘

    Returns:
        CompiledGraph: 编译后的子图
    """
    logger.info("创建中等辨证 Map-Reduce 子图...")

    # 创建状态图
    workflow = StateGraph(ModerateState)

    # ============== 添加节点 ==============

    # 1. 任务分解节点
    workflow.add_node("plan_queries", plan_queries)

    # 2. 查询执行节点（会被并行调用多次）
    workflow.add_node("execute_query", execute_query)

    # 3. 综合分析节点
    workflow.add_node("synthesize_diagnosis", synthesize_diagnosis)

    # ============== 添加边 ==============

    # 1. 入口 → 任务分解
    workflow.add_edge(START, "plan_queries")

    # 2. 任务分解 → 并行执行（Map-Reduce）
    # ★★★ 关键：使用 conditional_edges + map 函数实现并行 ★★★
    workflow.add_conditional_edges(
        "plan_queries",
        map_queries_to_executors,  # 返回 Send() 列表
        ["execute_query"],  # 目标节点列表
    )

    # 3. 并行执行 → 综合分析（自动 Reduce）
    # LangGraph 会自动等待所有 execute_query 完成后再调用 synthesize_diagnosis
    workflow.add_edge("execute_query", "synthesize_diagnosis")

    # 4. 综合分析 → 结束
    workflow.add_edge("synthesize_diagnosis", END)

    # 编译子图
    logger.info("中等辨证 Map-Reduce 子图创建完成")
    return workflow.compile()


# 创建全局子图实例
_moderate_map_reduce_graph = None


def get_moderate_map_reduce_graph():
    """获取中等辨证 Map-Reduce 子图实例（单例）"""
    global _moderate_map_reduce_graph
    if _moderate_map_reduce_graph is None:
        _moderate_map_reduce_graph = create_moderate_map_reduce_graph()
    return _moderate_map_reduce_graph
