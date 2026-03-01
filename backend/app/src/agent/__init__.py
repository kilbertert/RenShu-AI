"""
TCM Multi-Agent System
中医多智能体系统

基于 LangGraph 构建的中医问诊多智能体系统，支持：
- 中医闲聊/养生咨询
- 辨证问诊（四诊合参）
- 药材咨询（功效、禁忌、配伍）
- 方剂推荐
- 古籍检索
- 医案参考
- 舌诊图像分析（开发中）

使用方法：
    from app.src.agent import build_tcm_graph, TCMAgentService

    # 方式1：直接使用图
    graph = build_tcm_graph()
    result = await graph.ainvoke(input_state, config)

    # 方式2：使用服务
    service = TCMAgentService()
    result = await service.chat("我最近总是怕冷", user_id="xxx")

    # 方式3：FastAPI集成
    from app.src.agent import tcm_router
    app.include_router(tcm_router)
"""

# 核心构建器
from .tcm_builder import build_tcm_graph, new_thread_id, get_llm

# 状态定义
from .tcm_states import (
    # 主状态
    TCMInputState,
    TCMAgentState,
    TCMOutputState,
    # 路由
    TCMRouter,
    TCMQueryType,
    # 辨证相关
    DiagnoseStage,
    SyndromeResult,
    # 药材相关
    HerbInfo,
    HerbCompatibilityResult,
    # 方剂相关
    PrescriptionInfo,
    # 古籍/医案
    ClassicReference,
    CaseReference,
    # 舌诊
    TongueAnalysisResult,
    # 知识图谱子图状态
    KGInputState,
    KGOverallState,
    KGOutputState,
    KGTask,
    CypherExecutionState,
)

# 提示词
from .tcm_prompts import (
    TCM_ROUTER_SYSTEM_PROMPT,
    TCM_DIAGNOSE_SYSTEM_PROMPT,
    TCM_HERB_SYSTEM_PROMPT,
    TCM_PRESCRIPTION_SYSTEM_PROMPT,
    TCM_CLASSICS_SYSTEM_PROMPT,
    TCM_CASE_SYSTEM_PROMPT,
    TCM_GUARDRAILS_SYSTEM_PROMPT,
    TCM_WELLNESS_SYSTEM_PROMPT,
    TCM_TONGUE_ANALYSIS_PROMPT,
    TCM_PLANNER_SYSTEM_PROMPT,
    TCM_SUMMARIZE_SYSTEM_PROMPT,
)

# 工具定义
# from .tcm_tools import (
#     # 工具Schema
#     CypherQueryTool,
#     PredefinedCypherTool,
#     GraphRAGSearchTool,
#     VectorSearchTool,
#     # Cypher模板
#     PREDEFINED_CYPHER_TEMPLATES,
#     get_cypher_template,
#     list_available_templates,
#     # 配伍数据
#     HERB_INCOMPATIBILITY,
#     check_herb_compatibility,
#     # Schema
#     TCM_NEO4J_SCHEMA,
# )

# Neo4j连接
from .tcm_neo4j import (
    TCMNeo4jConnection,
    get_tcm_neo4j_connection,
    get_neo4j_graph,
    execute_cypher_async,
)

# 知识图谱子图
# from .kg_subgraph import build_kg_subgraph

# 校验器
from .tcm_validators import (
    validate_tcm_prescription,
    validate_syndrome_herb_match,
    validate_pregnancy_safety,
    TCMValidationResult,
    SyndromeHerbValidationResult,
    PregnancyValidationResult,
    SYNDROME_HERB_NATURE_MAP,
    HERB_NATURE_DATA,
    PREGNANCY_CONTRAINDICATED_HERBS,
)

# 服务
from .tcm_service import (
    TCMAgentService,
    get_tcm_agent_service,
)

# 控制器（FastAPI路由）
# from .tcm_controller import router as tcm_router
#
# # 主入口
# from .main import run_tcm_agent, interactive_session


__all__ = [
    # 构建器
    "build_tcm_graph",
    "new_thread_id",
    "get_llm",
    # "build_kg_subgraph",

    # 状态
    "TCMInputState",
    "TCMAgentState",
    "TCMOutputState",
    "TCMRouter",
    "TCMQueryType",
    "DiagnoseStage",
    "SyndromeResult",
    "HerbInfo",
    "HerbCompatibilityResult",
    "PrescriptionInfo",
    "ClassicReference",
    "CaseReference",
    "TongueAnalysisResult",
    "KGInputState",
    "KGOverallState",
    "KGOutputState",
    "KGTask",
    "CypherExecutionState",

    # 提示词
    "TCM_ROUTER_SYSTEM_PROMPT",
    "TCM_DIAGNOSE_SYSTEM_PROMPT",
    "TCM_HERB_SYSTEM_PROMPT",
    "TCM_PRESCRIPTION_SYSTEM_PROMPT",
    "TCM_CLASSICS_SYSTEM_PROMPT",
    "TCM_CASE_SYSTEM_PROMPT",
    "TCM_GUARDRAILS_SYSTEM_PROMPT",
    "TCM_WELLNESS_SYSTEM_PROMPT",
    "TCM_TONGUE_ANALYSIS_PROMPT",
    "TCM_PLANNER_SYSTEM_PROMPT",
    "TCM_SUMMARIZE_SYSTEM_PROMPT",

    # 工具
    # "CypherQueryTool",
    # "PredefinedCypherTool",
    # "GraphRAGSearchTool",
    # "VectorSearchTool",
    # "PREDEFINED_CYPHER_TEMPLATES",
    # "get_cypher_template",
    # "list_available_templates",
    # "HERB_INCOMPATIBILITY",
    # "check_herb_compatibility",
    # "TCM_NEO4J_SCHEMA",

    # Neo4j
    "TCMNeo4jConnection",
    "get_tcm_neo4j_connection",
    "get_neo4j_graph",
    "execute_cypher_async",

    # 校验器
    "validate_tcm_prescription",
    "validate_syndrome_herb_match",
    "validate_pregnancy_safety",
    "TCMValidationResult",
    "SyndromeHerbValidationResult",
    "PregnancyValidationResult",
    "SYNDROME_HERB_NATURE_MAP",
    "HERB_NATURE_DATA",
    "PREGNANCY_CONTRAINDICATED_HERBS",

    # 服务
    "TCMAgentService",
    "get_tcm_agent_service",

    # 路由
    # "tcm_router",
    #
    # # 主入口
    # "run_tcm_agent",
    # "interactive_session",
]

__version__ = "1.0.0"
__author__ = "SmartTCM Team"
