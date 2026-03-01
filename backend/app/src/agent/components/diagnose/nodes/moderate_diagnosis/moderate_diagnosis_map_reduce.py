"""
中等辨证节点 - Map-Reduce 并行版本

使用 LangGraph Send() 实现 Map-Reduce 模式的并行查询
"""

from typing import Dict, Any, List, Annotated
from operator import add
from typing_extensions import TypedDict
from langgraph.types import Send

from ...states import DiagnoseOverallState
from ...models import CollectedDiagnoseInfo
from app.src.utils import get_logger

logger = get_logger("moderate_diagnosis_map_reduce")


# ============== 状态定义 ==============

class QueryTask(TypedDict):
    """单个查询任务"""
    task_id: str
    task_type: str  # "syndrome" | "case" | "prescription"
    query: str
    collected_info: Dict[str, Any]


class QueryResult(TypedDict):
    """单个查询结果"""
    task_id: str
    task_type: str
    result: Any
    error: str | None


class ModerateState(TypedDict):
    """中等辨证的内部状态"""
    # 输入
    collected_info: Dict[str, Any]
    tongue_analysis: Dict[str, Any] | None
    user_profile: Dict[str, Any]
    llm_config: Any

    # 中间状态
    query_tasks: Annotated[List[QueryTask], add]  # 查询任务列表
    query_results: Annotated[List[QueryResult], add]  # 查询结果列表（自动 Reduce）

    # 输出
    answer: str
    steps: Annotated[List[str], add]


# ============== 节点 1：Planner（任务分解） ==============

async def plan_queries(state: DiagnoseOverallState) -> Dict[str, Any]:
    """
    任务分解节点：将辨证需求分解为并行查询任务

    分解策略：
    1. 证型查询（Neo4j）
    2. 医案检索（向量检索）
    3. 方剂查询（知识图谱）
    """
    logger.info("开始任务分解...")

    # 获取已收集的信息
    collected_info_dict = state.get("collected_info", {})
    if not collected_info_dict:
        return {
            "answer": "暂无收集到的症状信息，无法进行辨证。",
            "steps": ["中等辨证: 信息不足"],
        }

    collected_info = CollectedDiagnoseInfo(**collected_info_dict)
    symptoms = collected_info.get_all_symptoms()

    if not symptoms:
        return {
            "answer": "暂无明确症状，无法进行辨证分析。",
            "steps": ["中等辨证: 症状不足"],
        }

    # 构建查询描述
    symptoms_str = "、".join(symptoms[:5])  # 最多5个症状

    # 分解为3个并行任务
    query_tasks = [
        QueryTask(
            task_id="syndrome_query",
            task_type="syndrome",
            query=f"查询症状【{symptoms_str}】对应的证型",
            collected_info=collected_info_dict,
        ),
        QueryTask(
            task_id="case_query",
            task_type="case",
            query=f"检索与症状【{symptoms_str}】相似的医案",
            collected_info=collected_info_dict,
        ),
        QueryTask(
            task_id="prescription_query",
            task_type="prescription",
            query=f"查询适用于症状【{symptoms_str}】的方剂",
            collected_info=collected_info_dict,
        ),
    ]

    logger.info(f"任务分解完成：{len(query_tasks)} 个并行任务")

    return {
        "query_tasks": query_tasks,
        "steps": ["中等辨证: 任务分解完成"],
    }


# ============== Map 函数：并行分发 ==============

def map_queries_to_executors(state: ModerateState) -> List[Send]:
    """
    Map 函数：将每个查询任务并行分发到执行节点

    Returns:
        List[Send]: Send 对象列表，LangGraph 会并行执行
    """
    tasks = state.get("query_tasks", [])

    logger.info(f"并行分发 {len(tasks)} 个查询任务")

    return [
        Send(
            "execute_query",  # 目标节点
            {
                "task_id": task["task_id"],
                "task_type": task["task_type"],
                "query": task["query"],
                "collected_info": task["collected_info"],
            }
        )
        for task in tasks
    ]


# ============== 节点 2：Execute Query（并行执行） ==============

async def execute_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    查询执行节点：根据任务类型执行对应的查询

    这个节点会被并行调用多次（每个任务一次）
    """
    task_id = state.get("task_id")
    task_type = state.get("task_type")
    query = state.get("query")
    collected_info_dict = state.get("collected_info", {})

    logger.info(f"执行查询任务: {task_id} ({task_type})")

    try:
        collected_info = CollectedDiagnoseInfo(**collected_info_dict)

        # 根据任务类型调用不同的查询函数
        if task_type == "syndrome":
            result = await _query_similar_syndromes(collected_info)
        elif task_type == "case":
            result = await _query_similar_cases(collected_info)
        elif task_type == "prescription":
            result = await _query_related_prescriptions(collected_info)
        else:
            result = []

        logger.info(f"查询任务 {task_id} 完成，结果数: {len(result)}")

        return {
            "query_results": [
                QueryResult(
                    task_id=task_id,
                    task_type=task_type,
                    result=result,
                    error=None,
                )
            ],
            "steps": [f"查询执行: {task_id} 完成"],
        }

    except Exception as e:
        logger.error(f"查询任务 {task_id} 失败: {e}", exc_info=True)
        return {
            "query_results": [
                QueryResult(
                    task_id=task_id,
                    task_type=task_type,
                    result=[],
                    error=str(e),
                )
            ],
            "steps": [f"查询执行: {task_id} 失败 - {str(e)}"],
        }


# ============== 节点 3：Synthesize（结果综合） ==============

async def synthesize_diagnosis(state: ModerateState) -> Dict[str, Any]:
    """
    综合分析节点：汇总所有查询结果，生成最终辨证

    LangGraph 会自动等待所有并行任务完成后再调用此节点
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    from datetime import datetime
    from ...config import diagnose_config
    from .....tcm_builder import get_llm
    from ...prompts.diagnosis_prompts import MODERATE_DIAGNOSIS_PROMPT

    logger.info("开始综合分析...")
    
    def _get_current_solar_term() -> str:
        """获取当前节气（简化版）"""
        now = datetime.now()
        month = now.month
        
        # 简化的节气映射
        solar_terms = {
            1: "小寒/大寒", 2: "立春/雨水", 3: "惊蛰/春分",
            4: "清明/谷雨", 5: "立夏/小满", 6: "芒种/夏至",
            7: "小暑/大暑", 8: "立秋/处暑", 9: "白露/秋分",
            10: "寒露/霜降", 11: "立冬/小雪", 12: "大雪/冬至"
        }
        
        return solar_terms.get(month, "未知节气")

    # 获取所有查询结果
    query_results = state.get("query_results", [])

    # 按类型分组结果
    syndrome_results = []
    case_results = []
    prescription_results = []

    for qr in query_results:
        if qr.get("error"):
            logger.warning(f"查询 {qr['task_id']} 有错误: {qr['error']}")
            continue

        if qr["task_type"] == "syndrome":
            syndrome_results = qr["result"]
        elif qr["task_type"] == "case":
            case_results = qr["result"]
        elif qr["task_type"] == "prescription":
            prescription_results = qr["result"]

    # 获取其他上下文信息
    collected_info_dict = state.get("collected_info", {})
    collected_info = CollectedDiagnoseInfo(**collected_info_dict)
    collected_summary = collected_info.to_summary()

    tongue_analysis = state.get("tongue_analysis")
    tongue_desc = _format_tongue_analysis(tongue_analysis)

    user_profile = state.get("user_profile", {})
    user_profile_desc = _format_user_profile(user_profile)
    
    # 获取当前节气
    solar_term = _get_current_solar_term()

    # 构建提示词
    prompt = MODERATE_DIAGNOSIS_PROMPT.format(
        collected_info=collected_summary,
        tongue_analysis=tongue_desc,
        user_profile=user_profile_desc,
        syndrome_matches=_format_syndromes(syndrome_results),
        similar_cases=_format_cases(case_results),
        related_prescriptions=_format_prescriptions(prescription_results),
        solar_term=solar_term,
    )

    # 调用 LLM 生成辨证结果
    llm = get_llm(
        llm_config=state.get("llm_config"),
        temperature=diagnose_config.DIAGNOSIS_TEMPERATURE
    )

    response = await llm.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content="请结合参考资料开始您的辨证分析。")
    ])

    answer = response.content

    logger.info("综合分析完成")

    return {
        "answer": answer,
        "steps": ["中等辨证: 综合分析完成（Map-Reduce 并行）"],
    }


# ============== 查询函数（与原版相同） ==============

async def _query_similar_syndromes(collected_info: CollectedDiagnoseInfo) -> List[Dict[str, Any]]:
    """查询相似证型（预定义 Cypher）"""
    import asyncio
    # TODO: 实现知识图谱查询
    symptoms = collected_info.get_all_symptoms()
    if not symptoms:
        return []

    # 模拟网络延迟
    await asyncio.sleep(0.5)

    return [
        {
            "name": "气虚证",
            "symptoms": ["乏力", "气短", "自汗"],
            "similarity": 0.8,
        },
        {
            "name": "脾气虚证",
            "symptoms": ["乏力", "食欲不振", "便溏"],
            "similarity": 0.7,
        },
    ]


async def _query_similar_cases(collected_info: CollectedDiagnoseInfo) -> List[Dict[str, Any]]:
    """查询相似医案（向量检索）"""
    import asyncio
    # TODO: 实现向量检索
    await asyncio.sleep(0.8)

    return [
        {
            "case_id": "case_001",
            "chief_complaint": "乏力、食欲不振",
            "syndrome": "脾气虚证",
            "treatment": "补中益气汤加减",
            "similarity": 0.75,
        },
    ]


async def _query_related_prescriptions(collected_info: CollectedDiagnoseInfo) -> List[Dict[str, Any]]:
    """查询常用方剂（知识图谱）"""
    import asyncio
    # TODO: 实现知识图谱查询
    await asyncio.sleep(0.6)

    return [
        {
            "name": "补中益气汤",
            "indication": "脾胃气虚",
            "composition": ["黄芪", "人参", "白术", "甘草"],
        },
    ]


# ============== 格式化函数 ==============

def _format_tongue_analysis(tongue_analysis: Dict[str, Any] | None) -> str:
    """格式化舌像分析"""
    if not tongue_analysis:
        return "未提供"

    parts = []
    if tongue_analysis.get("tongue_color"): parts.append(f"舌色：{tongue_analysis['tongue_color']}")
    if tongue_analysis.get("tongue_shape"): parts.append(f"舌形：{tongue_analysis['tongue_shape']}")
    if tongue_analysis.get("coating_color"): parts.append(f"苔色：{tongue_analysis['coating_color']}")
    if tongue_analysis.get("coating_quality"): parts.append(f"苔质：{tongue_analysis['coating_quality']}")
    if tongue_analysis.get("analysis"): parts.append(f"分析：{tongue_analysis['analysis']}")

    return "\n".join(parts) if parts else "未提供"


def _format_user_profile(user_profile: Dict[str, Any]) -> str:
    """格式化用户画像"""
    if not user_profile:
        return "暂无用户画像"

    parts = []
    if user_profile.get("age"):
        parts.append(f"年龄：{user_profile['age']}岁")
    if user_profile.get("gender"):
        parts.append(f"性别：{user_profile['gender']}")
    if user_profile.get("constitution"):
        parts.append(f"体质：{user_profile['constitution']}")
    if user_profile.get("chronic_diseases"):
        parts.append(f"慢性病：{', '.join(user_profile['chronic_diseases'])}")

    return "\n".join(parts) if parts else "暂无用户画像"


def _format_syndromes(syndromes: List[Dict[str, Any]]) -> str:
    """格式化证型列表"""
    if not syndromes:
        return "暂无相似证型"

    parts = []
    for i, syndrome in enumerate(syndromes[:3], 1):
        symptoms_str = "、".join(syndrome.get("symptoms", []))
        similarity = syndrome.get("similarity", 0)
        parts.append(f"{i}. {syndrome['name']} (相似度: {similarity:.0%})\n   主要症状：{symptoms_str}")

    return "\n".join(parts)


def _format_cases(cases: List[Dict[str, Any]]) -> str:
    """格式化医案列表"""
    if not cases:
        return "暂无相似医案"

    parts = []
    for i, case in enumerate(cases[:2], 1):
        parts.append(
            f"{i}. 主诉：{case.get('chief_complaint', '未知')}\n"
            f"   证型：{case.get('syndrome', '未知')}\n"
            f"   治疗：{case.get('treatment', '未知')}"
        )

    return "\n".join(parts)


def _format_prescriptions(prescriptions: List[Dict[str, Any]]) -> str:
    """格式化方剂列表"""
    if not prescriptions:
        return "暂无相关方剂"

    parts = []
    for i, prescription in enumerate(prescriptions[:3], 1):
        composition = prescription.get("composition", [])
        composition_str = "、".join(composition[:5])
        parts.append(
            f"{i}. {prescription['name']}\n"
            f"   主治：{prescription.get('indication', '未知')}\n"
            f"   组成：{composition_str}等"
        )

    return "\n".join(parts)
