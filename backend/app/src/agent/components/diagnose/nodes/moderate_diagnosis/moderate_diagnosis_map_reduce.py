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
    task_type: str  # "graphrag"
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
    query: str
    collected_info: Dict[str, Any]
    tongue_analysis: Dict[str, Any] | None
    report_analysis: Dict[str, Any] | None
    user_profile: Dict[str, Any]
    llm_config: Any

    # 中间状态
    query_tasks: Annotated[List[QueryTask], add]  # 查询任务列表
    query_results: Annotated[List[QueryResult], add]  # 查询结果列表（自动 Reduce）

    # 输出
    answer: str
    diagnosis_result: Dict[str, Any]
    steps: Annotated[List[str], add]


# ============== 节点 1：Planner（任务分解） ==============

async def plan_queries(state: DiagnoseOverallState) -> Dict[str, Any]:
    """
    任务分解节点：将辨证需求分解为并行查询任务

    分解策略：先只检索诊断轴。方剂必须等最终主证确定后，再通过显式
    证型-方剂关系查询，不能与证型检索并行或由症状反推。
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

    # 诊断阶段只检索证候证据；证方关系在综合辨证完成后查询。
    query_tasks = [
        QueryTask(
            task_id="graphrag_query",
            task_type="graphrag",
            query=f"检索症状【{symptoms_str}】对应的可审计诊断子图",
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
        if task_type == "graphrag":
            result = (await _query_diagnostic_graph(collected_info)).model_dump()
        else:
            result = []

        result_count = (
            len(result.get("candidates", []))
            if task_type == "graphrag" and isinstance(result, dict)
            else len(result)
        )
        logger.info(f"查询任务 {task_id} 完成，结果数: {result_count}")

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
    from ...models import (
        DiagnosisCitation,
        DiagnosisPrescription,
        PrescriptionRelationEvidence,
    )
    from ...structured_diagnosis import (
        apply_clinical_safety_bounds,
        generate_structured_diagnosis,
    )

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
    verified_prescriptions = []
    from app.src.agent.retrieval.graphrag import GraphRAGResult

    graph_rag_result = GraphRAGResult(
        query="",
        retrieval_mode="unavailable",
        graph_available=False,
        warnings=["GraphRAG 任务没有返回结果。"],
    )

    for qr in query_results:
        if qr.get("error"):
            logger.warning(f"查询 {qr['task_id']} 有错误: {qr['error']}")
            continue

        if qr["task_type"] == "graphrag":
            graph_rag_result = GraphRAGResult.model_validate(qr["result"])
            syndrome_results = graph_rag_result.to_legacy_candidates()

    # 获取其他上下文信息
    collected_info_dict = state.get("collected_info", {})
    collected_info = CollectedDiagnoseInfo(**collected_info_dict)
    collected_summary = collected_info.to_summary()

    tongue_analysis = state.get("tongue_analysis")
    tongue_desc = _format_tongue_analysis(tongue_analysis)
    report_desc = _format_report_analysis(state.get("report_analysis"))

    user_profile = state.get("user_profile", {})
    user_profile_desc = _format_user_profile(user_profile)
    
    # 获取当前节气
    solar_term = _get_current_solar_term()

    # 构建提示词
    prompt = MODERATE_DIAGNOSIS_PROMPT.format(
        user_request=state.get("query", "") or "未特别说明",
        collected_info=collected_summary,
        tongue_analysis=tongue_desc,
        report_analysis=report_desc,
        user_profile=user_profile_desc,
        syndrome_matches=_format_syndromes(syndrome_results),
        similar_cases="当前未接入真实患者医案，不使用症状相似度拼接治疗模式。",
        related_prescriptions=(
            "方剂将在最终主证确定后由后端查询显式证方关系；"
            "本阶段 prescriptions 必须置空。"
        ),
        solar_term=solar_term,
    )
    from app.src.agent.retrieval.graphrag import format_graph_rag_context

    prompt += "\n\n## 可审计 GraphRAG 证据\n" + format_graph_rag_context(
        graph_rag_result
    )

    # 调用 LLM 生成辨证结果
    llm = get_llm(
        llm_config=state.get("llm_config"),
        temperature=diagnose_config.DIAGNOSIS_TEMPERATURE
    )

    diagnosis_result = await generate_structured_diagnosis(
        llm,
        [
            SystemMessage(content=prompt),
            HumanMessage(content="请结合参考资料开始您的辨证分析。"),
        ],
    )
    _attach_graph_identity(diagnosis_result, graph_rag_result)
    verified_prescriptions = await _query_prescriptions_for_syndrome(
        diagnosis_result.syndrome,
        diagnosis_result.syndrome_id,
    )
    _ground_prescriptions_to_syndrome(
        diagnosis_result,
        verified_prescriptions,
        DiagnosisPrescription,
        PrescriptionRelationEvidence,
    )
    diagnosis_result.citations = _build_retrieval_citations(
        syndrome_results,
        [],
        verified_prescriptions,
        DiagnosisCitation,
        graph_rag_result=graph_rag_result,
    )
    apply_clinical_safety_bounds(
        diagnosis_result,
        collected_info,
        report_analysis=state.get("report_analysis"),
    )
    _apply_real_case_boundary(
        diagnosis_result,
        state.get("query", ""),
    )
    answer = diagnosis_result.to_display()

    logger.info("综合分析完成")

    return {
        "answer": answer,
        "diagnosis_result": diagnosis_result.model_dump(),
        "steps": ["中等辨证: 结构化综合分析完成（Map-Reduce 并行）"],
    }


# ============== 查询函数（与原版相同） ==============

async def _query_similar_syndromes(collected_info: CollectedDiagnoseInfo) -> List[Dict[str, Any]]:
    """查询相似证型（Neo4j 知识图谱）。

    委托到主模块实现以避免代码漂移（多策略 HPO 桥接，
    详见 moderate_diagnosis.py）。
    """
    import asyncio
    from .moderate_diagnosis import _query_similar_syndromes as _impl
    await asyncio.sleep(0)
    return await _impl(collected_info)


async def _query_diagnostic_graph(collected_info: CollectedDiagnoseInfo):
    """委托主模块执行 GraphRAG，避免查询逻辑分叉。"""
    from .moderate_diagnosis import _query_diagnostic_graph as _impl

    return await _impl(collected_info)


async def _query_similar_cases(collected_info: CollectedDiagnoseInfo) -> List[Dict[str, Any]]:
    """查询相似医案代理结果，委托主模块的真实 Neo4j 检索实现。"""
    from .moderate_diagnosis import _query_similar_cases as _impl
    return await _impl(collected_info)


async def _query_related_prescriptions(collected_info: CollectedDiagnoseInfo) -> List[Dict[str, Any]]:
    """查询常用方剂（Neo4j 知识图谱）。

    委托到主模块实现以避免代码漂移（ITCM Formula 主数据源 +
    tcm_graph 旧 Prescription 兜底，详见 moderate_diagnosis.py）。
    """
    from .moderate_diagnosis import _query_related_prescriptions as _impl
    return await _impl(collected_info)


async def _query_prescriptions_for_syndrome(
    syndrome: str,
    syndrome_id: str | None = None,
) -> List[Dict[str, Any]]:
    """按最终主证查询显式证方关系，委托主模块实现。"""
    from .moderate_diagnosis import _query_prescriptions_for_syndrome as _impl

    return await _impl(syndrome, syndrome_id)


def _build_retrieval_citations(
    syndromes: List[Dict[str, Any]],
    cases: List[Dict[str, Any]],
    prescriptions: List[Dict[str, Any]],
    citation_model,
    *,
    graph_rag_result=None,
) -> list:
    """只把真实检索结果写入 citations，避免模型虚构来源。"""
    citations = []
    if graph_rag_result is not None:
        for evidence in graph_rag_result.evidences[:6]:
            citations.append(citation_model(
                source_type="graph_path",
                title=f"{evidence.syndrome_name} → {evidence.symptom_name}",
                source=evidence.source_db,
                evidence=evidence.statement,
                score=evidence.score,
                citation_id=evidence.evidence_id,
                node_ids=evidence.node_ids,
                relationship_ids=evidence.relationship_ids,
                relationship_path=evidence.relationship_path,
                matched_keywords=evidence.matched_keywords,
                symptom_role=evidence.symptom_role,
                evidence_weight=evidence.evidence_weight,
            ))
    else:
        for item in syndromes[:3]:
            citations.append(citation_model(
                source_type="syndrome",
                title=item.get("name") or "未命名证型",
                source=item.get("source_db") or item.get("source"),
                evidence="、".join(item.get("symptoms") or item.get("matched_keywords") or []),
                score=min(1.0, float(item.get("similarity") or 0)),
            ))
    for item in cases[:2]:
        citations.append(citation_model(
            source_type="treatment_pattern",
            title=f"{item.get('syndrome', '未知证型')} → {item.get('treatment', '未知治法')}",
            source=item.get("source"),
            evidence=item.get("chief_complaint") or "",
            score=min(1.0, float(item.get("match_score") or item.get("similarity") or 0)),
        ))
    for item in prescriptions[:3]:
        citations.append(citation_model(
            source_type="formula_relation",
            title=(
                f"{item.get('syndrome_name', '未知证型')} → "
                f"{item.get('name') or '未命名方剂'}"
            ),
            source=item.get("source_db") or item.get("source"),
            evidence=item.get("indications") or item.get("effects") or "",
            citation_id=(
                f"{item.get('source_db', 'neo4j')}:"
                f"{item.get('relationship_id', '')}"
            ),
            node_ids=[
                str(value)
                for value in (item.get("syndrome_id"), item.get("formula_id"))
                if value
            ],
            relationship_ids=(
                [str(item["relationship_id"])]
                if item.get("relationship_id")
                else []
            ),
            relationship_path=item.get("relationship_path") or [],
        ))
    return citations


def _apply_real_case_boundary(diagnosis_result, user_query: str) -> None:
    """真实病例库未接入时，禁止把知识图谱治疗模式冒充患者医案。"""
    query = str(user_query or "")
    requests_real_cases = any(
        token in query
        for token in ("真实医案", "真实病例", "病例编号", "医案编号", "病例来源", "相似医案")
    )
    if not requests_real_cases:
        return
    notice = (
        "**真实医案检索边界**：当前尚未配置可用的真实患者医案向量库，"
        "因此无法提供病例编号、原始病例来源或患者级相似度。"
        "下方辨证仅使用知识图谱中的证候、症状和方剂资料，不能视为真实医案匹配。"
    )
    current = diagnosis_result.patient_answer.strip()
    if notice not in current:
        diagnosis_result.patient_answer = f"{notice}\n\n{current}".strip()


def _attach_graph_identity(diagnosis_result, graph_rag_result) -> None:
    """仅在模型主证与图谱规范名一致时附加知识库 ID。"""
    if diagnosis_result.syndrome_id or not graph_rag_result.candidates:
        return
    from app.src.agent.retrieval.graphrag import _canonical_syndrome

    diagnosed_key = _canonical_syndrome(diagnosis_result.syndrome)
    candidate = next(
        (
            item
            for item in graph_rag_result.candidates
            if item.canonical_name == diagnosed_key
        ),
        None,
    )
    if candidate is not None:
        diagnosis_result.syndrome_id = candidate.syndrome_id


def _ground_prescriptions_to_syndrome(
    diagnosis_result,
    prescriptions: List[Dict[str, Any]],
    prescription_model,
    evidence_model,
) -> None:
    """只保留与最终主证存在显式关系、且被模型选中的方剂。"""
    retrieved = {
        _normalize_formula_name(str(item.get("name"))): item
        for item in prescriptions
        if item.get("name")
    }
    grounded = []
    for item in diagnosis_result.prescriptions or []:
        candidate = retrieved.get(_normalize_formula_name(str(item.name)))
        if not candidate:
            continue
        grounded.append(prescription_model(
            name=candidate["name"],
            composition=candidate.get("composition") or [],
            source=candidate.get("source"),
            rationale=(
                candidate.get("effects")
                or candidate.get("effect_zh")
                or candidate.get("indications")
                or candidate.get("indication")
            ),
            cautions=["仅作知识图谱检索参考，具体组成、剂量与加减必须由中医师面诊确定。"],
            relation_evidence=evidence_model(
                source_db=candidate.get("source_db") or "neo4j",
                syndrome_id=candidate.get("syndrome_id"),
                syndrome_name=candidate.get("syndrome_name") or diagnosis_result.syndrome,
                formula_id=candidate.get("formula_id"),
                formula_name=candidate["name"],
                relationship_type=candidate["relationship_type"],
                relationship_id=str(candidate["relationship_id"]),
                relationship_path=candidate.get("relationship_path") or [],
            ),
        ))
    had_unverified_prescription = bool(diagnosis_result.prescriptions)
    diagnosis_result.prescriptions = grounded
    if had_unverified_prescription and not grounded:
        warning = "候选方剂与最终主证之间缺少可追溯图谱关系，本次不输出具体方剂。"
        if warning not in diagnosis_result.warnings:
            diagnosis_result.warnings.append(warning)


def _normalize_formula_name(value: str) -> str:
    """生成方名比对键，避免全半角标点和空白影响关系校验。"""
    import re

    return re.sub(r"[\s，,。；;、（）()【】\[\]·—_\-]+", "", value).lower()


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


def _format_report_analysis(report: Dict[str, Any] | None) -> str:
    if not report:
        return "未提供"
    parts: list[str] = []
    if report.get("report_type"):
        parts.append(f"报告类型：{report['report_type']}")
    if report.get("summary"):
        parts.append(f"摘要：{report['summary']}")
    findings = report.get("key_findings") or []
    if findings:
        parts.append("关键发现：" + "；".join(str(item) for item in findings[:10]))
    abnormal_metrics = [
        item for item in (report.get("metrics") or [])
        if isinstance(item, dict)
        and item.get("abnormal_flag") in {"high", "low", "abnormal", "positive"}
    ]
    if abnormal_metrics:
        parts.append(
            "异常指标：" + "；".join(
                f"{item.get('name', '')} {item.get('value', '')}{item.get('unit', '')}"
                for item in abnormal_metrics[:12]
            )
        )
    parts.append("报告只作辅助证据，不能由单项指标直接确定中医证型。")
    return "\n".join(parts)


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
    """复用主模块格式化，确保 active Map-Reduce 展示完整诊断轴。"""
    from .moderate_diagnosis import _format_syndromes as _impl

    return _impl(syndromes)


def _format_cases(cases: List[Dict[str, Any]]) -> str:
    """格式化证型-方剂治疗模式；这些结果不是患者医案。"""
    if not cases:
        return "当前没有真实患者医案结果，也没有可用的证型-方剂治疗模式。"

    parts = [
        "以下内容是知识图谱中的证型-方剂治疗模式，不是患者病例，"
        "不包含病例编号、原始病历或疗效随访："
    ]
    for i, case in enumerate(cases[:2], 1):
        parts.append(
            f"{i}. 匹配症状：{case.get('chief_complaint', '未知')}\n"
            f"   证型：{case.get('syndrome', '未知')}\n"
            f"   关联方剂：{case.get('treatment', '未知')}"
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
