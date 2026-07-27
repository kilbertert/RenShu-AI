"""
中等辨证节点

中等复杂度的 RAG 辅助辨证
"""

import re
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from datetime import datetime

from app.src.agent.components.diagnose.states import DiagnoseOverallState



from app.src.utils import get_logger

from app.src.agent.components.diagnose.models import CollectedDiagnoseInfo

from app.src.agent.tcm_builder import get_llm
from app.src.agent.components.diagnose.config import diagnose_config
from app.src.agent.components.diagnose.prompts import MODERATE_DIAGNOSIS_PROMPT

logger = get_logger("moderate_diagnosis")


_ATOMIC_SYMPTOM_TERMS = (
    "头痛", "头晕", "眩晕", "乏力", "腰痛", "身痛", "耳鸣",
    "心悸", "气短", "胸闷", "胸痛", "腹胀", "腹痛",
    "便秘", "腹泻", "便溏", "尿频", "夜尿", "失眠", "多梦", "易醒",
    "怕冷", "恶寒", "发热", "潮热", "无汗", "自汗", "盗汗",
    "口渴", "口干", "口苦", "食欲不振", "鼻塞", "流清涕", "咳嗽",
)


def _expand_retrieval_token(token: str) -> List[str]:
    """保留复合症状原文，并追加其中可稳定识别的原子症状。"""
    value = str(token or "").strip()
    if len(value) < 2:
        return []
    expanded = [value]
    expanded.extend(
        term
        for term in _ATOMIC_SYMPTOM_TERMS
        if term != value and term in value
    )
    return list(dict.fromkeys(expanded))


def _canonical_candidate_key(name: str, canonical_name: str | None = None) -> str:
    """生成跨数据源证候去重键，兼容名称末尾是否带“证”。"""
    value = str(canonical_name or name or "").strip().lower()
    value = re.sub(r"[\s，,。；;、（）()【】\[\]·—_\-]+", "", value)
    if value.endswith("证") and len(value) > 1:
        value = value[:-1]
    return value


def _diagnostic_keywords(collected_info: CollectedDiagnoseInfo) -> List[str]:
    """提取 moderate 检索关键词，并把结构化舌象转为可匹配短语。"""
    raw_values = list(collected_info.get_all_symptoms() or [])
    tongue = collected_info.tongue or {}
    tongue_prefixes = {
        "tongue_color": "舌",
        "tongue_shape": "舌",
        "coating_color": "苔",
        "coating_quality": "苔",
    }
    for key, value in tongue.items():
        if key in {"source", "description", "image_quality"}:
            continue
        text = str(value or "").strip()
        if not text:
            continue
        prefix = tongue_prefixes.get(key, "")
        raw_values.append(f"{prefix}{text}" if prefix and not text.startswith(prefix) else text)
    pulse = collected_info.pulse or {}
    pulse_text = pulse.get("description") or pulse.get("pulse")
    if pulse_text:
        pulse_text = str(pulse_text).strip()
        raw_values.append(
            pulse_text if pulse_text.startswith("脉") else f"脉{pulse_text}"
        )

    keywords: List[str] = []
    for value in raw_values:
        if not value:
            continue
        for token in re.split(r"[，。；,;\s、]+", str(value)):
            token = token.strip()
            keywords.extend(_expand_retrieval_token(token))
    return list(dict.fromkeys(keywords))[:20]


def _get_current_solar_term() -> str:
    """获取当前节气（简化版）"""
    now = datetime.now()
    month = now.month
    
    # 简化的节气映射（实际应该更精确）
    solar_terms = {
        1: "小寒/大寒", 2: "立春/雨水", 3: "惊蛰/春分",
        4: "清明/谷雨", 5: "立夏/小满", 6: "芒种/夏至",
        7: "小暑/大暑", 8: "立秋/处暑", 9: "白露/秋分",
        10: "寒露/霜降", 11: "立冬/小雪", 12: "大雪/冬至"
    }
    
    return solar_terms.get(month, "未知节气")


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


async def moderate_diagnosis(state: DiagnoseOverallState, config: RunnableConfig) -> Dict[str, Any]:
    """
    中等复杂度的 RAG 辅助辨证

    优化（2026-02-05）：
    - 使用 Map-Reduce 子图实现并行查询
    - 降低延迟 50-70%（从 7秒降到 3-4秒）

    方法：
    1. 任务分解：将辨证需求分解为并行查询任务
    2. 并行执行：
       - 查询相似证型（Neo4j）
       - 查询相关医案（向量检索）
       - 查询常用方剂（知识图谱）
    3. 结果汇总：LangGraph 自动 Reduce
    4. 综合分析：LLM 结合所有结果生成辨证

    适用场景：
    - 多个症状需要综合判断
    - 可能存在兼证
    - 需要医案参考

    Args:
        state: 当前状态

    Returns:
        dict: 更新的状态字段
    """
    try:
        # 使用 Map-Reduce 子图
        from .moderate_map_reduce_builder import get_moderate_map_reduce_graph

        logger.info("使用 Map-Reduce 子图进行并行查询")

        # 获取子图
        map_reduce_graph = get_moderate_map_reduce_graph()

        # 准备子图输入
        tongue_analysis = state.get("tongue_analysis")
        collected_info_input = dict(state.get("collected_info", {}))
        if tongue_analysis and not collected_info_input.get("tongue"):
            collected_info_input["tongue"] = {
                key: str(value)
                for key, value in tongue_analysis.items()
                if key in {"tongue_color", "tongue_shape", "coating_color", "coating_quality"}
                and value
            }
        subgraph_input = {
            "query": state.get("query", ""),
            "collected_info": collected_info_input,
            "tongue_analysis": tongue_analysis,
            "report_analysis": state.get("report_analysis"),
            "user_profile": state.get("user_profile", {}),
            "llm_config": state.get("llm_config"),
        }

        # 调用子图
        import time
        start_time = time.time()
        result = await map_reduce_graph.ainvoke(subgraph_input, config)
        elapsed = time.time() - start_time

        logger.info(f"Map-Reduce 子图完成，总耗时: {elapsed:.2f}秒")

        # 提取结果
        answer = result.get("answer", "")
        diagnosis_result = result.get("diagnosis_result")
        steps = result.get("steps", [])

        return {
            "answer": answer,
            "diagnosis_result": diagnosis_result,
            "steps": steps + [f"中等辨证: 完成 (Map-Reduce 并行, 耗时 {elapsed:.2f}秒)"],
        }

    except Exception as e:
        logger.error(f"Map-Reduce 子图失败: {e}", exc_info=True)
        # 降级到串行版本
        logger.warning("降级到串行版本")
        return await _fallback_serial_diagnosis(state)


async def _fallback_serial_diagnosis(state: DiagnoseOverallState) -> Dict[str, Any]:
    """
    降级处理：使用 asyncio.gather 并行执行（当 Map-Reduce 子图失败时使用）
    """
    import asyncio
    import time

    try:
        # 获取已收集的信息
        collected_info_dict = state.get("collected_info", {})
        if collected_info_dict:
            collected_info_input = dict(collected_info_dict)
            tongue_analysis = state.get("tongue_analysis")
            if tongue_analysis and not collected_info_input.get("tongue"):
                collected_info_input["tongue"] = {
                    key: str(value)
                    for key, value in tongue_analysis.items()
                    if key in {"tongue_color", "tongue_shape", "coating_color", "coating_quality"}
                    and value
                }
            collected_info = CollectedDiagnoseInfo(**collected_info_input)
            collected_summary = collected_info.to_summary()
        else:
            collected_info = CollectedDiagnoseInfo()
            collected_summary = "暂无详细信息"

        # 获取舌像分析
        tongue_analysis = state.get("tongue_analysis")
        tongue_desc = "未提供"
        if tongue_analysis:
            parts = []
            if tongue_analysis.get("tongue_color"): parts.append(f"舌色：{tongue_analysis['tongue_color']}")
            if tongue_analysis.get("tongue_shape"): parts.append(f"舌形：{tongue_analysis['tongue_shape']}")
            if tongue_analysis.get("coating_color"): parts.append(f"苔色：{tongue_analysis['coating_color']}")
            if tongue_analysis.get("coating_quality"): parts.append(f"苔质：{tongue_analysis['coating_quality']}")
            if tongue_analysis.get("analysis"): parts.append(f"分析：{tongue_analysis['analysis']}")
            tongue_desc = "\n".join(parts)
        report_desc = _format_report_analysis(state.get("report_analysis"))

        # 获取用户画像
        user_profile = state.get("user_profile", {})
        user_profile_desc = _format_user_profile(user_profile)

        # === 并行 RAG 检索（使用 asyncio.gather）===
        logger.info("开始并行 RAG 检索（asyncio.gather）...")
        start_time = time.time()

        # ★★★ 关键：使用 asyncio.gather 并行执行 ★★★
        graph_rag_result = await _query_diagnostic_graph(collected_info)
        similar_cases: List[Dict[str, Any]] = []
        related_prescriptions: List[Dict[str, Any]] = []

        # 处理异常结果
        similar_syndromes = graph_rag_result.to_legacy_candidates()

        elapsed = time.time() - start_time
        logger.info(f"并行 RAG 检索完成 (耗时: {elapsed:.2f}秒): 证型 {len(similar_syndromes)} 个, 医案 {len(similar_cases)} 个, 方剂 {len(related_prescriptions)} 个")

        # === 构建提示词 ===
        solar_term = _get_current_solar_term()
        
        prompt = MODERATE_DIAGNOSIS_PROMPT.format(
            user_request=state.get("query", "") or "未特别说明",
            collected_info=collected_summary,
            tongue_analysis=tongue_desc,
            report_analysis=report_desc,
            user_profile=user_profile_desc,
            syndrome_matches=_format_syndromes(similar_syndromes),
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

        # === 调用 LLM ===
        llm = get_llm(
            llm_config=state.get("llm_config"),
            temperature=diagnose_config.DIAGNOSIS_TEMPERATURE
        )

        from app.src.agent.components.diagnose.models import (
            DiagnosisCitation,
            DiagnosisPrescription,
            PrescriptionRelationEvidence,
        )
        from app.src.agent.components.diagnose.structured_diagnosis import (
            apply_clinical_safety_bounds,
            generate_structured_diagnosis,
        )
        from .moderate_diagnosis_map_reduce import (
            _apply_real_case_boundary,
            _attach_graph_identity,
            _build_retrieval_citations,
            _ground_prescriptions_to_syndrome,
            _query_prescriptions_for_syndrome,
        )

        diagnosis_result = await generate_structured_diagnosis(
            llm,
            [
                SystemMessage(content=prompt),
                HumanMessage(content="请结合参考资料开始您的辨证分析。"),
            ],
        )
        _attach_graph_identity(diagnosis_result, graph_rag_result)
        related_prescriptions = await _query_prescriptions_for_syndrome(
            diagnosis_result.syndrome,
            diagnosis_result.syndrome_id,
        )
        _ground_prescriptions_to_syndrome(
            diagnosis_result,
            related_prescriptions,
            DiagnosisPrescription,
            PrescriptionRelationEvidence,
        )
        diagnosis_result.citations = _build_retrieval_citations(
            similar_syndromes,
            [],
            related_prescriptions,
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

        logger.info(f"降级并行辨证完成")

        return {
            "answer": answer,
            "diagnosis_result": diagnosis_result.model_dump(),
            "steps": [f"中等辨证: 完成 (降级并行, 耗时 {elapsed:.2f}秒)"],
        }

    except Exception as e:
        logger.error(f"中等辨证失败: {e}", exc_info=True)
        # 尝试降级
        try:
            from app.src.agent.components.diagnose.nodes.simple.simple_diagnosis import simple_diagnosis
            return await simple_diagnosis(state)
        except:
            return {
                "answer": f"抱歉，中等辨证过程中出现错误：{str(e)}。建议您前往医院进行详细检查。",
                "steps": [f"中等辨证: 失败 - {str(e)}"],
            }

async def _query_diagnostic_graph(collected_info: CollectedDiagnoseInfo):
    """执行可审计的 Neo4j GraphRAG 诊断子图检索。"""
    from app.src.agent.retrieval.graphrag import retrieve_diagnostic_graph

    return await retrieve_diagnostic_graph(collected_info, top_k=5)


async def _query_similar_syndromes(collected_info: CollectedDiagnoseInfo) -> List[Dict[str, Any]]:
    """查询相似证型（med 真实诊断关系优先，多策略图谱补充）。

    旧实现：``(Symptom)-[:INDICATES]->(Syndrome)``，但 Neo4j 中
    实际无此关系 → 旧实现永远返回空。

    当前实现：
        - 首选：med TCM 的证候→主症/兼症/舌象/脉象真实关系加权匹配
        - 数据源：``db=neo4j``（SymMap_v2 7 类 + HPOA 3 类节点）
        - 策略 A：用户症状 → 关键词匹配 SymMap MMSymptom.name → 关联
                  HPOA Disease（按 hpo_id 桥接 273 节点）→ 找相关 SymMap Disease
        - 策略 B：用户症状 → 关键词 CONTAINS 匹配 SymMap Syndrome（name_zh/definition）
        - 策略 C：用户症状 → 关键词匹配 SymMap TCMSymptom.name_zh（中医症状）

    Neo4j 不可用时返回空列表，由上层 LLM 自行降级到"仅基于症状推理"。

    Returns:
        每条含 ``name`` / ``symptoms`` (list) / ``similarity`` (0-1) /
        ``source`` / ``match_count`` / ``source_db``。
        兼容 :func:`_format_syndromes` 的 ``name`` / ``symptoms`` / ``similarity`` 字段。
    """
    graph_rag_result = await _query_diagnostic_graph(collected_info)
    return graph_rag_result.to_legacy_candidates()

    # 旧多策略属性检索代码保留为历史参考，但不再进入活跃主路径。
    keywords = _diagnostic_keywords(collected_info)
    if not keywords:
        return []

    try:
        from app.src.core.graph_db import get_neo4j_graph
    except ImportError as exc:
        logger.warning("graph_db 模块不可用: %s", exc)
        return []

    graph = get_neo4j_graph(database="neo4j")
    if graph is None:
        logger.warning("Neo4j 不可用，跳过证型查询")
        return []

    candidates: List[Dict[str, Any]] = []
    seen: set[str] = set()

    # ===== 首选策略: med TCM 证候 → 症状真实关系 =====
    try:
        cypher_med = """
        UNWIND $keywords AS kw
        MATCH (sy:Syndrome {source_db: 'med_tcm'})
              -[r:SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM]->
              (ts:TCMSymptom {source_db: 'med_tcm'})
        WHERE coalesce(ts.is_informative, false) = true
          AND (
            toLower(coalesce(ts.normalized_name, ts.name_zh, '')) CONTAINS toLower(kw)
            OR toLower(kw) CONTAINS toLower(coalesce(ts.normalized_name, ts.name_zh, ''))
          )
        WITH sy, kw,
             max(coalesce(r.evidence_weight, 1.0)) AS keyword_weight,
             collect(DISTINCT {
               keyword: kw,
               symptom: ts.name_zh,
               role: coalesce(r.symptom_role, ts.symptom_kind, 'unknown'),
               weight: coalesce(r.evidence_weight, 1.0)
             }) AS keyword_evidence
        WITH sy,
             collect(DISTINCT kw) AS matched_keywords,
             sum(keyword_weight) AS weighted_score,
             reduce(evidence = [], items IN collect(keyword_evidence) |
               evidence + items
             ) AS symptom_evidence
        RETURN sy.name_zh AS name,
               sy.canonical_name AS canonical_name,
               matched_keywords,
               symptom_evidence,
               weighted_score,
               size(matched_keywords) AS match_count,
               [(sy)-[:SYNDROME_PATTERN_OF_TCM_DISEASE]
                 ->(d:TCMDisease {source_db: 'med_tcm'}) | d.name_zh]
                 AS related_tcm_diseases,
               [(sy)-[:SYNDROME_ASSOCIATED_WITH_CONSTITUTION]
                 ->(constitution:Constitution {source_db: 'med_tcm'}) | constitution.name_zh]
                 AS constitutions,
               [(sy)-[:SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM {symptom_role: 'main'}]
                 ->(main:TCMSymptom {source_db: 'med_tcm'})
                 WHERE coalesce(main.is_informative, false) = true | main.name_zh][0..5]
                 AS main_symptoms,
               [(sy)-[:SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM {symptom_role: 'supplement'}]
                 ->(supplement:TCMSymptom {source_db: 'med_tcm'})
                 WHERE coalesce(supplement.is_informative, false) = true | supplement.name_zh][0..5]
                 AS supplement_symptoms,
               [(sy)-[:SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM {symptom_role: 'tongue'}]
                 ->(tongue:TCMSymptom {source_db: 'med_tcm'})
                 WHERE coalesce(tongue.is_informative, false) = true | tongue.name_zh][0..5]
                 AS tongue_symptoms,
               [(sy)-[:SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM {symptom_role: 'pulse'}]
                 ->(pulse:TCMSymptom {source_db: 'med_tcm'})
                 WHERE coalesce(pulse.is_informative, false) = true | pulse.name_zh][0..5]
                 AS pulse_symptoms
        ORDER BY weighted_score DESC, match_count DESC, sy.name_zh ASC
        LIMIT $top_k
        """
        rows_med = graph.query(
            cypher_med,
            params={"keywords": keywords, "top_k": 8},
        )
        for row in rows_med:
            name = row.get("name")
            canonical_key = _canonical_candidate_key(
                name,
                row.get("canonical_name"),
            )
            if not name or not canonical_key or canonical_key in seen:
                continue
            seen.add(canonical_key)

            best_evidence_by_keyword: dict[str, dict[str, Any]] = {}
            for evidence in row.get("symptom_evidence") or []:
                if not isinstance(evidence, dict):
                    continue
                keyword = str(evidence.get("keyword") or "")
                weight = float(evidence.get("weight") or 0)
                current = best_evidence_by_keyword.get(keyword)
                if current is None or weight > float(current.get("weight") or 0):
                    best_evidence_by_keyword[keyword] = evidence
            symptom_evidence = list(best_evidence_by_keyword.values())
            matched_symptoms = list(dict.fromkeys(
                str(evidence.get("symptom"))
                for evidence in symptom_evidence
                if evidence.get("symptom")
            ))
            match_count = int(row.get("match_count") or 0)
            weighted_score = float(row.get("weighted_score") or 0)
            coverage = min(1.0, match_count / max(1, min(len(keywords), 8)))
            role_quality = min(1.0, weighted_score / max(1.0, match_count * 3.0))
            candidates.append({
                "name": name,
                "canonical_name": row.get("canonical_name"),
                "symptoms": matched_symptoms,
                "symptom_evidence": symptom_evidence,
                "similarity": min(1.0, coverage * 0.7 + role_quality * 0.3),
                "match_count": match_count,
                "weighted_score": weighted_score,
                "source": "med_tcm_diagnostic_axis",
                "source_db": "med_tcm",
                "related_tcm_diseases": row.get("related_tcm_diseases") or [],
                "constitutions": row.get("constitutions") or [],
                "diagnostic_axis": {
                    "main": row.get("main_symptoms") or [],
                    "supplement": row.get("supplement_symptoms") or [],
                    "tongue": row.get("tongue_symptoms") or [],
                    "pulse": row.get("pulse_symptoms") or [],
                },
            })
    except Exception as exc:
        logger.error("证型查询 med 诊断轴策略失败: %s", exc)

    # ===== 策略 A: MMSymptom → HPOA Disease 桥接 =====
    try:
        cypher_a = """
        UNWIND $keywords AS kw
        MATCH (m:MMSymptom)
        WHERE toLower(coalesce(m.name, '')) CONTAINS toLower(kw)
              AND coalesce(m.hpo_id, '') <> ''
        WITH m, collect(DISTINCT kw) AS matched_kws
        // 桥接到 HPOA MMSymptom（按 hpo_id 字符串精确匹配）
        OPTIONAL MATCH (hp:MMSymptom {source_db: 'HPO'})
              WHERE hp.hpo_id = m.hpo_id
        OPTIONAL MATCH (hp)-[r_dhs:DISEASE_HAS_MM_SYMPTOM]-(d:Disease)
              WHERE d.source_db = 'OMIM' OR d.source_db = 'ORPHA'
        WITH m, matched_kws, d,
             size(matched_kws) AS match_count
        WHERE d IS NOT NULL
        RETURN d.name        AS name,
               d.source_db   AS source_db,
               d.omim_id     AS omim_id,
               d.mesh_id     AS mesh_id,
               collect(DISTINCT m.name)[0..3] AS matched_symptoms,
               match_count
        ORDER BY match_count DESC
        LIMIT $top_k
        """
        rows_a = graph.query(cypher_a, params={"keywords": keywords, "top_k": 8})
        for row in rows_a:
            name = row.get("name")
            canonical_key = _canonical_candidate_key(name)
            if not name or not canonical_key or canonical_key in seen:
                continue
            seen.add(canonical_key)
            mc = row.get("match_count", 0) or 0
            candidates.append({
                "name": name,
                "symptoms": row.get("matched_symptoms") or [],
                "similarity": min(1.0, mc * 0.25),
                "match_count": mc,
                "source": "hpoa_disease_via_mmsymptom",
                "source_db": f"hpoa({row.get('source_db', '')})",
                "omim_id": row.get("omim_id"),
                "mesh_id": row.get("mesh_id"),
            })
    except Exception as exc:
        logger.error("证型查询策略 A 失败: %s", exc)

    # ===== 策略 B: SymMap Syndrome 直接匹配 =====
    try:
        cypher_b = """
        UNWIND $keywords AS kw
        MATCH (sy:Syndrome)
        WHERE coalesce(sy.source_db, '') <> 'med_tcm'
          AND (toLower(coalesce(sy.name_zh, '')) CONTAINS toLower(kw)
              OR toLower(coalesce(sy.definition, '')) CONTAINS toLower(kw))
        WITH sy, collect(DISTINCT kw) AS matched_kws,
             size(collect(DISTINCT kw)) AS match_count
        RETURN sy.name_zh    AS name,
               sy.name_en    AS name_en,
               sy.definition AS definition,
               matched_kws   AS matched_keywords,
               match_count,
               // 字段加权：definition 命中 +2, name_zh 命中 +1
               reduce(s = 0, kw IN matched_kws |
                 s + (CASE WHEN toLower(coalesce(sy.definition, '')) CONTAINS toLower(kw) THEN 2 ELSE 0 END)
                   + (CASE WHEN toLower(coalesce(sy.name_zh, '')) CONTAINS toLower(kw) THEN 1 ELSE 0 END)
               ) AS score
        ORDER BY score DESC, sy.name_zh ASC
        LIMIT $top_k
        """
        rows_b = graph.query(cypher_b, params={"keywords": keywords, "top_k": 6})
        for row in rows_b:
            name = row.get("name")
            canonical_key = _canonical_candidate_key(name)
            if not name or not canonical_key or canonical_key in seen:
                continue
            seen.add(canonical_key)
            mc = row.get("match_count", 0) or 0
            score = row.get("score", 0) or 0
            candidates.append({
                "name": name,
                "symptoms": list(row.get("matched_keywords") or []),
                "similarity": min(1.0, score * 0.2),
                "match_count": mc,
                "source": "symmap_syndrome_direct",
                "source_db": "symmap",
                "name_en": row.get("name_en"),
                "definition": row.get("definition"),
            })
    except Exception as exc:
        logger.error("证型查询策略 B 失败: %s", exc)

    # ===== 策略 C: TCMSymptom 关键词匹配 =====
    try:
        cypher_c = """
        UNWIND $keywords AS kw
        MATCH (ts:TCMSymptom)
        WHERE coalesce(ts.source_db, '') <> 'med_tcm'
          AND toLower(coalesce(ts.name_zh, '')) CONTAINS toLower(kw)
        WITH ts, collect(DISTINCT kw) AS matched_kws,
             size(collect(DISTINCT kw)) AS match_count
        RETURN ts.name_zh    AS name,
               ts.definition AS definition,
               ts.locus      AS locus,
               matched_kws   AS matched_keywords,
               match_count
        ORDER BY match_count DESC
        LIMIT $top_k
        """
        rows_c = graph.query(cypher_c, params={"keywords": keywords, "top_k": 5})
        for row in rows_c:
            name = row.get("name")
            canonical_key = _canonical_candidate_key(name)
            if not name or not canonical_key or canonical_key in seen:
                continue
            seen.add(canonical_key)
            mc = row.get("match_count", 0) or 0
            candidates.append({
                "name": name,
                "symptoms": list(row.get("matched_keywords") or []),
                "similarity": min(1.0, mc * 0.3),
                "match_count": mc,
                "source": "symmap_tcm_symptom",
                "source_db": "symmap",
                "definition": row.get("definition"),
                "locus": row.get("locus"),
            })
    except Exception as exc:
        logger.error("证型查询策略 C 失败: %s", exc)

    if candidates:
        logger.info(
            "证型查询命中 %d 条候选 (keywords=%d, MED=%d A=%d B=%d C=%d)",
            len(candidates), len(keywords),
            sum(1 for c in candidates if c["source"] == "med_tcm_diagnostic_axis"),
            sum(1 for c in candidates if c["source"] == "hpoa_disease_via_mmsymptom"),
            sum(1 for c in candidates if c["source"] == "symmap_syndrome_direct"),
            sum(1 for c in candidates if c["source"] == "symmap_tcm_symptom"),
        )

    # 真实证候-症状关系优先，其余候选按相似度补充，最多 5 条。
    candidates.sort(
        key=lambda c: (
            c.get("source") == "med_tcm_diagnostic_axis",
            c.get("similarity", 0),
            c.get("match_count", 0),
        ),
        reverse=True,
    )
    return candidates[:5]


async def _query_similar_cases(collected_info: CollectedDiagnoseInfo) -> List[Dict[str, Any]]:
    """查询证型-方剂治疗模式（不是患者医案）。

    P3 Task 1 改造（2026-06-09）：
    真实图谱中**没有患者医案节点**，当前 Qdrant 也没有真实病例集合。
    因此这里只返回“证型/疾病 + 方剂”知识图谱治疗模式。调用方必须明确标为
    ``treatment_pattern``，不得提供病例编号、疗效或患者级相似度。

    设计选择（与 :func:`_query_similar_syndromes` 一致的模式）：
    **关键词直接匹配节点属性**。P1 阶段只灌了节点没灌关系，所以多跳
    关系查询不可用；改用"在 Python 端做两段匹配"：
        段 1：用户症状关键词 → 命中节点（Syndrome / Disease）
        段 2：相同关键词 → 命中节点（Formula.effect / Formula.indications）
        关联：按相同关键词关联两者，组装"案例"。

    Neo4j 不可用时返回空列表（上层 LLM 自行降级到"仅基于症状推理"）。

    Returns:
        每条含 ``chief_complaint`` / ``syndrome`` / ``treatment`` /
        ``matched_symptoms`` / ``match_score`` / ``source`` 字段，
        兼容 :func:`_format_cases` 的字段约定。
    """
    # 1. 收集并切分症状关键词
    symptoms_raw: List[str] = []
    if hasattr(collected_info, "get_all_symptoms"):
        symptoms_raw = collected_info.get_all_symptoms() or []
    if not symptoms_raw:
        return []

    keywords: List[str] = []
    for s in symptoms_raw:
        if not s:
            continue
        for token in re.split(r"[，。；,;\s、]+", str(s)):
            token = token.strip()
            if len(token) >= 2:
                keywords.append(token)
    if not keywords:
        return []
    keywords = list(dict.fromkeys(keywords))[:20]

    # 2. 连接 Neo4j
    try:
        from app.src.core.graph_db import get_neo4j_graph
    except ImportError as exc:
        logger.warning("graph_db 模块不可用: %s", exc)
        return []

    graph = get_neo4j_graph(database="neo4j")
    if graph is None:
        logger.warning("Neo4j 不可用，跳过治疗模式查询")
        return []

    candidates: List[Dict[str, Any]] = []
    seen: set = set()

    # ===== 段 1a: 命中医案"病证"（Syndrome / Disease）=====
    syndrome_records: List[Dict[str, Any]] = []
    disease_records: List[Dict[str, Any]] = []

    # 1a-i: SymMap Syndrome（与 _query_similar_syndromes 策略 B 相同）
    try:
        cypher_syn = """
        UNWIND $keywords AS kw
        MATCH (sy:Syndrome)
        WHERE toLower(coalesce(sy.name_zh, '')) CONTAINS toLower(kw)
              OR toLower(coalesce(sy.definition, '')) CONTAINS toLower(kw)
        WITH sy, collect(DISTINCT kw) AS matched_kws,
             size(collect(DISTINCT kw)) AS match_count
        RETURN sy.name_zh    AS name,
               sy.definition AS definition,
               matched_kws   AS matched_keywords,
               match_count
        ORDER BY match_count DESC
        LIMIT $top_k
        """
        rows = graph.query(cypher_syn, params={"keywords": keywords, "top_k": 5})
        for row in rows:
            syndrome_records.append({
                "name": row.get("name"),
                "definition": row.get("definition"),
                "matched_keywords": list(row.get("matched_keywords") or []),
                "match_count": row.get("match_count", 0) or 0,
                "kind": "syndrome",
            })
    except Exception as exc:
        logger.error("相似医案段 1a-i 失败: %s", exc)

    # 1a-ii: HPOA Disease via MMSymptom 桥接（与 _query_similar_syndromes 策略 A 相同）
    try:
        cypher_dis = """
        UNWIND $keywords AS kw
        MATCH (m:MMSymptom)
        WHERE toLower(coalesce(m.name, '')) CONTAINS toLower(kw)
              AND coalesce(m.hpo_id, '') <> ''
        WITH m, collect(DISTINCT kw) AS matched_kws
        OPTIONAL MATCH (hp:MMSymptom {source_db: 'HPO'})
              WHERE hp.hpo_id = m.hpo_id
        OPTIONAL MATCH (hp)-[r_dhs:DISEASE_HAS_MM_SYMPTOM]-(d:Disease)
              WHERE d.source_db = 'OMIM' OR d.source_db = 'ORPHA'
        WITH m, matched_kws, d,
             size(matched_kws) AS match_count
        WHERE d IS NOT NULL
        RETURN d.name        AS name,
               d.source_db   AS source_db,
               d.mesh_id     AS mesh_id,
               collect(DISTINCT m.name)[0..3] AS matched_keywords,
               match_count
        ORDER BY match_count DESC
        LIMIT $top_k
        """
        rows = graph.query(cypher_dis, params={"keywords": keywords, "top_k": 4})
        for row in rows:
            disease_records.append({
                "name": row.get("name"),
                "source_db": row.get("source_db"),
                "mesh_id": row.get("mesh_id"),
                "matched_keywords": list(row.get("matched_keywords") or []),
                "match_count": row.get("match_count", 0) or 0,
                "kind": "disease",
            })
    except Exception as exc:
        logger.error("相似医案段 1a-ii 失败: %s", exc)

    # ===== 段 1b: 命中医案"治疗方剂"（ITCM Formula）=====
    formula_records: List[Dict[str, Any]] = []
    try:
        cypher_f = """
        UNWIND $keywords AS kw
        MATCH (f:Formula)
        WHERE (toLower(coalesce(f.effect_zh, '')) CONTAINS toLower(kw)
            OR toLower(coalesce(f.indications_zh, '')) CONTAINS toLower(kw))
        WITH f, collect(DISTINCT kw) AS matched_kws,
             size(collect(DISTINCT kw)) AS match_count
        WITH f, matched_kws, match_count,
             reduce(s = 0, kw IN matched_kws |
               s + (CASE WHEN toLower(coalesce(f.effect_zh, '')) CONTAINS toLower(kw) THEN 3 ELSE 0 END)
                 + (CASE WHEN toLower(coalesce(f.indications_zh, '')) CONTAINS toLower(kw) THEN 1 ELSE 0 END)
             ) AS score
        RETURN f.name_zh       AS name,
               f.effect_zh     AS effect,
               f.indications_zh AS indications,
               f.source        AS source,
               score,
               matched_kws     AS matched_keywords
        ORDER BY score DESC
        LIMIT $top_k
        """
        rows = graph.query(cypher_f, params={"keywords": keywords, "top_k": 5})
        for row in rows:
            formula_records.append({
                "name": row.get("name"),
                "effect": row.get("effect"),
                "indications": row.get("indications"),
                "source": row.get("source"),
                "score": row.get("score", 0) or 0,
                "matched_keywords": list(row.get("matched_keywords") or []),
            })
    except Exception as exc:
        logger.error("相似医案段 1b 失败: %s", exc)

    # ===== 段 2: 关联组装"案例" =====
    # 对每个病证，找与其共享关键词最多的方剂；找不到则用最高分方剂兜底
    if not formula_records:
        logger.info("治疗模式：无方剂候选，仅返回病证摘要")
        # 仍可输出"病证型"作为参考
        for rec in syndrome_records[:2]:
            syndrome = rec["name"]
            if not syndrome or syndrome in seen:
                continue
            seen.add(syndrome)
            candidates.append({
                "chief_complaint": "、".join(rec["matched_keywords"])[:80],
                "syndrome": syndrome,
                "syndrome_definition": rec.get("definition"),
                "treatment": "（无对应方剂，请结合辨证施治）",
                "matched_symptoms": rec["matched_keywords"],
                "match_score": min(1.0, rec["match_count"] * 0.2),
                "source": "syndrome_only",
            })
        candidates.sort(key=lambda c: c.get("match_score", 0), reverse=True)
        return candidates[:5]

    # 病证 ↔ 方剂配对
    for rec in (syndrome_records + disease_records):
        rec_name = rec["name"]
        if not rec_name or rec_name in seen:
            continue
        seen.add(rec_name)

        # 找共享关键词最多的方剂
        shared = sorted(
            formula_records,
            key=lambda f: (
                -len(set(f["matched_keywords"]) & set(rec["matched_keywords"])),
                -f["score"],
            ),
        )
        best_formula = shared[0] if shared else None
        if best_formula is None:
            continue

        # 共享关键词
        overlap = list(set(best_formula["matched_keywords"]) & set(rec["matched_keywords"]))
        if not overlap:
            overlap = rec["matched_keywords"][:3]  # 兜底用病证自身的关键词

        if rec["kind"] == "disease":
            syndrome_label = f"现代医学参考：{rec_name}"
            source_tag = "hpoa_disease_to_itcm_formula"
            base_score = rec["match_count"] * 0.2
        else:
            syndrome_label = rec_name
            source_tag = "symmap_syndrome_to_itcm_formula"
            base_score = rec["match_count"] * 0.3

        # 治疗置信度：病证与方剂共享关键词越多越好
        overlap_score = len(overlap) * 0.15
        # 加方剂评分（限幅）
        formula_score = min(0.4, best_formula["score"] * 0.05)
        final_score = min(1.0, base_score + overlap_score + formula_score)

        candidates.append({
            "chief_complaint": "、".join(overlap)[:80],
            "syndrome": syndrome_label,
            "syndrome_definition": rec.get("definition"),
            "treatment": best_formula["name"],
            "treatment_effect": best_formula.get("effect"),
            "treatment_indications": best_formula.get("indications"),
            "treatment_source": best_formula.get("source"),
            "matched_symptoms": overlap,
            "match_score": final_score,
            "source": source_tag,
            **({"mesh_id": rec["mesh_id"]} if rec.get("mesh_id") else {}),
            **({"source_db": rec["source_db"]} if rec.get("source_db") else {}),
        })

    if candidates:
        logger.info(
            "证型-方剂治疗模式查询命中 %d 条 (keywords=%d, syndrome=%d, disease=%d, formula=%d)",
            len(candidates), len(keywords),
            len(syndrome_records), len(disease_records), len(formula_records),
        )

    candidates.sort(key=lambda c: c.get("match_score", 0), reverse=True)
    return candidates[:5]


async def _query_related_prescriptions(collected_info: CollectedDiagnoseInfo) -> List[Dict[str, Any]]:
    """查询常用方剂（Neo4j 知识图谱）。

    主数据源：ITCM Formula（25,857 条，存于 ``neo4j`` database）
        - 按症状关键词匹配 ``effect_zh``（功效）和 ``indications_zh``（主治）
        - 评分：effect_zh 命中 +3 / indications_zh 命中 +1
        - 去重后取 Top-K

    兜底数据源：tcm_graph 库的 ``(Prescription)-[:TREATS]->(Syndrome)``
        - 旧系统残留的小型方剂库（P001-P100 / S001-S291）
        - 仅在主数据源无结果时启用

    Neo4j 不可用时返回空列表，由上层 LLM 自行降级。
    """
    try:
        from app.src.core.graph_db import get_neo4j_graph
    except ImportError as exc:
        logger.warning("graph_db 模块不可用: %s", exc)
        return []

    # 收集症状关键词（去空、去短词）
    symptoms_raw: List[str] = []
    if hasattr(collected_info, "get_all_symptoms"):
        symptoms_raw = collected_info.get_all_symptoms() or []
    if not symptoms_raw:
        return []

    # 中文症状切分：按逗号、句号、分号、顿号拆词
    keywords: List[str] = []
    for s in symptoms_raw:
        if not s:
            continue
        for token in re.split(r"[，。；,;\s、]+", str(s)):
            token = token.strip()
            keywords.extend(_expand_retrieval_token(token))
    if not keywords:
        return []

    # 限制关键词数量，避免 Cypher 参数过大
    keywords = list(dict.fromkeys(keywords))[:20]

    # ===== 主数据源：ITCM Formula (neo4j db) =====
    graph = get_neo4j_graph(database="neo4j")
    if graph is not None and keywords:
        try:
            itcm_results = _query_itcm_formulas(graph, keywords, top_k=8)
            if itcm_results:
                logger.info(
                    "ITCM Formula 查询命中 %d 条 (keywords=%d)",
                    len(itcm_results), len(keywords),
                )
                return itcm_results
        except Exception as exc:
            logger.error("ITCM Formula 查询失败: %s", exc)

    # ===== 兜底：tcm_graph 库旧 Prescription =====
    graph_old = get_neo4j_graph(database="tcm_graph")
    if graph_old is None:
        return []
    try:
        cypher = """
        MATCH (p:Prescription)
        RETURN p.name AS name, p.source AS source, p.id AS id
        LIMIT $top_k
        """
        rows = graph_old.query(cypher, params={"top_k": 5})
        return [
            {
                "name": row.get("name"),
                "source": row.get("source"),
                "id": row.get("id"),
                "match_score": 0,
                "matched_keywords": [],
                "source_db": "tcm_graph_fallback",
            }
            for row in rows
        ]
    except Exception as exc:
        logger.error("tcm_graph fallback 方剂查询失败: %s", exc)
        return []


async def _query_prescriptions_for_syndrome(
    syndrome: str,
    syndrome_id: str | None = None,
    *,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """只查询与最终主证存在显式 Neo4j 关系的方剂。

    症状、功效或主治文本相似不能替代证方关系。正式关系协议统一为
    ``(Syndrome)-[:TREATS_WITH]->(Formula)``；没有关系时必须返回空列表。
    """
    canonical_name = _canonical_candidate_key(syndrome)
    if not canonical_name:
        return []

    try:
        from app.src.core.graph_db import get_neo4j_graph
    except ImportError as exc:
        logger.warning("graph_db 模块不可用: %s", exc)
        return []

    graph = get_neo4j_graph(database="neo4j")
    if graph is None:
        return []

    try:
        relationship_types = graph.query(
            """
            CALL db.relationshipTypes() YIELD relationshipType
            WHERE relationshipType = $relationship_type
            RETURN count(*) AS count
            """,
            params={"relationship_type": "TREATS_WITH"},
        )
    except Exception as exc:
        logger.error("证方关系元数据查询失败: %s", exc)
        return []
    if not relationship_types or int(relationship_types[0].get("count") or 0) == 0:
        return []

    syndrome_names = list(dict.fromkeys([
        str(syndrome or "").strip(),
        canonical_name,
        f"{canonical_name}证",
    ]))
    cypher = """
    MATCH (sy:Syndrome)-[rel:TREATS_WITH]->(formula:Formula)
    WHERE (
        ($syndrome_id <> '' AND toString(coalesce(sy.id, '')) = $syndrome_id)
        OR toLower(coalesce(sy.canonical_name, '')) = toLower($canonical_name)
        OR coalesce(sy.name_zh, '') IN $syndrome_names
      )
    WITH sy, rel, formula,
         coalesce(sy.name_zh, '') AS syndrome_name,
         coalesce(formula.name_zh, '') AS formula_name
    WHERE formula_name <> ''
    RETURN toString(coalesce(sy.id, elementId(sy))) AS syndrome_id,
           syndrome_name,
           toString(coalesce(formula.id, elementId(formula))) AS formula_id,
           formula_name AS name,
           type(rel) AS relationship_type,
           toString(coalesce(rel.id, rel.med_relationship_id, elementId(rel))) AS relationship_id,
           coalesce(rel.source_db, formula.source_db, sy.source_db, 'neo4j') AS source_db,
           coalesce(formula.source, formula.reference, '') AS source,
           coalesce(formula.effect_zh, '') AS effects,
           coalesce(formula.indications_zh, '') AS indications
    ORDER BY formula_name ASC
    LIMIT $top_k
    """
    try:
        rows = graph.query(
            cypher,
            params={
                "syndrome_id": str(syndrome_id or ""),
                "canonical_name": canonical_name,
                "syndrome_names": syndrome_names,
                "top_k": top_k,
            },
        )
    except Exception as exc:
        logger.error("最终证型方剂关系查询失败: %s", exc)
        return []

    results: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        name = str(row.get("name") or "").strip()
        relationship_id = str(row.get("relationship_id") or "").strip()
        relationship_type = str(row.get("relationship_type") or "").strip()
        if not name or not relationship_id or relationship_type != "TREATS_WITH":
            continue
        formula_key = _canonical_candidate_key(name)
        if not formula_key or formula_key in seen:
            continue
        seen.add(formula_key)
        source_db = str(row.get("source_db") or "neo4j")
        resolved_syndrome_id = str(row.get("syndrome_id") or syndrome_id or "")
        resolved_formula_id = str(row.get("formula_id") or "")
        resolved_syndrome_name = str(row.get("syndrome_name") or syndrome)
        results.append({
            "name": name,
            "source": row.get("source"),
            "effects": row.get("effects"),
            "indications": row.get("indications"),
            "source_db": source_db,
            "syndrome_id": resolved_syndrome_id or None,
            "syndrome_name": resolved_syndrome_name,
            "formula_id": resolved_formula_id or None,
            "relationship_type": relationship_type,
            "relationship_id": relationship_id,
            "relationship_path": [
                f"Syndrome[{resolved_syndrome_id or resolved_syndrome_name}]",
                f"-[:{relationship_type} {{{relationship_id}}}]-",
                f"Formula[{resolved_formula_id or name}]",
            ],
        })
    return results


def _query_itcm_formulas(
    graph, keywords: List[str], top_k: int = 8,
) -> List[Dict[str, Any]]:
    """在 ITCM Formula 节点上按关键词匹配 effect_zh / indications_zh。

    策略：用 ``toLower()`` + ``CONTAINS`` 多关键词联合查询；
    Cypher 端计算 score，Python 端只负责排序与裁剪。

    Returns:
        按 match_score 倒序的方剂列表，每条含
        ``name`` / ``effect_zh`` / ``indications_zh`` / ``source`` /
        ``match_score`` / ``matched_keywords`` / ``source_db``
    """
    cypher = """
    UNWIND $keywords AS kw
    MATCH (f:Formula)
    WHERE (toLower(coalesce(f.effect_zh, '')) CONTAINS toLower(kw)
        OR toLower(coalesce(f.indications_zh, '')) CONTAINS toLower(kw))
    WITH f, collect(DISTINCT kw) AS matched_kws,
         size(collect(DISTINCT kw)) AS match_count
    // 评分：功效字段命中 +3，主治字段命中 +1
    WITH f, matched_kws, match_count,
         reduce(
           s = 0,
           kw IN matched_kws |
             s + (CASE WHEN toLower(coalesce(f.effect_zh, '')) CONTAINS toLower(kw) THEN 3 ELSE 0 END)
               + (CASE WHEN toLower(coalesce(f.indications_zh, '')) CONTAINS toLower(kw) THEN 1 ELSE 0 END)
         ) AS score
    RETURN f.name_zh      AS name,
           f.effect_zh    AS effect_zh,
           f.indications_zh AS indications_zh,
           f.source       AS source,
           score,
           matched_kws    AS matched_keywords
    ORDER BY score DESC, f.name_zh ASC
    LIMIT $top_k
    """
    rows = graph.query(cypher, params={"keywords": keywords, "top_k": top_k})
    results: List[Dict[str, Any]] = []
    for row in rows:
        results.append({
            "name": row.get("name"),
            "effect_zh": row.get("effect_zh"),
            "indications_zh": row.get("indications_zh"),
            "source": row.get("source"),
            "match_score": row.get("score", 0),
            "matched_keywords": row.get("matched_keywords", []),
            "source_db": "itcm",
        })
    return results


def _format_syndromes(syndromes: List[Dict[str, Any]]) -> str:
    """格式化证型列表。

    兼容两种 schema：
    - 旧：``name`` / ``symptoms`` (list) / ``similarity``
    - 新：增加 ``source`` / ``source_db`` / ``definition`` 字段
    """
    if not syndromes:
        return "暂无相似证型"

    parts = []
    for i, syndrome in enumerate(syndromes[:3], 1):  # 最多3个
        symptoms_str = "、".join(syndrome.get("symptoms", []))
        similarity = syndrome.get("similarity", 0)
        source = syndrome.get("source", "")
        source_tag = f" [{source}]" if source else ""
        definition = syndrome.get("definition")
        parts.append(
            f"{i}. {syndrome['name']} (相似度: {similarity:.0%}){source_tag}\n"
            f"   匹配症状：{symptoms_str or '无'}"
        )
        if definition:
            parts.append(f"   定义：{definition[:80]}{'...' if len(str(definition)) > 80 else ''}")
        diagnostic_axis = syndrome.get("diagnostic_axis") or {}
        axis_labels = {
            "main": "主症",
            "supplement": "兼症",
            "tongue": "舌象",
            "pulse": "脉象",
        }
        for role, label in axis_labels.items():
            values = diagnostic_axis.get(role) or []
            if values:
                parts.append(f"   {label}：{'、'.join(str(value) for value in values[:5])}")
        constitutions = syndrome.get("constitutions") or []
        if constitutions:
            parts.append(f"   相关体质：{'、'.join(str(value) for value in constitutions[:3])}")
        diseases = syndrome.get("related_tcm_diseases") or []
        if diseases:
            parts.append(f"   相关中医病种：{'、'.join(str(value) for value in diseases[:3])}")

    return "\n".join(parts)


def _format_cases(cases: List[Dict[str, Any]]) -> str:
    """格式化证型-方剂治疗模式列表；不得描述为真实患者医案。

    兼容两种数据形态：
    - 旧 mock：``chief_complaint`` / ``syndrome`` / ``treatment`` / ``similarity``
    - 新 RAG（P3 Task 1）：增加 ``treatment_effect`` / ``treatment_indications`` /
      ``match_score`` / ``source`` 字段
    """
    if not cases:
        return "当前没有真实患者医案结果，也没有可用的证型-方剂治疗模式。"

    parts = [
        "以下内容是知识图谱中的证型-方剂治疗模式，不是患者病例，"
        "不包含病例编号、原始病历或疗效随访："
    ]
    for i, case in enumerate(cases[:2], 1):  # 最多2个
        score = case.get("match_score", case.get("similarity", 0))
        score_tag = f" (匹配度: {score:.0%})" if score else ""
        syndrome = case.get("syndrome", "未知")
        syndrome_def = case.get("syndrome_definition")
        syndrome_line = f"   证型：{syndrome}{score_tag}"
        if syndrome_def and len(str(syndrome_def)) > 0:
            truncated = str(syndrome_def)[:60]
            syndrome_line += f"\n   定义：{truncated}{'...' if len(str(syndrome_def)) > 60 else ''}"

        effect = case.get("treatment_effect")
        indications = case.get("treatment_indications")
        extra_lines = []
        if effect:
            extra_lines.append(f"   功效：{str(effect)[:80]}")
        if indications:
            extra_lines.append(f"   主治：{str(indications)[:80]}")

        parts.append(
            f"{i}. 匹配症状：{case.get('chief_complaint', '未知')}\n"
            f"{syndrome_line}\n"
            f"   关联方剂：{case.get('treatment', '未知')}"
            + ("\n" + "\n".join(extra_lines) if extra_lines else "")
        )

    return "\n".join(parts)


def _format_prescriptions(prescriptions: List[Dict[str, Any]]) -> str:
    """格式化方剂列表。

    兼容两种 schema：
    - 旧 tcm_graph Prescription：``name`` / ``composition`` (list) / ``indication``
    - 新 ITCM Formula：``name`` / ``effect_zh`` / ``indications_zh`` / ``match_score``
    """
    if not prescriptions:
        return "暂无相关方剂"

    parts = []
    for i, p in enumerate(prescriptions[:3], 1):  # 最多3个
        # 兼容：功效（effect_zh）或 主治（indications_zh 或 indication）
        indication = p.get("indications_zh") or p.get("indication") or "未知"
        effect = p.get("effect_zh")
        # 兼容：组成（composition list）— ITCM 没有则省略
        composition = p.get("composition") or []
        comp_str = "、".join(composition[:5]) + "等" if composition else "（ITCM 暂无组成数据）"
        score = p.get("match_score")
        score_tag = f" (匹配分: {score})" if score else ""
        source = p.get("source") or p.get("source_db") or ""
        source_tag = f" [{source}]" if source else ""
        parts.append(
            f"{i}. {p['name']}{source_tag}{score_tag}\n"
            f"   主治：{indication}\n"
            f"   功效：{effect or '未知'}\n"
            f"   组成：{comp_str}"
        )

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
