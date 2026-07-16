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
        subgraph_input = {
            "collected_info": state.get("collected_info", {}),
            "tongue_analysis": state.get("tongue_analysis"),
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
        steps = result.get("steps", [])

        return {
            "answer": answer,
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
            collected_info = CollectedDiagnoseInfo(**collected_info_dict)
            collected_summary = collected_info.to_summary()
        else:
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

        # 获取用户画像
        user_profile = state.get("user_profile", {})
        user_profile_desc = _format_user_profile(user_profile)

        # === 并行 RAG 检索（使用 asyncio.gather）===
        logger.info("开始并行 RAG 检索（asyncio.gather）...")
        start_time = time.time()

        # ★★★ 关键：使用 asyncio.gather 并行执行 ★★★
        similar_syndromes, similar_cases, related_prescriptions = await asyncio.gather(
            _query_similar_syndromes(collected_info),
            _query_similar_cases(collected_info),
            _query_related_prescriptions(collected_info),
            return_exceptions=True  # 即使某个查询失败，其他查询继续
        )

        # 处理异常结果
        if isinstance(similar_syndromes, Exception):
            logger.error(f"证型查询失败: {similar_syndromes}")
            similar_syndromes = []
        if isinstance(similar_cases, Exception):
            logger.error(f"医案查询失败: {similar_cases}")
            similar_cases = []
        if isinstance(related_prescriptions, Exception):
            logger.error(f"方剂查询失败: {related_prescriptions}")
            related_prescriptions = []

        elapsed = time.time() - start_time
        logger.info(f"并行 RAG 检索完成 (耗时: {elapsed:.2f}秒): 证型 {len(similar_syndromes)} 个, 医案 {len(similar_cases)} 个, 方剂 {len(related_prescriptions)} 个")

        # === 构建提示词 ===
        solar_term = _get_current_solar_term()
        
        prompt = MODERATE_DIAGNOSIS_PROMPT.format(
            collected_info=collected_summary,
            tongue_analysis=tongue_desc,
            user_profile=user_profile_desc,
            syndrome_matches=_format_syndromes(similar_syndromes),
            similar_cases=_format_cases(similar_cases),
            related_prescriptions=_format_prescriptions(related_prescriptions),
            solar_term=solar_term,
        )

        # === 调用 LLM ===
        llm = get_llm(
            llm_config=state.get("llm_config"),
            temperature=diagnose_config.DIAGNOSIS_TEMPERATURE
        )

        # 直接调用返回非结构化文本
        response = await llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content="请结合参考资料开始您的辨证分析。")
        ])

        answer = response.content

        logger.info(f"降级并行辨证完成")

        return {
            "answer": answer,
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



async def _query_similar_syndromes(collected_info: CollectedDiagnoseInfo) -> List[Dict[str, Any]]:
    """查询相似证型（Neo4j 知识图谱，多策略 HPO 桥接）。

    旧实现：``(Symptom)-[:INDICATES]->(Syndrome)``，但 Neo4j 中
    实际无此关系 → 旧实现永远返回空。

    新实现（2026-06-09 P1-3 改造）：
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
    symptoms = collected_info.get_all_symptoms()
    if not symptoms:
        return []

    # 中文症状切分
    keywords: List[str] = []
    for s in symptoms:
        if not s:
            continue
        for token in re.split(r"[，。；,;\s、]+", str(s)):
            token = token.strip()
            if len(token) >= 2:
                keywords.append(token)
    if not keywords:
        return []
    keywords = list(dict.fromkeys(keywords))[:20]

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
    seen: set = set()

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
            if not name or name in seen:
                continue
            seen.add(name)
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
        WHERE toLower(coalesce(sy.name_zh, '')) CONTAINS toLower(kw)
              OR toLower(coalesce(sy.definition, '')) CONTAINS toLower(kw)
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
            if not name or name in seen:
                continue
            seen.add(name)
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
        WHERE toLower(coalesce(ts.name_zh, '')) CONTAINS toLower(kw)
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
            if not name or name in seen:
                continue
            seen.add(name)
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
            "证型查询命中 %d 条候选 (keywords=%d, A=%d B=%d)",
            len(candidates), len(keywords),
            sum(1 for c in candidates if c["source"] == "hpoa_disease_via_mmsymptom"),
            sum(1 for c in candidates if c["source"] == "symmap_syndrome_direct"),
        )

    # 按 similarity 倒序，最多 5 条
    candidates.sort(key=lambda c: c.get("similarity", 0), reverse=True)
    return candidates[:5]


async def _query_similar_cases(collected_info: CollectedDiagnoseInfo) -> List[Dict[str, Any]]:
    """查询相似医案（Neo4j 关键词匹配 + Formula 关联 RAG）。

    P3 Task 1 改造（2026-06-09）：
    真实图谱中**没有"医案"节点**——传统医案是自然语言病例描述，
    不在 SymMap_v2 / ITCM / TCMBank 公开数据集覆盖范围。因此采用
    "证型/疾病 + 方剂"组合作为相似医案的代理：每个候选 = (证型, 治疗方剂)，
    这是中医辨证施治的标准范式。

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
        logger.warning("Neo4j 不可用，跳过相似医案查询")
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
        logger.info("相似医案：无方剂候选，仅返回病证摘要")
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
            "相似医案查询命中 %d 条 (keywords=%d, syndrome=%d, disease=%d, formula=%d)",
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
            # 过滤单字和空串（避免无意义匹配）
            if len(token) >= 2:
                keywords.append(token)
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

    return "\n".join(parts)


def _format_cases(cases: List[Dict[str, Any]]) -> str:
    """格式化相似医案列表。

    兼容两种数据形态：
    - 旧 mock：``chief_complaint`` / ``syndrome`` / ``treatment`` / ``similarity``
    - 新 RAG（P3 Task 1）：增加 ``treatment_effect`` / ``treatment_indications`` /
      ``match_score`` / ``source`` 字段
    """
    if not cases:
        return "暂无相似医案"

    parts = []
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
            f"{i}. 主诉：{case.get('chief_complaint', '未知')}\n"
            f"{syndrome_line}\n"
            f"   治疗：{case.get('treatment', '未知')}"
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
