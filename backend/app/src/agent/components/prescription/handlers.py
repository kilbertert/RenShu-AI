"""方剂咨询处理器：从 Neo4j Formula 主数据实时检索。"""

import asyncio
from typing import Any
from ...tcm_states import PrescriptionInfo, TCMAgentState
from app.src.utils import get_logger

logger = get_logger("prescription_handler")


PRESCRIPTION_QUERY = """
UNWIND $terms AS term
MATCH (f:Formula)
WITH f, term, f.name_zh AS formula_name
WHERE formula_name IS NOT NULL AND trim(formula_name) <> ''
  AND (
    toLower(term) CONTAINS toLower(formula_name)
    OR toLower(formula_name) CONTAINS toLower(term)
    OR toLower(coalesce(f.effect_zh, '')) CONTAINS toLower(term)
    OR toLower(coalesce(f.indications_zh, '')) CONTAINS toLower(term)
  )
WITH f, formula_name,
     sum(CASE
       WHEN toLower(formula_name) = toLower(term) THEN 5
       WHEN toLower(term) CONTAINS toLower(formula_name) THEN 3
       WHEN toLower(formula_name) CONTAINS toLower(term) THEN 2
       ELSE 1
     END) AS match_score
OPTIONAL MATCH (f)-[r:FORMULA_CONTAINS_HERB]->(h:Herb)
WITH f, formula_name, match_score,
     [item IN collect(DISTINCT {
        herb: head([v IN [h.name_zh, h.chinese_name]
                    WHERE v IS NOT NULL AND trim(toString(v)) <> '']),
        dosage: ''
     }) WHERE item.herb <> ''] AS composition
RETURN formula_name AS name,
       head([v IN [f.source, f.reference]
             WHERE v IS NOT NULL AND trim(toString(v)) <> '']) AS source,
       composition,
       coalesce(f.effect_zh, '') AS effects,
       coalesce(f.indications_zh, '') AS indications,
       '' AS syndrome,
       head([v IN [f.administration, f.procedure]
             WHERE v IS NOT NULL AND trim(toString(v)) <> '']) AS usage,
       '' AS cautions,
       match_score
ORDER BY match_score DESC, name ASC
LIMIT $limit
"""


def _query_text(state: TCMAgentState) -> str:
    return str(state.messages[-1].content) if state.messages else ""


def _terms(state: TCMAgentState, query: str) -> list[str]:
    entities = state.router.extracted_entities if state.router else {}
    values: list[Any] = []
    if isinstance(entities, dict):
        for key in ("prescriptions", "syndromes", "symptoms"):
            values.extend(entities.get(key, []) or [])
    terms = [str(item).strip() for item in values if str(item).strip()]
    if query.strip():
        terms.append(query.strip())
    return list(dict.fromkeys(terms))[:15]


def _as_warnings(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


async def _search_prescriptions(
    terms: list[str],
    *,
    query: str = "",
) -> tuple[list[PrescriptionInfo], list[str]]:
    if not terms:
        return [], []
    try:
        from app.src.core.graph_db import get_neo4j_graph

        graph = get_neo4j_graph(database="neo4j")
        if graph is None:
            return [], []
        rows = await asyncio.to_thread(
            graph.query,
            PRESCRIPTION_QUERY,
            params={"terms": terms, "limit": 8},
        )
    except Exception as exc:
        logger.warning("方剂知识图谱查询失败: %s", exc)
        return [], []

    if query:
        explicit_rows = [row for row in rows if str(row.get("name") or "") in query]
        if explicit_rows:
            # 用户明确点名方剂时只返回完整名称命中的实体，避免同名变方/汉方混入。
            rows = explicit_rows

    prescriptions: list[PrescriptionInfo] = []
    seen_names: set[str] = set()
    for row in rows:
        name = row.get("name") or ""
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        prescriptions.append(PrescriptionInfo(
            name=row.get("name") or "未知方剂",
            source=row.get("source") or "",
            composition=row.get("composition") or [],
            effects=row.get("effects") or "",
            indications=row.get("indications") or "",
            syndrome=row.get("syndrome") or "",
            usage=row.get("usage") or "",
            cautions=_as_warnings(row.get("cautions")),
        ))
    return prescriptions, [PRESCRIPTION_QUERY]


async def handle_prescription_query(state: TCMAgentState) -> dict:
    """处理方剂查询；患者可见内容严格由当前图谱字段生成。"""
    query = _query_text(state)
    terms = _terms(state, query)
    prescriptions, cypher_queries = await _search_prescriptions(terms, query=query)
    answer = _format_grounded_prescription_answer(prescriptions)

    return {
        "answer": answer,
        "prescriptions": prescriptions,
        "cypher_queries": cypher_queries,
        "steps": [f"方剂咨询: Neo4j 实时检索 {len(prescriptions)} 条", "方剂咨询回复生成完成"],
    }


def _format_grounded_prescription_answer(
    prescriptions: list[PrescriptionInfo],
) -> str:
    """不使用模型补全缺失组成、剂量或出处。"""
    if not prescriptions:
        return (
            "当前知识图谱未检索到匹配方剂，无法可靠提供古籍出处、组成或剂量。"
            "请核对方名；不要根据相似名称自行拼接处方。"
        )

    parts = [
        "以下内容仅来自当前 Neo4j 知识图谱；缺失字段会明确标注，"
        "不会根据常识自动补齐组成或剂量。"
    ]
    for item in prescriptions[:5]:
        composition = [
            str(entry.get("herb") or "").strip()
            for entry in item.composition
            if isinstance(entry, dict) and entry.get("herb")
        ]
        lines = [f"### {item.name}"]
        lines.append(f"- 出处/来源：{item.source or '未提供'}")
        lines.append(f"- 组成：{'、'.join(composition) if composition else '图谱未提供'}")
        lines.append(f"- 功效：{item.effects or '图谱未提供'}")
        lines.append(f"- 主治/适用信息：{item.indications or '图谱未提供'}")
        lines.append(f"- 适用证型：{item.syndrome or '图谱未提供'}")
        lines.append(f"- 用法：{item.usage or '图谱未提供；不提供个体化剂量'}")
        lines.append(
            f"- 注意事项：{'；'.join(item.cautions) if item.cautions else '图谱未提供'}"
        )
        parts.append("\n".join(lines))
    parts.append("方剂应用必须建立在明确辨证基础上，实际组成、剂量和加减请由中医师面诊确定。")
    return "\n\n".join(parts)
