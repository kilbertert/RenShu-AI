"""药材咨询处理器：真实 Neo4j 检索 + 安全配伍规则 + LLM 解释。"""

import asyncio
import re
from typing import Any

from ...tcm_states import HerbCompatibilityResult, HerbInfo, TCMAgentState
from app.src.utils import get_logger

logger = get_logger("herb_handler")


HERB_QUERY = """
UNWIND $terms AS term
MATCH (h:Herb)
WITH h, term,
     head([v IN [h.chinese_name, h.name_zh]
           WHERE v IS NOT NULL AND trim(toString(v)) <> '']) AS herb_name
WHERE herb_name IS NOT NULL
  AND (
    toLower(term) CONTAINS toLower(herb_name)
    OR toLower(herb_name) CONTAINS toLower(term)
    OR toLower(coalesce(h.function, '')) CONTAINS toLower(term)
    OR toLower(coalesce(h.indication_zh, '')) CONTAINS toLower(term)
  )
WITH h, herb_name,
     sum(CASE
       WHEN toLower(herb_name) = toLower(term) THEN 5
       WHEN toLower(term) CONTAINS toLower(herb_name) THEN 3
       WHEN toLower(herb_name) CONTAINS toLower(term) THEN 2
       ELSE 1
     END) AS match_score
RETURN herb_name AS name,
       head([v IN [h.pinyin_name, h.pinyin]
             WHERE v IS NOT NULL AND trim(toString(v)) <> '']) AS pinyin,
       head([v IN [h.class_zh, h.therapeutic_class_zh]
             WHERE v IS NOT NULL AND trim(toString(v)) <> '']) AS category,
       head([v IN [h.properties_zh, h.property_zh, h.properties]
             WHERE v IS NOT NULL AND trim(toString(v)) <> '']) AS nature,
       coalesce(h.flavor_zh, '') AS flavor,
       head([v IN [h.meridians_zh, h.meridian_zh, h.meridians]
             WHERE v IS NOT NULL AND trim(toString(v)) <> '']) AS meridians,
       coalesce(h.function, '') AS effects,
       coalesce(h.indication_zh, '') AS indications,
       '' AS contraindications,
       '' AS dosage,
       coalesce(h.source_db, h.source, '') AS source_db,
       match_score
ORDER BY match_score DESC, name ASC
LIMIT $limit
"""


_INCOMPATIBLE_GROUPS = [
    ({"甘草"}, {"甘遂", "大戟", "海藻", "芫花"}, "十八反"),
    ({"川乌", "草乌", "乌头"}, {"贝母", "瓜蒌", "半夏", "白蔹", "白及"}, "十八反"),
    ({"藜芦"}, {"人参", "沙参", "丹参", "玄参", "细辛", "芍药"}, "十八反"),
    ({"硫黄"}, {"朴硝"}, "十九畏"),
    ({"水银"}, {"砒霜"}, "十九畏"),
    ({"狼毒"}, {"密陀僧"}, "十九畏"),
    ({"巴豆"}, {"牵牛"}, "十九畏"),
    ({"丁香"}, {"郁金"}, "十九畏"),
    ({"川乌", "草乌"}, {"犀角"}, "十九畏"),
    ({"牙硝"}, {"三棱"}, "十九畏"),
    ({"官桂", "肉桂"}, {"赤石脂", "石脂"}, "十九畏"),
    ({"人参"}, {"五灵脂"}, "十九畏"),
]


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in re.split(r"[、，,；;|/]+", str(value)) if part.strip()]


_TCM_TERM_TRANSLATIONS = {
    "cold": "寒", "cool": "凉", "neutral": "平", "warm": "温", "hot": "热",
    "slightly warm": "微温", "slightly cold": "微寒",
    "sour": "酸", "bitter": "苦", "sweet": "甘", "pungent": "辛",
    "acrid": "辛", "salty": "咸", "bland": "淡",
    "heart": "心", "liver": "肝", "spleen": "脾", "lung": "肺",
    "kidney": "肾", "stomach": "胃", "large intestine": "大肠",
    "small intestine": "小肠", "gallbladder": "胆", "bladder": "膀胱",
    "pericardium": "心包", "triple energizer": "三焦",
}


def _display_tcm_terms(values: list[str]) -> list[str]:
    """把可确定映射的性味归经英文术语规范为中文并去重。"""
    result: list[str] = []
    for raw in values:
        for value in _as_list(raw):
            mapped = _TCM_TERM_TRANSLATIONS.get(value.strip().lower(), value.strip())
            if mapped and mapped not in result:
                result.append(mapped)
    return result


def _contains_chinese(value: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", value))


def _display_grounded_text(values: list[str], *, field: str) -> list[str]:
    """优先展示图谱已有中文；仅翻译少量可确定的常见功效短语。"""
    normalized = [str(value).strip() for value in values if str(value).strip()]
    chinese = [value for value in normalized if _contains_chinese(value)]
    if chinese:
        return list(dict.fromkeys(chinese))
    if field == "effects":
        translated: list[str] = []
        for value in normalized:
            lowered = value.lower()
            if (
                ("reinoforce qi" in lowered or "reinforce qi" in lowered or "tonify qi" in lowered)
                and "spleen" in lowered
            ):
                translated.append("补气健脾")
        if translated:
            return list(dict.fromkeys(translated))
    return normalized


def _query_text(state: TCMAgentState) -> str:
    return str(state.messages[-1].content) if state.messages else ""


def _herb_terms(state: TCMAgentState, query: str) -> list[str]:
    entities = state.router.extracted_entities if state.router else {}
    herbs = entities.get("herbs", []) if isinstance(entities, dict) else []
    terms = [str(item).strip() for item in herbs if str(item).strip()]
    known_names = set().union(
        *(left | right for left, right, _rule_name in _INCOMPATIBLE_GROUPS)
    )
    terms.extend(name for name in known_names if name in query)
    if query.strip():
        terms.append(query.strip())
    return list(dict.fromkeys(terms))[:10]


async def _search_herbs(
    terms: list[str],
    *,
    query: str = "",
) -> tuple[list[HerbInfo], list[str]]:
    if not terms:
        return [], []
    try:
        from app.src.core.graph_db import get_neo4j_graph

        graph = get_neo4j_graph(database="neo4j")
        if graph is None:
            return [], []
        rows = await asyncio.to_thread(
            graph.query,
            HERB_QUERY,
            params={"terms": terms, "limit": 8},
        )
    except Exception as exc:
        logger.warning("药材知识图谱查询失败: %s", exc)
        return [], []

    if query:
        explicit_rows = [row for row in rows if str(row.get("name") or "") in query]
        if explicit_rows:
            rows = explicit_rows

    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row.get("name") or ""
        if not name:
            continue
        item = grouped.setdefault(name, {
            "name": name,
            "pinyin": "",
            "category": "",
            "nature": "",
            "flavor": [],
            "meridians": [],
            "effects": [],
            "indications": [],
            "contraindications": [],
            "dosage": "",
            "sources": [],
        })
        for field in ("pinyin", "category", "nature", "dosage"):
            if not item[field] and row.get(field):
                item[field] = str(row[field])
        for field in ("flavor", "meridians", "effects", "indications", "contraindications"):
            for value in _as_list(row.get(field)):
                if value not in item[field]:
                    item[field].append(value)
        source = str(row.get("source_db") or "").strip()
        if source and source not in item["sources"]:
            item["sources"].append(source)

    herbs = [HerbInfo(**item) for item in grouped.values()]
    return herbs, [HERB_QUERY]


def _check_compatibility(names: list[str]) -> HerbCompatibilityResult | None:
    normalized = {name.strip() for name in names if name.strip()}
    if len(normalized) < 2:
        return None

    pairs: list[tuple[str, str]] = []
    warnings: list[str] = []
    for left_group, right_group, rule_name in _INCOMPATIBLE_GROUPS:
        left_hits = normalized.intersection(left_group)
        right_hits = normalized.intersection(right_group)
        for left in left_hits:
            for right in right_hits:
                pairs.append((left, right))
                warnings.append(f"{left}与{right}属于{rule_name}传统配伍禁忌")

    return HerbCompatibilityResult(
        is_compatible=not pairs,
        warnings=warnings,
        incompatible_pairs=pairs,
        suggestions=(
            ["不要自行合用，需由中医师结合剂量、炮制和病情复核"]
            if pairs
            else ["未命中十八反、十九畏规则仍不等于绝对安全，请结合过敏史、妊娠及基础疾病评估"]
        ),
    )


async def handle_herb_query(state: TCMAgentState) -> dict:
    """处理药材咨询；患者可见内容严格由图谱字段和确定性安全规则生成。"""
    query = _query_text(state)
    terms = _herb_terms(state, query)
    herbs, cypher_queries = await _search_herbs(terms, query=query)
    compatibility = _check_compatibility([*terms, *(herb.name for herb in herbs)])

    answer = _format_grounded_herb_answer(query, herbs, compatibility)

    return {
        "answer": answer,
        "herbs": herbs,
        "compatibility_check": compatibility,
        "cypher_queries": cypher_queries,
        "steps": [f"药材咨询: Neo4j 实时检索 {len(herbs)} 条", "药材咨询回复生成完成"],
    }


def _format_grounded_herb_answer(
    query: str,
    herbs: list[HerbInfo],
    compatibility: HerbCompatibilityResult | None,
) -> str:
    """只展示已核验字段，同时把数据边界翻译成患者可理解的表达。"""
    forbid_dosage = bool(
        any(
            marker in query
            for marker in (
                "不要猜测剂量", "不要给剂量", "不提供剂量", "不需要剂量",
                "不要剂量", "无需剂量", "不用剂量",
            )
        )
        or re.search(
            r"(?:不要|无需|不用|不需要|请勿|别)"
            r"(?:给|提供|说明|写|列出|涉及|提及|展示)?"
            r"(?:具体)?(?:用药)?剂量",
            query,
        )
    )
    if not herbs:
        parts = [
            "当前知识资料中没有查到对应药材，因此不能可靠提供性味、归经、功效、剂量或出处。"
        ]
    else:
        boundary = (
            "以下仅整理已核验的基础资料，不补充未核验的信息。"
            if forbid_dosage
            else "以下仅整理已核验的基础资料；未记录的剂量和出处不会自行补全。"
        )
        parts = [
            "下面是知识资料中可核验到的基础信息。\n" + boundary
        ]
        for herb in herbs[:3]:
            lines = [f"### {herb.name}"]
            nature_flavor = "、".join(_display_tcm_terms([herb.nature, *herb.flavor]))
            meridians = _display_tcm_terms(herb.meridians)
            effects = _display_grounded_text(herb.effects, field="effects")
            indications = _display_grounded_text(herb.indications, field="indications")
            lines.append(f"- 性味：{nature_flavor or '当前资料未提供'}")
            lines.append(f"- 归经：{'、'.join(meridians) if meridians else '当前资料未提供'}")
            lines.append(f"- 主要功效：{'；'.join(effects) if effects else '当前资料未提供'}")
            lines.append(f"- 常见应用：{'；'.join(indications) if indications else '当前资料未提供'}")
            lines.append(
                f"- 禁忌与注意：{'；'.join(herb.contraindications) if herb.contraindications else '当前资料未提供'}"
            )
            if not forbid_dosage:
                lines.append(f"- 用量：{herb.dosage or '当前资料未提供；不能据此形成个体化剂量建议'}")
            parts.append("\n".join(lines))

    if compatibility:
        if compatibility.warnings:
            parts.append("**配伍安全提示**：" + "；".join(compatibility.warnings))
        elif compatibility.suggestions:
            parts.append("**配伍边界**：" + "；".join(compatibility.suggestions))
    parts.append("涉及实际服用、妊娠、儿童、慢性病或合并用药时，请由中医师或药师复核。")
    return "\n\n".join(parts)
