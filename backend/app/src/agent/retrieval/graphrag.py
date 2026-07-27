"""基于 Neo4j 真实临床关系的可审计 GraphRAG 检索。"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.src.utils import get_logger


logger = get_logger("diagnostic_graphrag")

MED_SOURCE_DB = "med_tcm"
DIAGNOSTIC_RELATIONSHIP = "SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM"
ROLE_LABELS = {
    "main": "主症",
    "supplement": "兼症",
    "tongue": "舌象",
    "pulse": "脉象",
}


class GraphRAGEvidence(BaseModel):
    """一条可以回溯到 Neo4j 节点与关系的检索证据。"""

    evidence_id: str
    source_db: str
    source_archive_sha256: str | None = None
    source_tenant: str | None = None
    syndrome_name: str
    symptom_name: str
    matched_keywords: list[str] = Field(default_factory=list)
    symptom_role: Literal["main", "supplement", "tongue", "pulse", "unknown"]
    evidence_weight: float = Field(ge=0.0)
    score: float = Field(ge=0.0, le=1.0)
    node_ids: list[str] = Field(default_factory=list)
    relationship_ids: list[str] = Field(default_factory=list)
    relationship_path: list[str] = Field(default_factory=list)
    statement: str


class GraphRAGSyndromeCandidate(BaseModel):
    """按规范证候名聚合后的 GraphRAG 候选。"""

    syndrome_id: str | None = None
    syndrome_node_ids: list[str] = Field(default_factory=list)
    name: str
    canonical_name: str
    score: float = Field(ge=0.0, le=1.0)
    match_count: int = Field(ge=0)
    weighted_score: float = Field(ge=0.0)
    matched_keywords: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    related_tcm_diseases: list[str] = Field(default_factory=list)
    constitutions: list[str] = Field(default_factory=list)
    diagnostic_axis: dict[str, list[str]] = Field(default_factory=dict)
    source_db: str = MED_SOURCE_DB

    def to_legacy_dict(self) -> dict[str, Any]:
        """兼容 moderate 既有提示词和格式化函数。"""
        return {
            "id": self.syndrome_id,
            "name": self.name,
            "canonical_name": self.canonical_name,
            "symptoms": [
                symptom
                for role in ("main", "supplement", "tongue", "pulse")
                for symptom in self.diagnostic_axis.get(role, [])
            ][:12],
            "matched_keywords": self.matched_keywords,
            "similarity": self.score,
            "match_count": self.match_count,
            "weighted_score": self.weighted_score,
            "source": "med_tcm_diagnostic_axis",
            "retrieval_mode": "neo4j_graph_rag",
            "source_db": self.source_db,
            "related_tcm_diseases": self.related_tcm_diseases,
            "constitutions": self.constitutions,
            "diagnostic_axis": self.diagnostic_axis,
            "graph_evidence_ids": self.evidence_ids,
            "syndrome_node_ids": self.syndrome_node_ids,
        }


class GraphRAGResult(BaseModel):
    """一次 GraphRAG 检索的结构化、可审计返回。"""

    query: str
    keywords: list[str] = Field(default_factory=list)
    retrieval_mode: Literal["neo4j_graph", "unavailable"]
    graph_available: bool
    vector_index_used: bool = False
    candidates: list[GraphRAGSyndromeCandidate] = Field(default_factory=list)
    evidences: list[GraphRAGEvidence] = Field(default_factory=list)
    source_dbs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def to_legacy_candidates(self) -> list[dict[str, Any]]:
        return [candidate.to_legacy_dict() for candidate in self.candidates]


def _normalize_text(value: str) -> str:
    value = str(value or "").strip().lower()
    return re.sub(r"[\s，,。；;、（）()【】\[\]·—_\-]+", "", value)


def _canonical_syndrome(value: str) -> str:
    normalized = _normalize_text(value)
    if normalized.endswith("证") and len(normalized) > 1:
        return normalized[:-1]
    return normalized


def extract_graph_rag_keywords(collected_info: Any) -> list[str]:
    """从十问、阳性症状、文字/图片舌象和脉象中提取去重检索词。"""
    raw_values: list[str] = []
    if hasattr(collected_info, "get_all_symptoms"):
        raw_values.extend(collected_info.get_all_symptoms() or [])
        tongue = getattr(collected_info, "tongue", None) or {}
        pulse = getattr(collected_info, "pulse", None) or {}
    elif isinstance(collected_info, dict):
        raw_values.extend(
            str(value)
            for key, value in collected_info.items()
            if key not in {"tongue", "pulse"} and isinstance(value, str) and value
        )
        other = collected_info.get("other_symptoms") or []
        raw_values.extend(str(value) for value in other if value)
        tongue = collected_info.get("tongue") or {}
        pulse = collected_info.get("pulse") or {}
    elif isinstance(collected_info, str):
        raw_values.append(collected_info)
        tongue = {}
        pulse = {}
    else:
        raw_values.extend(str(value) for value in (collected_info or []) if value)
        tongue = {}
        pulse = {}

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
        raw_values.append(
            f"{prefix}{text}" if prefix and not text.startswith(prefix) else text
        )

    if isinstance(pulse, dict):
        pulse_text = pulse.get("description") or pulse.get("pulse")
    else:
        pulse_text = pulse
    if pulse_text:
        pulse_text = str(pulse_text).strip()
        raw_values.append(
            pulse_text if pulse_text.startswith("脉") else f"脉{pulse_text}"
        )

    keywords: list[str] = []
    for value in raw_values:
        for token in re.split(r"[，。；,;\s、]+", str(value)):
            normalized = _normalize_text(token)
            if len(normalized) >= 2:
                keywords.append(normalized)
    return list(dict.fromkeys(keywords))[:20]


DIAGNOSTIC_GRAPH_QUERY = """
UNWIND $keywords AS keyword
MATCH (syndrome:Syndrome {source_db: 'med_tcm'})
      -[relationship:SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM]->
      (symptom:TCMSymptom {source_db: 'med_tcm'})
WHERE coalesce(symptom.is_informative, false) = true
  AND (
    toLower(coalesce(symptom.normalized_name, symptom.name_zh, ''))
      CONTAINS toLower(keyword)
    OR toLower(keyword) CONTAINS
      toLower(coalesce(symptom.normalized_name, symptom.name_zh, ''))
  )
WITH syndrome, relationship, symptom,
     collect(DISTINCT keyword) AS matched_keywords
WITH syndrome,
     collect({
       syndrome_id: syndrome.id,
       symptom_id: symptom.id,
       relationship_id: relationship.med_relationship_id,
       symptom_name: symptom.name_zh,
       symptom_role: coalesce(relationship.symptom_role, symptom.symptom_kind, 'unknown'),
       evidence_weight: coalesce(relationship.evidence_weight, 1.0),
       matched_keywords: matched_keywords,
       source_archive_sha256: relationship.source_archive_sha256,
       source_tenant: relationship.source_tenant
     }) AS evidence_rows,
     sum(coalesce(relationship.evidence_weight, 1.0) * size(matched_keywords))
       AS raw_score
RETURN syndrome.id AS syndrome_id,
       syndrome.name_zh AS name,
       syndrome.canonical_name AS canonical_name,
       syndrome.source_archive_sha256 AS source_archive_sha256,
       syndrome.source_tenant AS source_tenant,
       evidence_rows,
       raw_score,
       [(syndrome)-[:SYNDROME_PATTERN_OF_TCM_DISEASE]
         ->(disease:TCMDisease {source_db: 'med_tcm'}) | disease.name_zh]
         AS related_tcm_diseases,
       [(syndrome)-[:SYNDROME_ASSOCIATED_WITH_CONSTITUTION]
         ->(constitution:Constitution {source_db: 'med_tcm'}) | constitution.name_zh]
         AS constitutions,
       [(syndrome)-[:SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM {symptom_role: 'main'}]
         ->(main:TCMSymptom {source_db: 'med_tcm'})
         WHERE coalesce(main.is_informative, false) = true | main.name_zh][0..8]
         AS main_symptoms,
       [(syndrome)-[:SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM {symptom_role: 'supplement'}]
         ->(supplement:TCMSymptom {source_db: 'med_tcm'})
         WHERE coalesce(supplement.is_informative, false) = true | supplement.name_zh][0..8]
         AS supplement_symptoms,
       [(syndrome)-[:SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM {symptom_role: 'tongue'}]
         ->(tongue:TCMSymptom {source_db: 'med_tcm'})
         WHERE coalesce(tongue.is_informative, false) = true | tongue.name_zh][0..8]
         AS tongue_symptoms,
       [(syndrome)-[:SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM {symptom_role: 'pulse'}]
         ->(pulse:TCMSymptom {source_db: 'med_tcm'})
         WHERE coalesce(pulse.is_informative, false) = true | pulse.name_zh][0..8]
         AS pulse_symptoms
ORDER BY raw_score DESC, syndrome.name_zh ASC
LIMIT $row_limit
"""


def _deduplicate(values: list[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _preferred_syndrome_name(names: list[str], canonical_name: str) -> str:
    preferred = f"{canonical_name}证"
    return preferred if preferred in names else names[0]


def _build_graph_rag_result(
    query: str,
    keywords: list[str],
    rows: list[dict[str, Any]],
    top_k: int,
) -> GraphRAGResult:
    grouped: dict[str, dict[str, Any]] = {}
    evidence_models: list[GraphRAGEvidence] = []

    for row in rows:
        name = str(row.get("name") or "").strip()
        canonical_name = str(
            row.get("canonical_name") or _canonical_syndrome(name)
        ).strip()
        raw_evidence_rows = row.get("evidence_rows") or []
        if not name or not canonical_name or not raw_evidence_rows:
            continue
        group = grouped.setdefault(canonical_name, {
            "names": [],
            "syndrome_node_ids": [],
            "evidence_ids": [],
            "evidences": [],
            "related_tcm_diseases": [],
            "constitutions": [],
            "diagnostic_axis": defaultdict(list),
        })
        group["names"].append(name)
        group["syndrome_node_ids"].append(row.get("syndrome_id"))
        group["related_tcm_diseases"].extend(row.get("related_tcm_diseases") or [])
        group["constitutions"].extend(row.get("constitutions") or [])
        for role, field in (
            ("main", "main_symptoms"),
            ("supplement", "supplement_symptoms"),
            ("tongue", "tongue_symptoms"),
            ("pulse", "pulse_symptoms"),
        ):
            group["diagnostic_axis"][role].extend(row.get(field) or [])

        for evidence_row in raw_evidence_rows:
            relationship_id = str(evidence_row.get("relationship_id") or "")
            syndrome_id = str(evidence_row.get("syndrome_id") or row.get("syndrome_id") or "")
            symptom_id = str(evidence_row.get("symptom_id") or "")
            symptom_name = str(evidence_row.get("symptom_name") or "")
            role = str(evidence_row.get("symptom_role") or "unknown")
            if role not in ROLE_LABELS:
                role = "unknown"
            weight = float(evidence_row.get("evidence_weight") or 0)
            matched_keywords = _deduplicate(evidence_row.get("matched_keywords") or [])
            evidence_id = f"{MED_SOURCE_DB}:{relationship_id}"
            evidence = GraphRAGEvidence(
                evidence_id=evidence_id,
                source_db=MED_SOURCE_DB,
                source_archive_sha256=(
                    evidence_row.get("source_archive_sha256")
                    or row.get("source_archive_sha256")
                ),
                source_tenant=(
                    evidence_row.get("source_tenant") or row.get("source_tenant")
                ),
                syndrome_name=name,
                symptom_name=symptom_name,
                matched_keywords=matched_keywords,
                symptom_role=role,
                evidence_weight=weight,
                score=min(1.0, weight / 3.0),
                node_ids=_deduplicate([syndrome_id, symptom_id]),
                relationship_ids=_deduplicate([relationship_id]),
                relationship_path=[
                    f"Syndrome[{syndrome_id}]",
                    f"-[:{DIAGNOSTIC_RELATIONSHIP} "
                    f"{{med_relationship_id:{relationship_id}}}]->",
                    f"TCMSymptom[{symptom_id}]",
                ],
                statement=(
                    f"{name}的{ROLE_LABELS.get(role, '症状')}“{symptom_name}”"
                    f"与输入“{'、'.join(matched_keywords)}”匹配"
                ),
            )
            group["evidence_ids"].append(evidence_id)
            group["evidences"].append(evidence)

    candidates: list[GraphRAGSyndromeCandidate] = []
    selected_evidence_ids: set[str] = set()
    for canonical_name, group in grouped.items():
        evidences: list[GraphRAGEvidence] = group["evidences"]
        best_weight_by_keyword: dict[str, float] = {}
        for evidence in evidences:
            for keyword in evidence.matched_keywords:
                best_weight_by_keyword[keyword] = max(
                    best_weight_by_keyword.get(keyword, 0.0),
                    evidence.evidence_weight,
                )
        matched_keywords = list(best_weight_by_keyword)
        match_count = len(matched_keywords)
        weighted_score = sum(best_weight_by_keyword.values())
        coverage = min(1.0, match_count / max(1, min(len(keywords), 8)))
        role_quality = min(
            1.0,
            weighted_score / max(1.0, match_count * 3.0),
        )
        score = min(1.0, coverage * 0.75 + role_quality * 0.25)
        names = _deduplicate(group["names"])
        syndrome_node_ids = _deduplicate(group["syndrome_node_ids"])
        evidence_ids = _deduplicate(group["evidence_ids"])
        candidates.append(GraphRAGSyndromeCandidate(
            syndrome_id=syndrome_node_ids[0] if syndrome_node_ids else None,
            syndrome_node_ids=syndrome_node_ids,
            name=_preferred_syndrome_name(names, canonical_name),
            canonical_name=canonical_name,
            score=score,
            match_count=match_count,
            weighted_score=weighted_score,
            matched_keywords=matched_keywords,
            evidence_ids=evidence_ids,
            related_tcm_diseases=_deduplicate(group["related_tcm_diseases"]),
            constitutions=_deduplicate(group["constitutions"]),
            diagnostic_axis={
                role: _deduplicate(group["diagnostic_axis"][role])
                for role in ("main", "supplement", "tongue", "pulse")
            },
        ))

    candidates.sort(
        key=lambda item: (item.score, item.match_count, item.weighted_score),
        reverse=True,
    )
    candidates = candidates[:top_k]
    for candidate in candidates:
        selected_evidence_ids.update(candidate.evidence_ids)
    for group in grouped.values():
        for evidence in group["evidences"]:
            if evidence.evidence_id in selected_evidence_ids:
                evidence_models.append(evidence)
    evidence_models.sort(
        key=lambda item: (item.evidence_weight, len(item.matched_keywords)),
        reverse=True,
    )

    return GraphRAGResult(
        query=query,
        keywords=keywords,
        retrieval_mode="neo4j_graph",
        graph_available=True,
        vector_index_used=False,
        candidates=candidates,
        evidences=evidence_models,
        source_dbs=[MED_SOURCE_DB] if candidates else [],
        warnings=(
            []
            if candidates
            else ["当前真实诊断关系未命中输入症状，未生成图谱候选。"]
        ),
    )


async def retrieve_diagnostic_graph(
    collected_info: Any,
    *,
    top_k: int = 5,
    graph: Any | None = None,
) -> GraphRAGResult:
    """检索并展开证候—症状—体质—病种子图，不调用模拟数据。"""
    keywords = extract_graph_rag_keywords(collected_info)
    query = "、".join(keywords)
    if not keywords:
        return GraphRAGResult(
            query=query,
            keywords=[],
            retrieval_mode="unavailable",
            graph_available=False,
            warnings=["没有可用于 GraphRAG 的有效症状关键词。"],
        )

    if graph is None:
        try:
            from app.src.core.graph_db import get_neo4j_graph

            graph = get_neo4j_graph(database="neo4j")
        except Exception as exc:
            logger.warning("GraphRAG 无法初始化 Neo4j: %s", exc)
            graph = None
    if graph is None:
        return GraphRAGResult(
            query=query,
            keywords=keywords,
            retrieval_mode="unavailable",
            graph_available=False,
            warnings=["Neo4j 当前不可用，未使用任何模拟检索结果。"],
        )

    try:
        rows = await asyncio.to_thread(
            graph.query,
            DIAGNOSTIC_GRAPH_QUERY,
            params={
                "keywords": keywords,
                "row_limit": max(top_k * 4, 12),
            },
        )
    except Exception as exc:
        logger.error("GraphRAG 诊断子图查询失败: %s", exc)
        return GraphRAGResult(
            query=query,
            keywords=keywords,
            retrieval_mode="unavailable",
            graph_available=False,
            warnings=["Neo4j 诊断子图查询失败，未使用任何模拟检索结果。"],
        )

    result = _build_graph_rag_result(query, keywords, rows, top_k)
    logger.info(
        "GraphRAG 检索完成: keywords=%d, candidates=%d, evidences=%d",
        len(result.keywords),
        len(result.candidates),
        len(result.evidences),
    )
    return result


def format_graph_rag_context(result: GraphRAGResult, evidence_limit: int = 8) -> str:
    """生成带证据编号的 LLM 上下文，不包含隐藏推理。"""
    if not result.candidates:
        return "GraphRAG 未命中真实诊断关系。"
    lines = [
        "以下内容来自 Neo4j 真实关系，只能作为辨证辅助证据：",
    ]
    for index, candidate in enumerate(result.candidates[:5], 1):
        lines.append(
            f"候选{index}：{candidate.name}；匹配词："
            f"{'、'.join(candidate.matched_keywords)}；得分：{candidate.score:.2f}"
        )
    for index, evidence in enumerate(result.evidences[:evidence_limit], 1):
        lines.append(
            f"[G{index}] {evidence.statement}；角色={evidence.symptom_role}；"
            f"权重={evidence.evidence_weight:.1f}；来源={evidence.source_db}；"
            f"路径={' '.join(evidence.relationship_path)}"
        )
    return "\n".join(lines)
