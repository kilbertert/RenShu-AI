"""可审计诊断 GraphRAG 测试。"""

from unittest.mock import MagicMock

import pytest

from app.src.agent.components.diagnose.models import (
    CollectedDiagnoseInfo,
    DiagnosisCitation,
    DiagnosisResult,
)
from app.src.agent.components.diagnose.nodes.moderate_diagnosis.moderate_diagnosis_map_reduce import (
    _apply_real_case_boundary,
    _build_retrieval_citations,
)
from app.src.agent.retrieval.graphrag import (
    extract_graph_rag_keywords,
    format_graph_rag_context,
    retrieve_diagnostic_graph,
)


def _graph_rows():
    return [
        {
            "syndrome_id": "MED:Disease:1",
            "name": "心脾两虚证",
            "canonical_name": "心脾两虚",
            "source_archive_sha256": "hash",
            "source_tenant": "10001",
            "evidence_rows": [
                {
                    "syndrome_id": "MED:Disease:1",
                    "symptom_id": "MED:MainSymptom:2",
                    "relationship_id": 10,
                    "symptom_name": "心悸怔忡",
                    "symptom_role": "main",
                    "evidence_weight": 3.0,
                    "matched_keywords": ["心悸"],
                    "source_archive_sha256": "hash",
                    "source_tenant": "10001",
                },
                {
                    "syndrome_id": "MED:Disease:1",
                    "symptom_id": "MED:TongueSymptom:3",
                    "relationship_id": 11,
                    "symptom_name": "舌淡",
                    "symptom_role": "tongue",
                    "evidence_weight": 1.5,
                    "matched_keywords": ["舌淡"],
                    "source_archive_sha256": "hash",
                    "source_tenant": "10001",
                },
            ],
            "related_tcm_diseases": ["不寐"],
            "constitutions": ["气虚体质"],
            "main_symptoms": ["心悸怔忡"],
            "supplement_symptoms": ["健忘"],
            "tongue_symptoms": ["舌淡"],
            "pulse_symptoms": ["脉细弱"],
        },
        {
            "syndrome_id": "MED:Disease:4",
            "name": "心脾两虚",
            "canonical_name": "心脾两虚",
            "source_archive_sha256": "hash",
            "source_tenant": "10001",
            "evidence_rows": [
                {
                    "syndrome_id": "MED:Disease:4",
                    "symptom_id": "MED:PulseSymptom:5",
                    "relationship_id": 12,
                    "symptom_name": "脉细弱",
                    "symptom_role": "pulse",
                    "evidence_weight": 1.5,
                    "matched_keywords": ["脉细弱"],
                    "source_archive_sha256": "hash",
                    "source_tenant": "10001",
                },
            ],
            "related_tcm_diseases": ["心悸"],
            "constitutions": ["气虚体质"],
            "main_symptoms": ["神疲乏力"],
            "supplement_symptoms": ["健忘"],
            "tongue_symptoms": ["舌淡"],
            "pulse_symptoms": ["脉细弱"],
        },
    ]


def test_extract_graph_rag_keywords_includes_structured_tongue_and_pulse():
    info = CollectedDiagnoseInfo(
        chief_complaint="心悸，健忘",
        tongue={"tongue_color": "淡", "source": "text"},
        pulse={"description": "细弱", "source": "text"},
    )

    assert extract_graph_rag_keywords(info) == ["心悸", "健忘", "舌淡", "脉细弱"]


@pytest.mark.asyncio
async def test_graph_rag_aggregates_canonical_syndromes_and_preserves_paths():
    graph = MagicMock()
    graph.query.return_value = _graph_rows()
    info = CollectedDiagnoseInfo(
        chief_complaint="心悸",
        tongue={"tongue_color": "淡"},
        pulse={"description": "细弱", "source": "text"},
    )

    result = await retrieve_diagnostic_graph(info, graph=graph)

    assert result.graph_available is True
    assert result.vector_index_used is False
    assert len(result.candidates) == 1
    assert result.candidates[0].name == "心脾两虚证"
    assert result.candidates[0].syndrome_node_ids == [
        "MED:Disease:1",
        "MED:Disease:4",
    ]
    assert {item.symptom_role for item in result.evidences} == {
        "main",
        "tongue",
        "pulse",
    }
    assert all(item.node_ids for item in result.evidences)
    assert all(item.relationship_path for item in result.evidences)
    assert "[G1]" in format_graph_rag_context(result)


@pytest.mark.asyncio
async def test_graph_rag_unavailable_never_returns_mock_candidates():
    result = await retrieve_diagnostic_graph(["心悸"], graph=None)

    # 本测试环境可能有真实 Neo4j；只约束不可用时必须为空，不能回落模拟数据。
    if not result.graph_available:
        assert result.candidates == []
        assert result.evidences == []


@pytest.mark.asyncio
async def test_graph_evidence_becomes_structured_diagnosis_citation():
    graph = MagicMock()
    graph.query.return_value = _graph_rows()
    result = await retrieve_diagnostic_graph(["心悸", "舌淡"], graph=graph)

    citations = _build_retrieval_citations(
        result.to_legacy_candidates(),
        [],
        [],
        DiagnosisCitation,
        graph_rag_result=result,
    )

    assert citations
    assert citations[0].source_type == "graph_path"
    assert citations[0].citation_id
    assert citations[0].node_ids
    assert citations[0].relationship_ids
    assert citations[0].evidence_weight is not None


def test_treatment_pattern_is_not_labeled_as_real_case() -> None:
    citations = _build_retrieval_citations(
        [],
        [{
            "syndrome": "心脾两虚证",
            "treatment": "归脾汤",
            "source": "symmap_syndrome_to_itcm_formula",
            "chief_complaint": "心悸、乏力",
            "match_score": 0.8,
        }],
        [],
        DiagnosisCitation,
    )

    assert citations[0].source_type == "treatment_pattern"


def test_verified_formula_relation_becomes_traceable_citation() -> None:
    citations = _build_retrieval_citations(
        [],
        [],
        [{
            "name": "归脾汤",
            "syndrome_name": "心脾两虚证",
            "syndrome_id": "SY-1",
            "formula_id": "F-1",
            "relationship_type": "TREATS_WITH",
            "relationship_id": "R-1",
            "relationship_path": [
                "Syndrome[SY-1]",
                "-[:TREATS_WITH {R-1}]-",
                "Formula[F-1]",
            ],
            "source_db": "curated_tcm",
            "indications": "心脾两虚",
        }],
        DiagnosisCitation,
    )

    assert citations[0].source_type == "formula_relation"
    assert citations[0].node_ids == ["SY-1", "F-1"]
    assert citations[0].relationship_ids == ["R-1"]
    assert citations[0].relationship_path


def test_real_case_request_gets_deterministic_empty_vector_boundary() -> None:
    diagnosis = DiagnosisResult(
        syndrome="心脾两虚证",
        patient_answer="根据症状考虑心脾两虚证。",
    )

    _apply_real_case_boundary(
        diagnosis,
        "请列出真实相似医案的病例编号和来源。",
    )

    assert diagnosis.patient_answer.startswith("**真实医案检索边界**")
    assert "尚未配置可用的真实患者医案向量库" in diagnosis.patient_answer
