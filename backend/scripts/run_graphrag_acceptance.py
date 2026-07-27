"""真实 Neo4j GraphRAG 与活跃问诊检索验收。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


async def main() -> int:
    from app.src.agent.components.diagnose.models import (
        CollectedDiagnoseInfo,
        DiagnosisCitation,
    )
    from app.src.agent.components.diagnose.nodes.moderate_diagnosis.moderate_diagnosis import (
        _query_prescriptions_for_syndrome,
        _query_similar_syndromes,
    )
    from app.src.agent.components.diagnose.nodes.moderate_diagnosis.moderate_diagnosis_map_reduce import (
        _build_retrieval_citations,
    )
    from app.src.agent.retrieval.graphrag import retrieve_diagnostic_graph

    info = CollectedDiagnoseInfo(
        chief_complaint="心悸怔忡，健忘多梦，食少乏力",
        chest_abdomen="心悸怔忡",
        sleep="多梦易醒",
        diet="食少",
        tongue={"tongue_color": "淡", "coating_color": "薄白"},
        other_symptoms=["脉细弱"],
    )
    result = await retrieve_diagnostic_graph(info, top_k=5)
    active_candidates = await _query_similar_syndromes(info)
    citations = _build_retrieval_citations(
        result.to_legacy_candidates(),
        [],
        [],
        DiagnosisCitation,
        graph_rag_result=result,
    )

    target = next(
        (item for item in result.candidates if item.canonical_name == "心脾两虚"),
        None,
    )
    verified_formulas = await _query_prescriptions_for_syndrome(
        target.name if target else "心脾两虚证",
        target.syndrome_id if target else None,
    )
    provenance_passed = bool(
        result.evidences
        and all(
            evidence.source_db == "med_tcm"
            and evidence.node_ids
            and evidence.relationship_ids
            and evidence.relationship_path
            and evidence.source_archive_sha256
            and evidence.source_tenant == "10001"
            for evidence in result.evidences
        )
    )
    target_passed = bool(
        target
        and target.match_count >= 3
        and target.diagnostic_axis.get("main")
        and target.diagnostic_axis.get("supplement")
        and target.diagnostic_axis.get("tongue")
        and target.diagnostic_axis.get("pulse")
        and target.constitutions
        and target.related_tcm_diseases
    )
    active_passed = bool(
        active_candidates
        and any(item.get("canonical_name") == "心脾两虚" for item in active_candidates)
        and citations
        and all(citation.source_type == "graph_path" for citation in citations)
    )
    formula_guard_passed = all(
        item.get("relationship_type") == "TREATS_WITH"
        and item.get("relationship_id")
        and item.get("relationship_path")
        for item in verified_formulas
    )

    print(f"GRAPHRAG_CANDIDATES={len(result.candidates)}")
    print(f"GRAPHRAG_EVIDENCES={len(result.evidences)}")
    print(f"GRAPHRAG_TOP={result.candidates[0].name if result.candidates else ''}")
    print("GRAPHRAG_VECTOR_INDEX=NOT_CONFIGURED")
    print(f"GRAPHRAG_VERIFIED_FORMULAS={len(verified_formulas)}")
    print(
        "GRAPHRAG_PROVENANCE_ACCEPTANCE="
        f"{'PASS' if provenance_passed else 'FAIL'}"
    )
    print(
        "GRAPHRAG_DIAGNOSTIC_AXIS_ACCEPTANCE="
        f"{'PASS' if target_passed else 'FAIL'}"
    )
    print(
        "GRAPHRAG_ACTIVE_QUERY_ACCEPTANCE="
        f"{'PASS' if active_passed else 'FAIL'}"
    )
    print(
        "GRAPHRAG_FORMULA_RELATION_GUARD_ACCEPTANCE="
        f"{'PASS' if formula_guard_passed else 'FAIL'}"
    )
    passed = (
        result.graph_available
        and provenance_passed
        and target_passed
        and active_passed
        and formula_guard_passed
    )
    print(f"GRAPHRAG_ACCEPTANCE={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
