"""全量中医知识图谱与活跃查询验收。

默认验收全量药理图谱、med TCM 诊断轴、HPO 来源隔离和活跃检索。
当前仍缺少中医症状到现代医学症状的可信关系源，因此严格临床全链单独
报告为 ``BLOCKED_SOURCE_DATA``。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


NODE_THRESHOLDS = {
    "Formula": 25_000,
    "Herb": 18_000,
    "Ingredient": 130_000,
    "Target": 59_000,
    "Disease": 70_000,
    "TCMSymptom": 5_400,
    "MMSymptom": 21_000,
    "Syndrome": 630,
    "TCMDisease": 100,
    "Constitution": 8,
    "TCMDiseaseCategory": 15,
}

SUPPORTED_RELATIONSHIP_THRESHOLDS = {
    "FORMULA_CONTAINS_HERB": 1_000,
    "HERB_CONTAINS_INGREDIENT": 2_000,
    "INGREDIENT_TARGETS": 60_000,
    "TARGET_ASSOCIATED_WITH_DISEASE": 10_000,
    "DISEASE_HAS_MM_SYMPTOM": 280_000,
    "SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM": 5_000,
    "SYNDROME_PATTERN_OF_TCM_DISEASE": 500,
    "SYNDROME_ASSOCIATED_WITH_CONSTITUTION": 300,
    "TCM_DISEASE_IN_CATEGORY": 100,
}

SOURCE_THRESHOLDS = {
    "SymMap": 65_000,
    "ITCM": 107_000,
    "TCMBank": 118_000,
    "HPO": 25_000,
    "med_tcm": 3_700,
}

UNAVAILABLE_CORE_RELATIONSHIPS = (
    "TCM_SYMPTOM_MAPS_TO_MM_SYMPTOM",
)


class ThresholdResult(BaseModel):
    name: str
    actual: int = Field(ge=0)
    minimum: int = Field(ge=0)
    passed: bool


class ActiveQueryResult(BaseModel):
    name: str
    count: int = Field(ge=0)
    samples: list[str] = Field(default_factory=list)
    passed: bool


class FullGraphReport(BaseModel):
    node_results: list[ThresholdResult]
    relationship_results: list[ThresholdResult]
    source_results: list[ThresholdResult]
    active_queries: list[ActiveQueryResult]
    total_nodes: int
    total_relationships: int
    index_count: int
    constraint_count: int
    hpo_duplicate_keys: int
    overwritten_symmap_nodes: int
    pharmacology_chain_connected: bool
    diagnostic_axis_connected: bool
    tongue_pulse_axis_connected: bool
    unavailable_core_relationships: list[str]
    med_data_quality_passed: bool

    @property
    def import_passed(self) -> bool:
        from scripts.tcm_graph_schema import CONSTRAINT_SPECS, INDEX_SPECS

        return all(
            result.passed
            for result in (
                self.node_results
                + self.relationship_results
                + self.source_results
            )
        ) and self.index_count >= len(INDEX_SPECS) \
            and self.constraint_count >= len(CONSTRAINT_SPECS) \
            and self.hpo_duplicate_keys == 0 \
            and self.overwritten_symmap_nodes == 0 \
            and self.med_data_quality_passed

    @property
    def active_queries_passed(self) -> bool:
        return all(result.passed for result in self.active_queries)

    @property
    def clinical_six_hop_ready(self) -> bool:
        return (
            self.pharmacology_chain_connected
            and self.diagnostic_axis_connected
            and not self.unavailable_core_relationships
        )


def evaluate_thresholds(
    actual_counts: dict[str, int],
    thresholds: dict[str, int],
) -> list[ThresholdResult]:
    return [
        ThresholdResult(
            name=name,
            actual=int(actual_counts.get(name, 0)),
            minimum=minimum,
            passed=int(actual_counts.get(name, 0)) >= minimum,
        )
        for name, minimum in thresholds.items()
    ]


def _counts_by_key(rows: list[dict[str, Any]], key: str = "name") -> dict[str, int]:
    return {str(row[key]): int(row["count"]) for row in rows}


async def verify_active_queries() -> list[ActiveQueryResult]:
    from app.src.agent.components.diagnose.models import CollectedDiagnoseInfo
    from app.src.agent.components.diagnose.nodes.moderate_diagnosis.moderate_diagnosis import (
        _query_prescriptions_for_syndrome,
        _query_similar_syndromes,
    )
    from app.src.agent.components.herb.handlers import _search_herbs
    from app.src.agent.components.prescription.handlers import _search_prescriptions

    info = CollectedDiagnoseInfo(
        chief_complaint="头痛，心悸，失眠多梦",
        head_body="头痛乏力",
        chest_abdomen="心悸",
        sleep="失眠多梦",
    )
    syndromes, herb_result, formula_result = await asyncio.gather(
        _query_similar_syndromes(info),
        _search_herbs(["黄芪"]),
        _search_prescriptions(["归脾汤", "头痛", "失眠"]),
    )
    target_syndrome = next(
        (item for item in syndromes if item.get("canonical_name") == "心脾两虚"),
        syndromes[0] if syndromes else {},
    )
    verified_formulas = await _query_prescriptions_for_syndrome(
        str(target_syndrome.get("name") or ""),
        target_syndrome.get("id"),
    )
    herbs, _ = herb_result
    formulas, _ = formula_result

    raw_results = [
        ("moderate_similar_syndromes", syndromes, "name"),
        ("herb_handler_retrieval", herbs, "name"),
        ("prescription_handler_retrieval", formulas, "name"),
    ]
    results: list[ActiveQueryResult] = []
    for name, items, field in raw_results:
        samples: list[str] = []
        for item in items[:3]:
            value = item.get(field) if isinstance(item, dict) else getattr(item, field, None)
            if value:
                samples.append(str(value))
        results.append(ActiveQueryResult(
            name=name,
            count=len(items),
            samples=samples,
            passed=bool(items),
        ))
    results.append(ActiveQueryResult(
        name="diagnosis_formula_relation_guard",
        count=len(verified_formulas),
        samples=[str(item.get("name")) for item in verified_formulas[:3] if item.get("name")],
        passed=all(
            item.get("relationship_type") == "TREATS_WITH"
            and item.get("relationship_id")
            and item.get("relationship_path")
            for item in verified_formulas
        ),
    ))
    return results


async def build_report(database: str = "neo4j") -> FullGraphReport:
    from app.src.core.graph_db import get_neo4j_graph
    from scripts.audit_med_tcm_diagnostic_axis import audit_graph

    graph = get_neo4j_graph(database=database)
    if graph is None:
        raise RuntimeError(f"Neo4j 未连接（database={database}）")

    node_counts = _counts_by_key(graph.query("""
        MATCH (n)
        UNWIND labels(n) AS name
        RETURN name, count(*) AS count
        ORDER BY name
    """))
    relationship_counts = _counts_by_key(graph.query("""
        MATCH ()-[r]->()
        RETURN type(r) AS name, count(*) AS count
        ORDER BY name
    """))
    source_counts = _counts_by_key(graph.query("""
        MATCH (n)
        WHERE n.source_db IS NOT NULL
        RETURN n.source_db AS name, count(*) AS count
        ORDER BY name
    """))

    total_nodes = int(graph.query("MATCH (n) RETURN count(n) AS count")[0]["count"])
    total_relationships = int(
        graph.query("MATCH ()-[r]->() RETURN count(r) AS count")[0]["count"]
    )
    index_count = int(graph.query("""
        SHOW INDEXES YIELD name
        WHERE name STARTS WITH 'idx_'
        RETURN count(*) AS count
    """)[0]["count"])
    constraint_count = int(graph.query("""
        SHOW CONSTRAINTS YIELD name
        WHERE name STARTS WITH 'uq_'
        RETURN count(*) AS count
    """)[0]["count"])
    hpo_duplicate_keys = int(graph.query("""
        MATCH (m:MMSymptom {source_db: 'HPO'})
        WITH m.hpo_id AS hpo_id, count(*) AS count
        WHERE count > 1
        RETURN count(*) AS duplicate_keys
    """)[0]["duplicate_keys"])
    overwritten_symmap_nodes = int(graph.query("""
        MATCH (m:MMSymptom {source_db: 'HPO'})
        WHERE m.symmap_id IS NOT NULL
        RETURN count(*) AS count
    """)[0]["count"])
    pharmacology_chain_connected = bool(graph.query("""
        MATCH p=(:Formula)-[:FORMULA_CONTAINS_HERB]->(:Herb)
          -[:HERB_CONTAINS_INGREDIENT]->(:Ingredient)
          -[:INGREDIENT_TARGETS]->(:Target)
          -[:TARGET_ASSOCIATED_WITH_DISEASE]->(:Disease)
          -[:DISEASE_HAS_MM_SYMPTOM]->(:MMSymptom)
        RETURN length(p) AS hops
        LIMIT 1
    """))
    diagnostic_axis_connected = bool(graph.query("""
        MATCH (:TCMDisease)<-[:SYNDROME_PATTERN_OF_TCM_DISEASE]-(sy:Syndrome)
              -[:SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM]->(:TCMSymptom)
        MATCH (sy)-[:SYNDROME_ASSOCIATED_WITH_CONSTITUTION]->(:Constitution)
        RETURN sy.name_zh AS syndrome
        LIMIT 1
    """))
    tongue_pulse_axis_connected = bool(graph.query("""
        MATCH (sy:Syndrome)-[:SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM]
              ->(:TCMSymptom:TongueSymptom)
        MATCH (sy)-[:SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM]
              ->(:TCMSymptom:PulseSymptom)
        RETURN sy.name_zh AS syndrome
        LIMIT 1
    """))

    unavailable = [
        rel_type
        for rel_type in UNAVAILABLE_CORE_RELATIONSHIPS
        if relationship_counts.get(rel_type, 0) == 0
    ]
    if relationship_counts.get("TREATS_WITH", 0) == 0:
        unavailable.append("SYNDROME_TREATS_WITH_FORMULA")
    med_audit = audit_graph(graph, database)

    return FullGraphReport(
        node_results=evaluate_thresholds(node_counts, NODE_THRESHOLDS),
        relationship_results=evaluate_thresholds(
            relationship_counts,
            SUPPORTED_RELATIONSHIP_THRESHOLDS,
        ),
        source_results=evaluate_thresholds(source_counts, SOURCE_THRESHOLDS),
        active_queries=await verify_active_queries(),
        total_nodes=total_nodes,
        total_relationships=total_relationships,
        index_count=index_count,
        constraint_count=constraint_count,
        hpo_duplicate_keys=hpo_duplicate_keys,
        overwritten_symmap_nodes=overwritten_symmap_nodes,
        pharmacology_chain_connected=pharmacology_chain_connected,
        diagnostic_axis_connected=diagnostic_axis_connected,
        tongue_pulse_axis_connected=tongue_pulse_axis_connected,
        unavailable_core_relationships=unavailable,
        med_data_quality_passed=med_audit.passed,
    )


def print_report(report: FullGraphReport) -> None:
    def print_group(title: str, results: list[ThresholdResult]) -> None:
        print(title)
        for result in results:
            state = "PASS" if result.passed else "FAIL"
            print(
                f"  [{state}] {result.name}: "
                f"actual={result.actual}, minimum={result.minimum}"
            )

    print_group("[NODES]", report.node_results)
    print_group("[RELATIONSHIPS]", report.relationship_results)
    print_group("[SOURCES]", report.source_results)
    print("[ACTIVE_QUERIES]")
    for result in report.active_queries:
        state = "PASS" if result.passed else "FAIL"
        print(f"  [{state}] {result.name}: count={result.count}, samples={result.samples}")

    print(
        "[INTEGRITY] "
        f"indexes={report.index_count}, constraints={report.constraint_count}, "
        f"hpo_duplicate_keys={report.hpo_duplicate_keys}, "
        f"overwritten_symmap_nodes={report.overwritten_symmap_nodes}"
    )
    print(
        f"[TOTALS] nodes={report.total_nodes}, relationships={report.total_relationships}"
    )
    print(
        "NETWORK_PHARMACOLOGY_CHAIN_ACCEPTANCE="
        f"{'PASS' if report.pharmacology_chain_connected else 'FAIL'}"
    )
    print(
        "TCM_DIAGNOSTIC_AXIS_ACCEPTANCE="
        f"{'PASS' if report.diagnostic_axis_connected else 'FAIL'}"
    )
    print(
        "TONGUE_PULSE_AXIS_ACCEPTANCE="
        f"{'PASS' if report.tongue_pulse_axis_connected else 'FAIL'}"
    )
    print(
        "MED_TCM_DATA_QUALITY_ACCEPTANCE="
        f"{'PASS' if report.med_data_quality_passed else 'FAIL'}"
    )
    print(f"FULL_GRAPH_IMPORT_ACCEPTANCE={'PASS' if report.import_passed else 'FAIL'}")
    print(
        "ACTIVE_GRAPH_QUERY_ACCEPTANCE="
        f"{'PASS' if report.active_queries_passed else 'FAIL'}"
    )
    if report.clinical_six_hop_ready:
        print("CLINICAL_SIX_HOP_ACCEPTANCE=PASS")
    else:
        print("CLINICAL_SIX_HOP_ACCEPTANCE=BLOCKED_SOURCE_DATA")
        print(
            "MISSING_OFFICIAL_RELATIONSHIPS="
            + ",".join(report.unavailable_core_relationships)
        )


async def main(database: str, strict_seven_relations: bool) -> int:
    try:
        report = await build_report(database)
    except Exception as exc:
        print(f"FULL_GRAPH_IMPORT_ACCEPTANCE=FAIL\nERROR={exc}")
        return 1
    print_report(report)
    if not report.import_passed or not report.active_queries_passed:
        return 1
    if strict_seven_relations and not report.clinical_six_hop_ready:
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="neo4j", choices=["neo4j", "tcm_graph"])
    parser.add_argument(
        "--strict-seven-relations",
        action="store_true",
        help="蓝图七类核心关系缺任何一类时返回失败",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.db, args.strict_seven_relations)))
