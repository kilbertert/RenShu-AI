"""审计 med TCM 诊断轴的来源、完整性、幂等性和临床路径。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.import_med_tcm_diagnostic_axis import (
    DEFAULT_TENANT_ID,
    EXPECTED_SHA256,
    IMPORT_SCHEMA_VERSION,
    SOURCE_DB,
)


EXPECTED_NODE_COUNT = 3_777
EXPECTED_RELATIONSHIP_COUNTS = {
    "SYNDROME_ASSOCIATED_WITH_CONSTITUTION": 345,
    "SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM": 5_273,
    "SYNDROME_PATTERN_OF_TCM_DISEASE": 511,
    "TCM_DISEASE_HAS_TCM_SYMPTOM": 14,
    "TCM_DISEASE_IN_CATEGORY": 107,
}
MAX_NON_INFORMATIVE_RATIO = 0.02


class MedDiagnosticAxisAuditReport(BaseModel):
    database: str
    node_count: int = Field(ge=0)
    relationship_counts: dict[str, int]
    duplicate_source_id_groups: int = Field(ge=0)
    duplicate_relationship_id_groups: int = Field(ge=0)
    duplicate_semantic_edge_groups: int = Field(ge=0)
    empty_name_nodes: int = Field(ge=0)
    isolated_nodes: int = Field(ge=0)
    cross_source_relationships: int = Field(ge=0)
    invalid_node_mappings: int = Field(ge=0)
    invalid_role_mappings: int = Field(ge=0)
    missing_node_provenance: int = Field(ge=0)
    missing_relationship_provenance: int = Field(ge=0)
    tenant_ids: list[str]
    archive_hashes: list[str]
    import_schema_versions: list[str]
    symptom_count: int = Field(ge=0)
    non_informative_symptoms: int = Field(ge=0)
    canonical_syndrome_duplicate_groups: int = Field(ge=0)
    normalized_symptom_duplicate_groups: int = Field(ge=0)
    heart_spleen_axis_complete: bool

    @property
    def non_informative_ratio(self) -> float:
        if self.symptom_count == 0:
            return 0.0
        return self.non_informative_symptoms / self.symptom_count

    @property
    def passed(self) -> bool:
        return (
            self.node_count == EXPECTED_NODE_COUNT
            and self.relationship_counts == EXPECTED_RELATIONSHIP_COUNTS
            and self.duplicate_source_id_groups == 0
            and self.duplicate_relationship_id_groups == 0
            and self.duplicate_semantic_edge_groups == 0
            and self.empty_name_nodes == 0
            and self.isolated_nodes == 0
            and self.cross_source_relationships == 0
            and self.invalid_node_mappings == 0
            and self.invalid_role_mappings == 0
            and self.missing_node_provenance == 0
            and self.missing_relationship_provenance == 0
            and self.tenant_ids == [DEFAULT_TENANT_ID]
            and self.archive_hashes == [EXPECTED_SHA256]
            and self.import_schema_versions == [IMPORT_SCHEMA_VERSION]
            and self.symptom_count > 0
            and self.non_informative_ratio <= MAX_NON_INFORMATIVE_RATIO
            and self.heart_spleen_axis_complete
        )


def _scalar(graph: Any, query: str, key: str = "count") -> int:
    rows = graph.query(query)
    return int(rows[0][key]) if rows else 0


def audit_graph(graph: Any, database: str = "neo4j") -> MedDiagnosticAxisAuditReport:
    relationship_rows = graph.query("""
        MATCH ()-[r {source_db: 'med_tcm'}]->()
        RETURN type(r) AS type, count(*) AS count
        ORDER BY type
    """)
    relationship_counts = {
        str(row["type"]): int(row["count"])
        for row in relationship_rows
    }
    symptom_stats = graph.query("""
        MATCH (s:TCMSymptom {source_db: 'med_tcm'})
        RETURN count(s) AS total,
               sum(CASE WHEN s.is_informative = false THEN 1 ELSE 0 END)
                 AS non_informative
    """)[0]
    provenance = graph.query("""
        MATCH (n {source_db: 'med_tcm'})
        RETURN collect(DISTINCT n.source_tenant) AS tenant_ids,
               collect(DISTINCT n.source_archive_sha256) AS archive_hashes,
               collect(DISTINCT n.import_schema_version) AS schema_versions
    """)[0]
    axis_rows = graph.query("""
        MATCH (s:Syndrome {source_db: 'med_tcm', canonical_name: '心脾两虚'})
        MATCH (s)-[r:SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM]
          ->(:TCMSymptom {source_db: 'med_tcm'})
        RETURN collect(DISTINCT r.symptom_role) AS roles
    """)
    axis_roles = set(axis_rows[0]["roles"] if axis_rows else [])
    heart_spleen_axis_complete = (
        {"main", "supplement", "tongue", "pulse"}.issubset(axis_roles)
        and _scalar(graph, """
            MATCH (:Syndrome {source_db: 'med_tcm', canonical_name: '心脾两虚'})
              -[:SYNDROME_ASSOCIATED_WITH_CONSTITUTION]
              ->(:Constitution {source_db: 'med_tcm'})
            RETURN count(*) AS count
        """) > 0
        and _scalar(graph, """
            MATCH (:Syndrome {source_db: 'med_tcm', canonical_name: '心脾两虚'})
              -[:SYNDROME_PATTERN_OF_TCM_DISEASE]
              ->(:TCMDisease {source_db: 'med_tcm'})
            RETURN count(*) AS count
        """) > 0
    )

    return MedDiagnosticAxisAuditReport(
        database=database,
        node_count=_scalar(
            graph,
            "MATCH (n {source_db: 'med_tcm'}) RETURN count(n) AS count",
        ),
        relationship_counts=relationship_counts,
        duplicate_source_id_groups=_scalar(graph, """
            MATCH (n {source_db: 'med_tcm'})
            WITH n.source_label AS source_label, n.med_id AS med_id, count(*) AS copies
            WHERE copies > 1
            RETURN count(*) AS count
        """),
        duplicate_relationship_id_groups=_scalar(graph, """
            MATCH ()-[r {source_db: 'med_tcm'}]->()
            WITH r.med_relationship_id AS med_relationship_id, count(*) AS copies
            WHERE copies > 1
            RETURN count(*) AS count
        """),
        duplicate_semantic_edge_groups=_scalar(graph, """
            MATCH (source)-[r {source_db: 'med_tcm'}]->(target)
            WITH source, target, type(r) AS relationship_type,
                 coalesce(r.symptom_role, '') AS symptom_role,
                 count(*) AS copies
            WHERE copies > 1
            RETURN count(*) AS count
        """),
        empty_name_nodes=_scalar(graph, """
            MATCH (n {source_db: 'med_tcm'})
            WHERE n.name_zh IS NULL OR trim(n.name_zh) = ''
               OR n.normalized_name IS NULL OR trim(n.normalized_name) = ''
            RETURN count(n) AS count
        """),
        isolated_nodes=_scalar(graph, """
            MATCH (n {source_db: 'med_tcm'})
            WHERE NOT (n)--()
            RETURN count(n) AS count
        """),
        cross_source_relationships=_scalar(graph, """
            MATCH (source)-[r {source_db: 'med_tcm'}]->(target)
            WHERE source.source_db <> 'med_tcm' OR target.source_db <> 'med_tcm'
               OR source.source_db IS NULL OR target.source_db IS NULL
            RETURN count(r) AS count
        """),
        invalid_node_mappings=_scalar(graph, """
            MATCH (n {source_db: 'med_tcm'})
            WHERE NOT (
              (n.source_label = 'Disease' AND n:Syndrome) OR
              (n.source_label = 'DiseaseType' AND n:TCMDisease) OR
              (n.source_label IN ['MainSymptom', 'SupplementSymptom',
                 'TongueSymptom', 'VeinSymptom', 'ClinicalSymptom'] AND n:TCMSymptom) OR
              (n.source_label = 'Constitution' AND n:Constitution) OR
              (n.source_label = 'CommonDiseaseCategory' AND n:TCMDiseaseCategory)
            )
            RETURN count(n) AS count
        """),
        invalid_role_mappings=_scalar(graph, """
            MATCH (:Syndrome {source_db: 'med_tcm'})
              -[r:SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM]
              ->(symptom:TCMSymptom {source_db: 'med_tcm'})
            WHERE NOT (
              (r.symptom_role = 'main' AND r.evidence_weight = 3.0
                AND symptom:MainSymptom) OR
              (r.symptom_role = 'supplement' AND r.evidence_weight = 2.0
                AND symptom:SupplementSymptom) OR
              (r.symptom_role = 'tongue' AND r.evidence_weight = 1.5
                AND symptom:TongueSymptom) OR
              (r.symptom_role = 'pulse' AND r.evidence_weight = 1.5
                AND symptom:PulseSymptom)
            )
            RETURN count(r) AS count
        """),
        missing_node_provenance=_scalar(graph, """
            MATCH (n {source_db: 'med_tcm'})
            WHERE n.source_tenant IS NULL OR n.source_archive_sha256 IS NULL
               OR n.import_schema_version IS NULL OR n.raw_name IS NULL
            RETURN count(n) AS count
        """),
        missing_relationship_provenance=_scalar(graph, """
            MATCH ()-[r {source_db: 'med_tcm'}]->()
            WHERE r.source_tenant IS NULL OR r.source_archive_sha256 IS NULL
               OR r.import_schema_version IS NULL
               OR r.med_relationship_id IS NULL OR r.source_relationship IS NULL
            RETURN count(r) AS count
        """),
        tenant_ids=sorted(str(value) for value in provenance["tenant_ids"] if value),
        archive_hashes=sorted(
            str(value) for value in provenance["archive_hashes"] if value
        ),
        import_schema_versions=sorted(
            str(value) for value in provenance["schema_versions"] if value
        ),
        symptom_count=int(symptom_stats["total"]),
        non_informative_symptoms=int(symptom_stats["non_informative"] or 0),
        canonical_syndrome_duplicate_groups=_scalar(graph, """
            MATCH (s:Syndrome {source_db: 'med_tcm'})
            WITH s.canonical_name AS canonical_name, count(*) AS copies
            WHERE copies > 1
            RETURN count(*) AS count
        """),
        normalized_symptom_duplicate_groups=_scalar(graph, """
            MATCH (s:TCMSymptom {source_db: 'med_tcm'})
            WITH s.normalized_name AS normalized_name, count(*) AS copies
            WHERE copies > 1
            RETURN count(*) AS count
        """),
        heart_spleen_axis_complete=heart_spleen_axis_complete,
    )


def audit_database(database: str = "neo4j") -> MedDiagnosticAxisAuditReport:
    from app.src.core.graph_db import get_neo4j_graph

    graph = get_neo4j_graph(database=database)
    if graph is None:
        raise RuntimeError(f"Neo4j 未连接（database={database}）")
    return audit_graph(graph, database)


def print_audit_report(report: MedDiagnosticAxisAuditReport) -> None:
    print("[MED_TCM_QUALITY]")
    print(
        f"  nodes={report.node_count}, relationships="
        f"{sum(report.relationship_counts.values())}"
    )
    print(
        "  integrity="
        f"duplicate_ids:{report.duplicate_source_id_groups}, "
        f"duplicate_relationship_ids:{report.duplicate_relationship_id_groups}, "
        f"duplicate_edges:{report.duplicate_semantic_edge_groups}, "
        f"empty_names:{report.empty_name_nodes}, isolated:{report.isolated_nodes}, "
        f"cross_source:{report.cross_source_relationships}"
    )
    print(
        "  provenance="
        f"missing_nodes:{report.missing_node_provenance}, "
        f"missing_relationships:{report.missing_relationship_provenance}, "
        f"tenants:{report.tenant_ids}, schema:{report.import_schema_versions}"
    )
    print(
        "  normalization="
        f"non_informative:{report.non_informative_symptoms}/{report.symptom_count} "
        f"({report.non_informative_ratio:.2%}), "
        f"canonical_syndrome_groups:{report.canonical_syndrome_duplicate_groups}, "
        f"normalized_symptom_groups:{report.normalized_symptom_duplicate_groups}"
    )
    print(
        "MED_TCM_DATA_QUALITY_ACCEPTANCE="
        f"{'PASS' if report.passed else 'FAIL'}"
    )


def main(database: str = "neo4j") -> int:
    try:
        report = audit_database(database)
    except Exception as exc:
        print(f"MED_TCM_DATA_QUALITY_ACCEPTANCE=FAIL\nERROR={exc}")
        return 1
    print_audit_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="neo4j", choices=["neo4j", "tcm_graph"])
    args = parser.parse_args()
    raise SystemExit(main(args.db))
