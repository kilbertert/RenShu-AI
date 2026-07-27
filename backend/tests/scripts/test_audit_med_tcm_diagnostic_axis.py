"""med TCM 数据质量审计规则测试。"""

from scripts.audit_med_tcm_diagnostic_axis import (
    EXPECTED_NODE_COUNT,
    EXPECTED_RELATIONSHIP_COUNTS,
    MedDiagnosticAxisAuditReport,
)
from scripts.import_med_tcm_diagnostic_axis import (
    DEFAULT_TENANT_ID,
    EXPECTED_SHA256,
    IMPORT_SCHEMA_VERSION,
)


def _valid_report(**updates):
    values = {
        "database": "neo4j",
        "node_count": EXPECTED_NODE_COUNT,
        "relationship_counts": EXPECTED_RELATIONSHIP_COUNTS,
        "duplicate_source_id_groups": 0,
        "duplicate_relationship_id_groups": 0,
        "duplicate_semantic_edge_groups": 0,
        "empty_name_nodes": 0,
        "isolated_nodes": 0,
        "cross_source_relationships": 0,
        "invalid_node_mappings": 0,
        "invalid_role_mappings": 0,
        "missing_node_provenance": 0,
        "missing_relationship_provenance": 0,
        "tenant_ids": [DEFAULT_TENANT_ID],
        "archive_hashes": [EXPECTED_SHA256],
        "import_schema_versions": [IMPORT_SCHEMA_VERSION],
        "symptom_count": 3_242,
        "non_informative_symptoms": 29,
        "canonical_syndrome_duplicate_groups": 26,
        "normalized_symptom_duplicate_groups": 244,
        "heart_spleen_axis_complete": True,
    }
    values.update(updates)
    return MedDiagnosticAxisAuditReport(**values)


def test_audit_accepts_known_normalization_duplicates_but_not_identity_duplicates():
    report = _valid_report()

    assert report.passed is True
    assert report.canonical_syndrome_duplicate_groups == 26
    assert report.normalized_symptom_duplicate_groups == 244


def test_audit_rejects_missing_provenance_or_cross_source_edges():
    assert _valid_report(missing_node_provenance=1).passed is False
    assert _valid_report(cross_source_relationships=1).passed is False


def test_audit_rejects_excessive_non_informative_symptoms():
    assert _valid_report(non_informative_symptoms=100).passed is False
