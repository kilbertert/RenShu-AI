"""统一知识图谱 schema 测试。"""

from scripts.tcm_graph_schema import (
    CONSTRAINT_SPECS,
    CORE_NODE_LABELS,
    CORE_RELATIONSHIP_TYPES,
    DIAGNOSTIC_NODE_LABELS,
    DIAGNOSTIC_RELATIONSHIP_TYPES,
    INDEX_SPECS,
    NODE_LABELS,
    RELATIONSHIP_TYPES,
)


def test_unified_schema_contains_eight_labels_and_seven_relationships():
    assert len(CORE_NODE_LABELS) == 8
    assert len(CORE_RELATIONSHIP_TYPES) == 7
    assert "FORMULA_CONTAINS_HERB" in RELATIONSHIP_TYPES
    assert "SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM" in RELATIONSHIP_TYPES
    assert "TCMDisease" in DIAGNOSTIC_NODE_LABELS
    assert "TongueSymptom" in DIAGNOSTIC_NODE_LABELS
    assert "PulseSymptom" in DIAGNOSTIC_NODE_LABELS
    assert "SYNDROME_ASSOCIATED_WITH_CONSTITUTION" in DIAGNOSTIC_RELATIONSHIP_TYPES
    assert len(NODE_LABELS) > len(CORE_NODE_LABELS)


def test_schema_index_names_are_unique():
    names = [spec.name for spec in INDEX_SPECS]

    assert len(names) == len(set(names))
    assert all("IF NOT EXISTS" in spec.cypher() for spec in INDEX_SPECS)
    assert any(spec.name == "idx_target_ncbi" for spec in INDEX_SPECS)


def test_med_source_identity_constraints_are_composite_and_unique():
    names = [spec.name for spec in CONSTRAINT_SPECS]

    assert len(names) == len(set(names))
    assert {spec.label for spec in CONSTRAINT_SPECS} >= {
        "Syndrome",
        "TCMSymptom",
        "TCMDisease",
        "Constitution",
        "TCMDiseaseCategory",
    }
    assert all(spec.properties == ("source_db", "med_id") for spec in CONSTRAINT_SPECS)
    assert all("IS UNIQUE" in spec.cypher() for spec in CONSTRAINT_SPECS)
