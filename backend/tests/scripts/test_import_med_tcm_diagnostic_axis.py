"""med TCM 诊断轴导入映射测试。"""

import csv
import io
import tarfile
from pathlib import Path

from scripts.import_med_tcm_diagnostic_axis import (
    EXPECTED_SHA256,
    IMPORT_SCHEMA_VERSION,
    MedSourceNode,
    MedSourceRelationship,
    canonical_syndrome_name,
    clean_display_name,
    is_informative_symptom,
    load_diagnostic_axis_dataset,
    map_node,
    map_relationship,
    _relationship_upsert_cypher,
)


def test_name_cleaning_preserves_raw_semantics_and_normalizes_noise():
    assert clean_display_name(" 头晕目 花。 ") == "头晕目花"
    assert clean_display_name("面色huang(恍)白") == "面色㿠白"
    assert clean_display_name("月经持续8-l0天始净") == "月经持续8-10天始净"
    assert canonical_syndrome_name("心脾两虚证") == "心脾两虚"
    assert is_informative_symptom("脉细弱") is True
    assert is_informative_symptom("脉") is False


def test_source_disease_maps_to_syndrome():
    mapped = map_node(MedSourceNode(
        id=135,
        labels="Disease",
        tenantId="10001",
        name="心脾两虚证",
    ))

    assert mapped.primary_label == "Syndrome"
    assert mapped.properties["name_zh"] == "心脾两虚证"
    assert mapped.properties["canonical_name"] == "心脾两虚"
    assert mapped.properties["source_db"] == "med_tcm"


def test_tongue_relationship_maps_to_weighted_core_relation():
    syndrome = MedSourceNode(
        id=1,
        labels="Disease",
        tenantId="10001",
        name="心脾两虚证",
    )
    tongue = MedSourceNode(
        id=2,
        labels="TongueSymptom",
        tenantId="10001",
        name="舌淡苔薄",
    )
    relationship = map_relationship(
        MedSourceRelationship(id=3, source=1, target=2, type="has_symptom"),
        syndrome,
        tongue,
    )

    assert relationship.relationship_type == "SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM"
    assert relationship.properties["symptom_role"] == "tongue"
    assert relationship.properties["evidence_weight"] == 1.5


def test_relationship_merge_uses_source_relationship_identity():
    cypher = _relationship_upsert_cypher(
        "Syndrome",
        "SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM",
        "TCMSymptom",
    )

    assert "med_relationship_id: row.med_relationship_id" in cypher
    assert "source_db: $source_db" in cypher


def _write_csv_tar(path: Path) -> None:
    nodes_buffer = io.StringIO()
    nodes_writer = csv.DictWriter(
        nodes_buffer,
        fieldnames=["id", "labels", "tenantId", "name"],
    )
    nodes_writer.writeheader()
    nodes_writer.writerows([
        {"id": 1, "labels": "Disease", "tenantId": "10000", "name": "旧证"},
        {"id": 2, "labels": "Disease", "tenantId": "10001", "name": "风寒感冒证"},
        {"id": 3, "labels": "MainSymptom", "tenantId": "10001", "name": "恶寒。"},
    ])
    relationships_buffer = io.StringIO()
    relationships_writer = csv.DictWriter(
        relationships_buffer,
        fieldnames=["id", "source", "target", "type"],
    )
    relationships_writer.writeheader()
    relationships_writer.writerows([
        {"id": 1, "source": 2, "target": 3, "type": "has_symptom"},
    ])

    with tarfile.open(path, "w:gz") as archive:
        for name, value in (
            ("tcm/nodes.csv", nodes_buffer.getvalue()),
            ("tcm/relationships.csv", relationships_buffer.getvalue()),
        ):
            payload = value.encode("utf-8-sig")
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_dataset_loader_filters_tenant_and_maps_all_relationships(tmp_path: Path):
    archive = tmp_path / "fixture.tar.gz"
    _write_csv_tar(archive)

    dataset = load_diagnostic_axis_dataset(
        archive,
        "10001",
        verify_hash=False,
    )

    assert dataset.stats.mapped_nodes == 2
    assert dataset.stats.mapped_relationships == 1
    assert dataset.stats.cleaned_names == 1
    assert dataset.nodes[0].properties["source_archive_sha256"]
    assert dataset.nodes[0].properties["import_schema_version"] == IMPORT_SCHEMA_VERSION
    assert dataset.relationships[0].properties["source_tenant"] == "10001"
    assert dataset.relationships[0].properties["source_archive_sha256"]
    assert {node.properties["name_zh"] for node in dataset.nodes} == {
        "风寒感冒证",
        "恶寒",
    }
