"""导入 med TCM 诊断轴归档。

只读取 ``tcm_export.tar.gz`` 中 tenant 10001 的超集数据，不执行归档内
``tcm.cypher``。原始 schema 会被重映射为 RenShu-AI 语义：

- Disease -> Syndrome
- DiseaseType -> TCMDisease
- Main/Supplement/Tongue/Vein/ClinicalSymptom -> TCMSymptom + 细分类标签
- Constitution -> Constitution
- CommonDiseaseCategory -> TCMDiseaseCategory

原文保存在 ``raw_name``，展示名称写入 ``name_zh``，检索使用
``normalized_name``。泛化症状不删除，仅设置 ``is_informative=false``。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import sys
import tarfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field


BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.tcm_dataset_config import get_tcm_dataset_root


SOURCE_DB = "med_tcm"
DEFAULT_TENANT_ID = "10001"
DEFAULT_ARCHIVE = get_tcm_dataset_root() / "tcm_export.tar.gz"
EXPECTED_SHA256 = "bd065b8cc30cdc9ede24e7d50c908791091e3af653ee5a36d7914abfd6e80cc8"
IMPORT_SCHEMA_VERSION = "med_tcm_diagnostic_axis_v1"
BATCH_SIZE = 500

GENERIC_SYMPTOM_NAMES = {
    "无",
    "舌",
    "苔",
    "脉",
    "正常",
    "无特殊改变",
}

TRAILING_PUNCTUATION_RE = re.compile(r"[，,。；;、]+$")
NORMALIZED_PUNCTUATION_RE = re.compile(r"[\s，,。；;、（）()【】\[\]·—_\-]+")
KNOWN_OCR_REPLACEMENTS = {
    "huang(恍)": "㿠",
}


class MedSourceNode(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: int
    source_label: str = Field(alias="labels")
    tenant_id: str = Field(alias="tenantId")
    name: str


class MedSourceRelationship(BaseModel):
    id: int
    source: int
    target: int
    type: str


class NodeMappingSpec(BaseModel):
    primary_label: str
    extra_labels: tuple[str, ...] = ()
    symptom_kind: str | None = None

    @property
    def cypher_labels(self) -> str:
        return ":".join((self.primary_label, *self.extra_labels))


class MappedNode(BaseModel):
    med_id: int
    source_label: str
    primary_label: str
    cypher_labels: str
    properties: dict[str, Any]


class MappedRelationship(BaseModel):
    med_relationship_id: int
    source_id: int
    target_id: int
    source_label: str
    target_label: str
    relationship_type: str
    properties: dict[str, Any]


class DiagnosticAxisStats(BaseModel):
    archive_sha256: str
    tenant_id: str
    source_node_counts: dict[str, int]
    mapped_node_counts: dict[str, int]
    relationship_counts: dict[str, int]
    source_nodes: int
    mapped_nodes: int
    source_relationships: int
    mapped_relationships: int
    cleaned_names: int
    non_informative_symptoms: int


class DiagnosticAxisDataset(BaseModel):
    nodes: list[MappedNode]
    relationships: list[MappedRelationship]
    stats: DiagnosticAxisStats


NODE_MAPPINGS: dict[str, NodeMappingSpec] = {
    "Disease": NodeMappingSpec(primary_label="Syndrome"),
    "DiseaseType": NodeMappingSpec(primary_label="TCMDisease"),
    "MainSymptom": NodeMappingSpec(
        primary_label="TCMSymptom",
        extra_labels=("MainSymptom",),
        symptom_kind="main",
    ),
    "SupplementSymptom": NodeMappingSpec(
        primary_label="TCMSymptom",
        extra_labels=("SupplementSymptom",),
        symptom_kind="supplement",
    ),
    "TongueSymptom": NodeMappingSpec(
        primary_label="TCMSymptom",
        extra_labels=("TongueSymptom",),
        symptom_kind="tongue",
    ),
    "VeinSymptom": NodeMappingSpec(
        primary_label="TCMSymptom",
        extra_labels=("PulseSymptom",),
        symptom_kind="pulse",
    ),
    "ClinicalSymptom": NodeMappingSpec(
        primary_label="TCMSymptom",
        extra_labels=("ClinicalSymptom",),
        symptom_kind="clinical",
    ),
    "Constitution": NodeMappingSpec(primary_label="Constitution"),
    "CommonDiseaseCategory": NodeMappingSpec(primary_label="TCMDiseaseCategory"),
}

SYMPTOM_ROLE_WEIGHTS = {
    "MainSymptom": ("main", 3.0),
    "SupplementSymptom": ("supplement", 2.0),
    "TongueSymptom": ("tongue", 1.5),
    "VeinSymptom": ("pulse", 1.5),
}


def archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_display_name(value: str) -> str:
    """清理 Unicode、不可见字符、空白和结尾标点，同时保留源文。"""
    text = unicodedata.normalize("NFKC", value or "")
    text = text.replace("\ufeff", "").replace("\u200b", "").strip()
    text = re.sub(r"\s+", "", text)
    for noisy_text, replacement in KNOWN_OCR_REPLACEMENTS.items():
        text = text.replace(noisy_text, replacement)
    text = re.sub(r"(?<=\d)-[lI](?=\d)", "-1", text)
    return TRAILING_PUNCTUATION_RE.sub("", text).strip()


def normalize_name(value: str) -> str:
    text = clean_display_name(value).lower()
    return NORMALIZED_PUNCTUATION_RE.sub("", text)


def canonical_syndrome_name(value: str) -> str:
    normalized = normalize_name(value)
    if normalized.endswith("证") and len(normalized) > 1:
        return normalized[:-1]
    return normalized


def is_informative_symptom(name: str) -> bool:
    normalized = normalize_name(name)
    return len(normalized) >= 2 and normalized not in GENERIC_SYMPTOM_NAMES


def _read_csv_member(
    archive: tarfile.TarFile,
    member_name: str,
) -> list[dict[str, str]]:
    member = archive.extractfile(member_name)
    if member is None:
        raise FileNotFoundError(f"归档缺少 {member_name}")
    with io.TextIOWrapper(member, encoding="utf-8-sig", newline="") as text:
        return list(csv.DictReader(text))


def read_source_rows(
    archive_path: Path,
) -> tuple[list[MedSourceNode], list[MedSourceRelationship]]:
    with tarfile.open(archive_path, "r:gz") as archive:
        node_rows = _read_csv_member(archive, "tcm/nodes.csv")
        relationship_rows = _read_csv_member(archive, "tcm/relationships.csv")
    return (
        [MedSourceNode.model_validate(row) for row in node_rows],
        [MedSourceRelationship.model_validate(row) for row in relationship_rows],
    )


def map_node(node: MedSourceNode) -> MappedNode:
    spec = NODE_MAPPINGS.get(node.source_label)
    if spec is None:
        raise ValueError(f"未支持的 med 节点标签: {node.source_label}")

    display_name = clean_display_name(node.name)
    normalized_name = normalize_name(display_name)
    properties: dict[str, Any] = {
        "id": f"MED:{node.source_label}:{node.id}",
        "med_id": node.id,
        "source_db": SOURCE_DB,
        "source_tenant": node.tenant_id,
        "source_label": node.source_label,
        "raw_name": node.name,
        "name_zh": display_name,
        "normalized_name": normalized_name,
    }
    if spec.primary_label == "Syndrome":
        properties["canonical_name"] = canonical_syndrome_name(display_name)
    if spec.symptom_kind:
        properties["symptom_kind"] = spec.symptom_kind
        properties["is_informative"] = is_informative_symptom(display_name)

    return MappedNode(
        med_id=node.id,
        source_label=node.source_label,
        primary_label=spec.primary_label,
        cypher_labels=spec.cypher_labels,
        properties=properties,
    )


def map_relationship(
    relationship: MedSourceRelationship,
    source_node: MedSourceNode,
    target_node: MedSourceNode,
) -> MappedRelationship:
    properties: dict[str, Any] = {
        "source_db": SOURCE_DB,
        "source_relationship": relationship.type,
        "med_relationship_id": relationship.id,
    }

    if (
        source_node.source_label == "Disease"
        and relationship.type == "has_symptom"
        and target_node.source_label in SYMPTOM_ROLE_WEIGHTS
    ):
        relationship_type = "SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM"
        symptom_role, evidence_weight = SYMPTOM_ROLE_WEIGHTS[target_node.source_label]
        properties.update({
            "symptom_role": symptom_role,
            "evidence_weight": evidence_weight,
        })
    elif (
        source_node.source_label == "Disease"
        and relationship.type == "is_type"
        and target_node.source_label == "DiseaseType"
    ):
        relationship_type = "SYNDROME_PATTERN_OF_TCM_DISEASE"
    elif (
        source_node.source_label == "Disease"
        and relationship.type == "is_constitution"
        and target_node.source_label == "Constitution"
    ):
        relationship_type = "SYNDROME_ASSOCIATED_WITH_CONSTITUTION"
    elif (
        source_node.source_label == "DiseaseType"
        and relationship.type == "in_category"
        and target_node.source_label == "CommonDiseaseCategory"
    ):
        relationship_type = "TCM_DISEASE_IN_CATEGORY"
    elif (
        source_node.source_label == "DiseaseType"
        and relationship.type == "has_symptom"
        and target_node.source_label == "ClinicalSymptom"
    ):
        relationship_type = "TCM_DISEASE_HAS_TCM_SYMPTOM"
        properties.update({
            "symptom_role": "clinical",
            "evidence_weight": 1.0,
        })
    else:
        raise ValueError(
            "未支持的 med 关系: "
            f"{source_node.source_label}-[{relationship.type}]->{target_node.source_label}"
        )

    return MappedRelationship(
        med_relationship_id=relationship.id,
        source_id=relationship.source,
        target_id=relationship.target,
        source_label=NODE_MAPPINGS[source_node.source_label].primary_label,
        target_label=NODE_MAPPINGS[target_node.source_label].primary_label,
        relationship_type=relationship_type,
        properties=properties,
    )


def load_diagnostic_axis_dataset(
    archive_path: Path = DEFAULT_ARCHIVE,
    tenant_id: str = DEFAULT_TENANT_ID,
    *,
    verify_hash: bool = True,
) -> DiagnosticAxisDataset:
    if not archive_path.exists():
        raise FileNotFoundError(f"med TCM 归档不存在: {archive_path}")
    digest = archive_sha256(archive_path)
    if verify_hash and digest != EXPECTED_SHA256:
        raise ValueError(
            f"归档 SHA256 不一致: expected={EXPECTED_SHA256}, actual={digest}"
        )

    source_nodes, source_relationships = read_source_rows(archive_path)
    selected_nodes = [node for node in source_nodes if node.tenant_id == tenant_id]
    selected_by_id = {node.id: node for node in selected_nodes}
    if not selected_nodes:
        raise ValueError(f"归档中没有 tenantId={tenant_id} 的节点")

    mapped_nodes = [map_node(node) for node in selected_nodes]
    mapped_relationships: list[MappedRelationship] = []
    for relationship in source_relationships:
        source_node = selected_by_id.get(relationship.source)
        target_node = selected_by_id.get(relationship.target)
        if source_node is None or target_node is None:
            continue
        mapped_relationships.append(
            map_relationship(relationship, source_node, target_node)
        )

    provenance = {
        "source_archive_sha256": digest,
        "import_schema_version": IMPORT_SCHEMA_VERSION,
    }
    for node in mapped_nodes:
        node.properties.update(provenance)
    for relationship in mapped_relationships:
        relationship.properties.update({
            **provenance,
            "source_tenant": tenant_id,
        })

    source_node_counts = Counter(node.source_label for node in selected_nodes)
    mapped_node_counts = Counter(node.primary_label for node in mapped_nodes)
    relationship_counts = Counter(
        relationship.relationship_type for relationship in mapped_relationships
    )
    stats = DiagnosticAxisStats(
        archive_sha256=digest,
        tenant_id=tenant_id,
        source_node_counts=dict(sorted(source_node_counts.items())),
        mapped_node_counts=dict(sorted(mapped_node_counts.items())),
        relationship_counts=dict(sorted(relationship_counts.items())),
        source_nodes=len(selected_nodes),
        mapped_nodes=len(mapped_nodes),
        source_relationships=sum(
            1
            for relationship in source_relationships
            if relationship.source in selected_by_id
            and relationship.target in selected_by_id
        ),
        mapped_relationships=len(mapped_relationships),
        cleaned_names=sum(
            node.name != clean_display_name(node.name) for node in selected_nodes
        ),
        non_informative_symptoms=sum(
            node.primary_label == "TCMSymptom"
            and not bool(node.properties.get("is_informative"))
            for node in mapped_nodes
        ),
    )
    if stats.source_relationships != stats.mapped_relationships:
        raise ValueError(
            "存在未映射关系: "
            f"source={stats.source_relationships}, mapped={stats.mapped_relationships}"
        )
    return DiagnosticAxisDataset(
        nodes=mapped_nodes,
        relationships=mapped_relationships,
        stats=stats,
    )


def _batches(items: list[Any], batch_size: int = BATCH_SIZE) -> Iterable[list[Any]]:
    for offset in range(0, len(items), batch_size):
        yield items[offset:offset + batch_size]


def _node_upsert_cypher(cypher_labels: str) -> str:
    return f"""
    UNWIND $batch AS row
    MERGE (n:{cypher_labels} {{source_db: $source_db, med_id: row.med_id}})
    SET n += row.properties
    """


def _relationship_upsert_cypher(
    source_label: str,
    relationship_type: str,
    target_label: str,
) -> str:
    return f"""
    UNWIND $batch AS row
    MATCH (source:{source_label} {{source_db: $source_db, med_id: row.source_id}})
    MATCH (target:{target_label} {{source_db: $source_db, med_id: row.target_id}})
    MERGE (source)-[relationship:{relationship_type} {{
      source_db: $source_db,
      med_relationship_id: row.med_relationship_id
    }}]->(target)
    SET relationship += row.properties
    """


def write_to_neo4j(
    dataset: DiagnosticAxisDataset,
    database: str = "neo4j",
    *,
    replace_source: bool = False,
) -> bool:
    from app.src.core.graph_db import get_neo4j_graph
    from scripts.tcm_graph_schema import prepare_schema

    if prepare_schema(database) != 0:
        return False
    graph = get_neo4j_graph(database=database)
    if graph is None:
        print(f"[FAIL] Neo4j 未连接（database={database}）")
        return False

    if replace_source:
        rows = graph.query(
            "MATCH (n {source_db: $source_db}) DETACH DELETE n RETURN count(n) AS count",
            params={"source_db": SOURCE_DB},
        )
        print(f"[REPLACE] deleted={rows[0]['count'] if rows else 0}")

    node_groups: dict[str, list[MappedNode]] = {}
    for node in dataset.nodes:
        node_groups.setdefault(node.cypher_labels, []).append(node)
    for cypher_labels, nodes in sorted(node_groups.items()):
        written = 0
        for batch in _batches(nodes):
            graph.query(
                _node_upsert_cypher(cypher_labels),
                params={
                    "source_db": SOURCE_DB,
                    "batch": [node.model_dump() for node in batch],
                },
            )
            written += len(batch)
        print(f"  [NODE] {cypher_labels}: {written}")

    relationship_groups: dict[tuple[str, str, str], list[MappedRelationship]] = {}
    for relationship in dataset.relationships:
        key = (
            relationship.source_label,
            relationship.relationship_type,
            relationship.target_label,
        )
        relationship_groups.setdefault(key, []).append(relationship)
    for (source_label, relationship_type, target_label), relationships in sorted(
        relationship_groups.items()
    ):
        written = 0
        for batch in _batches(relationships, batch_size=1000):
            graph.query(
                _relationship_upsert_cypher(
                    source_label,
                    relationship_type,
                    target_label,
                ),
                params={
                    "source_db": SOURCE_DB,
                    "batch": [relationship.model_dump() for relationship in batch],
                },
            )
            written += len(batch)
        print(
            f"  [REL] {source_label}-[{relationship_type}]->{target_label}: {written}"
        )
    return True


def print_stats(stats: DiagnosticAxisStats) -> None:
    print(f"[ARCHIVE] sha256={stats.archive_sha256}")
    print(f"[TENANT] {stats.tenant_id}")
    print("[SOURCE_NODES]")
    for label, count in stats.source_node_counts.items():
        print(f"  {label:24s}: {count:5d}")
    print("[MAPPED_NODES]")
    for label, count in stats.mapped_node_counts.items():
        print(f"  {label:24s}: {count:5d}")
    print("[MAPPED_RELATIONSHIPS]")
    for relationship_type, count in stats.relationship_counts.items():
        print(f"  {relationship_type:42s}: {count:5d}")
    print(
        "[QUALITY] "
        f"cleaned_names={stats.cleaned_names}, "
        f"non_informative_symptoms={stats.non_informative_symptoms}"
    )
    print(
        "[TOTALS] "
        f"nodes={stats.mapped_nodes}, relationships={stats.mapped_relationships}"
    )


def main(
    archive_path: Path,
    tenant_id: str,
    database: str,
    *,
    dry_run: bool,
    replace_source: bool,
    verify_hash: bool,
) -> int:
    try:
        dataset = load_diagnostic_axis_dataset(
            archive_path,
            tenant_id,
            verify_hash=verify_hash,
        )
    except Exception as exc:
        print(f"MED_TCM_IMPORT=FAIL\nERROR={exc}")
        return 1
    print_stats(dataset.stats)
    if dry_run:
        print("MED_TCM_DRY_RUN=PASS")
        return 0
    if not write_to_neo4j(dataset, database, replace_source=replace_source):
        print("MED_TCM_IMPORT=FAIL")
        return 1
    from scripts.audit_med_tcm_diagnostic_axis import audit_database, print_audit_report

    report = audit_database(database)
    print_audit_report(report)
    if not report.passed:
        print("MED_TCM_IMPORT=FAIL")
        return 1
    print("MED_TCM_IMPORT=PASS")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--tenant", default=DEFAULT_TENANT_ID)
    parser.add_argument("--db", default="neo4j", choices=["neo4j", "tcm_graph"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--replace-source",
        action="store_true",
        help="导入前删除 source_db=med_tcm 的旧节点和关系",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="允许导入与已审核 SHA256 不一致的新归档",
    )
    args = parser.parse_args()
    raise SystemExit(main(
        args.archive,
        args.tenant,
        args.db,
        dry_run=args.dry_run,
        replace_source=args.replace_source,
        verify_hash=not args.skip_hash_check,
    ))
