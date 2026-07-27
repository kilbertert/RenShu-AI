"""统一 TCM Neo4j 标签、关系类型和索引。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT.parent / ".env", encoding="utf-8", override=False)


CORE_NODE_LABELS = (
    "Formula",
    "Herb",
    "Ingredient",
    "Target",
    "Disease",
    "TCMSymptom",
    "MMSymptom",
    "Syndrome",
)

DIAGNOSTIC_NODE_LABELS = (
    "TCMDisease",
    "Constitution",
    "TCMDiseaseCategory",
    "MainSymptom",
    "SupplementSymptom",
    "TongueSymptom",
    "PulseSymptom",
    "ClinicalSymptom",
)

NODE_LABELS = CORE_NODE_LABELS + DIAGNOSTIC_NODE_LABELS

CORE_RELATIONSHIP_TYPES = (
    "FORMULA_CONTAINS_HERB",
    "HERB_CONTAINS_INGREDIENT",
    "INGREDIENT_TARGETS",
    "TARGET_ASSOCIATED_WITH_DISEASE",
    "DISEASE_HAS_MM_SYMPTOM",
    "TCM_SYMPTOM_MAPS_TO_MM_SYMPTOM",
    "SYNDROME_ASSOCIATED_WITH_TCM_SYMPTOM",
)

DIAGNOSTIC_RELATIONSHIP_TYPES = (
    "SYNDROME_PATTERN_OF_TCM_DISEASE",
    "SYNDROME_ASSOCIATED_WITH_CONSTITUTION",
    "TCM_DISEASE_IN_CATEGORY",
    "TCM_DISEASE_HAS_TCM_SYMPTOM",
)

RELATIONSHIP_TYPES = CORE_RELATIONSHIP_TYPES + DIAGNOSTIC_RELATIONSHIP_TYPES


class Neo4jIndexSpec(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    label: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    property: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")

    def cypher(self) -> str:
        return (
            f"CREATE INDEX {self.name} IF NOT EXISTS "
            f"FOR (n:{self.label}) ON (n.{self.property})"
        )


class Neo4jConstraintSpec(BaseModel):
    """来源级节点身份约束，避免幂等导入产生重复实体。"""

    name: str = Field(pattern=r"^[a-z0-9_]+$")
    label: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    properties: tuple[str, ...]

    def cypher(self) -> str:
        properties = ", ".join(f"n.{name}" for name in self.properties)
        return (
            f"CREATE CONSTRAINT {self.name} IF NOT EXISTS "
            f"FOR (n:{self.label}) REQUIRE ({properties}) IS UNIQUE"
        )


CONSTRAINT_SPECS = (
    Neo4jConstraintSpec(
        name="uq_syndrome_source_med_id",
        label="Syndrome",
        properties=("source_db", "med_id"),
    ),
    Neo4jConstraintSpec(
        name="uq_tcmsymptom_source_med_id",
        label="TCMSymptom",
        properties=("source_db", "med_id"),
    ),
    Neo4jConstraintSpec(
        name="uq_tcmdisease_source_med_id",
        label="TCMDisease",
        properties=("source_db", "med_id"),
    ),
    Neo4jConstraintSpec(
        name="uq_constitution_source_med_id",
        label="Constitution",
        properties=("source_db", "med_id"),
    ),
    Neo4jConstraintSpec(
        name="uq_tcm_category_source_med_id",
        label="TCMDiseaseCategory",
        properties=("source_db", "med_id"),
    ),
)


INDEX_SPECS = (
    Neo4jIndexSpec(name="idx_formula_nid", label="Formula", property="nid"),
    Neo4jIndexSpec(name="idx_formula_name_zh", label="Formula", property="name_zh"),
    Neo4jIndexSpec(name="idx_herb_id", label="Herb", property="id"),
    Neo4jIndexSpec(name="idx_herb_nid", label="Herb", property="nid"),
    Neo4jIndexSpec(name="idx_herb_tcmbank_id", label="Herb", property="tcmbank_id"),
    Neo4jIndexSpec(name="idx_herb_name_zh", label="Herb", property="name_zh"),
    Neo4jIndexSpec(name="idx_herb_chinese_name", label="Herb", property="chinese_name"),
    Neo4jIndexSpec(name="idx_ingredient_id", label="Ingredient", property="id"),
    Neo4jIndexSpec(name="idx_ingredient_nid", label="Ingredient", property="nid"),
    Neo4jIndexSpec(name="idx_ingredient_tcmbank_id", label="Ingredient", property="tcmbank_id"),
    Neo4jIndexSpec(name="idx_ingredient_name", label="Ingredient", property="name"),
    Neo4jIndexSpec(name="idx_ingredient_pubchem", label="Ingredient", property="pubchem_cid"),
    Neo4jIndexSpec(name="idx_target_nid", label="Target", property="nid"),
    Neo4jIndexSpec(name="idx_target_symmap_id", label="Target", property="symmap_id"),
    Neo4jIndexSpec(name="idx_target_tcmbank_id", label="Target", property="tcmbank_id"),
    Neo4jIndexSpec(name="idx_target_gene_symbol", label="Target", property="gene_symbol"),
    Neo4jIndexSpec(name="idx_target_ncbi", label="Target", property="ncbi_id"),
    Neo4jIndexSpec(name="idx_target_hpo_ncbi", label="Target", property="ncbi_gene_id"),
    Neo4jIndexSpec(name="idx_disease_nid", label="Disease", property="nid"),
    Neo4jIndexSpec(name="idx_disease_id", label="Disease", property="id"),
    Neo4jIndexSpec(name="idx_disease_symmap_id", label="Disease", property="symmap_id"),
    Neo4jIndexSpec(name="idx_disease_tcmbank_id", label="Disease", property="tcmbank_id"),
    Neo4jIndexSpec(name="idx_mmsymptom_symmap", label="MMSymptom", property="symmap_id"),
    Neo4jIndexSpec(name="idx_mmsymptom_hpo", label="MMSymptom", property="hpo_id"),
    Neo4jIndexSpec(name="idx_mmsymptom_name", label="MMSymptom", property="name"),
    Neo4jIndexSpec(name="idx_tcmsymptom_id", label="TCMSymptom", property="id"),
    Neo4jIndexSpec(name="idx_tcmsymptom_name", label="TCMSymptom", property="name_zh"),
    Neo4jIndexSpec(name="idx_tcmsymptom_med_id", label="TCMSymptom", property="med_id"),
    Neo4jIndexSpec(name="idx_tcmsymptom_normalized", label="TCMSymptom", property="normalized_name"),
    Neo4jIndexSpec(name="idx_syndrome_id", label="Syndrome", property="id"),
    Neo4jIndexSpec(name="idx_syndrome_name", label="Syndrome", property="name_zh"),
    Neo4jIndexSpec(name="idx_syndrome_med_id", label="Syndrome", property="med_id"),
    Neo4jIndexSpec(name="idx_syndrome_normalized", label="Syndrome", property="normalized_name"),
    Neo4jIndexSpec(name="idx_syndrome_canonical", label="Syndrome", property="canonical_name"),
    Neo4jIndexSpec(name="idx_tcmdisease_med_id", label="TCMDisease", property="med_id"),
    Neo4jIndexSpec(name="idx_tcmdisease_name", label="TCMDisease", property="name_zh"),
    Neo4jIndexSpec(name="idx_tcmdisease_normalized", label="TCMDisease", property="normalized_name"),
    Neo4jIndexSpec(name="idx_constitution_med_id", label="Constitution", property="med_id"),
    Neo4jIndexSpec(name="idx_constitution_name", label="Constitution", property="name_zh"),
    Neo4jIndexSpec(name="idx_tcm_category_med_id", label="TCMDiseaseCategory", property="med_id"),
    Neo4jIndexSpec(name="idx_tcm_category_name", label="TCMDiseaseCategory", property="name_zh"),
)


def prepare_schema(database: str = "neo4j") -> int:
    from app.src.core.graph_db import get_neo4j_graph

    os.environ["NEO4J_DB"] = database
    graph = get_neo4j_graph(database=database)
    if graph is None:
        print(f"[FAIL] Neo4j 未连接（database={database}）")
        return 1
    for spec in CONSTRAINT_SPECS:
        graph.query(spec.cypher())
        print(f"  [CONSTRAINT] {spec.name}")
    for spec in INDEX_SPECS:
        graph.query(spec.cypher())
        print(f"  [INDEX] {spec.name}")
    print(
        f"[DONE] {len(CONSTRAINT_SPECS)} constraints and "
        f"{len(INDEX_SPECS)} indexes ready"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="neo4j", choices=["neo4j", "tcm_graph"])
    args = parser.parse_args()
    raise SystemExit(prepare_schema(args.db))
