"""ITCM 中医药数据库节点导入 (P2 阶段)

ITCM（Integrative Traditional Chinese Medicine）公开数据，灌入 Neo4j。

节点（5 类，源 TSV detail 文件）：
  - Herb        (herb_detail.txt,        ~8.5K 行)  NID + 中/英/拉丁名 + 性味归经
  - Formula     (formula_detail.txt,     ~26K 行)   方剂名 + 功效 + 主治
  - Ingredient  (ingredient_detail.txt,  ~43K 行)   成分名 + CAS/PubChem/SMILES
  - Target      (target_detail.txt,      ~19K 行)   Gene Symbol + NCBI/UniProt
  - Disease     (disease_detail.txt,     ~11K 行)   diseaseId + OMIM/Orphanet

依据：``判断.md`` §3.4 P2 范围 + 2026-06-08-p0.5-import-execution.md §3.4
执行::

    cd D:/AI/project/RenShu-AI/backend
    .venv/Scripts/python.exe -m scripts.import_itcm --db neo4j
    .venv/Scripts/python.exe -m scripts.import_itcm --db neo4j --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.tcm_dataset_config import get_tcm_dataset_root

ITCM_DIR = get_tcm_dataset_root() / "ITCM"
BATCH_SIZE = 1000


@dataclass
class NodeTypeSpec:
    label: str
    file_path: Path
    pk_col: str
    field_map: dict[str, str]  # {tsv_col -> neo4j_prop}
    sep: str = "\t"
    skip_blank_pk: bool = True


@dataclass
class Stats:
    node_counts: dict[str, int] = field(default_factory=dict)
    skipped_blank: dict[str, int] = field(default_factory=dict)


# ===== 各类型规格定义 =====

HERB_SPEC = NodeTypeSpec(
    label="Herb",
    file_path=ITCM_DIR / "herb_detail.txt",
    pk_col="NID",
    field_map={
        "NID": "nid",
        "CHN": "name_zh",
        "PINYIN": "pinyin",
        "LATIN": "latin",
        "English_Name": "name_en",
        "ETCM": "etcm_id",
        "TCMID": "tcmid_id",
        "SYM": "sym_id",
        "TCMSP": "tcmsp_id",
        "FAMILY_CHN": "family_zh",
        "FAMILY_ENG": "family_en",
        "Property_CHN": "property_zh",
        "Property_Eng": "property_en",
        "Flavor_CHN": "flavor_zh",
        "Flavor_ENG": "flavor_en",
        "Meridian_Tropism_CHN": "meridian_zh",
        "Meridian_Tropism_ENG": "meridian_en",
        "Indication_CHN": "indication_zh",
        "Indication_ENG": "indication_en",
        "Class_Chinese": "class_zh",
        "UsePart": "use_part",
        "Syndromes": "syndromes",
        "TYPE": "type",
    },
)

FORMULA_SPEC = NodeTypeSpec(
    label="Formula",
    file_path=ITCM_DIR / "formula_detail.txt",
    pk_col="NID",
    field_map={
        "NID": "nid",
        "id": "external_id",
        "CHN": "name_zh",
        "PY": "pinyin",
        "source": "source",
        "DOSAGE_FORM": "dosage_form",
        "Administration": "administration",
        "TYPE": "type",
        "effect_CHN": "effect_zh",
        "effect_ENG": "effect_en",
        "Indications_CHN": "indications_zh",
        "Indications_ENG": "indications_en",
        "dose": "dose",
        "procedure": "procedure",
        "reference": "reference",
        "nation_CHN": "nation_zh",
        "nation_ENG": "nation_en",
        "is_china_pharmacopoeia_2020": "is_pharmacopoeia_2020",
        "is_Ethnic": "is_ethnic",
        "is_covid19": "is_covid19",
        "is_kampo": "is_kampo",
    },
)

INGREDIENT_SPEC = NodeTypeSpec(
    label="Ingredient",
    file_path=ITCM_DIR / "ingredient_detail.txt",
    pk_col="NID",
    field_map={
        "NID": "nid",
        "name": "name",
        "CAS": "cas_id",
        "PUBCHEM_CID": "pubchem_cid",
        "CHEMBL": "chembl_id",
        "REFERENCE": "reference",
        "iSMILES": "ismiles",
        "cSMILES": "csmiles",
        "aromatic_ring_count": "aromatic_ring_count",
        "SymMap_id": "symmap_id",
        "ETCM_id": "etcm_id",
        "TCMID_id": "tcmid_id",
        "TCMSP_id": "tcmsp_id",
        "ITCM": "itcm_id",
    },
)

TARGET_SPEC = NodeTypeSpec(
    label="Target",
    file_path=ITCM_DIR / "target_detail.txt",
    pk_col="NID",
    field_map={
        "NID": "nid",
        "gene_symbol": "gene_symbol",
        "ETCM": "etcm_id",
        "TCMID": "tcmid_id",
        "SYM": "sym_id",
        "TCMSP": "tcmsp_id",
        "UniProtKB": "uniprot_id",
        "Gene_name": "gene_name",
        "Protein_name": "protein_name",
        "Chromosome": "chromosome",
        "HIT_id": "hit_id",
        "Esembl_id": "ensembl_id",  # 注：ITCM 文件里这个字段名是 Esembl（拼写错误）
        "NCBI_id": "ncbi_id",
        "HGNC_id": "hgnc_id",
        "GenBank_Gene_id": "genbank_gene_id",
        "GenBank_Protein_id": "genbank_protein_id",
        "PDB_id": "pdb_id",
        "OMIM_id": "omim_id",
        "miRBase_id": "mirbase_id",
        "IMGT_GENE_DB_id": "imgt_id",
        "GenAtlas_ID": "genatlas_id",
        "Species": "species",
        "drugbank_ID": "drugbank_id",
        "ALIAS": "alias",
    },
)

DISEASE_SPEC = NodeTypeSpec(
    label="Disease",
    file_path=ITCM_DIR / "disease_detail.txt",
    pk_col="NID",
    field_map={
        "NID": "nid",
        "diseaseId": "disease_id",
        "diseaseName": "name",
        "ETCM": "etcm_id",
        "SYM": "sym_id",
        "TCMSP": "tcmsp_id",
        "disaease_definition": "definition",  # 注：文件里是 disaease（拼写错误）
        "MeSH_id": "mesh_id",
        "OMIM_id": "omim_id",
        "Orphanet_id": "orphanet_id",
        "ICD10CM_id": "icd10cm_id",
        "UMLS_id": "umls_id",
        "MedDRA_id": "meddra_id",
    },
)

ALL_SPECS: list[NodeTypeSpec] = [
    HERB_SPEC, FORMULA_SPEC, INGREDIENT_SPEC, TARGET_SPEC, DISEASE_SPEC,
]


# ===== 解析 =====

def _project_row(headers: list[str], values: list[str], spec: NodeTypeSpec) -> dict | None:
    """按 field_map 提取并重命名；PK 为空返回 None"""
    d = dict(zip(headers, values))
    pk_val = d.get(spec.pk_col)
    if not pk_val or pk_val.strip() == "":
        return None
    out = {}
    for src, dst in spec.field_map.items():
        v = d.get(src)
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        out[dst] = v
    if spec.label == "Target" and out.get("ncbi_id"):
        out["ncbi_gene_id"] = f"NCBI:{str(out['ncbi_id']).strip()}"
    out["source_db"] = "ITCM"
    return out


def iter_spec_rows(spec: NodeTypeSpec) -> Iterable[dict]:
    with open(spec.file_path, "r", encoding="utf-8") as f:
        headers = f.readline().rstrip("\n").split(spec.sep)
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            values = line.split(spec.sep)
            if len(values) < len(headers):
                values += [""] * (len(headers) - len(values))
            elif len(values) > len(headers):
                values = values[:len(headers)]
            row = _project_row(headers, values, spec)
            if row is None:
                yield {"__blank_pk__": True}
            else:
                yield row


def collect_stats() -> Stats:
    s = Stats()
    print("[STATS] ITCM 节点规模探查 ...")
    for spec in ALL_SPECS:
        if not spec.file_path.exists():
            print(f"  [WARN] {spec.label}: 源文件不存在 {spec.file_path}")
            s.node_counts[spec.label] = 0
            continue
        n_kept = 0
        n_skip = 0
        for row in iter_spec_rows(spec):
            if "__blank_pk__" in row:
                n_skip += 1
            else:
                n_kept += 1
        s.node_counts[spec.label] = n_kept
        s.skipped_blank[spec.label] = n_skip
        print(f"  {spec.label:14s}: {n_kept:7d} (跳过空 PK={n_skip})")
    return s


# ===== 写入 =====

def _cypher_upsert(spec: NodeTypeSpec) -> str:
    props = list(spec.field_map.values())
    pk_prop = props[0]
    cypher = f"""
    UNWIND $batch AS row
    MERGE (n:{spec.label} {{{pk_prop}: row.{pk_prop}}})
    SET n += row
    """
    return cypher.strip()


def write_to_neo4j(database: str) -> bool:
    try:
        from app.src.core.graph_db import get_neo4j_graph
    except ImportError as exc:
        print(f"[FAIL] 无法导入 graph_db: {exc}")
        return False

    os.environ["NEO4J_DB"] = database
    g = get_neo4j_graph()
    if g is None:
        print(f"[FAIL] Neo4j 未连接（database={database}）")
        return False

    total_start = time.monotonic()
    for spec in ALL_SPECS:
        if not spec.file_path.exists():
            continue
        cypher = _cypher_upsert(spec)
        batch: list[dict] = []
        t0 = time.monotonic()
        written = 0
        for row in iter_spec_rows(spec):
            if "__blank_pk__" in row:
                continue
            batch.append(row)
            if len(batch) >= BATCH_SIZE:
                g.query(cypher, params={"batch": batch})
                written += len(batch)
                batch = []
        if batch:
            g.query(cypher, params={"batch": batch})
            written += len(batch)
        dt = time.monotonic() - t0
        print(f"  [WRITE] {spec.label:14s}: {written:7d} nodes ({dt:.1f}s)")
    print(f"[DONE] 全部写入完成，总耗时 {time.monotonic() - total_start:.1f}s")
    return True


def main(dry_run: bool = False, database: str = "neo4j") -> int:
    missing = [s.label for s in ALL_SPECS if not s.file_path.exists()]
    if missing:
        print(f"[FAIL] 以下节点源文件缺失: {missing}")
        return 1

    s = collect_stats()
    if dry_run:
        print("\n[DRY-RUN] 不写入 Neo4j")
        return 0

    print(f"\n[WRITE] -> Neo4j (database={database}) ...")
    return 0 if write_to_neo4j(database) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只统计不写入")
    parser.add_argument("--db", default="neo4j",
                        choices=["tcm_graph", "neo4j"],
                        help="目标 Neo4j database (默认 neo4j)")
    args = parser.parse_args()
    raise SystemExit(main(dry_run=args.dry_run, database=args.db))
