"""HPOA 外部数据导入 (P0.5 阶段)

解析 HPO 本体 + HPOA 注释文件，灌入 Neo4j 知识图谱：

节点：
  - MMSymptom (id=HPO:xxxxxxx, name, definition, synonyms)
  - Disease   (id=OMIM:xxx / ORPHA:xxx / DECIPHER:xxx, name, source)
  - Target    (id=NCBI:xxx, symbol, name)

关系：
  - (Disease)-[DISEASE_HAS_MM_SYMPTOM {frequency, onset, aspect}]->(MMSymptom)
  - (Target)-[GENE_ASSOCIATED_WITH_PHENOTYPE]->(MMSymptom)
  - (Target)-[GENE_ASSOCIATED_WITH_DISEASE {via_phenotype}]->(Disease)

依据：``docs/external_relation_fetch_plan.md`` 方案 A
执行::

    cd D:/AI/project/RenShu-AI/backend
    .venv/Scripts/python.exe -m scripts.import_hpoa
    .venv/Scripts/python.exe -m scripts.import_hpoa --dry-run
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

EXTERNAL_ROOT = Path("D:/AI/project/RenShu-AI/TCM_Dataset/_external")
HPO_DIR = EXTERNAL_ROOT / "HPO"

BATCH_SIZE = 1000


@dataclass
class HPOTerm:
    hpo_id: str
    name: str
    definition: str = ""
    synonyms: list[str] = field(default_factory=list)


def parse_hp_obo(path: Path) -> dict[str, HPOTerm]:
    terms: dict[str, HPOTerm] = {}
    cur: HPOTerm | None = None
    in_term = False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if line.startswith("["):
                if in_term and cur:
                    terms[cur.hpo_id] = cur
                in_term = line == "[Term]"
                cur = HPOTerm(hpo_id="", name="") if in_term else None
                continue
            if not in_term or cur is None:
                continue
            if line.startswith("id: "):
                cur.hpo_id = line[4:].strip()
            elif line.startswith("name: "):
                cur.name = line[6:].strip()
            elif line.startswith("def: "):
                rest = line[5:]
                cur.definition = rest.split('"')[1] if '"' in rest else rest.strip()
            elif line.startswith("synonym: "):
                m = re.search(r'"([^"]+)"', line)
                if m:
                    cur.synonyms.append(m.group(1))
    if in_term and cur and cur.hpo_id:
        terms[cur.hpo_id] = cur
    return terms


def parse_phenotype_hpoa(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 12 or parts[0] == "database_id":
                continue
            yield {
                "database_id": parts[0],
                "disease_name": parts[1],
                "qualifier": parts[2],
                "hpo_id": parts[3],
                "onset": parts[6],
                "frequency": parts[7],
                "aspect": parts[10],
            }


def parse_genes_to_phenotype(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        header = None
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            if len(parts) < 6:
                continue
            yield {
                "ncbi_gene_id": parts[0],
                "gene_symbol": parts[1],
                "hpo_id": parts[2],
                "hpo_name": parts[3],
                "frequency": parts[4],
                "disease_id": parts[5] if parts[5] != "-" else "",
            }


@dataclass
class Stats:
    hpo_terms: int = 0
    hpoa_relations: int = 0
    hpoa_qualifier_not: int = 0
    gene_relations: int = 0
    unique_diseases: set = field(default_factory=set)
    unique_phenotypes: set = field(default_factory=set)
    unique_genes: set = field(default_factory=set)
    disease_bridge: dict = field(default_factory=lambda: defaultdict(int))


def _split_db_id(database_id: str) -> tuple[str, str]:
    if ":" in database_id:
        src, local = database_id.split(":", 1)
        return src, local
    return "UNKNOWN", database_id


def collect_stats(hpoa_path: Path, genes_path: Path, hpo_terms: dict[str, HPOTerm]) -> Stats:
    s = Stats()
    s.hpo_terms = len(hpo_terms)
    for rel in parse_phenotype_hpoa(hpoa_path):
        src, _ = _split_db_id(rel["database_id"])
        s.disease_bridge[src] += 1
        s.unique_diseases.add(rel["database_id"])
        s.unique_phenotypes.add(rel["hpo_id"])
        if rel["qualifier"] == "NOT":
            s.hpoa_qualifier_not += 1
        else:
            s.hpoa_relations += 1
    for rel in parse_genes_to_phenotype(genes_path):
        s.unique_genes.add(rel["gene_symbol"])
        if rel["disease_id"]:
            s.unique_diseases.add(rel["disease_id"])
        s.unique_phenotypes.add(rel["hpo_id"])
        s.gene_relations += 1
    return s


def _write_hpoa_rels(g, batch: list[dict]) -> None:
    g.query("""
    UNWIND $batch AS r
    MERGE (d:Disease {id: r.database_id})
      ON CREATE SET d.name = r.disease_name, d.source_db = split(r.database_id, ':')[0]
    MERGE (m:MMSymptom {hpo_id: r.hpo_id})
    MERGE (d)-[rel:DISEASE_HAS_MM_SYMPTOM]->(m)
      ON CREATE SET rel.frequency = r.frequency, rel.onset = r.onset, rel.aspect = r.aspect
    """, params={"batch": batch})


def _write_gene_rels(g, batch: list[dict]) -> None:
    g.query("""
    UNWIND $batch AS r
    MERGE (t:Target {ncbi_gene_id: r.ncbi_gene_id})
      ON CREATE SET t.symbol = r.gene_symbol
    MERGE (m:MMSymptom {hpo_id: r.hpo_id})
    MERGE (t)-[:GENE_ASSOCIATED_WITH_PHENOTYPE]->(m)
    WITH t, r
    WHERE r.disease_id <> ''
    MERGE (d:Disease {id: r.disease_id})
    MERGE (t)-[gd:GENE_ASSOCIATED_WITH_DISEASE]->(d)
      ON CREATE SET gd.via_phenotype = r.hpo_id
    """, params={"batch": batch})


def write_to_neo4j(hpo_terms: dict[str, HPOTerm], hpoa_path: Path, genes_path: Path) -> bool:
    try:
        from app.src.core.graph_db import get_neo4j_graph
    except ImportError as exc:
        print(f"[FAIL] 无法导入 graph_db: {exc}")
        return False

    g = get_neo4j_graph()
    if g is None:
        print("[FAIL] Neo4j 未连接，请启动 Neo4j 后重试。可用 --dry-run 仅统计。")
        return False

    t0 = time.monotonic()
    print("[WRITE] MMSymptom 节点 (from hp.obo) ...")
    batch: list[dict] = []
    written = 0
    for term in hpo_terms.values():
        if not term.hpo_id.startswith("HP:"):
            continue
        batch.append({
            "hpo_id": term.hpo_id,
            "name": term.name,
            "definition": term.definition[:500] if term.definition else "",
            "synonyms": term.synonyms[:5],
        })
        if len(batch) >= BATCH_SIZE:
            g.query("""
            UNWIND $batch AS t
            MERGE (m:MMSymptom {hpo_id: t.hpo_id})
            SET m.name = t.name, m.definition = t.definition, m.synonyms = t.synonyms
            """, params={"batch": batch})
            written += len(batch)
            batch = []
    if batch:
        g.query("""
        UNWIND $batch AS t
        MERGE (m:MMSymptom {hpo_id: t.hpo_id})
        SET m.name = t.name, m.definition = t.definition, m.synonyms = t.synonyms
        """, params={"batch": batch})
        written += len(batch)
    print(f"  -> wrote/updated {written} MMSymptom in {time.monotonic() - t0:.1f}s")

    t0 = time.monotonic()
    print("[WRITE] DISEASE_HAS_MM_SYMPTOM 关系 (from phenotype.hpoa) ...")
    batch, written = [], 0
    for rel in parse_phenotype_hpoa(hpoa_path):
        if rel["qualifier"] == "NOT":
            continue
        batch.append({
            "database_id": rel["database_id"],
            "disease_name": rel["disease_name"][:200],
            "hpo_id": rel["hpo_id"],
            "frequency": rel["frequency"],
            "onset": rel["onset"],
            "aspect": rel["aspect"],
        })
        if len(batch) >= BATCH_SIZE:
            _write_hpoa_rels(g, batch)
            written += len(batch)
            batch = []
    if batch:
        _write_hpoa_rels(g, batch)
        written += len(batch)
    print(f"  -> wrote {written} relations in {time.monotonic() - t0:.1f}s")

    t0 = time.monotonic()
    print("[WRITE] Target + GENE_ASSOCIATED_WITH_* 关系 (from genes_to_phenotype.txt) ...")
    batch, written = [], 0
    for rel in parse_genes_to_phenotype(genes_path):
        batch.append({
            "ncbi_gene_id": f"NCBI:{rel['ncbi_gene_id']}",
            "gene_symbol": rel["gene_symbol"],
            "hpo_id": rel["hpo_id"],
            "disease_id": rel["disease_id"],
        })
        if len(batch) >= BATCH_SIZE:
            _write_gene_rels(g, batch)
            written += len(batch)
            batch = []
    if batch:
        _write_gene_rels(g, batch)
        written += len(batch)
    print(f"  -> wrote {written} relations in {time.monotonic() - t0:.1f}s")
    return True


def main(dry_run: bool = False) -> int:
    hp_obo = HPO_DIR / "hp.obo"
    hpoa = HPO_DIR / "phenotype.hpoa"
    genes = HPO_DIR / "genes_to_phenotype.txt"

    for p in [hp_obo, hpoa, genes]:
        if not p.exists():
            print(f"[FAIL] 缺失文件: {p}")
            return 1

    print(f"[PARSE] hp.obo -> HPOTerm dict ...")
    t0 = time.monotonic()
    hpo_terms = parse_hp_obo(hp_obo)
    print(f"  -> {len(hpo_terms)} terms in {time.monotonic() - t0:.1f}s")

    print(f"[STATS] phenotype.hpoa + genes_to_phenotype.txt ...")
    stats = collect_stats(hpoa, genes, hpo_terms)
    print(f"  HPOA 关系（含 qualifier=NOT）: {stats.hpoa_relations + stats.hpoa_qualifier_not}")
    print(f"    其中 qualifier=NOT（不灌）: {stats.hpoa_qualifier_not}")
    print(f"    实际可灌: {stats.hpoa_relations}")
    print(f"  Gene 关系: {stats.gene_relations}")
    print(f"  唯一疾病: {len(stats.unique_diseases)}  唯一表型: {len(stats.unique_phenotypes)}  唯一基因: {len(stats.unique_genes)}")
    print(f"  Disease 来源分布: {dict(stats.disease_bridge)}")

    if dry_run:
        print("\n[DRY-RUN] 不写入 Neo4j")
        return 0

    print(f"\n[WRITE] -> Neo4j ...")
    ok = write_to_neo4j(hpo_terms, hpoa, genes)
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="仅解析+统计，不写入 Neo4j")
    args = parser.parse_args()
    raise SystemExit(main(dry_run=args.dry_run))
