"""SIDER 药物-副作用数据导入 (P0.5 阶段，旁路可选)

SIDER 4.1 公开数据集：drugs + side effects + MedDRA 编码。

节点：
  - Drug       (id=CID_flat, name, atc)
  - SideEffect (id=MedDRA:UMLS, name)

关系：
  - (Drug)-[HAS_SIDE_EFFECT]->(SideEffect)

依据：``docs/external_relation_fetch_plan.md`` 方案 A 旁路选项
执行::

    cd D:/AI/project/RenShu-AI/backend
    .venv/Scripts/python.exe -m scripts.import_sider
    .venv/Scripts/python.exe -m scripts.import_sider --dry-run
"""
from __future__ import annotations

import argparse
import gzip
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

SIDER_DIR = Path("D:/AI/project/RenShu-AI/TCM_Dataset/_external/SIDER")
BATCH_SIZE = 1000


@dataclass
class Stats:
    drugs: int = 0
    side_effects: int = 0
    se_relations: int = 0
    unique_drugs: set = field(default_factory=set)
    unique_se: set = field(default_factory=set)
    atc_drugs: int = 0


def load_drug_names(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                out[parts[0]] = parts[1]
    return out


def load_drug_atc(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                out[parts[0]].append(parts[1])
    return out


def iter_se_relations(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            yield {
                "stitch_flat": parts[0],
                "umls_id": parts[2],
                "meddra_type": parts[3],
                "meddra_id": parts[4],
                "side_effect": parts[5],
            }


def collect_stats() -> Stats:
    s = Stats()
    drug_names = load_drug_names(SIDER_DIR / "drug_names.tsv")
    drug_atc = load_drug_atc(SIDER_DIR / "drug_atc.tsv")
    s.drugs = len(drug_names)
    s.atc_drugs = len(drug_atc)
    for rel in iter_se_relations(SIDER_DIR / "meddra_all_se.tsv.gz"):
        s.unique_drugs.add(rel["stitch_flat"])
        s.unique_se.add(rel["umls_id"])
        s.se_relations += 1
    s.side_effects = len(s.unique_se)
    return s


def write_to_neo4j() -> bool:
    try:
        from app.src.core.graph_db import get_neo4j_graph
    except ImportError as exc:
        print(f"[FAIL] 无法导入 graph_db: {exc}")
        return False
    g = get_neo4j_graph()
    if g is None:
        print("[FAIL] Neo4j 未连接")
        return False

    t0 = time.monotonic()
    print("[WRITE] Drug 节点 (drug_names + drug_atc) ...")
    drug_names = load_drug_names(SIDER_DIR / "drug_names.tsv")
    drug_atc = load_drug_atc(SIDER_DIR / "drug_atc.tsv")
    batch = [
        {"stitch_id": cid, "name": name, "atc": drug_atc.get(cid, [])}
        for cid, name in drug_names.items()
    ]
    written = 0
    for i in range(0, len(batch), BATCH_SIZE):
        chunk = batch[i:i + BATCH_SIZE]
        g.query("""
        UNWIND $batch AS d
        MERGE (dr:Drug {stitch_id: d.stitch_id})
        SET dr.name = d.name, dr.atc = d.atc
        """, params={"batch": chunk})
        written += len(chunk)
    print(f"  -> wrote/updated {written} Drug in {time.monotonic() - t0:.1f}s")

    t0 = time.monotonic()
    print("[WRITE] SideEffect + HAS_SIDE_EFFECT 关系 ...")
    se_buffer: dict[str, str] = {}
    rel_buffer: list[dict] = []
    written_rel = 0
    written_se = 0
    for rel in iter_se_relations(SIDER_DIR / "meddra_all_se.tsv.gz"):
        if rel["umls_id"] not in se_buffer:
            se_buffer[rel["umls_id"]] = rel["side_effect"]
        rel_buffer.append({"stitch_id": rel["stitch_flat"], "umls_id": rel["umls_id"]})
        if len(rel_buffer) >= BATCH_SIZE * 10:
            g.query("""
            UNWIND $se AS s
            MERGE (se:SideEffect {umls_id: s.umls_id})
            SET se.name = s.name
            """, params={"se": [{"umls_id": k, "name": v} for k, v in se_buffer.items()]})
            written_se += len(se_buffer)
            se_buffer.clear()
            g.query("""
            UNWIND $batch AS r
            MATCH (dr:Drug {stitch_id: r.stitch_id})
            MATCH (se:SideEffect {umls_id: r.umls_id})
            MERGE (dr)-[rel:HAS_SIDE_EFFECT]->(se)
            """, params={"batch": rel_buffer})
            written_rel += len(rel_buffer)
            rel_buffer = []
    if se_buffer:
        g.query("""
        UNWIND $se AS s
        MERGE (se:SideEffect {umls_id: s.umls_id})
        SET se.name = s.name
        """, params={"se": [{"umls_id": k, "name": v} for k, v in se_buffer.items()]})
        written_se += len(se_buffer)
    if rel_buffer:
        g.query("""
        UNWIND $batch AS r
        MATCH (dr:Drug {stitch_id: r.stitch_id})
        MATCH (se:SideEffect {umls_id: r.umls_id})
        MERGE (dr)-[rel:HAS_SIDE_EFFECT]->(se)
        """, params={"batch": rel_buffer})
        written_rel += len(rel_buffer)
    print(f"  -> wrote {written_se} SideEffect, {written_rel} HAS_SIDE_EFFECT in {time.monotonic() - t0:.1f}s")
    return True


def main(dry_run: bool = False) -> int:
    for name in ["drug_names.tsv", "drug_atc.tsv", "meddra_all_se.tsv.gz"]:
        p = SIDER_DIR / name
        if not p.exists():
            print(f"[FAIL] 缺失文件: {p}")
            return 1
    if not (SIDER_DIR / "meddra_all_label_se.tsv.gz").exists():
        print(f"[WARN] 缺失 meddra_all_label_se.tsv.gz（之前下载失败），继续用 meddra_all_se.tsv.gz")

    print(f"[STATS] SIDER 4.1 ...")
    t0 = time.monotonic()
    s = collect_stats()
    print(f"  Drug 总数: {s.drugs} (有 ATC 的: {s.atc_drugs})")
    print(f"  SideEffect 唯一: {s.side_effects}")
    print(f"  HAS_SIDE_EFFECT 关系（未去重 PT+LLT）: {s.se_relations}")
    print(f"  -> 涉及唯一 Drug: {len(s.unique_drugs)}, 唯一 SideEffect: {len(s.unique_se)}")
    print(f"  解析耗时: {time.monotonic() - t0:.1f}s")

    if dry_run:
        print("\n[DRY-RUN] 不写入 Neo4j")
        return 0

    print(f"\n[WRITE] -> Neo4j ...")
    ok = write_to_neo4j()
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(dry_run=args.dry_run))
