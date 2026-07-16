"""TCM 语料导出（P3 Task 2）

将 Neo4j 中已灌库的 TCM 节点导出为 markdown 文本，存到 ``graphrag/data/input/``，
供 GraphRAG 索引使用。

执行::

    cd D:/AI/project/RenShu-AI/backend
    .venv/Scripts/python.exe -m scripts.export_tcm_corpus --db neo4j

依据：``判断.md`` §5.2.2 P3 Task 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# 输出目录：backend/graphrag/data/input/
# 注意：GraphRAG 的 settings.yaml 隐式默认 input/ 目录（相对项目根）；
#       项目根 = backend/graphrag/，所以 input/ 位于 backend/graphrag/data/input/。
DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "graphrag" / "data" / "input"
RECORDS_PER_FILE = 5000  # 单文件最大记录数，避免单文件过大


# ===== 通用工具 =====

def _safe_str(value: Any, max_len: int = 500) -> str:
    """转字符串，截断、去 None"""
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


# ===== 各节点类型的格式化器 =====

def _format_formula(f: dict) -> str:
    """ITCM Formula → markdown"""
    name = _safe_str(f.get("name_zh") or f.get("name") or "（无名方剂）")
    pinyin = _safe_str(f.get("pinyin"))
    source = _safe_str(f.get("source"))
    effect = _safe_str(f.get("effect_zh") or f.get("effect"), 2000)
    indications = _safe_str(f.get("indications_zh") or f.get("indications"), 2000)
    dose = _safe_str(f.get("dose"))
    procedure = _safe_str(f.get("procedure"), 1000)
    ref = _safe_str(f.get("reference"), 500)
    type_ = _safe_str(f.get("type"))
    flags = []
    if f.get("is_pharmacopoeia_2020"):
        flags.append("中国药典2020")
    if f.get("is_covid19"):
        flags.append("新冠")
    if f.get("is_kampo"):
        flags.append("汉方")

    parts = [f"# {name}", ""]
    meta = []
    if pinyin:
        meta.append(f"**拼音**: {pinyin}")
    if source:
        meta.append(f"**出处**: {source}")
    if type_:
        meta.append(f"**类型**: {type_}")
    if flags:
        meta.append(f"**标签**: {', '.join(flags)}")
    if meta:
        parts.append("  \n".join(meta) + "  ")
    if effect:
        parts.append(f"## 功效\n\n{effect}\n")
    if indications:
        parts.append(f"## 主治\n\n{indications}\n")
    if dose:
        parts.append(f"## 用法用量\n\n{dose}\n")
    if procedure:
        parts.append(f"## 制法\n\n{procedure}\n")
    if ref:
        parts.append(f"## 参考\n\n{ref}\n")
    return "\n".join(parts)


def _format_syndrome(s: dict) -> str:
    """SymMap Syndrome → markdown"""
    name = _safe_str(s.get("name_zh") or s.get("name") or "（无名证型）")
    name_en = _safe_str(s.get("name_en"))
    pinyin = _safe_str(s.get("name_pinyin"))
    definition = _safe_str(s.get("definition"), 2000)
    type_ = _safe_str(s.get("term_type"))

    parts = [f"# {name}", ""]
    meta = []
    if name_en:
        meta.append(f"**English**: {name_en}")
    if pinyin:
        meta.append(f"**拼音**: {pinyin}")
    if type_:
        meta.append(f"**类型**: {type_}")
    if meta:
        parts.append("  \n".join(meta) + "  ")
    if definition:
        parts.append(f"## 定义\n\n{definition}\n")
    return "\n".join(parts)


def _format_disease(d: dict) -> str:
    """Disease (SymMap/HPOA) → markdown"""
    name = _safe_str(d.get("name") or "（无名疾病）")
    source_db = _safe_str(d.get("source_db"))
    definition = _safe_str(d.get("definition"), 1500)
    omim = _safe_str(d.get("omim_id"))
    mesh = _safe_str(d.get("mesh_id"))
    orphanet = _safe_str(d.get("orphanet_id"))
    icd10 = _safe_str(d.get("icd10cm_id"))
    umls = _safe_str(d.get("umls_id"))

    parts = [f"# {name}", ""]
    meta = []
    if source_db:
        meta.append(f"**来源**: {source_db}")
    if omim:
        meta.append(f"**OMIM**: {omim}")
    if mesh:
        meta.append(f"**MeSH**: {mesh}")
    if orphanet:
        meta.append(f"**Orphanet**: {orphanet}")
    if icd10:
        meta.append(f"**ICD10**: {icd10}")
    if umls:
        meta.append(f"**UMLS**: {umls}")
    if meta:
        parts.append("  \n".join(meta) + "  ")
    if definition:
        parts.append(f"## 描述\n\n{definition}\n")
    return "\n".join(parts)


def _format_tcm_symptom(ts: dict) -> str:
    """SymMap TCMSymptom → markdown"""
    name = _safe_str(ts.get("name_zh") or ts.get("name") or "（无名症状）")
    pinyin = _safe_str(ts.get("name_pinyin"))
    definition = _safe_str(ts.get("definition"), 1500)
    locus = _safe_str(ts.get("locus"))
    property_ = _safe_str(ts.get("property"))
    type_ = _safe_str(ts.get("term_type"))

    parts = [f"# {name}", ""]
    meta = []
    if pinyin:
        meta.append(f"**拼音**: {pinyin}")
    if locus:
        meta.append(f"**部位**: {locus}")
    if property_:
        meta.append(f"**性质**: {property_}")
    if type_:
        meta.append(f"**类型**: {type_}")
    if meta:
        parts.append("  \n".join(meta) + "  ")
    if definition:
        parts.append(f"## 定义\n\n{definition}\n")
    return "\n".join(parts)


def _format_mm_symptom(m: dict) -> str:
    """MMSymptom (SymMap/HPOA) → markdown"""
    name = _safe_str(m.get("name") or "（无名现代症状）")
    source_db = _safe_str(m.get("source_db"))
    definition = _safe_str(m.get("definition"), 1500)
    hpo = _safe_str(m.get("hpo_id"))
    umls = _safe_str(m.get("umls_id"))
    omim = _safe_str(m.get("omim_id"))
    mesh = _safe_str(m.get("mesh_id"))

    parts = [f"# {name}", ""]
    meta = []
    if source_db:
        meta.append(f"**来源**: {source_db}")
    if hpo:
        meta.append(f"**HPO**: {hpo}")
    if umls:
        meta.append(f"**UMLS**: {umls}")
    if omim:
        meta.append(f"**OMIM**: {omim}")
    if mesh:
        meta.append(f"**MeSH**: {mesh}")
    if meta:
        parts.append("  \n".join(meta) + "  ")
    if definition:
        parts.append(f"## 描述\n\n{definition}\n")
    return "\n".join(parts)


def _format_herb(h: dict) -> str:
    """Herb (SymMap/ITCM) → markdown"""
    name = (
        _safe_str(h.get("chinese_name"))
        or _safe_str(h.get("name_zh"))
        or _safe_str(h.get("name"))
        or "（无名药材）"
    )
    pinyin = _safe_str(h.get("pinyin_name") or h.get("pinyin"))
    latin = _safe_str(h.get("latin_name") or h.get("latin"))
    name_en = _safe_str(h.get("english_name") or h.get("name_en"))
    property_ = _safe_str(h.get("properties_zh") or h.get("property_zh"))
    flavor = _safe_str(h.get("flavor_zh"))
    meridian = _safe_str(h.get("meridians_zh") or h.get("meridian_zh"))
    function = _safe_str(h.get("function"), 2000)
    indication = _safe_str(h.get("indication_zh"), 2000)
    use_part = _safe_str(h.get("use_part"))
    class_zh = _safe_str(h.get("class_zh"))

    parts = [f"# {name}", ""]
    meta = []
    if pinyin:
        meta.append(f"**拼音**: {pinyin}")
    if latin:
        meta.append(f"**拉丁名**: {latin}")
    if name_en:
        meta.append(f"**English**: {name_en}")
    if class_zh:
        meta.append(f"**类别**: {class_zh}")
    if use_part:
        meta.append(f"**入药部位**: {use_part}")
    if meta:
        parts.append("  \n".join(meta) + "  ")

    nature_lines = []
    if property_:
        nature_lines.append(f"**性**: {property_}")
    if flavor:
        nature_lines.append(f"**味**: {flavor}")
    if meridian:
        nature_lines.append(f"**归经**: {meridian}")
    if nature_lines:
        parts.append("  \n".join(nature_lines) + "  ")

    if function:
        parts.append(f"## 功效\n\n{function}\n")
    if indication:
        parts.append(f"## 应用\n\n{indication}\n")
    return "\n".join(parts)


def _format_ingredient(i: dict) -> str:
    """Ingredient (SymMap/ITCM) → markdown"""
    name = (
        _safe_str(i.get("molecule_name"))
        or _safe_str(i.get("name"))
        or "（无名的化学成分）"
    )
    formula = _safe_str(i.get("molecule_formula"))
    weight = _safe_str(i.get("molecule_weight"))
    ob = _safe_str(i.get("ob_score"))
    pubchem = _safe_str(i.get("pubchem_cid") or i.get("PUBCHEM_CID"))
    cas = _safe_str(i.get("cas_id") or i.get("CAS"))
    type_ = _safe_str(i.get("ingredient_type"))

    parts = [f"# {name}", ""]
    meta = []
    if formula:
        meta.append(f"**分子式**: {formula}")
    if weight:
        meta.append(f"**分子量**: {weight}")
    if ob:
        meta.append(f"**OB_score**: {ob}")
    if pubchem:
        meta.append(f"**PubChem**: {pubchem}")
    if cas:
        meta.append(f"**CAS**: {cas}")
    if type_:
        meta.append(f"**类型**: {type_}")
    if meta:
        parts.append("  \n".join(meta) + "  ")
    return "\n".join(parts)


# ===== 节点查询 + 导出 =====

# 注意：cypher 的字段名按 import_symmap_v2.py / import_itcm.py 实际落地属性

NODE_QUERIES = [
    {
        "label": "Formula",
        "filename_prefix": "formulas",
        "format_fn": _format_formula,
        "cypher": """
        MATCH (f:Formula)
        RETURN f.nid AS nid,
               f.name_zh AS name_zh,
               f.pinyin AS pinyin,
               f.source AS source,
               f.effect_zh AS effect_zh,
               f.indications_zh AS indications_zh,
               f.dose AS dose,
               f.procedure AS procedure,
               f.reference AS reference,
               f.type AS type,
               f.is_pharmacopoeia_2020 AS is_pharmacopoeia_2020,
               f.is_covid19 AS is_covid19,
               f.is_kampo AS is_kampo
        ORDER BY f.nid
        """,
    },
    {
        "label": "Syndrome",
        "filename_prefix": "syndromes",
        "format_fn": _format_syndrome,
        "cypher": """
        MATCH (s:Syndrome)
        RETURN s.id AS id,
               s.name_zh AS name_zh,
               s.name_en AS name_en,
               s.name_pinyin AS name_pinyin,
               s.definition AS definition,
               s.term_type AS term_type
        ORDER BY s.id
        """,
    },
    {
        "label": "Disease",
        "filename_prefix": "diseases",
        "format_fn": _format_disease,
        "cypher": """
        MATCH (d:Disease)
        RETURN d.name AS name,
               d.source_db AS source_db,
               d.definition AS definition,
               d.omim_id AS omim_id,
               d.mesh_id AS mesh_id,
               d.orphanet_id AS orphanet_id,
               d.icd10cm_id AS icd10cm_id,
               d.umls_id AS umls_id
        ORDER BY d.name
        """,
    },
    {
        "label": "TCMSymptom",
        "filename_prefix": "tcm_symptoms",
        "format_fn": _format_tcm_symptom,
        "cypher": """
        MATCH (t:TCMSymptom)
        RETURN t.id AS id,
               t.name_zh AS name_zh,
               t.name_pinyin AS name_pinyin,
               t.definition AS definition,
               t.locus AS locus,
               t.property AS property,
               t.term_type AS term_type
        ORDER BY t.id
        """,
    },
    {
        "label": "MMSymptom",
        "filename_prefix": "mm_symptoms",
        "format_fn": _format_mm_symptom,
        "cypher": """
        MATCH (m:MMSymptom)
        RETURN m.name AS name,
               m.source_db AS source_db,
               m.definition AS definition,
               m.hpo_id AS hpo_id,
               m.umls_id AS umls_id,
               m.omim_id AS omim_id,
               m.mesh_id AS mesh_id
        ORDER BY m.name
        """,
    },
    {
        "label": "Herb",
        "filename_prefix": "herbs",
        "format_fn": _format_herb,
        "cypher": """
        MATCH (h:Herb)
        RETURN h.id AS id,
               h.chinese_name AS chinese_name,
               h.pinyin_name AS pinyin_name,
               h.latin_name AS latin_name,
               h.english_name AS english_name,
               h.properties_zh AS properties_zh,
               h.meridians_zh AS meridians_zh,
               h.function AS function,
               h.class_zh AS class_zh,
               h.use_part AS use_part
        ORDER BY h.id
        """,
    },
    {
        "label": "Ingredient",
        "filename_prefix": "ingredients",
        "format_fn": _format_ingredient,
        "cypher": """
        MATCH (i:Ingredient)
        RETURN i.id AS id,
               i.molecule_name AS molecule_name,
               i.molecule_formula AS molecule_formula,
               i.molecule_weight AS molecule_weight,
               i.ob_score AS ob_score,
               i.pubchem_cid AS pubchem_cid,
               i.cas_id AS cas_id,
               i.ingredient_type AS ingredient_type
        ORDER BY i.id
        """,
    },
]


def export_one_label(graph, spec: dict, output_dir: Path) -> dict:
    """导出单个节点类型；按 RECORDS_PER_FILE 切分文件"""
    label = spec["label"]
    prefix = spec["filename_prefix"]
    format_fn = spec["format_fn"]
    cypher = spec["cypher"]

    print(f"  [{label}] 查询中...")
    t0 = time.monotonic()
    rows = graph.query(cypher)
    dt_query = time.monotonic() - t0
    print(f"  [{label}] 查询 {len(rows)} 条 ({dt_query:.1f}s)")

    if not rows:
        return {"label": label, "count": 0, "files": []}

    file_paths = []
    total = len(rows)
    for file_idx, start in enumerate(range(0, total, RECORDS_PER_FILE)):
        chunk = rows[start:start + RECORDS_PER_FILE]
        if file_idx == 0 and total <= RECORDS_PER_FILE:
            fname = f"{prefix}.md"
        else:
            fname = f"{prefix}_{file_idx + 1:02d}.md"
        fpath = output_dir / fname
        t0 = time.monotonic()
        with open(fpath, "w", encoding="utf-8") as f:
            for i, row in enumerate(chunk):
                f.write(format_fn(row))
                f.write("\n\n---\n\n")
                if (i + 1) % 1000 == 0:
                    print(f"    [{label}] 写入 {i+1}/{len(chunk)} -> {fname}")
        dt = time.monotonic() - t0
        print(f"  [{label}] 写入 {fname}: {len(chunk)} 条 ({dt:.1f}s)")
        file_paths.append(fname)

    return {"label": label, "count": total, "files": file_paths}


def cleanup_old_corpus(output_dir: Path) -> int:
    """清理旧的非 TCM 语料（merged_review.csv / technology_companies.txt 等）"""
    removed = []
    backup_dir = output_dir.parent / "input_backup_nontcm"
    backup_dir.mkdir(exist_ok=True)
    for p in output_dir.iterdir():
        if p.is_file() and p.is_file():
            backup_path = backup_dir / p.name
            if not backup_path.exists():
                p.rename(backup_path)
                removed.append(p.name)
    if removed:
        print(f"[CLEAN] 已备份 {len(removed)} 个旧语料文件到 {backup_dir}")
        for n in removed:
            print(f"  - {n}")
    return len(removed)


def main(database: str = "neo4j", output_dir: Path | None = None) -> int:
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUT] 输出目录: {output_dir}")

    removed = cleanup_old_corpus(output_dir)
    if removed == 0:
        print("[CLEAN] 没有需要清理的旧语料")

    try:
        from app.src.core.graph_db import get_neo4j_graph
    except ImportError as exc:
        print(f"[FAIL] 无法导入 graph_db: {exc}")
        return 1

    os.environ["NEO4J_DB"] = database
    graph = get_neo4j_graph()
    if graph is None:
        print(f"[FAIL] Neo4j 未连接（database={database}）")
        return 1
    print(f"[OK] Neo4j 已连接 (db={database})")

    total_start = time.monotonic()
    summary = []
    for spec in NODE_QUERIES:
        try:
            result = export_one_label(graph, spec, output_dir)
            summary.append(result)
        except Exception as exc:
            print(f"  [{spec['label']}] 导出失败: {exc}")
            summary.append({
                "label": spec["label"],
                "count": 0,
                "files": [],
                "error": str(exc),
            })

    total_dt = time.monotonic() - total_start
    print(f"\n[DONE] 全部导出完成，总耗时 {total_dt:.1f}s")
    print("\n=== 汇总 ===")
    for s in summary:
        files = ", ".join(s.get("files", [])) or "(无文件)"
        err = f" ERR={s.get('error')}" if "error" in s else ""
        print(f"  {s['label']:14s}: {s['count']:6d} 条 -> {files}{err}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="neo4j",
                        choices=["tcm_graph", "neo4j"],
                        help="目标 Neo4j database (默认 neo4j，因为 ITCM/HPOA 都在这里)")
    parser.add_argument("--out", default=None,
                        help="输出目录（默认 graphrag/data/input/）")
    args = parser.parse_args()
    out = Path(args.out) if args.out else None
    raise SystemExit(main(database=args.db, output_dir=out))
