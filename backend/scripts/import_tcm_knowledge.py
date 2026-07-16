"""TCM knowledge graph CSV ingestion script.

Imports the TCM ontology (syndromes / symptoms / prescriptions / herbs) and
their four relationship types (INDICATES / TREATS / CONTAINS / INCOMPATIBLE_WITH)
from ``backend/data/tcm/`` into Neo4j via the ``graph_db`` facade.

P1-6 deliverable per ``D:\\AI\\project\\RenShu-AI\\判断.md`` §3.2.

Usage::

    python -m scripts.import_tcm_knowledge                    # full import
    python -m scripts.import_tcm_knowledge --dry-run          # parse only
    python -m scripts.import_tcm_knowledge --clear            # wipe TCM nodes first
    python -m scripts.import_tcm_knowledge --nodes-only
    python -m scripts.import_tcm_knowledge --relationships-only
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# Allow running as ``python scripts/import_tcm_knowledge.py`` from the backend dir.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.src.core.graph_db import get_neo4j_graph, is_graph_db_available  # noqa: E402
from app.src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

DATA_ROOT = _BACKEND_ROOT / "data" / "tcm"
REL_ROOT = DATA_ROOT / "relationships"


# --- Schema declarations (drive both node and relationship imports) ---

@dataclass(frozen=True)
class NodeSpec:
    """Declarative description of a node CSV → Neo4j label mapping."""

    label: str
    csv_path: Path
    id_column: str
    property_columns: tuple[str, ...]


@dataclass(frozen=True)
class RelSpec:
    """Declarative description of a relationship CSV → Neo4j edge mapping."""

    rel_type: str
    from_label: str
    to_label: str
    csv_path: Path
    from_id_column: str
    to_id_column: str
    property_columns: tuple[str, ...] = ()


NODE_SPECS: tuple[NodeSpec, ...] = (
    NodeSpec(
        label="Syndrome",
        csv_path=DATA_ROOT / "syndromes.csv",
        id_column="id",
        property_columns=("name", "category", "description", "treatment"),
    ),
    NodeSpec(
        label="Symptom",
        csv_path=DATA_ROOT / "symptoms.csv",
        id_column="id",
        property_columns=("name", "category", "description"),
    ),
    NodeSpec(
        label="Prescription",
        csv_path=DATA_ROOT / "prescriptions.csv",
        id_column="id",
        property_columns=("name", "source", "category", "description", "indication"),
    ),
    NodeSpec(
        label="Herb",
        csv_path=DATA_ROOT / "herbs.csv",
        id_column="id",
        property_columns=(
            "name", "pinyin", "nature", "flavor", "meridian",
            "toxicity", "dosage", "effects", "indications",
        ),
    ),
)

REL_SPECS: tuple[RelSpec, ...] = (
    RelSpec(
        rel_type="INDICATES",
        from_label="Symptom",
        to_label="Syndrome",
        csv_path=REL_ROOT / "symptom_indicates_syndrome.csv",
        from_id_column="symptom_id",
        to_id_column="syndrome_id",
        property_columns=("weight",),
    ),
    RelSpec(
        rel_type="TREATS",
        from_label="Prescription",
        to_label="Syndrome",
        csv_path=REL_ROOT / "prescription_treats_syndrome.csv",
        from_id_column="prescription_id",
        to_id_column="syndrome_id",
    ),
    RelSpec(
        rel_type="CONTAINS",
        from_label="Prescription",
        to_label="Herb",
        csv_path=REL_ROOT / "prescription_contains_herb.csv",
        from_id_column="prescription_id",
        to_id_column="herb_id",
        property_columns=("dosage",),
    ),
    RelSpec(
        rel_type="INCOMPATIBLE_WITH",
        from_label="Herb",
        to_label="Herb",
        csv_path=REL_ROOT / "herb_incompatible_herb.csv",
        from_id_column="herb1_id",
        to_id_column="herb2_id",
        property_columns=("reason",),
    ),
)

ALL_TCM_LABELS: tuple[str, ...] = ("Syndrome", "Symptom", "Prescription", "Herb")


# --- Result accumulator (immutable per coding-style immutability rule) ---

@dataclass(frozen=True)
class ImportResult:
    """Aggregated counts of nodes / relationships imported."""

    nodes_by_label: dict[str, int]
    rels_by_type: dict[str, int]
    skipped: int = 0

    @property
    def total_nodes(self) -> int:
        return sum(self.nodes_by_label.values())

    @property
    def total_relationships(self) -> int:
        return sum(self.rels_by_type.values())


# --- CSV utilities ---

def _read_csv(path: Path) -> Iterable[dict[str, str]]:
    """Yield non-empty rows from ``path`` as plain dicts (UTF-8)."""
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not row or not any((v or "").strip() for v in row.values()):
                continue
            yield {k: (v or "").strip() for k, v in row.items() if k is not None}


# --- Graph operations ---

def import_nodes(graph, specs: tuple[NodeSpec, ...]) -> dict[str, int]:
    """MERGE each node row; return {label: count}."""
    counts: dict[str, int] = {}
    for spec in specs:
        if not spec.csv_path.exists():
            logger.warning("node CSV missing: %s", spec.csv_path)
            continue
        cypher = (
            f"MERGE (n:{spec.label} {{id: $node_id}}) "
            f"SET n += $props"
        )
        n = 0
        for row in _read_csv(spec.csv_path):
            node_id = row[spec.id_column]
            props = {col: row[col] for col in spec.property_columns if col in row and row[col]}
            graph.query(cypher, params={"node_id": node_id, "props": props})
            n += 1
        counts[spec.label] = n
        logger.info("imported %d %s nodes from %s", n, spec.label, spec.csv_path.name)
    return counts


def import_relationship(graph, spec: RelSpec) -> int:
    """MERGE each relationship row; return count imported."""
    if not spec.csv_path.exists():
        logger.warning("rel CSV missing: %s", spec.csv_path)
        return 0
    cypher = (
        f"MATCH (a:{spec.from_label} {{id: $from_id}}) "
        f"MATCH (b:{spec.to_label} {{id: $to_id}}) "
        f"MERGE (a)-[r:{spec.rel_type}]->(b) "
        f"SET r += $props"
    )
    n = 0
    skipped = 0
    for row in _read_csv(spec.csv_path):
        from_id = row[spec.from_id_column]
        to_id = row[spec.to_id_column]
        props = {col: row[col] for col in spec.property_columns if col in row and row[col]}
        try:
            # Probe that both endpoints exist before writing.
            graph.query(cypher, params={"from_id": from_id, "to_id": to_id, "props": props})
            n += 1
        except Exception as exc:  # endpoint missing or other graph error
            skipped += 1
            logger.debug("skip %s(%s)->%s: %s", spec.rel_type, from_id, to_id, exc)
    logger.info(
        "imported %d %s relationships from %s (skipped %d)",
        n, spec.rel_type, spec.csv_path.name, skipped,
    )
    return n


def import_relationships(graph, specs: tuple[RelSpec, ...]) -> dict[str, int]:
    """Import all relationship specs; return {rel_type: count}."""
    counts: dict[str, int] = {}
    for spec in specs:
        counts[spec.rel_type] = import_relationship(graph, spec)
    return counts


def clear_tcm_nodes(graph) -> int:
    """DETACH DELETE every TCM-labeled node; return count deleted."""
    cypher = (
        "MATCH (n) WHERE n:Syndrome OR n:Symptom "
        "OR n:Prescription OR n:Herb DETACH DELETE n RETURN count(n) AS n"
    )
    result = graph.query(cypher)
    n = result[0]["n"] if result else 0
    logger.info("cleared %d TCM nodes", n)
    return n


# --- Orchestrator ---

def run_import(
    *,
    dry_run: bool = False,
    clear: bool = False,
    nodes_only: bool = False,
    relationships_only: bool = False,
) -> Optional[ImportResult]:
    """Top-level entry. ``None`` on dry-run or unrecoverable failure."""
    if not is_graph_db_available():
        logger.error("Neo4j unavailable; set NEO4J_* env vars and ensure the service is up.")
        return None

    graph = get_neo4j_graph(database="tcm_graph")
    if graph is None:
        logger.error("could not acquire Neo4j 'tcm_graph' handle.")
        return None

    if clear and not dry_run:
        clear_tcm_nodes(graph)

    nodes: dict[str, int] = {}
    rels: dict[str, int] = {}

    if dry_run:
        # Validate files parse and print expected counts without touching Neo4j.
        for spec in NODE_SPECS:
            if not spec.csv_path.exists():
                logger.warning("missing: %s", spec.csv_path)
                continue
            nodes[spec.label] = sum(1 for _ in _read_csv(spec.csv_path))
        for spec in REL_SPECS:
            if not spec.csv_path.exists():
                logger.warning("missing: %s", spec.csv_path)
                continue
            rels[spec.rel_type] = sum(1 for _ in _read_csv(spec.csv_path))
        return ImportResult(nodes_by_label=nodes, rels_by_type=rels)

    if not relationships_only:
        nodes = import_nodes(graph, NODE_SPECS)
    if not nodes_only:
        rels = import_relationships(graph, REL_SPECS)

    return ImportResult(nodes_by_label=nodes, rels_by_type=rels)


# --- CLI ---

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Import TCM knowledge graph CSVs into Neo4j.",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Parse CSVs and print counts without writing to Neo4j.")
    p.add_argument("--clear", action="store_true",
                   help="DETACH DELETE all TCM nodes before importing.")
    p.add_argument("--nodes-only", action="store_true",
                   help="Only import node CSVs.")
    p.add_argument("--relationships-only", action="store_true",
                   help="Only import relationship CSVs.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    result = run_import(
        dry_run=args.dry_run,
        clear=args.clear,
        nodes_only=args.nodes_only,
        relationships_only=args.relationships_only,
    )
    if result is None:
        return 1
    logger.info(
        "done. nodes=%d rels=%d (per label: %s; per type: %s)",
        result.total_nodes, result.total_relationships,
        result.nodes_by_label, result.rels_by_type,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
