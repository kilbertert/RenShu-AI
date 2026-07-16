"""TCM knowledge graph verification script (P1-7 per 判断.md §3.6).

Performs two layers of checks:

1. **Local CSV validation** — 8 count assertions + ID reference integrity
   for every node and relationship CSV. Always runs.
2. **Neo4j query smoke tests** — two semantic queries that confirm the data
   was imported correctly (symptom → syndrome; syndrome → prescription).
   Gated on ``--with-neo4j`` so the script stays runnable in environments
   without a live Neo4j instance.

Usage::

    python -m scripts.verify_tcm_data                # local only
    python -m scripts.verify_tcm_data --with-neo4j   # local + Neo4j
    python -m scripts.verify_tcm_data --strict       # exit 1 on any failure
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Allow running as ``python scripts/verify_tcm_data.py`` from the backend dir.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

DATA_ROOT = _BACKEND_ROOT / "data" / "tcm"
REL_ROOT = DATA_ROOT / "relationships"


# --- Result accumulator (immutable per coding-style immutability rule) ---

@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single named check."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class VerificationReport:
    """Aggregated verification result."""

    checks: tuple[CheckResult, ...] = ()

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if not c.passed)

    def summary(self) -> str:
        total = len(self.checks)
        ok = total - len(self.failures)
        verdict = "PASS" if self.passed else "FAIL"
        return f"[{verdict}] {ok}/{total} checks passed"


# --- Local CSV utilities ---

def _read_ids(path: Path, id_col: str) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return {row[id_col] for row in csv.DictReader(fh) if row.get(id_col)}


def _row_count(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if any((v or "").strip() for v in row.values()):
                n += 1
    return n


def _check_minimum(name: str, actual: int, minimum: int) -> CheckResult:
    if actual >= minimum:
        return CheckResult(name, True, f"actual={actual} >= {minimum}")
    return CheckResult(name, False, f"actual={actual} < required {minimum}")


# --- Local validation (8 count assertions) ---

def validate_locally() -> VerificationReport:
    """Run all CSV-side checks; return a frozen report."""
    checks: list[CheckResult] = []

    # Node counts (4 assertions)
    n_syn = len(_read_ids(DATA_ROOT / "syndromes.csv", "id"))
    n_sym = len(_read_ids(DATA_ROOT / "symptoms.csv", "id"))
    n_pr = len(_read_ids(DATA_ROOT / "prescriptions.csv", "id"))
    n_hb = len(_read_ids(DATA_ROOT / "herbs.csv", "id"))
    checks.append(_check_minimum("Node.Syndrome >=50", n_syn, 50))
    checks.append(_check_minimum("Node.Symptom >=200", n_sym, 200))
    checks.append(_check_minimum("Node.Prescription >=100", n_pr, 100))
    checks.append(_check_minimum("Node.Herb >=200", n_hb, 200))

    # Relationship counts + ID reference integrity (4 checks)
    syn_ids = _read_ids(DATA_ROOT / "syndromes.csv", "id")
    sym_ids = _read_ids(DATA_ROOT / "symptoms.csv", "id")
    pr_ids = _read_ids(DATA_ROOT / "prescriptions.csv", "id")
    hb_ids = _read_ids(DATA_ROOT / "herbs.csv", "id")

    rel_specs = [
        (
            "Rel.INDICATES >=500",
            REL_ROOT / "symptom_indicates_syndrome.csv",
            "symptom_id", sym_ids, "syndrome_id", syn_ids, 500,
        ),
        (
            "Rel.TREATS >=120",
            REL_ROOT / "prescription_treats_syndrome.csv",
            "prescription_id", pr_ids, "syndrome_id", syn_ids, 120,
        ),
        (
            "Rel.CONTAINS >=400",
            REL_ROOT / "prescription_contains_herb.csv",
            "prescription_id", pr_ids, "herb_id", hb_ids, 400,
        ),
        (
            "Rel.INCOMPATIBLE_WITH >=30",
            REL_ROOT / "herb_incompatible_herb.csv",
            "herb1_id", hb_ids, "herb2_id", hb_ids, 30,
        ),
    ]
    for name, path, sc, src, fc, dst, mn in rel_specs:
        if not path.exists():
            checks.append(CheckResult(name, False, f"missing file: {path}"))
            continue
        total, bad_src, bad_dst = 0, set(), set()
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                total += 1
                if row.get(sc) not in src:
                    bad_src.add(row.get(sc))
                if row.get(fc) not in dst:
                    bad_dst.add(row.get(fc))
        count_ok = total >= mn
        integrity_ok = not bad_src and not bad_dst
        ok = count_ok and integrity_ok
        detail = f"rows={total} target>={mn} bad_{sc}={len(bad_src)} bad_{fc}={len(bad_dst)}"
        checks.append(CheckResult(name, ok, detail))

    return VerificationReport(tuple(checks))


# --- Neo4j smoke tests ---

def _lookup_node_id_by_name(graph, label: str, name: str) -> Optional[str]:
    """Resolve a node id by its ``name`` field; ``None`` if not found."""
    cypher = f"MATCH (n:{label} {{name: $name}}) RETURN n.id AS id LIMIT 1"
    result = graph.query(cypher, params={"name": name})
    if not result:
        return None
    return result[0].get("id")


def validate_neo4j() -> VerificationReport:
    """Run two semantic smoke tests against the live Neo4j graph."""
    checks: list[CheckResult] = []
    try:
        from app.src.core.graph_db import get_neo4j_graph, is_graph_db_available
    except Exception as exc:
        checks.append(CheckResult("Neo4j.import", False, f"import failed: {exc}"))
        return VerificationReport(tuple(checks))

    if not is_graph_db_available():
        checks.append(CheckResult("Neo4j.connection", False, "graph_db unavailable"))
        return VerificationReport(tuple(checks))

    graph = get_neo4j_graph(database="tcm_graph")
    if graph is None:
        checks.append(CheckResult("Neo4j.handle", False, "could not acquire handle"))
        return VerificationReport(tuple(checks))

    # Test 1: 恶寒 / 鼻塞 / 流涕 → 风寒感冒
    symptom_names = ["恶寒", "鼻塞", "流涕"]
    target_syndrome = "风寒感冒"
    syn_id = _lookup_node_id_by_name(graph, "Syndrome", target_syndrome)
    if syn_id is None:
        checks.append(CheckResult(
            "Smoke.symptom_to_syndrome",
            False,
            f"syndrome '{target_syndrome}' missing in Neo4j",
        ))
    else:
        results: list[tuple[str, float]] = []
        cypher = (
            "MATCH (s:Symptom {name: $sname})-[r:INDICATES]->(syn:Syndrome {id: $syn_id}) "
            "RETURN s.name AS sname, r.weight AS w ORDER BY w DESC LIMIT 1"
        )
        for sname in symptom_names:
            rows = graph.query(cypher, params={"sname": sname, "syn_id": syn_id})
            if rows:
                results.append((sname, float(rows[0].get("w") or 0.0)))
        matched = [s for s, _ in results]
        if len(matched) == len(symptom_names):
            checks.append(CheckResult(
                "Smoke.symptom_to_syndrome",
                True,
                f"{symptom_names} -> {target_syndrome} (weights={[w for _, w in results]})",
            ))
        else:
            missing = set(symptom_names) - set(matched)
            checks.append(CheckResult(
                "Smoke.symptom_to_syndrome",
                False,
                f"missing INDICATES edges from: {sorted(missing)}",
            ))

    # Test 2: 风寒感冒 → 方剂 (Prescriptions)
    if syn_id is None:
        checks.append(CheckResult(
            "Smoke.syndrome_to_prescription",
            False,
            "depends on previous test (syndrome id unresolved)",
        ))
    else:
        cypher = (
            "MATCH (p:Prescription)-[:TREATS]->(syn:Syndrome {id: $syn_id}) "
            "RETURN p.id AS pid, p.name AS pname"
        )
        rows = graph.query(cypher, params={"syn_id": syn_id})
        names = sorted({r.get("pname") for r in rows if r.get("pname")})
        if names:
            preview = ", ".join(names[:3]) + (f" (+{len(names) - 3} more)" if len(names) > 3 else "")
            checks.append(CheckResult(
                "Smoke.syndrome_to_prescription",
                True,
                f"{target_syndrome} -> {len(names)} prescription(s): {preview}",
            ))
        else:
            checks.append(CheckResult(
                "Smoke.syndrome_to_prescription",
                False,
                f"no TREATS edges into {target_syndrome}",
            ))

    return VerificationReport(tuple(checks))


# --- CLI ---

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Verify TCM knowledge graph CSVs and (optionally) Neo4j state.",
    )
    p.add_argument("--with-neo4j", action="store_true",
                   help="Also run semantic smoke tests against Neo4j.")
    p.add_argument("--strict", action="store_true",
                   help="Exit with non-zero status on any check failure.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    local = validate_locally()
    logger.info("local checks: %s", local.summary())
    for c in local.checks:
        marker = "OK" if c.passed else "FAIL"
        logger.info("  %s %-32s %s", marker, c.name, c.detail)

    reports: list[VerificationReport] = [local]
    if args.with_neo4j:
        neo = validate_neo4j()
        logger.info("neo4j checks: %s", neo.summary())
        for c in neo.checks:
            marker = "OK" if c.passed else "FAIL"
            logger.info("  %s %-32s %s", marker, c.name, c.detail)
        reports.append(neo)

    overall = all(r.passed for r in reports)
    logger.info("OVERALL: %s", "PASS" if overall else "FAIL")
    return 1 if (not overall and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
