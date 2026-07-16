"""外部 TCM 关系数据源下载器 (P0.5)

下载 4 HPO + 4 SIDER 文件到 ``TCM_Dataset/_external/``。
幂等：已存在则跳过。可用 ``--force`` 强制覆盖。

依据：``docs/external_relation_fetch_plan.md`` 方案 A。

执行::

    cd D:/AI/project/RenShu-AI/backend
    python -m scripts.fetch_external_sources
    # 或强制重下：
    python -m scripts.fetch_external_sources --force
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

TCM_ROOT = Path("D:/AI/project/RenShu-AI/TCM_Dataset")
EXTERNAL_ROOT = TCM_ROOT / "_external"

HPO_DIR = EXTERNAL_ROOT / "HPO"
SIDER_DIR = EXTERNAL_ROOT / "SIDER"


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    dest: Path
    expected_size: int


HPO_ASSETS: tuple[Asset, ...] = (
    Asset("hp.obo", "https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo", HPO_DIR / "hp.obo", 10_703_106),
    Asset("phenotype.hpoa", "http://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa", HPO_DIR / "phenotype.hpoa", 35_261_380),
    Asset("genes_to_phenotype.txt", "http://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt", HPO_DIR / "genes_to_phenotype.txt", 20_533_481),
    Asset("en_product4.xml", "http://www.orphadata.org/data/xml/en_product4.xml", HPO_DIR / "en_product4.xml", 48_788_342),
)

SIDER_ASSETS: tuple[Asset, ...] = (
    Asset("meddra_all_se.tsv.gz", "http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz", SIDER_DIR / "meddra_all_se.tsv.gz", 2_381_171),
    Asset("meddra_all_label_se.tsv.gz", "http://sideeffects.embl.de/media/download/meddra_all_label_se.tsv.gz", SIDER_DIR / "meddra_all_label_se.tsv.gz", 42_534_383),
    Asset("drug_names.tsv", "http://sideeffects.embl.de/media/download/drug_names.tsv", SIDER_DIR / "drug_names.tsv", 34_759),
    Asset("drug_atc.tsv", "http://sideeffects.embl.de/media/download/drug_atc.tsv", SIDER_DIR / "drug_atc.tsv", 32_760),
)


def _size_matches(dest: Path, expected: int) -> bool:
    return dest.exists() and expected > 0 and dest.stat().st_size == expected


def _download(asset: Asset, force: bool) -> tuple[str, int]:
    asset.dest.parent.mkdir(parents=True, exist_ok=True)
    if not force and _size_matches(asset.dest, asset.expected_size):
        return "skip", asset.dest.stat().st_size
    if not force and asset.dest.exists() and asset.expected_size == 0:
        return "skip", asset.dest.stat().st_size

    from urllib.request import Request, urlopen

    headers = {"User-Agent": "RenShu-AI/0.1 (TCM Graph Import)"}
    req = Request(asset.url, headers=headers)
    tmp = asset.dest.with_suffix(asset.dest.suffix + ".part")
    started = time.monotonic()
    with urlopen(req, timeout=300) as r, tmp.open("wb") as f:
        chunk = 256 * 1024
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
    elapsed = time.monotonic() - started
    actual_size = tmp.stat().st_size
    tmp.replace(asset.dest)
    if asset.expected_size and actual_size != asset.expected_size:
        return f"size-mismatch(got={actual_size},exp={asset.expected_size})", actual_size
    return f"ok({actual_size / 1_048_576:.1f}MB/{elapsed:.1f}s)", actual_size


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(force: bool = False, verify: bool = True) -> int:
    print(f"[DIR] External data root: {EXTERNAL_ROOT}")
    HPO_DIR.mkdir(parents=True, exist_ok=True)
    SIDER_DIR.mkdir(parents=True, exist_ok=True)

    plan: list[Asset] = [*HPO_ASSETS, *SIDER_ASSETS]
    failed: list[str] = []
    total_bytes_new = 0

    for asset in plan:
        rel = str(asset.dest.relative_to(EXTERNAL_ROOT))
        try:
            status, size = _download(asset, force=force)
        except Exception as exc:
            failed.append(f"{rel}: {type(exc).__name__}: {exc}")
            print(f"  [FAIL] {rel:45s} {type(exc).__name__}: {str(exc)[:60]}")
            continue
        print(f"  [{status:38s}] {rel}")
        if status.startswith("ok") or status.startswith("size-mismatch"):
            total_bytes_new += size

    if verify:
        print("\n[VERIFY] size check ...")
        for asset in plan:
            if asset.dest.exists():
                size = asset.dest.stat().st_size
                match = "OK" if (asset.expected_size == 0 or size == asset.expected_size) else "MISMATCH"
                md5 = _md5(asset.dest)
                rel = str(asset.dest.relative_to(EXTERNAL_ROOT))
                print(f"  [{match:8s}] {rel:45s} size={size:>10d} md5={md5[:8]}")

    print(f"\n[SUMMARY] {len(plan) - len(failed)}/{len(plan)} ok, new bytes={total_bytes_new / 1_048_576:.1f} MB")
    if failed:
        print("[FAILURES]")
        for f in failed:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HPO + SIDER external sources")
    parser.add_argument("--force", action="store_true", help="Re-download even if file size matches")
    parser.add_argument("--no-verify", action="store_true", help="Skip size verification step")
    args = parser.parse_args()
    raise SystemExit(main(force=args.force, verify=not args.no_verify))
