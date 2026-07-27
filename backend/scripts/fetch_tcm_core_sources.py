"""从公开数据库官方下载 SymMap v2、ITCM 与 TCMBank 核心文件。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.tcm_dataset_config import get_tcm_dataset_root


class SourceAsset(BaseModel):
    source: str
    url: str
    relative_path: Path
    expected_size: int = Field(gt=0)


SYMMAP_BASE = "http://www.symmap.org/static/download/V2.0"
ITCM_BASE = "http://itcm.biotcm.net"
TCMBANK_BASE = "http://tcmbank.cn/file/TCM_database"

ASSETS: tuple[SourceAsset, ...] = (
    SourceAsset(source="symmap", url=f"{SYMMAP_BASE}/SymMap%20v2.0%2C%20SMHB%20file.xlsx", relative_path=Path("SymMap_v2/Herb/SymMap v2.0, SMHB file.xlsx"), expected_size=99_614),
    SourceAsset(source="symmap", url=f"{SYMMAP_BASE}/SymMap%20v2.0%2C%20SMTS%20file.xlsx", relative_path=Path("SymMap_v2/TCM symptom/SymMap v2.0, SMTS file.xlsx"), expected_size=225_683),
    SourceAsset(source="symmap", url=f"{SYMMAP_BASE}/SymMap%20v2.0%2C%20SMMS%20file.xlsx", relative_path=Path("SymMap_v2/MM symptom/SymMap v2.0, SMMS file.xlsx"), expected_size=262_393),
    SourceAsset(source="symmap", url=f"{SYMMAP_BASE}/SymMap%20v2.0%2C%20SMIT%20file.xlsx", relative_path=Path("SymMap_v2/Ingredient/SymMap v2.0, SMIT file.xlsx"), expected_size=1_889_366),
    SourceAsset(source="symmap", url=f"{SYMMAP_BASE}/SymMap%20v2.0%2C%20SMTT%20file.xlsx", relative_path=Path("SymMap_v2/Target/SymMap v2.0, SMTT file.xlsx"), expected_size=1_677_922),
    SourceAsset(source="symmap", url=f"{SYMMAP_BASE}/SymMap%20v2.0%2C%20SMDE%20file.xlsx", relative_path=Path("SymMap_v2/Disease/SymMap v2.0, SMDE file.xlsx"), expected_size=2_255_321),
    SourceAsset(source="symmap", url=f"{SYMMAP_BASE}/SymMap%20v2.0%2C%20SMSY%20file.xlsx", relative_path=Path("SymMap_v2/Syndrome/SymMap v2.0, SMSY file.xlsx"), expected_size=31_666),
    SourceAsset(source="itcm", url=f"{ITCM_BASE}/downDetail/formula", relative_path=Path("ITCM/formula_detail.txt"), expected_size=11_391_995),
    SourceAsset(source="itcm", url=f"{ITCM_BASE}/downDetail/herb", relative_path=Path("ITCM/herb_detail.txt"), expected_size=1_980_675),
    SourceAsset(source="itcm", url=f"{ITCM_BASE}/downDetail/ingredient", relative_path=Path("ITCM/ingredient_detail.txt"), expected_size=9_272_801),
    SourceAsset(source="itcm", url=f"{ITCM_BASE}/downDetail/target", relative_path=Path("ITCM/target_detail.txt"), expected_size=2_867_467),
    SourceAsset(source="itcm", url=f"{ITCM_BASE}/downDetail/disease", relative_path=Path("ITCM/disease_detail.txt"), expected_size=830_520),
    SourceAsset(source="itcm", url=f"{ITCM_BASE}/downloaCuration/3", relative_path=Path("ITCM/Manual Curation of Formula.xlsx"), expected_size=790_710),
    SourceAsset(source="itcm", url=f"{ITCM_BASE}/downloaCuration/4", relative_path=Path("ITCM/Manual Curation of Herb Ingredient and Target.xlsx"), expected_size=5_041_999),
    SourceAsset(source="tcmbank", url=f"{TCMBANK_BASE}/herb_all.xlsx", relative_path=Path("TCMBank/herb_all.xlsx"), expected_size=1_475_016),
    SourceAsset(source="tcmbank", url=f"{TCMBANK_BASE}/ingredient_all.xlsx", relative_path=Path("TCMBank/ingredient_all.xlsx"), expected_size=64_110_358),
    SourceAsset(source="tcmbank", url=f"{TCMBANK_BASE}/gene_all.xlsx", relative_path=Path("TCMBank/gene_all.xlsx"), expected_size=1_684_300),
    SourceAsset(source="tcmbank", url=f"{TCMBANK_BASE}/disease_all.xlsx", relative_path=Path("TCMBank/disease_all.xlsx"), expected_size=2_445_433),
)


def _validate_download(path: Path, asset: SourceAsset) -> None:
    actual_size = path.stat().st_size
    if actual_size != asset.expected_size:
        raise RuntimeError(
            f"文件大小不匹配: {asset.relative_path}, "
            f"got={actual_size}, expected={asset.expected_size}"
        )
    if path.suffix.lower() == ".xlsx":
        with path.open("rb") as source:
            if source.read(2) != b"PK":
                raise RuntimeError(f"下载结果不是有效 XLSX: {asset.relative_path}")


def download_asset(root: Path, asset: SourceAsset, force: bool) -> str:
    destination = root / asset.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        try:
            _validate_download(destination, asset)
            return "skip"
        except RuntimeError:
            pass

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(asset.url, headers={"User-Agent": "RenShu-AI/1.0"})
    started = time.monotonic()
    with urlopen(request, timeout=300) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    _validate_download(temporary, asset)
    temporary.replace(destination)
    return f"ok({destination.stat().st_size / 1_048_576:.1f}MB/{time.monotonic() - started:.1f}s)"


def main(sources: set[str], force: bool = False) -> int:
    root = get_tcm_dataset_root()
    root.mkdir(parents=True, exist_ok=True)
    selected = [asset for asset in ASSETS if asset.source in sources]
    print(f"[DIR] TCM_Dataset: {root}")
    failures: list[str] = []
    for asset in selected:
        try:
            status = download_asset(root, asset, force)
            print(f"  [{status:20s}] {asset.relative_path}")
        except Exception as exc:
            failures.append(f"{asset.relative_path}: {type(exc).__name__}: {exc}")
            print(f"  [FAIL] {asset.relative_path}: {type(exc).__name__}: {exc}")
    print(f"[SUMMARY] {len(selected) - len(failures)}/{len(selected)} files ready")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["symmap", "itcm", "tcmbank"],
        default=["symmap", "itcm", "tcmbank"],
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(set(args.sources), force=args.force))
