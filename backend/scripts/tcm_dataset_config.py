"""TCM_Dataset 跨平台路径发现与配置。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict


BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent
load_dotenv(PROJECT_ROOT / ".env", encoding="utf-8", override=False)


class TCMDatasetConfig(BaseModel):
    """本地数据集目录及其发现来源。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: Path
    source: str


def candidate_tcm_dataset_roots(explicit: str | Path | None = None) -> list[TCMDatasetConfig]:
    """按显式参数、环境变量、仓库目录、Windows/WSL 路径排序候选。"""
    candidates: list[TCMDatasetConfig] = []
    if explicit:
        candidates.append(TCMDatasetConfig(root=Path(explicit).expanduser(), source="explicit"))
    env_root = os.environ.get("TCM_DATASET_ROOT")
    if env_root:
        candidates.append(TCMDatasetConfig(root=Path(env_root).expanduser(), source="env"))
    candidates.extend([
        TCMDatasetConfig(root=PROJECT_ROOT / "TCM_Dataset", source="project"),
        TCMDatasetConfig(
            root=Path("D:/AI/project/RenShu-AI/TCM_Dataset"),
            source="windows",
        ),
        TCMDatasetConfig(
            root=Path("/mnt/d/AI/project/RenShu-AI/TCM_Dataset"),
            source="wsl",
        ),
    ])

    unique: list[TCMDatasetConfig] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.root)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def get_tcm_dataset_config(explicit: str | Path | None = None) -> TCMDatasetConfig:
    """返回第一个已存在的目录；均不存在时返回最高优先级目标目录。"""
    candidates = candidate_tcm_dataset_roots(explicit)
    for candidate in candidates:
        if candidate.root.exists():
            return candidate
    return candidates[0]


def get_tcm_dataset_root(explicit: str | Path | None = None) -> Path:
    return get_tcm_dataset_config(explicit).root
