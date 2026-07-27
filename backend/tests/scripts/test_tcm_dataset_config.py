"""TCM_Dataset 跨平台路径配置测试。"""

from pathlib import Path

from scripts.tcm_dataset_config import (
    candidate_tcm_dataset_roots,
    get_tcm_dataset_config,
)


def test_explicit_dataset_root_has_highest_priority(tmp_path: Path):
    explicit = tmp_path / "dataset"
    explicit.mkdir()

    config = get_tcm_dataset_config(explicit)

    assert config.root == explicit
    assert config.source == "explicit"


def test_project_dataset_root_is_a_candidate():
    candidates = candidate_tcm_dataset_roots()

    assert any(candidate.root.name == "TCM_Dataset" for candidate in candidates)
