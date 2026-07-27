"""E2E 证方一致性断言回归测试。"""

import pytest

from scripts.run_e2e_acceptance import assert_prescription_grounding


def _grounded_payload() -> dict:
    return {
        "syndrome": "心脾两虚证",
        "prescriptions": [{
            "name": "归脾汤",
            "relation_evidence": {
                "source_db": "curated_tcm",
                "syndrome_id": "SY-1",
                "syndrome_name": "心脾两虚",
                "formula_id": "F-1",
                "formula_name": "归脾汤",
                "relationship_type": "TREATS_WITH",
                "relationship_id": "R-1",
                "relationship_path": [
                    "Syndrome[SY-1]",
                    "-[:TREATS_WITH {R-1}]-",
                    "Formula[F-1]",
                ],
            },
        }],
    }


def test_allows_no_formula_when_graph_has_no_verified_relation():
    assert_prescription_grounding(
        {"syndrome": "风寒感冒证", "prescriptions": []},
        [],
    )


def test_accepts_formula_with_matching_relation_path():
    assert_prescription_grounding(
        _grounded_payload(),
        [{"name": "归脾汤"}],
    )


def test_rejects_formula_without_relation_evidence():
    with pytest.raises(RuntimeError, match="缺少可追溯证方关系"):
        assert_prescription_grounding(
            {"syndrome": "风寒感冒证", "prescriptions": [{"name": "归脾汤"}]},
            [{"name": "归脾汤"}],
        )


def test_rejects_relation_for_different_syndrome():
    payload = _grounded_payload()
    payload["prescriptions"][0]["relation_evidence"]["syndrome_name"] = "肝郁脾虚证"

    with pytest.raises(RuntimeError, match="关系证型与最终主证不一致"):
        assert_prescription_grounding(payload, [{"name": "归脾汤"}])
