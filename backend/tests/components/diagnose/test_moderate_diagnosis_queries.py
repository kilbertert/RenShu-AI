"""
moderate_diagnosis 节点的查询函数单元测试

覆盖：
- _query_itcm_formulas: 关键词 CONTAINS 匹配 + scoring
- _format_prescriptions: 兼容旧 tcm_graph / 新 ITCM 两种 schema
- _format_syndromes: 兼容 source / source_db / definition 新字段
- _query_related_prescriptions: 无 Neo4j / 无症状 / 关键词切分
- _query_similar_syndromes: 三策略 + 排序 + 截断
- map_reduce 委托路径

Mock Neo4j（图连接通过 MagicMock 注入），不依赖真实数据库。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from app.src.agent.components.diagnose.models import CollectedDiagnoseInfo
from app.src.agent.components.diagnose.nodes.moderate_diagnosis.moderate_diagnosis import (
    _format_cases,
    _format_prescriptions,
    _format_syndromes,
    _query_itcm_formulas,
    _query_related_prescriptions,
    _query_similar_cases,
    _query_similar_syndromes,
)


# ============ Fixtures ============

@pytest.fixture
def insomnia_info() -> CollectedDiagnoseInfo:
    """典型失眠+头痛+心悸 患者信息"""
    return CollectedDiagnoseInfo(
        chief_complaint="失眠多梦，头痛头晕",
        cold_heat="手足心热",
        head_body="头痛，乏力",
        sleep="失眠多梦",
        other_symptoms=["心悸", "口苦"],
    )


@pytest.fixture
def empty_info() -> CollectedDiagnoseInfo:
    return CollectedDiagnoseInfo()


@pytest.fixture
def fake_graph_factory():
    """Factory that returns a mock graph + record list."""
    def _make(rows):
        g = MagicMock()
        g.query.return_value = rows
        return g
    return _make


# ============ _query_itcm_formulas ============

class TestQueryITCMFormulas:

    def test_returns_formulas_with_score(self, fake_graph_factory):
        g = fake_graph_factory([
            {"name": "金珠化痰丸", "effect_zh": "清痰热，安神志，除头痛",
             "indications_zh": "头痛眩运", "source": "TCMID",
             "score": 4, "matched_keywords": ["头痛"]},
            {"name": "参松养心胶囊", "effect_zh": "益气养阴",
             "indications_zh": "心悸不安,失眠多梦", "source": "itcm",
             "score": 3, "matched_keywords": ["失眠多梦", "心悸"]},
        ])
        results = _query_itcm_formulas(g, ["头痛", "失眠多梦"], top_k=8)
        assert len(results) == 2
        assert results[0]["name"] == "金珠化痰丸"
        assert results[0]["match_score"] == 4
        assert results[0]["source_db"] == "itcm"
        assert results[0]["matched_keywords"] == ["头痛"]

    def test_empty_result(self, fake_graph_factory):
        g = fake_graph_factory([])
        results = _query_itcm_formulas(g, ["某种无匹配关键词xyz"], top_k=8)
        assert results == []

    def test_cypher_uses_keywords_param(self, fake_graph_factory):
        g = fake_graph_factory([{"name": "X", "effect_zh": "Y", "indications_zh": "Z",
                                  "source": "S", "score": 1, "matched_keywords": ["kw1"]}])
        _query_itcm_formulas(g, ["kw1", "kw2"], top_k=5)
        call_args = g.query.call_args
        assert "keywords" in call_args.kwargs["params"]
        assert call_args.kwargs["params"]["keywords"] == ["kw1", "kw2"]
        assert call_args.kwargs["params"]["top_k"] == 5


# ============ _format_prescriptions ============

class TestFormatPrescriptions:

    def test_empty_list(self):
        assert _format_prescriptions([]) == "暂无相关方剂"

    def test_itcm_schema(self):
        items = [{
            "name": "金珠化痰丸",
            "effect_zh": "清痰热",
            "indications_zh": "头痛眩运",
            "source": "TCMID",
            "match_score": 4,
            "source_db": "itcm",
        }]
        out = _format_prescriptions(items)
        assert "金珠化痰丸" in out
        assert "(匹配分: 4)" in out
        assert "[TCMID]" in out
        assert "头痛眩运" in out
        assert "ITCM 暂无组成数据" in out

    def test_old_tcm_graph_schema(self):
        items = [{
            "name": "桂枝汤",
            "composition": ["桂枝", "白芍", "甘草", "生姜", "大枣", "（多余项）"],
            "indication": "风寒感冒",
            "source": "伤寒论",
            "match_score": 0,
        }]
        out = _format_prescriptions(items)
        assert "桂枝汤" in out
        assert "桂枝、白芍、甘草、生姜、大枣等" in out
        assert "风寒感冒" in out
        assert "(匹配分: 0)" not in out

    def test_top3_limit(self):
        items = [{"name": f"P{i}", "effect_zh": "e", "indications_zh": "i",
                  "source": "s", "match_score": i} for i in range(10)]
        out = _format_prescriptions(items)
        assert "1. P0" in out
        assert "3. P2" in out
        assert "4. P3" not in out


# ============ _format_syndromes ============

class TestFormatSyndromes:

    def test_empty(self):
        assert _format_syndromes([]) == "暂无相似证型"

    def test_basic(self):
        items = [{
            "name": "心脾不足",
            "symptoms": ["失眠多梦", "心悸"],
            "similarity": 0.8,
        }]
        out = _format_syndromes(items)
        assert "心脾不足" in out
        assert "80%" in out
        assert "失眠多梦" in out

    def test_with_source_and_definition(self):
        items = [{
            "name": "心虚",
            "symptoms": ["心悸"],
            "similarity": 0.4,
            "source": "symmap_syndrome_direct",
            # >80 chars: 截断逻辑才会追加 "..."
            "definition": "心气虚证是指心气不足，鼓动无力，以心悸怔忡及气虚症状为主要表现的虚证，多因久病伤气、劳倦过度或先天不足所致，临床表现为心悸怔忡、胸闷气短、神疲乏力、面色㿠白、舌淡苔白、脉细弱等，治宜补益心气，方用养心汤或归脾汤加减，临床需与心阳虚、心血虚等证候相鉴别。",
        }]
        out = _format_syndromes(items)
        assert "[symmap_syndrome_direct]" in out
        assert "定义：" in out
        assert "..." in out
        # 截断后 ≤ 82 chars（含省略号）
        assert len(out) < 200

    def test_short_definition_not_truncated(self):
        items = [{
            "name": "心虚",
            "symptoms": ["心悸"],
            "similarity": 0.4,
            "definition": "心气虚证。",
        }]
        out = _format_syndromes(items)
        assert "..." not in out
        assert "心气虚证。" in out

    def test_top3_limit(self):
        items = [{"name": f"S{i}", "symptoms": [], "similarity": 0.5} for i in range(5)]
        out = _format_syndromes(items)
        assert "1. S0" in out
        assert "3. S2" in out
        assert "4. S3" not in out


# ============ _query_related_prescriptions ============

class TestQueryRelatedPrescriptions:

    def test_returns_empty_when_no_symptoms(self, empty_info):
        result = asyncio.run(_query_related_prescriptions(empty_info))
        assert result == []

    def test_returns_empty_when_neo4j_unavailable(self, insomnia_info):
        with patch("app.src.core.graph_db.get_neo4j_graph", return_value=None):
            result = asyncio.run(_query_related_prescriptions(insomnia_info))
        assert result == []

    def test_neo4j_query_called_with_keywords(self, insomnia_info):
        fake_graph = MagicMock()
        fake_graph.query.return_value = [
            {"name": "金珠化痰丸", "effect_zh": "x", "indications_zh": "y",
             "source": "s", "score": 4, "matched_keywords": ["头痛"]}
        ]
        with patch("app.src.core.graph_db.get_neo4j_graph", return_value=fake_graph):
            result = asyncio.run(_query_related_prescriptions(insomnia_info))
        assert len(result) == 1
        kws = fake_graph.query.call_args.kwargs["params"]["keywords"]
        assert "头痛" in kws
        assert "心悸" in kws
        assert "失眠多梦" in kws
        assert all(len(k) >= 2 for k in kws)

    def test_keyword_dedup_and_limit_20(self):
        info = CollectedDiagnoseInfo(
            chief_complaint="头痛头痛头痛",
            other_symptoms=[f"症状{i}" for i in range(30)],
        )
        # ITCM 返回空 → 触发 tcm_graph fallback，tcm_graph 也返回空
        fake_graph = MagicMock()
        fake_graph.query.return_value = []
        with patch("app.src.core.graph_db.get_neo4j_graph", return_value=fake_graph):
            asyncio.run(_query_related_prescriptions(info))
        # 取首次（ITCM）调用的 keywords；fallback 调用没有 keywords 参数
        all_calls = fake_graph.query.call_args_list
        first_call_params = all_calls[0].kwargs["params"]
        assert "keywords" in first_call_params
        kws = first_call_params["keywords"]
        assert len(kws) <= 20
        assert len(set(kws)) == len(kws)  # 去重

    def test_fallback_to_tcm_graph(self, insomnia_info):
        neo4j_g = MagicMock()
        neo4j_g.query.return_value = []
        tcm_g = MagicMock()
        tcm_g.query.return_value = [{"name": "桂枝汤", "source": "伤寒论", "id": "P001"}]
        def side_effect(database=None):
            return {"neo4j": neo4j_g, "tcm_graph": tcm_g}.get(database)
        with patch("app.src.core.graph_db.get_neo4j_graph", side_effect=side_effect):
            result = asyncio.run(_query_related_prescriptions(insomnia_info))
        assert len(result) == 1
        assert result[0]["name"] == "桂枝汤"
        assert result[0]["source_db"] == "tcm_graph_fallback"


# ============ _query_similar_syndromes ============

class TestQuerySimilarSyndromes:

    def test_returns_empty_when_no_symptoms(self, empty_info):
        result = asyncio.run(_query_similar_syndromes(empty_info))
        assert result == []

    def test_returns_empty_when_neo4j_unavailable(self, insomnia_info):
        with patch("app.src.core.graph_db.get_neo4j_graph", return_value=None):
            result = asyncio.run(_query_similar_syndromes(insomnia_info))
        assert result == []

    def test_strategy_b_direct_syndrome_match(self, insomnia_info):
        fake_graph = MagicMock()
        responses = [
            [],
            [
                {"name": "心脾不足", "name_en": "Heart-Spleen Deficiency",
                 "definition": "心脾两虚证...", "matched_keywords": ["失眠多梦", "心悸"],
                 "match_count": 2, "score": 5},
                {"name": "心虚", "name_en": "Heart Deficiency",
                 "definition": "心气虚证...", "matched_keywords": ["心悸"],
                 "match_count": 1, "score": 2},
            ],
            [],
        ]
        fake_graph.query.side_effect = responses
        with patch("app.src.core.graph_db.get_neo4j_graph", return_value=fake_graph):
            result = asyncio.run(_query_similar_syndromes(insomnia_info))
        assert len(result) == 2
        assert result[0]["name"] == "心脾不足"
        assert result[0]["source"] == "symmap_syndrome_direct"

    def test_strategy_c_tcm_symptom_match(self, insomnia_info):
        fake_graph = MagicMock()
        responses = [
            [],
            [],
            [{"name": "失眠", "definition": "睡眠障碍...", "locus": "心神",
              "matched_keywords": ["失眠多梦"], "match_count": 1}],
        ]
        fake_graph.query.side_effect = responses
        with patch("app.src.core.graph_db.get_neo4j_graph", return_value=fake_graph):
            result = asyncio.run(_query_similar_syndromes(insomnia_info))
        assert len(result) == 1
        assert result[0]["name"] == "失眠"
        assert result[0]["source"] == "symmap_tcm_symptom"

    def test_strategy_a_hpoa_disease_match(self, insomnia_info):
        fake_graph = MagicMock()
        responses = [
            [{"name": "Hypertension", "source_db": "OMIM", "omim_id": "145500",
              "mesh_id": "D006973", "matched_symptoms": ["Headache"], "match_count": 1}],
            [],
            [],
        ]
        fake_graph.query.side_effect = responses
        with patch("app.src.core.graph_db.get_neo4j_graph", return_value=fake_graph):
            result = asyncio.run(_query_similar_syndromes(insomnia_info))
        assert len(result) == 1
        assert result[0]["name"] == "Hypertension"
        assert result[0]["source"] == "hpoa_disease_via_mmsymptom"
        assert result[0]["omim_id"] == "145500"

    def test_dedupe_across_strategies(self, insomnia_info):
        fake_graph = MagicMock()
        responses = [
            [{"name": "X", "source_db": "OMIM", "omim_id": "1", "mesh_id": "M",
              "matched_symptoms": ["a"], "match_count": 1}],
            [{"name": "X", "name_en": "X", "definition": "d", "matched_keywords": ["a"],
              "match_count": 1, "score": 2}],
            [],
        ]
        fake_graph.query.side_effect = responses
        with patch("app.src.core.graph_db.get_neo4j_graph", return_value=fake_graph):
            result = asyncio.run(_query_similar_syndromes(insomnia_info))
        assert len(result) == 1
        assert result[0]["source"] == "hpoa_disease_via_mmsymptom"

    def test_top5_limit(self, insomnia_info):
        fake_graph = MagicMock()
        responses = [
            [],
            [
                {"name": f"S{i}", "name_en": f"S{i}", "definition": "",
                 "matched_keywords": ["心悸"], "match_count": 1,
                 "score": int(10 * (i + 1) / 2)}
                for i in range(10)
            ],
            [],
        ]
        fake_graph.query.side_effect = responses
        with patch("app.src.core.graph_db.get_neo4j_graph", return_value=fake_graph):
            result = asyncio.run(_query_similar_syndromes(insomnia_info))
        assert len(result) == 5

    def test_neo4j_exception_does_not_crash(self, insomnia_info):
        fake_graph = MagicMock()
        responses = [
            Exception("策略 A 出错"),
            [{"name": "B1", "name_en": "x", "definition": "d", "matched_keywords": ["心悸"],
              "match_count": 1, "score": 1}],
            [],
        ]
        fake_graph.query.side_effect = responses
        with patch("app.src.core.graph_db.get_neo4j_graph", return_value=fake_graph):
            result = asyncio.run(_query_similar_syndromes(insomnia_info))
        assert len(result) == 1
        assert result[0]["name"] == "B1"


# ============ map_reduce 委托 ============

class TestMapReduceDelegation:

    def test_query_related_prescriptions_delegates(self, insomnia_info):
        from app.src.agent.components.diagnose.nodes.moderate_diagnosis.moderate_diagnosis_map_reduce import (
            _query_related_prescriptions as mr_query,
        )
        fake_graph = MagicMock()
        fake_graph.query.return_value = [
            {"name": "X", "effect_zh": "Y", "indications_zh": "Z",
             "source": "S", "score": 1, "matched_keywords": ["心悸"]}
        ]
        with patch("app.src.core.graph_db.get_neo4j_graph", return_value=fake_graph):
            main_result = asyncio.run(_query_related_prescriptions(insomnia_info))
            mr_result = asyncio.run(mr_query(insomnia_info))
        assert len(mr_result) == len(main_result)
        assert mr_result[0]["name"] == main_result[0]["name"]

    def test_query_similar_syndromes_delegates(self, insomnia_info):
        from app.src.agent.components.diagnose.nodes.moderate_diagnosis.moderate_diagnosis_map_reduce import (
            _query_similar_syndromes as mr_query,
        )
        # 用函数式 side_effect：每次调用都返回完整的 [A, B, C] 三段响应
        # 否则 list 类型的 side_effect 在第二次外层调用时已耗尽
        responses_cycle = [
            [],
            [{"name": "Y", "name_en": "y", "definition": "d", "matched_keywords": ["心悸"],
              "match_count": 1, "score": 1}],
            [],
        ]
        fake_graph = MagicMock()
        call_count = {"n": 0}

        def cyclic_side_effect(*args, **kwargs):
            idx = call_count["n"] % 3
            call_count["n"] += 1
            return responses_cycle[idx]
        fake_graph.query.side_effect = cyclic_side_effect

        with patch("app.src.core.graph_db.get_neo4j_graph", return_value=fake_graph):
            main_result = asyncio.run(_query_similar_syndromes(insomnia_info))
            mr_result = asyncio.run(mr_query(insomnia_info))
        assert len(mr_result) == len(main_result)
        assert mr_result[0]["name"] == main_result[0]["name"]


# ============ _query_similar_cases (P3 Task 1) ============

class TestQuerySimilarCases:
    """P3 Task 1：相似医案 = 病证 + 方剂的代理组合"""

    def test_neo4j_unavailable_returns_empty(self, insomnia_info):
        with patch("app.src.core.graph_db.get_neo4j_graph", return_value=None):
            result = asyncio.run(_query_similar_cases(insomnia_info))
        assert result == []

    def test_no_symptoms_returns_empty(self):
        from app.src.agent.components.diagnose.models import CollectedDiagnoseInfo
        empty = CollectedDiagnoseInfo(chief_complaint="", head_body="", cold_heat="",
                                       sweat="", urine_stool="", diet="", chest_abdomen="",
                                       sleep="", emotion="", other_symptoms=[])
        fake_graph = MagicMock()
        with patch("app.src.core.graph_db.get_neo4j_graph", return_value=fake_graph):
            result = asyncio.run(_query_similar_cases(empty))
        assert result == []
        fake_graph.query.assert_not_called()

    def test_full_path_syndrome_formula_match(self, insomnia_info):
        """段 1 全部命中：Syndrome + Disease + Formula"""
        fake_graph = MagicMock()
        responses = [
            # 段 1a-i: SymMap Syndrome
            [{"name": "心脾两虚", "definition": "心血不足脾气虚弱",
              "matched_keywords": ["心悸", "失眠多梦"], "match_count": 2}],
            # 段 1a-ii: HPOA Disease
            [],
            # 段 1b: ITCM Formula
            [{"name": "归脾汤", "effect": "益气补血健脾养心",
              "indications": "心悸失眠", "source": "济生方",
              "score": 4, "matched_keywords": ["心悸", "失眠多梦"]}],
        ]
        fake_graph.query.side_effect = responses
        with patch("app.src.core.graph_db.get_neo4j_graph", return_value=fake_graph):
            result = asyncio.run(_query_similar_cases(insomnia_info))
        assert len(result) == 1
        case = result[0]
        assert case["syndrome"] == "心脾两虚"
        assert case["treatment"] == "归脾汤"
        assert case["source"] == "symmap_syndrome_to_itcm_formula"
        assert case["match_score"] > 0
        # 共享关键词作为主诉
        assert "心悸" in case["chief_complaint"] or "失眠" in case["chief_complaint"]

    def test_disease_path(self, insomnia_info):
        """HPOA Disease 路径"""
        fake_graph = MagicMock()
        responses = [
            # 段 1a-i: SymMap Syndrome 空
            [],
            # 段 1a-ii: HPOA Disease 命中
            [{"name": "Insomnia", "source_db": "OMIM", "mesh_id": "D007319",
              "matched_keywords": ["失眠"], "match_count": 1}],
            # 段 1b: Formula
            [{"name": "酸枣仁汤", "effect": "养血安神", "indications": "虚烦不眠",
              "source": "金匮要略", "score": 3,
              "matched_keywords": ["失眠多梦", "心悸"]}],
        ]
        fake_graph.query.side_effect = responses
        with patch("app.src.core.graph_db.get_neo4j_graph", return_value=fake_graph):
            result = asyncio.run(_query_similar_cases(insomnia_info))
        assert len(result) == 1
        case = result[0]
        assert "Insomnia" in case["syndrome"]
        assert case["source"] == "hpoa_disease_to_itcm_formula"
        assert case["mesh_id"] == "D007319"

    def test_no_formula_only_syndrome(self, insomnia_info):
        """无方剂候选时，仅返回病证摘要"""
        fake_graph = MagicMock()
        responses = [
            [{"name": "肝郁化火", "definition": "肝气郁结化火",
              "matched_keywords": ["心悸"], "match_count": 1}],
            [],
            [],  # 无方剂
        ]
        fake_graph.query.side_effect = responses
        with patch("app.src.core.graph_db.get_neo4j_graph", return_value=fake_graph):
            result = asyncio.run(_query_similar_cases(insomnia_info))
        assert len(result) == 1
        case = result[0]
        assert case["syndrome"] == "肝郁化火"
        assert "无对应方剂" in case["treatment"]
        assert case["source"] == "syndrome_only"

    def test_query_exception_does_not_crash(self, insomnia_info):
        """Neo4j 异常时返回空，不影响上层"""
        fake_graph = MagicMock()
        fake_graph.query.side_effect = Exception("Neo4j 连接断开")
        with patch("app.src.core.graph_db.get_neo4j_graph", return_value=fake_graph):
            result = asyncio.run(_query_similar_cases(insomnia_info))
        assert result == []


# ============ _format_cases (P3 Task 1 增强) ============

class TestFormatCases:

    def test_empty_list(self):
        assert _format_cases([]) == "暂无相似医案"

    def test_p3_enhanced_fields(self):
        """P3 增强字段（treatment_effect / treatment_indications / definition）展示"""
        items = [{
            "chief_complaint": "心悸、失眠多梦",
            "syndrome": "心脾两虚",
            "syndrome_definition": "心血不足与脾气虚弱并见，主要表现为心悸、失眠、食欲不振等",
            "treatment": "归脾汤",
            "treatment_effect": "益气补血，健脾养心",
            "treatment_indications": "心脾两虚，气血不足",
            "match_score": 0.85,
            "source": "symmap_syndrome_to_itcm_formula",
        }]
        out = _format_cases(items)
        assert "心脾两虚" in out
        assert "归脾汤" in out
        assert "益气补血" in out  # effect
        assert "心脾两虚" in out  # indications
        assert "82%" in out or "85%" in out  # match_score
        assert "定义" in out  # syndrome_definition

    def test_legacy_compatibility(self):
        """兼容旧 mock 字段（chief_complaint / syndrome / treatment / similarity）"""
        items = [{
            "chief_complaint": "乏力、食欲不振",
            "syndrome": "脾气虚证",
            "treatment": "补中益气汤",
            "similarity": 0.75,
        }]
        out = _format_cases(items)
        assert "脾气虚证" in out
        assert "补中益气汤" in out
        assert "75%" in out
