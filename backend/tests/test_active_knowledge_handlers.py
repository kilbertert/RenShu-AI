"""活跃药材、方剂和图像路由不再返回空 mock。"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.src.agent.components.herb.handlers import HERB_QUERY, handle_herb_query
from app.src.agent.components.image.handlers import handle_image_query
from app.src.agent.components.prescription.handlers import (
    PRESCRIPTION_QUERY,
    handle_prescription_query,
)
from app.src.agent.components.router.router import route_query
from app.src.agent.components.diagnose.router import route_by_complexity
from app.src.agent.tcm_states import TCMAgentState, TCMRouter


class _FakeLLM:
    async def ainvoke(self, _messages):
        return AIMessage(content="基于实时检索资料生成的说明。")


@pytest.mark.asyncio
async def test_herb_handler_returns_retrieved_structured_data():
    state = TCMAgentState(
        messages=[HumanMessage(content="黄芪有什么作用")],
        router=TCMRouter(
            query_type="tcm-herb",
            extracted_entities={"herbs": ["黄芪"]},
        ),
    )
    graph = MagicMock()
    graph.query.return_value = [{
        "name": "黄芪",
        "pinyin": "Huang Qi",
        "category": "补气药",
        "nature": "微温",
        "flavor": "甘",
        "meridians": "脾、肺",
        "effects": "补气升阳、固表止汗",
        "indications": "气虚乏力",
        "contraindications": "",
        "dosage": "",
        "source_db": "ITCM",
    }]

    with (
        patch("app.src.core.graph_db.get_neo4j_graph", return_value=graph),
        patch("app.src.agent.tcm_builder.get_llm", return_value=_FakeLLM()),
    ):
        result = await handle_herb_query(state)

    assert result["herbs"][0].name == "黄芪"
    assert "补气升阳" in result["herbs"][0].effects
    assert result["herbs"][0].sources == ["ITCM"]
    assert result["answer"]
    assert result["cypher_queries"]


@pytest.mark.asyncio
async def test_herb_handler_checks_known_incompatibility_without_mock_data():
    state = TCMAgentState(
        messages=[HumanMessage(content="甘草和海藻能一起用吗")],
        router=TCMRouter(
            query_type="tcm-herb",
            extracted_entities={"herbs": ["甘草", "海藻"]},
        ),
    )
    with (
        patch("app.src.core.graph_db.get_neo4j_graph", return_value=None),
        patch("app.src.agent.tcm_builder.get_llm", return_value=_FakeLLM()),
    ):
        result = await handle_herb_query(state)

    assert result["compatibility_check"].is_compatible is False
    assert ("甘草", "海藻") in result["compatibility_check"].incompatible_pairs


@pytest.mark.asyncio
async def test_prescription_handler_returns_retrieved_structured_data():
    state = TCMAgentState(
        messages=[HumanMessage(content="四君子汤的功效是什么")],
        router=TCMRouter(
            query_type="tcm-prescription",
            extracted_entities={"prescriptions": ["四君子汤"]},
        ),
    )
    graph = MagicMock()
    graph.query.return_value = [{
        "name": "四君子汤",
        "source": "《太平惠民和剂局方》",
        "composition": [{"herb": "人参", "dosage": ""}],
        "effects": "益气健脾",
        "indications": "脾胃气虚",
        "syndrome": "脾气虚证",
        "usage": "",
        "cautions": "",
    }]

    with (
        patch("app.src.core.graph_db.get_neo4j_graph", return_value=graph),
        patch("app.src.agent.tcm_builder.get_llm", return_value=_FakeLLM()),
    ):
        result = await handle_prescription_query(state)

    assert result["prescriptions"][0].name == "四君子汤"
    assert result["prescriptions"][0].effects == "益气健脾"
    assert result["answer"]


@pytest.mark.asyncio
async def test_herb_answer_never_invents_dosage_when_user_forbids_it():
    state = TCMAgentState(
        messages=[HumanMessage(content="黄芪有什么作用？不要猜测剂量。")],
        router=TCMRouter(
            query_type="tcm-herb",
            extracted_entities={"herbs": ["黄芪"]},
        ),
    )
    graph = MagicMock()
    graph.query.return_value = [
        {
            "name": "黄芪",
            "nature": "微温",
            "flavor": "甘",
            "meridians": "肺、脾",
            "effects": "补气升阳",
            "indications": "气虚乏力",
            "source_db": "ITCM",
        },
        {
            "name": "黄芪",
            "nature": "甘,微温",
            "meridians": "肺、脾",
            "effects": "固表止汗",
            "source_db": "SymMap",
        },
    ]

    with patch("app.src.core.graph_db.get_neo4j_graph", return_value=graph):
        result = await handle_herb_query(state)

    assert result["herbs"][0].sources == ["ITCM", "SymMap"]
    assert "补气升阳" in result["answer"]
    assert "固表止汗" in result["answer"]
    assert "9-30" not in result["answer"]
    assert "图谱用量" not in result["answer"]


@pytest.mark.asyncio
async def test_herb_answer_understands_natural_no_dosage_wording():
    state = TCMAgentState(
        messages=[HumanMessage(content="黄芪有什么作用？不要给具体用药剂量。")],
        router=TCMRouter(
            query_type="tcm-herb",
            extracted_entities={"herbs": ["黄芪"]},
        ),
    )
    graph = MagicMock()
    graph.query.return_value = [{
        "name": "黄芪",
        "effects": "补气升阳",
        "dosage": "9-30g",
        "source_db": "ITCM",
    }]

    with patch("app.src.core.graph_db.get_neo4j_graph", return_value=graph):
        result = await handle_herb_query(state)

    assert "9-30" not in result["answer"]
    assert "图谱用量" not in result["answer"]


def test_herb_display_normalizes_common_english_graph_terms():
    from app.src.agent.components.herb.handlers import _format_grounded_herb_answer
    from app.src.agent.tcm_states import HerbInfo

    answer = _format_grounded_herb_answer(
        "黄芪有什么作用",
        [HerbInfo(
            name="黄芪",
            nature="Warm; Sweet",
            flavor=["甘"],
            meridians=["Lung", "Spleen", "肺"],
            effects=["To reinoforce qi and invigorate the function of the spleen."],
            indications=["Common cold", "补气", "升阳"],
            sources=["ITCM"],
        )],
        None,
    )

    assert "性味：温、甘" in answer
    assert "归经：肺、脾" in answer
    assert "功效：补气健脾" in answer
    assert "Common cold" not in answer


@pytest.mark.asyncio
async def test_explicit_formula_name_does_not_mix_kampo_variant():
    state = TCMAgentState(
        messages=[HumanMessage(content="四君子汤由哪些药组成？")],
        router=TCMRouter(
            query_type="tcm-prescription",
            extracted_entities={"prescriptions": ["四君子汤"]},
        ),
    )
    graph = MagicMock()
    graph.query.return_value = [
        {
            "name": "四君子汤",
            "source": "《太平惠民和剂局方》",
            "composition": [],
            "effects": "益气健脾",
            "indications": "脾胃气虚",
            "syndrome": "",
            "usage": "",
            "cautions": "",
        },
        {
            "name": "四君子湯（四君子汤）",
            "source": "kampo",
            "composition": [{"herb": "干姜"}, {"herb": "大枣"}],
            "effects": "",
            "indications": "慢性胃炎",
            "syndrome": "",
            "usage": "Oral",
            "cautions": "",
        },
    ]

    with patch("app.src.core.graph_db.get_neo4j_graph", return_value=graph):
        result = await handle_prescription_query(state)

    assert [item.name for item in result["prescriptions"]] == ["四君子汤"]
    assert "干姜" not in result["answer"]
    assert "大枣" not in result["answer"]
    assert "图谱未提供" in result["answer"]


def test_image_query_type_routes_to_explicit_handler():
    state = TCMAgentState(router=TCMRouter(query_type="tcm-image"))
    assert route_query(state) == "handle_image_query"


@pytest.mark.asyncio
async def test_image_handler_is_truthful_when_no_image_was_received():
    state = TCMAgentState(router=TCMRouter(query_type="tcm-image"))
    result = await handle_image_query(state)
    assert "没有收到可分析的图片" in result["answer"]


def test_complex_diagnosis_uses_explicit_safe_degradation_node():
    assert route_by_complexity({"complexity": {"level": "complex"}}) == "complex_diagnosis"


def test_active_queries_require_non_empty_names_and_current_schema():
    assert "herb_name IS NOT NULL" in HERB_QUERY
    assert "Herb_ITCM" not in HERB_QUERY
    assert "formula_name IS NOT NULL" in PRESCRIPTION_QUERY
    assert "CONTAINS|" not in PRESCRIPTION_QUERY
