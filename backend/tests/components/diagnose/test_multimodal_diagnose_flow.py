"""舌像从诊断输入进入十问信息结构的回归测试。"""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.src.agent.components.diagnose.nodes.collect_info import ExtractedInfo, collect_info


@pytest.mark.asyncio
async def test_collect_info_keeps_authenticated_tongue_analysis() -> None:
    state = {
        "messages": [HumanMessage(content="最近心悸乏力")],
        "tongue_analysis": {
            "tongue_color": "淡白",
            "tongue_shape": "胖大齿痕",
            "coating_color": "白",
            "coating_quality": "滑",
            "image_quality": "good",
        },
        "llm_config": None,
    }

    with (
        patch(
            "app.src.agent.components.diagnose.nodes.collect_info.get_llm",
            return_value=object(),
        ),
        patch(
            "app.src.agent.components.diagnose.nodes.collect_info.invoke_structured_with_json_fallback",
            new=AsyncMock(return_value=ExtractedInfo(chief_complaint="心悸乏力")),
        ),
    ):
        result = await collect_info(state)

    assert result["collected_info"]["tongue"] == {
        "tongue_color": "淡白",
        "tongue_shape": "胖大齿痕",
        "coating_color": "白",
        "coating_quality": "滑",
        "image_quality": "good",
        "source": "image",
    }
