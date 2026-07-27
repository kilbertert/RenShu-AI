"""TCM Agent 流式内容兼容测试。"""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.src.agent.tcm_service import (
    TCMAgentService,
    _extract_query_type,
    _extract_stream_text,
    _extract_unanswered_state_error,
    _is_user_visible_llm_event,
)
from app.src.agent.tcm_states import LLMConfig, TCMRouter


def test_extract_stream_text_from_anthropic_blocks() -> None:
    content = [
        {"type": "thinking", "thinking": "内部推理"},
        {"type": "text", "text": "辨证"},
        {"type": "text", "text": "完成"},
    ]

    assert _extract_stream_text(content) == "辨证完成"


def test_extract_stream_text_keeps_plain_string() -> None:
    assert _extract_stream_text("普通文本") == "普通文本"


def test_internal_classifier_stream_is_not_user_visible() -> None:
    event = {"metadata": {"langgraph_node": "analyze_and_route_query"}}

    assert not _is_user_visible_llm_event(event)


def test_diagnosis_synthesis_json_stream_is_not_user_visible() -> None:
    event = {"metadata": {"langgraph_node": "synthesize_diagnosis"}}

    assert not _is_user_visible_llm_event(event)


def test_structured_diagnosis_json_stream_is_not_user_visible() -> None:
    event = {
        "metadata": {"langgraph_node": "synthesize_diagnosis"},
        "tags": ["internal_structured_diagnosis"],
    }

    assert not _is_user_visible_llm_event(event)


def test_extract_query_type_from_pydantic_router() -> None:
    router = TCMRouter(query_type="tcm-diagnose")

    assert _extract_query_type(router) == "tcm-diagnose"


def test_unanswered_graph_error_is_not_silently_treated_as_done() -> None:
    state = SimpleNamespace(values={"error": "路由失败", "answer": ""})

    assert _extract_unanswered_state_error(state) == "路由失败"


def test_graph_error_with_patient_answer_does_not_override_answer() -> None:
    state = SimpleNamespace(values={"error": "内部提示", "answer": "可展示回答"})

    assert _extract_unanswered_state_error(state) is None


class _ResumeGraph:
    def __init__(self, user_id: str, conversation_id: str):
        self.state = SimpleNamespace(
            values={
                "user_id": user_id,
                "conversation_id": conversation_id,
                "messages": [],
                "answer": "恢复完成",
                "router": {"query_type": "tcm-diagnose"},
            },
            tasks=[],
        )
        self.updates = []
        self.resume_command = None

    async def aget_state(self, _config):
        return self.state

    async def aupdate_state(self, _config, update):
        self.updates.append(update)

    async def astream_events(self, command, _config, version):
        self.resume_command = command
        if False:
            yield version


@pytest.mark.asyncio
async def test_resume_stream_updates_checkpoint_and_uses_structured_tongue_payload() -> None:
    user_id = str(uuid4())
    conversation_id = str(uuid4())
    graph = _ResumeGraph(user_id, conversation_id)
    service = TCMAgentService()
    service._graph = graph
    tongue = {"tongue_color": "淡白", "coating_quality": "滑"}
    attachments = [{
        "id": str(uuid4()),
        "kind": "tongue_image",
        "original_filename": "舌像.png",
        "mime_type": "image/png",
        "size_bytes": 128,
        "status": "analyzed",
        "download_url": "/api/v1/attachments/example/content",
        "analysis_result": tongue,
        "analysis_error": None,
    }]
    llm_config = LLMConfig(
        provider_name="anthropic",
        model_name="LongCat-2.0",
        base_url="https://api.longcat.chat/anthropic",
    )

    chunks = [
        json.loads(chunk)
        async for chunk in service.resume_stream(
            thread_id="thread-tongue",
            user_answer="",
            user_id=user_id,
            conversation_id=conversation_id,
            attachments=attachments,
            tongue_analysis=tongue,
            llm_config=llm_config,
        )
    ]

    assert graph.updates[0]["tongue_analysis"] == tongue
    assert graph.updates[0]["attachments"] == attachments
    assert graph.updates[0]["llm_config"].base_url == "https://api.longcat.chat/anthropic"
    assert graph.resume_command.resume["tongue_analysis"] == tongue
    assert graph.resume_command.resume["llm_config"]["base_url"] == "https://api.longcat.chat/anthropic"
    assert graph.resume_command.resume["text"] == "我已上传舌像，请结合舌像继续问诊。"
    assert any(chunk.get("type") == "done" for chunk in chunks)
