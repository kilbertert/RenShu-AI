"""会话绑定、状态重置和患者画像注入回归测试。"""

from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from langgraph.types import Overwrite
from langchain_core.messages import AIMessage, HumanMessage

from app.src.agent.tcm_service import TCMAgentService
from app.src.agent.tcm_states import LLMConfig
from app.src.common.context.request_context import UserContext, set_current_context
from app.src.response.exception.exceptions import AuthorizationException
from app.src.schema.chat_schema import ChatResumeRequest, ModelConfiguration
from app.src.service.chat_servcie import ChatService
from app.src.utils.auth_utils import encrypt_api_key


class _Graph:
    def __init__(self, values):
        self.values = values
        self.updated = None

    async def aget_state(self, _config):
        return SimpleNamespace(values=self.values)

    async def aupdate_state(self, _config, values):
        self.updated = values


@pytest.mark.asyncio
async def test_new_turn_keeps_thread_but_resets_previous_outputs():
    service = TCMAgentService()
    graph = _Graph({
        "user_id": "user-1",
        "conversation_id": "conversation-1",
        "messages": ["history"],
        "answer": "previous answer",
        "steps": ["old step"],
    })
    service._graph = graph

    await service._prepare_new_turn("thread-1", "user-1", "conversation-1")

    assert graph.updated["answer"] == ""
    assert isinstance(graph.updated["steps"], Overwrite)
    assert graph.updated["steps"].value == []
    assert "messages" not in graph.updated


@pytest.mark.asyncio
async def test_new_turn_overwrites_expired_checkpoint_llm_config():
    service = TCMAgentService()
    graph = _Graph({
        "user_id": "user-1",
        "conversation_id": "conversation-1",
        "llm_config": LLMConfig(
            provider_name="anthropic",
            model_name="LongCat-2.0",
            base_url="http://127.0.0.1:8317",
        ),
    })
    service._graph = graph
    current = LLMConfig(
        provider_name="anthropic",
        model_name="LongCat-2.0",
        base_url="https://api.longcat.chat/anthropic",
    )

    await service._prepare_new_turn(
        "thread-1",
        "user-1",
        "conversation-1",
        llm_config=current,
    )

    assert graph.updated["llm_config"].base_url == "https://api.longcat.chat/anthropic"


@pytest.mark.asyncio
async def test_completed_turn_appends_assistant_message_to_checkpoint_history():
    service = TCMAgentService()
    graph = _Graph({"messages": [HumanMessage(content="上一轮问题")]})
    service._graph = graph

    await service._append_checkpoint_messages(
        {"configurable": {"thread_id": "thread-1"}},
        [AIMessage(content="上一轮回答")],
    )

    assert graph.updated["messages"][0].content == "上一轮回答"


@pytest.mark.asyncio
async def test_resume_appends_user_and_assistant_messages_to_checkpoint_history():
    service = TCMAgentService()
    graph = _Graph({"messages": [AIMessage(content="请补充信息")]})
    service._graph = graph

    await service._append_checkpoint_messages(
        {"configurable": {"thread_id": "thread-1"}},
        [HumanMessage(content="补充回答"), AIMessage(content="辨证结果")],
    )

    assert [message.content for message in graph.updated["messages"]] == [
        "补充回答",
        "辨证结果",
    ]


@pytest.mark.asyncio
async def test_checkpoint_identity_rejects_cross_user_resume():
    service = TCMAgentService()
    service._graph = _Graph({
        "user_id": "owner",
        "conversation_id": "conversation-1",
    })

    with pytest.raises(AuthorizationException):
        await service.assert_thread_binding("thread-1", "attacker", "conversation-1")


class _ProfileSession:
    def __init__(self, patient, health_profile):
        self.patient = patient
        self.health_profile = health_profile

    async def exec(self, _statement):
        return SimpleNamespace(first=lambda: self.patient)

    async def get(self, model, _identity):
        return self.health_profile


class _ConversationService:
    def __init__(self, session=None, conversation=None):
        self.session = session
        self.conversation = conversation

    async def _get_owned_conversation(self, _conversation_id, _user_id):
        return self.conversation


@pytest.mark.asyncio
async def test_agent_profile_merges_patient_and_longitudinal_health_data():
    patient = SimpleNamespace(
        birth_date=date(1990, 1, 1),
        gender="female",
        base_profile={
            "constitution_type": "气虚质",
            "medical_history": ["高血压"],
            "allergy_info": ["青霉素"],
            "taboo_items": ["酒精"],
        },
    )
    health = SimpleNamespace(
        constitution="阳虚质",
        chronic_conditions=["失眠"],
        allergies=["花粉"],
        most_common_syndrome="心脾两虚证",
        total_cases=3,
    )
    conversation = SimpleNamespace(session_metadata={"chief_complaint": "乏力"})
    chat_service = ChatService(
        _ConversationService(_ProfileSession(patient, health)),
        model_service=SimpleNamespace(),
    )

    profile = await chat_service._build_agent_user_profile(str(uuid4()), conversation)

    assert profile["constitution"] == "阳虚质"
    assert set(profile["allergies"]) == {"青霉素", "花粉"}
    assert profile["most_common_syndrome"] == "心脾两虚证"
    assert profile["session_persona"]["chief_complaint"] == "乏力"


@pytest.mark.asyncio
async def test_resume_request_cannot_replace_server_bound_thread():
    user_id = str(uuid4())
    conversation = SimpleNamespace(
        agent_thread_id=uuid4(),
        agent_interrupt={"pending": True},
    )
    chat_service = ChatService(
        _ConversationService(conversation=conversation),
        model_service=SimpleNamespace(),
    )
    set_current_context(UserContext(
        user_id=user_id,
        is_authenticated=True,
        roles=["patient"],
    ))
    request = ChatResumeRequest(
        conversation_id=str(uuid4()),
        thread_id=str(uuid4()),
        query="继续回答",
        model_configuration=ModelConfiguration(
            provider_id=str(uuid4()),
            model_id=str(uuid4()),
            model_name="test",
        ),
    )

    try:
        with pytest.raises(AuthorizationException):
            await chat_service.resume_agent_chat(request)
    finally:
        set_current_context(UserContext())


def test_checkpoint_serialization_contains_only_encrypted_api_key():
    ciphertext = encrypt_api_key("plain-secret")
    config = LLMConfig(
        provider_name="openai",
        model_name="test",
        api_key_encrypted=ciphertext,
    )

    serialized = config.model_dump()
    assert "api_key" not in serialized
    assert serialized["api_key_encrypted"] == ciphertext
    assert "plain-secret" not in str(serialized)
    assert config.api_key == "plain-secret"
