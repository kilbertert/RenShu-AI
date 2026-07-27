"""安全权限收口回归测试。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from app.src.common.context.request_context import UserContext, set_current_context
from app.src.controller import attachment_controller, model_config_controller, tongue_analysis_controller
from app.src.controller.account_controller import register_admin
from app.src.response.exception.exceptions import (
    AuthenticationException,
    AuthorizationException,
)
from app.src.response.response_middleware import ResponseMiddleware
from app.src.service.conversation_service import ConversationService
from app.src.service.language_model_service import ModelProviderService
from app.src.schema.model_config_schema import ModelProviderUpdate


@pytest.fixture(autouse=True)
def reset_user_context():
    set_current_context(UserContext())
    yield
    set_current_context(UserContext())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint",
    [
        model_config_controller.create_provider,
        model_config_controller.update_provider,
        model_config_controller.delete_provider,
        model_config_controller.verify_provider_api_key,
        model_config_controller.create_model_config,
        model_config_controller.update_model_config,
        model_config_controller.delete_model_config,
    ],
)
async def test_model_write_endpoints_require_login(endpoint):
    with pytest.raises(HTTPException) as exc_info:
        await endpoint()
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint",
    [
        attachment_controller.upload_attachment,
        attachment_controller.get_attachment,
        attachment_controller.get_attachment_content,
        tongue_analysis_controller.analyze_tongue_image,
        tongue_analysis_controller.get_tongue_history,
    ],
)
async def test_attachment_and_tongue_endpoints_require_login(endpoint):
    with pytest.raises(HTTPException) as exc_info:
        await endpoint()
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_provider_write_rejects_missing_user_id():
    service = ModelProviderService(session=SimpleNamespace())
    with pytest.raises(AuthenticationException):
        await service.delete_provider_safe(uuid4(), None, is_admin=False)


@pytest.mark.asyncio
async def test_owner_base_url_update_clears_stale_personal_override():
    user_id = uuid4()
    provider_id = uuid4()
    provider = SimpleNamespace(
        id=provider_id,
        owner_id=user_id,
        default_base_url="http://127.0.0.1:8317",
    )
    session = SimpleNamespace(add=lambda _item: None, flush=AsyncMock())
    service = ModelProviderService(session=session)
    service.get = AsyncMock(return_value=provider)
    service.update_user_config = AsyncMock(return_value=SimpleNamespace())

    await service.update_provider_safe(
        provider_id,
        ModelProviderUpdate(
            provider_id=provider_id,
            base_url="https://api.longcat.chat/anthropic",
        ),
        user_id,
        is_admin=False,
    )

    assert provider.default_base_url == "https://api.longcat.chat/anthropic"
    update_data = service.update_user_config.await_args.args[1]
    assert update_data["base_url"] == ""


class _Result:
    def __init__(self, value):
        self.value = value

    def first(self):
        return self.value


class _ConversationSession:
    def __init__(self, conversation):
        self.conversation = conversation

    async def exec(self, _statement):
        return _Result(self.conversation)


@pytest.mark.asyncio
async def test_conversation_read_rejects_cross_user_access():
    conversation = SimpleNamespace(id=uuid4(), user_id=uuid4())
    service = ConversationService(_ConversationSession(conversation))

    with pytest.raises(AuthorizationException):
        await service._get_owned_conversation(str(conversation.id), str(uuid4()))


@pytest.mark.asyncio
async def test_existing_conversation_cannot_be_rebound_to_another_user():
    conversation = SimpleNamespace(id=uuid4(), user_id=uuid4())
    service = ConversationService(_ConversationSession(conversation))

    with pytest.raises(AuthorizationException):
        await service._get_or_create_conversation(
            str(conversation.id), str(uuid4()), "越权会话"
        )


class _AdminService:
    def __init__(self, has_admin: bool):
        self._has_admin = has_admin

    async def has_admin_accounts(self) -> bool:
        return self._has_admin


@pytest.mark.asyncio
async def test_public_cannot_register_additional_admin():
    request = SimpleNamespace(state=SimpleNamespace(client_ip="test", request_id="req"))
    admin_data = SimpleNamespace(username="attacker", password="not-used")

    with pytest.raises(HTTPException) as exc_info:
        await register_admin(
            request=request,
            admin_data=admin_data,
            bootstrap_token=None,
            account_service=_AdminService(has_admin=True),
        )
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_first_admin_bootstrap_is_disabled_by_default():
    request = SimpleNamespace(state=SimpleNamespace(client_ip="test", request_id="req"))
    admin_data = SimpleNamespace(username="bootstrap", password="not-used")

    with pytest.raises(HTTPException) as exc_info:
        await register_admin(
            request=request,
            admin_data=admin_data,
            bootstrap_token=None,
            account_service=_AdminService(has_admin=False),
        )
    assert exc_info.value.status_code == 403


async def _cors_endpoint(_request):
    return JSONResponse({"ok": True})


def _cors_client() -> TestClient:
    app = Starlette(routes=[Route("/", _cors_endpoint)])
    app.add_middleware(
        ResponseMiddleware,
        allowed_origins=["https://allowed.example"],
    )
    return TestClient(app)


def test_cors_allows_configured_origin():
    response = _cors_client().get("/", headers={"Origin": "https://allowed.example"})
    assert response.headers["access-control-allow-origin"] == "https://allowed.example"


def test_cors_does_not_reflect_untrusted_origin():
    response = _cors_client().get("/", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers
