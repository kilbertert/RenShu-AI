"""M0 私有附件边界与聊天请求契约测试。"""

from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError
from starlette.datastructures import Headers, UploadFile

from app.src.common.config.setting_config import settings
from app.src.model.attachment_models import ChatAttachment
from app.src.response.exception.exceptions import AuthorizationException, BusinessException
from app.src.schema.attachment_schema import AttachmentKind
from app.src.schema.chat_schema import ChatRequest, ModelConfiguration
from app.src.service.attachment_service import AttachmentService


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"safe-image-payload"
PDF_BYTES = b"%PDF-1.4\n% safe-test-pdf\n"


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Session:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.added = []

    def add(self, value):
        self.added.append(value)

    async def exec(self, _statement):
        return _Rows(self.rows)

    async def commit(self):
        return None

    async def rollback(self):
        return None

    async def refresh(self, _value):
        return None


class _ConversationService:
    def __init__(self, session):
        self.session = session

    async def _get_or_create_conversation(self, conversation_id, user_id, title):
        return SimpleNamespace(id=conversation_id, user_id=user_id, title=title)


def _upload(data: bytes, mime: str = "image/png", filename: str = "舌像.png") -> UploadFile:
    return UploadFile(
        file=BytesIO(data),
        filename=filename,
        headers=Headers({"content-type": mime}),
    )


@pytest.mark.asyncio
async def test_medical_report_accepts_pdf_but_tongue_kind_rejects_it(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ATTACHMENT_STORAGE_ROOT", str(tmp_path))
    service = _service(_Session())

    report = await service.create_attachment(
        user_id=uuid4(),
        conversation_id=uuid4(),
        kind=AttachmentKind.MEDICAL_REPORT,
        upload=_upload(PDF_BYTES, mime="application/pdf", filename="检验报告.pdf"),
    )
    assert report.mime_type == "application/pdf"
    assert service.path_for_attachment(report).suffix == ".pdf"

    with pytest.raises(BusinessException, match="JPEG"):
        await _service(_Session()).create_attachment(
            user_id=uuid4(),
            conversation_id=uuid4(),
            kind=AttachmentKind.TONGUE_IMAGE,
            upload=_upload(PDF_BYTES, mime="application/pdf", filename="伪装舌像.pdf"),
        )


def _service(session: _Session) -> AttachmentService:
    return AttachmentService(session, _ConversationService(session))


@pytest.mark.asyncio
async def test_valid_image_is_private_and_context_excludes_storage_path(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ATTACHMENT_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setattr(settings, "ATTACHMENT_MAX_BYTES", 1024)
    session = _Session()
    service = _service(session)

    attachment = await service.create_image_attachment(
        user_id=uuid4(),
        conversation_id=uuid4(),
        kind=AttachmentKind.TONGUE_IMAGE,
        upload=_upload(PNG_BYTES),
    )

    assert service.path_for_attachment(attachment).is_file()
    response = service.to_response(attachment).model_dump(mode="json")
    context = service.to_context(attachment).model_dump(mode="json")
    serialized = str({"response": response, "context": context})
    assert attachment.storage_key not in serialized
    assert str(tmp_path) not in serialized
    assert "base64" not in serialized


@pytest.mark.asyncio
async def test_mime_and_magic_bytes_must_match(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ATTACHMENT_STORAGE_ROOT", str(tmp_path))

    with pytest.raises(BusinessException, match="不一致"):
        await _service(_Session()).create_image_attachment(
            user_id=uuid4(),
            conversation_id=uuid4(),
            kind=AttachmentKind.TONGUE_IMAGE,
            upload=_upload(PNG_BYTES, mime="image/jpeg"),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [b"", PNG_BYTES + b"x" * 100])
async def test_empty_and_oversized_files_are_rejected(tmp_path, monkeypatch, data):
    monkeypatch.setattr(settings, "ATTACHMENT_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setattr(settings, "ATTACHMENT_MAX_BYTES", 32)

    with pytest.raises(BusinessException):
        await _service(_Session()).create_image_attachment(
            user_id=uuid4(),
            conversation_id=uuid4(),
            kind=AttachmentKind.TONGUE_IMAGE,
            upload=_upload(data),
        )


def test_storage_path_traversal_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ATTACHMENT_STORAGE_ROOT", str(tmp_path))

    with pytest.raises(BusinessException, match="路径非法"):
        _service(_Session())._path_for_key("../outside.png")


@pytest.mark.asyncio
async def test_cross_user_cross_conversation_and_reuse_are_rejected():
    owner_id = uuid4()
    conversation_id = uuid4()
    attachment = ChatAttachment(
        user_id=owner_id,
        conversation_id=conversation_id,
        kind=AttachmentKind.TONGUE_IMAGE.value,
        original_filename="舌像.png",
        storage_key=f"{owner_id}/{uuid4()}.png",
        mime_type="image/png",
        size_bytes=len(PNG_BYTES),
        sha256="0" * 64,
    )
    ref = SimpleNamespace(id=attachment.id)

    with pytest.raises(AuthorizationException):
        await _service(_Session([attachment])).resolve_for_chat(
            refs=[ref], user_id=uuid4(), conversation_id=conversation_id
        )
    with pytest.raises(AuthorizationException):
        await _service(_Session([attachment])).resolve_for_chat(
            refs=[ref], user_id=owner_id, conversation_id=uuid4()
        )

    attachment.message_id = uuid4()
    with pytest.raises(BusinessException, match="已经发送"):
        await _service(_Session([attachment])).resolve_for_chat(
            refs=[ref], user_id=owner_id, conversation_id=conversation_id
        )


def test_chat_request_accepts_attachment_only_and_limits_count():
    model = ModelConfiguration(
        provider_id=str(uuid4()),
        model_id=str(uuid4()),
        model_name="vision-model",
    )
    request = ChatRequest(
        user_id=str(uuid4()),
        conversation_id=str(uuid4()),
        query="",
        attachments=[{"id": uuid4()}],
        model_configuration=model,
    )
    assert request.query == ""

    with pytest.raises(ValidationError):
        ChatRequest(
            user_id=str(uuid4()),
            conversation_id=str(uuid4()),
            query="",
            model_configuration=model,
        )
    with pytest.raises(ValidationError):
        ChatRequest(
            user_id=str(uuid4()),
            conversation_id=str(uuid4()),
            query="舌像",
            attachments=[{"id": uuid4()} for _ in range(5)],
            model_configuration=model,
        )
