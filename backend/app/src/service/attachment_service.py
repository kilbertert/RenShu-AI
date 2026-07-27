"""私有聊天附件的安全存储、所有权校验与消息绑定。"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from datetime import datetime
from pathlib import Path
from uuid import UUID

from fastapi import UploadFile
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.src.common.config.setting_config import ROOT_DIR, settings
from app.src.model.attachment_models import ChatAttachment
from app.src.response.exception.exceptions import (
    AuthorizationException,
    BusinessException,
    ResourceNotFoundException,
)
from app.src.schema.attachment_schema import (
    AttachmentContext,
    AttachmentKind,
    AttachmentResponse,
    AttachmentStatus,
)
from app.src.service.conversation_service import ConversationService


ALLOWED_IMAGE_MIME_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
PDF_MIME_TYPE = "application/pdf"


def _sniff_image_mime(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _sniff_attachment_mime(data: bytes) -> str | None:
    image_mime = _sniff_image_mime(data)
    if image_mime:
        return image_mime
    if data.startswith(b"%PDF-"):
        return PDF_MIME_TYPE
    return None


def _safe_filename(filename: str | None) -> str:
    value = Path(filename or "attachment").name.strip()
    value = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", value)
    return value[:255] or "attachment"


class AttachmentService:
    def __init__(
        self,
        session: AsyncSession,
        conversation_service: ConversationService,
    ):
        self.session = session
        self.conversation_service = conversation_service

    @property
    def storage_root(self) -> Path:
        configured = Path(settings.ATTACHMENT_STORAGE_ROOT)
        root = configured if configured.is_absolute() else ROOT_DIR / configured
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve()

    async def create_attachment(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        kind: AttachmentKind,
        upload: UploadFile,
    ) -> ChatAttachment:
        if kind not in {
            AttachmentKind.GENERIC_IMAGE,
            AttachmentKind.TONGUE_IMAGE,
            AttachmentKind.MEDICAL_REPORT,
        }:
            raise BusinessException("附件业务类型不受支持")

        await self.conversation_service._get_or_create_conversation(
            str(conversation_id),
            str(user_id),
            (
                "舌像问诊"
                if kind == AttachmentKind.TONGUE_IMAGE
                else "医疗报告解读"
                if kind == AttachmentKind.MEDICAL_REPORT
                else "图片问诊"
            ),
        )

        data = await self._read_limited(upload)
        detected_mime = _sniff_attachment_mime(data)
        declared_mime = (upload.content_type or "").lower()
        allowed_mimes = set(ALLOWED_IMAGE_MIME_TYPES)
        if kind == AttachmentKind.MEDICAL_REPORT:
            allowed_mimes.add(PDF_MIME_TYPE)
        if detected_mime is None or detected_mime not in allowed_mimes:
            if kind == AttachmentKind.MEDICAL_REPORT:
                raise BusinessException("医疗报告仅支持有效的 JPEG、PNG、WebP 或 PDF 文件")
            raise BusinessException("附件不是有效的 JPEG、PNG 或 WebP 图片")
        if kind in {AttachmentKind.GENERIC_IMAGE, AttachmentKind.TONGUE_IMAGE} and detected_mime == PDF_MIME_TYPE:
            raise BusinessException("舌像和普通图片不支持 PDF")
        if declared_mime and declared_mime not in allowed_mimes:
            raise BusinessException("附件声明的 MIME 类型不受支持")
        if declared_mime and declared_mime != detected_mime:
            raise BusinessException("附件内容与声明的 MIME 类型不一致")

        attachment = ChatAttachment(
            user_id=user_id,
            conversation_id=conversation_id,
            kind=kind.value,
            original_filename=_safe_filename(upload.filename),
            storage_key="pending",
            mime_type=detected_mime,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            status=AttachmentStatus.UPLOADED.value,
        )
        suffix = ALLOWED_IMAGE_MIME_TYPES.get(detected_mime, ".pdf")
        attachment.storage_key = f"{user_id}/{attachment.id}{suffix}"
        target = self._path_for_key(attachment.storage_key)

        self.session.add(attachment)
        try:
            await self._atomic_write(target, data)
            await self.session.commit()
            await self.session.refresh(attachment)
            return attachment
        except Exception:
            await self.session.rollback()
            target.unlink(missing_ok=True)
            raise

    async def create_image_attachment(self, **kwargs) -> ChatAttachment:
        """兼容 M0/M1 既有调用；新代码统一走 create_attachment。"""
        return await self.create_attachment(**kwargs)

    async def _read_limited(self, upload: UploadFile) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await upload.read(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > settings.ATTACHMENT_MAX_BYTES:
                raise BusinessException(
                    f"附件大小不能超过 {settings.ATTACHMENT_MAX_BYTES // (1024 * 1024)} MB"
                )
            chunks.append(chunk)
        if total == 0:
            raise BusinessException("附件内容为空")
        return b"".join(chunks)

    async def _atomic_write(self, target: Path, data: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")

        def write() -> None:
            temp.write_bytes(data)
            os.replace(temp, target)

        await asyncio.to_thread(write)

    def _path_for_key(self, storage_key: str) -> Path:
        target = (self.storage_root / storage_key).resolve()
        if self.storage_root not in target.parents:
            raise BusinessException("附件存储路径非法")
        return target

    async def get_owned(self, attachment_id: UUID, user_id: UUID) -> ChatAttachment:
        attachment = await self.session.get(ChatAttachment, attachment_id)
        if attachment is None:
            raise ResourceNotFoundException("附件不存在")
        if attachment.user_id != user_id:
            raise AuthorizationException("附件不存在或无权访问")
        return attachment

    async def get_owned_path(self, attachment_id: UUID, user_id: UUID) -> tuple[ChatAttachment, Path]:
        attachment = await self.get_owned(attachment_id, user_id)
        path = self._path_for_key(attachment.storage_key)
        if not path.is_file():
            raise ResourceNotFoundException("附件文件不存在")
        return attachment, path

    def path_for_attachment(self, attachment: ChatAttachment) -> Path:
        path = self._path_for_key(attachment.storage_key)
        if not path.is_file():
            raise ResourceNotFoundException("附件文件不存在")
        return path

    async def resolve_for_chat(
        self,
        *,
        refs: list,
        user_id: UUID,
        conversation_id: UUID,
    ) -> list[ChatAttachment]:
        if not refs:
            return []
        ids = [UUID(str(ref.id)) for ref in refs]
        result = await self.session.exec(
            select(ChatAttachment).where(ChatAttachment.id.in_(ids))
        )
        rows = list(result.all())
        by_id = {row.id: row for row in rows}
        if len(by_id) != len(ids):
            raise ResourceNotFoundException("部分附件不存在")
        attachments = [by_id[item_id] for item_id in ids]
        for attachment in attachments:
            if attachment.user_id != user_id:
                raise AuthorizationException("附件不存在或无权访问")
            if attachment.conversation_id != conversation_id:
                raise AuthorizationException("附件不属于当前会话")
            if attachment.message_id is not None:
                raise BusinessException("附件已经发送，不能重复绑定到其他消息")
        return attachments

    async def bind_to_message(self, attachments: list[ChatAttachment], message_id: UUID) -> None:
        now = datetime.now()
        for attachment in attachments:
            attachment.message_id = message_id
            if attachment.status == AttachmentStatus.UPLOADED.value:
                attachment.status = AttachmentStatus.ATTACHED.value
            attachment.updated_at = now
            self.session.add(attachment)

    async def save_analysis(
        self,
        attachment: ChatAttachment,
        analysis_result: dict | None,
        error: str | None = None,
    ) -> None:
        attachment.analysis_result = analysis_result
        attachment.analysis_error = error
        attachment.status = (
            AttachmentStatus.ANALYZED.value
            if analysis_result is not None
            else AttachmentStatus.ANALYSIS_FAILED.value
        )
        attachment.updated_at = datetime.now()
        self.session.add(attachment)
        await self.session.commit()

    def to_response(self, attachment: ChatAttachment) -> AttachmentResponse:
        return AttachmentResponse(
            id=attachment.id,
            conversation_id=attachment.conversation_id,
            kind=AttachmentKind(attachment.kind),
            original_filename=attachment.original_filename,
            mime_type=attachment.mime_type,
            size_bytes=attachment.size_bytes,
            sha256=attachment.sha256,
            status=AttachmentStatus(attachment.status),
            download_url=f"/api/v1/attachments/{attachment.id}/content",
            analysis_result=attachment.analysis_result,
        )

    def to_context(self, attachment: ChatAttachment) -> AttachmentContext:
        return AttachmentContext(
            id=attachment.id,
            kind=AttachmentKind(attachment.kind),
            original_filename=attachment.original_filename,
            mime_type=attachment.mime_type,
            size_bytes=attachment.size_bytes,
            status=AttachmentStatus(attachment.status),
            download_url=f"/api/v1/attachments/{attachment.id}/content",
            analysis_result=attachment.analysis_result,
            analysis_error=attachment.analysis_error,
        )
