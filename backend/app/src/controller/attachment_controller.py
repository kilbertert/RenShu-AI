"""认证聊天附件 API。"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse

from app.src.common.context import get_current_user_id
from app.src.common.decorators import require_login
from app.src.dependencies.dependency import AttachmentServiceDep
from app.src.response.utils import success_200
from app.src.schema.attachment_schema import AttachmentKind


router = APIRouter(prefix="/api/v1/attachments", tags=["聊天附件"])


@router.post("")
@require_login
async def upload_attachment(
    attachment_service: AttachmentServiceDep,
    conversation_id: UUID = Form(...),
    kind: AttachmentKind = Form(AttachmentKind.GENERIC_IMAGE),
    file: UploadFile = File(...),
):
    user_id = UUID(str(get_current_user_id()))
    attachment = await attachment_service.create_attachment(
        user_id=user_id,
        conversation_id=conversation_id,
        kind=kind,
        upload=file,
    )
    return success_200(data=attachment_service.to_response(attachment).model_dump(mode="json"))


@router.get("/{attachment_id}")
@require_login
async def get_attachment(
    attachment_id: UUID,
    attachment_service: AttachmentServiceDep,
):
    user_id = UUID(str(get_current_user_id()))
    attachment = await attachment_service.get_owned(attachment_id, user_id)
    return success_200(data=attachment_service.to_response(attachment).model_dump(mode="json"))


@router.get("/{attachment_id}/content")
@require_login
async def get_attachment_content(
    attachment_id: UUID,
    attachment_service: AttachmentServiceDep,
):
    user_id = UUID(str(get_current_user_id()))
    attachment, path = await attachment_service.get_owned_path(attachment_id, user_id)
    return FileResponse(
        path,
        media_type=attachment.mime_type,
        filename=attachment.original_filename,
        headers={"Cache-Control": "private, max-age=300"},
    )
