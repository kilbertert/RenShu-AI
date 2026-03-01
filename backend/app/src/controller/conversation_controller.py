from fastapi import APIRouter, Request, Depends
from app.src.response.utils import success_200
from app.src.service.conversation_service import ConversationService
from app.src.dependencies.dependency import get_conversation_service
from app.src.dependencies.dependency import ConversationServiceDep
from app.src.schema.conversation_schema import ConversationMessageRequest, ConversationDeleteRequest, MessageDeleteRequest

router = APIRouter(prefix="/api/v1/conversations", tags=["对话管理"])



@router.get("/me")
async def get_user_conversations(
        request: Request,
        conversation_service: ConversationServiceDep
):
    """
    获取当前用户的对话列表
    """
    
    # 获取用户的会话
    conversations = await conversation_service.get_my_conversations()
    
    client_ip = request.state.client_ip
    request_id = request.state.request_id
    
    return success_200(
        data=[conversation.model_dump() for conversation in conversations],
        message="获取对话列表成功",
        request_id=request_id,
        host_id=client_ip
    )


@router.post("/messages")
async def get_conversation_messages(
        data: ConversationMessageRequest,
        request: Request,
        conversation_service: ConversationServiceDep
):
    """
    获取指定会话的消息列表
    """
    messages = await conversation_service.get_messages_by_conversation_id(data.conversation_id)
    
    client_ip = request.state.client_ip
    request_id = request.state.request_id
    
    return success_200(
        data=[message.model_dump() for message in messages],
        message="获取消息列表成功",
        request_id=request_id,
        host_id=client_ip
    )


@router.post("/delete")
async def delete_conversation(
        data: ConversationDeleteRequest,
        request: Request,
        conversation_service: ConversationServiceDep
):
    """
    删除指定会话
    """
    await conversation_service.delete_conversation(data.conversation_id)
    
    client_ip = request.state.client_ip
    request_id = request.state.request_id
    
    return success_200(
        data=None,
        message="删除会话成功",
        request_id=request_id,
        host_id=client_ip
    )


@router.post("/messages/delete")
async def delete_message(
        data: MessageDeleteRequest,
        request: Request,
        conversation_service: ConversationServiceDep
):
    """
    删除指定消息
    """
    await conversation_service.delete_message(data.message_id)
    
    client_ip = request.state.client_ip
    request_id = request.state.request_id
    
    return success_200(
        data=None,
        message="删除消息成功",
        request_id=request_id,
        host_id=client_ip
    )