import uuid

from app.src.model import Conversation, Message
from app.src.response.exception.exceptions import InternalServerException
from app.src.service.base_service import BaseService
from app.src.common.decorators import require_login
from app.src.common.context import get_current_user_id
from sqlmodel import select

from app.src.model import Account


class ConversationService(BaseService):
    def __init__(self, session):
        super().__init__(Conversation, session)

    # ========== 外部接口方法（带装饰器） ==========

    @require_login
    async def get_my_conversations(self):
        """获取当前登录用户的会话列表"""
        user_id = get_current_user_id()
        return await self._get_conversation_by_user_id(user_id)

    @require_login
    async def create_my_conversation(self, **kwargs):
        """为当前登录用户创建会话"""
        user_id = get_current_user_id()
        return await self._create_conversation(user_id, **kwargs)

    @require_login
    async def get_messages_by_conversation_id(self, conversation_id: str):
        """获取指定会话的消息列表"""
        return await self._get_messages_by_conversation_id(conversation_id)

    @require_login
    async def delete_conversation(self, conversation_id: str):
        """删除指定会话"""
        user_id = get_current_user_id()
        return await self._delete_conversation(conversation_id, user_id)

    @require_login
    async def delete_message(self, message_id: str):
        """删除指定消息"""
        user_id = get_current_user_id()
        return await self._delete_message(message_id, user_id)

    @require_login
    async def get_or_create_conversation(self, conversation_id: str, title: str = "New Chat"):
        """获取或创建会话（处理并发）"""
        user_id = get_current_user_id()
        return await self._get_or_create_conversation(conversation_id, user_id, title)

    # ========== 内部方法（不加装饰器） ==========

    async def _get_or_create_conversation(self, conversation_id: str, user_id: str, title: str):
        """内部方法：获取或创建会话（健壮的并发处理）"""
        # 1. 尝试查询
        stmt = select(Conversation).where(Conversation.id == conversation_id)
        result = await self.session.exec(stmt)
        conversation = result.first()
        
        if conversation:
            return conversation
            
        # 2. 尝试创建
        try:
            conversation = Conversation(
                id=conversation_id,
                user_id=user_id,
                session_id=str(uuid.uuid4()),
                conversation_type="general_chat",
                title=title
            )
            self.session.add(conversation)
            await self.session.flush()
            return conversation
        except Exception:
            # 3. 如果创建失败（可能是并发写入冲突），回滚并重试查询
            await self.session.rollback()
            stmt = select(Conversation).where(Conversation.id == conversation_id)
            result = await self.session.exec(stmt)
            conversation = result.first()
            if conversation:
                return conversation
            # 如果依然拿不到，说明是真正的数据库错误，抛出异常
            raise InternalServerException(message="无法创建或获取会话")


    async def _get_conversation_by_user_id(self, user_id: str):
        """内部方法：根据用户id获取会话"""
        stmt = select(Account).where(Account.id == user_id)
        res = await self.session.exec(stmt)
        user = res.one_or_none()

        if not user:
            raise ValueError("用户不存在")
        if not user.is_active:
            raise ValueError("用户账户未激活")

        conversation_stmt = select(Conversation).where(Conversation.user_id == user_id).order_by(Conversation.updated_at.desc())
        conversation_res = await self.session.exec(conversation_stmt)
        return conversation_res.all()

    async def _create_conversation(self, user_id: str, **kwargs):
        """内部方法：创建会话"""
        try:
            conversation = await self.get(user_id)
            if not conversation:
                conversation = Conversation(
                    user_id=user_id,
                    session_id=str(uuid.uuid4()),
                    conversation_type="日常交流",
                )
                return await self.create(conversation)
            return conversation
        except Exception as e:
            raise InternalServerException(message="会话添加错误", details={"error": str(e)})

    async def _get_messages_by_conversation_id(self, conversation_id: str):
        """内部方法：根据会话ID获取消息列表"""
        stmt = select(Message).where(
            Message.conversation_id == conversation_id,
            Message.is_deleted == False
        ).order_by(Message.created_at)
        result = await self.session.exec(stmt)
        return result.all()

    async def _delete_conversation(self, conversation_id: str, user_id: str):
        """内部方法：删除会话"""
        # 1. 验证会话属于当前用户
        stmt = select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user_id)
        result = await self.session.exec(stmt)
        conversation = result.one_or_none()
        
        if not conversation:
            raise ValueError("会话不存在或无权访问")

        # 2. 删除会话下的所有消息 (物理删除或软删除均可，这里选择物理删除以保持整洁，或者软删除如果为了审计)
        # 由于Message有is_deleted，我们可以选择软删除所有消息，或者直接物理删除。
        # 考虑到 Conversation 是物理删除（假设），那么消息也应该物理删除。
        # 但为了稳妥，先尝试物理删除会话。如果配置了级联删除，消息会自动删除。
        # 如果没有配置级联，我们需要手动删除消息。
        
        # 手动物理删除消息
        messages_stmt = select(Message).where(Message.conversation_id == conversation_id)
        messages_result = await self.session.exec(messages_stmt)
        messages = messages_result.all()
        for msg in messages:
            await self.session.delete(msg)
            
        # 3. 删除会话
        await self.session.delete(conversation)
        return True

    async def _delete_message(self, message_id: str, user_id: str):
        """内部方法：删除消息（成对删除逻辑：删除指定消息及其关联的问/答）"""
        # 1. 查询目标消息
        stmt = select(Message).where(Message.id == message_id)
        result = await self.session.exec(stmt)
        target_message = result.one_or_none()
        
        if not target_message:
            raise ValueError("消息不存在")
            
        # 2. 验证消息所属会话属于当前用户 (安全性检查)
        conv_stmt = select(Conversation).where(Conversation.id == target_message.conversation_id, Conversation.user_id == user_id)
        conv_result = await self.session.exec(conv_stmt)
        conversation = conv_result.one_or_none()
        
        if not conversation:
             raise ValueError("无权操作此消息")

        # 3. 查找关联消息并成对删除
        # 获取会话下所有未删除消息，按时间排序
        all_msgs = await self._get_messages_by_conversation_id(target_message.conversation_id)
        
        # 找到目标消息的索引
        target_index = -1
        for i, msg in enumerate(all_msgs):
            if msg.id == target_message.id:
                target_index = i
                break
        
        if target_index == -1:
             raise ValueError("消息索引定位失败")

        messages_to_delete = [target_message]

        # 逻辑：
        # 如果是 user 消息 (index)，检查 index+1 是否是 assistant 消息，如果是则一起删除
        # 如果是 assistant 消息 (index)，检查 index-1 是否是 user 消息，如果是则一起删除
        
        if target_message.role == 'user':
            # 尝试删除后续的回答
            if target_index + 1 < len(all_msgs):
                next_msg = all_msgs[target_index + 1]
                if next_msg.role == 'assistant':
                    messages_to_delete.append(next_msg)
        
        elif target_message.role == 'assistant':
            # 尝试删除前置的提问
            if target_index - 1 >= 0:
                prev_msg = all_msgs[target_index - 1]
                if prev_msg.role == 'user':
                    messages_to_delete.append(prev_msg)

        # 执行删除
        for msg in messages_to_delete:
            msg.is_deleted = True
            self.session.add(msg)
            
        return True

















