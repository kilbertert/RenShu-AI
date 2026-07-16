import uuid
import asyncio
import json
from typing import AsyncGenerator
from fastapi import BackgroundTasks
from app.src.model.conversation_models import Message, Conversation
from app.src.model.account_model import Patient
from sqlmodel import select, desc
from app.src.schema.chat_schema import ChatRequest, PersonaAnalysisRequest, ChatResumeRequest
from app.src.service.base_service import BaseService
from app.src.service.conversation_service import ConversationService
from app.src.service.language_model_service import LanguageModelService
from app.src.common.decorators import require_login
from app.src.common.context import get_current_user_id
from app.src.worker.tasks import update_base_profile_task
from uuid import UUID
from app.src.common.config.redis_config import redis_manager
from app.src.utils import get_logger

# 导入TCM Agent服务
from app.src.agent.tcm_service import get_tcm_agent_service
from app.src.agent.tcm_states import TCMOutputState

logger = get_logger("chat_service")


class ChatService:
    def __init__(self,
                 conversation_service: ConversationService,
                 model_service: LanguageModelService
                 ):
        self.conversation_service = conversation_service
        self.model_service = model_service
        # 初始化TCM Agent服务
        self.tcm_agent_service = get_tcm_agent_service()

    # ========== 使用装饰器的方法 ==========

    # @require_login
    # async def generate_chat(self, chat_request: ChatRequest, background_tasks: BackgroundTasks = None):
    #     """
    #     生成聊天回复（需要登录）- 支持流式输出
    #     :param chat_request: 聊天请求
    #     :param background_tasks: FastAPI 后台任务对象
    #     :return: 聊天回复或流式生成器
    #     """
    #     user_id = get_current_user_id()
    #
    #     # 如果请求流式输出，返回异步生成器
    #     if chat_request.stream:
    #         return self._generate_stream(chat_request, user_id, background_tasks)
    #     else:
    #         # 非流式输出
    #         return await self._generate(chat_request, user_id)

    @require_login
    async def analyze_persona(self, request: PersonaAnalysisRequest):
        """
        分析用户画像（需要登录）- 企业级优化
        优化点：
        1. 缓存 base_profile（用户基础信息）
        2. 快速提交事务
        3. 并行化 LLM 调用
        """
        user_id = get_current_user_id()
        import json
        from app.src.utils.token_counter import estimate_tokens
        
        conversation_id = request.conversation_id




        stmt_patient = select(Patient).where(Patient.account_id == user_id)
        result_patient = await self.conversation_service.session.exec(stmt_patient)
        patient = result_patient.first()
        base_profile = patient.base_profile if patient else {}



        config = request.model_configuration
        analyzed_data = request.current_persona

        # 🚀 优化：如果前端没传画像，尝试从数据库加载
        if not analyzed_data and conversation_id:
            try:
                # 使用 conversation_service 的 session 获取会话对象
                conversation = await self.conversation_service.session.get(Conversation, conversation_id)
                if conversation and conversation.session_metadata:
                    analyzed_data = conversation.session_metadata
                    logger.info(f"📂 从数据库加载了历史画像, conversation_id={conversation_id}")
            except Exception as e:
                logger.warning(f"⚠️ 加载历史画像失败: {e}")

        # 如果还是为空，设置初始默认结构
        if not analyzed_data:
            analyzed_data = {
                "age": "",
                "gender": "",
                "healthScore": 100,
                "chiefComplaint": "待分析...",
                "suspectedDiagnosis": "分析待定",
                "recommendedTreatment": " wellness 建议"
            }
            logger.info("🆕 使用默认初始画像")

        # 构造 Prompt
        prompt = f"""
        你是一名动态医疗记录员。
        你的任务是根据用户最新的输入：“{request.text}” 来更新会话画像。
    
        当前会话画像：
        {json.dumps(analyzed_data, ensure_ascii=False)}
    
        指令：
        1. 提取“年龄”和“性别”（如果本次会话中提到）。如果没有提到，保持为 null 或空字符串。
        2. 使用本次会话中提到的最新症状增量总结“主诉”。
        3. 基于累积的主诉，再次推断出“疑似诊断”、“禁忌症”和“建议治疗方案”。
        4. 评估用户的整体健康状况，给出一个 0-100 的整数评分（healthScore）。
        5. 保持描述简洁（尽可能在10个字以内）。
    
        仅返回符合此结构的有效 JSON：
        {{
          "age": "string",
          "gender": "string",
          "healthScore": 0,
          "chiefComplaint": "string",
          "suspectedDiagnosis": "string",
          "recommendedTreatment": "string"
        }}
        """
            


        try:
            # 获取 LLM 客户端
            client = await self.model_service.get_client(
                user_id=UUID(user_id),
                provider_id=config.provider_id
            )
    
            # 🚀 直接调用 LLM（快速，max_tokens=512）
            response = await client.chat.completions.create(
                model=config.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=0.1, 
                top_p=0.95,
                max_tokens=512
            )
                
            content = response.choices[0].message.content
            cleaned_content = content.replace("```json\n", "").replace("```\n", "").replace("```", "").strip()
            new_session_data = json.loads(cleaned_content)
                
            analyzed_data = { **new_session_data}
            analyzed_data['baseProfile'] = base_profile if base_profile else []
            logger.info(f"✅ [analyze_persona] LLM 分析完成，conversation_id={conversation_id}")
                
            # 2. Update Session Metadata (快速提交，避免长时间锁定)
            if conversation_id:
                try:
                    title = request.text[:50] if request.text else "New Chat"
                    conversation = await self.conversation_service._get_or_create_conversation(conversation_id, user_id, title)
                        
                    if conversation:
                        conversation.session_metadata = analyzed_data
                        self.conversation_service.session.add(conversation)
                        # ⚡ 关键：立即提交，不等待请求结束
                        await self.conversation_service.session.commit()
                        logger.info(f"✅ [analyze_persona] 会话画像已更新并提交，conversation_id={conversation_id}")
                except Exception as db_error:
                    logger.warning(f"⚠️ [analyze_persona] 数据库更新失败: {db_error}")
                    await self.conversation_service.session.rollback()
    
            return analyzed_data
                
        except Exception as e:
            logger.error(f"❌ [analyze_persona] 画像分析失败: {e}")
            return request.current_persona

    # ========== 内部方法 ==========
    #
    # async def _generate_stream(self, chat_request: ChatRequest, user_id: str, background_tasks: BackgroundTasks = None) -> AsyncGenerator[str, None]:
    #     """流式生成聊天回复 - 企业级优化
    #
    #     优化点：
    #     1. SSE 流式输出，降低 TTFB（首字节时间）
    #     2. 并行处理 DB 保存和 LLM 生成
    #     3. 缓存热点配置
    #     4. 🚀 智能批量发送：当内容累积较多时，合并发送以提升渲染效率
    #     """
    #     from app.src.utils.token_counter import estimate_tokens
    #     import time
    #
    #     conversation_id = chat_request.conversation_id
    #     title = chat_request.query[:50]
    #
    #     # 1. 快速保存用户消息（短事务）
    #     conversation = await self.conversation_service._get_or_create_conversation(conversation_id, user_id, title)
    #
    #     user_message = Message(
    #         conversation_id=conversation_id,
    #         role="user",
    #         content=chat_request.query
    #     )
    #     self.conversation_service.session.add(user_message)
    #     await self.conversation_service.session.flush()
    #
    #     user_input_tokens = estimate_tokens(chat_request.query)
    #     conversation.accumulated_tokens += user_input_tokens
    #     conversation.total_tokens += user_input_tokens
    #     self.conversation_service.session.add(conversation)
    #
    #     # 提前提交，释放锁
    #     await self.conversation_service.session.commit()
    #     logger.info(f"✅ [流式] 用户消息已保存，conversation_id={conversation_id}")
    #
    #     # 2. 获取历史消息
    #     history_stmt = select(Message).where(
    #         Message.conversation_id == conversation_id
    #     ).order_by(Message.created_at)
    #
    #     history_result = await self.conversation_service.session.exec(history_stmt)
    #     history_messages = history_result.all()
    #
    #     messages_payload = [
    #         {"role": msg.role, "content": msg.content}
    #         for msg in history_messages
    #     ]
    #
    #     # 3. 流式调用 LLM
    #     accumulated_content = []
    #     try:
    #         config = chat_request.model_configuration
    #
    #         # 🔥 关键：使用流式模式
    #         response = await self.model_service.generate_chat_completion(
    #             user_id=UUID(user_id),
    #             model_id=config.model_id,
    #             provider_id=config.provider_id,
    #             model_name=config.model_name,
    #             messages=messages_payload,
    #             stream=True,  # 开启流式
    #             temperature=config.temperature,
    #             top_p=config.top_p,
    #             max_tokens=config.max_tokens
    #         )
    #
    #         # 4. 🚀 智能批量流式输出
    #         # 策略：累积内容，每 50ms 或累积超过 50 字符时发送一次
    #         batch_buffer = []
    #         last_send_time = time.time()
    #         BATCH_INTERVAL = 0.05  # 50ms
    #         BATCH_SIZE_THRESHOLD = 50  # 50 字符
    #
    #         async for chunk in response:
    #             if chunk.choices and len(chunk.choices) > 0:
    #                 delta = chunk.choices[0].delta
    #                 if delta.content:
    #                     accumulated_content.append(delta.content)
    #                     batch_buffer.append(delta.content)
    #
    #                     current_time = time.time()
    #                     batch_content = "".join(batch_buffer)
    #
    #                     # 检查是否需要发送：时间间隔超过 50ms 或内容超过 50 字符
    #                     should_send = (
    #                         current_time - last_send_time >= BATCH_INTERVAL or
    #                         len(batch_content) >= BATCH_SIZE_THRESHOLD
    #                     )
    #
    #                     if should_send and batch_content:
    #                         # 输出 SSE 格式（合并后的内容）
    #                         yield f"data: {json.dumps({'content': batch_content}, ensure_ascii=False)}\n\n"
    #                         batch_buffer = []
    #                         last_send_time = current_time
    #
    #         # 发送剩余的 buffer 内容
    #         if batch_buffer:
    #             remaining_content = "".join(batch_buffer)
    #             yield f"data: {json.dumps({'content': remaining_content}, ensure_ascii=False)}\n\n"
    #
    #         # 5. ✅ 流结束后，注册后台保存任务
    #         full_content = "".join(accumulated_content)
    #         if full_content and background_tasks:
    #             background_tasks.add_task(self._save_ai_message, conversation_id, user_id, full_content)
    #             logger.info(f"🚀 [流式] 已注册后台保存任务，总长度={len(full_content)}")
    #         yield "data: [DONE]\n\n"
    #     except Exception as e:
    #         logger.error(f"❌ [流式] 错误: {e}")
    #         await self.conversation_service.session.rollback()
    #         yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
    #
    #
    #
    async def _save_ai_message(self, conversation_id: str, user_id: str, content: str):
        """异步保存 AI 消息（不阻塞流式输出）"""
        try:
            from app.src.utils.token_counter import estimate_tokens
            
            ai_message = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=content
            )
            self.conversation_service.session.add(ai_message)
            await self.conversation_service.session.flush()
            
            # 更新 token 统计
            conversation = await self.conversation_service.session.get(Conversation, conversation_id)
            if conversation:
                ai_output_tokens = estimate_tokens(content)
                conversation.accumulated_tokens += ai_output_tokens
                conversation.total_tokens += ai_output_tokens
                conversation.updated_at = ai_message.created_at
                
                # 检查阈值
                THRESHOLD = 2000
                if conversation.accumulated_tokens >= THRESHOLD:
                    conversation.accumulated_tokens = 0
                    update_base_profile_task.delay(str(conversation.id), str(user_id))
                
                self.conversation_service.session.add(conversation)
                await self.conversation_service.session.commit()
                
            logger.info(f"✅ [流式] AI 消息已异步保存")
        except Exception as e:
            logger.error(f"❌ [流式] 保存 AI 消息失败: {e}")
            await self.conversation_service.session.rollback()

    @require_login
    async def generate_clg_agenthat(self, chat_request: ChatRequest, background_tasks: BackgroundTasks = None):
        """
        使用TCM多智能体架构生成聊天回复

        Args:
            chat_request: 聊天请求
            background_tasks: FastAPI 后台任务对象

        Returns:
            聊天回复或流式生成器
        """
        user_id = get_current_user_id()

        # 获取模型配置（API Key, Base URL 等）
        model_config = await self._get_llm_config_for_agent(user_id, chat_request.model_configuration)

        # 如果请求流式输出，返回异步生成器
        if chat_request.stream:
            return self._generate_tcm_agent_stream(chat_request, user_id, model_config, background_tasks)
        else:
            # 非流式输出
            return await self._generate_tcm_agent(chat_request, user_id, model_config)

    async def _get_llm_config_for_agent(self, user_id: str, model_configuration) -> dict:
        """
        获取 TCM Agent 所需的 LLM 配置

        从用户配置中获取 API Key 和 Base URL，避免硬编码

        Args:
            user_id: 用户ID
            model_configuration: 前端传入的模型配置

        Returns:
            dict: 包含 provider_name, model_name, api_key, base_url 等
        """
        from app.src.utils.auth_utils import decrypt_api_key

        try:
            raw_provider_id = model_configuration.provider_id

            # 1. 解析 provider：优先按 UUID 查，失败则按 name 查
            provider = None
            try:
                provider_id = UUID(raw_provider_id)
                provider = await self.model_service.model_config_service.provider_service.get(provider_id)
            except (ValueError, TypeError):
                # 不是 UUID，按 name 查
                provider = await self.model_service.model_config_service.provider_service.get_provider_by_name(raw_provider_id)
                if provider:
                    provider_id = provider.id
            if not provider:
                logger.warning(f"供应商不存在: {raw_provider_id}")
                return {}

            # 2. 获取用户配置（API Key, Base URL）
            user_config = await self.model_service.model_config_service.provider_service.get_user_config(
                UUID(user_id), provider_id
            )

            # 3. 解密 API Key
            api_key = None
            if user_config and user_config.api_key:
                api_key = decrypt_api_key(user_config.api_key)

            # 4. 获取 Base URL（用户配置优先，否则使用供应商默认值）
            base_url = None
            if user_config and user_config.base_url_override:
                base_url = user_config.base_url_override
            elif provider.default_base_url:
                base_url = provider.default_base_url

            # 5. 本地服务（如 Ollama）可能不需要 API Key
            if not api_key and base_url and ("localhost" in base_url or "127.0.0.1" in base_url):
                api_key = "ollama"

            logger.info(
                f"获取 LLM 配置成功: provider={provider.name}, "
                f"model={model_configuration.model_name}, has_api_key={bool(api_key)}"
            )

            return {
                "provider_name": provider.name,
                "model_name": model_configuration.model_name,
                "api_key": api_key,
                "base_url": base_url,
                "temperature": model_configuration.temperature or 0.7,
                "top_p": model_configuration.top_p or 1.0,
                "max_tokens": model_configuration.max_tokens or 2000,
            }

        except Exception as e:
            logger.error(f"获取 LLM 配置失败: {e}", exc_info=True)
            return {}

    async def _generate_tcm_agent_stream(self, chat_request: ChatRequest, user_id: str, model_config: dict, background_tasks: BackgroundTasks = None):
        """使用TCM多智能体架构流式生成聊天回复

        tcm_service 返回的 chunk 已经是序列化好的 JSON 字符串，
        直接作为 SSE data 字段透传，不要再包装。

        流前：保存用户消息到数据库
        流中：透传所有 chunk，累积 type=="content" 的文本
        流后：通过 background_tasks 异步保存 AI 消息
        """
        from app.src.utils.token_counter import estimate_tokens

        conversation_id = chat_request.conversation_id
        title = chat_request.query[:50]

        # 1. 保存用户消息（短事务，快速释放锁）
        try:
            conversation = await self.conversation_service._get_or_create_conversation(conversation_id, user_id, title)
            user_message = Message(
                conversation_id=conversation_id,
                role="user",
                content=chat_request.query
            )
            self.conversation_service.session.add(user_message)
            await self.conversation_service.session.flush()

            user_input_tokens = estimate_tokens(chat_request.query)
            conversation.accumulated_tokens += user_input_tokens
            conversation.total_tokens += user_input_tokens
            self.conversation_service.session.add(conversation)
            await self.conversation_service.session.commit()
            logger.info(f"✅ [TCM流式] 用户消息已保存，conversation_id={conversation_id}")
        except Exception as e:
            logger.error(f"❌ [TCM流式] 保存用户消息失败: {e}")
            await self.conversation_service.session.rollback()

        # 2. 流式输出 + 累积 content 类型的文本
        accumulated_content = []
        try:
            async for chunk in self.tcm_agent_service.chat_stream_with_tcm_agent(
                message=chat_request.query,
                user_id=user_id,
                conversation_id=conversation_id,
                provider_name=model_config.get("provider_name"),
                model_name=model_config.get("model_name"),
                api_key=model_config.get("api_key"),
                base_url=model_config.get("base_url"),
                temperature=model_config.get("temperature", 0.7),
                top_p=model_config.get("top_p", 1.0),
                max_tokens=model_config.get("max_tokens", 2000),
                enable_thinking=chat_request.enable_thinking or False,
            ):
                try:
                    parsed = json.loads(chunk)
                    if parsed.get("type") == "content" and parsed.get("content"):
                        accumulated_content.append(parsed["content"])
                except (json.JSONDecodeError, TypeError):
                    pass
                yield f"data: {chunk}\n\n"

            # 3. 流结束后，注册后台保存 AI 消息
            full_content = "".join(accumulated_content)
            if full_content and background_tasks:
                background_tasks.add_task(self._save_ai_message, conversation_id, user_id, full_content)
                logger.info(f"🚀 [TCM流式] 已注册后台保存任务，总长度={len(full_content)}")
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"❌ [TCM流式] 流式生成错误: {e}")
            await self.conversation_service.session.rollback()
            yield f'data: {json.dumps({"type": "error", "content": str(e)}, ensure_ascii=False)}\n\n'
            yield "data: [DONE]\n\n"

    @require_login
    async def resume_agent_chat(self, resume_request: ChatResumeRequest, background_tasks: BackgroundTasks = None):
        """
        恢复被 interrupt 暂停的 TCM Agent 聊天

        Args:
            resume_request: 恢复请求（包含 thread_id 和用户回答）
            background_tasks: FastAPI 后台任务对象

        Returns:
            流式生成器
        """
        user_id = get_current_user_id()
        model_config = await self._get_llm_config_for_agent(user_id, resume_request.model_configuration)
        return self._resume_tcm_agent_stream(resume_request, user_id, model_config, background_tasks)

    async def _resume_tcm_agent_stream(self, resume_request: ChatResumeRequest, user_id: str, model_config: dict, background_tasks: BackgroundTasks = None):
        """恢复被 interrupt 暂停的 TCM Agent 流"""
        from app.src.utils.token_counter import estimate_tokens

        conversation_id = resume_request.conversation_id
        thread_id = resume_request.thread_id

        # 保存用户追问回答到数据库
        try:
            conversation = await self.conversation_service._get_or_create_conversation(
                conversation_id, user_id, resume_request.query[:50]
            )
            user_message = Message(
                conversation_id=conversation_id,
                role="user",
                content=resume_request.query
            )
            self.conversation_service.session.add(user_message)
            await self.conversation_service.session.flush()

            user_input_tokens = estimate_tokens(resume_request.query)
            conversation.accumulated_tokens += user_input_tokens
            conversation.total_tokens += user_input_tokens
            self.conversation_service.session.add(conversation)
            await self.conversation_service.session.commit()
            logger.info(f"✅ [TCM Resume] 用户追问已保存，conversation_id={conversation_id}")
        except Exception as e:
            logger.error(f"❌ [TCM Resume] 保存用户追问失败: {e}")
            await self.conversation_service.session.rollback()

        # 恢复图执行并流式输出
        accumulated_content = []
        try:
            async for chunk in self.tcm_agent_service.resume_stream(
                thread_id=thread_id,
                user_answer=resume_request.query,
            ):
                try:
                    parsed = json.loads(chunk)
                    if parsed.get("type") == "content" and parsed.get("content"):
                        accumulated_content.append(parsed["content"])
                except (json.JSONDecodeError, TypeError):
                    pass
                yield f"data: {chunk}\n\n"

            # 流结束后，注册后台保存 AI 消息
            full_content = "".join(accumulated_content)
            if full_content and background_tasks:
                background_tasks.add_task(self._save_ai_message, conversation_id, user_id, full_content)
                logger.info(f"🚀 [TCM Resume] 已注册后台保存任务，总长度={len(full_content)}")
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"❌ [TCM Resume] 流式恢复错误: {e}")
            yield f'data: {json.dumps({"type": "error", "content": str(e)}, ensure_ascii=False)}\n\n'
            yield "data: [DONE]\n\n"

    async def _generate_tcm_agent(self, chat_request: ChatRequest, user_id: str, model_config: dict):
        """使用TCM多智能体架构生成聊天回复（非流式）"""
        # 调用TCM Agent服务，传入模型配置
        result: TCMOutputState = await self.tcm_agent_service.chat_with_tcm_agent(
            message=chat_request.query,
            user_id=user_id,
            conversation_id=chat_request.conversation_id,
            user_profile={},
            provider_name=model_config.get("provider_name"),
            model_name=model_config.get("model_name"),
            api_key=model_config.get("api_key"),
            base_url=model_config.get("base_url"),
            temperature=model_config.get("temperature", 0.7),
            top_p=model_config.get("top_p", 1.0),
            max_tokens=model_config.get("max_tokens", 2000),
        )

        # 返回格式化的响应
        return {
            "role": "assistant",
            "content": result.answer,
            "query_type": result.query_type,
            "steps": result.steps,
            "syndrome_result": result.syndrome_result,
            "herbs": [h.model_dump() for h in result.herbs] if result.herbs else [],
            "prescriptions": [p.model_dump() for p in result.prescriptions] if result.prescriptions else [],
            "classics": [c.model_dump() for c in result.classics] if result.classics else [],
            "cases": [c.model_dump() for c in result.cases] if result.cases else [],
        }

    # async def _generate(self, chat_request: ChatRequest, user_id: str):
    #     """内部方法：生成聊天回复
    #
    #     企业级优化：
    #     1. 使用 SELECT FOR UPDATE SKIP LOCKED 避免锁等待
    #     2. 提前提交短事务，释放行锁
    #     3. LLM 调用在事务外执行
    #     """
    #     from app.src.utils.token_counter import estimate_tokens
    #
    #     # 1. 获取或创建会话
    #     conversation_id = chat_request.conversation_id
    #     title = chat_request.query[:50]
    #
    #     conversation = await self.conversation_service._get_or_create_conversation(conversation_id, user_id, title)
    #
    #     # 2. 保存用户提问（短事务，快速提交）
    #     user_message = Message(
    #         conversation_id=conversation_id,
    #         role="user",
    #         content=chat_request.query
    #     )
    #     self.conversation_service.session.add(user_message)
    #     await self.conversation_service.session.flush()
    #
    #     # 更新累积 Token
    #     user_input_tokens = estimate_tokens(chat_request.query)
    #     conversation.accumulated_tokens += user_input_tokens
    #     conversation.total_tokens += user_input_tokens
    #     self.conversation_service.session.add(conversation)
    #
    #     # ⚡ 企业级模式：提前提交事务，释放数据库行锁
    #     # 这样 analyze_persona 请求可以立即获取锁并更新 session_metadata
    #     await self.conversation_service.session.commit()
    #     print(f"✅ [generate_chat] 用户消息已保存并提交，conversation_id={conversation_id}")
    #
    #     # 3. 获取历史消息 (用于构建上下文)
    #     # 获取最近 N 条消息
    #     history_stmt = select(Message).where(
    #         Message.conversation_id == conversation_id
    #     ).order_by(Message.created_at)
    #
    #     history_result = await self.conversation_service.session.exec(history_stmt)
    #     history_messages = history_result.all()
    #
    #     # 转换为 LLM 需要的格式
    #     messages_payload = [
    #         {"role": msg.role, "content": msg.content}
    #         for msg in history_messages
    #     ]
    #
    #     try:
    #         config = chat_request.model_configuration
    #         # 🚀 这里调用 LLM，耗时较长，但数据库锁已经释放
    #         response = await self.model_service.generate_chat_completion(
    #             user_id=UUID(user_id),
    #             model_id=config.model_id,
    #             provider_id=config.provider_id,
    #             model_name=config.model_name,
    #             messages=messages_payload,
    #             stream=False,
    #             temperature=config.temperature,
    #             top_p=config.top_p,
    #             max_tokens=config.max_tokens
    #         )
    #
    #         content = response.choices[0].message.content
    #         print(f"✅ [generate_chat] LLM 响应完成，conversation_id={conversation_id}")
    #
    #         # 6. 保存 AI 回复 (重新开启一个新事务)
    #         ai_message = Message(
    #             conversation_id=conversation_id,
    #             role="assistant",
    #             content=content
    #         )
    #         self.conversation_service.session.add(ai_message)
    #         await self.conversation_service.session.flush()
    #
    #         # 重新获取 conversation 对象 (避免 detached 状态)
    #         conversation = await self.conversation_service.session.get(Conversation, conversation_id)
    #
    #         # 更新累积 Token (AI 输出)
    #         ai_output_tokens = estimate_tokens(content)
    #         conversation.accumulated_tokens += ai_output_tokens
    #         conversation.total_tokens += ai_output_tokens
    #
    #         # 更新会话更新时间
    #         conversation.updated_at = ai_message.created_at
    #
    #         # Check Threshold (e.g., 2000 tokens)
    #         # We use a threshold of 2000 (approx 3300 chars CN)
    #         THRESHOLD = 2000
    #         if conversation.accumulated_tokens >= THRESHOLD:
    #             print(f"Token threshold reached ({conversation.accumulated_tokens}). Triggering base profile update.")
    #             # Reset counter
    #             conversation.accumulated_tokens = 0
    #             # Trigger Celery Task
    #             # Pass IDs as strings to Celery
    #             update_base_profile_task.delay(str(conversation.id), str(user_id))
    #
    #         self.conversation_service.session.add(conversation)
    #         await self.conversation_service.session.flush()
    #         # 第二次提交 (保存 AI 响应)
    #         await self.conversation_service.session.commit()
    #         print(f"✅ [generate_chat] AI 响应已保存并提交，conversation_id={conversation_id}")
    #
    #         return {
    #             "role": "assistant",
    #             "content": content
    #         }
    #
    #     except Exception as e:
    #         # 异常已在底层处理或在此捕获
    #         await self.conversation_service.session.rollback()
    #         raise e
    #

























           




















