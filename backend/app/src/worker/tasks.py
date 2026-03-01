import asyncio
import json
from typing import Dict, Any, List
from uuid import UUID
from celery import shared_task
from sqlmodel import select
from app.src.worker.celery_app import celery_app
# 切换回异步配置
from app.src.common.config.prosgresql_config import async_db_manager
from app.src.model.conversation_models import Conversation, Message
from app.src.model.account_model import Patient
from app.src.service.language_model_service import LanguageModelService, ModelConfigService, ModelProviderService

async def process_profile_update(conversation_id: str, user_id: str):
    """
    异步执行画像更新逻辑 (复用 LanguageModelService)
    """
    # 确保数据库连接已初始化
    if not async_db_manager.async_engine:
        await async_db_manager.init()
        
    async with async_db_manager.async_session_factory() as session:
        # 初始化服务层
        provider_service = ModelProviderService(session)
        config_service = ModelConfigService(session, provider_service)
        llm_service = LanguageModelService(session, config_service)
        
        # 1. 获取基础数据
        conversation = await session.get(Conversation, conversation_id)
        if not conversation:
            print(f"Conversation {conversation_id} not found")
            return

        stmt = select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at)
        result = await session.exec(stmt)
        messages = result.all()
        
        stmt_patient = select(Patient).where(Patient.account_id == UUID(user_id))
        result_patient = await session.exec(stmt_patient)
        patient = result_patient.first()
        
        if not patient:
            print(f"Patient profile not found for user {user_id}")
            return
            
        current_base_profile = patient.base_profile or {}
        
        # 2. 自动选择模型 (复用 Service 逻辑)
        # 获取用户所有可用模型
        providers = await llm_service.get_providers_with_models(user_id=UUID(user_id))
        
        target_model_id = None
        target_provider_id = None
        target_model_name = None
        
        # 策略：选择第一个启用的模型（不再限制 model_type，因为很多模型没有设置这个字段）
        found = False
        for p in providers:
            if not p.get("is_enabled"): continue
            # 确保提供商有 API Key 配置
            if not p.get("api_key"): continue
            for m in p.get("models", []):
                if m.get("is_enabled"):
                    target_model_id = m["id"]
                    target_provider_id = p["id"]
                    target_model_name = m["model_name"]
                    found = True
                    break
            if found: break
        
        if not found:
            print(f"No available chat model found for user {user_id}")
            return
            
        print(f"Using model {target_model_name} for background analysis.")

        # 3. 构造 Prompt
        history_text = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        prompt = f"""
        你是专业的中医健康分析师。请根据以下对话历史和用户当前的健康档案，分析并更新用户的长期健康画像。
        
        【当前健康档案】：
        {json.dumps(current_base_profile, ensure_ascii=False)}
        
        【对话历史】：
        {history_text}
        
        【任务】：
        请提取或更新以下长期健康特征（如果对话中包含相关新信息）：
        1. 体质类型 (Constitution Type)
        2. 既往病史 (Medical History)
        3. 家族病史 (Family History)
        4. 过敏信息 (Allergy Info)
        5. 长期生活习惯 (Lifestyle)
        
        【输出格式】：
        只返回一个 JSON 对象，包含更新后的字段。如果没有新信息，对应字段可省略或保持原样。
        不要返回任何解释性文字。
        """
        
        # 4. 准备客户端并调用 (复用 Service 逻辑)
        try:
            # 这一步会自动处理：获取配置、解密Key、合并用户偏好参数
            client, params = await llm_service.prepare_chat_completion(
                user_id=UUID(user_id),
                model_id=UUID(target_model_id),
                provider_id=UUID(target_provider_id),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024
            )
            
            # 执行调用
            response = await client.chat.completions.create(**params)
            content = response.choices[0].message.content
            
            # 5. 解析结果并更新
            cleaned_content = content.replace("```json\n", "").replace("```\n", "").replace("```", "").strip()
            start = cleaned_content.find("{")
            end = cleaned_content.rfind("}")
            
            if start != -1 and end != -1:
                analyzed_data = json.loads(cleaned_content[start:end+1])
                if analyzed_data:
                    new_profile = {**current_base_profile, **analyzed_data}
                    patient.base_profile = new_profile
                    session.add(patient)
                    await session.commit()
                    print(f"Base profile updated for user {user_id}")
                else:
                    print("No new data extracted.")
            else:
                print(f"JSON parse failed: {content[:100]}...")
                
        except Exception as e:
            print(f"LLM analysis failed: {e}")


@celery_app.task(name="app.src.worker.tasks.update_base_profile_task")
def update_base_profile_task(conversation_id: str, user_id: str):
    """
    Celery 任务入口 (同步)
    通过 asyncio.run 桥接异步业务逻辑
    """
    print(f"Task started: update_base_profile_task for {user_id}")
    try:
        asyncio.run(process_profile_update(conversation_id, user_id))
    except Exception as e:
        print(f"Task failed: {e}")
