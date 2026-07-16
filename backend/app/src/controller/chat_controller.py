from fastapi import APIRouter, Request, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

from app.src.dependencies.dependency import ChatServiceDep
from app.src.schema.chat_schema import ChatRequest, PersonaAnalysisRequest, ChatResumeRequest
from app.src.utils import get_logger

from app.src.response.utils import success_200

router = APIRouter(prefix="/api/v1/chat", tags=["聊天模块"])

logger = get_logger("chat_controller")

@router.post("/generate")
async def chat_generate(request: Request,
                        chat_service: ChatServiceDep,
                        chat_request: ChatRequest,
                        background_tasks: BackgroundTasks):  # ✅ 添加 BackgroundTasks
    """
    生成聊天回复 - 支持流式输出
    如果 chat_request.stream=True，返回 SSE 流
    否则返回完整响应
    """
    logger.info(f"开始生成，流式模式={chat_request.stream}")
    
    response = await chat_service.generate_clg_agenthat(chat_request,background_tasks)
    
    # 如果是流式生成器，返回 StreamingResponse
    if hasattr(response, '__aiter__'):
        return StreamingResponse(
            response,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    # 非流式响应
    return success_200(data=response)

@router.post("/analyze_persona")
async def analyze_persona(request: Request,
                          chat_service: ChatServiceDep,
                          analysis_request: PersonaAnalysisRequest):
    logger.info("开始分析用户画像")
    response = await chat_service.analyze_persona(analysis_request)
    return success_200(data=response)

@router.post("/resume")
async def chat_resume(request: Request,
                      chat_service: ChatServiceDep,
                      resume_request: ChatResumeRequest,
                      background_tasks: BackgroundTasks):
    """
    恢复被 interrupt 暂停的聊天 - 用户追问回答后调用
    """
    logger.info(f"恢复聊天，thread_id={resume_request.thread_id}")
    response = await chat_service.resume_agent_chat(resume_request, background_tasks)
    if hasattr(response, '__aiter__'):
        return StreamingResponse(
            response,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    return success_200(data=response)
