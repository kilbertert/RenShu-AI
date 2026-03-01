"""
Tongue Analysis Controller
舌诊分析API控制器
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from app.src.schema.tongue_analysis_schema import (
    TongueAnalysisRequest,
    TongueAnalysisResponse,
    TongueHistoryResponse,
    TongueHistoryItem,
)
from app.src.agent.tcm_image_analyzer import TongueAnalyzer
from app.src.utils import get_logger

logger = get_logger("tongue_analysis")

router = APIRouter(prefix="/api/v1/tongue", tags=["舌诊分析"])


@router.post("/analyze", response_model=TongueAnalysisResponse)
async def analyze_tongue_image(request: TongueAnalysisRequest):
    """
    分析舌诊图片

    Args:
        request: 包含图片URL和可选补充信息的请求

    Returns:
        TongueAnalysisResponse: 舌诊分析结果
    """
    try:
        analyzer = TongueAnalyzer()
        result = await analyzer.analyze_tongue_image(
            image_url=request.image_url,
            additional_info=request.additional_info
        )

        return TongueAnalysisResponse(
            tongue_color=result.tongue_color,
            tongue_shape=result.tongue_shape,
            coating_color=result.coating_color,
            coating_texture=result.coating_texture,
            analysis=result.analysis,
            syndrome_hints=result.syndrome_hints,
            suggestions=[]  # 可以根据分析结果生成建议
        )

    except Exception as e:
        logger.error(f"舌诊分析失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"舌诊分析失败: {str(e)}"
        )


@router.get("/history", response_model=TongueHistoryResponse)
async def get_tongue_history(
    page: int = 1,
    page_size: int = 10,
    user_id: Optional[str] = None
):
    """
    获取舌诊历史记录

    Args:
        page: 页码
        page_size: 每页数量
        user_id: 用户ID（可选）

    Returns:
        TongueHistoryResponse: 历史记录列表
    """
    # TODO: 实现历史记录存储和查询
    # 目前返回空列表
    return TongueHistoryResponse(
        total=0,
        items=[]
    )
