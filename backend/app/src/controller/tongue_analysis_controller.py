"""
Tongue Analysis Controller
舌诊分析API控制器
"""

from fastapi import APIRouter, HTTPException
from sqlalchemy import func
from sqlmodel import select
from uuid import UUID

from app.src.common.config.prosgresql_config import SessionDep
from app.src.common.context import get_current_user_id
from app.src.common.decorators import require_login
from app.src.model.medical_models import TongueAnalysis
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
@require_login
async def analyze_tongue_image(request: TongueAnalysisRequest):
    """
    分析舌诊图片

    Args:
        request: 包含图片URL和可选补充信息的请求

    Returns:
        TongueAnalysisResponse: 舌诊分析结果
    """
    try:
        image_value = request.image_url.strip()
        if image_value.lower().startswith(("http://", "https://", "file://")):
            raise HTTPException(
                status_code=400,
                detail="旧舌诊接口禁止读取远程 URL，请使用认证附件上传接口",
            )
        if len(image_value) > 12 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="舌像数据过大")
        analyzer = TongueAnalyzer()
        result = await analyzer.analyze_tongue_image(
            image_url=image_value,
            additional_info=request.additional_info
        )

        return TongueAnalysisResponse(
            tongue_color=result.tongue_color,
            tongue_shape=result.tongue_shape,
            coating_color=result.coating_color,
            coating_texture=result.coating_quality,
            analysis=result.analysis,
            syndrome_hints=result.syndrome_hints,
            suggestions=[]  # 可以根据分析结果生成建议
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"舌诊分析失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"舌诊分析失败: {str(e)}"
        )


@router.get("/history", response_model=TongueHistoryResponse)
@require_login
async def get_tongue_history(
    session: SessionDep,
    page: int = 1,
    page_size: int = 10,
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
    page = max(page, 1)
    page_size = min(max(page_size, 1), 100)
    user_id = UUID(str(get_current_user_id()))
    total_result = await session.exec(
        select(func.count(TongueAnalysis.id)).where(
            TongueAnalysis.user_id == user_id
        )
    )
    total = int(total_result.one() or 0)
    result = await session.exec(
        select(TongueAnalysis)
        .where(TongueAnalysis.user_id == user_id)
        .order_by(TongueAnalysis.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    items = []
    for record in result.all():
        payload = record.analysis_result or {}
        hints = payload.get("syndrome_hints") or []
        items.append(TongueHistoryItem(
            id=str(record.id),
            created_at=record.created_at.isoformat(),
            image_url=record.image_url,
            tongue_color=record.color_analysis or payload.get("tongue_color") or "",
            coating_color=record.coating_color or payload.get("coating_color") or "",
            syndrome_hints=[str(item) for item in hints],
            analysis_summary=str(payload.get("analysis") or ""),
        ))
    return TongueHistoryResponse(
        total=total,
        items=items,
    )
