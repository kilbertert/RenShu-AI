"""
Tongue Analysis Schema
舌诊分析请求/响应模式
"""

from typing import Optional
from pydantic import BaseModel, Field


class TongueAnalysisRequest(BaseModel):
    """舌诊分析请求"""
    image_url: str = Field(
        description="图片URL或base64编码的图片数据"
    )
    additional_info: Optional[str] = Field(
        default=None,
        description="用户补充信息，如症状描述"
    )


class TongueAnalysisResponse(BaseModel):
    """舌诊分析响应"""
    tongue_color: str = Field(
        default="",
        description="舌色（淡红/红/绛/紫/淡白）"
    )
    tongue_shape: str = Field(
        default="",
        description="舌形（胖大/瘦薄/齿痕/裂纹）"
    )
    coating_color: str = Field(
        default="",
        description="苔色（白/黄/灰/黑）"
    )
    coating_texture: str = Field(
        default="",
        description="苔质（薄/厚/腻/燥/滑）"
    )
    analysis: str = Field(
        default="",
        description="舌诊分析详情"
    )
    syndrome_hints: list[str] = Field(
        default_factory=list,
        description="证型提示"
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="养生建议"
    )


class TongueHistoryItem(BaseModel):
    """舌诊历史记录项"""
    id: str = Field(description="记录ID")
    created_at: str = Field(description="创建时间")
    image_url: Optional[str] = Field(default=None, description="图片URL")
    tongue_color: str = Field(default="", description="舌色")
    coating_color: str = Field(default="", description="苔色")
    syndrome_hints: list[str] = Field(default_factory=list, description="证型提示")
    analysis_summary: str = Field(default="", description="分析摘要")


class TongueHistoryResponse(BaseModel):
    """舌诊历史响应"""
    total: int = Field(description="总记录数")
    items: list[TongueHistoryItem] = Field(
        default_factory=list,
        description="历史记录列表"
    )
