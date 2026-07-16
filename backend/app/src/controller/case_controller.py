"""
P2 阶段：病例库 REST API

- GET  /api/v1/cases              用户历次问诊列表
- GET  /api/v1/cases/{case_id}    单次问诊详情（含症状/证型/方剂）
- GET  /api/v1/cases/profile      用户健康档案聚合
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Query

from app.src.common.decorators.auth_decorators import require_login
from app.src.common.context import get_current_user_id
from app.src.dependencies.dependency import CaseServiceDep
from app.src.response.utils import success_200
from app.src.utils import get_logger

logger = get_logger("case_controller")

router = APIRouter(prefix="/api/v1/cases", tags=["病例库"])


def _case_to_dict(case) -> dict:
    return {
        "id": str(case.id),
        "conversation_id": str(case.conversation_id),
        "thread_id": case.thread_id,
        "chief_complaint": case.chief_complaint,
        "complexity_level": case.complexity_level,
        "syndrome_id": case.syndrome_id,
        "syndrome_name": case.syndrome_name,
        "syndrome_confidence": case.syndrome_confidence,
        "created_at": case.created_at.isoformat() if case.created_at else None,
    }


def _detail_to_dict(detail: dict) -> dict:
    case = detail["case"]
    return {
        "case": _case_to_dict(case),
        "diagnosis_text": case.diagnosis_text,
        "symptoms": [s.symptom_name for s in detail["symptoms"]],
        "syndromes": [
            {
                "name": s.syndrome_name,
                "confidence": s.confidence,
                "is_primary": s.is_primary,
            }
            for s in detail["syndromes"]
        ],
        "prescriptions": [
            {
                "name": p.prescription_name,
                "composition": p.composition,
                "usage": p.usage,
                "source": p.source,
                "rank": p.recommendation_rank,
            }
            for p in detail["prescriptions"]
        ],
    }


def _profile_to_dict(profile) -> Optional[dict]:
    if profile is None:
        return None
    return {
        "user_id": str(profile.user_id),
        "constitution": profile.constitution,
        "chronic_conditions": profile.chronic_conditions or [],
        "allergies": profile.allergies or [],
        "total_cases": profile.total_cases,
        "last_case_at": profile.last_case_at.isoformat() if profile.last_case_at else None,
        "most_common_syndrome": profile.most_common_syndrome,
        "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
    }


@router.get("")
@require_login
async def list_my_cases(
    case_service: CaseServiceDep,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """当前用户历次问诊列表（按时间倒序）。"""
    user_id = UUID(get_current_user_id())
    cases = await case_service.list_user_cases(user_id, limit=limit, offset=offset)
    return success_200(data={
        "total": len(cases),
        "items": [_case_to_dict(c) for c in cases],
    })


@router.get("/profile")
@require_login
async def get_my_health_profile(case_service: CaseServiceDep):
    """当前用户的健康档案聚合（cases 触发器维护）。"""
    user_id = UUID(get_current_user_id())
    profile = await case_service.get_health_profile(user_id)
    return success_200(data=_profile_to_dict(profile))


@router.get("/{case_id}")
@require_login
async def get_case_detail(case_id: UUID, case_service: CaseServiceDep):
    """单次问诊详情：病例 + 症状 + 证型 + 方剂。仅返回当前用户自己的。"""
    user_id = UUID(get_current_user_id())
    detail = await case_service.get_case_detail(case_id, user_id)
    if detail is None:
        return success_200(data=None, message="病例不存在或不属于当前用户")
    return success_200(data=_detail_to_dict(detail))
