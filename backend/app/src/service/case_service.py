"""
P2 阶段：病例库服务 (case library service)

把 LangGraph diagnose state 里的病例信息持久化到 PostgreSQL，
并提供跨会话查询 + 健康档案聚合。
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.src.model.case_models import (
    Case,
    CasePrescription,
    CaseSymptom,
    CaseSyndrome,
    UserHealthProfile,
)
from app.src.utils import get_logger

logger = get_logger("case_service")


class CaseService:
    """病例落库 + 跨会话查询服务。"""

    def __init__(self, session: AsyncSession):
        self.session = session

    # ------------------------------------------------------------------ writes

    async def save_case_from_state(
        self,
        user_id: UUID,
        conversation_id: UUID,
        thread_id: Optional[str],
        state: dict[str, Any],
    ) -> Optional[Case]:
        """从 LangGraph state 提取病例字段并落库。

        state 期望字段（容错读取，缺一不可时用占位）：
            - collected_info.dict 或 collected_info（内含 chief_complaint、各类问答）
            - syndrome_result.{syndrome_name, syndrome_id, confidence}  OR  state.syndrome_name
            - answer / diagnose_text（完整辨证文本）
            - complexity_level（simple / moderate / complex）
            - prescriptions: list[{name, composition, usage, source}]
        """
        try:
            chief_complaint = self._extract_chief_complaint(state)
            symptoms = self._extract_symptoms(state)
            syndrome = self._extract_syndrome(state)
            complexity = self._extract_complexity(state)
            diagnosis_text = self._extract_diagnosis_text(state)
            prescriptions = self._extract_prescriptions(state)

            case = Case(
                user_id=user_id,
                conversation_id=conversation_id,
                thread_id=thread_id,
                chief_complaint=chief_complaint,
                complexity_level=complexity,
                syndrome_id=syndrome.get("id"),
                syndrome_name=syndrome.get("name"),
                syndrome_confidence=syndrome.get("confidence"),
                diagnosis_text=diagnosis_text,
            )
            self.session.add(case)
            await self.session.flush()

            for s in symptoms:
                self.session.add(CaseSymptom(
                    case_id=case.id,
                    symptom_name=s["name"],
                    category=s.get("category"),
                    severity=s.get("severity"),
                ))

            if syndrome.get("name"):
                self.session.add(CaseSyndrome(
                    case_id=case.id,
                    syndrome_name=syndrome["name"],
                    confidence=syndrome.get("confidence"),
                    is_primary=True,
                ))

            for rank, p in enumerate(prescriptions, start=1):
                self.session.add(CasePrescription(
                    case_id=case.id,
                    prescription_name=p["name"],
                    composition=p.get("composition"),
                    usage=p.get("usage"),
                    source=p.get("source"),
                    recommendation_rank=rank,
                ))

            await self.session.commit()
            await self.session.refresh(case)
            logger.info(
                "病例已落库: case_id=%s user_id=%s syndrome=%s symptoms=%d prescriptions=%d",
                case.id, user_id, case.syndrome_name, len(symptoms), len(prescriptions),
            )
            return case
        except Exception as exc:
            await self.session.rollback()
            logger.error("病例落库失败 user_id=%s: %s", user_id, exc, exc_info=True)
            return None

    # ----------------------------------------------------------------- reads

    async def list_user_cases(
        self, user_id: UUID, limit: int = 50, offset: int = 0,
    ) -> list[Case]:
        """按时间倒序列出用户病例。"""
        stmt = (
            select(Case)
            .where(Case.user_id == user_id)
            .order_by(Case.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_case_detail(self, case_id: UUID, user_id: UUID) -> Optional[dict[str, Any]]:
        """拉取单次问诊的完整快照（病例 + 症状 + 证型 + 方剂）。"""
        case_row = await self.session.get(Case, case_id)
        if case_row is None or case_row.user_id != user_id:
            return None

        symptoms = (await self.session.execute(
            select(CaseSymptom).where(CaseSymptom.case_id == case_id)
        )).scalars().all()
        syndromes = (await self.session.execute(
            select(CaseSyndrome).where(CaseSyndrome.case_id == case_id)
        )).scalars().all()
        prescriptions = (await self.session.execute(
            select(CasePrescription)
            .where(CasePrescription.case_id == case_id)
            .order_by(CasePrescription.recommendation_rank)
        )).scalars().all()

        return {
            "case": case_row,
            "symptoms": list(symptoms),
            "syndromes": list(syndromes),
            "prescriptions": list(prescriptions),
        }

    async def get_health_profile(self, user_id: UUID) -> Optional[UserHealthProfile]:
        """取用户健康档案（由 cases 触发器维护）。无记录时返回 None。"""
        return await self.session.get(UserHealthProfile, user_id)

    # ---------------------------------------------------------------- extract

    @staticmethod
    def _extract_chief_complaint(state: dict[str, Any]) -> str:
        collected = state.get("collected_info")
        if isinstance(collected, dict):
            cc = collected.get("chief_complaint")
            if cc:
                return str(cc)
        for msg in state.get("messages", []) or []:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if content:
                    return str(content)[:500]
        return "未明确"

    @staticmethod
    def _extract_symptoms(state: dict[str, Any]) -> list[dict[str, Any]]:
        """从 collected_info 的十问字段拆出症状条目。"""
        out: list[dict[str, Any]] = []
        collected = state.get("collected_info")
        if not isinstance(collected, dict):
            for s in state.get("symptoms", []) or []:
                if isinstance(s, str):
                    out.append({"name": s})
                elif isinstance(s, dict) and s.get("name"):
                    out.append(s)
            return out

        for key in ("head_body", "cold_heat", "sweat", "urine_stool",
                    "diet", "chest_abdomen", "sleep", "emotion"):
            value = collected.get(key)
            if not value or not isinstance(value, str):
                continue
            parts = [p.strip() for p in value.replace("、", ",").split(",") if p.strip()]
            for p in parts:
                out.append({"name": p, "category": key})
        return out

    @staticmethod
    def _extract_syndrome(state: dict[str, Any]) -> dict[str, Any]:
        result = state.get("syndrome_result")
        if isinstance(result, dict):
            return {
                "id": result.get("syndrome_id") or result.get("id"),
                "name": result.get("syndrome_name") or result.get("name"),
                "confidence": result.get("confidence"),
            }
        return {
            "id": state.get("syndrome_id"),
            "name": state.get("syndrome_name"),
            "confidence": state.get("syndrome_confidence"),
        }

    @staticmethod
    def _extract_complexity(state: dict[str, Any]) -> Optional[str]:
        c = state.get("complexity_level")
        if c in ("simple", "moderate", "complex"):
            return c
        diagnose_stage = state.get("diagnose_stage")
        if isinstance(diagnose_stage, dict):
            inner = diagnose_stage.get("complexity")
            if isinstance(inner, dict) and inner.get("level") in ("simple", "moderate", "complex"):
                return inner["level"]
        return None

    @staticmethod
    def _extract_diagnosis_text(state: dict[str, Any]) -> Optional[str]:
        text = state.get("answer") or state.get("diagnosis_text")
        if isinstance(text, str) and text.strip():
            return text
        return None

    @staticmethod
    def _extract_prescriptions(state: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for p in state.get("prescriptions", []) or []:
            if isinstance(p, str):
                out.append({"name": p})
            elif isinstance(p, dict) and p.get("name"):
                out.append(p)
        return out
