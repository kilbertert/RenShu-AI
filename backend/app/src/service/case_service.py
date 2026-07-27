"""
P2 阶段：病例库服务 (case library service)

把 LangGraph diagnose state 里的病例信息持久化到 PostgreSQL，
并提供跨会话查询 + 健康档案聚合。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import func, select
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
            diagnosis_result = state.get("diagnosis_result")
            if state.get("error") or not isinstance(diagnosis_result, dict):
                logger.warning("病例落库跳过：诊断流程未成功完成")
                return None
            syndrome_value = str(diagnosis_result.get("syndrome") or "").strip()
            confidence_value = float(diagnosis_result.get("confidence") or 0)
            if not syndrome_value or syndrome_value == "未明确" or confidence_value <= 0:
                logger.warning(
                    "病例落库跳过：无可靠主证 syndrome=%s confidence=%s",
                    syndrome_value or "空",
                    confidence_value,
                )
                return None

            chief_complaint = self._extract_chief_complaint(state)
            symptoms = self._extract_symptoms(state)
            syndrome = self._extract_syndrome(state)
            complexity = self._extract_complexity(state)
            diagnosis_text = self._extract_diagnosis_text(state)
            diagnosis_payload = self._extract_diagnosis_payload(state)
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
                diagnosis_payload=diagnosis_payload,
                tongue_analysis=self._extract_tongue_analysis(state),
                report_analysis=self._extract_report_analysis(state),
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
            for secondary_name in self._extract_secondary_syndromes(state):
                if secondary_name and secondary_name != syndrome.get("name"):
                    self.session.add(CaseSyndrome(
                        case_id=case.id,
                        syndrome_name=secondary_name,
                        confidence=None,
                        is_primary=False,
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

            await self.session.flush()
            await self._refresh_health_profile(user_id, state)
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

    async def _refresh_health_profile(
        self,
        user_id: UUID,
        state: dict[str, Any],
    ) -> None:
        """在应用层刷新健康档案，避免依赖数据库专用触发器。"""
        totals = await self.session.execute(
            select(func.count(Case.id), func.max(Case.created_at)).where(
                Case.user_id == user_id
            )
        )
        total_cases, last_case_at = totals.one()

        common = await self.session.execute(
            select(Case.syndrome_name, func.count(Case.id).label("case_count"))
            .where(Case.user_id == user_id, Case.syndrome_name.is_not(None))
            .group_by(Case.syndrome_name)
            .order_by(func.count(Case.id).desc(), Case.syndrome_name.asc())
            .limit(1)
        )
        common_row = common.first()

        profile = await self.session.get(UserHealthProfile, user_id)
        if profile is None:
            profile = UserHealthProfile(user_id=user_id)

        user_profile = state.get("user_profile") or {}
        collected_info = state.get("collected_info") or {}
        if isinstance(user_profile, dict):
            profile.constitution = (
                user_profile.get("constitution")
                or user_profile.get("constitution_type")
                or profile.constitution
            )
            chronic_conditions = self._as_string_list(
                user_profile.get("chronic_conditions")
                or user_profile.get("medical_history")
            )
            if isinstance(collected_info, dict):
                chronic_conditions.extend(
                    self._as_string_list(collected_info.get("medical_history"))
                )
            if chronic_conditions:
                profile.chronic_conditions = list(dict.fromkeys(chronic_conditions))

            allergies = self._as_string_list(user_profile.get("allergies"))
            if isinstance(collected_info, dict):
                allergies.extend(self._as_string_list(collected_info.get("allergies")))
            if allergies:
                profile.allergies = list(dict.fromkeys(allergies))

        profile.total_cases = int(total_cases or 0)
        profile.last_case_at = last_case_at
        profile.most_common_syndrome = common_row[0] if common_row else None
        profile.updated_at = datetime.now()
        self.session.add(profile)

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
            if getattr(msg, "type", None) == "human":
                content = getattr(msg, "content", "")
                if content:
                    return str(content)[:500]
        return "未明确"

    @staticmethod
    def _as_string_list(value: Any) -> list[str]:
        if value is None:
            return []
        values = value if isinstance(value, list) else [value]
        return [str(item).strip() for item in values if str(item).strip()]

    @staticmethod
    def _extract_symptoms(state: dict[str, Any]) -> list[dict[str, Any]]:
        """从 collected_info 的十问字段拆出症状条目。"""
        from app.src.agent.components.diagnose.models import CollectedDiagnoseInfo

        out: list[dict[str, Any]] = []
        collected = state.get("collected_info")
        if not isinstance(collected, dict):
            for s in state.get("symptoms", []) or []:
                if isinstance(s, str):
                    out.append({"name": s})
                elif isinstance(s, dict) and s.get("name"):
                    out.append(s)
            return out

        negated = {
            CollectedDiagnoseInfo._normalize_observation(item)
            for item in collected.get("negated_symptoms", []) or []
            if item
        }

        def is_positive(value: str) -> bool:
            return not CollectedDiagnoseInfo._is_negated_statement(value, negated)

        chief_complaint = collected.get("chief_complaint")
        if chief_complaint:
            chief_parts = [
                part.strip()
                for part in re.split(r"[，,、；;\n]+", str(chief_complaint))
                if part.strip()
            ]
            for part in chief_parts:
                if is_positive(part):
                    out.append({"name": part, "category": "chief_complaint"})

        for key in ("head_body", "cold_heat", "sweat", "urine_stool",
                    "diet", "chest_abdomen", "sleep", "emotion"):
            value = collected.get(key)
            if not value or not isinstance(value, str):
                continue
            parts = [p.strip() for p in re.split(r"[，,、；;\n]+", value) if p.strip()]
            for p in parts:
                if is_positive(p):
                    out.append({"name": p, "category": key})

        for symptom in collected.get("other_symptoms", []) or []:
            if symptom and is_positive(str(symptom)):
                out.append({"name": str(symptom), "category": "other"})

        deduplicated: list[dict[str, Any]] = []
        seen: set[tuple[str, Optional[str]]] = set()
        for symptom in out:
            key = (symptom["name"], symptom.get("category"))
            if key in seen:
                continue
            normalized = CollectedDiagnoseInfo._normalize_observation(symptom["name"])
            is_atomic = not re.search(r"[，,、；;\n]", symptom["name"])
            duplicate_index = next(
                (
                    index
                    for index, existing in enumerate(deduplicated)
                    if is_atomic
                    and not re.search(r"[，,、；;\n]", existing["name"])
                    and (
                        normalized
                        in CollectedDiagnoseInfo._normalize_observation(existing["name"])
                        or CollectedDiagnoseInfo._normalize_observation(existing["name"])
                        in normalized
                    )
                ),
                None,
            )
            if duplicate_index is not None:
                existing = deduplicated[duplicate_index]
                if len(normalized) > len(
                    CollectedDiagnoseInfo._normalize_observation(existing["name"])
                ):
                    deduplicated[duplicate_index] = symptom
                continue
            seen.add(key)
            deduplicated.append(symptom)
        return deduplicated

    @staticmethod
    def _extract_syndrome(state: dict[str, Any]) -> dict[str, Any]:
        result = state.get("syndrome_result") or state.get("diagnosis_result")
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        if isinstance(result, dict):
            extracted = {
                "id": result.get("syndrome_id") or result.get("id"),
                "name": (
                    result.get("syndrome_name")
                    or result.get("syndrome")
                    or result.get("name")
                ),
                "confidence": result.get("confidence"),
            }
            if extracted["name"]:
                return extracted

        diagnosis_text = CaseService._extract_diagnosis_text(state) or ""
        match = re.search(
            r"(?:主证|证型)\s*[：:]\s*\**\s*([^\n，。；;、*]{2,40})",
            diagnosis_text,
        )
        if match:
            return {
                "id": None,
                "name": match.group(1).strip(),
                "confidence": state.get("syndrome_confidence"),
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
        complexity = state.get("complexity")
        if hasattr(complexity, "model_dump"):
            complexity = complexity.model_dump()
        if isinstance(complexity, dict):
            level = complexity.get("level")
            if level in ("simple", "moderate", "complex"):
                return level.value if hasattr(level, "value") else level
        diagnose_stage = state.get("diagnose_stage")
        if isinstance(diagnose_stage, dict):
            inner = diagnose_stage.get("complexity")
            if isinstance(inner, dict) and inner.get("level") in ("simple", "moderate", "complex"):
                return inner["level"]
        return None

    @staticmethod
    def _extract_tongue_analysis(state: dict[str, Any]) -> Optional[dict[str, Any]]:
        tongue = state.get("tongue_analysis")
        if hasattr(tongue, "model_dump"):
            tongue = tongue.model_dump(mode="json")
        if isinstance(tongue, dict) and tongue:
            return tongue
        collected = state.get("collected_info") or {}
        textual_tongue = collected.get("tongue") if isinstance(collected, dict) else None
        return textual_tongue if isinstance(textual_tongue, dict) and textual_tongue else None

    @staticmethod
    def _extract_report_analysis(state: dict[str, Any]) -> Optional[dict[str, Any]]:
        report = state.get("report_analysis")
        if hasattr(report, "model_dump"):
            report = report.model_dump(mode="json")
        return report if isinstance(report, dict) and report else None

    @staticmethod
    def _extract_diagnosis_text(state: dict[str, Any]) -> Optional[str]:
        text = state.get("answer") or state.get("diagnosis_text")
        if isinstance(text, str) and text.strip():
            return text
        return None

    @staticmethod
    def _extract_diagnosis_payload(state: dict[str, Any]) -> Optional[dict[str, Any]]:
        result = state.get("diagnosis_result")
        if hasattr(result, "model_dump"):
            result = result.model_dump(mode="json")
        if not isinstance(result, dict):
            return None
        payload = dict(result)
        collected = state.get("collected_info") or {}
        if isinstance(collected, dict):
            input_evidence = {
                key: collected.get(key)
                for key in (
                    "tongue",
                    "pulse",
                    "negated_symptoms",
                    "medical_history",
                    "current_medications",
                    "allergies",
                )
                if collected.get(key)
            }
            if input_evidence:
                payload["input_evidence"] = input_evidence
        return payload

    @staticmethod
    def _extract_secondary_syndromes(state: dict[str, Any]) -> list[str]:
        result = CaseService._extract_diagnosis_payload(state) or {}
        values = result.get("syndrome_secondary") or result.get("secondary_syndromes") or []
        return [str(value) for value in values if value]

    @staticmethod
    def _extract_prescriptions(state: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        diagnosis_payload = CaseService._extract_diagnosis_payload(state) or {}
        source_items = diagnosis_payload.get("prescriptions") or state.get("prescriptions", []) or []
        for p in source_items:
            if hasattr(p, "model_dump"):
                p = p.model_dump()
            if isinstance(p, str):
                out.append({"name": p})
            elif isinstance(p, dict) and p.get("name"):
                normalized = dict(p)
                composition = normalized.get("composition")
                if composition is not None and not isinstance(composition, str):
                    normalized["composition"] = (
                        json.dumps(composition, ensure_ascii=False)
                        if composition
                        else None
                    )
                out.append(normalized)

        deduplicated = []
        seen = set()
        for item in out:
            if item["name"] not in seen:
                seen.add(item["name"])
                deduplicated.append(item)
        return deduplicated
