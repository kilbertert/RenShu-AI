"""
简单辨证节点

简单病情的直接辨证（LLM 直接分析）
"""

from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage

from app.src.agent.tcm_builder import get_llm

from app.src.agent.components.diagnose.prompts import SIMPLE_DIAGNOSIS_PROMPT
from datetime import datetime

from app.src.agent.components.diagnose.states import DiagnoseOverallState
from app.src.agent.components.diagnose.models import CollectedDiagnoseInfo, DiagnosisResult
from app.src.agent.components.diagnose.structured_diagnosis import (
    apply_clinical_safety_bounds,
    generate_structured_diagnosis,
)
from app.src.agent.components.diagnose.config import diagnose_config
from app.src.utils import get_logger

logger = get_logger("simple_diagnosis")


def _get_current_solar_term() -> str:
    """获取当前节气（简化版）"""
    now = datetime.now()
    month = now.month
    
    # 简化的节气映射（实际应该更精确）
    solar_terms = {
        1: "小寒/大寒", 2: "立春/雨水", 3: "惊蛰/春分",
        4: "清明/谷雨", 5: "立夏/小满", 6: "芒种/夏至",
        7: "小暑/大暑", 8: "立秋/处暑", 9: "白露/秋分",
        10: "寒露/霜降", 11: "立冬/小雪", 12: "大雪/冬至"
    }
    
    return solar_terms.get(month, "未知节气")


async def simple_diagnosis(state: DiagnoseOverallState) -> Dict[str, Any]:
    """
    简单病情的直接辨证

    方法：
    - LLM 根据收集的信息直接进行八纲辨证
    - 结合望闻问切四诊信息
    - 给出证型、治则、建议
    - 如果启用 thinking 模式，会先输出思考过程

    适用场景：
    - 单一证型（如普通感冒）
    - 症状明确，指向清晰
    - 无复杂既往史

    Args:
        state: 当前状态

    Returns:
        dict: 更新的状态字段
    """
    try:
        # 获取已收集的信息
        collected_info_dict = state.get("collected_info", {})
        if collected_info_dict:
            collected_info = CollectedDiagnoseInfo(**collected_info_dict)
            collected_summary = collected_info.to_summary()
        else:
            collected_summary = "暂无详细信息"

        # 获取舌像分析
        tongue_analysis = state.get("tongue_analysis")
        tongue_desc = "未提供"
        if tongue_analysis:
            parts = []
            if tongue_analysis.get("tongue_color"): parts.append(f"舌色：{tongue_analysis['tongue_color']}")
            if tongue_analysis.get("tongue_shape"): parts.append(f"舌形：{tongue_analysis['tongue_shape']}")
            if tongue_analysis.get("coating_color"): parts.append(f"苔色：{tongue_analysis['coating_color']}")
            if tongue_analysis.get("coating_quality"): parts.append(f"苔质：{tongue_analysis['coating_quality']}")
            if tongue_analysis.get("analysis"): parts.append(f"分析：{tongue_analysis['analysis']}")
            tongue_desc = "\n".join(parts)

        report_analysis = state.get("report_analysis")
        report_desc = _format_report_analysis(report_analysis)

        # 获取用户画像
        user_profile = state.get("user_profile", {})

        user_profile_desc = _format_user_profile(user_profile)
        
        # 获取当前节气
        solar_term = _get_current_solar_term()

        # 构建提示词
        prompt = SIMPLE_DIAGNOSIS_PROMPT.format(
            collected_info=collected_summary,
            tongue_analysis=tongue_desc,
            report_analysis=report_desc,
            user_profile=user_profile_desc,
            solar_term=solar_term,
        )

        # 调用 LLM
        llm = get_llm(
            llm_config=state.get("llm_config"),
            temperature=diagnose_config.DIAGNOSIS_TEMPERATURE
        )

        diagnosis_result = await generate_structured_diagnosis(
            llm,
            [
                SystemMessage(content=prompt),
                HumanMessage(content="请开始您的辨证分析。"),
            ],
        )
        if diagnosis_result.prescriptions:
            diagnosis_result.prescriptions = []
            diagnosis_result.warnings.append(
                "简单辨证未经过方剂知识图谱核验，因此不输出具体方剂。"
            )
        apply_clinical_safety_bounds(
            diagnosis_result,
            collected_info,
            report_analysis=state.get("report_analysis"),
        )
        answer = diagnosis_result.to_display()

        logger.info("简单辨证完成（结构化输出）")
        return {
            "answer": answer,
            "diagnosis_result": diagnosis_result.model_dump(),
            "steps": ["简单辨证: 结构化辨证完成"],
        }

    except Exception as e:
        logger.error(f"简单辨证失败: {e}", exc_info=True)
        answer = "抱歉，本次辨证分析未能完整生成。若症状持续、加重或伴胸痛、呼吸困难等情况，请及时线下就医。"
        diagnosis_result = DiagnosisResult(
            syndrome="未明确",
            confidence=0.0,
            warnings=["结构化辨证生成失败，需要补充信息或由专业医师复核。"],
            should_seek_doctor=True,
            patient_answer=answer,
        )
        return {
            "answer": answer,
            "diagnosis_result": diagnosis_result.model_dump(),
            "error": f"结构化辨证生成失败: {type(e).__name__}",
            "steps": [f"简单辨证: 失败 - {str(e)}"],
        }



def _format_report_analysis(report: Dict[str, Any] | None) -> str:
    if not report:
        return "未提供"
    parts: list[str] = []
    if report.get("report_type"):
        parts.append(f"报告类型：{report['report_type']}")
    if report.get("summary"):
        parts.append(f"摘要：{report['summary']}")
    findings = report.get("key_findings") or []
    if findings:
        parts.append("关键发现：" + "；".join(str(item) for item in findings[:10]))
    abnormal_metrics = [
        item for item in (report.get("metrics") or [])
        if isinstance(item, dict)
        and item.get("abnormal_flag") in {"high", "low", "abnormal", "positive"}
    ]
    if abnormal_metrics:
        parts.append(
            "异常指标：" + "；".join(
                f"{item.get('name', '')} {item.get('value', '')}{item.get('unit', '')}"
                for item in abnormal_metrics[:12]
            )
        )
    parts.append("约束：报告只能作为辅助证据，不能由单项指标直接确定中医证型。")
    return "\n".join(parts)


def _format_user_profile(profile: Dict[str, Any]) -> str:
    """格式化用户画像"""
    if not profile:
        return "暂无用户健康档案"

    parts = []

    if profile.get("gender"):
        parts.append(f"性别：{profile['gender']}")
    if profile.get("age") or profile.get("age_group"):
        parts.append(f"年龄：{profile.get('age') or profile.get('age_group')}")
    if profile.get("constitution"):
        parts.append(f"体质类型：{profile['constitution']}")
    if profile.get("chronic_conditions"):
        conditions = profile['chronic_conditions']
        if isinstance(conditions, list):
            parts.append(f"既往病史：{', '.join(conditions)}")
        else:
            parts.append(f"既往病史：{conditions}")
    if profile.get("allergies"):
        allergies = profile['allergies']
        if isinstance(allergies, list):
            parts.append(f"过敏史：{', '.join(allergies)}")
        else:
            parts.append(f"过敏史：{allergies}")

    return "\n".join(parts) if parts else "暂无用户健康档案"
