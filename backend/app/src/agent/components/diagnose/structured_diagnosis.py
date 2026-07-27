"""统一生成结构化辨证结果，并对不规范模型输出做安全降级。"""

from __future__ import annotations

import re
from typing import Any, Sequence

from langchain_core.messages import HumanMessage

from app.src.core.language_model.structured_output import parse_json_model
from app.src.utils import get_logger

from .models import CollectedDiagnoseInfo, DiagnosisPrescription, DiagnosisResult

logger = get_logger("structured_diagnosis")


DIAGNOSIS_STRUCTURED_INSTRUCTION = """
请只返回符合 DiagnosisResult 的 JSON 对象，不要使用 Markdown 代码围栏。

必须包含：
- syndrome：主证名称；无法明确时填“未明确”
- syndrome_id：知识库有 ID 才填写
- confidence：0 到 1
- syndrome_evidence：支持主证的可核验症状依据，不要输出隐藏思维链
- syndrome_secondary：兼证列表
- treatment_principle / treatment_method
- prescriptions：方名、组成、用法、出处、推荐理由和注意事项；资料不足可为空列表，禁止编造剂量和出处
- recommendations / warnings / should_seek_doctor
- citations：实际使用的检索证据；没有检索证据可为空列表
- patient_answer：300-500 字以内、直接面向患者的中文 Markdown 回答，包含辨证结论、调理建议和就医提示

patient_answer 中不得出现具体方剂名称、药味组成或剂量；具体方剂只允许放在
prescriptions 结构字段中，并由后端根据真实检索结果再次核验。

安全要求：线上结果仅作健康参考；急重症、孕妇、儿童、复杂基础病或正在合并用药者应提示线下就医。
不得声称已经排除高血压、脑血管病、颈椎病等现代医学疾病或器质性病变；
没有明确检查证据时只能表述为“线上无法排除，必要时线下检查”。
节气只能用于起居和环境建议，不能仅凭季节推断“内湿、湿热、寒包火”等
输入症状与检索证据未支持的新证候。中医“心、肝、脾、肺、肾”不得直接表述为
现代医学器官功能受损。
"""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content or "")


def _legacy_text_result(text: str) -> DiagnosisResult:
    """普通文本无法解析为 JSON 时，保留患者回答并提取最小结构。"""
    syndrome_match = re.search(
        r"(?:主证|证型|辨证结论)\s*[：:]\s*\**\s*([^\n，。；;、*]{2,40})",
        text,
    )
    if not syndrome_match:
        syndrome_match = re.search(r'"syndrome"\s*:\s*"([^"]{2,40})"', text)
    prescription_match = re.search(
        r"(?:推荐方剂|方剂参考|方药)\s*[：:]\s*\**\s*([^\n，。；;、*]{2,40})",
        text,
    )
    if not prescription_match:
        prescription_match = re.search(
            r'"prescriptions"\s*:\s*\[\s*\{[^{}]*?"name"\s*:\s*"([^"]{2,40})"',
            text,
            re.DOTALL,
        )
    prescriptions = []
    if prescription_match:
        prescriptions.append(DiagnosisPrescription(name=prescription_match.group(1).strip()))

    if text.lstrip().startswith("{"):
        if syndrome_match:
            patient_answer = f"**辨证结论**：考虑{syndrome_match.group(1).strip()}。\n\n"
            if prescription_match:
                patient_answer += f"**方剂参考**：{prescription_match.group(1).strip()}，具体组成与用法需由中医师面诊确认。\n\n"
            patient_answer += "线上信息有限，建议结合舌脉和既往用药由专业中医师复核；症状持续或加重请及时就医。"
        else:
            patient_answer = "本次结构化辨证输出不完整，无法可靠确定证型。请补充信息或由专业中医师线下复核。"
    else:
        patient_answer = text.strip() or "本次辨证结果生成不完整，建议补充信息或线下就医。"

    return DiagnosisResult(
        syndrome=syndrome_match.group(1).strip() if syndrome_match else "未明确",
        confidence=0.5 if syndrome_match else 0.0,
        prescriptions=prescriptions,
        warnings=["本次模型未返回完整结构化结果，已保留患者版说明；具体诊疗请线下复核。"],
        should_seek_doctor=not bool(syndrome_match),
        patient_answer=patient_answer,
    )


async def generate_structured_diagnosis(
    llm: Any,
    messages: Sequence[Any],
) -> DiagnosisResult:
    """优先 Pydantic 结构化输出，JSON 和普通文本依次降级。"""
    tagged_llm = llm.with_config(tags=["internal_structured_diagnosis"])
    structured_messages = [
        *messages,
        HumanMessage(content=DIAGNOSIS_STRUCTURED_INSTRUCTION),
    ]

    try:
        result = await tagged_llm.with_structured_output(DiagnosisResult).ainvoke(
            structured_messages
        )
        if result is not None:
            parsed = result if isinstance(result, DiagnosisResult) else DiagnosisResult.model_validate(result)
            if not parsed.patient_answer:
                parsed.patient_answer = parsed.to_display()
            return parsed
    except Exception as exc:
        logger.warning("DiagnosisResult 结构化输出失败，尝试普通 JSON: %s", exc)

    raw = await tagged_llm.ainvoke(structured_messages)
    try:
        parsed = parse_json_model(raw.content, DiagnosisResult)
        if not parsed.patient_answer:
            parsed.patient_answer = parsed.to_display()
        return parsed
    except Exception as exc:
        logger.warning("DiagnosisResult JSON 解析失败，降级保留普通文本: %s", exc)
        return _legacy_text_result(_content_to_text(raw.content))


def apply_clinical_safety_bounds(
    result: DiagnosisResult,
    collected_info: CollectedDiagnoseInfo,
    *,
    complexity_level: str | None = None,
    report_analysis: dict[str, Any] | None = None,
) -> DiagnosisResult:
    """根据可核验输入量和当前能力边界约束置信度与方剂推荐。"""
    positive_symptoms = collected_info.get_all_symptoms()
    result.syndrome_evidence = [
        cleaned
        for item in result.syndrome_evidence
        if (cleaned := _sanitize_report_derived_tcm_claim(str(item)))
    ]
    report = report_analysis or {}
    abnormal_metrics = [
        item
        for item in (report.get("metrics") or [])
        if isinstance(item, dict)
        and item.get("abnormal_flag") in {"high", "low", "abnormal", "positive"}
    ]
    report_summary = str(report.get("summary") or "").strip()
    urgent_report = bool(report.get("urgent_warning"))

    if abnormal_metrics or report.get("key_findings"):
        result.should_seek_doctor = True
        warning = (
            f"医疗报告存在需要结合临床复核的异常：{report_summary}"
            if report_summary
            else "医疗报告存在需要结合临床复核的异常指标。"
        )
        if warning not in result.warnings:
            result.warnings.append(warning)

    medical_history = [
        str(item).strip() for item in (collected_info.medical_history or []) if str(item).strip()
    ]
    current_medications = [
        str(item).strip() for item in (collected_info.current_medications or []) if str(item).strip()
    ]
    if current_medications or len(medical_history) >= 2:
        result.prescriptions = []
        result.should_seek_doctor = True
        context = "、".join([*medical_history, *current_medications])
        warning = (
            f"已知存在{context}；线上无法核对药物相互作用和禁忌，不提供具体方剂或自行停改药建议。"
        )
        if warning not in result.warnings:
            result.warnings.append(warning)

    if urgent_report:
        result.confidence = min(result.confidence, 0.35)
        result.prescriptions = []
        result.should_seek_doctor = True
        warning = "报告或伴随症状含紧急风险提示，应立即按报告建议就医，不应等待线上辨证。"
        if warning not in result.warnings:
            result.warnings.insert(0, warning)

    sparse_information = len(positive_symptoms) <= 2
    if sparse_information:
        result.confidence = min(result.confidence, 0.55)
        if result.syndrome == "未明确":
            result.confidence = min(result.confidence, 0.35)
        result.prescriptions = []
        warning = "当前阳性症状过少且缺少完整四诊证据，不宜据此确定个体化方剂。"
        if warning not in result.warnings:
            result.warnings.append(warning)
        notice = (
            "**信息边界**：目前只有少量非特异性阳性症状，"
            "只能给出低置信度辨证方向，暂不提供方剂推荐。"
        )
        if notice not in result.patient_answer:
            result.patient_answer = f"{notice}\n\n{result.patient_answer}".strip()

    if complexity_level == "complex":
        result.confidence = min(result.confidence, 0.65)
        result.prescriptions = []
        result.syndrome_evidence = [
            str(item).strip()
            for item in positive_symptoms[:8]
            if str(item).strip()
        ]
        warning = (
            "复杂病例当前使用可审计 GraphRAG 安全降级路径，尚未启用真实多专家"
            "DeepSearch，不应据此形成具体方剂或高置信度结论。"
        )
        if warning not in result.warnings:
            result.warnings.append(warning)
        evidence = "；".join(result.syndrome_evidence[:3])
        recommendations = "\n".join(
            f"{index}. {item}"
            for index, item in enumerate(result.recommendations[:4], 1)
        )
        report_notice = (
            f"**报告提示**：{report_summary.rstrip('。')}。\n\n"
            if report_summary
            else ""
        )
        attention_items = [
            str(item).strip()
            for item in (report.get("medical_attention_advice") or [])
            if str(item).strip()
        ]
        attention_notice = (
            "**报告复核建议**：\n"
            + "\n".join(
                f"{index}. {item}"
                for index, item in enumerate(attention_items[1:4] or attention_items[:3], 1)
            )
            + "\n\n"
            if attention_items
            else ""
        )
        safety_context = ""
        if medical_history or current_medications:
            history_text = "、".join(medical_history) or "未说明具体慢病名称"
            medication_text = "、".join(current_medications) or "未说明具体药名"
            safety_context = (
                "**慢病与用药安全**：已记录"
                f"{history_text}；当前用药为{medication_text}。"
                "需要携带完整用药清单线下核对相互作用和禁忌，不能自行停药、换药或叠加中药。\n\n"
            )
        urgent_notice = (
            "**紧急提示**：请立即就医，不要等待线上辨证结果。\n\n"
            if urgent_report
            else ""
        )
        result.patient_answer = (
            urgent_notice
            + report_notice
            + f"**初步辨证方向**：{result.syndrome or '未明确'}。"
            "由于症状跨多个系统、存在寒热或虚实矛盾，并伴复杂病史，"
            "当前结论置信度受限，不能替代面诊。\n\n"
            + (f"**可核验依据摘要**：{evidence}\n\n" if evidence else "")
            + (f"**生活调理建议**：\n{recommendations}\n\n" if recommendations else "")
            + attention_notice
            + safety_context
            + "**就医提示**：建议携带既往检查和完整用药清单，到正规医院由中医师"
            "结合舌脉及必要检查复核。当前不提供具体方剂、剂量或自行合方建议。"
        )
        result.should_seek_doctor = True

    if complexity_level != "complex":
        result.patient_answer = _build_deterministic_patient_answer(
            result,
            positive_symptoms=positive_symptoms,
            sparse_information=sparse_information,
        )
        if urgent_report:
            result.patient_answer = (
                "**紧急就医提示**：报告或伴随症状含紧急风险，请立即就医，"
                "不要等待线上辨证结果。\n\n" + result.patient_answer
            )
    result.patient_answer = _remove_formula_mentions_from_patient_answer(
        result.patient_answer,
        formula_names=[item.name for item in result.prescriptions],
    )
    result.patient_answer = _sanitize_unsupported_exclusion_claims(
        result.patient_answer
    )
    result.patient_answer = _sanitize_unsupported_seasonal_inference(
        result.patient_answer
    )
    return result


def _clean_patient_fragment(
    value: str,
    *,
    formula_names: Sequence[str],
) -> str:
    text = _remove_formula_mentions_from_patient_answer(
        str(value or ""),
        formula_names=formula_names,
    )
    text = _sanitize_unsupported_exclusion_claims(text)
    text = _sanitize_unsupported_seasonal_inference(text)
    text = re.sub(r"([：:])\s*[，,]", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip(" ，,；;。")
    return text


def _clean_warning_fragment(
    value: str,
    *,
    formula_names: Sequence[str],
) -> str:
    text = _clean_patient_fragment(value, formula_names=formula_names)
    text = re.sub(
        r"^(?:线上|本次)?(?:辨证|结果|建议)?(?:仅供参考|仅作健康参考)"
        r"[，,]?不能替代[^；;。]*(?:[；;。]|$)",
        "",
        text,
    ).strip(" ，,；;。")
    return text


def _build_deterministic_patient_answer(
    result: DiagnosisResult,
    *,
    positive_symptoms: Sequence[str],
    sparse_information: bool,
) -> str:
    """由结构化字段确定性生成患者 Markdown，避免模型自由文本直接外泄。"""
    formula_names = [item.name for item in result.prescriptions]
    parts: list[str] = []

    if sparse_information:
        parts.append(
            "**信息边界**：目前只有少量非特异性阳性表现，"
            "本次只能提供低置信度方向，不提供个体化方剂。"
        )

    if result.syndrome and result.syndrome != "未明确":
        parts.append(f"**辨证结论**：目前主要考虑为**{result.syndrome}**。")
    else:
        parts.append("**辨证结论**：现有信息不足，暂不能可靠确定具体证型。")

    symptoms = [str(item).strip() for item in positive_symptoms if str(item).strip()]
    evidence = [
        _clean_patient_fragment(item, formula_names=formula_names)
        for item in result.syndrome_evidence[:3]
    ]
    evidence = [item for item in evidence if item]
    basis_lines: list[str] = []
    if symptoms:
        basis_lines.append(f"- 已纳入的阳性表现：{'、'.join(symptoms[:8])}")
    basis_lines.extend(f"- {item}" for item in evidence)
    if basis_lines:
        parts.append("**辨证依据**：\n" + "\n".join(basis_lines))

    treatment = "、".join(
        item
        for item in (
            _clean_patient_fragment(result.treatment_principle or "", formula_names=formula_names),
            _clean_patient_fragment(result.treatment_method or "", formula_names=formula_names),
        )
        if item
    )
    if treatment:
        parts.append(f"**调理方向**：{treatment}。")

    recommendations = [
        _clean_patient_fragment(item, formula_names=formula_names)
        for item in result.recommendations[:4]
    ]
    recommendations = [item for item in recommendations if item]
    if recommendations:
        parts.append(
            "**生活调理建议**：\n"
            + "\n".join(
                f"{index}. {item}"
                for index, item in enumerate(recommendations, 1)
            )
        )

    warnings = [
        _clean_warning_fragment(item, formula_names=formula_names)
        for item in result.warnings[:4]
    ]
    warnings = [item for item in warnings if item]
    seek_prefix = (
        "建议结合症状变化和检查结果安排线下复核。"
        if result.should_seek_doctor
        else "若症状持续、加重或出现新的明显不适，请及时线下就医。"
    )
    warning_text = "；".join(warnings)
    parts.append(
        "**就医提示**："
        + seek_prefix
        + (f" 注意：{warning_text}。" if warning_text else "")
        + " 本结果仅作健康参考，不能替代面诊、检查和处方。"
    )
    return "\n\n".join(parts)


def _sanitize_unsupported_exclusion_claims(text: str) -> str:
    """禁止把未提供的现代医学检查结果描述成已经排除。"""
    value = str(text or "")
    value = re.sub(
        r"在(?:已经|已)?排除(?:了)?([^，。\n]{1,100})(?:的前提下|之后|后)[，,]?",
        r"线上尚无法排除\1，",
        value,
    )
    value = re.sub(
        r"(?:已经|已)排除(?:了)?([^，。\n]{1,100})",
        r"线上尚无法排除\1",
        value,
    )
    return value


def _sanitize_report_derived_tcm_claim(text: str) -> str:
    """报告异常可提示现代医学复核，但不能作为直接确定中医证型的依据。"""
    value = str(text or "").strip()
    value = re.sub(
        r"[，,；;]?\s*(?:结合|根据)(?:血红蛋白|血常规|化验|检验|报告)"
        r"[^。\n]*(?:符合|支持|提示)[^。\n]*(?:证|表现)?[。]?\s*$",
        "。",
        value,
    )
    value = re.sub(r"。{2,}", "。", value).strip()
    return value


def _sanitize_unsupported_seasonal_inference(text: str) -> str:
    """移除仅凭节气新增证候、而非由患者症状支持的推断。"""
    value = str(text or "")
    return re.sub(
        r"(?:当前|目前)正值[^。\n]{0,180}"
        r"(?:形成|属于|提示)[^。\n]{0,100}"
        r"(?:证候|证型|局面)[。]?'?",
        "节气仅作为起居调护参考，不据此增加证候判断。",
        value,
    )


def _remove_formula_mentions_from_patient_answer(
    text: str,
    *,
    formula_names: Sequence[str] = (),
) -> str:
    """隐藏真实方名和明确方药建议，同时保留姜汤、米汤等食疗词。"""
    value = str(text or "")
    formula_suffix = r"(?:汤|丸|散|丹|饮|胶囊|颗粒|口服液|合剂)"
    value = re.sub(
        rf"[（(](?=[^）)\n]{{0,20}}(?:方剂|方药|中成药|处方))"
        rf"[^）)\n]{{0,80}}{formula_suffix}[^）)\n]{{0,40}}[）)]",
        "（具体方剂需由中医师面诊确定）",
        value,
    )
    value = re.sub(
        rf"(?:(?:方剂|方药|中成药|处方)(?:例如|如|可参考|可选|[：:])|"
        rf"可选用|可使用|可考虑|(?<!不)建议使用|(?<!不)建议服用|"
        rf"(?<!不)推荐使用|(?<!不)推荐服用|开具)"
        rf"[^，。；\n]{{0,40}}{formula_suffix}(?:加减)?",
        "具体方剂需由中医师面诊确定",
        value,
    )
    for name in sorted(
        {str(item).strip() for item in formula_names if str(item).strip()},
        key=len,
        reverse=True,
    ):
        value = re.sub(
            re.escape(name),
            "具体方剂需由中医师面诊确定",
            value,
        )
    ordinary_food_soups = {
        "姜汤", "生姜汤", "米汤", "热汤", "菜汤", "鸡汤", "鱼汤", "骨头汤",
        "绿豆汤", "红豆汤", "酸梅汤", "冬瓜汤", "萝卜汤", "蛋花汤",
    }

    def remove_unapproved_formula(match: re.Match[str]) -> str:
        name = match.group(1)
        return name if any(name.endswith(food) for food in ordinary_food_soups) else ""

    value = re.sub(
        rf"([\u4e00-\u9fff]{{2,20}}{formula_suffix})"
        rf"(?=$|[、，,。；;：:\s]|等|或|及|与)",
        remove_unapproved_formula,
        value,
    )
    value = re.sub(r"[、，,]\s*(?=等)", "", value)
    value = re.sub(r"([、，,])\1+", r"\1", value)
    return value
