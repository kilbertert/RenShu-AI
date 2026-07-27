"""
信息收集节点

从用户输入中提取症状信息，更新 collected_info
"""

import re
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..states import DiagnoseOverallState
from ..models import CollectedDiagnoseInfo
from ....tcm_builder import get_llm
from app.src.core.language_model.structured_output import (
    invoke_structured_with_json_fallback,
)
from app.src.utils import get_logger

logger = get_logger("collect_info")


class ExtractedInfo(BaseModel):
    """从用户输入中提取的信息"""
    chief_complaint: str = Field(default="", description="主诉症状")
    onset_time: str = Field(default="", description="发病时间")
    duration: str = Field(default="", description="病程")
    cold_heat: str = Field(default="", description="寒热情况")
    sweat: str = Field(default="", description="汗出情况")
    head_body: str = Field(default="", description="头身症状")
    urine_stool: str = Field(default="", description="二便情况")
    diet: str = Field(default="", description="饮食情况")
    chest_abdomen: str = Field(default="", description="胸腹症状")
    sleep: str = Field(default="", description="睡眠情况")
    emotion: str = Field(default="", description="情志状态")
    complexion: str = Field(default="", description="面色")
    tongue: dict[str, str] = Field(
        default_factory=dict,
        description="患者用文字明确描述的舌色、舌形、苔色、苔质",
    )
    pulse: str = Field(default="", description="患者用文字明确描述的脉象")
    medical_history: list[str] = Field(default_factory=list, description="既往病史")
    current_medications: list[str] = Field(default_factory=list, description="当前用药")
    allergies: list[str] = Field(default_factory=list, description="过敏史")
    menstruation: str = Field(default="", description="月经、孕产等女性相关信息")
    negated_symptoms: list[str] = Field(
        default_factory=list,
        description="患者明确否认的症状，例如无发热、不胸痛",
    )
    other_symptoms: list[str] = Field(default_factory=list, description="其他症状")


COLLECTION_SYSTEM_PROMPT = """你是一位经验丰富的中医师，正在进行问诊。

你的任务是从患者的描述中提取关键信息，按照中医十问的框架进行分类。

**中医十问框架：**
1. 寒热：恶寒、发热、寒热往来
2. 汗出：有汗、无汗、盗汗、自汗
3. 头身：头痛、头晕、身痛、乏力
4. 二便：大便（便秘、腹泻）、小便（频数、不利）
5. 饮食：食欲、口渴、口苦、口淡
6. 胸腹：胸闷、腹胀、心悸
7. 睡眠：失眠、多梦、嗜睡
8. 情志：烦躁、抑郁、焦虑

**提取原则：**
- 只提取患者明确提到的信息，不要推测
- 保持患者的原始描述，不要过度解释
- 如果某个类别没有信息，留空
- 注意提取时间信息（何时开始、持续多久）
- 用户用文字提供的舌象和脉象必须提取，不得因为没有图片而忽略
- 明确否认的症状放入 negated_symptoms，不得同时作为阳性 other_symptoms
- “无汗”“口不渴”“不欲饮”是有辨证意义的表现，不按普通否定症状丢弃

**已收集的信息：**
{collected_summary}

**患者本轮描述：**
{user_input}

请提取本轮新增的信息。
"""


async def collect_info(state: DiagnoseOverallState) -> Dict[str, Any]:
    """
    收集用户输入的信息，更新 collected_info

    功能：
    1. 解析用户最新输入
    2. 提取症状、时间、程度等信息
    3. 映射到 CollectedDiagnoseInfo 的相应字段
    4. 检测是否有图片（舌像）或文件（报告）

    Args:
        state: 当前状态

    Returns:
        dict: 更新的状态字段
    """
    try:
        # 获取用户最新输入
        messages = state.get("messages", [])
        if not messages:
            return {"steps": ["信息收集: 无消息"]}

        last_user_message = None
        for msg in reversed(messages):
            # 兼容两种格式：HumanMessage 对象和字典
            if isinstance(msg, HumanMessage):
                last_user_message = msg.content
                break
            elif isinstance(msg, dict) and msg.get("role") == "user":
                last_user_message = msg.get("content", "")
                break

        if not last_user_message:
            return {"steps": ["信息收集: 未找到用户消息"]}

        # 获取已收集的信息
        collected_info_dict = state.get("collected_info", {})
        if collected_info_dict:
            collected_info = CollectedDiagnoseInfo(**collected_info_dict)
        else:
            collected_info = CollectedDiagnoseInfo()

        # 舌像由认证附件主路径产生，不依赖 LLM 从用户文字中再次猜测。
        tongue_analysis = state.get("tongue_analysis")
        if tongue_analysis:
            tongue_fields = {
                key: str(tongue_analysis.get(key) or "")
                for key in (
                    "tongue_color",
                    "tongue_shape",
                    "coating_color",
                    "coating_quality",
                    "image_quality",
                )
                if tongue_analysis.get(key)
            }
            if tongue_fields:
                tongue_fields["source"] = "image"
                collected_info.tongue = tongue_fields

        deterministic = _fallback_extract_info(last_user_message)
        deterministic_categories = sum(
            bool(getattr(deterministic, field))
            for field in (
                "cold_heat", "sweat", "head_body", "urine_stool", "diet",
                "chest_abdomen", "sleep", "emotion", "complexion", "tongue",
                "pulse", "medical_history", "current_medications", "allergies",
                "menstruation",
            )
        )

        # 常见十问已被确定性规则覆盖时直接快通道，避免兼容网关结构化输出失败后长时间重试。
        generic_complaint = bool(
            deterministic_categories == 0
            and len(last_user_message.strip()) <= 80
            and re.search(
                r"(?:不舒服|身体不适|有点难受|总觉得难受|哪里不对劲)",
                last_user_message,
            )
        )
        if deterministic_categories >= 2 or generic_complaint:
            response = deterministic.model_copy(deep=True)
            logger.info(
                "确定性信息抽取快通道命中 %s 类%s",
                deterministic_categories,
                "（泛化主诉）" if generic_complaint else "",
            )
        else:
            collected_summary = (
                collected_info.to_summary()
                if collected_info.get_filled_count() > 0
                else "暂无"
            )
            system_prompt = COLLECTION_SYSTEM_PROMPT.format(
                collected_summary=collected_summary,
                user_input=last_user_message,
            )
            llm = get_llm(llm_config=state.get("llm_config"))
            try:
                response = await invoke_structured_with_json_fallback(
                    llm,
                    ExtractedInfo,
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=last_user_message),
                    ],
                )
            except Exception as exc:
                logger.warning("LLM 信息提取失败，使用确定性关键词降级: %s", exc)
                response = deterministic.model_copy(deep=True)

        # 明确出现的十问关键词、文字舌脉和否定语义始终用确定性规则补齐，
        # 避免模型把“正常/没有”误当成未回答而重复追问。
        for field in (
            "chief_complaint", "onset_time", "duration", "cold_heat", "sweat",
            "head_body", "urine_stool", "diet", "chest_abdomen", "sleep",
            "emotion", "complexion", "pulse",
            "menstruation",
        ):
            if not getattr(response, field) and getattr(deterministic, field):
                setattr(response, field, getattr(deterministic, field))
        if not response.tongue and deterministic.tongue:
            response.tongue = deterministic.tongue
        if not response.other_symptoms and deterministic.other_symptoms:
            response.other_symptoms = deterministic.other_symptoms
        for field in ("medical_history", "current_medications", "allergies"):
            merged_values = list(dict.fromkeys([
                *(getattr(response, field) or []),
                *(getattr(deterministic, field) or []),
            ]))
            if field == "current_medications" and any(
                "多种药" in str(value) for value in merged_values
            ):
                merged_values = [
                    value for value in merged_values if "多种药" not in str(value)
                ] + ["多种药物（具体名称待补充）"]
            setattr(response, field, merged_values)

        textual_tongue, textual_pulse = _extract_textual_tongue_pulse(last_user_message)
        negated_symptoms = list(dict.fromkeys([
            *response.negated_symptoms,
            *deterministic.negated_symptoms,
        ]))
        for symptom in negated_symptoms:
            normalized = _normalize_negated_symptom(symptom)
            if normalized and normalized not in collected_info.negated_symptoms:
                collected_info.negated_symptoms.append(normalized)

        # 更新 collected_info
        if response.chief_complaint and not collected_info.chief_complaint:
            chief_complaint = _positive_observation_text(
                response.chief_complaint,
                collected_info.negated_symptoms,
            )
            if chief_complaint:
                collected_info.chief_complaint = chief_complaint
        if response.onset_time and not collected_info.onset_time:
            collected_info.onset_time = response.onset_time
        if response.duration and not collected_info.duration:
            collected_info.duration = response.duration

        # 更新十问信息（追加而不是覆盖）
        if response.cold_heat:
            collected_info.cold_heat = _merge_info(collected_info.cold_heat, response.cold_heat)
        if response.sweat:
            collected_info.sweat = _merge_info(collected_info.sweat, response.sweat)
        if response.head_body:
            collected_info.head_body = _merge_info(collected_info.head_body, response.head_body)
        if response.urine_stool:
            collected_info.urine_stool = _merge_info(collected_info.urine_stool, response.urine_stool)
        if response.diet:
            collected_info.diet = _merge_info(collected_info.diet, response.diet)
        if response.chest_abdomen:
            collected_info.chest_abdomen = _merge_info(collected_info.chest_abdomen, response.chest_abdomen)
        if response.sleep:
            collected_info.sleep = _merge_info(collected_info.sleep, response.sleep)
        if response.emotion:
            collected_info.emotion = _merge_info(collected_info.emotion, response.emotion)
        if response.complexion:
            collected_info.complexion = _merge_info(collected_info.complexion, response.complexion)
        if response.menstruation:
            collected_info.menstruation = _merge_info(
                collected_info.menstruation,
                response.menstruation,
            )

        # 图片分析优先；没有图片时，文字舌象和脉象同样是有效四诊证据。
        if not collected_info.tongue or collected_info.tongue.get("source") != "image":
            merged_tongue = {
                **(response.tongue or {}),
                **textual_tongue,
            }
            if merged_tongue:
                merged_tongue["source"] = "text"
                collected_info.tongue = merged_tongue
        pulse_description = textual_pulse or response.pulse
        if pulse_description:
            collected_info.pulse = {
                "description": pulse_description,
                "source": "text",
            }

        for field in ("medical_history", "current_medications", "allergies"):
            new_values = getattr(response, field) or []
            if not new_values:
                continue
            existing_values = list(getattr(collected_info, field) or [])
            for value in new_values:
                if value and value not in existing_values:
                    existing_values.append(value)
            setattr(collected_info, field, existing_values)

        # 更新其他症状
        if response.other_symptoms:
            if collected_info.other_symptoms is None:
                collected_info.other_symptoms = []
            for symptom in response.other_symptoms:
                if (
                    symptom
                    and not _is_negated_observation(
                        symptom,
                        collected_info.negated_symptoms,
                    )
                    and symptom not in collected_info.other_symptoms
                ):
                    collected_info.other_symptoms.append(symptom)

        # 记录收集历史
        collection_history = state.get("collection_history", [])
        collection_history.append({
            "round_number": len(collection_history) + 1,
            "user_input": last_user_message,
            "extracted_info": response.model_dump(),
        })

        logger.info(f"信息收集完成，已收集 {collected_info.get_filled_count()} 类信息")

        return {
            "collected_info": collected_info.model_dump(),
            "collection_history": collection_history,
            "steps": [f"信息收集: 提取了 {len([f for f in response.model_dump().values() if f])} 项信息"],
        }

    except Exception as e:
        logger.error(f"信息收集失败: {e}", exc_info=True)
        return {
            "steps": [f"信息收集: 失败 - {str(e)}"],
        }


def _merge_info(existing: str | None, new: str) -> str:
    """合并信息（追加而不是覆盖）"""
    if not existing:
        return new
    if not new:
        return existing
    # 如果新信息不在旧信息中，追加
    if new not in existing:
        return f"{existing}；{new}"
    return existing


_CANONICAL_CATEGORY_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "cold_heat": [
        ("怕冷", r"(?:容易|总是|比较|明显)?(?:怕冷|畏冷|畏寒|发冷)"),
        ("手脚冰凉", r"(?:手脚|手足|四肢)(?:总是|经常|一直|容易|比较|明显)?(?:发凉|冰凉|冰冷|凉)"),
        ("怕热", r"(?:容易|总是|比较|明显)?怕热"),
        ("五心烦热", r"(?:五心烦热|手足心热|手心脚心发热)"),
        ("无明显寒热", r"(?:没有|无|不觉得)(?:明显)?(?:怕冷|怕热|寒热异常)|寒热正常|体温正常"),
    ],
    "sweat": [
        ("少汗", r"(?:不怎么|不太|很少|较少|几乎不)出汗"),
        ("无汗", r"(?:没有|完全不|从不)出汗|无汗"),
        ("自汗", r"(?:稍微活动|一活动|动一动|白天).{0,8}(?:就)?出汗|容易出汗"),
        ("盗汗", r"(?:睡着|夜里|夜间).{0,8}出汗|醒后汗止|盗汗"),
        ("汗多", r"(?:出汗很多|汗特别多|大汗|汗多)"),
        ("汗出正常", r"(?:出汗|汗出)(?:正常|没有异常|还好)"),
    ],
    "head_body": [
        ("乏力", r"(?:容易|总是|经常)?(?:疲倦|疲劳|乏力)|没精神|浑身没劲|全身没力气|身上没劲"),
        ("身体酸沉", r"(?:身体|全身|浑身|四肢)(?:酸沉|困重|沉重)"),
        ("腰酸", r"腰(?:酸软|酸|发酸)"),
    ],
    "urine_stool": [
        ("大便偏稀", r"大便(?:偏|比较|有点)?(?:稀|溏)|便质偏稀|不太成形|不成形"),
        ("大便干结", r"大便(?:偏|比较|很)?(?:干|硬|干结)|排便费力"),
        ("大便正常", r"大便(?:基本|一直|还算|都)?(?:正常|如常|没问题)"),
        ("小便正常", r"小便(?:基本|一直|还算|都)?(?:正常|如常|没问题)"),
        ("二便正常", r"(?:大小便|二便)(?:基本|一直|都)?(?:正常|如常|没问题)"),
        ("小便偏多", r"小便(?:次数)?(?:偏|比较|明显)?多|尿得多"),
        ("小便偏少", r"小便(?:量)?(?:偏|比较|明显)?少|尿少"),
        ("尿色黄", r"(?:小便|尿)(?:颜色)?(?:偏|很)?黄"),
    ],
    "diet": [
        ("食欲一般", r"(?:食欲|胃口)(?:一般|普通|还行|尚可)"),
        ("食欲不振", r"(?:食欲|胃口)(?:不好|差|不佳)|吃不下|不想吃饭|纳差"),
        ("食量减少", r"(?:吃得|饭量)(?:少|不多|减少)"),
        ("饮食正常", r"(?:饮食|吃饭|饭量)(?:基本|一直|还算)?(?:正常|如常|没问题)|饮食和睡眠(?:尚可|正常|还行)"),
        ("不欲饮", r"(?:不想|不愿|不爱)(?:喝水|饮水)|不欲饮"),
        ("咽干", r"(?:偶有|有点|轻微|明显)?咽干"),
    ],
    "chest_abdomen": [
        ("胸闷", r"(?:胸口|胸部)(?:发闷|闷|憋闷)"),
        ("心悸", r"(?:心慌|心跳得慌|心悸)"),
        ("腹胀", r"(?:肚子|腹部|胃脘|胃里)(?:胀|发胀|胀气)"),
    ],
    "sleep": [
        ("入睡困难", r"(?:入睡困难|很难入睡|睡不着|难以入睡)"),
        ("易醒", r"(?:容易醒|总醒|睡一会就醒|夜里醒)"),
        ("多梦", r"(?:梦多|多梦)"),
        ("睡眠不踏实", r"(?:睡不踏实|睡得浅|睡眠浅)"),
        ("睡眠尚可", r"(?:睡眠|睡得)(?:一般|还行|尚可|正常|不错)|饮食和睡眠(?:尚可|正常|还行)"),
    ],
    "emotion": [
        ("焦虑", r"(?:焦虑|紧张|担心很多|心里不安)"),
        ("情绪低落", r"(?:情绪低落|心情不好|提不起兴趣)"),
    ],
}

_MEANINGFUL_NEGATIVE_OBSERVATIONS = {"少汗", "无汗", "不欲饮", "无明显寒热"}


def _append_observation(
    statements: list[str],
    positive_symptoms: list[str],
    observation: str,
) -> None:
    if observation not in statements:
        statements.append(observation)
    normalized = CollectedDiagnoseInfo._normalize_observation(observation)
    if (
        not CollectedDiagnoseInfo._is_non_pathological_statement(normalized)
        and not CollectedDiagnoseInfo._is_negated_statement(normalized, set())
        and observation not in positive_symptoms
    ):
        positive_symptoms.append(observation)


def _fallback_extract_info(text: str) -> ExtractedInfo:
    """模型无法返回 JSON 时，用保守关键词规则提取患者明确陈述的信息。"""
    categories = {
        "cold_heat": ["怕冷", "恶寒", "畏寒", "怕热", "发热", "发烧", "潮热", "手足心热", "五心烦热"],
        "sweat": ["无汗", "有汗", "自汗", "盗汗", "汗多"],
        "head_body": ["头痛", "头晕", "眩晕", "乏力", "腰痛", "腰酸", "身痛", "耳鸣"],
        "urine_stool": ["便秘", "腹泻", "便溏", "大便干", "大便稀", "大便偏稀", "大便正常", "小便正常", "小便偏多", "尿频", "夜尿", "小便不利"],
        "diet": ["食欲不振", "食欲一般", "没胃口", "饮食正常", "饭量正常", "口渴", "口干", "口苦", "咽干", "不欲饮"],
        "chest_abdomen": ["胸闷", "心悸", "气短", "腹胀", "腹痛"],
        "sleep": ["失眠", "多梦", "易醒", "嗜睡", "睡眠正常"],
        "emotion": ["焦虑", "烦躁", "抑郁", "易怒", "情绪低落"],
        "complexion": ["面色苍白", "面色萎黄", "面红", "面色晦暗"],
    }

    negated_symptoms = _extract_negated_symptoms(text)
    extracted: dict[str, Any] = {}
    positive_symptoms: list[str] = []
    for key, keywords in categories.items():
        statements: list[str] = []
        for keyword in keywords:
            for match in re.finditer(re.escape(keyword), text):
                if _mention_is_negated(text, match.start(), keyword):
                    statement = _normalize_negated_symptom(keyword)
                else:
                    statement = keyword
                    normalized = CollectedDiagnoseInfo._normalize_observation(keyword)
                    if (
                        not CollectedDiagnoseInfo._is_non_pathological_statement(normalized)
                        and keyword not in positive_symptoms
                    ):
                        positive_symptoms.append(keyword)
                if statement and statement not in statements:
                    statements.append(statement)
        for observation, pattern in _CANONICAL_CATEGORY_PATTERNS.get(key, []):
            match = re.search(pattern, text)
            if not match:
                continue
            if (
                observation not in _MEANINGFUL_NEGATIVE_OBSERVATIONS
                and _mention_is_negated(text, match.start(), observation)
            ):
                negated_observation = _normalize_negated_symptom(observation)
                if negated_observation and negated_observation not in statements:
                    statements.append(negated_observation)
                continue
            _append_observation(statements, positive_symptoms, observation)
        extracted[key] = "、".join(statements)
    duration_match = re.search(
        r"(?:大约|约|近|最近|持续|已经|断断续续|反复)?"
        r"(?:半|数|十来|[一二三四五六七八九十\d]+(?:来|多|余)?|[一二三四五六七八九十\d]+[到至-][一二三四五六七八九十\d]+)"
        r"(?:天|周|个?月|年)",
        text,
    )
    extracted["duration"] = duration_match.group(0) if duration_match else ""

    if "大便" in text and re.search(r"(?:有时|时而).{0,8}干", text) and re.search(
        r"(?:有时|时而).{0,8}(?:稀|溏)", text
    ):
        extracted["urine_stool"] = _merge_info(
            extracted.get("urine_stool"),
            "大便有时干、有时稀",
        )
    if re.search(r"(?:口干).{0,8}(?:不想|不愿|不欲)(?:喝水|饮水|饮)", text):
        extracted["diet"] = _merge_info(extracted.get("diet"), "口干但不欲饮")

    known_conditions = [
        "高血压", "糖尿病", "冠心病", "心律失常", "脑梗死", "脑出血",
        "慢性肾病", "肾功能不全", "慢性肝病", "甲状腺疾病", "哮喘",
        "慢阻肺", "肿瘤", "贫血",
    ]
    extracted["medical_history"] = [
        condition for condition in known_conditions if condition in text
    ]

    medications: list[str] = []
    if re.search(r"(?:正在|目前|长期)?(?:服用|吃着|使用).{0,8}(?:多种|多类)(?:药物|药)", text):
        medications.append("多种药物（具体名称待补充）")
    for match in re.finditer(
        r"(?:服用|口服|使用)([\u4e00-\u9fffA-Za-z0-9-]{2,30}(?:片|胶囊|颗粒|丸|注射液))",
        text,
    ):
        medication = match.group(1)
        if medication not in medications:
            medications.append(medication)
    known_medications = [
        "二甲双胍", "阿卡波糖", "胰岛素", "氨氯地平", "硝苯地平", "缬沙坦",
        "厄贝沙坦", "氯沙坦", "美托洛尔", "阿司匹林", "氯吡格雷", "华法林",
        "利伐沙班", "阿哌沙班", "达比加群", "阿托伐他汀", "瑞舒伐他汀",
        "左甲状腺素", "优甲乐",
    ]
    for medication in known_medications:
        if medication in text and medication not in medications:
            medications.append(medication)
    extracted["current_medications"] = medications
    extracted["allergies"] = list(dict.fromkeys(
        match.group(1).strip()
        for match in re.finditer(
            r"(?:对|曾对)([\u4e00-\u9fffA-Za-z0-9-]{1,20})(?:过敏|有过敏反应)",
            text,
        )
        if match.group(1).strip()
    ))
    menstruation_patterns = [
        r"(?:月经|经期)[^，,。；;\n]{0,30}",
        r"(?:怀孕|妊娠|孕期)[^，,。；;\n]{0,20}",
        r"(?:白带)[^，,。；;\n]{0,20}",
    ]
    menstruation = [
        match.group(0).strip()
        for pattern in menstruation_patterns
        for match in re.finditer(pattern, text)
    ]
    extracted["menstruation"] = "；".join(dict.fromkeys(menstruation))

    tongue, pulse = _extract_textual_tongue_pulse(text)
    extracted["tongue"] = tongue
    extracted["pulse"] = pulse
    extracted["negated_symptoms"] = negated_symptoms
    extracted["chief_complaint"] = "、".join(positive_symptoms[:8]) or text[:200]
    extracted["other_symptoms"] = positive_symptoms
    return ExtractedInfo(**extracted)


_NEGATION_PREFIXES = (
    "无", "没有", "未见", "未出现", "否认", "不伴", "并无", "没", "不",
)
_NEGATION_EXCEPTIONS = {"无汗", "口不渴", "不欲饮"}


def _mention_is_negated(text: str, start: int, keyword: str) -> bool:
    """判断关键词在当前分句内是否被明确否定。"""
    if keyword == "无汗":
        return False
    prefix = text[max(0, start - 14):start]
    compact = re.sub(r"\s+", "", prefix)
    if any(token in compact for token in ("不是没有", "并非没有", "不能排除", "不排除")):
        return False
    if re.search(r"(?:无|没有|未见|未出现|否认|不伴|并无|没|不)(?:明显|任何|持续|剧烈|异常)?$", compact):
        return True
    clause_start = max(
        text.rfind(mark, 0, start)
        for mark in ("，", ",", "。", "！", "!", "？", "?", "；", ";", "\n")
    )
    clause = re.sub(r"\s+", "", text[clause_start + 1:start])
    if any(token in clause for token in ("但", "却", "然而", "现", "出现", "转为")):
        return False
    return clause.startswith(_NEGATION_PREFIXES)


def _normalize_negated_symptom(value: str) -> str:
    compact = re.sub(r"[\s，,。；;、]", "", str(value or ""))
    if not compact:
        return ""
    if compact in _NEGATION_EXCEPTIONS:
        return compact
    for prefix in _NEGATION_PREFIXES:
        if compact.startswith(prefix):
            remainder = compact[len(prefix):]
            return f"无{remainder}" if remainder else ""
    return f"无{compact}"


def _extract_negated_symptoms(text: str) -> list[str]:
    """提取明确否认项；“无汗”等辨证体征不放入否定列表。"""
    keywords = [
        "发热", "怕冷", "怕热", "胸痛", "胸闷", "心悸", "气短",
        "呼吸困难", "腹胀", "腹痛", "腹泻", "便秘", "便溏",
        "口干", "口苦", "口渴", "头痛", "头晕", "乏力", "失眠",
        "多梦", "咳嗽", "恶心", "呕吐", "尿频", "夜尿", "耳鸣",
        "异常出汗", "出汗异常",
    ]
    negated: list[str] = []
    for keyword in keywords:
        for match in re.finditer(re.escape(keyword), text):
            if _mention_is_negated(text, match.start(), keyword):
                value = _normalize_negated_symptom(keyword)
                if value and value not in negated:
                    negated.append(value)
    return negated


def _is_negated_observation(value: str, negated_symptoms: list[str]) -> bool:
    normalized = CollectedDiagnoseInfo._normalize_observation(value)
    if normalized in _NEGATION_EXCEPTIONS:
        return False
    negated = {
        CollectedDiagnoseInfo._normalize_observation(item)
        for item in negated_symptoms
        if item
    }
    return CollectedDiagnoseInfo._is_negated_statement(normalized, negated)


def _positive_observation_text(value: str, negated_symptoms: list[str]) -> str:
    parts = [
        part.strip()
        for part in re.split(r"[，,、；;\n]+", str(value or ""))
        if part.strip()
    ]
    positive = [
        part for part in parts
        if not _is_negated_observation(part, negated_symptoms)
    ]
    return "、".join(positive)


def _extract_textual_tongue_pulse(text: str) -> tuple[dict[str, str], str]:
    """从患者文字中保守提取舌象和脉象，不对未描述内容做推断。"""
    tongue: dict[str, str] = {}
    tongue_clauses = [
        clause.strip()
        for clause in re.split(r"[。；;\n]+", text)
        if "舌" in clause or "苔" in clause
    ]
    if tongue_clauses:
        tongue["description"] = "；".join(tongue_clauses)[:200]

        colors = re.findall(
            r"(?:舌(?:质|体|边|尖)?(?:色)?(?:为|是|呈|：|:)?)(淡红偏暗|淡红|淡白|淡|暗红|红|绛|紫暗|青紫|紫)",
            text,
        )
        if colors:
            tongue["tongue_color"] = "、".join(dict.fromkeys(colors))

        shapes = [
            item for item in ("胖大", "淡胖", "胖", "瘦薄", "瘦", "齿痕", "裂纹", "点刺")
            if item in text
        ]
        if shapes:
            tongue["tongue_shape"] = "、".join(dict.fromkeys(shapes))

        coating_colors = re.findall(
            r"苔(?:色)?(?:为|是|呈|：|:)?(?:薄|厚|腻|润|燥|滑|少|剥)?(黄白相兼|灰黑|薄白|白|黄|灰|黑)",
            text,
        )
        if coating_colors:
            normalized_colors = [
                "白" if color == "薄白" else color
                for color in coating_colors
            ]
            tongue["coating_color"] = "、".join(dict.fromkeys(normalized_colors))

        coating_quality = [
            item for item in ("薄", "厚", "腻", "白腻", "黄腻", "润", "燥", "滑", "少苔", "剥苔", "无苔")
            if re.search(rf"苔[^，,。；;\n]{{0,6}}{re.escape(item)}|{re.escape(item)}苔", text)
        ]
        if "苔薄白" in text and "薄" not in coating_quality:
            coating_quality.append("薄")
        if coating_quality:
            tongue["coating_quality"] = "、".join(dict.fromkeys(coating_quality))

    pulse = ""
    pulse_match = re.search(
        r"脉(?:象)?(?:为|是|呈|：|:)?\s*([^，,。；;\n]{1,16})",
        text,
    )
    if pulse_match:
        pulse = pulse_match.group(1).strip()
        pulse = re.sub(r"^(?:象|见)", "", pulse)
    return tongue, pulse
