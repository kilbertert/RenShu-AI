"""
分析追问节点

分析已收集信息，决定下一步行动
"""

import re
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from langgraph.errors import GraphInterrupt
from ..states import DiagnoseOverallState
from ..models import CollectedDiagnoseInfo
from ..config import diagnose_config
from app.src.agent.safety import (
    detect_psychological_crisis,
    psychological_crisis_response,
)
from app.src.agent.tcm_states import LLMConfig
from app.src.utils import get_logger

logger = get_logger("analyze_follow_up")


class NextAction(str, Enum):
    """下一步行动"""

    ASK_SYMPTOM = "ask_symptom"  # 追问症状
    REQUEST_TONGUE = "request_tongue"  # 请求上传舌像
    REQUEST_REPORT = "request_report"  # 请求上传检验报告
    ASSESS_COMPLEXITY = "assess_complexity"  # 信息足够，进入复杂度评估
    INTENT_SWITCH = "intent_switch"  # 检测到意图切换，退出子图


class FollowUpDecision(BaseModel):
    """追问决策"""

    action: str = Field(
        description="下一步行动: ask_symptom/request_tongue/assess_complexity"
    )
    question: str = Field(default="", description="追问问题（如果需要追问）")
    reasoning: str = Field(default="", description="决策理由")
    missing_info: list[str] = Field(default_factory=list, description="缺失的信息类别")

    @field_validator("action", mode="before")
    @classmethod
    def normalize_action(cls, value):
        normalized = str(value or NextAction.ASK_SYMPTOM.value).strip().lower()
        aliases = {
            "ask_follow_up": NextAction.ASK_SYMPTOM.value,
            "follow_up": NextAction.ASK_SYMPTOM.value,
            "继续追问": NextAction.ASK_SYMPTOM.value,
            "追问症状": NextAction.ASK_SYMPTOM.value,
            "请求舌像": NextAction.REQUEST_TONGUE.value,
            "进入辨证": NextAction.ASSESS_COMPLEXITY.value,
            "信息足够": NextAction.ASSESS_COMPLEXITY.value,
        }
        return aliases.get(normalized, normalized)

    @field_validator("question", "reasoning", mode="before")
    @classmethod
    def normalize_nullable_text(cls, value):
        return "" if value is None else str(value)

    @field_validator("missing_info", mode="before")
    @classmethod
    def normalize_missing_info(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [
                item.strip()
                for item in re.split(r"[、,，;；\n]", value)
                if item.strip()
            ]
        return value


FOLLOW_UP_SYSTEM_PROMPT = """你是一位经验丰富的中医师，正在进行问诊。

**你的任务：**
分析已收集的信息，决定下一步行动。

**已收集的信息：**
{collected_summary}

**已追问轮数：** {follow_up_count} / {max_rounds}

**对话历史（最近3轮）：**
{conversation_history}

**决策规则：**
1. 如果主诉不明确，优先追问主诉
2. 如果寒热、汗出、头身、二便、饮食、睡眠这6类信息收集不足4类，继续追问
3. 舌像只作为可选增强信息；已有文字舌象、用户拒绝或已询问过时不得再次请求
4. 如果已追问 {max_rounds} 轮或信息足够，进入复杂度评估
5. **重要**：如果用户明确表示"没有"、"都没有"、"全部没有"等否定回答，不要重复问相同维度的问题
6. **重要**：避免重复之前已经问过的问题，要从不同维度收集信息

**追问技巧：**
- 根据已有症状推测可能的证型，针对性追问
- 一次只问1-2个相关问题
- 用通俗易懂的语言
- 如果用户否定了某个维度，转向其他维度（如从寒热转向饮食、睡眠等）
- 注意识别否定表达：没有、无、不、全都没有、都没有等

**请决定下一步行动：**
- ask_symptom: 继续追问症状（必须是新的维度，不能重复）
- request_tongue: 可选请求上传舌像；必须同时允许用户回复“跳过”继续
- assess_complexity: 信息足够，进入辨证分析

**用户最新输入：**
{user_input}
"""


# 需要舌像的关键词
TONGUE_RELATED_KEYWORDS = [
    "脾胃",
    "湿热",
    "痰湿",
    "食欲",
    "消化",
    "腹胀",
    "便溏",
    "口苦",
    "口干",
    "口臭",
    "舌",
    "苔",
]

TONGUE_DECLINE_KEYWORDS = [
    "跳过", "不方便", "无法上传", "不能上传", "暂时不能", "不想上传",
    "拒绝上传", "没有照片", "没有图片", "按文字继续", "直接继续",
]

TONGUE_QUESTION = (
    "如果方便，可以在自然光下上传一张舌头照片，帮助补充判断；"
    "这不是完成问诊的必要条件。若不方便，请直接回复“跳过”，我会根据文字信息继续分析。"
)


def _build_deterministic_follow_up_question(
    collected_info: CollectedDiagnoseInfo,
    report_analysis: Dict[str, Any] | None = None,
) -> str:
    """按十问缺口生成稳定追问；报告异常先给出明确、可行动的医学边界。"""
    prefix = _report_follow_up_prefix(report_analysis)
    if not collected_info.chief_complaint:
        question = (
            "请先具体描述最主要的不舒服是什么、持续了多久；另外请补充"
            "怕冷还是怕热、出汗、饮食、大小便和睡眠情况。"
        )
        return f"{prefix}{question}".strip()

    missing = collected_info.get_missing_categories()
    missing_text = "、".join(missing[:4])
    duration_text = "症状持续多久" if not collected_info.duration else ""
    requested = "、".join(item for item in (duration_text, missing_text) if item)
    question = (
        f"为了继续辨证，请补充{requested or '尚未说明的症状变化'}；"
        "如果正在服药或有慢性病，也请一并说明。"
    )
    return f"{prefix}{question}".strip()


def _report_follow_up_prefix(report_analysis: Dict[str, Any] | None) -> str:
    report = report_analysis or {}
    metrics = report.get("metrics") or []
    abnormal = [
        item
        for item in metrics
        if isinstance(item, dict)
        and item.get("abnormal_flag") in {"high", "low", "abnormal", "positive"}
    ]
    if not abnormal and not report.get("key_findings"):
        return ""

    summary = str(report.get("summary") or "").strip().rstrip("。；;，,")
    if not summary and abnormal:
        item = abnormal[0]
        summary = (
            f"{item.get('name', '报告指标')} {item.get('value', '')} {item.get('unit', '')}"
            "超出报告参考范围"
        ).strip()
    urgent = (
        "报告或症状含紧急风险提示，请优先按报告建议立即就医。"
        if report.get("urgent_warning")
        else "这需要结合症状尽快线下复核，单份报告不能直接确定病因或中医证型。"
    )
    return f"先说明报告：{summary}。{urgent}\n\n"


def _normalize_resume_value(
    value: Any,
) -> tuple[
    str,
    Dict[str, Any] | None,
    Dict[str, Any] | None,
    LLMConfig | None,
]:
    """兼容旧的纯文字 resume，并接收安全的结构化附件恢复载荷。"""
    if isinstance(value, dict):
        text = str(value.get("text") or "").strip()
        tongue_analysis = value.get("tongue_analysis")
        if not isinstance(tongue_analysis, dict):
            tongue_analysis = None
        report_analysis = value.get("report_analysis")
        if not isinstance(report_analysis, dict):
            report_analysis = None
        raw_llm_config = value.get("llm_config")
        try:
            llm_config = (
                raw_llm_config
                if isinstance(raw_llm_config, LLMConfig)
                else LLMConfig.model_validate(raw_llm_config)
                if isinstance(raw_llm_config, dict)
                else None
            )
        except (TypeError, ValueError):
            llm_config = None
        fallback = (
            "我已上传舌像，请结合舌像继续问诊。"
            if tongue_analysis
            else "我已上传医疗报告，请结合报告继续问诊。"
            if report_analysis
            else "请继续问诊。"
        )
        return text or fallback, tongue_analysis, report_analysis, llm_config
    return str(value or "").strip(), None, None, None


def _declines_tongue_image(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    return any(keyword in compact for keyword in TONGUE_DECLINE_KEYWORDS)


def _resume_updates(
    value: Any,
    state: DiagnoseOverallState,
    *,
    requested_tongue: bool = False,
) -> Dict[str, Any]:
    text, tongue_analysis, report_analysis, llm_config = _normalize_resume_value(value)
    updates: Dict[str, Any] = {
        "messages": [HumanMessage(content=text)],
    }
    if llm_config is not None:
        updates["llm_config"] = llm_config
    if tongue_analysis:
        updates["tongue_analysis"] = tongue_analysis
    elif state.get("tongue_analysis"):
        updates["tongue_analysis"] = state["tongue_analysis"]
    if report_analysis:
        updates["report_analysis"] = report_analysis
    elif state.get("report_analysis"):
        updates["report_analysis"] = state["report_analysis"]
    if requested_tongue:
        updates["tongue_request_count"] = int(
            state.get("tongue_request_count", 0) or 0
        ) + 1
        if not tongue_analysis and _declines_tongue_image(text):
            updates["tongue_request_declined"] = True
    crisis = detect_psychological_crisis(text)
    if crisis.is_crisis:
        updates.update({
            "next_action": NextAction.INTENT_SWITCH.value,
            "answer": psychological_crisis_response(),
            "follow_up_question": "",
            "steps": [f"安全检查: 心理危机拦截 ({crisis.matched_text})"],
        })
    return updates


async def analyze_and_follow_up(state: DiagnoseOverallState) -> Dict[str, Any]:
    """
    分析已收集信息，决定下一步行动

    输出 next_action:
    - "ask_symptom": 追问症状
    - "request_tongue": 请求上传舌像
    - "request_report": 请求上传检验报告
    - "assess_complexity": 信息足够，进入复杂度评估
    - "intent_switch": 检测到意图切换，退出子图

    Args:
        state: 当前状态

    Returns:
        dict: 更新的状态字段
    """
    collected_info = CollectedDiagnoseInfo()
    follow_up_count = int(state.get("follow_up_count", 0) or 0)
    max_rounds = diagnose_config.MAX_FOLLOW_UP_ROUNDS
    try:
        # 获取已收集的信息
        collected_info_dict = state.get("collected_info", {})
        if collected_info_dict:
            collected_info = CollectedDiagnoseInfo(**collected_info_dict)
        else:
            collected_info = CollectedDiagnoseInfo()

        # 获取追问轮数
        # 获取用户最新输入
        messages = state.get("messages", [])
        last_user_message = ""
        for msg in reversed(messages):
            # 兼容两种格式：HumanMessage 对象和字典
            if isinstance(msg, HumanMessage):
                last_user_message = msg.content
                break
            elif isinstance(msg, dict) and msg.get("role") == "user":
                last_user_message = msg.get("content", "")
                break

        # === 规则判断（快速路径）===

        # 规则0: 检测否定回答（用户表示没有症状）
        negative_keywords = ["没有", "无", "全都没有", "都没有", "全部没有", "没啥", "没什么"]
        is_negative_response = any(keyword in last_user_message for keyword in negative_keywords)
        
        # 如果是否定回答且已经追问过2轮以上，考虑进入评估
        if is_negative_response and follow_up_count >= 2:
            logger.info(f"检测到否定回答且已追问{follow_up_count}轮，考虑进入评估")
            # 如果已有基本信息（至少2类），直接进入评估
            if collected_info.get_filled_count() >= 2:
                logger.info("已有基本信息，进入复杂度评估")
                return {
                    "next_action": NextAction.ASSESS_COMPLEXITY.value,
                    "follow_up_count": follow_up_count,
                    "steps": ["分析追问: 用户否定回答且已有基本信息，进入评估"],
                }

        # 规则1: 达到最大追问轮数，强制进入评估
        if follow_up_count >= max_rounds:
            logger.info(f"达到最大追问轮数 {max_rounds}，进入复杂度评估")
            return {
                "next_action": NextAction.ASSESS_COMPLEXITY.value,
                "follow_up_count": follow_up_count,
                "steps": [f"分析追问: 达到最大轮数，进入复杂度评估"],
            }

        report_analysis = state.get("report_analysis") or None

        # 规则2: 信息足够，直接进入评估。舌像仅接受用户主动上传，永不阻塞主路径。
        if collected_info.is_sufficient(
            min_categories=diagnose_config.MIN_REQUIRED_CATEGORIES
        ):
            logger.info("信息收集充分，进入复杂度评估")
            return {
                "next_action": NextAction.ASSESS_COMPLEXITY.value,
                "follow_up_count": follow_up_count,
                "steps": ["分析追问: 信息充分，进入复杂度评估"],
            }

        # 十问信息充分度是确定性规则，不再调用 LLM 决策，避免慢调用和同质化追问。
        question = _build_deterministic_follow_up_question(
            collected_info,
            report_analysis,
        )
        result = {
            "next_action": NextAction.ASK_SYMPTOM.value,
            "follow_up_count": follow_up_count + 1,
            "follow_up_question": question,
            "answer": question,
            "steps": ["分析追问: 按十问缺失维度生成确定性追问"],
        }
        user_response = interrupt(
            {
                "question": question,
                "action": NextAction.ASK_SYMPTOM.value,
                "missing_info": collected_info.get_missing_categories(),
            }
        )
        result.update(_resume_updates(user_response, state))
        return result

    except GraphInterrupt:
        # interrupt 抛出的异常应该直接传播给 LangGraph 处理
        raise
    except Exception as e:
        # 信息不足时，结构化决策失败必须降级为确定性追问，不能误入辨证。
        logger.error(f"分析追问失败: {e}", exc_info=True)
        if (
            follow_up_count < max_rounds
            and not collected_info.is_sufficient(
                min_categories=diagnose_config.MIN_REQUIRED_CATEGORIES
            )
        ):
            question = _build_deterministic_follow_up_question(
                collected_info,
                state.get("report_analysis"),
            )
            user_response = interrupt({
                "question": question,
                "action": NextAction.ASK_SYMPTOM.value,
                "missing_info": [],
            })
            return {
                "next_action": NextAction.ASK_SYMPTOM.value,
                "follow_up_count": follow_up_count + 1,
                "follow_up_question": question,
                "answer": question,
                "steps": [f"分析追问: 确定性追问降级 - {str(e)}"],
                **_resume_updates(user_response, state),
            }

        return {
            "next_action": NextAction.ASSESS_COMPLEXITY.value,
            "follow_up_count": follow_up_count,
            "steps": [f"分析追问: 已达追问上限，进入复杂度评估 - {str(e)}"],
        }


def _should_request_tongue(collected_info: CollectedDiagnoseInfo, state: Dict) -> bool:
    """判断是否应该请求舌像"""
    # 如果已有舌像信息，不再请求
    if collected_info.tongue or state.get("tongue_analysis"):
        return False

    # 如果配置禁用舌像请求
    if not diagnose_config.ENABLE_TONGUE_REQUEST:
        return False

    # 舌像是可选增强项，只能询问一次；明确跳过后不得再次请求。
    if state.get("tongue_request_declined"):
        return False
    if int(state.get("tongue_request_count", 0) or 0) >= 1:
        return False

    # 只检查阳性症状，避免“没有口干口苦”触发舌像请求。
    all_symptoms = "、".join(collected_info.get_all_symptoms()).lower()
    for keyword in TONGUE_RELATED_KEYWORDS:
        if keyword in all_symptoms:
            return True

    return False
