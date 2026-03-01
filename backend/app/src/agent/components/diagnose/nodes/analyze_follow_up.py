"""
分析追问节点

分析已收集信息，决定下一步行动
"""

from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from enum import Enum
from langgraph.errors import GraphInterrupt
from ..states import DiagnoseOverallState
from ..models import CollectedDiagnoseInfo
from ..config import diagnose_config
from ....tcm_builder import get_llm
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
3. 如果涉及脾胃、湿热问题且无舌像信息，建议上传舌像
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
- request_tongue: 请求上传舌像
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
    try:
        # 获取已收集的信息
        collected_info_dict = state.get("collected_info", {})
        if collected_info_dict:
            collected_info = CollectedDiagnoseInfo(**collected_info_dict)
        else:
            collected_info = CollectedDiagnoseInfo()

        # 获取追问轮数
        follow_up_count = state.get("follow_up_count", 0)
        max_rounds = diagnose_config.MAX_FOLLOW_UP_ROUNDS

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

        # 规则2: 信息足够，进入评估
        if collected_info.is_sufficient(
            min_categories=diagnose_config.MIN_REQUIRED_CATEGORIES
        ):
            # 检查是否需要舌像
            if _should_request_tongue(collected_info, state):
                tongue_question = "为了更准确地判断您的情况，能否拍一张舌头的照片上传？请在自然光下，伸出舌头拍摄。"
                # 暂停图执行，等待用户上传舌像
                user_response = interrupt(
                    {
                        "question": tongue_question,
                        "action": NextAction.REQUEST_TONGUE.value,
                    }
                )
                # 恢复后，将用户回答注入 messages
                return {
                    "next_action": NextAction.ASK_SYMPTOM.value,
                    "follow_up_count": follow_up_count,
                    "follow_up_question": tongue_question,
                    "messages": [HumanMessage(content=user_response)],
                    "steps": ["分析追问: 建议上传舌像"],
                }

            logger.info("信息收集充分，进入复杂度评估")
            return {
                "next_action": NextAction.ASSESS_COMPLEXITY.value,
                "follow_up_count": follow_up_count,
                "steps": ["分析追问: 信息充分，进入复杂度评估"],
            }

        # === LLM 决策（需要智能判断）===
        collected_summary = collected_info.to_summary()

        # 构建对话历史（最近3轮）
        collection_history = state.get("collection_history", [])
        conversation_history = ""
        if collection_history:
            recent_history = collection_history[-3:]  # 最近3轮
            for i, h in enumerate(recent_history, 1):
                conversation_history += f"第{h['round_number']}轮 - 用户: {h['user_input'][:50]}...\n"
        else:
            conversation_history = "暂无历史"

        system_prompt = FOLLOW_UP_SYSTEM_PROMPT.format(
            collected_summary=collected_summary,
            follow_up_count=follow_up_count,
            max_rounds=max_rounds,
            conversation_history=conversation_history,
            user_input=last_user_message,
        )

        llm = get_llm(
            llm_config=state.get("llm_config"),
            temperature=diagnose_config.COLLECTION_TEMPERATURE,
        )

        structured_llm = llm.with_structured_output(FollowUpDecision)

        decision = await structured_llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"请分析并决定下一步行动。"),
            ]
        )

        # 处理决策结果
        action = decision.action.lower()
        if action not in [a.value for a in NextAction]:
            action = NextAction.ASK_SYMPTOM.value

        result = {
            "next_action": action,
            "follow_up_count": follow_up_count + 1,
            "steps": [f"分析追问: {decision.reasoning}"],
        }

        # 如果需要追问，添加追问问题
        if action == NextAction.ASK_SYMPTOM.value and decision.question:
            result["follow_up_question"] = decision.question
            result["answer"] = decision.question
            # 暂停图执行，等待用户输入
            user_response = interrupt(
                {
                    "question": decision.question,
                    "action": action,
                    "missing_info": decision.missing_info,
                }
            )
            # 恢复后，将用户回答注入 messages，路由回 collect_info 继续收集
            result["messages"] = [HumanMessage(content=user_response)]
            result["next_action"] = NextAction.ASK_SYMPTOM.value

        # 如果请求舌像
        if action == NextAction.REQUEST_TONGUE.value:
            tongue_question = "为了更准确地判断您的情况，能否拍一张舌头的照片上传？请在自然光下，伸出舌头拍摄。"
            result["follow_up_question"] = tongue_question
            result["answer"] = tongue_question
            # 暂停图执行，等待用户上传舌像
            user_response = interrupt(
                {
                    "question": tongue_question,
                    "action": action,
                }
            )
            # 恢复后，将用户回答注入 messages，继续收集信息
            result["messages"] = [HumanMessage(content=user_response)]
            result["next_action"] = NextAction.ASK_SYMPTOM.value

        logger.info(
            f"追问决策: {action}, 问题: {decision.question[:50] if decision.question else 'N/A'}"
        )

        return result

    except GraphInterrupt:
        # interrupt 抛出的异常应该直接传播给 LangGraph 处理
        raise
    except Exception as e:
        # 其他真正的异常才需要处理降级
        logger.error(f"分析追问失败: {e}", exc_info=True)
        # 降级：直接进入评估
        return {
            "next_action": NextAction.ASSESS_COMPLEXITY.value,
            "steps": [f"分析追问: 失败，降级到复杂度评估 - {str(e)}"],
        }


def _should_request_tongue(collected_info: CollectedDiagnoseInfo, state: Dict) -> bool:
    """判断是否应该请求舌像"""
    # 如果已有舌像信息，不再请求
    if collected_info.tongue:
        return False

    # 如果配置禁用舌像请求
    if not diagnose_config.ENABLE_TONGUE_REQUEST:
        return False

    # 检查症状是否涉及需要舌诊的情况
    all_symptoms = collected_info.to_summary().lower()
    for keyword in TONGUE_RELATED_KEYWORDS:
        if keyword in all_symptoms:
            return True

    return False
