"""
诊断流处理器

处理中医问诊相关的查询：
- 症状咨询
- 综合问诊
- 舌诊分析（需要图片）

架构升级（2026-02）：
- 集成诊断子图（DiagnoseSubgraph）
- 支持多轮追问、复杂度评估、分级辨证
"""

import re

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt

from ...tcm_states import PrescriptionInfo, SyndromeResult, TCMAgentState
from .builder import get_diagnose_graph
from app.src.utils import get_logger

logger = get_logger("diagnose_handler")


async def handle_diagnose_query(state: TCMAgentState, config: RunnableConfig) -> dict:
    """
    处理诊断类查询

    通过调用诊断子图实现：
    - 多轮信息收集
    - 复杂度评估
    - 分级辨证（简单/中等/复杂）

    Args:
        state: 当前状态

    Returns:
        dict: 更新的状态字段
    """
    messages = state.messages
    if not messages:
        return {"error": "No messages to process"}

    # 获取最后一条用户消息
    last_user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_query = msg.content
            break

    if not last_user_query:
        return {"error": "No user query found"}

    try:
        # 获取诊断子图
        diagnose_graph = get_diagnose_graph()

        # 准备子图输入
        subgraph_input = {
            "query": last_user_query,
            "messages": list(messages),
            "user_id": state.user_id,
            "conversation_id": state.conversation_id,
            "thread_id": config.get("configurable", {}).get("thread_id"),
            "user_profile": state.user_profile or {},
            "llm_config": state.llm_config,
            "extracted_entities": state.router.extracted_entities if state.router else {},
            "tongue_analysis": (
                state.tongue_analysis.model_dump(mode="json")
                if state.tongue_analysis else None
            ),
            "report_analysis": (
                state.report_analysis.model_dump(mode="json")
                if hasattr(state.report_analysis, "model_dump")
                else state.report_analysis
            ),
        }

        # 调用诊断子图（传递 config 使子图事件传播到父图的 astream_events）
        logger.info("调用诊断子图...")
        result = await diagnose_graph.ainvoke(subgraph_input, config)

        # 提取结果
        answer = result.get("answer", "")
        diagnosis_result = result.get("diagnosis_result")
        diagnosis_error = result.get("error")
        steps = result.get("steps", [])
        follow_up_question = result.get("follow_up_question")

        # 如果有追问问题，使用追问问题作为回复
        if follow_up_question and not answer:
            answer = follow_up_question

        # 更新用户画像中的症状记录
        user_profile = state.user_profile or {}
        new_symptoms = _extract_symptoms(last_user_query)
        if new_symptoms:
            existing = user_profile.get("recent_symptoms", [])
            existing.extend([s for s in new_symptoms if s not in existing])
            user_profile["recent_symptoms"] = existing[-10:]  # 最多保留10个

        logger.info(f"诊断子图完成，步骤: {steps}")

        syndrome_result = None
        prescriptions = []
        should_seek_doctor = False
        if isinstance(diagnosis_result, dict):
            syndrome_name = diagnosis_result.get("syndrome") or "未明确"
            syndrome_result = SyndromeResult(
                syndrome_name=syndrome_name,
                confidence=float(diagnosis_result.get("confidence") or 0),
                symptoms_matched=diagnosis_result.get("syndrome_evidence") or [],
                treatment_principle=diagnosis_result.get("treatment_principle") or "",
                recommended_prescriptions=[
                    item.get("name")
                    for item in diagnosis_result.get("prescriptions", [])
                    if isinstance(item, dict) and item.get("name")
                ],
            )
            for item in diagnosis_result.get("prescriptions", []) or []:
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                prescriptions.append(PrescriptionInfo(
                    name=item["name"],
                    source=item.get("source") or "",
                    composition=item.get("composition") or [],
                    effects=item.get("rationale") or "",
                    usage=item.get("usage") or "",
                    cautions=item.get("cautions") or [],
                    syndrome=syndrome_name,
                ))
            should_seek_doctor = bool(diagnosis_result.get("should_seek_doctor"))

        return {
            "answer": answer,
            "diagnosis_result": diagnosis_result,
            "syndrome_result": syndrome_result,
            "prescriptions": prescriptions,
            "should_seek_doctor": should_seek_doctor,
            "error": diagnosis_error,
            "user_profile": user_profile,
            "steps": steps or ["诊断处理: 完成中医问诊分析"],
        }

    except GraphInterrupt:
        # GraphInterrupt 是 LangGraph 的中断机制，必须向上传播
        # 这是多轮追问的核心机制，不能捕获
        logger.info("诊断子图触发中断（多轮追问），向上传播")
        raise
    except Exception as e:
        logger.error(f"诊断处理失败: {e}", exc_info=True)
        # 降级到简单处理
        return await _fallback_diagnose(state, last_user_query, str(e))


async def _fallback_diagnose(state: TCMAgentState, query: str, error: str) -> dict:
    """
    降级处理：当子图调用失败时使用简单 LLM 处理

    Args:
        state: 当前状态
        query: 用户查询
        error: 错误信息

    Returns:
        dict: 更新的状态字段
    """
    from langchain_core.messages import SystemMessage
    from ...tcm_builder import get_llm

    logger.warning(f"诊断子图失败，降级到简单处理: {error}")

    FALLBACK_PROMPT = """你是一位经验丰富的中医师，擅长望闻问切四诊合参。

## 问诊原则
1. 详细���解患者的症状、病史、生活习惯
2. 注意症状的性质、部位、时间规律
3. 结合四诊信息进行辨证分析
4. 给出中医辨证和调理建议

## 注意事项
- 始终提醒患者：中医建议仅供参考，严重情况请及时就医
- 不做西医诊断，不开具处方药
- 关注患者的整体状态，而非单一症状

请根据患者的描述进行分析。
"""

    try:
        llm = get_llm(llm_config=state.llm_config)

        response = await llm.ainvoke([
            SystemMessage(content=FALLBACK_PROMPT),
            HumanMessage(content=query)
        ])

        return {
            "answer": response.content,
            "steps": [f"诊断处理: 降级处理完成 (原因: {error})"],
        }

    except Exception as e2:
        logger.error(f"降级处理也失败: {e2}", exc_info=True)
        return {
            "answer": "抱歉，问诊分析过程中出现问题。请稍后重试，或换种方式描述您的症状。",
            "error": str(e2),
            "steps": [f"诊断处理: 失败 - {str(e2)}"],
        }


def _extract_symptoms(text: str) -> list:
    """从文本中提取症状关键词"""
    from .nodes.collect_info import _mention_is_negated

    symptom_keywords = [
        "头痛", "头晕", "发热", "咳嗽", "乏力", "失眠", "腹痛", "腹泻",
        "便秘", "恶心", "呕吐", "胸闷", "心悸", "气短", "盗汗", "自汗",
        "口干", "口苦", "食欲不振", "怕冷", "怕热", "腰痛", "关节痛",
        "眩晕", "耳鸣", "健忘", "多梦", "烦躁", "焦虑", "抑郁",
    ]

    found = []
    for symptom in symptom_keywords:
        for match in re.finditer(re.escape(symptom), text):
            if not _mention_is_negated(text, match.start(), symptom):
                found.append(symptom)
                break

    return found


# 保留旧函数名以兼容
async def call_diagnose_subgraph(state: TCMAgentState) -> dict:
    """调用诊断子图（兼容旧接口）"""
    return await handle_diagnose_query(state)
