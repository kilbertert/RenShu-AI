"""
简单辨证节点

简单病情的直接辨证（LLM 直接分析）
"""

from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.src.agent.tcm_builder import get_llm

from app.src.agent.components.diagnose.prompts import SIMPLE_DIAGNOSIS_PROMPT
from datetime import datetime

from app.src.agent.components.diagnose.states import DiagnoseOverallState
from app.src.agent.components.diagnose.models import CollectedDiagnoseInfo
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

        # 获取用户画像
        user_profile = state.get("user_profile", {})

        user_profile_desc = _format_user_profile(user_profile)
        
        # 获取当前节气
        solar_term = _get_current_solar_term()

        # 构建提示词
        prompt = SIMPLE_DIAGNOSIS_PROMPT.format(
            collected_info=collected_summary,
            tongue_analysis=tongue_desc,
            user_profile=user_profile_desc,
            solar_term=solar_term,
        )

        # 调用 LLM
        llm = get_llm(
            llm_config=state.get("llm_config"),
            temperature=diagnose_config.DIAGNOSIS_TEMPERATURE
        )

        # 检查是否启用 thinking 模式（从 llm_config 中获取）
        llm_config = state.get("llm_config")
        enable_thinking = llm_config.enable_thinking if llm_config else False
        
        if enable_thinking:
            # Thinking 模式：需要解析思考过程和诊断报告
            # 这里我们使用流式输出来分别发送 thinking 和 content
            # 但由于当前是 ainvoke，我们需要先获取完整响应，然后解析
            response = await llm.ainvoke([
                SystemMessage(content=prompt),
                HumanMessage(content="请开始您的辨证分析。")
            ])
            
            full_content = response.content
            
            # 尝试分离思考过程和诊断报告
            # 思考过程通常在 "🤔 **正在分析患者症状...**" 和 "---" 之间
            # 或者在 "【第一部分：思考过程】" 和 "【第二部分：诊断报告】" 之间
            thinking_content = ""
            diagnosis_content = full_content
            
            # 方法1：查找标记分隔符
            if "【第一部分：思考过程】" in full_content and "【第二部分：诊断报告】" in full_content:
                parts = full_content.split("【第二部分：诊断报告】")
                if len(parts) >= 2:
                    thinking_part = parts[0]
                    # 移除"【第一部分：思考过程】"标题
                    thinking_content = thinking_part.replace("【第一部分：思考过程】", "").strip()
                    diagnosis_content = parts[1].strip()
            # 方法2：查找思考emoji和分隔线
            elif "🤔" in full_content:
                # 查找第一个 "---" 作为分隔符
                parts = full_content.split("---", 1)
                if len(parts) >= 2:
                    thinking_content = parts[0].strip()
                    diagnosis_content = parts[1].strip()
            
            # 如果成功分离了思考过程，返回时包含 thinking_content
            # 注意：这里我们只能返回状态更新，实际的流式输出需要在上层处理
            # 为了支持 thinking 的流式输出，我们需要修改架构
            # 暂时先返回完整内容，后续可以优化为真正的流式 thinking
            
            answer = diagnosis_content if diagnosis_content else full_content
            
            logger.info(f"简单辨证完成（Thinking模式）")
            
            return {
                "answer": answer,
                "thinking_process": thinking_content if thinking_content else None,
                "steps": ["简单辨证: 完成（含思考过程）"],
            }
        else:
            # 非 Thinking 模式：直接返回诊断结果
            response = await llm.ainvoke([
                SystemMessage(content=prompt),
                HumanMessage(content="请开始您的辨证分析。")
            ])

            answer = response.content

            logger.info(f"简单辨证完成")

            return {
                "answer": answer,
                "steps": ["简单辨证: 完成"],
            }

    except Exception as e:
        logger.error(f"简单辨证失败: {e}", exc_info=True)
        return {
            "answer": f"抱歉，辨证分析过程中出现错误：{str(e)}。建议您前往医院进行详细检查。",
            "steps": [f"简单辨证: 失败 - {str(e)}"],
        }



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
