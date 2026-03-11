"""
TCM 共享工具函数

统一替代以下文件中的重复函数：
- context/message_priority.py: _estimate_tokens, _get_content
- context/tool_trimmer.py: _estimate_tokens, _get_content
- middleware/context_manager.py: _estimate_tokens, _get_message_content
- middleware/filesystem.py: _estimate_tokens
- intent_recognition/context_enricher.py: _get_current_solar_term, _get_current_season
- components/diagnose/nodes/simple/simple_diagnosis.py: _get_current_solar_term, _format_user_profile
- components/diagnose/nodes/moderate_diagnosis/moderate_diagnosis.py: _get_current_solar_term, _format_user_profile
- components/diagnose/nodes/moderate_diagnosis/moderate_diagnosis_map_reduce.py: _get_current_solar_term, _format_user_profile
- context/summarization.py: _estimate_tokens, _get_content, _get_role
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
)

from .tcm_constants import SOLAR_TERMS_BY_MONTH, SOLAR_TERMS_PRECISE, SEASONS


# ============== Token 估算 ==============

def estimate_tokens(text: Union[str, Any]) -> int:
    """
    估算文本的 token 数量

    中文约 1.5 字符/token，英文约 4 字符/token

    统一替代 6 个文件中的 _estimate_tokens()

    Args:
        text: 文本或可序列化对象

    Returns:
        估算的 token 数量
    """
    if text is None:
        return 0

    if not isinstance(text, str):
        try:
            text = json.dumps(text, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(text)

    if not text:
        return 0

    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars

    return int(chinese_chars / 1.5 + other_chars / 4)


def estimate_messages_tokens(messages: List[Any]) -> int:
    """
    估算消息列表的总 token 数

    替代 context_manager.py:_estimate_total_tokens()

    Args:
        messages: 消息列表（BaseMessage 或 Dict）

    Returns:
        总 token 数
    """
    total = 0
    for msg in messages:
        content = get_message_content(msg)
        total += estimate_tokens(content)
    return total


# ============== 消息内容提取 ==============

def get_message_content(message: Any) -> str:
    """
    获取消息内容

    统一替代 5 个文件中的 _get_content()/_get_message_content()

    Args:
        message: BaseMessage、Dict 或其他消息对象

    Returns:
        消息内容字符串
    """
    if isinstance(message, dict):
        return message.get("content", "")
    elif hasattr(message, "content"):
        return message.content or ""
    return ""


def get_message_role(message: Any) -> str:
    """
    获取消息角色

    统一替代 4 个文件中的 _get_role()/_get_message_role()

    Args:
        message: BaseMessage、Dict 或其他消息对象

    Returns:
        消息角色字符串
    """
    if isinstance(message, dict):
        return message.get("role", "")
    elif isinstance(message, SystemMessage):
        return "system"
    elif isinstance(message, HumanMessage):
        return "human"
    elif isinstance(message, AIMessage):
        return "assistant"
    elif hasattr(message, "type"):
        return message.type
    return ""


# ============== 节气和季节 ==============

def get_current_solar_term(dt: Optional[datetime] = None, precise: bool = False) -> str:
    """
    获取当前节气

    统一替代 4 个文件中的 _get_current_solar_term()

    Args:
        dt: 日期时间（默认当前时间）
        precise: 是否使用精确日期计算

    Returns:
        当前节气名称
    """
    if dt is None:
        dt = datetime.now()

    if not precise:
        return SOLAR_TERMS_BY_MONTH.get(dt.month, "未知节气")

    # 精确版：根据月日匹配
    month, day = dt.month, dt.day

    for i, (term, (m, d)) in enumerate(SOLAR_TERMS_PRECISE):
        if month == m and day >= d:
            return term
        if month == m and day < d and i > 0:
            return SOLAR_TERMS_PRECISE[i - 1][0]

    # 年末年初
    if month == 1 and day < 6:
        return "冬至"

    return "未知"


def get_current_season(dt: Optional[datetime] = None) -> str:
    """
    获取当前季节

    替代 context_enricher.py 中的季节逻辑

    Args:
        dt: 日期时间（默认当前时间）

    Returns:
        当前季节名称
    """
    if dt is None:
        dt = datetime.now()

    month = dt.month
    for season, months in SEASONS.items():
        if month in months:
            return season
    return "未知"


# ============== 用户画像格式化 ==============

def format_user_profile(profile: Dict[str, Any]) -> str:
    """
    格式化用户画像

    统一替代 3 个文件中的 _format_user_profile()
    合并了 simple_diagnosis.py（更完整版）和 moderate_diagnosis.py 的实现

    Args:
        profile: 用户画像字典

    Returns:
        格式化的画像文本
    """
    if not profile:
        return "暂无用户健康档案"

    parts = []

    if profile.get("gender"):
        parts.append(f"性别：{profile['gender']}")

    if profile.get("age"):
        parts.append(f"年龄：{profile['age']}岁")
    elif profile.get("age_group"):
        parts.append(f"年龄段：{profile['age_group']}")

    if profile.get("constitution"):
        parts.append(f"体质类型：{profile['constitution']}")

    # 兼容两种字段名
    chronic = profile.get("chronic_conditions") or profile.get("chronic_diseases")
    if chronic:
        if isinstance(chronic, list):
            parts.append(f"既往病史：{', '.join(chronic)}")
        else:
            parts.append(f"既往病史：{chronic}")

    if profile.get("allergies"):
        allergies = profile["allergies"]
        if isinstance(allergies, list):
            parts.append(f"过敏史：{', '.join(allergies)}")
        else:
            parts.append(f"过敏史：{allergies}")

    return "\n".join(parts) if parts else "暂无用户健康档案"
