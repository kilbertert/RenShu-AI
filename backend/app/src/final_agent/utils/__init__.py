"""
共享工具函数和常量

消除 agent/ 目录中 6+ 个文件的重复函数：
- estimate_tokens: 统一 token 估算
- get_message_content: 统一消息内容提取
- get_current_solar_term: 统一节气获取
- format_user_profile: 统一画像格式化
"""

from .tcm_utils import (
    estimate_tokens,
    estimate_messages_tokens,
    get_message_content,
    get_message_role,
    get_current_solar_term,
    get_current_season,
    format_user_profile,
)

from .tcm_constants import (
    TCM_KEYWORDS,
    TCM_KEYWORDS_FLAT,
    SOLAR_TERMS_BY_MONTH,
    SOLAR_TERMS_PRECISE,
    SEASONS,
    SYMPTOM_KEYWORDS,
    HERB_KEYWORDS,
    PRESCRIPTION_KEYWORDS,
    REGION_CHARACTERISTICS,
)

__all__ = [
    "estimate_tokens",
    "estimate_messages_tokens",
    "get_message_content",
    "get_message_role",
    "get_current_solar_term",
    "get_current_season",
    "format_user_profile",
    "TCM_KEYWORDS",
    "TCM_KEYWORDS_FLAT",
    "SOLAR_TERMS_BY_MONTH",
    "SOLAR_TERMS_PRECISE",
    "SEASONS",
    "SYMPTOM_KEYWORDS",
    "HERB_KEYWORDS",
    "PRESCRIPTION_KEYWORDS",
    "REGION_CHARACTERISTICS",
]
