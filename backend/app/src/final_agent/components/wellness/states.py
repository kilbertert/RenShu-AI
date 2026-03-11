from typing import List, Optional, Dict, Any, Annotated
from typing_extensions import TypedDict
from operator import add

from ...tcm_states import LLMConfig


class WellnessInputState(TypedDict):
    """养生子图的输入状态"""
    query: str
    user_profile: Dict[str, Any]
    sub_type: str  # seasonal, daily, constitution, general
    llm_config: Optional[LLMConfig]  # LLM 配置


class WellnessOverallState(TypedDict):
    """养生子图的内部状态"""
    query: str
    user_profile: Dict[str, Any]
    sub_type: str
    llm_config: Optional[LLMConfig]  # LLM 配置
    answer: Optional[str]
    steps: Annotated[List[str], add]


class WellnessOutputState(TypedDict):
    """养生子图的输出状态"""
    answer: str
    steps: List[str]
