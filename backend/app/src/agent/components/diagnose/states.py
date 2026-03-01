"""
诊断子图状态定义

定义诊断子图的输入、内部和输出状态
"""

from typing import List, Optional, Dict, Any, Annotated
from typing_extensions import TypedDict, NotRequired
from operator import add
from langchain_core.messages import BaseMessage

from ...tcm_states import LLMConfig


class DiagnoseInputState(TypedDict):
    """诊断子图输入状态"""
    query: str                          # 用户当前输入
    messages: List[BaseMessage]         # 对话历史
    user_profile: Dict[str, Any]        # 用户画像（体质、既往史等）
    llm_config: Optional[LLMConfig]     # LLM 配置
    extracted_entities: NotRequired[Dict[str, Any]]  # 意图识别提取的实体


class DiagnoseOverallState(TypedDict):
    """诊断子图内部状态"""
    # === 输入继承 ===
    query: str
    messages: List[BaseMessage]
    user_profile: Dict[str, Any]
    llm_config: Optional[LLMConfig]
    extracted_entities: NotRequired[Dict[str, Any]]

    # === 信息收集 ===
    collected_info: NotRequired[Dict[str, Any]]      # 已收集的诊断信息
    collection_history: NotRequired[Annotated[List[Dict[str, Any]], add]]  # 收集历史
    follow_up_count: NotRequired[int]                # 追问轮数

    # === 多模态 ===
    tongue_analysis: NotRequired[Dict[str, Any]]     # 舌像分析结果
    report_analysis: NotRequired[Dict[str, Any]]     # 报告解读结果

    # === 复杂度评估 ===
    complexity: NotRequired[Dict[str, Any]]          # 复杂度评估结果

    # === 辨证结果 ===
    diagnosis_result: NotRequired[Dict[str, Any]]    # 辨证结果

    # === 流程控制 ===
    next_action: NotRequired[str]                    # 路由信号
    steps: NotRequired[Annotated[List[str], add]]    # 执行步骤记录

    # === 输出字段 ===
    answer: NotRequired[str]                         # 回复内容
    follow_up_question: NotRequired[str]             # 追问问题（如果需要）


class DiagnoseOutputState(TypedDict):
    """诊断子图输出状态"""
    answer: str                                      # 回复内容
    diagnosis_result: NotRequired[Dict[str, Any]]    # 辨证结果
    steps: List[str]                                 # 执行步骤
    follow_up_question: NotRequired[str]             # 追问问题（如果需要）
