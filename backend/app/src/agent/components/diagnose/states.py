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
    user_id: NotRequired[str]           # 用户ID（病例落库）
    conversation_id: NotRequired[str]   # 会话ID（病例落库）
    thread_id: NotRequired[str]         # LangGraph线程ID（病例落库）
    user_profile: Dict[str, Any]        # 用户画像（体质、既往史等）
    llm_config: Optional[LLMConfig]     # LLM 配置
    extracted_entities: NotRequired[Dict[str, Any]]  # 意图识别提取的实体
    tongue_analysis: NotRequired[Dict[str, Any]]
    report_analysis: NotRequired[Dict[str, Any]]


class DiagnoseOverallState(TypedDict):
    """诊断子图内部状态"""
    # === 输入继承 ===
    query: str
    messages: List[BaseMessage]
    user_id: NotRequired[str]
    conversation_id: NotRequired[str]
    thread_id: NotRequired[str]
    user_profile: Dict[str, Any]
    llm_config: Optional[LLMConfig]
    extracted_entities: NotRequired[Dict[str, Any]]

    # === 信息收集 ===
    collected_info: NotRequired[Dict[str, Any]]      # 已收集的诊断信息
    collection_history: NotRequired[Annotated[List[Dict[str, Any]], add]]  # 收集历史
    follow_up_count: NotRequired[int]                # 追问轮数
    tongue_request_count: NotRequired[int]           # 舌像可选请求次数（最多一次）
    tongue_request_declined: NotRequired[bool]       # 用户已明确跳过舌像

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
    error: NotRequired[str]                          # 诊断生成失败；失败病例不得落库

    # === 输出字段 ===
    answer: NotRequired[str]                         # 回复内容
    follow_up_question: NotRequired[str]             # 追问问题（如果需要）
    error: NotRequired[str]                          # 明确失败状态


class DiagnoseOutputState(TypedDict):
    """诊断子图输出状态"""
    answer: str                                      # 回复内容
    diagnosis_result: NotRequired[Dict[str, Any]]    # 辨证结果
    steps: List[str]                                 # 执行步骤
    follow_up_question: NotRequired[str]             # 追问问题（如果需要）
