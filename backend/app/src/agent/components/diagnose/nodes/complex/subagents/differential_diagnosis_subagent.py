"""
鉴别诊断专家 SubAgent

使用 DeepAgents 框架创建，专门负责区分相似证型，避免误诊
"""

from typing import Any
from langchain_core.language_models import BaseChatModel

from langchain.agents import create_agent
from deepagents import create_deep_agent

from app.src.agent.components.diagnose.prompts.deepsearch_prompts import (
    DIFFERENTIAL_DIAGNOSIS_EXPERT_PROMPT
)
from app.src.utils import get_logger

logger = get_logger("differential_diagnosis_expert")


def create_differential_expert(llm: BaseChatModel) -> Any:
    """
    创建鉴别诊断专家子 Agent
    
    功能：
    - 区分相似证型（如肝郁脾虚 vs 脾虚肝郁）
    - 分析寒热真假、虚实真假
    - 识别一症多因的情况
    - 提供鉴别诊断建议
    
    Args:
        llm: 语言模型
        
    Returns:
        鉴别诊断专家 Agent 实例
    """
    logger.info("创建鉴别诊断专家 SubAgent")
    
    # 使用 DeepAgents 创建子 Agent
    # 鉴别诊断专家不需要工具，直接基于知识库进行推理
    agent = create_agent(
        name="differential_diagnosis_expert",
        model=llm,
        system_prompt=DIFFERENTIAL_DIAGNOSIS_EXPERT_PROMPT,
        # 无需工具，直接基于提供的数据进行推理分析
    )
    
    return agent
