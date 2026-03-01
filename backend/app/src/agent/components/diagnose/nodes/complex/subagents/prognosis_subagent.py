"""
预后评估专家 SubAgent

使用 DeepAgents 框架创建，专门判断疾病的发展趋势和转归
"""

from typing import Any
from langchain_core.language_models import BaseChatModel

from langchain.agents import create_agent

from app.src.agent.components.diagnose.prompts.deepsearch_prompts import (
    PROGNOSIS_EVALUATION_EXPERT_PROMPT
)
from app.src.utils import get_logger

logger = get_logger("prognosis_expert")


def create_prognosis_expert(llm: BaseChatModel) -> Any:
    """
    创建预后评估专家子 Agent
    
    功能：
    - 评估正气盛衰
    - 分析邪正进退
    - 判断疾病传变规律
    - 预测病情发展趋势
    
    Args:
        llm: 语言模型
        
    Returns:
        预后评估专家 Agent 实例
    """
    logger.info("创建预后评估专家 SubAgent")
    
    # 使用 DeepAgents 创建子 Agent
    agent = create_agent(
        name="prognosis_evaluation_expert",
        model=llm,
        system_prompt=PROGNOSIS_EVALUATION_EXPERT_PROMPT,
    )
    
    return agent
