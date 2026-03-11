"""
治则治法专家 SubAgent

使用 DeepAgents 框架创建，专门根据辨证结果制定精准的治疗策略
"""

from typing import Any
from langchain_core.language_models import BaseChatModel

from langchain.agents import create_agent

from app.src.agent.components.diagnose.prompts.deepsearch_prompts import (
    TREATMENT_PRINCIPLE_EXPERT_PROMPT
)
from app.src.utils import get_logger

logger = get_logger("treatment_principle_expert")


def create_treatment_expert(llm: BaseChatModel) -> Any:
    """
    创建治则治法专家子 Agent
    
    功能：
    - 确定治本/治标/标本兼治策略
    - 制定扶正祛邪方案
    - 调整阴阳平衡
    - 考虑因时因地因人制宜
    
    Args:
        llm: 语言模型
        
    Returns:
        治则治法专家 Agent 实例
    """
    logger.info("创建治则治法专家 SubAgent")
    
    # 使用 DeepAgents 创建子 Agent
    agent = create_agent(
        name="treatment_principle_expert",
        model=llm,
        system_prompt=TREATMENT_PRINCIPLE_EXPERT_PROMPT,
    )
    
    return agent
