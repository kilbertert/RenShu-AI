"""
方药推荐专家 SubAgent

使用 DeepAgents 框架创建，专门根据治则治法推荐合适的方剂和调理方案
"""

from typing import Any
from langchain_core.language_models import BaseChatModel

from langchain.agents import create_agent

from app.src.agent.components.diagnose.prompts.deepsearch_prompts import (
    PRESCRIPTION_RECOMMENDATION_EXPERT_PROMPT
)
from app.src.utils import get_logger

logger = get_logger("prescription_expert")


def create_prescription_expert(llm: BaseChatModel) -> Any:
    """
    创建方药推荐专家子 Agent
    
    功能：
    - 推荐主方及加减
    - 分析君臣佐使
    - 提供食疗方案
    - 建议穴位调理
    
    重要提示：本系统不开具处方，仅提供方剂参考和调理建议
    
    Args:
        llm: 语言模型
        
    Returns:
        方药推荐专家 Agent 实例
    """
    logger.info("创建方药推荐专家 SubAgent")
    
    # 使用 DeepAgents 创建子 Agent
    agent = create_agent(
        name="prescription_recommendation_expert",
        model=llm,
        system_prompt=PRESCRIPTION_RECOMMENDATION_EXPERT_PROMPT,
    )
    
    return agent
