"""
质疑验证专家 SubAgent

使用 DeepAgents 框架创建，专门对诊断结论进行质疑和验证
"""

from typing import Any
from langchain_core.language_models import BaseChatModel

from langchain.agents import create_agent

from app.src.agent.components.diagnose.prompts.deepsearch_prompts import (
    VERIFICATION_EXPERT_PROMPT
)
from app.src.utils import get_logger

logger = get_logger("verification_expert")


def create_verification_expert(llm: BaseChatModel) -> Any:
    """
    创建质疑验证专家子 Agent
    
    作为"魔鬼代言人"角色：
    - 对当前诊断提出质疑
    - 寻找可能的漏洞
    - 验证逻辑链的完整性
    - 确保没有遗漏重要的鉴别诊断
    
    验证维度：
    1. 证据充分性验证
    2. 逻辑一致性验证
    3. 鉴别诊断验证
    4. 治法合理性验证
    5. 安全性验证
    
    Args:
        llm: 语言模型
        
    Returns:
        质疑验证专家 Agent 实例
    """
    logger.info("创建质疑验证专家 SubAgent")
    
    # 使用 DeepAgents 创建子 Agent
    agent = create_agent(
        name="verification_expert",
        model=llm,
        system_prompt=VERIFICATION_EXPERT_PROMPT,
    )
    
    return agent
