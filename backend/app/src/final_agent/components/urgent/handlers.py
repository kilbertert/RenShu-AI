from langchain_core.messages import SystemMessage
from ...tcm_states import TCMAgentState

async def handle_urgent_query(state: TCMAgentState, user_query: str) -> dict:
    """处理紧急医疗情况"""
    from ...tcm_builder import get_llm
    llm = get_llm( llm_config=state.llm_config)
    emergency_prompt = f"检测到紧急医疗情况：{user_query}。请以中医专家的身份，立即给出紧急避险建议，并强烈建议其联系急救中心(120)或前往最近的医院。"
    response = await llm.ainvoke([SystemMessage(content=emergency_prompt)])
    return {
        "answer": response.content,
        "steps": ["逻辑执行: 触发紧急告急处理"],
    }
