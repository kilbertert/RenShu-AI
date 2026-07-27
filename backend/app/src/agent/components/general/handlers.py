from langchain_core.messages import SystemMessage
from ...tcm_states import TCMAgentState
from ...tcm_prompts import TCM_WELLNESS_SYSTEM_PROMPT, TCM_ERROR_RESPONSE

async def respond_to_general_query(state: TCMAgentState) -> dict:
    """处理一般性回答（闲聊/通用对话）"""
    from ...tcm_builder import get_llm
    messages = state.messages
    user_profile = state.user_profile or {}

    # 使用 state.llm_config 获取 LLM
    llm = get_llm(llm_config=state.llm_config)

    # 构建系统提示
    system_prompt = """"
    你是一个乐于助人的聊天助手，你的名字叫做小义，你擅长帮助用户解决各种各样的问题。
    请你使用欢快的语气和通俗易懂的方式回答用户的问题。

    下面是交互的上下文，请根据上下文内容生成回答。
    """
    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            *messages[-12:],  # 保留最近多轮用户/助手消息作为上下文
        ])

        return {
            "answer": response.content,
            "steps": ["一般性回答生成完成"],
        }
    except Exception as e:
        return {
            "answer": TCM_ERROR_RESPONSE,
            "error": str(e),
        }
