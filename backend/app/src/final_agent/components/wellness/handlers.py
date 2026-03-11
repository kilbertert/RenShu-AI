from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from .states import WellnessOverallState
from ...tcm_prompts import TCM_WELLNESS_SYSTEM_PROMPT
from ...tcm_states import TCMAgentState

async def call_wellness_subgraph(state: TCMAgentState, config: RunnableConfig) -> dict:
    """封装养生子图的调用节点 (Teacher Pattern)"""
    from .builder import create_wellness_graph

    # 1. 编译子图
    wellness_graph = create_wellness_graph()

    # 2. 准备输入
    # 从 state.router 中获取意图识别的子类型和级别，而不是解析reasoning字符串
    # 通过意图识别模块，sub_type 应该已经在 route_result.classification 中被确定
    sub_type = "general"  # 默认值
    if state.router and hasattr(state.router, 'classification'):
        # 从意图识别结果中获取 sub_type，如果没有则默认为 'general'
        # 这里假设 intent_router 已经将 sub_type 传递给了 TCMRouter
        # 如果没有传递，我们需要在 router.py 中的 analyze_and_route_query 中确保传递
        sub_type = getattr(state.router.classification, 'sub_type', 'general')

    input_data = {
        "query": state.messages[-1].content if state.messages else "",
        "user_profile": state.user_profile or {},
        "sub_type": sub_type,
        "llm_config": state.llm_config,  # 传递 LLM 配置到子图
    }

    # 3. 执行子图（传递 config 使子图事件传播到父图的 astream_events）
    result = await wellness_graph.ainvoke(input_data, config)

    # 4. 返回结果合并到主图
    return {
        "answer": result["answer"],
        "steps": [f"子图执行: 养生流 -> {sub_type}"] + result.get("steps", [])
    }

async def handle_wellness_seasonal(state: WellnessOverallState) -> dict:
    """处理季节养生"""
    from ...tcm_builder import get_llm
    llm_config = state.get('llm_config')
    llm = get_llm(temperature=0.7, llm_config=llm_config)
    system_prompt = "你是一位中医专家，专门负责【季节养生】咨询。请根据当前的节气、季节和用户的体质（如有），提供专业的饮食、起居和经络调理建议。"
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state['query'])
    ])
    return {"answer": response.content, "steps": ["季节养生逻辑执行完成"]}

async def handle_wellness_daily(state: WellnessOverallState) -> dict:
    """处理日常科普"""
    from ...tcm_builder import get_llm
    llm_config = state.get('llm_config')
    llm = get_llm(temperature=0.7, llm_config=llm_config)
    system_prompt = "你是一位中医科普专家，专门负责【日常养生】咨询。请用通俗易懂的语言解释中医养生常识，回答日常保健问题。"
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state['query'])
    ])
    return {"answer": response.content, "steps": ["日常养生逻辑执行完成"]}

async def handle_wellness_constitution(state: WellnessOverallState) -> dict:
    """处理体质调理"""
    from ...tcm_builder import get_llm
    llm_config = state.get('llm_config')
    llm = get_llm(temperature=0.7, llm_config=llm_config)
    system_prompt = "你是一位中医体质辨识与调理专家，专门负责【体质调理】咨询。请根据用户描述的症状或体质（如气虚、湿热等），提供针对性的个性化调理方案。"
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state['query'])
    ])
    return {"answer": response.content, "steps": ["体质调理逻辑执行完成"]}

async def handle_wellness_general(state: WellnessOverallState) -> dict:
    """处理通用/复杂养生"""
    from ...tcm_builder import get_llm
    llm_config = state.get('llm_config')
    llm = get_llm(temperature=0.7, llm_config=llm_config)
    system_prompt = TCM_WELLNESS_SYSTEM_PROMPT.format(
        user_profile=state.get('user_profile') or "未知",
        query=state['query'],
    )
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state['query'])
    ])
    return {"answer": response.content, "steps": ["通用养生逻辑执行完成"]}
