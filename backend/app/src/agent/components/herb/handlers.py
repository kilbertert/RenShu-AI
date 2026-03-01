from langchain_core.messages import HumanMessage, SystemMessage
from ...tcm_states import TCMAgentState
from ...tcm_prompts import TCM_HERB_SYSTEM_PROMPT
# from ...kg_subgraph import build_kg_subgraph

async def handle_herb_query(state: TCMAgentState) -> dict:
    """处理药材咨询查询"""
    return {}
    # from ...tcm_builder import get_llm
    # messages = state.messages
    # router = state.router
    # entities = router.extracted_entities if router else {}
    #
    # # 提取药材名称
    # herbs = entities.get("herbs", [])
    # query = messages[-1].content if messages else ""
    #
    # # 调用知识图谱子图查询
    # kg_subgraph = build_kg_subgraph()
    #
    # try:
    #     kg_result = await kg_subgraph.ainvoke({
    #         "question": query,
    #         "query_type": "tcm-herb",
    #         "entities": entities,
    #         "history": [],
    #     })
    #
    #     # 使用LLM生成回答
    #     llm = get_llm(llm_config=state.llm_config)
    #     system_prompt = TCM_HERB_SYSTEM_PROMPT.format(
    #         query=query,
    #         herbs=herbs or "未指定",
    #     )
    #
    #     response = await llm.ainvoke([
    #         SystemMessage(content=system_prompt),
    #         HumanMessage(content=f"查询结果：{kg_result.get('query_results', [])}"),
    #         HumanMessage(content=query),
    #     ])
    #
    #     return {
    #         "answer": response.content,
    #         "cypher_queries": kg_result.get("cypher_queries", []),
    #         "steps": kg_result.get("steps", []) + ["药材咨询回复生成完成"],
    #     }
    # except Exception as e:
    #     # 降级处理：直接使用LLM回答
    #     llm = get_llm( llm_config=state.llm_config)
    #     response = await llm.ainvoke([
    #         SystemMessage(content=TCM_HERB_SYSTEM_PROMPT.format(query=query, herbs=herbs or "未指定")),
    #         HumanMessage(content=query),
    #     ])
    #
    #     return {
    #         "answer": response.content,
    #         "steps": [f"药材咨询（降级模式）: {str(e)}"],
    #     }
