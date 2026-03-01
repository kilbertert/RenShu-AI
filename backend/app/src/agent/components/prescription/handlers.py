from langchain_core.messages import HumanMessage, SystemMessage
from ...tcm_states import TCMAgentState
from ...tcm_prompts import TCM_PRESCRIPTION_SYSTEM_PROMPT
# from ...kg_subgraph import build_kg_subgraph

async def handle_prescription_query(state: TCMAgentState) -> dict:
    """处理方剂推荐查询"""
    return {}
    # from ...tcm_builder import get_llm
    # messages = state.messages
    # router = state.router
    # entities = router.extracted_entities if router else {}
    # syndrome = entities.get("syndrome", "")
    #
    # query = messages[-1].content if messages else ""
    #
    # # 调用知识图谱子图
    # kg_subgraph = build_kg_subgraph()
    #
    # try:
    #     kg_result = await kg_subgraph.ainvoke({
    #         "question": query,
    #         "query_type": "tcm-prescription",
    #         "entities": entities,
    #         "history": [],
    #     })
    #
    #     llm = get_llm( llm_config=state.llm_config)
    #     system_prompt = TCM_PRESCRIPTION_SYSTEM_PROMPT.format(
    #         query=query,
    #         syndrome=syndrome or "未指定",
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
    #         "steps": kg_result.get("steps", []) + ["方剂推荐回复生成完成"],
    #     }
    # except Exception as e:
    #     llm = get_llm( llm_config=state.llm_config)
    #     response = await llm.ainvoke([
    #         SystemMessage(content=TCM_PRESCRIPTION_SYSTEM_PROMPT.format(query=query, syndrome=syndrome)),
    #         HumanMessage(content=query),
    #     ])
    #
    #     return {
    #         "answer": response.content,
    #         "steps": [f"方剂推荐（降级模式）: {str(e)}"],
    #     }
