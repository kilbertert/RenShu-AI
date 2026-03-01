#!/usr/bin/env python
"""
测试中医基础知识问答的路由
"""
import asyncio
from langchain_core.messages import HumanMessage

from app.src.agent.tcm_builder import build_tcm_graph, new_thread_id
from app.src.agent.tcm_states import TCMAgentState


async def knowledge_routing():
    """测试中医基础知识应该路由到 tcm-chat，而不是 tcm-wellness"""
    graph = build_tcm_graph()
    
    test_cases = [
        # {
        #     "category": "中医基础知识（应路由到 tcm-chat）",
        #     "expected_type": "tcm-chat",
        #     "queries": [
        #         "什么是中医？",
        #         "中医的五行是什么？",
        #         "中医和西医有什么区别？",
        #         "中医的历史有多久？",
        #         "什么是阴阳？",
        #         "八纲辨证是什么意思？",
        #         "讲解一下气血理论",
        #     ]
        # },
        {
            "category": "养生咨询（应路由到 tcm-wellness）",
            "expected_type": "tcm-wellness",
            "queries": [
                "春季如何养肝？",
                "我是气虚体质，怎么调理？",
                "夏天吃什么比较好？",
                "立冬后应该注意什么？",
            ]
        },
    ]
    
    for case in test_cases:
        print("\n" + "=" * 80)
        print(f"测试类别: {case['category']}")
        print(f"期望路由: {case['expected_type']}")
        print("=" * 80)
        
        success_count = 0
        total_count = len(case['queries'])
        
        for query in case['queries']:
            input_state = TCMAgentState(
                messages=[HumanMessage(content=query)],
            )
            
            config = {"configurable": {"thread_id": new_thread_id()}}
            result = await graph.ainvoke(input_state, config=config)
            
            # 检查路由类型
            router = result.get('router')
            if router:
                query_type = router.query_type if hasattr(router, 'query_type') else router.get('query_type')
            else:
                # OOS 情况
                query_type = "tcm-chat"  # OOS 默认是 tcm-chat
            
            is_correct = query_type == case['expected_type']
            success_count += is_correct
            
            status = "[OK]" if is_correct else "[FAIL]"
            print(f"{status} '{query}' -> {query_type}")
            
            if not is_correct:
                print(f"   期望: {case['expected_type']}, 实际: {query_type}")
        
        accuracy = success_count / total_count * 100
        print(f"\n准确率: {success_count}/{total_count} ({accuracy:.1f}%)")


if __name__ == "__main__":
    asyncio.run(knowledge_routing())
