#!/usr/bin/env python
"""
快速验证修复效果
"""
import asyncio
from langchain_core.messages import HumanMessage

from app.src.agent.tcm_builder import build_tcm_graph, new_thread_id
from app.src.agent.tcm_states import TCMAgentState


async def test_fixes():
    """快速测试修复效果"""
    graph = build_tcm_graph()
    
    test_cases = [
        {
            "name": "问题1&2修复：中医基础知识不应被拦截",
            "queries": [
                "什么是中医？",
                "中医的五行是什么？",
                "中医和西医有什么区别？",
            ]
        },
        {
            "name": "问题3修复：中间件步骤应该可见",
            "queries": [
                "我想了解养生",
            ]
        },
    ]
    
    for case in test_cases:
        print("\n" + "=" * 80)
        print(f"测试: {case['name']}")
        print("=" * 80)
        
        for query in case['queries']:
            print(f"\n查询: {query}")
            
            input_state = TCMAgentState(
                messages=[HumanMessage(content=query)],
            )
            
            config = {"configurable": {"thread_id": new_thread_id()}}
            result = await graph.ainvoke(input_state, config=config)
            
            # 检查结果
            steps = result.get('steps', [])
            answer = result.get('answer', '')
            
            # 检查是否被错误拦截
            if "西医诊疗范畴" in answer or "超范围" in ''.join(steps):
                print(f"❌ 仍被拦截: {answer[:100]}")
            else:
                print(f"✅ 通过检查")
            
            # 检查中间件步骤
            middleware_steps = [s for s in steps if '[中间件]' in s or '安全检查' in s]
            if middleware_steps:
                print(f"✅ 中间件可见: {middleware_steps[0]}")
            else:
                print(f"⚠️ 中间件步骤缺失")
            
            print(f"📝 回答: {answer[:100]}...")


if __name__ == "__main__":
    asyncio.run(test_fixes())
