#!/usr/bin/env python
"""
TCM Builder 全场景自动化测试
测试中间件效果和四类回答场景
"""
import asyncio
from langchain_core.messages import HumanMessage

from app.src.agent.tcm_builder import build_tcm_graph, new_thread_id
from app.src.agent.tcm_states import TCMAgentState


def print_section(title: str):
    """打印分隔线"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_result(query: str, result: dict):
    """打印测试结果"""
    print(f"📝 查询: {query}")
    print(f"✅ 路由: {result.get('router', {}).get('business_type', 'N/A')}")
    print(f"📊 置信度: {result.get('router', {}).get('confidence', 0):.2f}")
    print(f"🎯 答案: {result.get('answer', 'N/A')[:200]}...")
    if result.get('steps'):
        print(f"🔧 步骤: {', '.join(result['steps'][:3])}...")
    print()


async def test_middleware():
    """测试中间件功能"""
    print_section("测试 1: 中间件效果验证")
    
    graph = build_tcm_graph()
    
    test_cases = [
        {
            "query": "我的身份证号是 110101199001011234，手机号 13800138000",
            "expected": "PII 脱敏中间件应检测到个人信息"
        },
        {
            "query": "我想了解中医",
            "expected": "正常通过所有中间件"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"[中间件测试 {i}] {case['expected']}")
        print(f"查询: {case['query']}")
        
        input_state = TCMAgentState(
            messages=[HumanMessage(content=case['query'])],
        )
        
        config = {"configurable": {"thread_id": new_thread_id()}}
        result = await graph.ainvoke(input_state, config=config)
        
        # 检查中间件执行步骤
        steps = result.get('steps', [])
        middleware_steps = [s for s in steps if any(kw in s for kw in 
            ['[中间件]', '安全检查', 'PII检测', 'Context'])]
        
        if middleware_steps:
            print(f"✅ 中间件执行: {', '.join(middleware_steps[:3])}")
        else:
            print(f"⚠️ 未检测到中间件步骤（可能被其他步骤覆盖）")
        
        print(f"📄 答案: {result.get('answer', 'N/A')[:150]}...")
        print("-" * 80)


async def test_general_qa():
    """测试场景 1: 一般性回答（General）"""
    print_section("测试 2: 一般性回答 (General)")
    
    graph = build_tcm_graph()
    
    test_queries = [
        "什么是中医？",
        "中医和西医有什么区别？",
        "中医的五行是什么？",
        "中医的历史有多久？"
    ]
    
    for query in test_queries:
        input_state = TCMAgentState(
            messages=[HumanMessage(content=query)],
        )
        
        config = {"configurable": {"thread_id": new_thread_id()}}
        result = await graph.ainvoke(input_state, config=config)
        
        print_result(query, result)
        
        # 验证：一般性回答应该路由到 tcm-chat (对应 respond_to_general_query 节点)
        # 注意：OOS/闲聊/一般知识问答都是 tcm-chat
        router = result.get('router')
        if router:
            query_type = router.query_type if hasattr(router, 'query_type') else router.get('query_type')
            # 一般性问答可能是 tcm-chat 或 tcm-wellness (养生知识也算)
            assert query_type in ['tcm-chat', 'tcm-wellness'], \
                f"一般性问答路由错误: 应为 tcm-chat 或 tcm-wellness，实际为 {query_type}"
        else:
            # 如果没有 router，检查是否有答案（可能被 OOS 处理）
            assert result.get('answer'), "应该有回答内容"


async def test_urgent():
    """测试场景 2: 紧急情况（Urgent）"""
    print_section("测试 3: 紧急情况 (Urgent)")
    
    graph = build_tcm_graph()
    
    test_queries = [
        "我胸口剧痛，呼吸困难，冒冷汗",
        "突然头晕目眩，站不稳，想吐",
        "高烧40度，浑身发抖",
        "突然晕倒了，刚醒过来"
    ]
    
    for query in test_queries:
        input_state = TCMAgentState(
            messages=[HumanMessage(content=query)],
        )
        
        config = {"configurable": {"thread_id": new_thread_id()}}
        result = await graph.ainvoke(input_state, config=config)
        
        print_result(query, result)
        
        # 验证紧急标记
        assert result.get('should_seek_doctor') == True, \
            "紧急情况应标记 should_seek_doctor=True"
        assert "就医" in result.get('answer', ''), \
            "紧急回答应包含就医建议"


async def test_diagnose():
    """测试场景 3: 诊断回答（Diagnose）"""
    print_section("测试 4: 诊断回答 (Diagnose)")
    
    graph = build_tcm_graph()
    
    test_queries = [
        "我最近失眠多梦，心烦意乱",
        "总是感觉疲劳乏力，食欲不振",
        "手脚冰凉，怕冷",
        "口干舌燥，容易上火"
    ]
    
    for query in test_queries:
        input_state = TCMAgentState(
            messages=[HumanMessage(content=query)],
        )
        
        config = {"configurable": {"thread_id": new_thread_id()}}
        result = await graph.ainvoke(input_state, config=config)
        
        print_result(query, result)
        
        # 验证：诊断应该路由到 tcm-diagnose
        router = result.get('router')
        if router:
            query_type = router.query_type if hasattr(router, 'query_type') else router.get('query_type')
            assert query_type == 'tcm-diagnose', \
                f"诊断路由错误: 应为 tcm-diagnose，实际为 {query_type}"


async def test_wellness():
    """测试场景 4: 养生回答（Wellness）"""
    print_section("测试 5: 养生回答 (Wellness)")
    
    graph = build_tcm_graph()
    
    test_queries = [
        "如何养生保健？",
        "春季如何调理身体？",
        "吃什么食物可以补气血？",
        "平时怎么预防感冒？"
    ]
    
    for query in test_queries:
        input_state = TCMAgentState(
            messages=[HumanMessage(content=query)],
        )
        
        config = {"configurable": {"thread_id": new_thread_id()}}
        result = await graph.ainvoke(input_state, config=config)
        
        print_result(query, result)
        
        # 验证：养生应该路由到 tcm-wellness
        router = result.get('router')
        if router:
            query_type = router.query_type if hasattr(router, 'query_type') else router.get('query_type')
            assert query_type == 'tcm-wellness', \
                f"养生路由错误: 应为 tcm-wellness，实际为 {query_type}"


async def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("  TCM Builder 全场景自动化测试")
    print("=" * 80)
    
    try:
        # 1. 中间件测试
        await test_middleware()
        
        # 2. 一般性回答
        await test_general_qa()
        
        # 3. 紧急情况
        await test_urgent()
        
        # 4. 诊断回答
        await test_diagnose()
        
        # 5. 养生回答
        await test_wellness()
        
        print_section("✅ 所有测试完成")
        print("所有场景测试通过！中间件和四类回答功能正常。")
        
    except AssertionError as e:
        print_section("❌ 测试失败")
        print(f"错误: {e}")
        raise
    except Exception as e:
        print_section("❌ 测试异常")
        print(f"异常: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())
