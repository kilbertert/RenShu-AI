"""
测试完整的诊断子图

包括：
- 信息收集循环
- 复杂度评估
- 简单/中等/复杂诊断（DeepSearch Agent）
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("正在导入模块...")

try:
    from app.src.agent.components.diagnose.builder import create_diagnose_graph
    print("✅ 诊断子图模块导入成功")
except Exception as e:
    print(f"❌ 诊断子图模块导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# 测试数据
SIMPLE_CASE = {
    "messages": [{
        "role": "user",
        "content": "我最近怕冷，手脚冰凉"
    }],
    "user_id": "test_user_001",
}

MODERATE_CASE = {
    "messages": [{
        "role": "user",
        "content": "我最近总是失眠，心烦意乱，口干舌燥，舌红苔黄"
    }],
    "user_id": "test_user_002",
}

COMPLEX_CASE = {
    "messages": [{
        "role": "user",
        "content": """
我的症状比较复杂：
- 胸闷心悸已经2个多月了
- 晚上经常失眠，多梦易醒
- 头晕目眩，记忆力下降
- 食欲不好，大便有时偏稀
- 容易疲劳，精神不振
- 情绪焦虑，容易烦躁

我5年前查出有高血压，一直在服药控制。
舌象：舌淡红，苔薄白
脉象：脉细弦
"""
    }],
    "user_id": "test_user_003",
}


async def test_diagnose_interactive(initial_input: str, test_name: str):
    """交互式诊断测试（支持追问）"""
    print("\n" + "=" * 60)
    print(f"测试: {test_name}")
    print("=" * 60)
    
    try:
        from langchain_core.messages import HumanMessage
        from langgraph.types import Command
        
        graph = create_diagnose_graph()  # 内部会自动创建 MemorySaver
        print("✅ 诊断子图创建成功")
        
        config = {"configurable": {"thread_id": f"interactive_{test_name}"}}
        
        # 初始化输入
        initial_state = {
            "messages": [HumanMessage(content=initial_input)],
            "user_id": "test_interactive",
        }
        
        print(f"\n👤 用户: {initial_input}")
        print("\n正在分析...\n")
        
        max_rounds = 10  # 最多10轮对话
        current_input = initial_state
        
        for round_num in range(max_rounds):
            # 执行图
            result = await graph.ainvoke(current_input, config=config)
            
            # 打印执行步骤
            if result.get("steps"):
                print(f"[执行步骤]")
                for step in result["steps"][-2:]:  # 只显示最后2步
                    print(f"  • {step}")
                print()
            
            # 检查是否有 interrupt
            if "__interrupt__" in result:
                interrupts = result["__interrupt__"]
                if interrupts:
                    interrupt_data = interrupts[0].value
                    question = interrupt_data.get("question", "")
                    
                    print(f"🤖 医生追问: {question}\n")
                    
                    # 等待用户输入
                    user_answer = input("👤 您的回答 (输入'q'退出, 按回车跳过): ").strip()
                    print()  # 空行
                    
                    if user_answer.lower() == 'q':
                        print("❌ 用户退出测试\n")
                        return False
                    
                    if not user_answer:
                        print("⚠️ 跳过追问，直接进入诊断\n")
                        user_answer = "没有了"
                    
                    print(f"正在分析您的回答...\n")
                    
                    # 使用 Command(resume=...) 恢复执行
                    current_input = Command(resume=user_answer)
                    continue
            
            # 检查是否完成诊断
            if result.get("answer"):
                print("=" * 60)
                print("✅ 诊断完成！")
                print("=" * 60)
                
                answer = result.get("answer", "")
                print(f"\n{answer}\n")
                
                if result.get("complexity"):
                    complexity = result["complexity"]
                    print(f"复杂度评估:")
                    print(f"  级别: {complexity.get('level')}")
                    print(f"  得分: {complexity.get('score')}/10")
                    print(f"  理由: {complexity.get('reasoning')}\n")
                
                return True
            
            # 没有 interrupt 也没有 answer，可能还在处理
            print(f"[继续执行...]\n")
        
        print("⚠️ 达到最大轮数\n")
        return False
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_diagnose_simple():
    """测试简单诊断流程（不追问）"""
    return await test_diagnose_interactive(
        initial_input="我最近怕冷，手脚冰凉",
        test_name="简单病例（阳虚）"
    )


async def test_diagnose_moderate():
    """测试中等诊断流程（需要追问）"""
    return await test_diagnose_interactive(
        initial_input="我最近总是失眠，心烦意乱，口干舌燥",
        test_name="中等病例（阴虚火旺）"
    )


async def test_diagnose_complex():
    """测试复杂诊断流程（DeepSearch Agent，需要追问）"""
    return await test_diagnose_interactive(
        initial_input="""我的症状比较复杂：
- 胸闷心悸已经2个多月了
- 晚上经常失眠，多梦易醒
- 头晕目眩，记忆力下降
- 食欲不好，大便有时偏稀
- 容易疲劳，精神不振
- 情绪焦虑，容易烦躁

我5年前查出有高血压，一直在服药控制。""",
        test_name="复杂病例（心脾两虚 + DeepSearch）"
    )


async def test_graph_structure():
    """测试图结构"""
    print("\n" + "=" * 60)
    print("测试 0: 图结构验证")
    print("=" * 60)
    
    try:
        graph = create_diagnose_graph()
        print("✅ 诊断子图创建成功")
        
        # 检查节点
        nodes = graph.nodes
        print(f"\n节点列表:")
        for node in nodes:
            print(f"  - {node}")
        
        expected_nodes = {
            "collect_info",
            "analyze_follow_up",
            "assess_complexity",
            "simple_diagnosis",
            "moderate_diagnosis",
            "complex_diagnosis"
        }
        
        missing_nodes = expected_nodes - set(nodes.keys())
        if missing_nodes:
            print(f"\n⚠️ 缺少节点: {missing_nodes}")
        else:
            print("\n✅ 所有必要节点都已添加")
        
        return len(missing_nodes) == 0
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("诊断子图完整测试")
    print("=" * 60)
    
    results = {}
    
    # 测试 0: 图结构
    print("\n>>> 测试图结构")
    results["structure"] = asyncio.run(test_graph_structure())
    
    if not results["structure"]:
        print("\n❌ 图结构测试失败，请检查配置")
        sys.exit(1)
    
    # 测试 1: 简单诊断
    print("\n>>> 是否测试简单诊断？(y/n)")
    if input().strip().lower() == 'y':
        results["simple"] = asyncio.run(test_diagnose_simple())
    
    # 测试 2: 中等诊断
    print("\n>>> 是否测试中等诊断？(y/n)")
    if input().strip().lower() == 'y':
        results["moderate"] = asyncio.run(test_diagnose_moderate())
    
    # 测试 3: 复杂诊断（DeepSearch Agent）
    print("\n>>> 是否测试复杂诊断（DeepSearch Agent，较慢）？(y/n)")
    if input().strip().lower() == 'y':
        results["complex"] = asyncio.run(test_diagnose_complex())
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for test_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n✅ 所有测试通过！诊断子图集成成功！")
    else:
        print("\n⚠️ 部分测试失败")


if __name__ == "__main__":
    main()
