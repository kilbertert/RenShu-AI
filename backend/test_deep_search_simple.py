"""
DeepSearch Agent 简单测试脚本

快速验证 DeepSearch Agent 是否能够正常运行

运行方式：
    cd backend
    python test_deep_search_simple.py
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("正在导入模块...")

try:
    # 先导入 get_llm，避免循环导入
    from app.src.agent.tcm_builder import get_llm
    print("✅ get_llm 导入成功")
except Exception as e:
    print(f"❌ get_llm 导入失败: {e}")
    sys.exit(1)

try:
    from app.src.agent.components.diagnose.nodes.complex.deep_search_agent import (
        create_deep_search_agent,
    )
    print("✅ deep_search_agent 模块导入成功")
except Exception as e:
    print(f"❌ deep_search_agent 模块导入失败: {e}")
    sys.exit(1)


async def test_basic():
    """基础测试：创建 Agent"""
    print("\n" + "=" * 60)
    print("测试：创建 DeepSearch Agent")
    print("=" * 60)
    
    try:
        print("正在创建 Agent...")
        agent = create_deep_search_agent(enable_doctor_approval=False)
        print(f"✅ Agent 创建成功！类型: {type(agent)}")
        
        # 测试简单调用
        print("\n正在测试 Agent 调用...")
        test_input = {
            "messages": [{
                "role": "user",
                "content": "你好，请介绍一下你的功能"
            }]
        }
        
        config = {"configurable": {"thread_id": "test_001"}}
        result = await agent.ainvoke(test_input, config=config)
        
        print("✅ Agent 调用成功！")
        if result.get("messages"):
            response = result["messages"][-1].content
            print(f"\nAgent 回复:\n{response[:200]}...\n")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_diagnosis():
    """诊断测试：简单病例"""
    print("\n" + "=" * 60)
    print("测试：简单病例诊断")
    print("=" * 60)
    
    try:
        agent = create_deep_search_agent(enable_doctor_approval=False)
        
        # 简单病例
        test_input = {
            "messages": [{
                "role": "user",
                "content": """
请分析以下患者信息：

主诉：怕冷，手脚冰凉
症状：怕冷明显，手脚冰凉，疲乏，小便清长
舌象：舌淡苔白
脉象：脉沉细

请给出辨证分析。
"""
            }]
        }
        
        config = {"configurable": {"thread_id": "test_diagnosis_001"}}
        
        print("正在执行辨证分析...")
        result = await agent.ainvoke(test_input, config=config)
        
        print("✅ 辨证分析完成！")
        if result.get("messages"):
            response = result["messages"][-1].content
            print(f"\n辨证结果:\n{response[:500]}...\n")
        
        return True
        
    except Exception as e:
        print(f"❌ 诊断测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("DeepSearch Agent 简单测试")
    print("=" * 60)
    
    # 测试 1：基础功能
    success1 = asyncio.run(test_basic())
    
    if not success1:
        print("\n❌ 基础测试失败，请检查配置")
        sys.exit(1)
    
    # 测试 2：诊断功能（可选）
    print("\n是否继续测试诊断功能？(y/n)")
    choice = input().strip().lower()
    
    if choice == 'y':
        success2 = asyncio.run(test_diagnosis())
        if success2:
            print("\n✅ 所有测试通过！")
        else:
            print("\n⚠️ 诊断测试失败，但基础功能正常")
    else:
        print("\n✅ 基础测试通过！")


if __name__ == "__main__":
    main()
