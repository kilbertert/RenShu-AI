"""
DeepSearch Agent 测试脚本

测试 DeepSearch Agent 的基本功能：
1. Agent 创建
2. 中间件栈初始化
3. 子 Agent 注册
4. 工具调用
5. 简单诊断流程

运行方式：
    python test_deep_search_agent.py
    或
    python -m pytest test_deep_search_agent.py -v
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.src.agent.components.diagnose.nodes.complex.deep_search_agent import (
    create_deep_search_agent,
    run_deep_search_diagnosis,
)
from app.src.utils import get_logger

logger = get_logger("test_deep_search")


# ============================================================
# 测试数据
# ============================================================

SIMPLE_PATIENT_INFO = {
    "主诉": "怕冷，手脚冰凉",
    "现病史": "近 3 个月出现怕冷症状，尤其是手脚冰凉，冬天更明显",
    "症状": {
        "寒热": "怕冷",
        "汗出": "少汗",
        "精神": "疲乏",
        "睡眠": "易醒",
        "二便": "小便清长"
    },
    "舌象": "舌淡苔白",
    "脉象": "脉沉细"
}

COMPLEX_PATIENT_INFO = {
    "主诉": "胸闷心悸，失眠多梦 2 月余",
    "现病史": "患者 2 个月前因工作压力大出现胸闷心悸，夜间难以入睡，多梦易醒，伴有头晕目眩",
    "症状": {
        "胸闷": "时有胸闷气短",
        "心悸": "心跳加快，尤其夜间明显",
        "失眠": "入睡困难，易醒多梦",
        "头晕": "头晕目眩",
        "精神": "疲乏无力，健忘",
        "情绪": "焦虑烦躁",
        "食欲": "纳差",
        "大便": "便溏"
    },
    "既往史": "有高血压病史 5 年，血压控制尚可",
    "舌象": "舌淡红，苔薄白",
    "脉象": "脉细弦"
}


# ============================================================
# 测试函数
# ============================================================

async def test_agent_creation():
    """测试 1：Agent 创建"""
    logger.info("=" * 60)
    logger.info("测试 1：DeepSearch Agent 创建")
    logger.info("=" * 60)
    
    try:
        agent = create_deep_search_agent(enable_doctor_approval=False)
        logger.info("✅ Agent 创建成功")
        logger.info(f"Agent 类型: {type(agent)}")
        
        # 检查 Agent 属性
        if hasattr(agent, 'graph'):
            logger.info(f"✅ Graph 存在: {type(agent.graph)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Agent 创建失败: {e}", exc_info=True)
        return False


async def test_simple_diagnosis():
    """测试 2：简单诊断（阳虚）"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 2：简单诊断流程（阳虚证）")
    logger.info("=" * 60)
    
    try:
        result = await run_deep_search_diagnosis(
            collected_info=SIMPLE_PATIENT_INFO,
            thread_id="test_simple_001",
            enable_doctor_approval=False
        )
        
        logger.info("✅ 诊断完成")
        logger.info(f"Thread ID: {result.get('thread_id')}")
        logger.info(f"步骤: {result.get('steps')}")
        
        if result.get('answer'):
            logger.info(f"\n诊断结果（前 500 字符）:\n{result['answer'][:500]}...")
        
        return True
    except Exception as e:
        logger.error(f"❌ 简单诊断失败: {e}", exc_info=True)
        return False


async def test_complex_diagnosis():
    """测试 3：复杂诊断（心脾两虚）"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 3：复杂诊断流程（心脾两虚）")
    logger.info("=" * 60)
    
    try:
        result = await run_deep_search_diagnosis(
            collected_info=COMPLEX_PATIENT_INFO,
            thread_id="test_complex_001",
            enable_doctor_approval=False
        )
        
        logger.info("✅ 复杂诊断完成")
        logger.info(f"Thread ID: {result.get('thread_id')}")
        logger.info(f"步骤: {result.get('steps')}")
        
        if result.get('answer'):
            logger.info(f"\n诊断结果（前 500 字符）:\n{result['answer'][:500]}...")
        
        return True
    except Exception as e:
        logger.error(f"❌ 复杂诊断失败: {e}", exc_info=True)
        return False


async def test_agent_with_custom_prompt():
    """测试 4：自定义提示词测试"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 4：自定义提示词")
    logger.info("=" * 60)
    
    try:
        agent = create_deep_search_agent(enable_doctor_approval=False)
        
        # 自定义简单问题
        custom_input = {
            "messages": [{
                "role": "user",
                "content": "请简单介绍一下中医辨证的基本流程"
            }]
        }
        
        config = {"configurable": {"thread_id": "test_custom_001"}}
        result = await agent.ainvoke(custom_input, config=config)
        
        logger.info("✅ 自定义提示词测试完成")
        if result.get("messages"):
            response = result["messages"][-1].content
            logger.info(f"\nAgent 回复（前 300 字符）:\n{response[:300]}...")
        
        return True
    except Exception as e:
        logger.error(f"❌ 自定义提示词测试失败: {e}", exc_info=True)
        return False


async def test_middleware_stack():
    """测试 5：中间件栈验证"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 5：中间件栈验证")
    logger.info("=" * 60)
    
    try:
        agent = create_deep_search_agent(enable_doctor_approval=False)
        
        # 检查中间件
        logger.info("✅ 中间件栈初始化成功")
        
        # 尝试获取中间件信息（如果 API 支持）
        if hasattr(agent, 'middleware'):
            logger.info(f"中间件数量: {len(agent.middleware)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ 中间件栈验证失败: {e}", exc_info=True)
        return False


async def test_subagent_registration():
    """测试 6：子 Agent 注册验证"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 6：子 Agent 注册")
    logger.info("=" * 60)
    
    try:
        from app.src.agent.components.diagnose.nodes.complex.subagents import (
            create_differential_expert,
            create_treatment_expert,
            create_prescription_expert,
            create_prognosis_expert,
            create_verification_expert,
        )
        from app.src.agent import get_llm
        
        llm = get_llm()
        
        # 创建所有子 Agent
        experts = {
            "differential": create_differential_expert(llm),
            "treatment": create_treatment_expert(llm),
            "prescription": create_prescription_expert(llm),
            "prognosis": create_prognosis_expert(llm),
            "verification": create_verification_expert(llm),
        }
        
        logger.info(f"✅ 成功创建 {len(experts)} 个子 Agent:")
        for name, expert in experts.items():
            logger.info(f"  - {name}: {type(expert)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ 子 Agent 注册失败: {e}", exc_info=True)
        return False


async def test_tools_registration():
    """测试 7：工具注册验证"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 7：工具注册")
    logger.info("=" * 60)
    
    try:
        from app.src.agent.components.diagnose.nodes.complex.tools import (
            kg_syndrome_search,
            case_vector_search,
            classics_search,
            web_search,
        )
        
        tools = {
            "kg_syndrome_search": kg_syndrome_search,
            "case_vector_search": case_vector_search,
            "classics_search": classics_search,
            "web_search": web_search,
        }
        
        logger.info(f"✅ 成功注册 {len(tools)} 个工具:")
        for name, tool in tools.items():
            logger.info(f"  - {name}: {type(tool)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ 工具注册失败: {e}", exc_info=True)
        return False


# ============================================================
# 主测试流程
# ============================================================

async def run_all_tests():
    """运行所有测试"""
    logger.info("\n" + "=" * 60)
    logger.info("开始 DeepSearch Agent 完整测试")
    logger.info("=" * 60 + "\n")
    
    results = {}
    
    # 测试 1: Agent 创建
    results["agent_creation"] = await test_agent_creation()
    
    # 测试 2: 工具注册
    results["tools_registration"] = await test_tools_registration()
    
    # 测试 3: 子 Agent 注册
    results["subagent_registration"] = await test_subagent_registration()
    
    # 测试 4: 中间件栈
    results["middleware_stack"] = await test_middleware_stack()
    
    # 测试 5: 自定义提示词
    results["custom_prompt"] = await test_agent_with_custom_prompt()
    
    # 测试 6: 简单诊断
    results["simple_diagnosis"] = await test_simple_diagnosis()
    
    # 测试 7: 复杂诊断（可选，较慢）
    # results["complex_diagnosis"] = await test_complex_diagnosis()
    
    # 汇总结果
    logger.info("\n" + "=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"总计: {passed}/{total} 通过")
    logger.info("=" * 60)
    
    return passed == total


async def quick_test():
    """快速测试（只测试 Agent 创建）"""
    logger.info("运行快速测试...")
    
    success = await test_agent_creation()
    
    if success:
        logger.info("\n✅ 快速测试通过！DeepSearch Agent 可以正常创建。")
    else:
        logger.error("\n❌ 快速测试失败！请检查依赖和配置。")
    
    return success


# ============================================================
# 命令行入口
# ============================================================

def main():
    """主入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepSearch Agent 测试脚本")
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="测试模式：quick（快速测试）或 full（完整测试）"
    )
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        success = asyncio.run(quick_test())
    else:
        success = asyncio.run(run_all_tests())
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
