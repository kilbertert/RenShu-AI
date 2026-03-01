"""
TCM Builder 调试脚本

快速调试 TCM Multi-Agent Builder，支持：
1. 单次查询调试
2. 多轮对话调试
3. 流式输出调试
4. 中间件调试
5. 交互式对话

运行方式：
    python test_tcm_builder.py
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from langchain_core.messages import HumanMessage, AIMessage
from app.src.agent.tcm_builder import (
    build_tcm_graph,
    get_middleware_chain,
    new_thread_id,
)
from app.src.agent.tcm_states import LLMConfig
from app.src.utils import get_logger

logger = get_logger("tcm_builder_debug")




# ============================================================
# 调试场景示例
# ============================================================

async def debug_single_query():
    """场景1：单次查询调试"""
    print("\n" + "="*80)
    print("场景1：单次查询调试")
    print("="*80)
    
    # 构建图
    graph = build_tcm_graph()
    print("✅ 图构建完成\n")
    
    # 准备查询
    query = "我想调理体质，最近总是感觉疲劳"
    print(f"用户查询: {query}\n")
    
    # 执行
    input_state = {
        "messages": [HumanMessage(content=query)],
        "user_id": "debug_user",
        "conversation_id": "debug_conv_001",
    }
    
    config = {"configurable": {"thread_id": new_thread_id()}}
    
    print("开始执行图...\n")
    result = await graph.ainvoke(input_state, config=config)
    
    # 输出结果
    print("-" * 80)
    print("执行结果:")
    print("-" * 80)
    
    if result.get("router"):
        router = result["router"]
        print(f"路由类型: {router.query_type}")
        print(f"置信度: {router.confidence:.2f}")
        print(f"推理: {router.reasoning}")
    
    if result.get("steps"):
        print(f"\n执行步骤:")
        for step in result["steps"]:
            print(f"  - {step}")
    
    if result.get("answer"):
        print(f"\n回答:\n{result['answer']}")
    
    print("\n" + "="*80)


async def debug_streaming():
    """场景2：流式输出调试"""
    print("\n" + "="*80)
    print("场景2：流式输出调试")
    print("="*80)
    
    graph = build_tcm_graph()
    print("✅ 图构建完成\n")
    
    query = "头疼怎么办？"
    print(f"用户查询: {query}\n")
    
    input_state = {
        "messages": [HumanMessage(content=query)],
        "user_id": "debug_user",
        "conversation_id": "debug_stream_001",
    }
    
    config = {"configurable": {"thread_id": new_thread_id()}}
    
    print("流式输出（节点执行顺序）:\n")
    
    async for chunk in graph.astream(input_state, config=config):
        for node_name, node_output in chunk.items():
            print(f"  [节点] {node_name}")
            
            # 如果有answer，显示部分内容
            if isinstance(node_output, dict) and "answer" in node_output:
                answer_preview = node_output["answer"][:100]
                print(f"         回答预览: {answer_preview}...")
    
    print("\n✅ 流式输出完成")
    print("="*80)


async def debug_multi_turn():
    """场景3：多轮对话调试"""
    print("\n" + "="*80)
    print("场景3：多轮对话调试")
    print("="*80)
    
    graph = build_tcm_graph()
    print("✅ 图构建完成\n")
    
    # 模拟多轮对话
    conversation = [
        "你好",
        "我最近失眠，睡不着",
        "对，还有心悸的症状",
        "舌苔是薄白色的",
    ]
    
    thread_id = new_thread_id()
    config = {"configurable": {"thread_id": thread_id}}
    messages = []
    
    for turn, query in enumerate(conversation, 1):
        print(f"\n--- 第 {turn} 轮 ---")
        print(f"用户: {query}")
        
        messages.append(HumanMessage(content=query))
        
        input_state = {
            "messages": messages.copy(),
            "user_id": "debug_user",
            "conversation_id": f"debug_multiturn_{thread_id}",
        }
        
        result = await graph.ainvoke(input_state, config=config)
        
        answer = result.get("answer", "")
        print(f"助手: {answer[:200]}...")
        
        if answer:
            messages.append(AIMessage(content=answer))
    
    print("\n✅ 多轮对话完成")
    print("="*80)


async def debug_middleware():
    """场景4：中间件调试"""
    print("\n" + "="*80)
    print("场景4：中间件调试")
    print("="*80)
    
    # 获取中间件链
    middleware_chain = get_middleware_chain()
    middlewares = middleware_chain.middlewares
    
    print(f"中间件总数: {len(middlewares)}\n")
    print("中间件列表（按优先级）:\n")
    
    for idx, mw in enumerate(middlewares, 1):
        status = "✅" if mw.enabled else "❌"
        print(f"{idx:2d}. {status} [P{mw.priority:2d}] {mw.name}")
    
    print("\n" + "="*80)
    
    # 测试拦截
    print("\n测试紧急情况拦截:")
    print("-" * 80)
    
    graph = build_tcm_graph()
    
    query = "我突然剧烈胸痛，喘不上气"
    print(f"查询: {query}\n")
    
    input_state = {
        "messages": [HumanMessage(content=query)],
        "user_id": "debug_user",
        "conversation_id": "debug_emergency",
    }
    
    config = {"configurable": {"thread_id": new_thread_id()}}
    result = await graph.ainvoke(input_state, config=config)
    
    answer = result.get("answer", "")
    print(f"回答:\n{answer}\n")
    
    if "紧急" in answer or "120" in answer:
        print("✅ 紧急情况正确拦截")
    else:
        print("⚠️ 未触发紧急拦截")
    
    print("="*80)


async def debug_interactive():
    """场景5：交互式调试"""
    print("\n" + "="*80)
    print("场景5：交互式调试")
    print("="*80)
    
    graph = build_tcm_graph()
    print("✅ 图构建完成")
    print("\n输入查询进行调试（输入 'quit' 退出，'clear' 清空历史）:\n")
    
    thread_id = new_thread_id()
    config = {"configurable": {"thread_id": thread_id}}
    messages = []
    
    while True:
        try:
            query = input("\n👤 用户: ").strip()
            
            if query.lower() in ["quit", "exit", "q"]:
                print("\n退出交互模式")
                break
            
            if query.lower() == "clear":
                messages = []
                thread_id = new_thread_id()
                config = {"configurable": {"thread_id": thread_id}}
                print("✅ 历史已清空")
                continue
            
            if not query:
                continue
            
            messages.append(HumanMessage(content=query))
            
            input_state = {
                "messages": messages.copy(),
                "user_id": "debug_user",
                "conversation_id": f"interactive_{thread_id}",
            }
            
            print("⏳ 处理中...\n")
            result = await graph.ainvoke(input_state, config=config)
            
            # 显示路由信息
            if result.get("router"):
                router = result["router"]
                print(f"📍 [路由: {router.query_type}] [置信度: {router.confidence:.2f}]")
            
            # 显示回答
            answer = result.get("answer", "")
            print(f"\n🤖 助手: {answer}")
            
            if answer:
                messages.append(AIMessage(content=answer))
        
        except KeyboardInterrupt:
            print("\n\n中断，退出交互模式")
            break
        except Exception as e:
            print(f"\n❌ 执行出错: {e}")
            import traceback
            traceback.print_exc()


# ============================================================
# 主菜单
# ============================================================

async def main_menu():
    """主菜单"""
    print("\n" + "="*80)
    print("TCM Builder 调试工具")
    print("="*80)
    print("\n请选择调试场景:\n")
    print("  1. 单次查询调试")
    print("  2. 流式输出调试")
    print("  3. 多轮对话调试")
    print("  4. 中间件调试")
    print("  5. 交互式调试")
    print("  0. 退出")
    print("\n" + "="*80)
    
    choice = input("\n请输入选项 (0-5): ").strip()
    
    if choice == "1":
        await debug_single_query()
    elif choice == "2":
        await debug_streaming()
    elif choice == "3":
        await debug_multi_turn()
    elif choice == "4":
        await debug_middleware()
    elif choice == "5":
        await debug_interactive()
    elif choice == "0":
        print("\n退出程序")
        return False
    else:
        print("\n❌ 无效选项，请重新选择")
    
    return True


async def run():
    """运行主循环"""
    while True:
        continue_flag = await main_menu()
        if not continue_flag:
            break
        
        # 询问是否继续
        print("\n")
        again = input("是否继续调试? (y/n): ").strip().lower()
        if again not in ["y", "yes", ""]:
            print("\n退出程序")
            break


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n\n程序已终止")
