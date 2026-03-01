"""
测试豆包 (Doubao) Thinking Mode

验证三种调用方式：
1. 同步流式 (stream)
2. 异步流式 (astream)
3. 非流式 (invoke) - 使用父类实现
"""

import sys
import asyncio
from pathlib import Path

# 添加项目根目录到 Python 路径
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.src.core.language_model.llm_provider import get_langchain_llm
from langchain_core.messages import HumanMessage

# 测试配置 - 请替换为你的 API Key
API_KEY = "sk-Tz0aHC6HJoHHfKHeFHY2VICuawHRP1QteP0Hy1j0pAbOOFLn"
BASE_URL = "https://www.dmxapi.cn/v1"

def test_doubao_sync_stream():
    """测试 1: 同步流式 (stream) - 启用 thinking mode"""
    
    print("=" * 80)
    print("测试 1: 豆包 同步流式 (stream) - 启用 thinking mode")
    print("=" * 80)
    
    llm = get_langchain_llm(
        provider_name="doubao",
        model="doubao-seed-1-6-251015",
        api_key=API_KEY,
        base_url=BASE_URL,
        enable_thinking=True,
        streaming=True,
    )
    
    messages = [HumanMessage(content="9.11 和 9.8 哪个更大？")]
    
    print("\n📝 问题: 9.11 和 9.8 哪个更大？\n")
    
    thinking_content = ""
    answer_content = ""
    thinking_started = False
    answer_started = False
    
    try:
        for chunk in llm.stream(input=messages):
            # 提取思考内容
            if hasattr(chunk, 'additional_kwargs') and 'reasoning_content' in chunk.additional_kwargs:
                thinking = chunk.additional_kwargs['reasoning_content']
                if thinking:
                    if not thinking_started:
                        print("🧠 思考过程：")
                        thinking_started = True
                    thinking_content += thinking
                    print(thinking, end="", flush=True)
            
            # 提取答案内容
            if hasattr(chunk, 'content') and chunk.content:
                if not answer_started:
                    print("\n\n💬 回答内容：")
                    answer_started = True
                answer_content += chunk.content
                print(chunk.content, end="", flush=True)
        
        print("\n")
        print("=" * 80)
        print("✅ 同步流式测试成功！")
        print(f"思考内容长度: {len(thinking_content)} 字符")
        print(f"答案内容长度: {len(answer_content)} 字符")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 同步流式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_doubao_async_stream():
    """测试 2: 异步流式 (astream) - 启用 thinking mode"""
    
    print("\n" + "=" * 80)
    print("测试 2: 豆包 异步流式 (astream) - 启用 thinking mode")
    print("=" * 80)
    
    llm = get_langchain_llm(
        provider_name="doubao",
        model="doubao-seed-1-6-251015",
        api_key=API_KEY,
        base_url=BASE_URL,
        enable_thinking=True,
        streaming=True,
    )
    
    messages = [HumanMessage(content="9.11 和 9.8 哪个更大？")]
    
    print("\n📝 问题: 9.11 和 9.8 哪个更大？\n")
    
    thinking_content = ""
    answer_content = ""
    thinking_started = False
    answer_started = False
    
    try:
        async for chunk in llm.astream(input=messages):
            # 提取思考内容
            if hasattr(chunk, 'additional_kwargs') and 'reasoning_content' in chunk.additional_kwargs:
                thinking = chunk.additional_kwargs['reasoning_content']
                if thinking:
                    if not thinking_started:
                        print("🧠 思考过程：")
                        thinking_started = True
                    thinking_content += thinking
                    print(thinking, end="", flush=True)
            
            # 提取答案内容
            if hasattr(chunk, 'content') and chunk.content:
                if not answer_started:
                    print("\n\n💬 回答内容：")
                    answer_started = True
                answer_content += chunk.content
                print(chunk.content, end="", flush=True)
        
        print("\n")
        print("=" * 80)
        print("✅ 异步流式测试成功！")
        print(f"思考内容长度: {len(thinking_content)} 字符")
        print(f"答案内容长度: {len(answer_content)} 字符")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 异步流式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_doubao_invoke():
    """测试 3: 非流式 (invoke) - 启用 thinking mode"""
    
    print("\n" + "=" * 80)
    print("测试 3: 豆包 非流式 (invoke) - 启用 thinking mode")
    print("=" * 80)
    
    llm = get_langchain_llm(
        provider_name="doubao",
        model="doubao-seed-1-6-251015",
        api_key=API_KEY,
        base_url=BASE_URL,
        enable_thinking=True,  # 启用 thinking mode
        streaming=False,
    )
    
    messages = [HumanMessage(content="9.11 和 9.8 哪个更大？")]
    
    print("\n📝 问题: 9.11 和 9.8 哪个更大？\n")
    
    try:
        response = llm.invoke(input=messages)
        
        # 提取思考内容
        thinking_content = response.additional_kwargs.get('reasoning_content', '')
        answer_content = response.content
        
        if thinking_content:
            print("🧠 思考过程：")
            print(thinking_content)
        else:
            print("⚠️  未找到思考内容")
        
        print("\n💬 回答内容：")
        print(answer_content)
        
        print("\n" + "=" * 80)
        print("✅ 非流式测试成功！")
        print(f"思考内容长度: {len(thinking_content)} 字符")
        print(f"答案内容长度: {len(answer_content)} 字符")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 非流式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_doubao_ainvoke():
    """测试 4: 异步非流式 (ainvoke) - 启用 thinking mode"""
    
    print("\n" + "=" * 80)
    print("测试 4: 豆包 异步非流式 (ainvoke) - 启用 thinking mode")
    print("=" * 80)
    
    llm = get_langchain_llm(
        provider_name="doubao",
        model="doubao-seed-1-6-251015",
        api_key=API_KEY,
        base_url=BASE_URL,
        enable_thinking=True,  # 启用 thinking mode
        streaming=False,
    )
    
    messages = [HumanMessage(content="9.11 和 9.8 哪个更大？")]
    
    print("\n📝 问题: 9.11 和 9.8 哪个更大？\n")
    
    try:
        response = await llm.ainvoke(input=messages)
        
        # 提取思考内容
        thinking_content = response.additional_kwargs.get('reasoning_content', '')
        answer_content = response.content
        
        if thinking_content:
            print("🧠 思考过程：")
            print(thinking_content)
        else:
            print("⚠️  未找到思考内容")
        
        print("\n💬 回答内容：")
        print(answer_content)
        
        print("\n" + "=" * 80)
        print("✅ 异步非流式测试成功！")
        print(f"思考内容长度: {len(thinking_content)} 字符")
        print(f"答案内容长度: {len(answer_content)} 字符")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 异步非流式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """运行所有测试"""
    
    print("\n⚠️ 请先在脚本中设置你的豆包 API Key！\n")
    
    if API_KEY == "your-doubao-api-key":
        print("❌ 请先替换 API_KEY 变量为实际的 API Key")
        return
    
    results = {}
    
    # 测试 1: 同步流式
    results["同步流式 (stream)"] = test_doubao_sync_stream()
    
    # 测试 2: 异步流式
    results["异步流式 (astream)"] = await test_doubao_async_stream()
    
    # 测试 3: 非流式
    results["非流式 (invoke)"] = test_doubao_invoke()
    
    # 测试 4: 异步非流式
    results["异步非流式 (ainvoke)"] = await test_doubao_ainvoke()
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    print("=" * 80)
    
    # 检查是否全部通过
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 所有测试通过！豆包 thinking mode 集成成功！")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
