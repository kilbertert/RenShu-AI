"""
测试 Qwen Thinking Mode - 标准格式
"""

import os
import sys
import asyncio

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入 ChatQwen
try:
    from app.src.core.language_model.langchain_qwen import Chat as ChatQwen
    HAS_CHAT_QWEN = True
except ImportError:
    HAS_CHAT_QWEN = False
    print("Warning: ChatQwen not found")
    print("Please install: pip install langchain-qwq")
    sys.exit(1)

# 直接使用提供的 API Key 和 Base URL
QWEN_API_KEY = "sk-Tz0aHC6HJoHHfKHeFHY2VICuawHRP1QteP0Hy1j0pAbOOFLn"
QWEN_BASE_URL = "https://www.dmxapi.cn/v1"

print(f"Qwen API Key: {QWEN_API_KEY[:10]}...")
print(f"Qwen Base URL: {QWEN_BASE_URL}")
print("=" * 80)

# 测试消息
messages = [{"role": "user", "content": "9.11 和 9.8 哪个更大？"}]

# ============================================================================
# 方式 1: LangChain ChatQwen 流式调用
# ============================================================================
print("\n【方式 1】LangChain ChatQwen - stream 流式调用")
print("=" * 80)

llm_stream = ChatQwen(
    model="qwen3-max-2026-01-23",
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    streaming=True,
    extra_body={"enable_thinking": True},  # 启用思考模式
)

try:
    print("\n【详细 Chunk 分析】")
    print("-" * 80)
    
    chunk_count = 0
    thinking_content = ""
    answer_content = ""
    
    for chunk in llm_stream.stream(input=messages):
        chunk_count += 1
        
        # 打印前3个 chunk 的完整结构
        if chunk_count <= 3:
            print(f"\n--- Chunk {chunk_count} ---")
            print(f"Type: {type(chunk)}")
            print(f"Content: {repr(chunk.content[:50]) if chunk.content else 'None'}")
            print(f"Additional kwargs keys: {list(chunk.additional_kwargs.keys())}")
            
            # 检查 reasoning_content
            if 'reasoning_content' in chunk.additional_kwargs:
                reasoning = chunk.additional_kwargs['reasoning_content']
                print(f"Reasoning content: {repr(reasoning[:50])}...")
        
        # 收集内容 - 标准格式
        if chunk.content:
            answer_content += chunk.content
        
        # 检查 additional_kwargs 中的 reasoning_content
        if 'reasoning_content' in chunk.additional_kwargs:
            thinking_content += chunk.additional_kwargs.get('reasoning_content', '')
    
    print(f"\n总共收到 {chunk_count} 个 chunks")
    print("-" * 80)
    
    if thinking_content:
        print("\n" + "=" * 20 + " 思考过程 " + "=" * 20)
        print(thinking_content[:300] + "..." if len(thinking_content) > 300 else thinking_content)
    else:
        print("\n⚠️  未找到思考内容")
    
    print("\n" + "=" * 20 + " 最终回复 " + "=" * 20)
    print(answer_content[:300] + "..." if len(answer_content) > 300 else answer_content)
    
    print("\n✅ LangChain ChatQwen 流式调用成功\n")
    
except Exception as e:
    print(f"❌ LangChain ChatQwen 流式调用失败: {e}\n")
    import traceback
    traceback.print_exc()


# ============================================================================
# 方式 2: LangChain ChatQwen ainvoke 调用（非流式异步）
# ============================================================================
print("\n【方式 2】LangChain ChatQwen - ainvoke 非流式异步调用")
print("=" * 80)

async def test_ainvoke():
    """测试 ainvoke 方法的推理输出"""
    llm_ainvoke = ChatQwen(
        model="qwen3-max-2026-01-23",
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
        streaming=False,  # 非流式
        extra_body={"enable_thinking": True},
    )
    
    try:
        print("\n调用 ainvoke 方法...")
        response = await llm_ainvoke.ainvoke(input=messages)
        
        print("\n【Response 结构分析】")
        print("-" * 80)
        print(f"Type: {type(response)}")
        print(f"Content length: {len(response.content) if response.content else 0}")
        print(f"Additional kwargs keys: {list(response.additional_kwargs.keys())}")
        
        # 提取推理内容 - 标准格式
        thinking_content = response.additional_kwargs.get('reasoning_content', '')
        answer_content = response.content
        
        if thinking_content:
            print("\n" + "=" * 20 + " 思考过程 " + "=" * 20)
            print(thinking_content[:300] + "..." if len(thinking_content) > 300 else thinking_content)
        else:
            print("\n⚠️  未找到思考内容")
        
        print("\n" + "=" * 20 + " 最终回复 " + "=" * 20)
        print(answer_content[:300] + "..." if len(answer_content) > 300 else answer_content)
        
        print("\n✅ LangChain ChatQwen ainvoke 调用成功\n")
        
    except Exception as e:
        print(f"❌ LangChain ChatQwen ainvoke 调用失败: {e}\n")
        import traceback
        traceback.print_exc()

# 运行异步测试
asyncio.run(test_ainvoke())


# ============================================================================
# 方式 3: LangChain ChatQwen invoke 调用（非流式同步）
# ============================================================================
print("\n【方式 3】LangChain ChatQwen - invoke 非流式同步调用")
print("=" * 80)

llm_invoke = ChatQwen(
    model="qwen3-max-2026-01-23",
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    streaming=False,
    extra_body={"enable_thinking": True},
)

try:
    print("\n调用 invoke 方法...")
    response = llm_invoke.invoke(input=messages)
    
    print("\n【Response 结构分析】")
    print("-" * 80)
    print(f"Type: {type(response)}")
    print(f"Content length: {len(response.content) if response.content else 0}")
    print(f"Additional kwargs keys: {list(response.additional_kwargs.keys())}")
    
    # 提取推理内容 - 标准格式
    thinking_content = response.additional_kwargs.get('reasoning_content', '')
    answer_content = response.content
    
    if thinking_content:
        print("\n" + "=" * 20 + " 思考过程 " + "=" * 20)
        print(thinking_content[:300] + "..." if len(thinking_content) > 300 else thinking_content)
    else:
        print("\n⚠️  未找到思考内容")
    
    print("\n" + "=" * 20 + " 最终回复 " + "=" * 20)
    print(answer_content[:300] + "..." if len(answer_content) > 300 else answer_content)
    
    print("\n✅ LangChain ChatQwen invoke 调用成功\n")
    
except Exception as e:
    print(f"❌ LangChain ChatQwen invoke 调用失败: {e}\n")
    import traceback
    traceback.print_exc()


print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
