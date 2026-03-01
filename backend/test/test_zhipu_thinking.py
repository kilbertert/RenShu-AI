"""
测试智谱 GLM Thinking Mode - 使用原生 OpenAI SDK
"""

import os
import sys
import asyncio

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入原生 OpenAI SDK
try:
    from openai import OpenAI, AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: OpenAI SDK not found")
    print("Please install: pip install openai")
    sys.exit(1)

# 直接使用提供的 API Key 和 Base URL
ZHIPU_API_KEY = "sk-Tz0aHC6HJoHHfKHeFHY2VICuawHRP1QteP0Hy1j0pAbOOFLn"
ZHIPU_BASE_URL = "https://www.dmxapi.cn/v1"

print(f"智谱 GLM API Key: {ZHIPU_API_KEY[:10]}...")
print(f"智谱 GLM Base URL: {ZHIPU_BASE_URL}")
print("=" * 80)

# 测试消息
messages = [{"role": "user", "content": "9.11 和 9.8 哪个更大？"}]

# ============================================================================
# 方式 1: 原生 OpenAI SDK 流式调用
# ============================================================================
print("\n【方式 1】原生 OpenAI SDK - stream 流式调用")
print("=" * 80)

client = OpenAI(
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL,
)

try:
    print("\n【详细 Chunk 分析】")
    print("-" * 80)
    
    response = client.chat.completions.create(
        model="glm-4.7",
        messages=messages,
        extra_body={"thinking": {"type": "enabled"}},  # 启用思考模式
        stream=True,
    )
    
    chunk_count = 0
    thinking_content = ""
    answer_content = ""
    
    for chunk in response:
        chunk_count += 1
        
        # 检查 choices 是否为空
        if not chunk.choices:
            continue
        
        # 获取 delta
        delta = chunk.choices[0].delta
        
        # 打印前3个 chunk 的完整结构
        if chunk_count <= 3:
            print(f"\n--- Chunk {chunk_count} ---")
            print(f"Type: {type(chunk)}")
            print(f"Delta: {delta}")
            
            # 检查 reasoning_content
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                print(f"Reasoning content: {repr(delta.reasoning_content[:50])}...")
            
            # 检查 content
            if hasattr(delta, 'content') and delta.content:
                print(f"Content: {repr(delta.content[:50])}...")
        
        # 收集思考内容
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
            thinking_content += delta.reasoning_content
        
        # 收集答案内容
        if hasattr(delta, 'content') and delta.content:
            answer_content += delta.content
    
    print(f"\n总共收到 {chunk_count} 个 chunks")
    print("-" * 80)
    
    if thinking_content:
        print("\n" + "=" * 20 + " 思考过程 " + "=" * 20)
        print(thinking_content[:300] + "..." if len(thinking_content) > 300 else thinking_content)
    else:
        print("\n⚠️  未找到思考内容")
    
    print("\n" + "=" * 20 + " 最终回复 " + "=" * 20)
    print(answer_content[:300] + "..." if len(answer_content) > 300 else answer_content)
    
    print("\n✅ 原生 OpenAI SDK 流式调用成功\n")
    
except Exception as e:
    print(f"❌ 原生 OpenAI SDK 流式调用失败: {e}\n")
    import traceback
    traceback.print_exc()


# ============================================================================
# 方式 2: 原生 OpenAI SDK 异步流式调用
# ============================================================================
print("\n【方式 2】原生 OpenAI SDK - astream 异步流式调用")
print("=" * 80)

async def test_astream():
    """测试异步流式调用"""
    async_client = AsyncOpenAI(
        api_key=ZHIPU_API_KEY,
        base_url=ZHIPU_BASE_URL,
    )
    
    try:
        print("\n调用异步流式方法...")
        
        response = await async_client.chat.completions.create(
            model="glm-4.7",
            messages=messages,
            extra_body={"thinking": {"type": "enabled"}},
            stream=True,
        )
        
        chunk_count = 0
        thinking_content = ""
        answer_content = ""
        
        async for chunk in response:
            chunk_count += 1
            
            # 检查 choices 是否为空
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta
            
            # 收集思考内容
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                thinking_content += delta.reasoning_content
            
            # 收集答案内容
            if hasattr(delta, 'content') and delta.content:
                answer_content += delta.content
        
        print(f"\n总共收到 {chunk_count} 个 chunks")
        print("-" * 80)
        
        if thinking_content:
            print("\n" + "=" * 20 + " 思考过程 " + "=" * 20)
            print(thinking_content[:300] + "..." if len(thinking_content) > 300 else thinking_content)
        else:
            print("\n⚠️  未找到思考内容")
        
        print("\n" + "=" * 20 + " 最终回复 " + "=" * 20)
        print(answer_content[:300] + "..." if len(answer_content) > 300 else answer_content)
        
        print("\n✅ 原生 OpenAI SDK 异步流式调用成功\n")
        
    except Exception as e:
        print(f"❌ 原生 OpenAI SDK 异步流式调用失败: {e}\n")
        import traceback
        traceback.print_exc()

# 运行异步测试
asyncio.run(test_astream())


# ============================================================================
# 方式 3: 原生 OpenAI SDK 非流式调用
# ============================================================================
print("\n【方式 3】原生 OpenAI SDK - 非流式调用")
print("=" * 80)

try:
    print("\n调用非流式方法...")
    
    response = client.chat.completions.create(
        model="glm-4.7",
        messages=messages,
        extra_body={"thinking": {"type": "enabled"}},
        stream=False,
    )
    
    print("\n【Response 结构分析】")
    print("-" * 80)
    print(f"Type: {type(response)}")
    print(f"Choices: {len(response.choices)}")
    
    # 获取 message
    message = response.choices[0].message
    print(f"Message type: {type(message)}")
    print(f"Message attributes: {dir(message)}")
    
    # 提取推理内容
    thinking_content = ""
    answer_content = message.content if message.content else ""
    
    # 检查是否有 reasoning_content 属性
    if hasattr(message, 'reasoning_content') and message.reasoning_content:
        thinking_content = message.reasoning_content
    
    if thinking_content:
        print("\n" + "=" * 20 + " 思考过程 " + "=" * 20)
        print(thinking_content[:300] + "..." if len(thinking_content) > 300 else thinking_content)
    else:
        print("\n⚠️  未找到思考内容")
    
    print("\n" + "=" * 20 + " 最终回复 " + "=" * 20)
    print(answer_content[:300] + "..." if len(answer_content) > 300 else answer_content)
    
    print("\n✅ 原生 OpenAI SDK 非流式调用成功\n")
    
except Exception as e:
    print(f"❌ 原生 OpenAI SDK 非流式调用失败: {e}\n")
    import traceback
    traceback.print_exc()


print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
