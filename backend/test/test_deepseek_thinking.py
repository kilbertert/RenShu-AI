"""
测试 DeepSeek Thinking Mode - 仅 LangChain 方式
"""

import os
import sys
import asyncio

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.src.common.config.setting_config import settings

# 导入 ChatDeepSeek
try:
    from app.src.core.language_model.langchain_deepseek import Chat as ChatDeepSeek
    HAS_CHAT_DEEPSEEK = True
except ImportError:
    HAS_CHAT_DEEPSEEK = False
    print("Warning: ChatDeepSeek not found")
    sys.exit(1)

print(f"DeepSeek API Key: {settings.DEEPSEEK_API_KEY[:10]}...")
print(f"DeepSeek Base URL: {settings.DEEPSEEK_BASE_URL}")
print("=" * 80)

# 测试消息
messages = [{"role": "user", "content": "9.11 和 9.8 哪个更大？"}]

# ============================================================================
# 方式 1: LangChain ChatDeepSeek 调用（deepseek-reasoner）
# ============================================================================
print("\n【方式 1】LangChain ChatDeepSeek - deepseek-reasoner")
print("=" * 80)

llm_reasoner = ChatDeepSeek(
    api_key=settings.DEEPSEEK_API_KEY,
    base_url=settings.DEEPSEEK_BASE_URL,
    model="deepseek-reasoner",
    streaming=True,
)

try:
    print("\n【详细 Chunk 分析】")
    print("-" * 80)
    
    chunk_count = 0
    thinking_content = ""
    answer_content = ""
    
    for chunk in llm_reasoner.stream(input=messages):
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
        
        # 收集内容
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
    
    print("\n✅ LangChain ChatDeepSeek (deepseek-reasoner) 调用成功\n")
    
except Exception as e:
    print(f"❌ LangChain ChatDeepSeek (deepseek-reasoner) 调用失败: {e}\n")
    import traceback
    traceback.print_exc()


# ============================================================================
# 方式 2: LangChain ChatDeepSeek ainvoke 调用（非流式异步）
# ============================================================================
print("\n【方式 2】LangChain ChatDeepSeek - ainvoke 非流式异步调用")
print("=" * 80)

async def test_ainvoke():
    """测试 ainvoke 方法的推理输出"""
    llm_ainvoke = ChatDeepSeek(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        model="deepseek-reasoner",
        streaming=False,  # 非流式
    )
    
    try:
        print("\n调用 ainvoke 方法...")
        response = await llm_ainvoke.ainvoke(input=messages)
        
        print("\n【Response 结构分析】")
        print("-" * 80)
        print(f"Type: {type(response)}")
        print(f"Content length: {len(response.content) if response.content else 0}")
        print(f"Additional kwargs keys: {list(response.additional_kwargs.keys())}")
        
        # 提取推理内容
        thinking_content = response.additional_kwargs.get('reasoning_content', '')
        answer_content = response.content
        
        if thinking_content:
            print("\n" + "=" * 20 + " 思考过程 " + "=" * 20)
            print(thinking_content[:300] + "..." if len(thinking_content) > 300 else thinking_content)
        else:
            print("\n⚠️  未找到思考内容")
        
        print("\n" + "=" * 20 + " 最终回复 " + "=" * 20)
        print(answer_content[:300] + "..." if len(answer_content) > 300 else answer_content)
        
        print("\n✅ LangChain ChatDeepSeek ainvoke 调用成功\n")
        
    except Exception as e:
        print(f"❌ LangChain ChatDeepSeek ainvoke 调用失败: {e}\n")
        import traceback
        traceback.print_exc()

# 运行异步测试
asyncio.run(test_ainvoke())


# ============================================================================
# 方式 3: LangChain ChatDeepSeek invoke 调用（非流式同步）
# ============================================================================
print("\n【方式 3】LangChain ChatDeepSeek - invoke 非流式同步调用")
print("=" * 80)

llm_invoke = ChatDeepSeek(
    api_key=settings.DEEPSEEK_API_KEY,
    base_url=settings.DEEPSEEK_BASE_URL,
    model="deepseek-reasoner",
    streaming=False,
)

try:
    print("\n调用 invoke 方法...")
    response = llm_invoke.invoke(input=messages)
    
    print("\n【Response 结构分析】")
    print("-" * 80)
    print(f"Type: {type(response)}")
    print(f"Content length: {len(response.content) if response.content else 0}")
    print(f"Additional kwargs keys: {list(response.additional_kwargs.keys())}")
    
    # 提取推理内容
    thinking_content = response.additional_kwargs.get('reasoning_content', '')
    answer_content = response.content
    
    if thinking_content:
        print("\n" + "=" * 20 + " 思考过程 " + "=" * 20)
        print(thinking_content[:300] + "..." if len(thinking_content) > 300 else thinking_content)
    else:
        print("\n⚠️  未找到思考内容")
    
    print("\n" + "=" * 20 + " 最终回复 " + "=" * 20)
    print(answer_content[:300] + "..." if len(answer_content) > 300 else answer_content)
    
    print("\n✅ LangChain ChatDeepSeek invoke 调用成功\n")
    
except Exception as e:
    print(f"❌ LangChain ChatDeepSeek invoke 调用失败: {e}\n")
    import traceback
    traceback.print_exc()


print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
