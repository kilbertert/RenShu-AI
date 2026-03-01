"""
测试 Claude Thinking Mode - ainvoke 方法
"""

import os
import sys
import asyncio

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入 ChatAnthropic
try:
    from langchain_anthropic import ChatAnthropic
    HAS_CHAT_ANTHROPIC = True
except ImportError:
    HAS_CHAT_ANTHROPIC = False
    print("Warning: ChatAnthropic not found")
    sys.exit(1)

# 直接使用提供的 API Key 和 Base URL
CLAUDE_API_KEY = "sk-Tz0aHC6HJoHHfKHeFHY2VICuawHRP1QteP0Hy1j0pAbOOFLn"
CLAUDE_BASE_URL = "https://www.dmxapi.cn"  # 移除 /v1，LangChain 会自动添加

print(f"Claude API Key: {CLAUDE_API_KEY[:10]}...")
print(f"Claude Base URL: {CLAUDE_BASE_URL}")
print("=" * 80)

# 测试消息
messages = [{"role": "user", "content": "9.11 和 9.8 哪个更大？"}]

# ============================================================================
# 方式 1: LangChain ChatAnthropic 流式调用
# ============================================================================
print("\n【方式 1】LangChain ChatAnthropic - stream 流式调用")
print("=" * 80)

llm_stream = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    api_key=CLAUDE_API_KEY,
    base_url=CLAUDE_BASE_URL,
    streaming=True,
    thinking={"type": "enabled", "budget_tokens": 2000}
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
            print(f"Content type: {type(chunk.content)}")
            print(f"Content: {repr(chunk.content)[:100]}")
        
        # 收集思考内容 - Claude 特殊格式
        if isinstance(chunk.content, list):
            for item in chunk.content:
                if isinstance(item, dict):
                    # 思考内容：type='thinking'
                    if item.get('type') == 'thinking':
                        thinking = item.get('thinking', '')
                        thinking_content += thinking
                    
                    # 答案内容：type='text', index=1
                    elif item.get('type') == 'text' and item.get('index') == 1:
                        answer = item.get('text', '')
                        answer_content += answer
        
        # 如果 content 是字符串（答案内容）
        elif isinstance(chunk.content, str) and chunk.content:
            answer_content += chunk.content
    
    print(f"\n总共收到 {chunk_count} 个 chunks")
    print("-" * 80)
    
    if thinking_content:
        print("\n" + "=" * 20 + " 思考过程 " + "=" * 20)
        print(thinking_content[:300] + "..." if len(thinking_content) > 300 else thinking_content)
    else:
        print("\n⚠️  未找到思考内容")
    
    print("\n" + "=" * 20 + " 最终回复 " + "=" * 20)
    print(answer_content[:300] + "..." if len(answer_content) > 300 else answer_content)
    
    print("\n✅ LangChain ChatAnthropic 流式调用成功\n")
    
except Exception as e:
    print(f"❌ LangChain ChatAnthropic 流式调用失败: {e}\n")
    import traceback
    traceback.print_exc()


# ============================================================================
# 方式 2: LangChain ChatAnthropic ainvoke 调用（非流式异步）
# ============================================================================
print("\n【方式 2】LangChain ChatAnthropic - ainvoke 非流式异步调用")
print("=" * 80)

async def test_ainvoke():
    """测试 ainvoke 方法的推理输出"""
    llm_ainvoke = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        api_key=CLAUDE_API_KEY,
        base_url=CLAUDE_BASE_URL,
        streaming=False,  # 非流式
        thinking={"type": "enabled", "budget_tokens": 2000}
    )
    
    try:
        print("\n调用 ainvoke 方法...")
        response = await llm_ainvoke.ainvoke(input=messages)
        
        print("\n【Response 结构分析】")
        print("-" * 80)
        print(f"Type: {type(response)}")
        print(f"Content type: {type(response.content)}")
        print(f"Content: {repr(response.content)[:200]}")
        
        # 提取推理内容 - Claude 特殊格式
        thinking_content = ""
        answer_content = ""
        
        if isinstance(response.content, list):
            for item in response.content:
                if isinstance(item, dict):
                    # 思考内容：type='thinking'
                    if item.get('type') == 'thinking':
                        thinking_content += item.get('thinking', '')
                    
                    # 答案内容：type='text'
                    elif item.get('type') == 'text':
                        answer_content += item.get('text', '')
        elif isinstance(response.content, str):
            answer_content = response.content
        
        if thinking_content:
            print("\n" + "=" * 20 + " 思考过程 " + "=" * 20)
            print(thinking_content[:300] + "..." if len(thinking_content) > 300 else thinking_content)
        else:
            print("\n⚠️  未找到思考内容")
        
        print("\n" + "=" * 20 + " 最终回复 " + "=" * 20)
        print(answer_content[:300] + "..." if len(answer_content) > 300 else answer_content)
        
        print("\n✅ LangChain ChatAnthropic ainvoke 调用成功\n")
        
    except Exception as e:
        print(f"❌ LangChain ChatAnthropic ainvoke 调用失败: {e}\n")
        import traceback
        traceback.print_exc()

# 运行异步测试
asyncio.run(test_ainvoke())


# ============================================================================
# 方式 3: LangChain ChatAnthropic invoke 调用（非流式同步）
# ============================================================================
print("\n【方式 3】LangChain ChatAnthropic - invoke 非流式同步调用")
print("=" * 80)

llm_invoke = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    api_key=CLAUDE_API_KEY,
    base_url=CLAUDE_BASE_URL,
    streaming=False,
    thinking={"type": "enabled", "budget_tokens": 2000}
)

try:
    print("\n调用 invoke 方法...")
    response = llm_invoke.invoke(input=messages)
    
    print("\n【Response 结构分析】")
    print("-" * 80)
    print(f"Type: {type(response)}")
    print(f"Content type: {type(response.content)}")
    print(f"Content: {repr(response.content)[:200]}")
    
    # 提取推理内容 - Claude 特殊格式
    thinking_content = ""
    answer_content = ""
    
    if isinstance(response.content, list):
        for item in response.content:
            if isinstance(item, dict):
                # 思考内容：type='thinking'
                if item.get('type') == 'thinking':
                    thinking_content += item.get('thinking', '')
                
                # 答案内容：type='text'
                elif item.get('type') == 'text':
                    answer_content += item.get('text', '')
    elif isinstance(response.content, str):
        answer_content = response.content
    
    if thinking_content:
        print("\n" + "=" * 20 + " 思考过程 " + "=" * 20)
        print(thinking_content[:300] + "..." if len(thinking_content) > 300 else thinking_content)
    else:
        print("\n⚠️  未找到思考内容")
    
    print("\n" + "=" * 20 + " 最终回复 " + "=" * 20)
    print(answer_content[:300] + "..." if len(answer_content) > 300 else answer_content)
    
    print("\n✅ LangChain ChatAnthropic invoke 调用成功\n")
    
except Exception as e:
    print(f"❌ LangChain ChatAnthropic invoke 调用失败: {e}\n")
    import traceback
    traceback.print_exc()


print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
