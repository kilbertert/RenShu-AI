"""
测试所有支持 Thinking Mode 的提供商

根据文档，支持 thinking mode 的提供商：
- OpenAI (gpt-5+): 特殊格式
- DeepSeek (deepseek-reasoner): 标准格式
- Claude (3.5+): 特殊格式
- Qwen: 标准格式
- Gemini (2.0+): 特殊格式
- 智谱 GLM: 标准格式
- Moonshot (Kimi): 标准格式
- 豆包 (Doubao): 标准格式
- MiniMax: 标准格式（自动转换）

不支持 thinking mode 的提供商：
- Grok: 不支持推理输出
- SiliconFlow: 暂未发现支持
- Ollama: 取决于具体模型
"""

import sys
import asyncio
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到 Python 路径
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.src.core.language_model.llm_provider import get_langchain_llm
from langchain_core.messages import HumanMessage

# 统一的 API 配置
API_KEY = "sk-Tz0aHC6HJoHHfKHeFHY2VICuawHRP1QteP0Hy1j0pAbOOFLn"
BASE_URL = "https://www.dmxapi.cn/v1"

# 支持 thinking mode 的提供商配置
THINKING_PROVIDERS = {
    "openai": {
        "model": "gpt-5.2",  # 需要 GPT-5+ 才支持 reasoning
        "api_key": API_KEY,
        "base_url": BASE_URL,
        "enable_thinking": True,
        "format": "openai",  # 特殊格式
    },
    # "deepseek": {
    #     "model": "deepseek-reasoner",  # 只有这个模型支持推理
    #     "api_key": API_KEY,
    #     "base_url": BASE_URL,
    #     "enable_thinking": False,  # deepseek-reasoner 自动启用
    #     "format": "standard",  # 标准格式
    # },
    # "qwen": {
    #     "model": "qwen-max",
    #     "api_key": API_KEY,
    #     "base_url": BASE_URL,
    #     "enable_thinking": True,
    #     "format": "standard",  # 标准格式
    # },
    # "anthropic": {
    #     "model": "claude-sonnet-4-5-20250929",
    #     "api_key": API_KEY,
    #     "base_url": BASE_URL,
    #     "enable_thinking": True,
    #     "format": "claude",  # 特殊格式
    # },
    # "google": {
    #     "model": "gemini-3-flash-preview-thinking",
    #     "api_key": API_KEY,
    #     "base_url": BASE_URL,
    #     "enable_thinking": True,
    #     "format": "claude",  # 与 Claude 相同的格式
    # },
    # "zhipu": {
    #     "model": "glm-4.7",
    #     "api_key": API_KEY,
    #     "base_url": BASE_URL,
    #     "enable_thinking": True,
    #     "format": "standard",  # 标准格式
    # },
    # "moonshot": {
    #     "model": "kimi-k2.5",
    #     "api_key": API_KEY,
    #     "base_url": BASE_URL,
    #     "enable_thinking": True,
    #     "format": "standard",  # 标准格式
    # },
    # "doubao": {
    #     "model": "doubao-seed-1-6-251015",
    #     "api_key": API_KEY,
    #     "base_url": BASE_URL,
    #     "enable_thinking": True,
    #     "format": "standard",  # 标准格式
    # },
    # "minimax": {
    #     "model": "MiniMax-M2.1",
    #     "api_key": API_KEY,
    #     "base_url": BASE_URL,
    #     "enable_thinking": True,
    #     "format": "standard",  # 标准格式（自动转换）
    # },
}

# 测试问题
TEST_QUESTION = "9.11 和 9.8 哪个更大？请详细解释你的推理过程。"


def extract_thinking_content(chunk, format_type: str) -> str:
    """根据格式类型提取思考内容"""
    
    if format_type == "standard":
        # 标准格式：DeepSeek, Qwen, 智谱 GLM, Moonshot, 豆包, MiniMax
        if hasattr(chunk, 'additional_kwargs') and 'reasoning_content' in chunk.additional_kwargs:
            return chunk.additional_kwargs['reasoning_content']
    
    elif format_type == "openai":
        # OpenAI 特殊格式
        if hasattr(chunk, 'content') and isinstance(chunk.content, list):
            for item in chunk.content:
                if isinstance(item, dict) and item.get('type') == 'reasoning':
                    summary_list = item.get('summary', [])
                    for summary_item in summary_list:
                        if isinstance(summary_item, dict):
                            return summary_item.get('text', '')
    
    elif format_type == "claude":
        # Claude/Gemini 特殊格式
        if hasattr(chunk, 'content') and isinstance(chunk.content, list):
            for item in chunk.content:
                if isinstance(item, dict) and item.get('type') == 'thinking':
                    return item.get('thinking', '')
    
    return ""


def extract_answer_content(chunk, format_type: str) -> str:
    """根据格式类型提取答案内容"""
    
    if format_type == "standard":
        # 标准格式：直接从 content 获取
        if hasattr(chunk, 'content') and chunk.content:
            return chunk.content
    
    elif format_type == "openai":
        # OpenAI 特殊格式
        if hasattr(chunk, 'content') and isinstance(chunk.content, list):
            for item in chunk.content:
                if isinstance(item, dict) and item.get('type') == 'text' and item.get('index') == 1:
                    return item.get('text', '')
    
    elif format_type == "claude":
        # Claude/Gemini 特殊格式
        if hasattr(chunk, 'content'):
            if isinstance(chunk.content, str):
                # Gemini 答案是字符串
                return chunk.content
            elif isinstance(chunk.content, list):
                # Claude 答案在列表中
                for item in chunk.content:
                    if isinstance(item, dict) and item.get('type') == 'text' and item.get('index') == 1:
                        return item.get('text', '')
    
    return ""


async def thinking_mode(provider_name: str, config: Dict[str, Any]) -> Dict[str, bool]:
    """测试 thinking mode 的四种方法"""
    
    print(f"\n{'='*80}")
    print(f"测试 {provider_name.upper()} Thinking Mode")
    print(f"模型: {config['model']}")
    print(f"格式: {config['format']}")
    print(f"{'='*80}")
    
    # 检查 API Key
    if not API_KEY or API_KEY.startswith("your-"):
        print(f"⚠️ 跳过 {provider_name}：请先设置 API Key")
        return {
            "同步流式 (stream)": False,
            "异步流式 (astream)": False,
            "非流式 (invoke)": False,
            "异步非流式 (ainvoke)": False,
        }
    
    results = {}
    format_type = config["format"]
    
    # 测试 1: 同步流式
    print(f"\n🔄 测试 1: 同步流式 (stream)")
    try:
        llm = get_langchain_llm(provider_name=provider_name, **{k: v for k, v in config.items() if k != "format"})
        messages = [HumanMessage(content=TEST_QUESTION)]
        
        thinking_content = ""
        answer_content = ""
        thinking_started = False
        answer_started = False
        
        for chunk in llm.stream(input=messages):
            # 提取思考内容
            thinking = extract_thinking_content(chunk, format_type)
            if thinking:
                if not thinking_started:
                    print("🧠 思考过程：")
                    thinking_started = True
                thinking_content += thinking
                print(thinking, end="", flush=True)
            
            # 提取答案内容
            answer = extract_answer_content(chunk, format_type)
            if answer:
                if not answer_started:
                    print("\n\n💬 回答内容：")
                    answer_started = True
                answer_content += answer
                print(answer, end="", flush=True)
        
        print(f"\n✅ 同步流式测试成功！")
        print(f"思考内容长度: {len(thinking_content)} 字符")
        print(f"答案内容长度: {len(answer_content)} 字符")
        results["同步流式 (stream)"] = True
        
    except Exception as e:
        print(f"❌ 同步流式测试失败: {e}")
        results["同步流式 (stream)"] = False
    
    # 测试 2: 异步流式
    print(f"\n🔄 测试 2: 异步流式 (astream)")
    try:
        llm = get_langchain_llm(provider_name=provider_name, **{k: v for k, v in config.items() if k != "format"})
        messages = [HumanMessage(content=TEST_QUESTION)]
        
        thinking_content = ""
        answer_content = ""
        thinking_started = False
        answer_started = False
        
        async for chunk in llm.astream(input=messages):
            # 提取思考内容
            thinking = extract_thinking_content(chunk, format_type)
            if thinking:
                if not thinking_started:
                    print("🧠 思考过程：")
                    thinking_started = True
                thinking_content += thinking
                print(thinking, end="", flush=True)
            
            # 提取答案内容
            answer = extract_answer_content(chunk, format_type)
            if answer:
                if not answer_started:
                    print("\n\n💬 回答内容：")
                    answer_started = True
                answer_content += answer
                print(answer, end="", flush=True)
        
        print(f"\n✅ 异步流式测试成功！")
        print(f"思考内容长度: {len(thinking_content)} 字符")
        print(f"答案内容长度: {len(answer_content)} 字符")
        results["异步流式 (astream)"] = True
        
    except Exception as e:
        print(f"❌ 异步流式测试失败: {e}")
        results["异步流式 (astream)"] = False
    
    # 测试 3: 非流式 (invoke) - 不启用 thinking mode
    print(f"\n🔄 测试 3: 非流式 (invoke) - 不启用 thinking mode")
    try:
        config_copy = config.copy()
        config_copy["enable_thinking"] = False
        
        llm = get_langchain_llm(provider_name=provider_name, **{k: v for k, v in config_copy.items() if k != "format"})
        messages = [HumanMessage(content=TEST_QUESTION)]
        
        response = llm.invoke(input=messages)
        
        print("💬 回答内容：")
        print(response.content)
        
        print(f"\n✅ 非流式测试成功！")
        print(f"答案内容长度: {len(response.content)} 字符")
        results["非流式 (invoke)"] = True
        
    except Exception as e:
        print(f"❌ 非流式测试失败: {e}")
        results["非流式 (invoke)"] = False
    
    # 测试 4: 异步非流式 (ainvoke) - 不启用 thinking mode
    print(f"\n🔄 测试 4: 异步非流式 (ainvoke) - 不启用 thinking mode")
    try:
        config_copy = config.copy()
        config_copy["enable_thinking"] = False
        
        llm = get_langchain_llm(provider_name=provider_name, **{k: v for k, v in config_copy.items() if k != "format"})
        messages = [HumanMessage(content=TEST_QUESTION)]
        
        response = await llm.ainvoke(input=messages)
        
        print("💬 回答内容：")
        print(response.content)
        
        print(f"\n✅ 异步非流式测试成功！")
        print(f"答案内容长度: {len(response.content)} 字符")
        results["异步非流式 (ainvoke)"] = True
        
    except Exception as e:
        print(f"❌ 异步非流式测试失败: {e}")
        results["异步非流式 (ainvoke)"] = False
    
    return results


async def main():
    """主函数"""
    
    print("🚀 测试所有支持 Thinking Mode 的提供商")
    print("=" * 80)
    print(f"✅ 使用统一 API Key: {API_KEY[:20]}...")
    print(f"✅ 使用统一 Base URL: {BASE_URL}")
    print()
    
    print(f"📋 将测试 {len(THINKING_PROVIDERS)} 个支持 thinking mode 的提供商:")
    for provider, config in THINKING_PROVIDERS.items():
        print(f"  - {provider}: {config['model']} ({config['format']} 格式)")
    print()
    
    all_results = {}
    
    for provider_name, config in THINKING_PROVIDERS.items():
        print(f"\n🔄 开始测试提供商: {provider_name}")
        
        try:
            results = await thinking_mode(provider_name, config)
            all_results[provider_name] = results
        except Exception as e:
            print(f"❌ 提供商 {provider_name} 测试出错: {e}")
            all_results[provider_name] = {
                "同步流式 (stream)": False,
                "异步流式 (astream)": False,
                "非流式 (invoke)": False,
                "异步非流式 (ainvoke)": False,
            }
    
    # 输出总结
    print("\n" + "=" * 80)
    print("📊 Thinking Mode 测试总结")
    print("=" * 80)
    
    for provider_name, results in all_results.items():
        config = THINKING_PROVIDERS[provider_name]
        print(f"\n🔹 {provider_name.upper()} ({config['format']} 格式):")
        for test_name, result in results.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"  {test_name}: {status}")
        
        # 统计通过率
        passed = sum(results.values())
        total = len(results)
        print(f"  通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    # 全局统计
    total_tests = sum(len(results) for results in all_results.values())
    total_passed = sum(sum(results.values()) for results in all_results.values())
    
    print(f"\n🎯 全局统计:")
    print(f"总测试数: {total_tests}")
    print(f"通过数: {total_passed}")
    print(f"失败数: {total_tests - total_passed}")
    print(f"总通过率: {total_passed/total_tests*100:.1f}%")
    
    if total_passed == total_tests:
        print("\n🎉 所有 Thinking Mode 测试通过！")
    else:
        print(f"\n⚠️ 有 {total_tests - total_passed} 个测试失败")


if __name__ == "__main__":
    asyncio.run(main())