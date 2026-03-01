"""
测试所有提供商的四种调用方式

验证四种调用方式：
1. 同步流式 (stream)
2. 异步流式 (astream)
3. 非流式 (invoke)
4. 异步非流式 (ainvoke)

支持的提供商：
- openai, azure, deepseek, qwen, zhipu, moonshot
- doubao, minimax, siliconflow, ollama
- anthropic, google, grok, xai
"""

import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目根目录到 Python 路径
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.src.core.language_model.llm_provider import get_langchain_llm, get_supported_providers
from langchain_core.messages import HumanMessage

# 统一的 API 配置
API_KEY = "sk-Tz0aHC6HJoHHfKHeFHY2VICuawHRP1QteP0Hy1j0pAbOOFLn"
BASE_URL = "https://www.dmxapi.cn/v1"

# 测试配置 - 使用统一的 API Key 和 Base URL
PROVIDER_CONFIGS = {
    "openai": {
        "model": "gpt-4o-mini",
        "api_key": API_KEY,
        "base_url": BASE_URL,
        "enable_thinking": False,
    },
    "deepseek": {
        "model": "deepseek-chat",  # 普通聊天模型
        "api_key": API_KEY,
        "base_url": BASE_URL,
        "enable_thinking": False,
    },
    "qwen": {
        "model": "qwen-max",
        "api_key": API_KEY,
        "base_url": BASE_URL,
        "enable_thinking": False,
    },
    "zhipu": {
        "model": "glm-4-plus",
        "api_key": API_KEY,
        "base_url": BASE_URL,
        "enable_thinking": False,
    },
    "moonshot": {
        "model": "moonshot-v1-8k",
        "api_key": API_KEY,
        "base_url": BASE_URL,
        "enable_thinking": False,
    },
    "doubao": {
        "model": "doubao-pro-4k",
        "api_key": API_KEY,
        "base_url": BASE_URL,
        "enable_thinking": False,
    },
    "minimax": {
        "model": "abab6.5-chat",
        "api_key": API_KEY,
        "base_url": BASE_URL,
        "enable_thinking": False,
    },
    "anthropic": {
        "model": "claude-3-haiku-20240307",
        "api_key": API_KEY,
        "base_url": BASE_URL,
        "enable_thinking": False,
    },
    "google": {
        "model": "gemini-pro",
        "api_key": API_KEY,
        "base_url": BASE_URL,
        "enable_thinking": False,
    },
    "grok": {
        "model": "grok-beta",
        "api_key": API_KEY,
        "base_url": BASE_URL,
        "enable_thinking": False,
    },
    "siliconflow": {
        "model": "deepseek-ai/DeepSeek-V2.5",
        "api_key": API_KEY,
        "base_url": BASE_URL,
        "enable_thinking": False,
    },
    "ollama": {
        "model": "llama3.2",
        "base_url": "http://localhost:11434",
        "enable_thinking": False,
    },
}

# 测试问题
TEST_QUESTION = "9.11 和 9.8 哪个更大？请简短回答。"


def test_provider_sync_stream(provider_name: str, config: Dict[str, Any]) -> bool:
    """测试同步流式 (stream)"""
    
    print(f"\n{'='*60}")
    print(f"测试 {provider_name.upper()} - 同步流式 (stream)")
    print(f"{'='*60}")
    
    try:
        llm = get_langchain_llm(
            provider_name=provider_name,
            **config
        )
        
        messages = [HumanMessage(content=TEST_QUESTION)]
        
        print(f"\n📝 问题: {TEST_QUESTION}\n")
        print("💬 回答内容：")
        
        content = ""
        thinking_content = ""
        
        for chunk in llm.stream(input=messages):
            # 检查是否有思考内容
            if hasattr(chunk, 'additional_kwargs') and 'reasoning_content' in chunk.additional_kwargs:
                thinking = chunk.additional_kwargs['reasoning_content']
                if thinking and not thinking_content:
                    print("🧠 思考过程：")
                thinking_content += thinking
                print(thinking, end="", flush=True)
            
            # 检查答案内容
            if hasattr(chunk, 'content') and chunk.content:
                if thinking_content and not content:
                    print("\n\n💬 回答内容：")
                content += chunk.content
                print(chunk.content, end="", flush=True)
        
        print(f"\n\n✅ {provider_name} 同步流式测试成功！")
        print(f"回答长度: {len(content)} 字符")
        if thinking_content:
            print(f"思考内容长度: {len(thinking_content)} 字符")
        
        return True
        
    except Exception as e:
        print(f"\n❌ {provider_name} 同步流式测试失败: {e}")
        return False


async def test_provider_async_stream(provider_name: str, config: Dict[str, Any]) -> bool:
    """测试异步流式 (astream)"""
    
    print(f"\n{'='*60}")
    print(f"测试 {provider_name.upper()} - 异步流式 (astream)")
    print(f"{'='*60}")
    
    try:
        llm = get_langchain_llm(
            provider_name=provider_name,
            **config
        )
        
        messages = [HumanMessage(content=TEST_QUESTION)]
        
        print(f"\n📝 问题: {TEST_QUESTION}\n")
        print("💬 回答内容：")
        
        content = ""
        thinking_content = ""
        
        async for chunk in llm.astream(input=messages):
            # 检查是否有思考内容
            if hasattr(chunk, 'additional_kwargs') and 'reasoning_content' in chunk.additional_kwargs:
                thinking = chunk.additional_kwargs['reasoning_content']
                if thinking and not thinking_content:
                    print("🧠 思考过程：")
                thinking_content += thinking
                print(thinking, end="", flush=True)
            
            # 检查答案内容
            if hasattr(chunk, 'content') and chunk.content:
                if thinking_content and not content:
                    print("\n\n💬 回答内容：")
                content += chunk.content
                print(chunk.content, end="", flush=True)
        
        print(f"\n\n✅ {provider_name} 异步流式测试成功！")
        print(f"回答长度: {len(content)} 字符")
        if thinking_content:
            print(f"思考内容长度: {len(thinking_content)} 字符")
        
        return True
        
    except Exception as e:
        print(f"\n❌ {provider_name} 异步流式测试失败: {e}")
        return False


def test_provider_invoke(provider_name: str, config: Dict[str, Any]) -> bool:
    """测试非流式 (invoke)"""
    
    print(f"\n{'='*60}")
    print(f"测试 {provider_name.upper()} - 非流式 (invoke)")
    print(f"{'='*60}")
    
    try:
        # 非流式不启用 thinking mode
        config_copy = config.copy()
        config_copy["enable_thinking"] = False
        
        llm = get_langchain_llm(
            provider_name=provider_name,
            **config_copy
        )
        
        messages = [HumanMessage(content=TEST_QUESTION)]
        
        print(f"\n📝 问题: {TEST_QUESTION}\n")
        print("💬 回答内容：")
        
        response = llm.invoke(input=messages)
        
        print(response.content)
        
        print(f"\n✅ {provider_name} 非流式测试成功！")
        print(f"回答长度: {len(response.content)} 字符")
        
        return True
        
    except Exception as e:
        print(f"\n❌ {provider_name} 非流式测试失败: {e}")
        return False


async def test_provider_ainvoke(provider_name: str, config: Dict[str, Any]) -> bool:
    """测试异步非流式 (ainvoke)"""
    
    print(f"\n{'='*60}")
    print(f"测试 {provider_name.upper()} - 异步非流式 (ainvoke)")
    print(f"{'='*60}")
    
    try:
        # 非流式不启用 thinking mode
        config_copy = config.copy()
        config_copy["enable_thinking"] = False
        
        llm = get_langchain_llm(
            provider_name=provider_name,
            **config_copy
        )
        
        messages = [HumanMessage(content=TEST_QUESTION)]
        
        print(f"\n📝 问题: {TEST_QUESTION}\n")
        print("💬 回答内容：")
        
        response = await llm.ainvoke(input=messages)
        
        print(response.content)
        
        print(f"\n✅ {provider_name} 异步非流式测试成功！")
        print(f"回答长度: {len(response.content)} 字符")
        
        return True
        
    except Exception as e:
        print(f"\n❌ {provider_name} 异步非流式测试失败: {e}")
        return False


async def test_single_provider(provider_name: str, config: Dict[str, Any]) -> Dict[str, bool]:
    """测试单个提供商的所有方法"""
    
    # 检查是否有 API Key（ollama 除外）
    if provider_name != "ollama" and (not API_KEY or API_KEY.startswith("your-")):
        print(f"\n⚠️ 跳过 {provider_name}：请先设置 API Key")
        return {
            "同步流式 (stream)": False,
            "异步流式 (astream)": False,
            "非流式 (invoke)": False,
            "异步非流式 (ainvoke)": False,
        }
    
    results = {}
    
    # 测试 1: 同步流式
    results["同步流式 (stream)"] = test_provider_sync_stream(provider_name, config)
    
    # 测试 2: 异步流式
    results["异步流式 (astream)"] = await test_provider_async_stream(provider_name, config)
    
    # 测试 3: 非流式
    results["非流式 (invoke)"] = test_provider_invoke(provider_name, config)
    
    # 测试 4: 异步非流式
    results["异步非流式 (ainvoke)"] = await test_provider_ainvoke(provider_name, config)
    
    return results


async def test_all_providers():
    """测试所有提供商"""
    
    print("🚀 开始测试所有提供商的四种调用方式")
    print("=" * 80)
    print(f"✅ 使用统一 API Key: {API_KEY[:20]}...")
    print(f"✅ 使用统一 Base URL: {BASE_URL}")
    print()
    
    # 获取支持的提供商列表
    supported_providers = get_supported_providers()
    
    # 过滤出有配置的提供商
    test_providers = [p for p in supported_providers if p in PROVIDER_CONFIGS]
    
    print(f"支持的提供商: {supported_providers}")
    print(f"有配置的提供商: {test_providers}")
    print(f"将测试 {len(test_providers)} 个提供商")
    print()
    
    for provider in test_providers:
        config = PROVIDER_CONFIGS[provider]
        print(f"  - {provider}: {config['model']}")
    print()
    
    all_results = {}
    
    for provider_name in test_providers:
        config = PROVIDER_CONFIGS[provider_name]
        
        print(f"\n🔄 开始测试提供商: {provider_name}")
        
        try:
            results = await test_single_provider(provider_name, config)
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
    print("📊 测试总结")
    print("=" * 80)
    
    for provider_name, results in all_results.items():
        print(f"\n🔹 {provider_name.upper()}:")
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
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠️ 有 {total_tests - total_passed} 个测试失败，请检查配置和网络连接")


def main():
    """主函数"""
    
    print(f"✅ 已配置统一 API Key: {API_KEY[:20]}...")
    print(f"✅ 已配置统一 Base URL: {BASE_URL}")
    print()
    
    # 检查是否有任何有效的 API Key
    if not API_KEY or API_KEY.startswith("your-"):
        print("❌ 没有找到有效的 API Key 配置，请先设置 API Key")
        return
    
    valid_configs = [p for p in PROVIDER_CONFIGS.keys() if p == "ollama" or API_KEY]
    
    print(f"✅ 找到 {len(valid_configs)} 个有效配置: {valid_configs}")
    print()
    
    # 运行测试
    asyncio.run(test_all_providers())


if __name__ == "__main__":
    main()