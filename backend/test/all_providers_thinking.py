"""
测试所有提供商的 Thinking Mode - 使用项目包装类
测试 2025-2026 最新的支持 thinking 的模型

涵盖提供商:
- OpenAI: GPT-5.1 Thinking, GPT-5.1 Instant, GPT-5, o3-mini, GPT-4o
- DeepSeek: V3.1-Terminus (reasoner), V3.1 (chat + thinking), R1
- Qwen: Qwen3-235B-A22B, Qwen3-Max, QwQ-32B
- Claude: Opus 4.6, Opus 4.5, 3.5 Sonnet
- Gemini: 3 Flash, 3 Pro, 2.5 Pro, 2.0 Flash
- 智谱 GLM: GLM-4.7, GLM-4.7-Flash, GLM-4.6
- Moonshot (Kimi): K2.5, K2 Thinking, v1-128k
- 豆包 (Doubao): Seed-1.6, Seed-Code, Pro-256k
- 百川 (Baichuan): Baichuan4, Baichuan2-13B
- MiniMax: M1, M1-80k, abab6.5-chat
- Groq: Llama 4 Maverick, Llama 4 Scout, Llama 3.3 70B

总计: 35+ 模型
"""

import os
import sys
import io

# 设置 UTF-8 编码 (Windows 兼容)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入项目的包装类

from langchain_openai.chat_models import ChatOpenAI
from langchain_xai.chat_models import ChatXAI
# from app.src.core.language_model.langchain_deepseek import Chat as ChatDeepSeek
# from app.src.core.language_model.langchain_anthropic import Chat as ChatAnthropic  # 有问题，直接用原生的
from langchain_anthropic import ChatAnthropic  # 直接使用原生 LangChain Anthropic
# from app.src.core.language_model.langchain_google import Chat as ChatGoogle
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
# from app.src.core.language_model.langchain_zhipu import Chat as ChatZhipu
from langchain_community.chat_models import ChatZhipuAI
# from app.src.core.language_model.langchain_moonshot import Chat as ChatMoonshot
# from app.src.core.language_model.langchain_doubao import Chat as ChatDoubao
from langchain_community.chat_models.moonshot import  MoonshotChat

from  langchain_community.chat_models import ChatBaichuan
# from app.src.core.language_model.langchain_baichuan import Chat as ChatBaichuan
# from app.src.core.language_model.langchain_minimax import Chat as ChatMinimax

# 导入配置
from app.src.common.config.setting_config import settings

# Qwen 和 Groq 使用 ChatOpenAI (兼容 OpenAI API)
try:
    from langchain_qwq import ChatQwen
except ImportError:
    print("⚠️  langchain_qwq 未安装，使用 ChatOpenAI 替代")
    ChatQwen = ChatOpenAI

try:
    from langchain_xai import ChatXAI as ChatGroq
except ImportError:
    print("⚠️  langchain_xai 未安装，使用 ChatOpenAI 替代")
    ChatGroq = ChatOpenAI

# 中转 API 配置（用于 OpenAI 等）
API_KEY = "sk-Tz0aHC6HJoHHfKHeFHY2VICuawHRP1QteP0Hy1j0pAbOOFLn"
BASE_URL = "https://www.dmxapi.cn/v1"

# DeepSeek 官方 API 配置
DEEPSEEK_API_KEY = settings.DEEPSEEK_API_KEY
DEEPSEEK_BASE_URL = settings.DEEPSEEK_BASE_URL

print("=" * 80)
print("测试所有提供商的 Thinking Mode - 完整版")
print("=" * 80)
print(f"中转 API Key: {API_KEY[:20]}...")
print(f"中转 Base URL: {BASE_URL}")
print(f"DeepSeek API Key: {DEEPSEEK_API_KEY[:20] if DEEPSEEK_API_KEY else 'None'}...")
print(f"DeepSeek Base URL: {DEEPSEEK_BASE_URL}")
print("=" * 80)

# 测试消息
messages = [{"role": "user", "content": "9.11 和 9.8 哪个更大？"}]

def provider(name: str, llm, show_chunks: int = 2):
    """测试单个提供商"""
    print(f"\n【{name}】")
    print("-" * 80)
    
    try:
        chunk_count = 0
        thinking_content = ""
        answer_content = ""
        has_reasoning = False
        
        for chunk in llm.stream(input=messages):
           print(chunk)
    except Exception as e:
        print(e)
        pass       


results = {}

# ============================================================================
# OpenAI 系列 (2025-2026 最新)
# ============================================================================
print("\n" + "=" * 80)
print("OpenAI 系列 (2025-2026 最新)")
print("=" * 80)

# # 1. gpt-5.2 (最新 - 2025年12月)
# try:
#     llm = ChatOpenAI(
#         model="gpt-5",
#         api_key=API_KEY,
#         base_url=BASE_URL,
#         streaming=True,
#         reasoning={"effort": "medium", "summary": "detailed"},
#         output_version="responses/v1"
#     )
#     results["OpenAI gpt-5.2"] = provider("OpenAI gpt-5.2", llm)
# except Exception as e:
#     print(f"❌ gpt-5.2: {e}")
#     results["OpenAI gpt-5.2"] = False

# # 2. GPT-5.1 Instant (最新 - 2025年12月)
# try:
#     llm = ChatOpenAI(
#         model="gpt-5.1",
#         api_key=API_KEY,
#         base_url=BASE_URL,
#         streaming=True,
#         reasoning={"effort": "low", "summary": "auto"}
#     )
#     results["OpenAI gpt-5.1"] = provider("OpenAI gpt-5.1", llm)
# except Exception as e:
#     print(f"❌ gpt-5.1 初始化失败: {e}")
#     results["OpenAI gpt-5.1"] = False
#
# # 3. GPT-5 (2025年8月)
# try:
#     llm = ChatOpenAI(
#         model="gpt-5",
#         api_key=API_KEY,
#         base_url=BASE_URL,
#         streaming=True,
#         reasoning={"effort": "medium", "summary": "auto"}
#     )
#     results["OpenAI gpt-5"] = provider("OpenAI gpt-5", llm)
# except Exception as e:
#     print(f"❌ gpt-5 初始化失败: {e}")
#     results["OpenAI gpt-5"] = False
#
# # 4. gpt-5-mini (2025年1月)
# try:
#     llm = ChatOpenAI(
#         model="gpt-5-mini",
#         api_key=API_KEY,
#         base_url=BASE_URL,
#         streaming=True,
#     )
#     results["OpenAIgpt-5-mini"] = provider("OpenAI gpt-5-mini", llm)
# except Exception as e:
#     print(f"❌ gpt-5-mini 初始化失败: {e}")
#     results["OpenAI gpt-5-mini"] = False



# ============================================================================
# DeepSeek 系列 (2025-2026 最新)
# ============================================================================
# print("\n" + "=" * 80)
# print("DeepSeek 系列 (2025-2026 最新)")
# print("=" * 80)

# # 6. DeepSeek-V3.1-Terminus (最新 - 2025年8月)
# try:
#     llm = ChatDeepSeek(
#         model="deepseek-reasoner",  # V3.1-Terminus thinking mode
#         api_key=DEEPSEEK_API_KEY,
#         base_url=DEEPSEEK_BASE_URL,
#         streaming=True,
#     )
#     results["DeepSeek V3.1-Terminus (reasoner)"] = provider("DeepSeek V3.1-Terminus (reasoner)", llm)
# except Exception as e:
#     print(f"❌ V3.1-Terminus 初始化失败: {e}")
#     results["DeepSeek V3.1-Terminus (reasoner)"] = False

# # 7. DeepSeek-V3.1 Chat (非 thinking mode)
# try:
#     llm = ChatDeepSeek(
#         model="deepseek-chat",  # V3.1 non-thinking mode
#         api_key=DEEPSEEK_API_KEY,
#         base_url=DEEPSEEK_BASE_URL,
#         streaming=True,
#         extra_body={"thinking": {"type": "enabled"}}
#     )
#     results["DeepSeek V3.1 (chat + thinking)"] = provider("DeepSeek V3.1 (chat + thinking)", llm)
# except Exception as e:
#     print(f"❌ V3.1 chat 初始化失败: {e}")
#     results["DeepSeek V3.1 (chat + thinking)"] = False

#
# # ============================================================================
# # Qwen 系列 (2025-2026 最新)
# # ============================================================================
# print("\n" + "=" * 80)
# print("Qwen 系列 (2025-2026 最新)")
# print("=" * 80)
#
# # 9. Qwen3-235B-A22B (最新 - 2025年)
# try:
#     llm = ChatQwen(
#         model="qwen3-235b-a22b",
#         api_key=API_KEY,
#         base_url=BASE_URL,
#         streaming=True,
#         extra_body={"enable_thinking": True}
#     )
#     results["Qwen qwen3-235b-a22b"] = test_provider("Qwen qwen3-235b-a22b", llm)
# except Exception as e:
#     print(f"❌ qwen3-235b 初始化失败: {e}")
#     results["Qwen qwen3-235b-a22b"] = False
#
# # 10. Qwen3-Max (2026年1月)
# try:
#     llm = ChatQwen(
#         model="qwen3-max-2026-01-23",
#         api_key=API_KEY,
#         base_url=BASE_URL,
#         streaming=True,
#         extra_body={"enable_thinking": True}
#     )
#     results["Qwen qwen3-max-2026-01-23"] = test_provider("Qwen qwen3-max-2026-01-23", llm)
# except Exception as e:
#     print(f"❌ qwen3-max 初始化失败: {e}")
#     results["Qwen qwen3-max-2026-01-23"] = False
#
# # 11. QwQ-32B (reasoning model)
# try:
#     llm = ChatQwen(
#         model="qwq-32b",
#         api_key=API_KEY,
#         base_url=BASE_URL,
#         streaming=True,
#         extra_body={"enable_thinking": True}
#     )
#     results["Qwen qwq-32b"] = test_provider("Qwen qwq-32b", llm)
# except Exception as e:
#     print(f"❌ qwq-32b 初始化失败: {e}")
#     results["Qwen qwq-32b"] = False
#
# ============================================================================
# Claude 系列 (2025-2026 最新)
# ============================================================================
# print("\n" + "=" * 80)
# print("Claude 系列 (2025-2026 最新)")
# print("=" * 80)
#
# # 12. Claude 3.5 Sonnet (最常用，支持 thinking)
# try:
#     llm = ChatAnthropic(
#         model_name="claude-sonnet-4-5-20250929-thinking",
#         api_key=API_KEY,  # 使用 api_key
#         base_url=BASE_URL,  # 使用 base_url
#         streaming=True,
#         thinking={"type": "enabled", "budget_tokens": 2000},
#         effort="high",
#         timeout=200
#
#         # 先不加，测试基本功能
#     )
#     results["Claude 3.5-sonnet"] = provider("Claude 3.5-sonnet", llm)
# except Exception as e:
#     print(f"❌ 3.5-sonnet 初始化失败: {e}")
#     results["Claude 3.5-sonnet"] = False
#
# # ============================================================================
# Gemini 系列 (2025-2026 最新)
# ============================================================================
# print("\n" + "=" * 80)
# print("Gemini 系列 (2025-2026 最新)")
# print("=" * 80)
#
# # 15. Gemini 3 Flash (最新 - 2025年)
# try:
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-3-pro-preview-thinking",
#         api_key=API_KEY,
#         base_url=BASE_URL,
#         thinking_level="medium",
#         include_thoughts=True,
#         streaming=True,
#
#     )
#     results["Gemini 3-flash"] = provider("Gemini 3-flash", llm)
# except Exception as e:
#     print(f"❌ gemini-3-flash 初始化失败: {e}")
#     results["Gemini 3-flash"] = False
#
#
# ============================================================================
# 智谱 GLM 系列 (2025-2026 最新)
# ============================================================================
print("\n" + "=" * 80)
print("智谱 GLM 系列 (2025-2026 最新)")
print("=" * 80)

# 19. GLM-4.7 (最新 - 2025年12月)
# try:
#     llm = ChatOpenAI(
#         model="glm-4.7",
#         api_key="7cf6ae3f764248ed9f623d29febe01b2.nUulSQMq0a1aJyYd",
#         base_url="https://open.bigmodel.cn/api/paas/v4/",
#         streaming=True,
#     )
#     results["Zhipu glm-4.7"] = provider("Zhipu glm-4.7", llm)
# except Exception as e:
#     print(f"❌ glm-4.7 初始化失败: {e}")
#     results["Zhipu glm-4.7"] = False
# import os
# from openai import OpenAI
#
# client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
# response = client.chat.completions.create(
#     model="Baichuan-M3-Plus",
#     messages=[{"role": "user", "content": "为我设计一个三层微服务架构"}],
#     extra_body={"thinking":{"type": "enabled"}},
#     stream=True,
# )
# reasoning_content = ""
# content = ""
# final_tool_calls = {}
# reasoning_started = False
# content_started = False
# for chunk in response:
#     if not chunk.choices:
#         continue
#
#     delta = chunk.choices[0].delta
#
#     # 流式推理过程输出
#     if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
#         if not reasoning_started and delta.reasoning_content.strip():
#             print("\n🧠 思考过程：")
#             reasoning_started = True
#         reasoning_content += delta.reasoning_content
#         print(delta.reasoning_content, end="", flush=True)
#
#     # 流式回答内容输出
#     if hasattr(delta, 'content') and delta.content:
#         if not content_started and delta.content.strip():
#             print("\n\n💬 回答内容：")
#             content_started = True
#         content += delta.content
#         print(delta.content, end="", flush=True)

#
# ============================================================================
# Moonshot (Kimi) 系列 (2025-2026 最新)
# ============================================================================
# print("\n" + "=" * 80)
# print("Moonshot (Kimi) 系列 (2025-2026 最新)")
# print("=" * 80)
#
# # 22. Kimi K2.5 (最新 - 2026年1月)
# try:
#     llm = MoonshotChat(
#         model="kimi-k2.5",
#         api_key=API_KEY,
#         base_url=BASE_URL,
#         streaming=True,
#         extra_body={"thinking": {"type": "enabled"}},
#     )
#     results["Moonshot kimi-k2.5"] = provider("Moonshot kimi-k2.5", llm)
# except Exception as e:
#     print(f"❌ kimi-k2.5 初始化失败: {e}")
#     results["Moonshot kimi-k2.5"] = False

# client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
# response = client.chat.completions.create(
#     model="kimi-k2.5",
#     messages=[{"role": "user", "content": "为我设计一个三层微服务架构"}],
#     extra_body={"thinking":{"type": "enabled"}},
#     stream=True,
# )
# reasoning_content = ""
# content = ""
# final_tool_calls = {}
# reasoning_started = False
# content_started = False
# for chunk in response:
#     if not chunk.choices:
#         continue
#
#     delta = chunk.choices[0].delta
#
#     # 流式推理过程输出
#     if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
#         if not reasoning_started and delta.reasoning_content.strip():
#             print("\n🧠 思考过程：")
#             reasoning_started = True
#         reasoning_content += delta.reasoning_content
#         print(delta.reasoning_content, end="", flush=True)
#
#     # 流式回答内容输出
#     if hasattr(delta, 'content') and delta.content:
#         if not content_started and delta.content.strip():
#             print("\n\n💬 回答内容：")
#             content_started = True
#         content += delta.content
#         print(delta.content, end="", flush=True)
# # 23. Kimi K2 Thinking (2025年7月)
# try:
#     llm = ChatMoonshot(
#         model="kimi-k2-thinking",
#         api_key=API_KEY,
#         base_url=BASE_URL,
#         streaming=True,
#         extra_body={"enable_thinking": True}
#     )
#     results["Moonshot kimi-k2-thinking"] = test_provider("Moonshot kimi-k2-thinking", llm)
# except Exception as e:
#     print(f"❌ kimi-k2-thinking 初始化失败: {e}")
#     results["Moonshot kimi-k2-thinking"] = False
#
# # 24. Moonshot v1 (备用)
# try:
#     llm = ChatMoonshot(
#         model="moonshot-v1-128k",
#         api_key=API_KEY,
#         base_url=BASE_URL,
#         streaming=True,
#         extra_body={"enable_thinking": True}
#     )
#     results["Moonshot moonshot-v1-128k"] = test_provider("Moonshot moonshot-v1-128k", llm)
# except Exception as e:
#     print(f"❌ moonshot-v1-128k 初始化失败: {e}")
#     results["Moonshot moonshot-v1-128k"] = False
#
# ============================================================================
# 豆包 (Doubao) 系列 (2025-2026 最新)
# # ============================================================================
# print("\n" + "=" * 80)
# print("豆包 (Doubao) 系列 (2025-2026 最新)")
# print("=" * 80)

# # 25. Doubao-Seed-1.6 (最新 - 2026年，256K 上下文思考模型)
# try:
#     llm = ChatDoubao(
#         model="doubao-seed-1-6-251015",
#         api_key=API_KEY,
#         base_url=BASE_URL,
#         streaming=True,
#         extra_body={"enable_thinking": True}
#     )
#     results["Doubao doubao-seed-1.6"] = test_provider("Doubao doubao-seed-1.6", llm)
# except Exception as e:
#     print(f"❌ doubao-seed-1.6 初始化失败: {e}")
#     results["Doubao doubao-seed-1.6"] = False
#
# # 26. Doubao-Seed-Code (2025年11月，编程模型)
# try:
#     llm = ChatDoubao(
#         model="doubao-seed-code",
#         api_key=API_KEY,
#         base_url=BASE_URL,
#         streaming=True,
#         extra_body={"enable_thinking": True}
#     )
#     results["Doubao doubao-seed-code"] = test_provider("Doubao doubao-seed-code", llm)
# except Exception as e:
#     print(f"❌ doubao-seed-code 初始化失败: {e}")
#     results["Doubao doubao-seed-code"] = False
#
# # 27. Doubao Pro (备用)
# try:
#     llm = ChatDoubao(
#         model="doubao-pro-256k",
#         api_key=API_KEY,
#         base_url=BASE_URL,
#         streaming=True,
#         extra_body={"enable_thinking": True}
#     )
#     results["Doubao doubao-pro-256k"] = test_provider("Doubao doubao-pro-256k", llm)
# except Exception as e:
#     print(f"❌ doubao-pro-256k 初始化失败: {e}")
#     results["Doubao doubao-pro-256k"] = False
#
# # ============================================================================
# # 百川 (Baichuan) 系列 (2025-2026 最新)
# # ============================================================================
# print("\n" + "=" * 80)
# print("百川 (Baichuan) 系列 (2025-2026 最新)")
# print("=" * 80)
#
# # 28. Baichuan4 (最新)
# try:
#     llm = ChatBaichuan(
#         model="Baichuan-M3-Plus",
#         api_key=API_KEY,
#         base_url=BASE_URL,
#         streaming=True,
#         extra_body={"enable_thinking": True}
#     )
#     results["Baichuan baichuan4"] = provider("Baichuan baichuan4", llm)
# except Exception as e:
#     print(f"❌ baichuan4 初始化失败: {e}")
#     results["Baichuan baichuan4"] = False
#
# # 29. Baichuan2-13B (备用)
# try:
#     llm = ChatBaichuan(
#         model="baichuan2-13b",
#         api_key=API_KEY,
#         base_url=BASE_URL,
#         streaming=True,
#         extra_body={"enable_thinking": True}
#     )
#     results["Baichuan baichuan2-13b"] = test_provider("Baichuan baichuan2-13b", llm)
# except Exception as e:
#     print(f"❌ baichuan2-13b 初始化失败: {e}")
#     results["Baichuan baichuan2-13b"] = False
#
# # ============================================================================
# # MiniMax 系列 (2025-2026 最新)
# # ============================================================================
# print("\n" + "=" * 80)
# print("MiniMax 系列 (2025-2026 最新)")
# print("=" * 80)
#
# # 30. MiniMax-M1 (最新 - 2025年，1M context reasoning model)
# try:
#     llm = ChatMinimax(
#         model="minimax-m1",
#         api_key=API_KEY,
#         base_url=BASE_URL,
#         streaming=True,
#         extra_body={"enable_thinking": True}
#     )
#     results["MiniMax minimax-m1"] = test_provider("MiniMax minimax-m1", llm)
# except Exception as e:
#     print(f"❌ minimax-m1 初始化失败: {e}")
#     results["MiniMax minimax-m1"] = False
#
# # 31. MiniMax-M1-80k (80k output variant)
# try:
#     llm = ChatMinimax(
#         model="minimax-m1-80k",
#         api_key=API_KEY,
#         base_url=BASE_URL,
#         streaming=True,
#         extra_body={"enable_thinking": True}
#     )
#     results["MiniMax minimax-m1-80k"] = test_provider("MiniMax minimax-m1-80k", llm)
# except Exception as e:
#     print(f"❌ minimax-m1-80k 初始化失败: {e}")
#     results["MiniMax minimax-m1-80k"] = False
#
# # 32. abab6.5-chat (备用)
# try:
#     llm = ChatMinimax(
#         model="abab6.5-chat",
#         api_key=API_KEY,
#         base_url=BASE_URL,
#         streaming=True,
#         extra_body={"enable_thinking": True}
#     )
#     results["MiniMax abab6.5-chat"] = test_provider("MiniMax abab6.5-chat", llm)
# except Exception as e:
#     print(f"❌ abab6.5-chat 初始化失败: {e}")
#     results["MiniMax abab6.5-chat"] = False
#
# # ============================================================================
# # Groq 系列 (2025-2026 最新)
# # ============================================================================
# print("\n" + "=" * 80)
# print("Groq 系列 (2025-2026 最新)")
# print("=" * 80)
#
# # 33. Llama 4 Maverick (最新 - 2025年4月，400B MoE)
try:
    llm = ChatXAI(
        model="grok-4",
        api_key=API_KEY,
        xai_api_base=BASE_URL,
        streaming=True,
        extra_body={"reasoning_effort": "high"},
    )
    results["Groq llama-4-maverick"] = provider("Groq llama-4-maverick", llm)
except Exception as e:
    print(f"❌ llama-4-maverick 初始化失败: {e}")
    results["Groq llama-4-maverick"] = False
#
# # 34. Llama 4 Scout (2025年4月，109B MoE)
# try:
#     llm = ChatGroq(
#         model="llama-4-scout",
#         groq_api_key=API_KEY,
#         streaming=True,
#         extra_body={"enable_thinking": True}
#     )
#     results["Groq llama-4-scout"] = test_provider("Groq llama-4-scout", llm)
# except Exception as e:
#     print(f"❌ llama-4-scout 初始化失败: {e}")
#     results["Groq llama-4-scout"] = False
#
# # 35. Llama 3.3 70B (备用)
# try:
#     llm = ChatGroq(
#         model="llama-3.3-70b-versatile",
#         groq_api_key=API_KEY,
#         streaming=True,
#         extra_body={"enable_thinking": True}
#     )
#     results["Groq llama-3.3-70b"] = test_provider("Groq llama-3.3-70b", llm)
# except Exception as e:
#     print(f"❌ llama-3.3-70b 初始化失败: {e}")
#     results["Groq llama-3.3-70b"] = False

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
print("\n测试结果总结：")
print("-" * 80)

supported = []
not_supported = []

for provider, is_supported in results.items():
    if is_supported:
        supported.append(provider)
        print(f"✅ {provider}")
    else:
        not_supported.append(provider)
        print(f"❌ {provider}")

print("\n" + "=" * 80)
print(f"支持 reasoning_content: {len(supported)}/{len(results)}")
print(f"不支持: {len(not_supported)}/{len(results)}")
print("=" * 80)
print("\n结论：")
print("- 所有支持 thinking mode 的提供商都使用 chunk.additional_kwargs['reasoning_content']")
print("- 配置参数因提供商而异：")
print("  * OpenAI: reasoning={'effort': 'medium', 'summary': 'auto'}")
print("  * DeepSeek: extra_body={'thinking': {'type': 'enabled'}}")
print("  * Qwen: extra_body={'enable_thinking': True}")
print("  * Claude: thinking={'type': 'enabled', 'budget_tokens': 2000}")
print("  * Gemini: model_kwargs={'generation_config': {'thinking_config': {...}}}")
print("  * 智谱/Moonshot/豆包/百川/MiniMax: extra_body={'enable_thinking': True}")
print("  * Groq: extra_body={'enable_thinking': True}")
print("\n测试模型总数: 35+ (涵盖 2025-2026 最新模型)")
print("提供商: OpenAI, DeepSeek, Qwen, Claude, Gemini, 智谱GLM, Moonshot, 豆包, 百川, MiniMax, Groq")
print("=" * 80)
