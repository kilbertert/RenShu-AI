"""
LLM Provider Factory - LLM提供商工厂
根据提供商名称实例化对应的LangChain类

支持的提供商：
- openai: OpenAI (默认)
- deepseek: 深度求索
- qwen/dashscope/tongyi/alibaba/aliyun: 通义千问
- zhipu/glm/chatglm: 智谱AI
- moonshot/kimi: 月之暗面
- doubao/bytedance/volcengine: 字节豆包
- minimax: MiniMax
- siliconflow: 硅基流动
- ollama: Ollama (本地)
- anthropic/claude: Anthropic Claude
- azure: Azure OpenAI
- google/gemini: Google Gemini
- xai: Grok

使用方式：
    from app.src.core.language_model.llm_provider import get_langchain_llm

    llm = get_langchain_llm(
        provider_name="openai",
        model="gpt-4o-mini",
        api_key="sk-xxx",
        base_url="https://api.openai.com/v1",
        temperature=0.3
    )
"""

from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel


# 提供商名称到LangChain模块的映射
PROVIDER_MODULE_MAP = {
    # OpenAI 系列
    "openai": "app.src.core.language_model.langchain_openai",
    "azure": "app.src.core.language_model.langchain_azure",

    # 国内主流
    "deepseek": "app.src.core.language_model.langchain_deepseek",
    "qwen": "app.src.core.language_model.langchain_qwen",
    "dashscope": "app.src.core.language_model.langchain_qwen",  # 别名
    "tongyi": "app.src.core.language_model.langchain_qwen",  # 别名：通义千问
    "alibaba": "app.src.core.language_model.langchain_qwen",  # 别名：阿里巴巴
    "aliyun": "app.src.core.language_model.langchain_qwen",  # 别名：阿里云
    "zhipu": "app.src.core.language_model.langchain_zhipu",
    "glm": "app.src.core.language_model.langchain_zhipu",  # 别名
    "chatglm": "app.src.core.language_model.langchain_zhipu",  # 别名
    "moonshot": "app.src.core.language_model.langchain_moonshot",
    "kimi": "app.src.core.language_model.langchain_moonshot",  # 别名
    "doubao": "app.src.core.language_model.langchain_doubao",
    "bytedance": "app.src.core.language_model.langchain_doubao",  # 别名
    "volcengine": "app.src.core.language_model.langchain_doubao",  # 别名：火山引擎
    "minimax": "app.src.core.language_model.langchain_minimax",

    # 聚合平台
    "siliconflow": "app.src.core.language_model.langchain_siliconflow",

    # 本地部署
    "ollama": "app.src.core.language_model.langchain_ollama",

    # 国际主流
    "anthropic": "app.src.core.language_model.langchain_anthropic",
    "claude": "app.src.core.language_model.langchain_anthropic",  # 别名
    "google": "app.src.core.language_model.langchain_google",
    "gemini": "app.src.core.language_model.langchain_google",  # 别名
    "grok": "app.src.core.language_model.langchain_grok",
    "xai":"app.src.core.language_model.langchain_grok",
}

# 提供商默认 Base URL
DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "tongyi": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "alibaba": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "aliyun": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "glm": "https://open.bigmodel.cn/api/paas/v4",
    "chatglm": "https://open.bigmodel.cn/api/paas/v4",
    "moonshot": "https://api.moonshot.cn/v1",
    "kimi": "https://api.moonshot.cn/v1",
    "doubao": "https://ark.cn-beijing.volces.com/api/v3",
    "bytedance": "https://ark.cn-beijing.volces.com/api/v3",
    "volcengine": "https://ark.cn-beijing.volces.com/api/v3",
    "minimax": "https://api.minimax.chat/v1",
    "siliconflow": "https://api.siliconflow.cn/v1",
    "ollama": "http://localhost:11434",
    "anthropic": "https://api.anthropic.com",
    "claude": "https://api.anthropic.com",
    "grok": "https://api.grok.com/openai/v1",
    "xai": "https://api.grok.com/openai/v1",
}


def get_supported_providers() -> list[str]:
    """获取所有支持的提供商列表（不含别名）"""
    primary_providers = [
        "openai", "azure", "deepseek", "qwen", "zhipu", "moonshot",
        "doubao", "minimax", "siliconflow", "ollama",
        "anthropic", "google", "grok","xai"
    ]
    return primary_providers


def get_default_base_url(provider_name: str) -> Optional[str]:
    """获取提供商的默认 Base URL"""
    return DEFAULT_BASE_URLS.get(provider_name.lower())


def _get_langchain_class(provider_name: str) -> type:
    """
    根据提供商名称动态导入并返回对应的LangChain Chat类

    Args:
        provider_name: 提供商名称

    Returns:
        对应的LangChain Chat类

    Raises:
        ValueError: 不支持的提供商
    """
    provider_key = provider_name.lower()

    if provider_key not in PROVIDER_MODULE_MAP:
        supported = get_supported_providers()
        raise ValueError(
            f"不支持的LLM提供商: {provider_name}. "
            f"支持的提供商: {supported}"
        )

    module_path = PROVIDER_MODULE_MAP[provider_key]

    # 动态导入模块
    import importlib
    module = importlib.import_module(module_path)

    # 获取 Chat 类
    return module.Chat




def get_langchain_llm(
    provider_name: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: str = "",
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: Optional[int] = None,
    enable_thinking: bool = False,
    enable_web_search: bool = False,
    **kwargs
) -> BaseChatModel:
    """
    根据提供商名称获取LangChain LLM实例

    Args:
        provider_name: 提供商名称，默认 "openai"
        model: 模型名称，默认 "gpt-4o-mini"
        api_key: API Key
        base_url: Base URL (可选，不提供则使用默认值)
        temperature: 温度参数，默认0.7
        top_p: Top P参数，默认1.0
        max_tokens: 最大token数 (可选)
        enable_thinking: 是否启用思考模式
        enable_web_search: 是否启用 Web 搜索工具（自动绑定 Tavily）
        **kwargs: 其他LangChain参数

    Thinking Mode 支持情况：
        - OpenAI (GPT-5+): reasoning={"effort": "medium", "summary": "auto"}
        - DeepSeek: extra_body={"thinking": {"type": "enabled"}}
        - Qwen/DashScope: extra_body={"enable_thinking": True}
        - Anthropic Claude (3.7+): thinking={"type": "enabled"}
        - Google Gemini (3.0+): generation_config={"thinking_config": {"thinking_level": "high"}}
        - 其他提供商: 暂不支持

    Returns:
        BaseChatModel: LangChain聊天模型实例

    Raises:
        ValueError: 不支持的提供商
    """
    provider_key = provider_name.lower()

    # 如果没有提供 base_url，使用默认值
    if not base_url:
        base_url = get_default_base_url(provider_key)

    # 获取对应的 LangChain 类（根据 enable_thinking 选择合适的实现）
    ChatClass = _get_langchain_class(provider_name)



    # 构建参数
    llm_kwargs = {
        "model": model,
        "temperature": temperature,
        # "top_p": top_p,
        **kwargs
    }

    if max_tokens:
        llm_kwargs["max_tokens"] = max_tokens

    # 根据不同提供商设置不同的参数
    if provider_key == "ollama":
        # Ollama 使用 base_url 参数，不需要 api_key
        if base_url:
            llm_kwargs["base_url"] = base_url

    elif provider_key == "azure":
        # Azure 需要特殊参数
        llm_kwargs["api_key"] = api_key
        if base_url:
            llm_kwargs["azure_endpoint"] = base_url
        # Azure 使用 deployment_name 而不是 model
        if "azure_deployment" not in kwargs:
            llm_kwargs["azure_deployment"] = model
            
    elif provider_key in ("anthropic", "claude"):
        # Anthropic Claude: 使用 thinking 参数（Claude 3.7+）
        llm_kwargs["anthropic_api_key"] = api_key
        if enable_thinking:
            # Claude extended thinking mode
            llm_kwargs["thinking"] = {"type": "enabled"}
            
    elif provider_key in ("google", "gemini"):
        # Google Gemini: 使用 thinking_config 参数
        llm_kwargs["google_api_key"] = api_key
        if enable_thinking:
            # Gemini 3: thinking_level (low/medium/high)
            # Gemini 2.5: thinking_budget (token count or -1 for dynamic)
            llm_kwargs["model_kwargs"]["generation_config"] = {
                "thinking_config": {
                    "thinking_level": "high"  # Gemini 3 默认使用 high
                }
            }
            
    elif provider_key == "grok":
        # Groq 使用 groq_api_key
        llm_kwargs["groq_api_key"] = api_key
        if base_url:
           llm_kwargs["base_url"]=base_url 
        
    elif provider_key == "openai":
        # OpenAI: 使用 reasoning 参数（GPT-5+）
        llm_kwargs["api_key"] = api_key
        if base_url:
            llm_kwargs["base_url"] = base_url
        if enable_thinking:
            # OpenAI reasoning mode (GPT-5+)
            llm_kwargs["reasoning"] = {
                "effort": "medium",  # low, medium, high, xhigh
                "summary": "auto"    # detailed, auto, or None
            }
            
    elif provider_key == "deepseek":
        # DeepSeek: 使用 extra_body 传递 thinking 参数
        llm_kwargs["api_key"] = api_key
        if base_url:
            llm_kwargs["base_url"] = base_url
        if enable_thinking:
            # DeepSeek thinking mode
            llm_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            
    elif provider_key in ("qwen", "dashscope", "tongyi", "alibaba", "aliyun"):
        # 通义千问: 使用 extra_body 传递 enable_thinking
        llm_kwargs["api_key"] = api_key
        if base_url:
            llm_kwargs["base_url"] = base_url
        if enable_thinking:
            # Qwen/DashScope thinking mode
            llm_kwargs["extra_body"] = {"enable_thinking": True}
    
    elif provider_key in ("zhipu", "glm", "chatglm"):
        # 智谱 GLM: 使用自定义包装器支持 thinking mode
        llm_kwargs["api_key"] = api_key
        if base_url:
            llm_kwargs["base_url"] = base_url
        if enable_thinking:
            # 智谱 GLM thinking mode
            llm_kwargs["enable_thinking"] = True
    
    elif provider_key in ("moonshot", "kimi"):
        # Moonshot (Kimi): 支持 thinking mode
        llm_kwargs["api_key"] = api_key
        if base_url:
            llm_kwargs["base_url"] = base_url
        if enable_thinking:
            llm_kwargs["enable_thinking"] = True
    
    elif provider_key in ("doubao", "bytedance", "volcengine"):
        # 豆包: 支持 thinking mode
        llm_kwargs["api_key"] = api_key
        if base_url:
            llm_kwargs["base_url"] = base_url
        if enable_thinking:
            llm_kwargs["enable_thinking"] = True
    
    elif provider_key == "minimax":
        # MiniMax: 支持 thinking mode，使用特殊的 reasoning_split 参数
        llm_kwargs["api_key"] = api_key
        if base_url:
            llm_kwargs["base_url"] = base_url
        if enable_thinking:
            llm_kwargs["enable_thinking"] = True

    else:
        # 其他 OpenAI 兼容类 (siliconflow)
        llm_kwargs["api_key"] = api_key
        if base_url:
            llm_kwargs["base_url"] = base_url
        
        

    # 创建 LLM 实例
    llm = ChatClass(**llm_kwargs)



    return llm


def get_intent_classifier_llm(
    provider_name: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: str = "",
    base_url: Optional[str] = None,
    top_p:Optional[str|int|float]=None,
    temperature:Optional[str|int|float]=None
) -> BaseChatModel:
    """
    获取用于意图分类的LLM实例

    意图分类使用较低的temperature确保结果稳定

    Args:
        provider_name: 提供商名称，默认 "openai"
        model: 模型名称，默认 "gpt-4o-mini"
        api_key: API Key
        base_url: Base URL (可选)

    Returns:
        BaseChatModel: 配置好的LLM实例
    """
    return get_langchain_llm(
        provider_name=provider_name,
        model=model,
        api_key=api_key,
        base_url=base_url,
        top_p=top_p,
        temperature=temperature,  # 低温度确保分类稳定
    )
