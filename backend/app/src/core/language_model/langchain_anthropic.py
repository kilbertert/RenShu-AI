"""Anthropic (Claude) 聊天模型"""
from langchain_anthropic.chat_models import ChatAnthropic


from langchain_openai.chat_models.base import BaseChatOpenAI

class Chat(ChatAnthropic,BaseChatOpenAI):
    """Anthropic Claude聊天模型

    支持模型: claude-3-opus, claude-3-sonnet, claude-3-haiku 等
    """





