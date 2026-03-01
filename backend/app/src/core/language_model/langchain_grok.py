"""Groq 聊天模型 (高速推理)"""

from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_xai.chat_models import ChatXAI

class Chat(ChatXAI,BaseChatOpenAI):
    """Groq高速推理聊天模型

    支持模型: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768 等
    特点: 极速推理
    """

    pass
