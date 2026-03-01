"""Azure OpenAI 聊天模型"""
from langchain_openai import AzureChatOpenAI

from langchain_openai.chat_models.base import BaseChatOpenAI


class Chat(AzureChatOpenAI,BaseChatOpenAI):
    """Azure OpenAI聊天模型

    支持模型: gpt-4, gpt-4-turbo, gpt-35-turbo 等
    需要配置 azure_endpoint, api_version, azure_deployment
    """

    pass
