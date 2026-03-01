"""Google Gemini 聊天模型"""

from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
class Chat(ChatGoogleGenerativeAI,BaseChatOpenAI):
    """Google Gemini聊天模型

    支持模型: gemini-pro, gemini-1.5-pro, gemini-1.5-flash 等

    """
    pass
