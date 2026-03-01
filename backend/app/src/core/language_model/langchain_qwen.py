from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_qwq import ChatQwen

class Chat(ChatQwen,BaseChatOpenAI):
    pass
