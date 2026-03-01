"""硅基流动 (SiliconFlow) 聊天模型"""
import tiktoken

from typing_extensions import Tuple

from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_xinference.chat_models import ChatXinference

class Chat(ChatXinference,BaseChatOpenAI):

    """硅基流动聊天模型

    支持多种开源模型: Qwen, DeepSeek, GLM, Yi 等
    API兼容OpenAI格式
    默认base_url: https://api.siliconflow.cn/v1
    """

    def _get_encoding_model(self) -> Tuple[str, tiktoken.Encoding]:
        model = "gpt-3.5-turbo"
        return model, tiktoken.encoding_for_model(model)
