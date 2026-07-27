"""Qwen OpenAI-compatible LangChain adapter.

DashScope/MaaS compatible-mode endpoints implement the OpenAI chat-completions
contract, including ``image_url`` content blocks.  Keep this adapter on the
installed ``langchain-openai`` dependency instead of the disabled
``langchain_qwq`` optional package.
"""

from langchain_openai import ChatOpenAI


class Chat(ChatOpenAI):
    """Qwen chat and multimodal models exposed through an OpenAI-compatible API."""
