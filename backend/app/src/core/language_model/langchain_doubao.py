"""字节跳动豆包 (Doubao) 聊天模型"""
from typing import Tuple, Iterator, AsyncIterator, Any, Optional, List

import tiktoken
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import BaseMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from openai import OpenAI, AsyncOpenAI
from pydantic import Field, PrivateAttr


class Chat(BaseChatOpenAI):
    """字节跳动豆包聊天模型

    支持模型: doubao-pro-4k, doubao-pro-32k, doubao-lite-4k 等
    API兼容OpenAI格式
    
    支持 Thinking Mode (需要设置 enable_thinking=True)
    """
    
    model: str = Field(default="doubao-pro-4k")
    api_key: str = Field(default="")
    base_url: str = Field(default="https://ark.cn-beijing.volces.com/api/v3")
    streaming: bool = Field(default=True)
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = Field(default=None)
    enable_thinking: bool = Field(default=False, description="是否启用思考模式")
    _native_client: Optional[OpenAI] = PrivateAttr(default=None)
    _native_async_client: Optional[AsyncOpenAI] = PrivateAttr(default=None)

    def _get_encoding_model(self) -> Tuple[str, tiktoken.Encoding]:
        model = "gpt-3.5-turbo"
        return model, tiktoken.encoding_for_model(model)
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        同步流式生成
        
        如果启用了 thinking mode，使用原生 OpenAI SDK 来支持 extra_body 参数
        否则使用父类的标准实现
        """
        if not self.enable_thinking:
            # 不启用 thinking mode，使用父类的标准实现
            yield from super()._stream(messages, stop, **kwargs)
            return
        
        # 启用 thinking mode，使用原生 OpenAI SDK
        if self._native_client is None:
            self._native_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or "https://ark.cn-beijing.volces.com/api/v3"
            )
        
        # 转换 LangChain 消息格式为 OpenAI 格式
        openai_messages = self._convert_messages_to_openai_format(messages)
        
        # 构建请求参数
        request_params = self._build_request_params(openai_messages)
        
        # 调用原生 OpenAI SDK
        response = self._native_client.chat.completions.create(**request_params)
        
        # 转换响应为 LangChain 格式
        yield from self._convert_response_to_langchain_format(response)
    
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        异步流式生成
        
        如果启用了 thinking mode，使用原生 AsyncOpenAI SDK
        否则使用父类的标准实现
        """
        if not self.enable_thinking:
            # 不启用 thinking mode，使用父类的标准实现
            async for chunk in super()._astream(messages, stop, **kwargs):
                yield chunk
            return
        
        # 启用 thinking mode，使用原生 AsyncOpenAI SDK
        if self._native_async_client is None:
            self._native_async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url or "https://ark.cn-beijing.volces.com/api/v3"
            )
        
        # 转换 LangChain 消息格式为 OpenAI 格式
        openai_messages = self._convert_messages_to_openai_format(messages)
        
        # 构建请求参数
        request_params = self._build_request_params(openai_messages)
        
        # 调用原生 AsyncOpenAI SDK
        response = await self._native_async_client.chat.completions.create(**request_params)
        
        # 转换响应为 LangChain 格式
        async for chunk in self._convert_async_response_to_langchain_format(response):
            yield chunk
    
    def _convert_messages_to_openai_format(self, messages: List[BaseMessage]) -> List[dict]:
        """转换 LangChain 消息格式为 OpenAI 格式"""
        openai_messages = []
        for msg in messages:
            if hasattr(msg, 'type'):
                role = msg.type
                if role == 'human':
                    role = 'user'
                elif role == 'ai':
                    role = 'assistant'
            else:
                role = 'user'
            
            openai_messages.append({
                "role": role,
                "content": msg.content
            })
        return openai_messages
    
    def _build_request_params(self, openai_messages: List[dict]) -> dict:
        """构建请求参数"""
        request_params = {
            "model": self.model,
            "messages": openai_messages,
            "stream": True,
            "temperature": self.temperature,
            "extra_body": {"thinking": {"type": "enabled"}},  # 启用 thinking mode
        }
        
        if self.max_tokens:
            request_params["max_tokens"] = self.max_tokens
        
        return request_params
    
    def _convert_response_to_langchain_format(self, response) -> Iterator[ChatGenerationChunk]:
        """转换同步响应为 LangChain 格式"""
        for chunk in response:
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta
            
            # 构建 LangChain AIMessageChunk
            chunk_content = ""
            additional_kwargs = {}
            
            # 提取思考内容
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                additional_kwargs['reasoning_content'] = delta.reasoning_content
            
            # 提取答案内容
            if hasattr(delta, 'content') and delta.content:
                chunk_content = delta.content
            
            # 创建 LangChain 格式的 chunk
            message_chunk = AIMessageChunk(
                content=chunk_content,
                additional_kwargs=additional_kwargs
            )
            
            # 包装为 ChatGenerationChunk
            yield ChatGenerationChunk(message=message_chunk)
    
    async def _convert_async_response_to_langchain_format(self, response) -> AsyncIterator[ChatGenerationChunk]:
        """转换异步响应为 LangChain 格式"""
        async for chunk in response:
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta
            
            # 构建 LangChain AIMessageChunk
            chunk_content = ""
            additional_kwargs = {}
            
            # 提取思考内容
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                additional_kwargs['reasoning_content'] = delta.reasoning_content
            
            # 提取答案内容
            if hasattr(delta, 'content') and delta.content:
                chunk_content = delta.content
            
            # 创建 LangChain 格式的 chunk
            message_chunk = AIMessageChunk(
                content=chunk_content,
                additional_kwargs=additional_kwargs
            )
            
            # 包装为 ChatGenerationChunk
            yield ChatGenerationChunk(message=message_chunk)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """
        同步非流式生成（invoke 方法会调用此方法）
        
        如果启用了 thinking mode，使用原生 OpenAI SDK
        否则使用父类的标准实现
        """
        if not self.enable_thinking:
            # 不启用 thinking mode，使用父类的标准实现
            return super()._generate(messages, stop, **kwargs)
        
        # 启用 thinking mode，使用原生 OpenAI SDK
        if self._native_client is None:
            self._native_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or "https://ark.cn-beijing.volces.com/api/v3"
            )
        
        # 转换 LangChain 消息格式为 OpenAI 格式
        openai_messages = self._convert_messages_to_openai_format(messages)
        
        # 构建请求参数（非流式）
        request_params = {
            "model": self.model,
            "messages": openai_messages,
            "stream": False,  # 非流式
            "temperature": self.temperature,
            "extra_body": {"thinking": {"type": "enabled"}},
        }
        
        if self.max_tokens:
            request_params["max_tokens"] = self.max_tokens
        
        # 调用原生 OpenAI SDK
        response = self._native_client.chat.completions.create(**request_params)
        
        # 转换响应为 LangChain 格式
        return self._convert_completion_to_langchain_format(response)
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """
        异步非流式生成（ainvoke 方法会调用此方法）
        
        如果启用了 thinking mode，使用原生 AsyncOpenAI SDK
        否则使用父类的标准实现
        """
        if not self.enable_thinking:
            # 不启用 thinking mode，使用父类的标准实现
            return await super()._agenerate(messages, stop, **kwargs)
        
        # 启用 thinking mode，使用原生 AsyncOpenAI SDK
        if self._native_async_client is None:
            self._native_async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url or "https://ark.cn-beijing.volces.com/api/v3"
            )
        
        # 转换 LangChain 消息格式为 OpenAI 格式
        openai_messages = self._convert_messages_to_openai_format(messages)
        
        # 构建请求参数（非流式）
        request_params = {
            "model": self.model,
            "messages": openai_messages,
            "stream": False,  # 非流式
            "temperature": self.temperature,
            "extra_body": {"thinking": {"type": "enabled"}},
        }
        
        if self.max_tokens:
            request_params["max_tokens"] = self.max_tokens
        
        # 调用原生 AsyncOpenAI SDK
        response = await self._native_async_client.chat.completions.create(**request_params)
        
        # 转换响应为 LangChain 格式
        return self._convert_completion_to_langchain_format(response)
    
    def _convert_completion_to_langchain_format(self, response):
        """转换非流式响应为 LangChain 格式"""
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        
        message = response.choices[0].message
        
        # 提取内容
        content = message.content if message.content else ""
        
        # 提取思考内容
        additional_kwargs = {}
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            additional_kwargs['reasoning_content'] = message.reasoning_content
        
        # 创建 LangChain AIMessage
        ai_message = AIMessage(
            content=content,
            additional_kwargs=additional_kwargs
        )
        
        # 包装为 ChatGeneration 和 ChatResult
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

