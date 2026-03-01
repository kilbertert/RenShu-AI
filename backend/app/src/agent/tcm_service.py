"""
TCM Agent Service
中医智能体服务

提供与FastAPI集成的服务接口，支持全流程状态流式传输
"""

import uuid
import json
from typing import AsyncGenerator, Optional, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

from .tcm_builder import build_tcm_graph, new_thread_id
from .tcm_states import TCMInputState, TCMOutputState, LLMConfig
from app.src.schema.chat_schema import StreamMessageType, NODE_DISPLAY_REGISTRY


class TCMAgentService:
    """中医智能体服务"""

    def __init__(self):
        self._graph = None
        self._thread_configs = {}  # 存储线程配置

    @property
    def graph(self):
        """懒加载图实例"""
        if self._graph is None:
            self._graph = build_tcm_graph()
        return self._graph

    def get_thread_config(self, thread_id: str) -> dict:
        """获取线程配置"""
        return {"configurable": {"thread_id": thread_id}}

    async def chat_with_tcm_agent(
        self,
        message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        user_profile: Optional[dict] = None,
        thread_id: Optional[str] = None,
        # 新增：模型配置参数
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 2000,
    ) -> TCMOutputState:
        """
        使用TCM多智能体架构处理用户消息

        Args:
            message: 用户消息
            user_id: 用户ID
            conversation_id: 会话ID
            user_profile: 用户画像
            thread_id: 线程ID（用于多轮对话）
            provider_name: LLM 提供商名称 (openai/deepseek/ollama)
            model_name: 模型名称
            api_key: API Key
            base_url: API Base URL
            temperature: 温度参数
            top_p: Top-P 采样参数
            max_tokens: 最大 token 数

        Returns:
            TCMOutputState: 输出状态
        """
        thread_id = thread_id or new_thread_id()
        config = self.get_thread_config(thread_id)

        # 构建 LLM 配置
        llm_config = None
        if provider_name and model_name:
            llm_config = LLMConfig(
                provider_name=provider_name,
                model_name=model_name,
                api_key=api_key or "",
                base_url=base_url,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        input_state = TCMInputState(
            messages=[HumanMessage(content=message)],
            user_id=user_id,
            conversation_id=conversation_id or str(uuid.uuid4()),
            user_profile=user_profile or {},
            llm_config=llm_config,
        )

        result = await self.graph.ainvoke(input_state, config)

        return TCMOutputState(
            answer=result.get("answer", ""),
            query_type=result.get("router", {}).get("query_type", "tcm-chat") if result.get("router") else "tcm-chat",
            syndrome_result=result.get("syndrome_result"),
            herbs=result.get("herbs", []),
            prescriptions=result.get("prescriptions", []),
            classics=result.get("classics", []),
            cases=result.get("cases", []),
            tongue_analysis=result.get("tongue_analysis"),
            steps=result.get("steps", []),
            cypher_queries=result.get("cypher_queries", []),
            follow_up_questions=[],
            should_seek_doctor=False,
        )

    async def chat_stream_with_tcm_agent(
        self,
        message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        user_profile: Optional[dict] = None,
        thread_id: Optional[str] = None,
        # 新增：模型配置参数
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 2000,
        enable_thinking: bool = False,  # 是否启用思考过程展示
    ) -> AsyncGenerator[str, None]:
        """
        使用TCM多智能体架构处理用户消息（流式 + 状态输出）

        使用 astream_events API 实现 Token 级流式输出，同时发送中文状态消息。
        子图节点事件通过 config 传播自动捕获（包括诊断子图、养生子图等）。

        流式消息格式：
        - 状态消息: {"type": "意图识别", "content": "正在识别您的意图..."}
        - 内容消息: {"type": "content", "content": "根据您的症状..."}
        - 完成消息: {"type": "done", "query_type": "tcm-diagnose", "steps": [...]}
        - 错误消息: {"type": "error", "content": "错误信息"}

        Yields:
            str: JSON 格式的流式消息
        """
        thread_id = thread_id or new_thread_id()
        config = self.get_thread_config(thread_id)

        # 构建 LLM 配置
        llm_config = None
        if provider_name and model_name:
            llm_config = LLMConfig(
                provider_name=provider_name,
                model_name=model_name,
                api_key=api_key or "",
                base_url=base_url,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )

        input_state = TCMInputState(
            messages=[HumanMessage(content=message)],
            user_id=user_id,
            conversation_id=conversation_id or str(uuid.uuid4()),
            user_profile=user_profile or {},
            llm_config=llm_config,
        )

        # 记录已处理的节点和状态
        processed_nodes: set[str] = set()
        query_type = "tcm-chat"
        executed_steps: list[str] = []

        try:
            # 发送 thread_id 给前端，用于后续 resume
            yield json.dumps({"type": "thread_init", "thread_id": thread_id}, ensure_ascii=False)

            async for event in self.graph.astream_events(input_state, config, version="v2"):
                event_kind = event.get("event")
                event_name = event.get("name", "")
                event_data = event.get("data", {})

                # === 1. 节点开始：从 NODE_DISPLAY_REGISTRY 查找并发送中文状态 ===
                if event_kind == "on_chain_start":
                    if event_name not in processed_nodes and event_name in NODE_DISPLAY_REGISTRY:
                        processed_nodes.add(event_name)
                        display = NODE_DISPLAY_REGISTRY[event_name]
                        yield json.dumps(display, ensure_ascii=False)
                        executed_steps.append(display["type"])

                # === 2. LLM Token 流：逐 token 输出内容 ===
                elif event_kind == "on_chat_model_stream":
                    chunk = event_data.get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield json.dumps({
                            "type": StreamMessageType.CONTENT.value,
                            "content": chunk.content,
                        }, ensure_ascii=False)

                # === 3. 提取路由信息（从 on_chain_end 中获取 query_type）===
                elif event_kind == "on_chain_end":
                    output = event_data.get("output", {})
                    if isinstance(output, dict) and "router" in output and output["router"]:
                        router_info = output["router"]
                        if isinstance(router_info, dict) and "query_type" in router_info:
                            query_type = router_info["query_type"]

            # 检查是否被 interrupt 暂停
            state = await self.graph.aget_state(config)
            if state and state.tasks:
                for task in state.tasks:
                    if task.interrupts:
                        interrupt_value = task.interrupts[0].value
                        yield json.dumps({
                            "type": "interrupt",
                            "question": interrupt_value.get("question", ""),
                            "action": interrupt_value.get("action", ""),
                            "thread_id": thread_id,
                        }, ensure_ascii=False)
                        return  # 不发 done，前端知道需要等待用户输入

            # 发送完成消息
            yield json.dumps({
                "type": StreamMessageType.DONE.value,
                "query_type": query_type,
                "steps": executed_steps,
            }, ensure_ascii=False)

        except Exception as e:
            yield json.dumps({
                "type": StreamMessageType.ERROR.value,
                "content": str(e),
            }, ensure_ascii=False)

    async def resume_stream(self, thread_id: str, user_answer: str) -> AsyncGenerator[str, None]:
        """用户回答追问后，恢复图执行

        Args:
            thread_id: LangGraph 线程ID
            user_answer: 用户追问回答

        Yields:
            str: JSON 格式的流式消息
        """
        config = self.get_thread_config(thread_id)
        processed_nodes: set[str] = set()
        query_type = "tcm-chat"
        executed_steps: list[str] = []

        try:
            async for event in self.graph.astream_events(
                Command(resume=user_answer), config, version="v2"
            ):
                event_kind = event.get("event")
                event_name = event.get("name", "")
                event_data = event.get("data", {})

                # 节点开始：发送中文状态
                if event_kind == "on_chain_start":
                    if event_name not in processed_nodes and event_name in NODE_DISPLAY_REGISTRY:
                        processed_nodes.add(event_name)
                        display = NODE_DISPLAY_REGISTRY[event_name]
                        yield json.dumps(display, ensure_ascii=False)
                        executed_steps.append(display["type"])

                # LLM Token 流
                elif event_kind == "on_chat_model_stream":
                    chunk = event_data.get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield json.dumps({
                            "type": StreamMessageType.CONTENT.value,
                            "content": chunk.content,
                        }, ensure_ascii=False)

                # 提取路由信息
                elif event_kind == "on_chain_end":
                    output = event_data.get("output", {})
                    if isinstance(output, dict) and "router" in output and output["router"]:
                        router_info = output["router"]
                        if isinstance(router_info, dict) and "query_type" in router_info:
                            query_type = router_info["query_type"]

            # 检查是否再次 interrupt（多轮追问）
            state = await self.graph.aget_state(config)
            if state and state.tasks:
                for task in state.tasks:
                    if task.interrupts:
                        interrupt_value = task.interrupts[0].value
                        yield json.dumps({
                            "type": "interrupt",
                            "question": interrupt_value.get("question", ""),
                            "action": interrupt_value.get("action", ""),
                            "thread_id": thread_id,
                        }, ensure_ascii=False)
                        return

            yield json.dumps({
                "type": StreamMessageType.DONE.value,
                "query_type": query_type,
                "steps": executed_steps,
            }, ensure_ascii=False)

        except Exception as e:
            yield json.dumps({
                "type": StreamMessageType.ERROR.value,
                "content": str(e),
            }, ensure_ascii=False)

    async def get_conversation_history(self, thread_id: str) -> list[dict]:
        """
        获取对话历史

        Args:
            thread_id: 线程ID

        Returns:
            list[dict]: 对话历史
        """
        config = self.get_thread_config(thread_id)

        try:
            state = await self.graph.aget_state(config)
            messages = state.values.get("messages", [])

            history = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    history.append({"role": "assistant", "content": msg.content})

            return history
        except Exception:
            return []

    async def clear_conversation(self, thread_id: str) -> bool:
        """
        清除对话历史

        Args:
            thread_id: 线程ID

        Returns:
            bool: 是否成功
        """
        # MemorySaver不支持直接清除，需要创建新线程
        if thread_id in self._thread_configs:
            del self._thread_configs[thread_id]
        return True


# 单例服务实例
_tcm_agent_service: Optional[TCMAgentService] = None


def get_tcm_agent_service() -> TCMAgentService:
    """获取TCM Agent服务单例"""
    global _tcm_agent_service
    if _tcm_agent_service is None:
        _tcm_agent_service = TCMAgentService()
    return _tcm_agent_service
