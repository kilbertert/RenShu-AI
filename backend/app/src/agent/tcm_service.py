"""
TCM Agent Service
中医智能体服务

提供与FastAPI集成的服务接口，支持全流程状态流式传输
"""

import uuid
import json
from typing import AsyncGenerator, Optional, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command, Overwrite

from .tcm_builder import build_tcm_graph, new_thread_id
from .tcm_states import TCMInputState, TCMOutputState, LLMConfig
from app.src.schema.attachment_schema import AttachmentContext
from app.src.schema.chat_schema import StreamMessageType, NODE_DISPLAY_REGISTRY
from app.src.response.exception.exceptions import AuthorizationException


def _extract_stream_text(content: Any) -> str:
    """从 OpenAI/Anthropic 流式内容中提取可展示文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


_INTERNAL_LLM_NODES = {
    "analyze_and_route_query",
    "collect_info",
    "analyze_follow_up",
    "assess_complexity",
    # 辨证节点调用的是结构化 JSON 生成；患者版回答统一从最终 state.answer 输出。
    "simple_diagnosis",
    "moderate_diagnosis",
    "complex_diagnosis",
    "plan_queries",
    "execute_query",
    "synthesize_diagnosis",
}


def _is_user_visible_llm_event(event: dict[str, Any]) -> bool:
    """内部分类/结构化推理不得作为最终回答流给前端。"""
    node_name = event.get("metadata", {}).get("langgraph_node")
    tags = set(event.get("tags") or [])
    if "internal_structured_diagnosis" in tags:
        return False
    return node_name not in _INTERNAL_LLM_NODES


def _extract_query_type(router_info: Any) -> Optional[str]:
    """兼容字典与 Pydantic TCMRouter 的 query_type 读取。"""
    if isinstance(router_info, dict):
        value = router_info.get("query_type")
    else:
        value = getattr(router_info, "query_type", None)
    return str(value) if value else None


def _extract_unanswered_state_error(state: Any) -> Optional[str]:
    """图执行失败且没有患者答案时，返回可显式透传的错误。"""
    values = getattr(state, "values", {}) or {}
    error = values.get("error")
    answer = _extract_stream_text(values.get("answer", ""))
    return str(error) if error and not answer else None


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

    async def assert_thread_binding(
        self,
        thread_id: str,
        user_id: str,
        conversation_id: str,
    ) -> None:
        """Checkpoint 中已有身份时必须与当前认证会话一致。"""
        state = await self.graph.aget_state(self.get_thread_config(thread_id))
        values = state.values if state else {}
        stored_user_id = values.get("user_id")
        stored_conversation_id = values.get("conversation_id")
        if stored_user_id and str(stored_user_id) != str(user_id):
            raise AuthorizationException("LangGraph 线程不属于当前用户")
        if stored_conversation_id and str(stored_conversation_id) != str(conversation_id):
            raise AuthorizationException("LangGraph 线程不属于当前会话")

    async def _prepare_new_turn(
        self,
        thread_id: str,
        user_id: str,
        conversation_id: str,
        llm_config: Optional[LLMConfig] = None,
    ) -> None:
        """清空上一轮临时输出，并以本轮服务端配置覆盖旧 checkpoint。"""
        config = self.get_thread_config(thread_id)
        await self.assert_thread_binding(thread_id, user_id, conversation_id)
        state = await self.graph.aget_state(config)
        if not state or not state.values:
            return
        updates = {
                "router": None,
                "diagnose_stage": None,
                "syndrome_result": None,
                "diagnosis_result": None,
                "herbs": Overwrite([]),
                "prescriptions": Overwrite([]),
                "classics": Overwrite([]),
                "cases": Overwrite([]),
                "tongue_analysis": None,
                "report_analysis": None,
                "compatibility_check": None,
                "steps": Overwrite([]),
                "cypher_queries": Overwrite([]),
                "answer": "",
                "error": None,
                "jump_to": None,
                "should_seek_doctor": False,
            }
        if llm_config is not None:
            updates["llm_config"] = llm_config
        await self.graph.aupdate_state(config, updates)

    async def _append_checkpoint_messages(
        self,
        config: dict,
        messages: list[Any],
        state: Any = None,
    ) -> None:
        """把中断问题、resume 回答和最终回复补入父图消息历史。"""
        if not messages:
            return
        if state is None:
            state = await self.graph.aget_state(config)
        existing = list((getattr(state, "values", {}) or {}).get("messages", []))
        pending: list[Any] = []
        for message in messages:
            previous = pending[-1] if pending else (existing[-1] if existing else None)
            if (
                previous is not None
                and getattr(previous, "type", None) == getattr(message, "type", None)
                and getattr(previous, "content", None) == getattr(message, "content", None)
            ):
                continue
            pending.append(message)
        if pending:
            await self.graph.aupdate_state(config, {"messages": pending})

    async def chat_with_tcm_agent(
        self,
        message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        user_profile: Optional[dict] = None,
        thread_id: Optional[str] = None,
        attachments: Optional[list[AttachmentContext | dict]] = None,
        tongue_analysis: Optional[dict] = None,
        report_analysis: Optional[dict] = None,
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
        conversation_id = conversation_id or str(uuid.uuid4())

        # 构建 LLM 配置
        llm_config = None
        if provider_name and model_name:
            llm_config = LLMConfig(
                provider_name=provider_name,
                model_name=model_name,
                api_key_encrypted=api_key or "",
                base_url=base_url,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        await self._prepare_new_turn(
            thread_id,
            user_id,
            conversation_id,
            llm_config=llm_config,
        )

        input_state = TCMInputState(
            messages=[HumanMessage(content=message)],
            user_id=user_id,
            conversation_id=conversation_id,
            attachments=attachments or [],
            tongue_analysis=tongue_analysis,
            report_analysis=report_analysis,
            user_profile=user_profile or {},
            llm_config=llm_config,
        )

        result = await self.graph.ainvoke(input_state, config)

        query_type = _extract_query_type(result.get("router")) or "tcm-chat"
        return TCMOutputState(
            answer=result.get("answer", ""),
            query_type=query_type,
            syndrome_result=result.get("syndrome_result"),
            diagnosis_result=result.get("diagnosis_result"),
            herbs=result.get("herbs", []),
            prescriptions=result.get("prescriptions", []),
            classics=result.get("classics", []),
            cases=result.get("cases", []),
            tongue_analysis=result.get("tongue_analysis"),
            report_analysis=result.get("report_analysis"),
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
        attachments: Optional[list[AttachmentContext | dict]] = None,
        tongue_analysis: Optional[dict] = None,
        report_analysis: Optional[dict] = None,
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
        conversation_id = conversation_id or str(uuid.uuid4())

        # 构建 LLM 配置
        llm_config = None
        if provider_name and model_name:
            llm_config = LLMConfig(
                provider_name=provider_name,
                model_name=model_name,
                api_key_encrypted=api_key or "",
                base_url=base_url,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )
        await self._prepare_new_turn(
            thread_id,
            user_id,
            conversation_id,
            llm_config=llm_config,
        )

        input_state = TCMInputState(
            messages=[HumanMessage(content=message)],
            user_id=user_id,
            conversation_id=conversation_id,
            attachments=attachments or [],
            tongue_analysis=tongue_analysis,
            report_analysis=report_analysis,
            user_profile=user_profile or {},
            llm_config=llm_config,
        )

        # 记录已处理的节点和状态
        processed_nodes: set[str] = set()
        query_type = "tcm-chat"
        executed_steps: list[str] = []
        streamed_content = False
        streamed_chunks: list[str] = []

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
                    if not _is_user_visible_llm_event(event):
                        continue
                    chunk = event_data.get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        content = _extract_stream_text(chunk.content)
                        if content:
                            streamed_content = True
                            streamed_chunks.append(content)
                            yield json.dumps({
                                "type": StreamMessageType.CONTENT.value,
                                "content": content,
                            }, ensure_ascii=False)

                # === 3. 提取路由信息（从 on_chain_end 中获取 query_type）===
                elif event_kind == "on_chain_end":
                    output = event_data.get("output", {})
                    if isinstance(output, dict) and "router" in output and output["router"]:
                        routed_type = _extract_query_type(output["router"])
                        if routed_type:
                            query_type = routed_type

            # 检查是否被 interrupt 暂停
            state = await self.graph.aget_state(config)
            if state:
                routed_type = _extract_query_type(state.values.get("router"))
                if routed_type:
                    query_type = routed_type
            if state and state.tasks:
                for task in state.tasks:
                    if task.interrupts:
                        interrupt_value = task.interrupts[0].value
                        question = interrupt_value.get("question", "")
                        if question:
                            await self._append_checkpoint_messages(
                                config,
                                [AIMessage(content=question)],
                                state,
                            )
                        yield json.dumps({
                            "type": "interrupt",
                            "question": question,
                            "action": interrupt_value.get("action", ""),
                            "thread_id": thread_id,
                        }, ensure_ascii=False)
                        return  # 不发 done，前端知道需要等待用户输入

            state_error = _extract_unanswered_state_error(state)
            if state_error:
                yield json.dumps({
                    "type": StreamMessageType.ERROR.value,
                    "content": state_error,
                }, ensure_ascii=False)
                return

            final_answer = _extract_stream_text(state.values.get("answer", "")) if state else ""
            final_answer = final_answer or "".join(streamed_chunks)
            if not streamed_content and state:
                if final_answer:
                    yield json.dumps({
                        "type": StreamMessageType.CONTENT.value,
                        "content": final_answer,
                    }, ensure_ascii=False)

            if final_answer:
                await self._append_checkpoint_messages(
                    config,
                    [AIMessage(content=final_answer)],
                    state,
                )

            if state and state.values.get("diagnosis_result"):
                yield json.dumps({
                    "type": "diagnosis_result",
                    "data": state.values["diagnosis_result"],
                }, ensure_ascii=False)

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

    async def resume_stream(
        self,
        thread_id: str,
        user_answer: str,
        user_id: str,
        conversation_id: str,
        attachments: Optional[list[AttachmentContext | dict]] = None,
        tongue_analysis: Optional[dict] = None,
        report_analysis: Optional[dict] = None,
        llm_config: Optional[LLMConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """用户回答追问后，恢复图执行

        Args:
            thread_id: LangGraph 线程ID
            user_answer: 用户追问回答

        Yields:
            str: JSON 格式的流式消息
        """
        config = self.get_thread_config(thread_id)
        await self.assert_thread_binding(thread_id, user_id, conversation_id)
        effective_answer = user_answer.strip() or (
            "我已上传舌像，请结合舌像继续问诊。"
            if tongue_analysis
            else "我已上传医疗报告，请结合报告继续问诊。"
            if report_analysis
            else "请继续问诊。"
        )

        # 父图状态与暂停中的诊断子图都要收到本轮安全附件语义。
        state_update: dict[str, Any] = {}
        if llm_config is not None:
            state_update["llm_config"] = llm_config
        if attachments:
            state_update["attachments"] = attachments
        if tongue_analysis:
            state_update["tongue_analysis"] = tongue_analysis
        if report_analysis:
            state_update["report_analysis"] = report_analysis
        if state_update:
            await self.graph.aupdate_state(config, state_update)

        resume_value: Any = effective_answer
        if attachments or tongue_analysis or report_analysis or llm_config:
            resume_value = {
                "text": effective_answer,
                "attachments": [
                    item.model_dump(mode="json")
                    if isinstance(item, AttachmentContext) else item
                    for item in (attachments or [])
                ],
                "tongue_analysis": tongue_analysis,
                "llm_config": (
                    llm_config.model_dump(mode="json")
                    if llm_config is not None else None
                ),
            }
            if report_analysis:
                resume_value["report_analysis"] = report_analysis
        processed_nodes: set[str] = set()
        query_type = "tcm-chat"
        executed_steps: list[str] = []
        streamed_content = False
        streamed_chunks: list[str] = []

        try:
            async for event in self.graph.astream_events(
                Command(resume=resume_value), config, version="v2"
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
                    if not _is_user_visible_llm_event(event):
                        continue
                    chunk = event_data.get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        content = _extract_stream_text(chunk.content)
                        if content:
                            streamed_content = True
                            streamed_chunks.append(content)
                            yield json.dumps({
                                "type": StreamMessageType.CONTENT.value,
                                "content": content,
                            }, ensure_ascii=False)

                # 提取路由信息
                elif event_kind == "on_chain_end":
                    output = event_data.get("output", {})
                    if isinstance(output, dict) and "router" in output and output["router"]:
                        routed_type = _extract_query_type(output["router"])
                        if routed_type:
                            query_type = routed_type

            # 检查是否再次 interrupt（多轮追问）
            state = await self.graph.aget_state(config)
            if state:
                routed_type = _extract_query_type(state.values.get("router"))
                if routed_type:
                    query_type = routed_type
            if state and state.tasks:
                for task in state.tasks:
                    if task.interrupts:
                        interrupt_value = task.interrupts[0].value
                        question = interrupt_value.get("question", "")
                        checkpoint_messages = [HumanMessage(content=effective_answer)]
                        if question:
                            checkpoint_messages.append(AIMessage(content=question))
                        await self._append_checkpoint_messages(
                            config,
                            checkpoint_messages,
                            state,
                        )
                        yield json.dumps({
                            "type": "interrupt",
                            "question": question,
                            "action": interrupt_value.get("action", ""),
                            "thread_id": thread_id,
                        }, ensure_ascii=False)
                        return

            state_error = _extract_unanswered_state_error(state)
            if state_error:
                yield json.dumps({
                    "type": StreamMessageType.ERROR.value,
                    "content": state_error,
                }, ensure_ascii=False)
                return

            final_answer = _extract_stream_text(state.values.get("answer", "")) if state else ""
            final_answer = final_answer or "".join(streamed_chunks)
            if not streamed_content and state:
                if final_answer:
                    yield json.dumps({
                        "type": StreamMessageType.CONTENT.value,
                        "content": final_answer,
                    }, ensure_ascii=False)

            checkpoint_messages = [HumanMessage(content=effective_answer)]
            if final_answer:
                checkpoint_messages.append(AIMessage(content=final_answer))
            await self._append_checkpoint_messages(config, checkpoint_messages, state)

            if state and state.values.get("diagnosis_result"):
                yield json.dumps({
                    "type": "diagnosis_result",
                    "data": state.values["diagnosis_result"],
                }, ensure_ascii=False)

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
