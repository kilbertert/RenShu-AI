"""
TCM Supervisor 多Agent协调器

基于2026年最佳实践的多Agent架构：

1. Supervisor模式：中心协调器负责任务分解和结果合成
2. Direct-Pass机制：避免"电话游戏"问题，子Agent可直接响应
3. 上下文隔离：每个子Agent拥有独立的上下文窗口
4. 明确的Handoff协议：Agent间切换的显式协议

架构：
    User Query → TCMSupervisor → [Specialist Agents] → Response
                            ↓                    ↓
                       Task Decomposition    Direct Pass / Synthesis
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Agent角色枚举"""
    SUPERVISOR = "supervisor"           # 主协调器
    DIAGNOSIS = "diagnosis"             # 诊断Agent
    PRESCRIPTION = "prescription"       # 方剂Agent
    WELLNESS = "wellness"               # 养生Agent
    HERB = "herb"                       # 药材Agent
    GENERAL = "general"                 # 通用咨询Agent


class ResponseType(str, Enum):
    """响应类型"""
    SYNTHESIZED = "synthesized"         # 需要Supervisor合成
    DIRECT_PASS = "direct_pass"         # 直接传递给用户
    PARTIAL = "partial"                 # 部分结果，需要继续
    HANDOFF = "handoff"                 # 移交给其他Agent
    ERROR = "error"                     # 错误响应


@dataclass
class AgentTask:
    """Agent任务定义"""
    task_id: str
    role: AgentRole
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    input_data: Dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    """Agent响应"""
    agent_role: AgentRole
    task_id: str
    response_type: ResponseType
    content: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Direct Pass 相关
    direct_pass_content: Optional[str] = None  # 直接传递给用户的内容
    handoff_target: Optional[AgentRole] = None  # 移交目标


@dataclass
class SupervisorConfig:
    """Supervisor配置"""
    enable_direct_pass: bool = True       # 启用直接传递
    enable_handoff: bool = True           # 启用Agent移交
    max_parallel_agents: int = 3          # 最大并行Agent数
    synthesis_prompt: str = "default"     # 合成提示模板
    fallback_to_general: bool = True      # 降级到通用Agent


class TCMSupervisor:
    """
    TCM 多Agent协调器（Supervisor模式）

    职责：
    1. 任务分解：将用户查询分解为子任务
    2. Agent调度：分配任务给专门的Agent
    3. 结果合成：整合多个Agent的输出
    4. Direct Pass：允许子Agent直接响应
    5. Handoff管理：处理Agent间的移交
    """

    def __init__(
        self,
        config: Optional[SupervisorConfig] = None,
        specialists: Optional[Dict[AgentRole, Callable]] = None
    ):
        """
        初始化Supervisor

        Args:
            config: Supervisor配置
            specialists: 专门Agent的执行函数映射
        """
        self.config = config or SupervisorConfig()
        self.specialists = specialists or {}
        self._active_tasks: Dict[str, AgentTask] = {}
        self._completed_tasks: Dict[str, AgentResponse] = {}

    def register_specialist(
        self,
        role: AgentRole,
        executor: Callable
    ):
        """
        注册专门Agent

        Args:
            role: Agent角色
            executor: 执行函数，签名为 async def executor(task: AgentTask) -> AgentResponse
        """
        self.specialists[role] = executor
        logger.info(f"[TCMSupervisor] 注册专门Agent: {role}")

    async def process(
        self,
        query: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
        memory_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        处理用户查询（Supervisor主入口）

        Args:
            query: 用户查询
            user_id: 用户ID
            context: 对话上下文
            memory_context: 记忆上下文

        Returns:
            处理结果
        """
        logger.info(f"[TCMSupervisor] 处理查询: {query[:50]}...")

        # 1. 分析查询，确定需要的Agent
        tasks = await self._decompose_query(query, user_id, context, memory_context)

        if not tasks:
            # 无需分解，直接由通用Agent处理
            return await self._handle_with_general(query, context)

        # 2. 检查Direct Pass机会
        if len(tasks) == 1 and self.config.enable_direct_pass:
            task = tasks[0]
            response = await self._execute_agent(task, context)
            if response.response_type == ResponseType.DIRECT_PASS:
                # 直接传递，无需Supervisor合成
                return {
                    "type": "direct_pass",
                    "agent": response.agent_role,
                    "content": response.direct_pass_content or response.content,
                    "data": response.data
                }

        # 3. 执行多个Agent（并行或串行）
        responses = await self._execute_tasks(tasks, context)

        # 4. 处理Handoff
        for response in responses:
            if response.response_type == ResponseType.HANDOFF and response.handoff_target:
                return await self._handle_handoff(response, query, context)

        # 5. 合成结果
        synthesized = await self._synthesize_responses(query, responses)

        return {
            "type": "synthesized",
            "content": synthesized["content"],
            "steps": synthesized["steps"],
            "agent_responses": [
                {
                    "agent": r.agent_role,
                    "content": r.content,
                    "confidence": r.confidence
                }
                for r in responses
            ]
        }

    async def _decompose_query(
        self,
        query: str,
        user_id: str,
        context: Optional[Dict[str, Any]],
        memory_context: Optional[Dict[str, Any]]
    ) -> List[AgentTask]:
        """
        分解查询为子任务

        根据查询内容判断需要哪些Agent参与：
        - 诊断相关 → DIAGNOSIS
        - 方剂相关 → PRESCRIPTION
        - 养生相关 → WELLNESS
        - 药材相关 → HERB
        """
        tasks = []
        query_lower = query.lower()

        # 简单规则匹配（实际可用LLM进行意图识别）
        if any(kw in query_lower for kw in ["诊断", "辨证", "症状", "病因"]):
            tasks.append(AgentTask(
                task_id=f"diagnosis_{datetime.now().timestamp()}",
                role=AgentRole.DIAGNOSIS,
                description="分析用户症状并进行中医辨证",
                input_data={"query": query, "symptoms": context.get("symptoms", {})},
                priority=1
            ))

        if any(kw in query_lower for kw in ["方剂", "处方", "药方"]):
            tasks.append(AgentTask(
                task_id=f"prescription_{datetime.now().timestamp()}",
                role=AgentRole.PRESCRIPTION,
                description="推荐合适的方剂",
                input_data={"query": query},
                priority=2
            ))

        if any(kw in query_lower for kw in ["养生", "保健", "调理", "食疗"]):
            tasks.append(AgentTask(
                task_id=f"wellness_{datetime.now().timestamp()}",
                role=AgentRole.WELLNESS,
                description="提供养生建议",
                input_data={"query": query},
                priority=2
            ))

        if any(kw in query_lower for kw in ["药材", "中药", "草药"]):
            tasks.append(AgentTask(
                task_id=f"herb_{datetime.now().timestamp()}",
                role=AgentRole.HERB,
                description="解答药材相关问题",
                input_data={"query": query},
                priority=2
            ))

        return tasks

    async def _execute_tasks(
        self,
        tasks: List[AgentTask],
        context: Optional[Dict[str, Any]]
    ) -> List[AgentResponse]:
        """
        执行多个Agent任务

        根据依赖关系决定并行或串行执行
        """
        responses = []

        # 简单实现：全部并行（不考虑依赖）
        import asyncio

        async def execute_single(task: AgentTask) -> AgentResponse:
            return await self._execute_agent(task, context)

        # 并行执行，限制最大并发数
        semaphore = asyncio.Semaphore(self.config.max_parallel_agents)

        async def bounded_execute(task: AgentTask) -> AgentResponse:
            async with semaphore:
                return await execute_single(task)

        results = await asyncio.gather(
            *[bounded_execute(task) for task in tasks],
            return_exceptions=True
        )

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"[TCMSupervisor] Agent执行失败: {result}")
            elif isinstance(result, AgentResponse):
                responses.append(result)

        return responses

    async def _execute_agent(
        self,
        task: AgentTask,
        context: Optional[Dict[str, Any]]
    ) -> AgentResponse:
        """
        执行单个Agent

        实现上下文隔离：每个Agent接收独立的上下文
        """
        specialist = self.specialists.get(task.role)

        if specialist is None:
            # Agent不存在，返回错误
            return AgentResponse(
                agent_role=task.role,
                task_id=task.task_id,
                response_type=ResponseType.ERROR,
                content=f"专门Agent {task.role} 未注册",
                confidence=0.0
            )

        try:
            # 创建隔离的上下文（包含任务所需的最小信息）
            isolated_context = {
                "task": task.description,
                "input": task.input_data,
                "user_context": context.get("user_profile", {}) if context else {},
                "memory_context": context.get("memory_context", {}) if context else {},
            }

            # 调用专门Agent（传入隔离的上下文）
            response = await specialist(task, isolated_context)

            # 确保返回的是AgentResponse
            if not isinstance(response, AgentResponse):
                response = AgentResponse(
                    agent_role=task.role,
                    task_id=task.task_id,
                    response_type=ResponseType.SYNTHESIZED,
                    content=str(response),
                    confidence=0.5
                )

            logger.info(
                f"[TCMSupervisor] Agent {task.role} 完成, "
                f"类型: {response.response_type}, "
                f"置信度: {response.confidence}"
            )

            return response

        except Exception as e:
            logger.error(f"[TCMSupervisor] Agent {task.role} 执行异常: {e}")
            return AgentResponse(
                agent_role=task.role,
                task_id=task.task_id,
                response_type=ResponseType.ERROR,
                content=f"执行失败: {str(e)}",
                confidence=0.0
            )

    async def _synthesize_responses(
        self,
        query: str,
        responses: List[AgentResponse]
    ) -> Dict[str, Any]:
        """
        合成多个Agent的响应

        Args:
            query: 原始查询
            responses: Agent响应列表

        Returns:
            合成结果
        """
        if not responses:
            return {
                "content": "抱歉，我无法处理您的问题。",
                "steps": []
            }

        if len(responses) == 1:
            response = responses[0]
            return {
                "content": response.content,
                "steps": [f"{response.agent_role}: {response.content[:50]}..."]
            }

        # 多个响应：需要合成
        # 按优先级排序
        sorted_responses = sorted(responses, key=lambda r: r.priority, reverse=True)

        # 构建合成内容
        synthesized_parts = []
        steps = []

        for response in sorted_responses:
            if response.content:
                synthesized_parts.append(f"【{response.agent_role}】\n{response.content}")
                steps.append(f"{response.agent_role}")

        synthesized_content = "\n\n".join(synthesized_parts)

        return {
            "content": synthesized_content,
            "steps": steps
        }

    async def _handle_handoff(
        self,
        response: AgentResponse,
        original_query: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        处理Agent移交

        当Agent认为自己不适合处理时，移交给其他Agent
        """
        target_role = response.handoff_target
        if not target_role:
            return {"type": "error", "content": "移交目标未指定"}

        logger.info(f"[TCMSupervisor] Agent {response.agent_role} 移交给 {target_role}")

        # 创建新任务
        new_task = AgentTask(
            task_id=f"handoff_{datetime.now().timestamp()}",
            role=target_role,
            description=f"从{response.agent_role}移交的任务",
            input_data={
                "original_query": original_query,
                "previous_response": response.content,
                "context": response.data
            }
        )

        # 执行目标Agent
        new_response = await self._execute_agent(new_task, context)

        return {
            "type": "handoff",
            "from": response.agent_role,
            "to": target_role,
            "content": new_response.content,
            "handoff_reason": response.data.get("handoff_reason", "")
        }

    async def _handle_with_general(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        由通用Agent处理
        """
        if not self.config.fallback_to_general:
            return {"type": "error", "content": "无法处理此查询"}

        specialist = self.specialists.get(AgentRole.GENERAL)
        if not specialist:
            return {"type": "error", "content": "通用Agent未注册"}

        task = AgentTask(
            task_id=f"general_{datetime.now().timestamp()}",
            role=AgentRole.GENERAL,
            description="通用中医咨询",
            input_data={"query": query}
        )

        response = await specialist(task, context or {})

        return {
            "type": "general",
            "content": response.content,
            "data": response.data
        }


def create_supervisor(
    enable_direct_pass: bool = True,
    enable_handoff: bool = True,
    max_parallel: int = 3
) -> TCMSupervisor:
    """
    创建Supervisor实例的便捷函数
    """
    config = SupervisorConfig(
        enable_direct_pass=enable_direct_pass,
        enable_handoff=enable_handoff,
        max_parallel_agents=max_parallel
    )
    return TCMSupervisor(config)


# Direct Pass 工具函数

def direct_pass_response(content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    创建Direct Pass响应

    子Agent使用此函数返回需要直接传递给用户的响应，
    避免Supervisor重写导致的信息丢失。

    Args:
        content: 要传递的内容
        metadata: 附加元数据

    Returns:
        Direct Pass响应字典
    """
    return {
        "response_type": ResponseType.DIRECT_PASS,
        "direct_pass_content": content,
        "metadata": metadata or {}
    }


def handoff_to(
    target_role: AgentRole,
    reason: str,
    context: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    创建Handoff请求

    子Agent使用此函数请求移交任务给其他Agent。

    Args:
        target_role: 目标Agent角色
        reason: 移交原因
        context: 传递给目标Agent的上下文

    Returns:
        Handoff响应字典
    """
    return {
        "response_type": ResponseType.HANDOFF,
        "handoff_target": target_role,
        "handoff_reason": reason,
        "data": context or {}
    }
