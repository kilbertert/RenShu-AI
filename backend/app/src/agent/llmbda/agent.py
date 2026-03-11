"""
LLMbda Agent 组合

提供形式化的 Agent 组合语义
支持串行、并行、条件等组合模式
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from .core import (
    LLMbdaExpr, LLMbdaType, BaseType, AnyType,
    Variable, Abstraction, Application, Sequence,
    EvaluationContext, LLMbdaEvaluator,
)
from .safety import SecurityLevel, FlowPolicy, InformationFlowAnalyzer


logger = logging.getLogger(__name__)


class ComposeMode(str, Enum):
    """Agent组合模式"""
    SEQUENTIAL = "sequential"       # 串行执行
    PARALLEL = "parallel"          # 并行执行
    CONDITIONAL = "conditional"    # 条件选择
    LOOP = "loop"                  # 循环执行
    PIPELINE = "pipeline"          # 流水线


@dataclass
class AgentDefinition:
    """
    Agent 定义

    使用 LLMbda 表达式定义 Agent 行为
    """
    name: str                              # Agent 名称
    input_type: LLMbdaType                 # 输入类型
    output_type: LLMbdaType                # 输出类型
    behavior: LLMbdaExpr                   # 行为表达式
    description: Optional[str] = None      # 描述
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate(
        self,
        input_value: Any,
        context: EvaluationContext
    ) -> Any:
        """执行 Agent"""
        from .core import Variable, Application

        # 创建输入变量
        input_var = Variable("input")
        context = context.with_binding("input", input_value, self.input_type)

        # 求值行为表达式
        return self.behavior.evaluate(context)

    def get_function_type(self) -> LLMbdaType:
        """获取函数类型"""
        from .core import FunctionType
        return FunctionType(self.input_type, self.output_type)


@dataclass
class AgentComposition:
    """
    Agent 组合

    使用 LLMbda 表达式组合多个 Agent
    """
    name: str
    mode: ComposeMode
    agents: List[AgentDefinition]
    flow_policy: Optional[FlowPolicy] = None

    def compose(self) -> AgentDefinition:
        """
        组合 Agent

        Returns:
            组合后的 Agent 定义
        """
        if self.mode == ComposeMode.SEQUENTIAL:
            return self._compose_sequential()
        elif self.mode == ComposeMode.PARALLEL:
            return self._compose_parallel()
        elif self.mode == ComposeMode.CONDITIONAL:
            return self._compose_conditional()
        elif self.mode == ComposeMode.PIPELINE:
            return self._compose_pipeline()
        else:
            raise ValueError(f"Unsupported compose mode: {self.mode}")

    def _compose_sequential(self) -> AgentDefinition:
        """串行组合: agent1 >> agent2 >> agent3"""
        from .core import Sequence, Variable, Application

        if len(self.agents) < 2:
            raise ValueError("Sequential composition requires at least 2 agents")

        # 构建序列表达式
        input_var = Variable("input")
        current_expr = self.agents[0].behavior

        for agent in self.agents[1:]:
            # 创建应用: agent(prev_result)
            current_expr = Sequence(current_expr, agent.behavior)

        return AgentDefinition(
            name=f"seq_{self.name}",
            input_type=self.agents[0].input_type,
            output_type=self.agents[-1].output_type,
            behavior=current_expr,
            description=f"Sequential composition of {len(self.agents)} agents"
        )

    def _compose_parallel(self) -> AgentDefinition:
        """并行组合: par(agent1, agent2, ...)"""
        # 并行执行需要实际的运行时支持
        # 这里创建结构化的组合，实际执行时需要并发处理

        return AgentDefinition(
            name=f"par_{self.name}",
            input_type=self.agents[0].input_type,
            output_type=BaseType("ParallelResult"),
            behavior=self._create_parallel_expr(),
            description=f"Parallel composition of {len(self.agents)} agents"
        )

    def _create_parallel_expr(self) -> LLMbdaExpr:
        """创建并行表达式"""
        from .core import Variable, ToolCall

        # 使用并行执行工具
        agent_names = [a.name for a in self.agents]
        return ToolCall(
            "parallel_execute",
            {
                "agents": Variable("input"),
                "agent_names": agent_names
            }
        )

    def _compose_conditional(self) -> AgentDefinition:
        """条件组合: if condition then agent1 else agent2"""
        from .core import ToolCall

        return AgentDefinition(
            name=f"cond_{self.name}",
            input_type=BaseType("Any"),
            output_type=BaseType("Any"),
            behavior=ToolCall(
                "conditional_execute",
                {
                    "condition": Variable("condition"),
                    "then_agent": self.agents[0].name if len(self.agents) > 0 else None,
                    "else_agent": self.agents[1].name if len(self.agents) > 1 else None
                }
            ),
            description="Conditional agent composition"
        )

    def _compose_pipeline(self) -> AgentDefinition:
        """流水线组合: 输出流式传递"""
        return self._compose_sequential()


class LLMbdaAgent:
    """
    LLMbda Agent

    形式化的 Agent 实现
    提供类型安全、信息流安全保证
    """

    def __init__(
        self,
        definition: AgentDefinition,
        tool_executor: Optional[Callable] = None,
        enable_safety: bool = True
    ):
        self.definition = definition
        self.evaluator = LLMbdaEvaluator(tool_executor)
        self.enable_safety = enable_safety

        # 设置安全分析器
        self.flow_analyzer = None
        if enable_safety:
            policy = FlowPolicy()
            policy.set_variable_label("input", definition.security_level)
            self.flow_analyzer = InformationFlowAnalyzer(policy)

    def execute(
        self,
        input_value: Any,
        context: Optional[EvaluationContext] = None
    ) -> Any:
        """
        执行 Agent

        Args:
            input_value: 输入值
            context: 执行上下文

        Returns:
            执行结果
        """
        ctx = context or self.evaluator.create_context()

        # 绑定输入
        ctx = ctx.with_binding("input", input_value, self.definition.input_type)

        # 安全检查
        if self.flow_analyzer:
            self.flow_analyzer.clear_violations()
            self.flow_analyzer.check_expr(self.definition.behavior)

            if not self.flow_analyzer.is_safe():
                violations = self.flow_analyzer.get_violations()
                logger.error(f"[LLMbda] Security violations detected: {violations}")
                raise SecurityError(f"Information flow violation: {violations[0] if violations else 'Unknown'}")

        # 执行
        try:
            result = self.definition.behavior.evaluate(ctx)
            return result
        except Exception as e:
            logger.error(f"[LLMbda] Agent {self.definition.name} execution error: {e}")
            raise

    def get_type(self) -> LLMbdaType:
        """获取 Agent 类型"""
        return self.definition.get_function_type()

    def get_info(self) -> Dict[str, Any]:
        """获取 Agent 信息"""
        return {
            "name": self.definition.name,
            "description": self.definition.description,
            "input_type": str(self.definition.input_type),
            "output_type": str(self.definition.output_type),
            "security_level": self.definition.security_level.name,
            "metadata": self.definition.metadata
        }


class SecurityError(Exception):
    """安全违规错误"""
    pass


# ==================== TCM Agent 工厂 ====================

def create_diagnosis_agent() -> LLMbdaAgent:
    """创建诊断 Agent"""
    from .core import ToolCall, Variable

    behavior = Sequence(
        ToolCall("collect_symptoms", {"input": Variable("input")}),
        ToolCall("analyze_syndrome", {"symptoms": Variable("result")})
    )

    definition = AgentDefinition(
        name="diagnosis_agent",
        input_type=BaseType("PatientQuery"),
        output_type=BaseType("DiagnosisResult"),
        behavior=behavior,
        description="中医辨证诊断 Agent",
        security_level=SecurityLevel.CONFIDENTIAL
    )

    return LLMbdaAgent(definition)


def create_prescription_agent() -> LLMbdaAgent:
    """创建处方 Agent"""
    from .core import ToolCall, Variable

    behavior = ToolCall(
        "generate_prescription",
        {
            "syndrome": Variable("input"),
            "patient_info": Variable("patient_info")
        }
    )

    definition = AgentDefinition(
        name="prescription_agent",
        input_type=BaseType("Syndrome"),
        output_type=BaseType("Prescription"),
        behavior=behavior,
        description="中医处方生成 Agent",
        security_level=SecurityLevel.CONFIDENTIAL
    )

    return LLMbdaAgent(definition)


def create_tcm_pipeline() -> LLMbdaAgent:
    """创建完整 TCM 诊疗流程"""
    # 创建子 Agent
    diagnosis = create_diagnosis_agent()
    prescription = create_prescription_agent()

    # 组合
    composition = AgentComposition(
        name="tcm_pipeline",
        mode=ComposeMode.SEQUENTIAL,
        agents=[diagnosis.definition, prescription.definition]
    )

    combined = composition.compose()

    return LLMbdaAgent(
        combined,
        enable_safety=True
    )
