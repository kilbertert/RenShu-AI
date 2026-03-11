"""
LLMbda - 形式化Agent模型

基于 arXiv:2602.20064 的 Lambda 演算与信息流控制

LLMbda 提供了：
1. 形式化的Agent组合语义
2. 信息流安全保证
3. 可验证的Agent行为
4. 类型系统的推理约束

参考论文: "LLMbda: A Lambda Calculus for Information Flow Control in LLM Agents"
"""

from .core import (
    # 核心表达式
    LLMbdaExpr,
    Variable,
    Abstraction,
    Application,
    ToolCall,
    Message,
    Sequence,

    # 类型系统
    LLMbdaType,
    BaseType,

    # 执行器
    LLMbdaEvaluator,
    EvaluationContext,
)
from .safety import (
    InformationFlowAnalyzer,
    FlowPolicy,
    SecurityLevel,
)
from .agent import (
    LLMbdaAgent,
    AgentDefinition,
    AgentComposition,
    ComposeMode,
)

__all__ = [
    # 核心表达式
    "LLMbdaExpr",
    "Variable",
    "Abstraction",
    "Application",
    "ToolCall",
    "Message",
    "Sequence",

    # 类型系统
    "LLMbdaType",
    "BaseType",

    # 执行器
    "LLMbdaEvaluator",
    "EvaluationContext",

    # 安全分析
    "InformationFlowAnalyzer",
    "FlowPolicy",
    "SecurityLevel",

    # Agent组合
    "LLMbdaAgent",
    "AgentDefinition",
    "AgentComposition",
    "ComposeMode",
]
