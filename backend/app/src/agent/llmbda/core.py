"""
LLMbda 核心表达式

基于 Lambda 演算的 Agent 形式化模型

语法:
  e ::= x                    变量
      | λx:T.e               抽象 (函数)
      | e e                  应用
      | tool(e)              工具调用
      | msg(e)               消息发送
      | e; e                 序列
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum


logger = logging.getLogger(__name__)


# ==================== 类型系统 ====================

class LLMbdaType(ABC):
    """LLMbda 类型基类"""

    def __str__(self) -> str:
        return self.__class__.__name__

    def __eq__(self, other) -> bool:
        return self.__class__ == other.__class__


@dataclass
class BaseType(LLMbdaType):
    """基础类型"""
    name: str

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return isinstance(other, BaseType) and self.name == other.name


@dataclass
class FunctionType(LLMbdaType):
    """函数类型 T1 -> T2"""
    input_type: LLMbdaType
    output_type: LLMbdaType

    def __str__(self) -> str:
        return f"({self.input_type} -> {self.output_type})"


@dataclass
class ToolType(LLMbdaType):
    """工具类型"""
    input_type: LLMbdaType
    output_type: LLMbdaType

    def __str__(self) -> str:
        return f"Tool<{self.input_type}, {self.output_type}>"


@dataclass
class MessageType(LLMbdaType):
    """消息类型"""
    content_type: LLMbdaType

    def __str__(self) -> str:
        return f"Msg<{self.content_type}>"


# 预定义基础类型
Str = BaseType("String")
Num = BaseType("Number")
Bool = BaseType("Boolean")
AnyType = BaseType("Any")
Context = BaseType("Context")


# ==================== 表达式 ====================

class LLMbdaExpr(ABC):
    """LLMbda 表达式基类"""

    @abstractmethod
    def evaluate(self, context: "EvaluationContext") -> Any:
        """求值"""
        pass

    @abstractmethod
    def get_type(self, context: "EvaluationContext") -> LLMbdaType:
        """类型推断"""
        pass

    @abstractmethod
    def free_variables(self) -> Set[str]:
        """获取自由变量"""
        pass


@dataclass
class Variable(LLMbdaExpr):
    """变量 x"""
    name: str

    def evaluate(self, context: "EvaluationContext") -> Any:
        return context.get(self.name)

    def get_type(self, context: "EvaluationContext") -> LLMbdaType:
        return context.get_type(self.name)

    def free_variables(self) -> Set[str]:
        return {self.name}


@dataclass
class Abstraction(LLMbdaExpr):
    """抽象 λx:T.e"""
    param: str                  # 参数名
    param_type: LLMbdaType      # 参数类型
    body: LLMbdaExpr            # 函数体

    def evaluate(self, context: "EvaluationContext") -> Any:
        # 返回闭包
        return Closure(self, context)

    def get_type(self, context: "EvaluationContext") -> LLMbdaType:
        body_type = self.body.get_type(
            context.with_binding(self.param, self.param_type)
        )
        return FunctionType(self.param_type, body_type)

    def free_variables(self) -> Set[str]:
        return self.body.free_variables() - {self.param}


@dataclass
class Application(LLMbdaExpr):
    """应用 e1 e2"""
    function: LLMbdaExpr
    argument: LLMbdaExpr

    def evaluate(self, context: "EvaluationContext") -> Any:
        func_value = self.function.evaluate(context)
        arg_value = self.argument.evaluate(context)

        if isinstance(func_value, Closure):
            # 应用闭包
            new_context = func_value.context.with_binding(
                func_value.abstraction.param,
                arg_value
            )
            return func_value.abstraction.body.evaluate(new_context)
        else:
            raise TypeError(f"Cannot apply non-function: {type(func_value)}")

    def get_type(self, context: "EvaluationContext") -> LLMbdaType:
        func_type = self.function.get_type(context)
        if isinstance(func_type, FunctionType):
            return func_type.output_type
        raise TypeError(f"Expected function type, got {func_type}")

    def free_variables(self) -> Set[str]:
        return self.function.free_variables() | self.argument.free_variables()


@dataclass
class ToolCall(LLMbdaExpr):
    """工具调用 tool(args)"""
    tool_name: str
    arguments: Dict[str, LLMbdaExpr]
    result_type: LLMbdaType = field(default_factory=lambda: AnyType)

    def evaluate(self, context: "EvaluationContext") -> Any:
        # 求值所有参数
        evaluated_args = {
            k: v.evaluate(context)
            for k, v in self.arguments.items()
        }

        # 从上下文获取工具执行器
        executor = context.get_tool_executor()
        if executor is None:
            raise RuntimeError("No tool executor available")

        # 执行工具
        return executor(self.tool_name, evaluated_args)

    def get_type(self, context: "EvaluationContext") -> LLMbdaType:
        return ToolType(AnyType, self.result_type)

    def free_variables(self) -> Set[str]:
        vars_set: Set[str] = set()
        for expr in self.arguments.values():
            vars_set |= expr.free_variables()
        return vars_set


@dataclass
class Message(LLMbdaExpr):
    """消息发送 msg(content)"""
    content: LLMbdaExpr
    recipient: Optional[str] = None  # 接收者 (用于多Agent)

    def evaluate(self, context: "EvaluationContext") -> Any:
        content_value = self.content.evaluate(context)

        # 记录到上下文
        context.add_message({
            "content": content_value,
            "recipient": self.recipient,
            "timestamp": context.get_timestamp()
        })

        return content_value

    def get_type(self, context: "EvaluationContext") -> LLMbdaType:
        return MessageType(self.content.get_type(context))

    def free_variables(self) -> Set[str]:
        return self.content.free_variables()


@dataclass
class Sequence(LLMbdaExpr):
    """序列 e1; e2"""
    first: LLMbdaExpr
    second: LLMbdaExpr

    def evaluate(self, context: "EvaluationContext") -> Any:
        self.first.evaluate(context)
        return self.second.evaluate(context)

    def get_type(self, context: "EvaluationContext") -> LLMbdaType:
        self.first.evaluate(context)  # 副作用
        return self.second.get_type(context)

    def free_variables(self) -> Set[str]:
        return self.first.free_variables() | self.second.free_variables()


# ==================== 闭包 ====================

@dataclass
class Closure:
    """闭包 - 抽象 + 捕获的上下文"""
    abstraction: Abstraction
    context: "EvaluationContext"


# ==================== 求值上下文 ====================

class EvaluationContext:
    """求值上下文"""

    def __init__(
        self,
        parent: Optional["EvaluationContext"] = None,
        bindings: Optional[Dict[str, object]] = None,
        type_bindings: Optional[Dict[str, LLMbdaType]] = None,
        tool_executor: Optional[object] = None
    ):
        self.parent = parent
        self.bindings = bindings or {}
        self.type_bindings = type_bindings or {}
        self.tool_executor = tool_executor
        self.messages: List[Dict] = []
        self.timestamp = 0

    def get(self, name: str) -> Any:
        """获取变量值"""
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.get(name)
        raise NameError(f"Undefined variable: {name}")

    def get_type(self, name: str) -> LLMbdaType:
        """获取变量类型"""
        if name in self.type_bindings:
            return self.type_bindings[name]
        if self.parent:
            return self.parent.get_type(name)
        raise NameError(f"Undefined type for variable: {name}")

    def with_binding(
        self,
        name: str,
        value: Any,
        value_type: Optional[LLMbdaType] = None
    ) -> "EvaluationContext":
        """创建扩展上下文"""
        new_bindings = self.bindings.copy()
        new_bindings[name] = value

        new_type_bindings = self.type_bindings.copy()
        if value_type:
            new_type_bindings[name] = value_type
        elif name in self.type_bindings:
            new_type_bindings[name] = self.type_bindings[name]

        return EvaluationContext(
            parent=self.parent or self,
            bindings=new_bindings,
            type_bindings=new_type_bindings,
            tool_executor=self.tool_executor
        )

    def get_tool_executor(self) -> Optional[callable]:
        """获取工具执行器"""
        return self.tool_executor or (self.parent.get_tool_executor() if self.parent else None)

    def add_message(self, message: Dict) -> None:
        """添加消息"""
        self.messages.append(message)
        self.timestamp += 1

    def get_timestamp(self) -> int:
        """获取当前时间戳"""
        return self.timestamp


# ==================== 求值器 ====================

class LLMbdaEvaluator:
    """LLMbda 表达式求值器"""

    def __init__(self, tool_executor: Optional[callable] = None):
        self.tool_executor = tool_executor

    def create_context(
        self,
        bindings: Optional[Dict[str, Any]] = None,
        type_bindings: Optional[Dict[str, LLMbdaType]] = None
    ) -> EvaluationContext:
        """创建求值上下文"""
        return EvaluationContext(
            bindings=bindings,
            type_bindings=type_bindings,
            tool_executor=self.tool_executor
        )

    def evaluate(
        self,
        expr: LLMbdaExpr,
        context: Optional[EvaluationContext] = None
    ) -> Any:
        """求值表达式"""
        ctx = context or self.create_context()
        return expr.evaluate(ctx)

    def infer_type(
        self,
        expr: LLMbdaExpr,
        context: Optional[EvaluationContext] = None
    ) -> LLMbdaType:
        """类型推断"""
        ctx = context or self.create_context()
        return expr.get_type(ctx)


# ==================== 辅助函数 ====================

def var(name: str) -> Variable:
    """创建变量"""
    return Variable(name)


def lam(param: str, param_type: LLMbdaType, body: LLMbdaExpr) -> Abstraction:
    """创建抽象"""
    return Abstraction(param, param_type, body)


def app(func: LLMbdaExpr, arg: LLMbdaExpr) -> Application:
    """创建应用"""
    return Application(func, arg)


def tool_call(tool_name: str, **kwargs: LLMbdaExpr) -> ToolCall:
    """创建工具调用"""
    return ToolCall(tool_name, kwargs)


def msg(content: LLMbdaExpr, recipient: Optional[str] = None) -> Message:
    """创建消息"""
    return Message(content, recipient)


def seq(first: LLMbdaExpr, second: LLMbdaExpr) -> Sequence:
    """创建序列"""
    return Sequence(first, second)
