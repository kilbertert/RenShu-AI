"""
中间件基类

定义中间件的标准接口，所有中间件都应继承此基类。

LangChain 1.0+ 中间件接口：
- before_model(): 模型调用前执行
- after_model(): 模型调用后执行
- wrap_tool_call(): 工具调用包装
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Callable
from dataclasses import dataclass


@dataclass
class MiddlewareConfig:
    """中间件配置"""
    enabled: bool = True
    priority: int = 0  # 优先级，数字越小越先执行
    name: str = ""


class BaseMiddleware(ABC):
    """
    中间件基类

    所有中间件都应继承此类并实现相应的钩子方法。

    执行顺序：
    1. before_model() - 模型调用前
    2. [模型执行]
    3. after_model() - 模型调用后

    工具调用时：
    - wrap_tool_call() 包装工具调用
    """

    def __init__(self, config: Optional[MiddlewareConfig] = None):
        """
        初始化中间件

        Args:
            config: 中间件配置
        """
        self.config = config or MiddlewareConfig()
        if not self.config.name:
            self.config.name = self.__class__.__name__

    @property
    def name(self) -> str:
        """中间件名称"""
        return self.config.name

    @property
    def enabled(self) -> bool:
        """是否启用"""
        return self.config.enabled

    @property
    def priority(self) -> int:
        """优先级"""
        return self.config.priority

    def _get_state_value(self, state: Any, key: str, default: Any = None) -> Any:
        """
        从状态中获取值（兼容字典和 Pydantic 模型）

        Args:
            state: 状态对象
            key: 键名
            default: 默认值

        Returns:
            对应的值
        """
        if isinstance(state, dict):
            return state.get(key, default)
        else:
            return getattr(state, key, default)

    def before_model(
            self,
            state: Dict[str, Any],
            runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """
        模型调用前执行

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            None: 继续执行
            Dict: 包含状态更新或拦截响应
                - 如果包含 "jump_to": "end"，则跳过后续执行
        """
        return None

    def after_model(
            self,
            state: Dict[str, Any],
            runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """
        模型调用后执行

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            None: 不修改
            Dict: 包含状态更新
        """
        return None

    def wrap_tool_call(
            self,
            tool_call: Callable,
            tool_name: str,
            state: Dict[str, Any]
    ) -> Callable:
        """
        包装工具调用

        可用于：
        - 工具调用前后的日志记录
        - 工具结果的大小检查和驱逐
        - 工具调用的超时控制

        Args:
            tool_call: 原始工具调用函数
            tool_name: 工具名称
            state: 当前状态

        Returns:
            包装后的工具调用函数
        """
        return tool_call


class MiddlewareChain:
    """
    中间件链

    管理多个中间件的执行顺序和调用。
    """

    def __init__(self):
        self._middlewares: list[BaseMiddleware] = []

    def add(self, middleware: BaseMiddleware) -> "MiddlewareChain":
        """添加中间件"""
        if middleware.enabled:
            self._middlewares.append(middleware)
            # 按优先级排序
            self._middlewares.sort(key=lambda m: m.priority)
        return self

    def remove(self, name: str) -> "MiddlewareChain":
        """移除中间件"""
        self._middlewares = [m for m in self._middlewares if m.name != name]
        return self

    def execute_before_model(
            self,
            state: Dict[str, Any],
            runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """
        执行所有中间件的 before_model

        Args:
            state: 状态（支持字典式访问的对象或字典）
            runtime: 运行时上下文

        Returns:
            None: 所有中间件都允许继续
            Dict: 累积的状态更新（包括拦截标记）
        """
        accumulated_updates = {}

        for middleware in self._middlewares:
            result = middleware.before_model(state, runtime)
            if result is not None:
                # 检查是否需要拦截
                if result.get("jump_to") == "end":
                    # 返回拦截结果（包含之前的累积更新）
                    accumulated_updates.update(result)
                    return accumulated_updates

                # 累积状态更新
                accumulated_updates.update(result)

                # 更新 state（用于后续中间件，支持字典式访问）
                for key, value in result.items():
                    state[key] = value

        # 返回所有累积的更新
        return accumulated_updates if accumulated_updates else None

    def execute_after_model(
            self,
            state: Dict[str, Any],
            runtime: Any
    ) -> Dict[str, Any]:
        """
        执行所有中间件的 after_model

        Args:
            state: 状态（支持字典式访问的对象或字典）
            runtime: 运行时上下文

        Returns:
            合并后的状态更新
        """
        accumulated_updates = {}

        # 逆序执行 after_model
        for middleware in reversed(self._middlewares):
            result = middleware.after_model(state, runtime)
            if result is not None:
                accumulated_updates.update(result)

                # 更新 state（用于后续中间件，支持字典式访问）
                for key, value in result.items():
                    state[key] = value

        return accumulated_updates

    def wrap_tool_call(
            self,
            tool_call: Callable,
            tool_name: str,
            state: Dict[str, Any]
    ) -> Callable:
        """
        用所有中间件包装工具调用
        """
        wrapped = tool_call
        for middleware in self._middlewares:
            wrapped = middleware.wrap_tool_call(wrapped, tool_name, state)
        return wrapped

    @property
    def middlewares(self) -> list[BaseMiddleware]:
        """获取所有中间件"""
        return self._middlewares.copy()
