"""
工具监控

提供工具调用计数、延迟追踪和限频功能。
"""

import logging
import time
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ToolCallStats:
    """单个工具的调用统计"""
    call_count: int = 0
    total_latency_ms: float = 0.0
    error_count: int = 0
    last_call_time: Optional[float] = None

    @property
    def avg_latency_ms(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_latency_ms / self.call_count


class ToolMonitor:
    """
    工具调用监控器

    功能：
    1. 调用计数
    2. 延迟追踪
    3. 限频（每工具调用上限）
    4. 错误统计
    """

    def __init__(
        self,
        default_limit: int = 20,
        tool_limits: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            default_limit: 默认每工具调用上限
            tool_limits: 自定义每工具调用上限
        """
        self.default_limit = default_limit
        self.tool_limits = tool_limits or {}
        self._stats: Dict[str, ToolCallStats] = {}

    def _get_stats(self, tool_name: str) -> ToolCallStats:
        if tool_name not in self._stats:
            self._stats[tool_name] = ToolCallStats()
        return self._stats[tool_name]

    def can_call(self, tool_name: str) -> bool:
        """检查工具是否还能调用（未超限）"""
        stats = self._get_stats(tool_name)
        limit = self.tool_limits.get(tool_name, self.default_limit)
        return stats.call_count < limit

    def record_call(
        self,
        tool_name: str,
        latency_ms: float,
        error: bool = False,
    ) -> None:
        """记录一次工具调用"""
        stats = self._get_stats(tool_name)
        stats.call_count += 1
        stats.total_latency_ms += latency_ms
        stats.last_call_time = time.time()
        if error:
            stats.error_count += 1

    def wrap_tool_call(
        self,
        tool_call: Callable,
        tool_name: str,
    ) -> Callable:
        """
        包装工具调用，自动记录统计

        Args:
            tool_call: 原始工具调用函数
            tool_name: 工具名称

        Returns:
            包装后的函数
        """
        monitor = self

        async def wrapped(*args, **kwargs) -> Any:
            if not monitor.can_call(tool_name):
                limit = monitor.tool_limits.get(tool_name, monitor.default_limit)
                logger.warning(f"工具 {tool_name} 已达调用上限 ({limit})")
                return {"error": f"工具 {tool_name} 调用次数已达上限 ({limit})"}

            start = time.time()
            error = False
            try:
                result = await tool_call(*args, **kwargs)
                return result
            except Exception as e:
                error = True
                raise
            finally:
                latency_ms = (time.time() - start) * 1000
                monitor.record_call(tool_name, latency_ms, error=error)

        return wrapped

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有工具的调用统计"""
        return {
            name: {
                "call_count": stats.call_count,
                "avg_latency_ms": round(stats.avg_latency_ms, 2),
                "error_count": stats.error_count,
                "limit": self.tool_limits.get(name, self.default_limit),
                "remaining": max(0, self.tool_limits.get(name, self.default_limit) - stats.call_count),
            }
            for name, stats in self._stats.items()
        }

    def reset(self) -> None:
        """重置所有统计"""
        self._stats.clear()
