"""
日志追踪中间件

功能：
1. 记录请求开始/结束时间
2. 统计 Token 使用量
3. 记录执行步骤
4. 记录错误信息
5. 生成请求追踪 ID
"""

import time
import uuid
import logging
from typing import Any, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime

from .base import BaseMiddleware, MiddlewareConfig


@dataclass
class LoggingConfig(MiddlewareConfig):
    """日志中间件配置"""
    log_input: bool = True    # 是否记录输入
    log_output: bool = True   # 是否记录输出
    log_steps: bool = True    # 是否记录执行步骤
    log_tokens: bool = True   # 是否记录 Token 统计
    log_timing: bool = True   # 是否记录耗时
    log_errors: bool = True   # 是否记录错误

    # 日志级别
    log_level: str = "INFO"

    # 是否记录到文件
    log_to_file: bool = False
    log_file_path: str = "logs/tcm_agent.log"


@dataclass
class RequestLog:
    """请求日志"""
    trace_id: str
    user_id: str = ""
    conversation_id: str = ""
    query: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    steps: list = field(default_factory=list)
    error: str = ""
    status: str = "success"  # success, error, timeout


class TCMLoggingMiddleware(BaseMiddleware):
    """
    日志追踪中间件

    在请求开始和结束时记录日志，统计性能指标。
    """

    def __init__(self, config: Optional[LoggingConfig] = None):
        """初始化日志中间件"""
        super().__init__(config or LoggingConfig(name="TCMLoggingMiddleware", priority=15))
        self.logging_config = config or LoggingConfig()

        # 配置 logger
        self.logger = logging.getLogger("tcm_agent")
        self.logger.setLevel(getattr(logging, self.logging_config.log_level))

        # 配置 handler
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # File handler (可选)
            if self.logging_config.log_to_file:
                file_handler = logging.FileHandler(self.logging_config.log_file_path)
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    def before_model(
        self,
        state: Dict[str, Any],
        runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """
        模型调用前：记录请求开始
        """
        # 生成追踪 ID
        trace_id = str(uuid.uuid4())
        start_time = time.time()

        # 提取用户输入
        messages = self._get_state_value(state, "messages", [])
        user_query = ""
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "content"):
                user_query = last_message.content

        # 创建日志记录
        log = RequestLog(
            trace_id=trace_id,
            user_id=self._get_state_value(state, "user_id", ""),
            conversation_id=self._get_state_value(state, "conversation_id", ""),
            query=user_query,
            start_time=start_time,
        )

        # 记录开始日志
        if self.logging_config.log_input:
            self.logger.info(
                f"[{trace_id}] 请求开始 | "
                f"用户: {log.user_id} | "
                f"查询: {user_query[:50]}..."
            )

        # 将日志对象保存到状态中
        return {
            "_tcm_log": log,
            "_tcm_trace_id": trace_id,
        }

    def after_model(
        self,
        state: Dict[str, Any],
        runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """
        模型调用后：记录请求结束和统计信息
        """
        # 获取日志对象
        log = self._get_state_value(state, "_tcm_log")
        if not log:
            return None

        # 计算耗时
        log.end_time = time.time()
        log.duration_ms = (log.end_time - log.start_time) * 1000

        # 提取执行步骤
        steps = self._get_state_value(state, "steps", [])
        log.steps = steps

        # 提取错误信息
        error = self._get_state_value(state, "error", "")
        if error:
            log.error = error
            log.status = "error"

        # 记录结束日志
        if self.logging_config.log_timing:
            self.logger.info(
                f"[{log.trace_id}] 请求完成 | "
                f"耗时: {log.duration_ms:.2f}ms | "
                f"步骤数: {len(log.steps)} | "
                f"状态: {log.status}"
            )

        # 记录执行步骤
        if self.logging_config.log_steps and log.steps:
            self.logger.debug(f"[{log.trace_id}] 执行步骤:")
            for idx, step in enumerate(log.steps, 1):
                self.logger.debug(f"  {idx}. {step}")

        # 记录错误
        if self.logging_config.log_errors and log.error:
            self.logger.error(f"[{log.trace_id}] 错误: {log.error}")

        # 记录 Token 统计（如果有）
        if self.logging_config.log_tokens:
            # TODO: 从 runtime 中提取 Token 统计
            pass

        return None


def get_tcm_logging_middleware(**kwargs) -> TCMLoggingMiddleware:
    """
    获取 TCM 日志中间件实例

    Returns:
        TCMLoggingMiddleware 实例
    """
    config = LoggingConfig(**kwargs)
    return TCMLoggingMiddleware(config)
