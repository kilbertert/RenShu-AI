"""
系统编排器 (System Orchestrator)

协调所有 2026 模块的运行
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .registry import ModuleRegistry, ModuleStatus, get_module_registry


logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """编排器配置"""
    # 自动启用模块
    auto_enable_modules: bool = True

    # 性能监控
    enable_metrics: bool = True

    # 错误处理
    continue_on_error: bool = True

    # 调试模式
    debug_mode: bool = False


@dataclass
class ModuleStatus:
    """模块运行状态"""
    name: str
    status: str                           # running, stopped, error
    last_activity: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Dict = field(default_factory=dict)


@dataclass
class IntegrationReport:
    """集成报告"""
    timestamp: datetime = field(default_factory=datetime.now)
    total_modules: int = 0
    enabled_modules: int = 0
    running_modules: int = 0
    error_modules: int = 0
    startup_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    module_statuses: Dict[str, Dict] = field(default_factory=dict)


class SystemOrchestrator:
    """
    系统编排器

    负责协调所有 2026 模块的初始化、运行和监控
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        registry: Optional[ModuleRegistry] = None
    ):
        self.config = config or OrchestratorConfig()
        self.registry = registry or get_module_registry()

        self.module_states: Dict[str, ModuleStatus] = {}
        self.is_running = False

    async def initialize(self) -> IntegrationReport:
        """
        初始化所有模块

        Returns:
            集成报告
        """
        start_time = datetime.now()
        report = IntegrationReport()

        logger.info("[Orchestrator] Starting system initialization...")

        # 获取所有模块
        all_modules = self.registry.list_all()
        report.total_modules = len(all_modules)

        # 按优先级排序
        sorted_modules = sorted(all_modules, key=lambda m: m.priority)

        # 逐个启动模块
        for module in sorted_modules:
            if module.enabled_by_default or module.status == ModuleStatus.ENABLED:
                try:
                    await self._start_module(module)
                    report.enabled_modules += 1
                except Exception as e:
                    error_msg = f"Failed to start {module.name}: {e}"
                    report.errors.append(error_msg)
                    logger.error(f"[Orchestrator] {error_msg}")
                    if not self.config.continue_on_error:
                        break

        # 更新运行状态
        report.running_modules = len(self._get_running_modules())
        report.error_modules = len(self._get_error_modules())

        # 收集模块状态
        for name, state in self.module_states.items():
            report.module_statuses[name] = {
                "status": state.status,
                "error": state.error_message,
                "metrics": state.metrics
            }

        # 计算启动时间
        report.startup_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        self.is_running = True
        logger.info(
            f"[Orchestrator] Initialization complete: "
            f"{report.enabled_modules}/{report.total_modules} modules enabled"
        )

        return report

    async def _start_module(self, module) -> None:
        """启动单个模块"""
        logger.info(f"[Orchestrator] Starting module: {module.name}")

        # 检查依赖
        deps_ok, missing = self.registry.check_dependencies(module.name)
        if not deps_ok:
            raise RuntimeError(f"Missing dependencies: {missing}")

        # 更新状态
        self.module_states[module.name] = ModuleStatus(
            name=module.name,
            status="running",
            last_activity=datetime.now()
        )

        # 如果有初始化函数，调用它
        if module.initializer:
            await module.initializer(module.current_config)

        # 启用模块
        if module.status != ModuleStatus.ENABLED:
            module.enable()

    async def shutdown(self) -> None:
        """关闭系统"""
        logger.info("[Orchestrator] Shutting down system...")

        # 按相反顺序关闭模块
        for name, state in reversed(list(self.module_states.items())):
            try:
                module = self.registry.get(name)
                if module:
                    module.disable()
                state.status = "stopped"
                logger.info(f"[Orchestrator] Stopped module: {name}")
            except Exception as e:
                logger.error(f"[Orchestrator] Error stopping {name}: {e}")

        self.is_running = False

    def _get_running_modules(self) -> List[ModuleStatus]:
        """获取运行中的模块"""
        return [s for s in self.module_states.values() if s.status == "running"]

    def _get_error_modules(self) -> List[ModuleStatus]:
        """获取出错的模块"""
        return [s for s in self.module_states.values() if s.status == "error"]

    async def process_request(
        self,
        request: Dict,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        处理请求（通过所有启用的模块）

        Args:
            request: 请求数据
            context: 上下文

        Returns:
            处理后的响应
        """
        if not self.is_running:
            raise RuntimeError("Orchestrator not running")

        ctx = context or {}
        result = {"request": request, "modules_applied": []}

        # 按优先级顺序应用模块
        enabled_modules = self.registry.list_enabled()

        for module in sorted(enabled_modules, key=lambda m: m.priority):
            try:
                # 应用模块处理
                processed = await self._apply_module(module, request, ctx)
                if processed:
                    result["modules_applied"].append(module.name)
                    result.update(processed)

                # 更新模块活动时间
                if module.name in self.module_states:
                    self.module_states[module.name].last_activity = datetime.now()

            except Exception as e:
                error_msg = f"Module {module.name} error: {e}"
                logger.warning(f"[Orchestrator] {error_msg}")
                if not self.config.continue_on_error:
                    raise

        return result

    async def _apply_module(
        self,
        module,
        request: Dict,
        context: Dict
    ) -> Optional[Dict]:
        """应用单个模块处理"""
        # 这里根据模块类型执行相应的处理
        # 实际实现中会调用各模块的处理器

        if module.name == "focus_compression":
            # Focus 压缩处理
            return await self._apply_focus_compression(request, context)

        elif module.name == "mcp_protocol":
            # MCP 工具调用
            return await self._apply_mcp_protocol(request, context)

        elif module.name == "trism_security":
            # TRiSM 安全检查
            return await self._apply_trism_security(request, context)

        elif module.name == "self_learning":
            # 自学习记录
            return await self._apply_self_learning(request, context)

        return None

    async def _apply_focus_compression(self, request: Dict, context: Dict) -> Dict:
        """应用 Focus 压缩"""
        # 实际实现中会调用 FocusContextManager
        return {"focus_compression_applied": True}

    async def _apply_mcp_protocol(self, request: Dict, context: Dict) -> Dict:
        """应用 MCP 协议"""
        # 实际实现中会调用 MCP 工具
        return {"mcp_protocol_applied": True}

    async def _apply_trism_security(self, request: Dict, context: Dict) -> Dict:
        """应用 TRiSM 安全"""
        # 实际实现中会调用 TRiSM 检查
        return {"trism_security_checked": True}

    async def _apply_self_learning(self, request: Dict, context: Dict) -> Dict:
        """应用自学习"""
        # 实际实现中会记录交互
        return {"self_learning_recorded": True}

    def get_system_status(self) -> Dict:
        """获取系统状态"""
        running = self._get_running_modules()
        errors = self._get_error_modules()

        return {
            "is_running": self.is_running,
            "total_modules": len(self.module_states),
            "running_modules": len(running),
            "error_modules": len(errors),
            "running_module_names": [s.name for s in running],
            "error_module_names": [s.name for s in errors],
            "registry_summary": self.registry.get_status_summary()
        }

    def get_metrics(self) -> Dict:
        """获取系统指标"""
        return {
            "modules": {
                name: state.metrics
                for name, state in self.module_states.items()
            },
            "registry": self.registry.get_status_summary()
        }
