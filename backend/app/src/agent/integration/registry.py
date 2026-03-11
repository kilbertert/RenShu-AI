"""
模块注册中心 (Module Registry)

管理所有 2026 模块的注册、依赖和状态
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


logger = logging.getLogger(__name__)


class ModuleStatus(str, Enum):
    """模块状态"""
    DISABLED = "disabled"
    ENABLED = "enabled"
    ERROR = "error"
    TESTING = "testing"


@dataclass
class ModuleDependency:
    """模块依赖"""
    module_name: str
    required: bool = True  # 是否必需
    min_version: Optional[str] = None


@dataclass
class ModuleDefinition:
    """模块定义"""
    name: str                           # 模块名称
    version: str                         # 版本号
    description: str                     # 描述
    category: str                        # 分类 (memory, context, mcp, llmbda, trism, learning)
    priority: int = 0                    # 优先级 (越小越优先)

    # 状态管理
    status: ModuleStatus = ModuleStatus.DISABLED
    enabled_by_default: bool = False

    # 依赖
    dependencies: List[ModuleDependency] = field(default_factory=list)

    # 配置
    config_schema: Optional[Dict] = None
    current_config: Dict = field(default_factory=dict)

    # 初始化函数
    initializer: Optional[Callable] = None

    # 统计
    usage_count: int = 0
    error_count: int = 0
    last_used: Optional[str] = None

    def enable(self) -> bool:
        """启用模块"""
        if self.status == ModuleStatus.ERROR:
            logger.warning(f"[Registry] Module {self.name} has errors, cannot enable")
            return False

        self.status = ModuleStatus.ENABLED
        logger.info(f"[Registry] Module {self.name} enabled")
        return True

    def disable(self) -> None:
        """禁用模块"""
        self.status = ModuleStatus.DISABLED
        logger.info(f"[Registry] Module {self.name} disabled")

    def is_available(self) -> bool:
        """检查模块是否可用"""
        return self.status in [ModuleStatus.ENABLED, ModuleStatus.TESTING]

    def update_config(self, config: Dict) -> None:
        """更新配置"""
        self.current_config.update(config)
        logger.info(f"[Registry] Module {self.name} config updated")


class ModuleRegistry:
    """
    模块注册中心

    管理所有 2026 特性模块
    """

    def __init__(self):
        self._modules: Dict[str, ModuleDefinition] = {}
        self._initialize_builtin_modules()

    def _initialize_builtin_modules(self) -> None:
        """初始化内置模块"""

        # === Memory 模块 ===
        self.register(ModuleDefinition(
            name="mem0_memory",
            version="1.0.0",
            description="Mem0 记忆层 - 持久化跨会话记忆",
            category="memory",
            priority=1,
            enabled_by_default=True,
            dependencies=[],
            config_schema={
                "enabled": {"type": "boolean", "default": True},
                "vector_store": {"type": "string", "default": "qdrant"},
                "graph_store": {"type": "string", "default": "neo4j"}
            },
            current_config={"enabled": True}
        ))

        # === Context 模块 ===
        self.register(ModuleDefinition(
            name="focus_compression",
            version="1.0.0",
            description="Focus 主动上下文压缩 - 22.7% token节省",
            category="context",
            priority=2,
            enabled_by_default=True,
            dependencies=[],
            config_schema={
                "enabled": {"type": "boolean", "default": True},
                "compress_interval": {"type": "integer", "default": 12},
                "aggressive": {"type": "boolean", "default": True}
            },
            current_config={"enabled": True, "compress_interval": 12}
        ))

        # === MCP 模块 ===
        self.register(ModuleDefinition(
            name="mcp_protocol",
            version="1.0.0",
            description="MCP 工具协议 - 统一工具调用标准",
            category="mcp",
            priority=3,
            enabled_by_default=True,
            dependencies=[],
            config_schema={
                "enabled": {"type": "boolean", "default": True},
                "server_url": {"type": "string", "default": ""}
            },
            current_config={"enabled": True}
        ))

        # === LLMbda 模块 ===
        self.register(ModuleDefinition(
            name="llmbda_model",
            version="1.0.0",
            description="LLMbda 形式化 Agent 模型 - 信息流安全",
            category="llmbda",
            priority=4,
            enabled_by_default=False,  # 默认禁用，实验性功能
            dependencies=[],
            config_schema={
                "enabled": {"type": "boolean", "default": False},
                "safety_check": {"type": "boolean", "default": True}
            },
            current_config={"enabled": False}
        ))

        # === TRiSM 模块 ===
        self.register(ModuleDefinition(
            name="trism_security",
            version="1.0.0",
            description="TRiSM 安全框架 - Trust/Risk/Security",
            category="trism",
            priority=5,
            enabled_by_default=True,
            dependencies=[],
            config_schema={
                "enabled": {"type": "boolean", "default": True},
                "risk_threshold": {"type": "float", "default": 0.7},
                "trust_threshold": {"type": "float", "default": 0.5}
            },
            current_config={"enabled": True}
        ))

        # === Learning 模块 ===
        self.register(ModuleDefinition(
            name="self_learning",
            version="1.0.0",
            description="自学习机制 - 反馈/反思/进化",
            category="learning",
            priority=6,
            enabled_by_default=True,
            dependencies=[],
            config_schema={
                "enabled": {"type": "boolean", "default": True},
                "feedback_collection": {"type": "boolean", "default": True},
                "self_reflection": {"type": "boolean", "default": True},
                "evolution": {"type": "boolean", "default": False}
            },
            current_config={"enabled": True, "evolution": False}
        ))

    def register(self, module: ModuleDefinition) -> None:
        """注册模块"""
        self._modules[module.name] = module
        logger.info(f"[Registry] Module registered: {module.name} v{module.version}")

    def unregister(self, name: str) -> None:
        """注销模块"""
        if name in self._modules:
            del self._modules[name]
            logger.info(f"[Registry] Module unregistered: {name}")

    def get(self, name: str) -> Optional[ModuleDefinition]:
        """获取模块"""
        return self._modules.get(name)

    def list_all(self) -> List[ModuleDefinition]:
        """列出所有模块"""
        return list(self._modules.values())

    def list_by_category(self, category: str) -> List[ModuleDefinition]:
        """按类别列出模块"""
        return [m for m in self._modules.values() if m.category == category]

    def list_enabled(self) -> List[ModuleDefinition]:
        """列出已启用的模块"""
        return [m for m in self._modules.values() if m.status == ModuleStatus.ENABLED]

    def enable(self, name: str) -> bool:
        """启用模块"""
        module = self.get(name)
        if module:
            return module.enable()
        return False

    def disable(self, name: str) -> None:
        """禁用模块"""
        module = self.get(name)
        if module:
            module.disable()

    def update_config(self, name: str, config: Dict) -> bool:
        """更新模块配置"""
        module = self.get(name)
        if module:
            module.update_config(config)
            return True
        return False

    def check_dependencies(self, name: str) -> tuple[bool, List[str]]:
        """
        检查模块依赖

        Returns:
            (是否满足, 缺失的依赖列表)
        """
        module = self.get(name)
        if not module:
            return False, [f"Module {name} not found"]

        missing = []
        for dep in module.dependencies:
            dep_module = self.get(dep.module_name)
            if not dep_module:
                if dep.required:
                    missing.append(dep.module_name)
            elif dep_module.status != ModuleStatus.ENABLED:
                if dep.required:
                    missing.append(dep.module_name)

        return len(missing) == 0, missing

    def get_status_summary(self) -> Dict:
        """获取状态摘要"""
        modules = self.list_all()

        by_status: Dict = {s.value: 0 for s in ModuleStatus}
        by_category: Dict = {}

        for module in modules:
            by_status[module.status.value] += 1
            by_category[module.category] = by_category.get(module.category, 0) + 1

        return {
            "total_modules": len(modules),
            "by_status": by_status,
            "by_category": by_category,
            "enabled_count": len(self.list_enabled())
        }

    def get_startup_order(self) -> List[str]:
        """获取模块启动顺序（按优先级）"""
        enabled = self.list_enabled()
        return [m.name for m in sorted(enabled, key=lambda m: m.priority)]


# 全局注册中心实例
_global_registry: Optional[ModuleRegistry] = None


def get_module_registry() -> ModuleRegistry:
    """获取全局模块注册中心"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModuleRegistry()
    return _global_registry
