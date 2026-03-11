"""
系统集成模块 (System Integration)

整合所有 2026 年最新特性到主工作流

Phase 7: 集成与部署
1. 模块集成 - 将所有新模块集成到主流程
2. 端到端测试 - 完整流程验证
3. 性能测试 - Token节省/响应时间/吞吐量
4. 灰度发布 - 逐步上线
"""

from .orchestrator import (
    SystemOrchestrator,
    OrchestratorConfig,
    ModuleStatus,
    IntegrationReport,
)
from .registry import (
    ModuleRegistry,
    ModuleDefinition,
    ModuleDependency,
    get_module_registry,
)
from .testing import (
    EndToEndTestRunner,
    TestScenario,
    TestResult,
)
from .deployment import (
    DeploymentConfig,
    CanaryDeployment,
    DeploymentStage,
    DeploymentMetrics,
)

__all__ = [
    # Orchestrator
    "SystemOrchestrator",
    "OrchestratorConfig",
    "ModuleStatus",
    "IntegrationReport",

    # Registry
    "ModuleRegistry",
    "ModuleDefinition",
    "ModuleDependency",
    "get_module_registry",

    # Testing
    "EndToEndTestRunner",
    "TestScenario",
    "TestResult",

    # Deployment
    "DeploymentConfig",
    "CanaryDeployment",
    "DeploymentStage",
    "DeploymentMetrics",
]
