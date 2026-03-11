"""
Final Agent - 统一架构的中医问诊系统

架构设计:
- 主图: 请求级别的通用处理（安全、记忆、路由、学习、脱敏）
- 子图: 具体业务逻辑执行（诊断、养生、药材、方剂等）

主图中间件（精简到 4 个）:
1. GuardrailsMiddleware - 安全检查
2. MemoryMiddleware - 用户记忆加载/保存
3. LearningMiddleware - 自我学习
4. PIIMiddleware - 输出脱敏

业务中间件（移到 DiagnoseSubgraph）:
- FocusContextMiddleware - 长对话压缩
- FilesystemMiddleware - 大结果驱逐
- Retry/Limit - 稳定性控制
"""

from .builder import build_main_graph, get_middleware_chain, get_llm
from .states import MainState, MainInput, MainOutput

__all__ = [
    "build_main_graph",
    "get_middleware_chain",
    "get_llm",
    "MainState",
    "MainInput",
    "MainOutput",
]
