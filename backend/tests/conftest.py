"""
Pytest 全局 fixtures / 配置

要点：
- 集成测试需要连接真实 Neo4j。Neo4j 连接在 :mod:`app.src.agent.tcm_neo4j`
  中是单例，初始化时读取 ``NEO4J_DB`` 环境变量；为保证测试在 ``db=neo4j``
  上跑（ITCM / SymMap / HPOA / SIDER / TCMBank 都在这里），必须在
  ``pytest`` 收集阶段之前就设置好这个变量。
- ``NEO4J_DB`` 在 ``conftest.py`` 顶部设置，确保任何测试模块
  ``import`` 时（间接触发 tcm_neo4j 单例）拿到的都是 ``neo4j``。
- 提供 ``neo4j_available`` fixture 供集成测试 skip 决策，
  避免在 CI / 本地无 Neo4j 环境下失败。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 在任何 app.* 导入前设置环境变量
os.environ.setdefault("NEO4J_DB", "neo4j")

# 确保 backend 根目录在 sys.path（tests/ 在 backend/ 下，相对导入会用到）
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pytest  # noqa: E402


def _check_neo4j_connection() -> bool:
    """尝试连接 Neo4j；若失败返回 False（用于 skip 决策）。"""
    try:
        from app.src.core.graph_db import get_neo4j_graph
        g = get_neo4j_graph(database="neo4j")
        if g is None:
            return False
        g.query("RETURN 1 AS n")
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def neo4j_available() -> bool:
    """Session-scoped：检测 Neo4j 是否可用"""
    return _check_neo4j_connection()


@pytest.fixture(scope="session")
def neo4j_graph(neo4j_available):
    """Session-scoped：提供 Neo4j 图连接；不可用时 skip 调用方测试"""
    if not neo4j_available:
        pytest.skip("Neo4j 不可用（未配置或连接失败）")
    from app.src.core.graph_db import get_neo4j_graph
    g = get_neo4j_graph(database="neo4j")
    assert g is not None
    return g


def pytest_collection_modifyitems(config, items):
    """自动给标记为 'integration' 的测试做 Neo4j skip 决策"""
    skip_marker = pytest.mark.skip(reason="Neo4j 不可用")
    for item in items:
        if "integration" in item.keywords:
            if not _check_neo4j_connection():
                item.add_marker(skip_marker)
