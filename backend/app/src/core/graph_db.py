"""
Neo4j 图数据库连接管理（轻量门面）

为下游模块（kg_tools、moderate_diagnosis 等）提供统一的
``get_neo4j_graph()`` 入口。底层连接复用
:mod:`app.src.agent.tcm_neo4j` 中已经维护的 ``TCMNeo4jConnection`` 单例，
避免出现两个独立的 Neo4j 连接池。

设计要点：
- 缺失配置或连接失败时返回 ``None``，调用方据此降级到"无图谱增强"路径。
- 通过 :func:`functools.lru_cache` 缓存返回的 ``Neo4jGraph`` 引用（同一个
  database 复用同一对象），同时保留对底层连接单例状态的查询入口。
- 不在导入阶段拉起 Neo4j 驱动；只有真正调用 ``get_neo4j_graph()`` 时才尝试
  初始化，便于在无 Neo4j 的开发/CI 环境里依然能 ``import`` 本模块。
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from langchain_community.graphs import Neo4jGraph

from app.src.utils import get_logger

logger = get_logger("graph_db")


@lru_cache(maxsize=4)
def get_neo4j_graph(database: Optional[str] = None) -> Optional[Neo4jGraph]:
    """
    获取 Neo4j 图连接（按 database 维度缓存）。

    实现上委托给 :class:`app.src.agent.tcm_neo4j.TCMNeo4jConnection`
    单例，因此无论多少处调用本函数，都只会有一个 Neo4j 会话。

    Args:
        database: 目标数据库名。``None`` 时沿用环境变量 ``NEO4J_DB`` 的值
            （默认 ``tcm_graph``）。如果传入与底层连接不同的 database，
            仅记录警告，不会创建新连接——因为现有连接对象在构造时已绑定
            database。

    Returns:
        :class:`Neo4jGraph` 实例；配置缺失或连接失败时返回 ``None``。
    """
    connection = get_connection()
    if connection is None or not connection.is_connected():
        logger.warning("Neo4j 未连接（配置缺失或初始化失败），graph_db 不可用")
        return None

    expected_db = os.getenv("NEO4J_DB", "tcm_graph")
    if database and database != expected_db:
        logger.warning(
            "请求的 database=%s 与底层连接绑定的 %s 不一致，沿用底层值",
            database,
            expected_db,
        )

    return connection.graph


def is_graph_db_available() -> bool:
    """快速判断 Neo4j 是否可用（无副作用，不抛异常）。"""
    return get_neo4j_graph() is not None


def get_connection():
    """
    获取底层 TCMNeo4jConnection 单例。

    拆出此方法有两个目的：
    1. 让 ``get_neo4j_graph`` 内的 import 不再是匿名函数内 import，
       便于在测试中通过 ``patch.object`` 替换。
    2. 调用方可以选择直接拿连接对象做更细粒度的查询控制。

    Returns:
        :class:`app.src.agent.tcm_neo4j.TCMNeo4jConnection` 实例；
        模块无法导入时返回 ``None``。
    """
    try:
        from app.src.agent.tcm_neo4j import get_tcm_neo4j_connection
    except ImportError as exc:
        logger.error("无法导入 tcm_neo4j 模块: %s", exc)
        return None
    return get_tcm_neo4j_connection()
