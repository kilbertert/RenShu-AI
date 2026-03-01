"""
TCM Knowledge Graph Connection
中医知识图谱Neo4j连接管理
"""

import os
from typing import Optional
from functools import lru_cache

from langchain_community.graphs import Neo4jGraph


class TCMNeo4jConnection:
    """中医知识图谱Neo4j连接管理器"""

    _instance: Optional["TCMNeo4jConnection"] = None
    _graph: Optional[Neo4jGraph] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._graph is None:
            self._initialize_connection()

    def _initialize_connection(self):
        """初始化Neo4j连接"""
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        database = os.getenv("NEO4J_DB", "tcm_graph")

        try:
            self._graph = Neo4jGraph(
                url=uri,
                username=username,
                password=password,
                database=database,
            )
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            self._graph = None

    @property
    def graph(self) -> Optional[Neo4jGraph]:
        """获取Neo4j图实例"""
        return self._graph

    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._graph is not None

    def execute_query(self, cypher: str, parameters: dict = None) -> list[dict]:
        """
        执行Cypher查询

        Args:
            cypher: Cypher查询语句
            parameters: 查询参数

        Returns:
            list[dict]: 查询结果
        """
        if not self.is_connected():
            raise ConnectionError("Neo4j connection not established")

        try:
            result = self._graph.query(cypher, params=parameters or {})
            return result
        except Exception as e:
            raise RuntimeError(f"Query execution failed: {e}")

    def get_schema(self) -> str:
        """获取图数据库Schema"""
        if not self.is_connected():
            return ""
        return self._graph.schema

    def refresh_schema(self):
        """刷新Schema缓存"""
        if self.is_connected():
            self._graph.refresh_schema()

    def close(self):
        """关闭连接"""
        if self._graph is not None:
            # Neo4jGraph doesn't have explicit close, but we can reset
            self._graph = None
            TCMNeo4jConnection._instance = None


@lru_cache(maxsize=1)
def get_tcm_neo4j_connection() -> TCMNeo4jConnection:
    """
    获取TCM Neo4j连接单例

    Returns:
        TCMNeo4jConnection: 连接实例
    """
    return TCMNeo4jConnection()


def get_neo4j_graph() -> Optional[Neo4jGraph]:
    """
    获取Neo4j图实例

    Returns:
        Neo4jGraph: Neo4j图实例
    """
    conn = get_tcm_neo4j_connection()
    return conn.graph


async def execute_cypher_async(cypher: str, parameters: dict = None) -> list[dict]:
    """
    异步执行Cypher查询（实际上是同步执行，但提供异步接口）

    Args:
        cypher: Cypher查询语句
        parameters: 查询参数

    Returns:
        list[dict]: 查询结果
    """
    conn = get_tcm_neo4j_connection()
    return conn.execute_query(cypher, parameters)
