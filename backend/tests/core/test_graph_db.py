"""
graph_db 模块单元测试

覆盖以下场景：
- 底层 tcm_neo4j 模块导入失败时返回 None
- 底层连接未建立（未配置 / 初始化失败）时返回 None
- 底层连接可用时返回 Neo4jGraph 实例
- 同一 database 参数复用缓存
- is_graph_db_available() 与 get_neo4j_graph() 行为一致
- 环境变量全部缺失时返回 None

兼容 pytest（`pytest tests/core/test_graph_db.py`）和
unittest（`python -m unittest tests.core.test_graph_db`）。
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from app.src.core import graph_db


def _reset_graph_db_cache() -> None:
    """重置 graph_db 模块的 lru_cache。"""
    graph_db.get_neo4j_graph.cache_clear()


# graph_db.py 暴露了独立的 get_connection() 工厂函数用于测试注入。
_GET_CONN_PATH = "app.src.core.graph_db.get_connection"


class TestGetNeo4jGraph(unittest.TestCase):
    """针对 get_neo4j_graph() 的行为测试。"""

    def setUp(self) -> None:
        _reset_graph_db_cache()

    def tearDown(self) -> None:
        _reset_graph_db_cache()

    def test_returns_none_when_tcm_neo4j_unimportable(self) -> None:
        """当 tcm_neo4j 模块无法导入时返回 None，不抛异常。"""
        with patch.dict(sys.modules, {"app.src.agent.tcm_neo4j": None}):
            _reset_graph_db_cache()
            result = graph_db.get_neo4j_graph()
            self.assertIsNone(result)

    def test_returns_none_when_connection_not_established(self) -> None:
        """底层连接未建立（未配置 / 初始化失败）时返回 None。"""
        fake_connection = MagicMock()
        fake_connection.is_connected.return_value = False
        fake_connection.graph = None

        with patch(_GET_CONN_PATH, return_value=fake_connection):
            result = graph_db.get_neo4j_graph()
            self.assertIsNone(result)
            fake_connection.is_connected.assert_called_once()

    def test_returns_graph_when_connection_available(self) -> None:
        """底层连接可用时返回 Neo4jGraph 实例。"""
        fake_graph = MagicMock(name="Neo4jGraph")
        fake_connection = MagicMock()
        fake_connection.is_connected.return_value = True
        fake_connection.graph = fake_graph

        with patch(_GET_CONN_PATH, return_value=fake_connection):
            result = graph_db.get_neo4j_graph()
            self.assertIs(result, fake_graph)

    def test_mismatched_database_logs_warning_but_returns_graph(self) -> None:
        """请求的 database 与底层不一致时，记录警告并仍返回图对象。"""
        fake_graph = MagicMock(name="Neo4jGraph")
        fake_connection = MagicMock()
        fake_connection.is_connected.return_value = True
        fake_connection.graph = fake_graph

        with patch.dict(os.environ, {"NEO4J_DB": "tcm_graph"}):
            with patch(_GET_CONN_PATH, return_value=fake_connection):
                result = graph_db.get_neo4j_graph(database="other_db")
                self.assertIs(result, fake_graph)

    def test_caches_result_by_database(self) -> None:
        """同 database 参数的重复调用只触发一次底层访问。"""
        fake_graph = MagicMock(name="Neo4jGraph")
        fake_connection = MagicMock()
        fake_connection.is_connected.return_value = True
        fake_connection.graph = fake_graph

        with patch(_GET_CONN_PATH, return_value=fake_connection) as mock_get:
            graph_db.get_neo4j_graph()
            graph_db.get_neo4j_graph()
            graph_db.get_neo4j_graph()
            self.assertEqual(mock_get.call_count, 1)

    def test_different_database_args_use_separate_cache_entries(self) -> None:
        """不同 database 参数应分别缓存（maxsize=4 的体现）。"""
        fake_graph = MagicMock(name="Neo4jGraph")
        fake_connection = MagicMock()
        fake_connection.is_connected.return_value = True
        fake_connection.graph = fake_graph

        with patch.dict(os.environ, {"NEO4J_DB": "tcm_graph"}):
            with patch(_GET_CONN_PATH, return_value=fake_connection) as mock_get:
                graph_db.get_neo4j_graph(database=None)
                graph_db.get_neo4j_graph(database=None)
                graph_db.get_neo4j_graph(database="other_db")
                graph_db.get_neo4j_graph(database="other_db")
                # None 与 "other_db" 是两个不同的 cache key
                self.assertEqual(mock_get.call_count, 2)


class TestIsGraphDbAvailable(unittest.TestCase):
    """针对 is_graph_db_available() 的行为测试。"""

    def setUp(self) -> None:
        _reset_graph_db_cache()

    def tearDown(self) -> None:
        _reset_graph_db_cache()

    def test_returns_true_when_graph_available(self) -> None:
        """Neo4jGraph 可用时返回 True。"""
        fake_graph = MagicMock(name="Neo4jGraph")
        fake_connection = MagicMock()
        fake_connection.is_connected.return_value = True
        fake_connection.graph = fake_graph

        with patch(_GET_CONN_PATH, return_value=fake_connection):
            self.assertTrue(graph_db.is_graph_db_available())

    def test_returns_false_when_graph_unavailable(self) -> None:
        """Neo4jGraph 不可用时返回 False。"""
        fake_connection = MagicMock()
        fake_connection.is_connected.return_value = False

        with patch(_GET_CONN_PATH, return_value=fake_connection):
            self.assertFalse(graph_db.is_graph_db_available())


class TestEnvironmentConfiguration(unittest.TestCase):
    """端到端环境配置场景。"""

    def setUp(self) -> None:
        _reset_graph_db_cache()

    def tearDown(self) -> None:
        _reset_graph_db_cache()

    def test_missing_env_returns_none(self) -> None:
        """NEO4J_URI / USER / PASSWORD 全部缺失时，get_neo4j_graph() 返回 None。"""
        env_without_neo4j = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("NEO4J_")
        }
        with patch.dict(os.environ, env_without_neo4j, clear=True):
            from app.src.agent import tcm_neo4j

            tcm_neo4j.TCMNeo4jConnection._instance = None
            tcm_neo4j.TCMNeo4jConnection._graph = None
            _reset_graph_db_cache()

            try:
                result = graph_db.get_neo4j_graph()
                self.assertIsNone(result)
            finally:
                tcm_neo4j.TCMNeo4jConnection._instance = None
                tcm_neo4j.TCMNeo4jConnection._graph = None


if __name__ == "__main__":
    unittest.main()
