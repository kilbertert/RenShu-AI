"""
TCM Neo4j Schema Initializer
中医知识图谱Schema初始化器

创建约束、索引（全文索引和向量索引）
"""

import os
from typing import Optional

from ..tcm_neo4j import get_tcm_neo4j_connection


# Schema定义
CONSTRAINTS = [
    # Classic节点约束
    """
    CREATE CONSTRAINT classic_unique IF NOT EXISTS
    FOR (c:Classic) REQUIRE (c.book_name, c.chapter, c.title) IS UNIQUE
    """,

    # Case节点约束
    """
    CREATE CONSTRAINT case_unique IF NOT EXISTS
    FOR (ca:Case) REQUIRE ca.case_id IS UNIQUE
    """,

    # Syndrome节点约束
    """
    CREATE CONSTRAINT syndrome_unique IF NOT EXISTS
    FOR (s:Syndrome) REQUIRE s.name IS UNIQUE
    """,

    # Prescription节点约束
    """
    CREATE CONSTRAINT prescription_unique IF NOT EXISTS
    FOR (p:Prescription) REQUIRE p.name IS UNIQUE
    """,

    # Herb节点约束
    """
    CREATE CONSTRAINT herb_unique IF NOT EXISTS
    FOR (h:Herb) REQUIRE h.name IS UNIQUE
    """,
]

FULLTEXT_INDEXES = [
    # Classic全文索引
    """
    CREATE FULLTEXT INDEX classic_content_fulltext IF NOT EXISTS
    FOR (c:Classic) ON EACH [c.content, c.interpretation]
    """,

    # Case全文索引
    """
    CREATE FULLTEXT INDEX case_content_fulltext IF NOT EXISTS
    FOR (ca:Case) ON EACH [ca.chief_complaint, ca.syndrome, ca.prescription]
    """,
]

# 向量索引（需要Neo4j 5.11+）
VECTOR_INDEXES = [
    # Classic向量索引
    """
    CREATE VECTOR INDEX classic_embedding IF NOT EXISTS
    FOR (c:Classic) ON c.content_embedding
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dimension},
            `vector.similarity_function`: 'cosine'
        }}
    }}
    """,

    # Case向量索引
    """
    CREATE VECTOR INDEX case_embedding IF NOT EXISTS
    FOR (ca:Case) ON ca.case_embedding
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dimension},
            `vector.similarity_function`: 'cosine'
        }}
    }}
    """,
]


def create_constraints() -> list[str]:
    """
    创建唯一性约束

    Returns:
        list[str]: 执行结果消息列表
    """
    conn = get_tcm_neo4j_connection()
    results = []

    for constraint in CONSTRAINTS:
        try:
            conn.execute_query(constraint.strip())
            results.append(f"✓ 约束创建成功")
        except Exception as e:
            error_msg = str(e)
            if "already exists" in error_msg.lower():
                results.append(f"○ 约束已存在，跳过")
            else:
                results.append(f"✗ 约束创建失败: {error_msg}")

    return results


def create_fulltext_indexes() -> list[str]:
    """
    创建全文索引

    Returns:
        list[str]: 执行结果消息列表
    """
    conn = get_tcm_neo4j_connection()
    results = []

    for index in FULLTEXT_INDEXES:
        try:
            conn.execute_query(index.strip())
            results.append(f"✓ 全文索引创建成功")
        except Exception as e:
            error_msg = str(e)
            if "already exists" in error_msg.lower():
                results.append(f"○ 全文索引已存在，跳过")
            else:
                results.append(f"✗ 全文索引创建失败: {error_msg}")

    return results


def create_vector_indexes(dimension: int = 1024) -> list[str]:
    """
    创建向量索引

    Args:
        dimension: 向量维度，默认1024（DashScope text-embedding-v3）

    Returns:
        list[str]: 执行结果消息列表
    """
    conn = get_tcm_neo4j_connection()
    results = []

    for index_template in VECTOR_INDEXES:
        index = index_template.format(dimension=dimension)
        try:
            conn.execute_query(index.strip())
            results.append(f"✓ 向量索引创建成功 (维度: {dimension})")
        except Exception as e:
            error_msg = str(e)
            if "already exists" in error_msg.lower():
                results.append(f"○ 向量索引已存在，跳过")
            elif "not supported" in error_msg.lower():
                results.append(f"⚠ 向量索引不支持（需要Neo4j 5.11+）")
            else:
                results.append(f"✗ 向量索引创建失败: {error_msg}")

    return results


def init_schema(vector_dimension: int = 1024) -> dict:
    """
    初始化完整Schema

    Args:
        vector_dimension: 向量维度

    Returns:
        dict: 初始化结果
    """
    results = {
        "constraints": [],
        "fulltext_indexes": [],
        "vector_indexes": [],
        "success": True,
        "message": ""
    }

    try:
        # 创建约束
        results["constraints"] = create_constraints()

        # 创建全文索引
        results["fulltext_indexes"] = create_fulltext_indexes()

        # 创建向量索引
        results["vector_indexes"] = create_vector_indexes(vector_dimension)

        # 统计结果
        total = (
            len(results["constraints"]) +
            len(results["fulltext_indexes"]) +
            len(results["vector_indexes"])
        )
        success_count = sum(
            1 for r in (
                results["constraints"] +
                results["fulltext_indexes"] +
                results["vector_indexes"]
            )
            if r.startswith("✓") or r.startswith("○")
        )

        results["message"] = f"Schema初始化完成: {success_count}/{total} 成功"

    except Exception as e:
        results["success"] = False
        results["message"] = f"Schema初始化失败: {str(e)}"

    return results


def drop_all_indexes() -> list[str]:
    """
    删除所有索引（用于重置）

    Returns:
        list[str]: 执行结果消息列表
    """
    conn = get_tcm_neo4j_connection()
    results = []

    # 获取所有索引
    try:
        indexes = conn.execute_query("SHOW INDEXES")
        for idx in indexes:
            idx_name = idx.get("name", "")
            if idx_name and not idx_name.startswith("constraint"):
                try:
                    conn.execute_query(f"DROP INDEX {idx_name} IF EXISTS")
                    results.append(f"✓ 删除索引: {idx_name}")
                except Exception as e:
                    results.append(f"✗ 删除索引失败 {idx_name}: {str(e)}")
    except Exception as e:
        results.append(f"✗ 获取索引列表失败: {str(e)}")

    return results


def get_schema_info() -> dict:
    """
    获取当前Schema信息

    Returns:
        dict: Schema信息
    """
    conn = get_tcm_neo4j_connection()

    info = {
        "constraints": [],
        "indexes": [],
        "node_labels": [],
        "relationship_types": [],
    }

    try:
        # 获取约束
        constraints = conn.execute_query("SHOW CONSTRAINTS")
        info["constraints"] = [c.get("name", "") for c in constraints]

        # 获取索引
        indexes = conn.execute_query("SHOW INDEXES")
        info["indexes"] = [
            {"name": i.get("name", ""), "type": i.get("type", "")}
            for i in indexes
        ]

        # 获取节点标签
        labels = conn.execute_query("CALL db.labels()")
        info["node_labels"] = [l.get("label", "") for l in labels]

        # 获取关系类型
        rel_types = conn.execute_query("CALL db.relationshipTypes()")
        info["relationship_types"] = [r.get("relationshipType", "") for r in rel_types]

    except Exception as e:
        info["error"] = str(e)

    return info
