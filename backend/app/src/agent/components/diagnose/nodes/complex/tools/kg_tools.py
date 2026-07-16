"""
知识图谱工具

从 Neo4j 知识图谱查询证型和脏腑信息。
Neo4j 不可用（配置缺失 / 连接失败 / 驱动缺失）时返回空结果，
由上层 LLM 自行降级到"仅基于症状推理"路径，不回落到 mock 数据。
"""

from typing import List, Dict
from langchain.tools import tool

from app.src.utils import get_logger

logger = get_logger("kg_tools")

try:
    from app.src.core.graph_db import get_neo4j_graph, is_graph_db_available
    GRAPH_DB_AVAILABLE = True
except ImportError as exc:
    logger.error("无法导入 graph_db 模块，知识图谱工具将返回空结果: %s", exc)
    GRAPH_DB_AVAILABLE = False

    def get_neo4j_graph(database=None):  # type: ignore[no-redef]
        return None

    def is_graph_db_available() -> bool:  # type: ignore[no-redef]
        return False


@tool
async def kg_syndrome_search(
    symptoms: List[str],
    min_match_count: int = 2,
) -> Dict:
    """
    从 Neo4j 知识图谱查询症状对应的证型

    根据输入的症状列表，在知识图谱中查找匹配的证型，
    并返回匹配度和相关信息。

    Args:
        symptoms: 症状列表，如 ["头痛", "胸闷", "失眠"]
        min_match_count: 最少匹配症状数，默认 2

    Returns:
        包含匹配证型的字典：

        - 成功：``{"syndromes": [{"name": ..., "matched_symptoms": [...], ...}]}``
        - 失败/不可用：``{"syndromes": []}``
    """
    logger.info("知识图谱查询证型，症状: %s", symptoms)

    graph = get_neo4j_graph()
    if graph is None:
        logger.warning("Neo4j 不可用，跳过证型查询，返回空结果")
        return {"syndromes": []}

    query = """
    MATCH (s:Symptom)-[:INDICATES]->(syn:Syndrome)
    WHERE s.name IN $symptoms
    WITH syn, COLLECT(s.name) as matched_symptoms
    WHERE SIZE(matched_symptoms) >= $min_match_count
    RETURN syn.name as syndrome,
           syn.description as description,
           matched_symptoms,
           SIZE(matched_symptoms) as match_count,
           toFloat(SIZE(matched_symptoms)) / $symptom_count as confidence
    ORDER BY confidence DESC
    LIMIT 10
    """

    try:
        results = await graph.aquery(
            query,
            params={
                "symptoms": symptoms,
                "min_match_count": min_match_count,
                "symptom_count": len(symptoms),
            },
        )
    except Exception as exc:
        logger.error("证型查询失败: %s", exc)
        return {"syndromes": []}

    syndromes = [
        {
            "name": r["syndrome"],
            "matched_symptoms": r["matched_symptoms"],
            "confidence": r["confidence"],
            "description": r["description"],
        }
        for r in results
    ]

    logger.info("找到 %d 个匹配证型", len(syndromes))
    return {"syndromes": syndromes}


@tool
async def kg_organ_query(symptoms: List[str]) -> Dict:
    """
    查询症状涉及的脏腑系统

    分析症状与脏腑的关联关系，判断病位。

    Args:
        symptoms: 症状列表

    Returns:
        包含脏腑信息的字典：

        - 成功：``{"organs": [{"name": ..., "related_symptoms": [...], ...}]}``
        - 失败/不可用：``{"organs": []}``
    """
    logger.info("查询脏腑系统，症状: %s", symptoms)

    graph = get_neo4j_graph()
    if graph is None:
        logger.warning("Neo4j 不可用，跳过脏腑查询，返回空结果")
        return {"organs": []}

    query = """
    MATCH (s:Symptom)-[:BELONGS_TO]->(o:Organ)
    WHERE s.name IN $symptoms
    WITH o, COLLECT(s.name) as related_symptoms
    RETURN o.name as organ,
           o.function as function,
           o.pathology as pathology,
           related_symptoms
    ORDER BY SIZE(related_symptoms) DESC
    """

    try:
        results = await graph.aquery(query, params={"symptoms": symptoms})
    except Exception as exc:
        logger.error("脏腑查询失败: %s", exc)
        return {"organs": []}

    organs = [
        {
            "name": r["organ"],
            "related_symptoms": r["related_symptoms"],
            "function": r["function"],
            "pathology": r["pathology"],
        }
        for r in results
    ]

    logger.info("找到 %d 个相关脏腑", len(organs))
    return {"organs": organs}
