"""
知识图谱工具

从 Neo4j 知识图谱查询证型和脏腑信息
"""

from typing import List, Dict, Optional
from langchain.tools import tool

from app.src.utils import get_logger

logger = get_logger("kg_tools")


@tool
async def kg_syndrome_search(
    symptoms: List[str],
    min_match_count: int = 2
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
        {
            "syndromes": [
                {
                    "name": "肝郁脾虚",
                    "matched_symptoms": ["头痛", "胸闷"],
                    "confidence": 0.85,
                    "description": "..."
                }
            ]
        }
    """
    logger.info(f"知识图谱查询证型，症状: {symptoms}")
    
    try:
        # 导入 Neo4j 图数据库
        from app.src.core.graph_db import get_neo4j_graph
        neo4j_graph = get_neo4j_graph()
        
        # Cypher 查询：查找症状匹配的证型
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
        
        results = await neo4j_graph.aquery(
            query,
            params={
                "symptoms": symptoms,
                "min_match_count": min_match_count,
                "symptom_count": len(symptoms)
            }
        )
        
        syndromes = [
            {
                "name": r["syndrome"],
                "matched_symptoms": r["matched_symptoms"],
                "confidence": r["confidence"],
                "description": r["description"]
            }
            for r in results
        ]
        
        logger.info(f"找到 {len(syndromes)} 个匹配证型")
        return {"syndromes": syndromes}
        
    except Exception as e:
        logger.warning(f"知识图谱查询失败: {e}，使用模拟数据")
        
        # 模拟数据（用于测试或 Neo4j 不可用时）
        mock_syndromes = _get_mock_syndromes(symptoms)
        return {"syndromes": mock_syndromes}


@tool
async def kg_organ_query(symptoms: List[str]) -> Dict:
    """
    查询症状涉及的脏腑系统
    
    分析症状与脏腑的关联关系，判断病位。
    
    Args:
        symptoms: 症状列表
    
    Returns:
        包含脏腑信息的字典：
        {
            "organs": [
                {
                    "name": "肝",
                    "related_symptoms": ["头痛", "胸闷"],
                    "function": "主疏泄，藏血",
                    "pathology": "肝郁气滞"
                }
            ]
        }
    """
    logger.info(f"查询脏腑系统，症状: {symptoms}")
    
    try:
        from app.src.core.graph_db import get_neo4j_graph
        neo4j_graph = get_neo4j_graph()
        
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
        
        results = await neo4j_graph.aquery(query, params={"symptoms": symptoms})
        
        organs = [
            {
                "name": r["organ"],
                "related_symptoms": r["related_symptoms"],
                "function": r["function"],
                "pathology": r["pathology"]
            }
            for r in results
        ]
        
        return {"organs": organs}
        
    except Exception as e:
        logger.warning(f"脏腑查询失败: {e}，使用模拟数据")
        return {"organs": _get_mock_organs(symptoms)}


def _get_mock_syndromes(symptoms: List[str]) -> List[Dict]:
    """生成模拟的证型数据"""
    # 基于常见症状的简单映射
    syndrome_map = {
        "头痛": ["肝阳上亢", "肝郁气滞", "血虚头痛"],
        "胸闷": ["肝郁气滞", "心气虚", "痰湿阻肺"],
        "失眠": ["心肾不交", "肝郁化火", "心血虚"],
        "腰膝酸软": ["肾阴虚", "肾阳虚", "肝肾亏虚"],
        "乏力": ["脾气虚", "气血两虚", "肾阳虚"],
        "食欲不振": ["脾胃虚弱", "肝郁脾虚", "湿困脾胃"],
        "腹胀": ["脾虚湿阻", "肝郁脾虚", "食积"],
    }
    
    # 统计证型出现次数
    syndrome_count = {}
    for symptom in symptoms:
        if symptom in syndrome_map:
            for syn in syndrome_map[symptom]:
                syndrome_count[syn] = syndrome_count.get(syn, 0) + 1
    
    # 排序并返回
    result = []
    for syn, count in sorted(syndrome_count.items(), key=lambda x: -x[1]):
        if count >= 1:
            result.append({
                "name": syn,
                "matched_symptoms": [s for s in symptoms if syn in syndrome_map.get(s, [])],
                "confidence": count / len(symptoms),
                "description": f"{syn}的典型表现"
            })
    
    return result[:5] if result else [
        {
            "name": "待进一步辨证",
            "matched_symptoms": symptoms,
            "confidence": 0.5,
            "description": "症状复杂，需要综合分析"
        }
    ]


def _get_mock_organs(symptoms: List[str]) -> List[Dict]:
    """生成模拟的脏腑数据"""
    organ_map = {
        "头痛": [("肝", "主疏泄", "肝阳上亢")],
        "胸闷": [("心", "主血脉", "心气不足"), ("肺", "主气", "痰湿阻肺")],
        "失眠": [("心", "藏神", "心神不宁"), ("肾", "主水", "心肾不交")],
        "腰膝酸软": [("肾", "主骨", "肾精亏虚")],
        "乏力": [("脾", "主运化", "脾气虚弱")],
        "食欲不振": [("脾", "主运化", "脾失健运"), ("胃", "主受纳", "胃气虚弱")],
    }
    
    organ_info = {}
    for symptom in symptoms:
        if symptom in organ_map:
            for organ, function, pathology in organ_map[symptom]:
                if organ not in organ_info:
                    organ_info[organ] = {
                        "name": organ,
                        "related_symptoms": [],
                        "function": function,
                        "pathology": pathology
                    }
                organ_info[organ]["related_symptoms"].append(symptom)
    
    return list(organ_info.values())
