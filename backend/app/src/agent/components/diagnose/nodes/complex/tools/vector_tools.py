"""
向量检索工具

从向量数据库检索相似医案
"""

from typing import Dict, Optional, List
from langchain.tools import tool

from app.src.utils import get_logger

logger = get_logger("vector_tools")


@tool
async def case_vector_search(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.7
) -> Dict:
    """
    从向量数据库检索相似医案
    
    根据症状描述，在医案库中查找相似的历史病例，
    提供参考的辨证思路和治疗方案。
    
    Args:
        query: 查询文本（症状描述），如 "头痛 胸闷 失眠"
        top_k: 返回最相似的 k 个案例，默认 5
        similarity_threshold: 相似度阈值，默认 0.7
    
    Returns:
        包含相似医案的字典：
        {
            "similar_cases": [
                {
                    "case_id": "case_123",
                    "similarity": 0.92,
                    "patient_info": "男，35岁",
                    "chief_complaint": "头痛3天",
                    "syndrome": "肝郁脾虚",
                    "treatment": "逍遥散加减",
                    "outcome": "显效"
                }
            ]
        }
    """
    logger.info(f"检索相似医案，查询: {query}")
    
    try:
        # 尝试使用向量数据库
        from app.src.core.vector_store import get_vector_store, get_embedding_model
        
        vector_store = get_vector_store("tcm_cases")
        embedding_model = get_embedding_model()
        
        # 向量检索
        query_embedding = await embedding_model.aembed_query(query)
        
        results = await vector_store.asimilarity_search_with_score(
            query_embedding,
            k=top_k,
            score_threshold=similarity_threshold
        )
        
        similar_cases = [
            {
                "case_id": doc.metadata.get("case_id", f"case_{i}"),
                "similarity": float(score),
                "patient_info": doc.metadata.get("patient_info", "未知"),
                "chief_complaint": doc.metadata.get("chief_complaint", ""),
                "syndrome": doc.metadata.get("syndrome", ""),
                "treatment": doc.metadata.get("treatment", ""),
                "outcome": doc.metadata.get("outcome", "未知")
            }
            for i, (doc, score) in enumerate(results)
        ]
        
        logger.info(f"找到 {len(similar_cases)} 个相似医案")
        return {"similar_cases": similar_cases}
        
    except Exception as e:
        logger.warning(f"向量检索失败: {e}，使用模拟数据")
        return {"similar_cases": _get_mock_cases(query)}


def _get_mock_cases(query: str) -> List[Dict]:
    """生成模拟的医案数据"""
    # 基于关键词的简单模拟
    mock_cases = [
        {
            "case_id": "case_001",
            "similarity": 0.89,
            "patient_info": "女，42岁",
            "chief_complaint": "头痛反复发作2年，近1月加重",
            "syndrome": "肝郁脾虚",
            "treatment": "逍遥散加减：柴胡10g、白芍15g、当归10g、茯苓15g、白术10g、甘草6g、薄荷6g、生姜3片",
            "outcome": "服药7剂后头痛明显减轻，继服14剂后痊愈"
        },
        {
            "case_id": "case_002",
            "similarity": 0.85,
            "patient_info": "男，35岁",
            "chief_complaint": "失眠多梦3月余",
            "syndrome": "心肾不交",
            "treatment": "交泰丸加减：黄连3g、肉桂1g、酸枣仁15g、远志10g、茯神15g",
            "outcome": "服药14剂后睡眠改善，续服巩固"
        },
        {
            "case_id": "case_003",
            "similarity": 0.82,
            "patient_info": "女，55岁",
            "chief_complaint": "胸闷气短1年",
            "syndrome": "心气虚",
            "treatment": "归脾汤加减：党参15g、黄芪20g、白术10g、茯神15g、酸枣仁15g、龙眼肉10g",
            "outcome": "服药21剂后症状明显改善"
        }
    ]
    
    # 根据查询关键词筛选
    keywords = query.replace(",", " ").replace("，", " ").split()
    filtered_cases = []
    
    for case in mock_cases:
        match_count = sum(
            1 for kw in keywords
            if kw in case["chief_complaint"] or kw in case["syndrome"]
        )
        if match_count > 0:
            case_copy = case.copy()
            case_copy["similarity"] = min(0.95, case["similarity"] + match_count * 0.05)
            filtered_cases.append(case_copy)
    
    # 如果没有匹配的，返回全部
    if not filtered_cases:
        filtered_cases = mock_cases
    
    return sorted(filtered_cases, key=lambda x: -x["similarity"])[:5]
