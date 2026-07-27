"""
向量检索工具

从向量数据库检索相似医案
"""

from typing import Dict
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
        logger.warning("真实医案向量检索不可用: %s", e)
        return {
            "similar_cases": [],
            "available": False,
            "retrieval_mode": "unavailable",
            "reason": "尚未配置真实医案 embedding 与 Qdrant 集合，未返回模拟数据。",
        }
