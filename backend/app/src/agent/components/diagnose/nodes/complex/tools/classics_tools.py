"""
古籍检索工具

从中医古籍中检索相关论述
"""

from typing import List, Dict
from langchain.tools import tool

from app.src.utils import get_logger

logger = get_logger("classics_tools")


@tool
async def classics_search(
    keywords: List[str],
    books: List[str] = None,
    max_results: int = 5
) -> Dict:
    """
    从中医古籍中检索相关论述
    
    根据关键词在古籍库中搜索相关的理论依据和经典条文。
    
    Args:
        keywords: 关键词列表，如 ["头痛", "肝郁"]
        books: 检索的书籍列表，默认为常用经典
        max_results: 最多返回结果数，默认 5
    
    Returns:
        包含古籍引用的字典：
        {
            "citations": [
                {
                    "book": "伤寒论",
                    "chapter": "辨太阳病脉证并治",
                    "section": "第5条",
                    "text": "太阳之为病，脉浮，头项强痛而恶寒。",
                    "keywords_matched": ["头痛"]
                }
            ]
        }
    """
    if books is None:
        books = ["伤寒论", "金匮要略", "温病条辨", "黄帝内经"]
    
    logger.info(f"检索古籍，关键词: {keywords}，书籍: {books}")
    
    try:
        # 尝试使用 Elasticsearch 全文检索
        from app.src.core.search import get_classics_index
        
        classics_index = get_classics_index()
        
        results = await classics_index.asearch(
            query={
                "bool": {
                    "should": [
                        {"match": {"text": keyword}} for keyword in keywords
                    ],
                    "filter": {"terms": {"book": books}}
                }
            },
            size=max_results
        )
        
        citations = [
            {
                "book": hit["_source"]["book"],
                "chapter": hit["_source"]["chapter"],
                "section": hit["_source"].get("section", ""),
                "text": hit["_source"]["text"],
                "keywords_matched": [
                    kw for kw in keywords
                    if kw in hit["_source"]["text"]
                ]
            }
            for hit in results["hits"]["hits"]
        ]
        
        logger.info(f"找到 {len(citations)} 条古籍引用")
        return {"citations": citations}
        
    except Exception as e:
        logger.warning("真实古籍检索不可用: %s", e)
        return {
            "citations": [],
            "available": False,
            "reason": "尚未建立可追溯古籍全文索引，未返回模拟条文。",
        }
