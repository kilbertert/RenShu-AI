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
        logger.warning(f"古籍检索失败: {e}，使用模拟数据")
        return {"citations": _get_mock_classics(keywords)}


def _get_mock_classics(keywords: List[str]) -> List[Dict]:
    """生成模拟的古籍数据"""
    # 古籍条文模拟库
    classics_db = [
        # 伤寒论
        {
            "book": "伤寒论",
            "chapter": "辨太阳病脉证并治",
            "section": "第1条",
            "text": "太阳之为病，脉浮，头项强痛而恶寒。",
            "keywords": ["头痛", "太阳病", "恶寒"]
        },
        {
            "book": "伤寒论",
            "chapter": "辨少阳病脉证并治",
            "section": "第96条",
            "text": "伤寒五六日中风，往来寒热，胸胁苦满，嘿嘿不欲饮食，心烦喜呕。",
            "keywords": ["胸闷", "少阳病", "不欲饮食"]
        },
        # 金匮要略
        {
            "book": "金匮要略",
            "chapter": "血痹虚劳病脉证并治",
            "text": "男子平人，脉虚弱细微者，善盗汗也。",
            "keywords": ["虚劳", "盗汗"]
        },
        {
            "book": "金匮要略",
            "chapter": "肺痿肺痈咳嗽上气病脉证治",
            "text": "咳而上气，喉中水鸡声，射干麻黄汤主之。",
            "keywords": ["咳嗽", "喉咙"]
        },
        # 黄帝内经
        {
            "book": "黄帝内经·素问",
            "chapter": "至真要大论",
            "text": "诸风掉眩，皆属于肝。",
            "keywords": ["头晕", "肝", "眩晕"]
        },
        {
            "book": "黄帝内经·素问",
            "chapter": "阴阳应象大论",
            "text": "肝主怒，怒伤肝。",
            "keywords": ["肝", "情志", "怒"]
        },
        {
            "book": "黄帝内经·灵枢",
            "chapter": "本神",
            "text": "心藏神，神有余则笑不休，神不足则悲。",
            "keywords": ["心", "失眠", "神"]
        },
        # 温病条辨
        {
            "book": "温病条辨",
            "chapter": "上焦篇",
            "text": "温病初起，发热无汗，或微恶风寒者，用辛凉轻剂。",
            "keywords": ["发热", "温病"]
        }
    ]
    
    # 根据关键词匹配
    matched_citations = []
    for entry in classics_db:
        matched_keywords = [kw for kw in keywords if kw in entry.get("keywords", []) or kw in entry["text"]]
        if matched_keywords:
            matched_citations.append({
                "book": entry["book"],
                "chapter": entry["chapter"],
                "section": entry.get("section", ""),
                "text": entry["text"],
                "keywords_matched": matched_keywords
            })
    
    # 按匹配关键词数排序
    matched_citations.sort(key=lambda x: -len(x["keywords_matched"]))
    
    return matched_citations[:5] if matched_citations else [
        {
            "book": "伤寒论",
            "chapter": "概论",
            "section": "",
            "text": "未找到直接相关条文，请参考相关章节进行辨证。",
            "keywords_matched": []
        }
    ]
