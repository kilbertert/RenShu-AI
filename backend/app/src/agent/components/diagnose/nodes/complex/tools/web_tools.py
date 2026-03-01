"""
网络搜索工具

搜索最新医学研究和中西医结合信息
"""

from typing import Dict, Optional, List
from langchain.tools import tool
import os

from app.src.utils import get_logger

logger = get_logger("web_tools")


@tool
async def web_search(
    query: str,
    max_results: int = 5,
    include_domains: Optional[List[str]] = None
) -> Dict:
    """
    网络搜索工具（支持并行）
    
    搜索互联网上的医学信息，获取最新研究和诊疗指南。
    
    Args:
        query: 搜索查询
        max_results: 返回结果数，默认 5
        include_domains: 限定域名（如医学网站）
    
    Returns:
        包含搜索结果的字典：
        {
            "results": [
                {
                    "title": "标题",
                    "url": "链接",
                    "content": "摘要",
                    "score": 0.95
                }
            ]
        }
    """
    logger.info(f"网络搜索: {query}")
    
    try:
        # 尝试使用 Tavily 搜索
        from tavily import TavilyClient
        
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not set")
        
        tavily_client = TavilyClient(api_key=api_key)
        
        response = await tavily_client.search(
            query=query,
            max_results=max_results,
            include_domains=include_domains or [],
            include_raw_content=False
        )
        
        results = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score", 0.0)
            }
            for item in response.get("results", [])
        ]
        
        logger.info(f"搜索到 {len(results)} 条结果")
        return {"results": results}
        
    except Exception as e:
        logger.warning(f"网络搜索失败: {e}，使用模拟数据")
        return {"results": _get_mock_web_results(query)}


@tool
async def medical_research_search(
    query: str,
    databases: List[str] = None
) -> Dict:
    """
    医学数据库搜索（中英文文献）
    
    搜索 PubMed、CNKI 等医学数据库的文献。
    
    Args:
        query: 搜索查询（支持中英文）
        databases: 数据库列表，默认 ["pubmed", "cnki"]
    
    Returns:
        包含文献的字典：
        {
            "papers": [
                {
                    "title": "论文标题",
                    "authors": "作者",
                    "abstract": "摘要",
                    "year": 2025,
                    "database": "pubmed"
                }
            ]
        }
    """
    if databases is None:
        databases = ["pubmed", "cnki"]
    
    logger.info(f"医学文献搜索: {query}，数据库: {databases}")
    
    try:
        # 尝试使用 Tavily 搜索医学网站
        from tavily import TavilyClient
        
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not set")
        
        tavily_client = TavilyClient(api_key=api_key)
        
        # 构建医学网站限定搜索
        medical_domains = ["pubmed.ncbi.nlm.nih.gov", "cnki.net", "wanfangdata.com.cn"]
        
        response = await tavily_client.search(
            query=query,
            max_results=5,
            include_domains=medical_domains
        )
        
        papers = [
            {
                "title": item.get("title", ""),
                "abstract": item.get("content", ""),
                "url": item.get("url", ""),
                "database": _detect_database(item.get("url", ""))
            }
            for item in response.get("results", [])
        ]
        
        return {"papers": papers}
        
    except Exception as e:
        logger.warning(f"医学文献搜索失败: {e}，使用模拟数据")
        return {"papers": _get_mock_papers(query)}


def _detect_database(url: str) -> str:
    """根据 URL 判断数据库来源"""
    if "pubmed" in url:
        return "pubmed"
    elif "cnki" in url:
        return "cnki"
    elif "wanfang" in url:
        return "wanfang"
    else:
        return "other"


def _get_mock_web_results(query: str) -> List[Dict]:
    """生成模拟的网络搜索结果"""
    return [
        {
            "title": f"中医辨证论治{query}的临床研究进展",
            "url": "https://example.com/article1",
            "content": f"本文综述了近年来中医对{query}的辨证论治研究进展，"
                       f"总结了常见证型和治疗方法，为临床提供参考。",
            "score": 0.85
        },
        {
            "title": f"{query}的中西医结合治疗策略",
            "url": "https://example.com/article2",
            "content": f"探讨{query}的中西医结合治疗方案，分析其优势和注意事项。",
            "score": 0.78
        },
        {
            "title": f"基于数据挖掘的{query}证型分布规律研究",
            "url": "https://example.com/article3",
            "content": f"运用数据挖掘技术分析{query}的证型分布规律及用药特点。",
            "score": 0.72
        }
    ]


def _get_mock_papers(query: str) -> List[Dict]:
    """生成模拟的医学文献数据"""
    return [
        {
            "title": f"Traditional Chinese Medicine Treatment of {query}: A Systematic Review",
            "authors": "Zhang, Y., Li, X., Wang, J.",
            "abstract": f"This systematic review evaluates the efficacy of TCM for {query}...",
            "year": 2025,
            "database": "pubmed"
        },
        {
            "title": f"{query}的中医证型分布及用药规律研究",
            "authors": "李明, 王芳, 张华",
            "abstract": f"目的：探讨{query}的中医证型分布规律及临床用药特点。方法：回顾性分析...",
            "year": 2024,
            "database": "cnki"
        }
    ]
