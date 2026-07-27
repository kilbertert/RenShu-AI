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
        logger.warning("真实网络搜索不可用: %s", e)
        return {
            "results": [],
            "available": False,
            "reason": "未配置可信网络检索服务，未返回模拟网页。",
        }


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
        logger.warning("真实医学文献搜索不可用: %s", e)
        return {
            "papers": [],
            "available": False,
            "reason": "未配置可信医学文献检索服务，未返回模拟论文。",
        }


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
