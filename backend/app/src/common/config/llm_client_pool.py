"""LLM 客户端连接池 - 企业级优化
预热连接、复用客户端、减少连接开销
"""
import hashlib
from typing import Dict, Optional, List, Any
from uuid import UUID
from openai import AsyncOpenAI
from app.src.utils import get_logger

logger = get_logger("llm_client_pool")


class LLMClientPool:
    """LLM 客户端连接池 - 支持预热"""
    
    def __init__(self, max_clients: int = 100):
        self._clients: Dict[str, AsyncOpenAI] = {}
        self._max_clients = max_clients
        self._warmed_up = False
        
    def _generate_key(self, provider_id: str, api_key_hash: str, base_url: str) -> str:
        """生成客户端缓存键"""
        return f"{provider_id}:{api_key_hash[:8]}:{base_url}"
    
    async def warmup(self, providers: List[Dict[str, Any]]) -> int:
        """预热 LLM 客户端
        
        在应用启动时调用，预创建常用的 LLM 客户端连接
        
        Args:
            providers: 供应商配置列表，每个元素包含:
                - provider_id: 供应商ID
                - api_key: API Key
                - base_url: Base URL
                
        Returns:
            成功预热的客户端数量
        """
        if self._warmed_up:
            logger.debug("客户端已预热，跳过")
            return len(self._clients)
            
        warmed_count = 0
        for config in providers:
            try:
                provider_id = str(config.get('provider_id', ''))
                api_key = config.get('api_key', '')
                base_url = config.get('base_url', '')
                
                if not api_key or not base_url:
                    continue
                    
                await self.get_or_create_client(
                    provider_id=provider_id,
                    api_key=api_key,
                    base_url=base_url
                )
                warmed_count += 1
                logger.debug(f"🔥 预热客户端: {provider_id}")
            except Exception as e:
                logger.warning(f"预热客户端失败 [{config.get('provider_id')}]: {e}")
                
        self._warmed_up = True
        logger.info(f"✅ LLM 客户端预热完成，共 {warmed_count} 个")
        return warmed_count
    
    async def get_or_create_client(
        self, 
        provider_id: str,
        api_key: str,
        base_url: str,
        timeout: float = 60.0
    ) -> AsyncOpenAI:
        """获取或创建 LLM 客户端"""
        # 生成缓存键
        api_key_hash = hashlib.md5(api_key.encode()).hexdigest()
        cache_key = self._generate_key(provider_id, api_key_hash, base_url)
        
        # 尝试从缓存获取
        if cache_key in self._clients:
            logger.debug(f"🎯 LLM 客户端缓存命中: {cache_key}")
            return self._clients[cache_key]
        
        # 创建新客户端
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=2
        )
        
        # 存入缓存
        self._clients[cache_key] = client
        logger.info(f"💾 创建并缓存 LLM 客户端: {cache_key}")
        
        # 限制缓存大小（LRU）
        if len(self._clients) > self._max_clients:
            oldest_key = next(iter(self._clients))
            removed_client = self._clients.pop(oldest_key)
            await removed_client.close()
            logger.debug(f"♻️ 移除旧客户端: {oldest_key}")
        
        return client
    
    def get_client_sync(self, provider_id: str, api_key: str, base_url: str) -> Optional[AsyncOpenAI]:
        """同步获取已缓存的客户端（不创建新的）"""
        api_key_hash = hashlib.md5(api_key.encode()).hexdigest()
        cache_key = self._generate_key(provider_id, api_key_hash, base_url)
        return self._clients.get(cache_key)
    
    async def close_all(self):
        """关闭所有客户端连接"""
        for key, client in self._clients.items():
            try:
                await client.close()
            except Exception as e:
                logger.warning(f"关闭客户端失败 [{key}]: {e}")
        self._clients.clear()
        self._warmed_up = False
        logger.info("✅ 所有 LLM 客户端已关闭")
    
    @property
    def client_count(self) -> int:
        """获取当前缓存的客户端数量"""
        return len(self._clients)
    
    @property
    def is_warmed_up(self) -> bool:
        """是否已预热"""
        return self._warmed_up


# 全局客户端池实例
llm_client_pool = LLMClientPool(max_clients=50)


async def get_cached_llm_client(
    provider_id: str,
    api_key: str,
    base_url: str,
    timeout: float = 60.0
) -> AsyncOpenAI:
    """获取缓存的 LLM 客户端"""
    return await llm_client_pool.get_or_create_client(
        provider_id=provider_id,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout
    )
