"""
Redis 配置和缓存管理器
用于高性能缓存：用户配置、模型信息、API Key等
"""
import asyncio
import json
import pickle
import uuid
from typing import Optional, Any, Union
from datetime import timedelta
import redis.asyncio as redis
from app.src.utils import get_logger

logger = get_logger("redis_config")


class RedisManager:
    """Redis 异步管理器 - 企业级缓存"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self._enabled = False
        self._local_lock_guard = asyncio.Lock()
        self._local_locks: dict[str, str] = {}
        
    async def init(self, redis_url: str = "redis://localhost:6379/0", enabled: bool = True):
        """初始化 Redis 连接
        
        Args:
            redis_url: Redis 连接URL
            enabled: 是否启用Redis（如果为False，降级为无缓存模式）
        """
        self._enabled = enabled
        
        if not enabled:
            logger.warning("⚠️ Redis 缓存已禁用，将使用无缓存模式")
            return
            
        try:
            self.redis_client = await redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=False,  # 我们手动处理序列化
                socket_connect_timeout=5,
                socket_keepalive=True,
                max_connections=50
            )
            # 测试连接
            await self.redis_client.ping()
            logger.info("✅ Redis 连接成功")
        except Exception as e:
            logger.error(f"❌ Redis 连接失败: {e}，降级为无缓存模式")
            self._enabled = False
            self.redis_client = None
    
    async def close(self):
        """关闭 Redis 连接"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("🔌 Redis 连接已关闭")
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if not self._enabled or not self.redis_client:
            return None
            
        try:
            value = await self.redis_client.get(key)
            if value:
                # 尝试反序列化
                try:
                    return pickle.loads(value)
                except:
                    return value.decode('utf-8')
            return None
        except Exception as e:
            logger.warning(f"Redis GET 失败 [{key}]: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 300):
        """设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值（自动序列化）
            ttl: 过期时间（秒），默认5分钟
        """
        if not self._enabled or not self.redis_client:
            return False
            
        try:
            # 序列化值
            if isinstance(value, (str, int, float)):
                serialized = str(value).encode('utf-8')
            else:
                serialized = pickle.dumps(value)
                
            await self.redis_client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.warning(f"Redis SET 失败 [{key}]: {e}")
            return False
    
    async def delete(self, *keys: str):
        """删除缓存"""
        if not self._enabled or not self.redis_client:
            return False
            
        try:
            await self.redis_client.delete(*keys)
            return True
        except Exception as e:
            logger.warning(f"Redis DELETE 失败: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        if not self._enabled or not self.redis_client:
            return False
            
        try:
            return await self.redis_client.exists(key) > 0
        except Exception as e:
            logger.warning(f"Redis EXISTS 失败 [{key}]: {e}")
            return False

    async def acquire_lock(self, key: str, ttl: int = 600) -> str | None:
        """非阻塞获取带所有权令牌的锁；Redis 不可用时退化为进程内锁。"""
        token = uuid.uuid4().hex
        if self._enabled and self.redis_client:
            try:
                acquired = await self.redis_client.set(
                    key,
                    token.encode("utf-8"),
                    ex=ttl,
                    nx=True,
                )
                return f"redis:{token}" if acquired else None
            except Exception as exc:
                logger.warning("Redis LOCK 失败 [%s]: %s，退化为进程内锁", key, exc)

        async with self._local_lock_guard:
            if key in self._local_locks:
                return None
            self._local_locks[key] = token
        return f"local:{token}"

    async def release_lock(self, key: str, lease: str | None) -> bool:
        """只允许锁持有者释放，避免旧请求误删新请求的锁。"""
        if not lease or ":" not in lease:
            return False
        backend, token = lease.split(":", 1)
        if backend == "redis" and self.redis_client:
            try:
                released = await self.redis_client.eval(
                    """
                    if redis.call('get', KEYS[1]) == ARGV[1] then
                        return redis.call('del', KEYS[1])
                    end
                    return 0
                    """,
                    1,
                    key,
                    token.encode("utf-8"),
                )
                return bool(released)
            except Exception as exc:
                logger.warning("Redis UNLOCK 失败 [%s]: %s", key, exc)
                return False

        if backend == "local":
            async with self._local_lock_guard:
                if self._local_locks.get(key) != token:
                    return False
                del self._local_locks[key]
                return True
        return False
    
    async def mget(self, *keys: str) -> list:
        """批量获取多个键"""
        if not self._enabled or not self.redis_client:
            return [None] * len(keys)
            
        try:
            values = await self.redis_client.mget(*keys)
            result = []
            for value in values:
                if value:
                    try:
                        result.append(pickle.loads(value))
                    except:
                        result.append(value.decode('utf-8'))
                else:
                    result.append(None)
            return result
        except Exception as e:
            logger.warning(f"Redis MGET 失败: {e}")
            return [None] * len(keys)
    
    def cache_key(self, prefix: str, *args) -> str:
        """生成缓存键
        
        Example:
            cache_key("user_config", user_id, provider_id)
            -> "user_config:123:456"
        """
        parts = [str(prefix)] + [str(arg) for arg in args]
        return ":".join(parts)


# 全局 Redis 管理器实例
redis_manager = RedisManager()


# 缓存装饰器
def cached(ttl: int = 300, key_prefix: str = "cache"):
    """缓存装饰器
    
    Args:
        ttl: 缓存过期时间（秒）
        key_prefix: 缓存键前缀
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 生成缓存键（基于函数名和参数）
            key_parts = [key_prefix, func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            cache_key = ":".join(key_parts)
            
            # 尝试从缓存获取
            cached_value = await redis_manager.get(cache_key)
            if cached_value is not None:
                logger.debug(f"🎯 缓存命中: {cache_key}")
                return cached_value
            
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 存入缓存
            await redis_manager.set(cache_key, result, ttl)
            logger.debug(f"💾 缓存更新: {cache_key}")
            
            return result
        return wrapper
    return decorator
