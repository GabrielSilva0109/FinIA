"""
Redis Cache Manager - Sistema de cache distribuído para FinAI
Substitui cache em memória por Redis para persistência e performance
"""

import redis
import json
import pickle
import time
import logging
from typing import Any, Optional, Union, Dict
from datetime import datetime, timedelta
import pandas as pd
from config import settings

logger = logging.getLogger(__name__)

class RedisCache:
    """Gerenciador de cache Redis com fallback para memória local"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.local_cache: Dict[str, tuple] = {}  # Fallback cache
        self.is_redis_available = False
        
        if settings.REDIS_ENABLED:
            self._initialize_redis()
    
    def _initialize_redis(self):
        """Inicializa conexão Redis com tratamento de erro"""
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                socket_timeout=settings.REDIS_TIMEOUT,
                socket_connect_timeout=settings.REDIS_TIMEOUT,
                health_check_interval=30,
                decode_responses=False  # Para permitir pickle
            )
            
            # Teste de conexão
            self.redis_client.ping()
            self.is_redis_available = True
            logger.info("✅ Redis conectado com sucesso!")
            
        except Exception as e:
            logger.warning(f"⚠️ Redis não disponível, usando cache local: {e}")
            self.is_redis_available = False
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serializa valores para armazenamento Redis"""
        try:
            if isinstance(value, (pd.DataFrame, pd.Series)):
                # DataFrames: usar pickle para preservar tipos
                return pickle.dumps(value)
            elif isinstance(value, (dict, list, tuple)):
                # JSON para tipos simples (mais legível no Redis)
                return json.dumps(value, default=str).encode('utf-8')
            else:
                # Outros tipos: pickle
                return pickle.dumps(value)
        except Exception as e:
            logger.warning(f"Erro na serialização: {e}")
            return pickle.dumps(value)  # Fallback para pickle
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserializa valores do Redis"""
        try:
            # Tentar JSON primeiro (mais rápido)
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            try:
                # Fallback para pickle
                return pickle.loads(data)
            except Exception as e:
                logger.warning(f"Erro na deserialização: {e}")
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Armazena valor no cache com TTL"""
        ttl = ttl or settings.CACHE_TTL
        timestamp = time.time()
        
        if self.is_redis_available and self.redis_client:
            try:
                # Serializar valor
                serialized_value = self._serialize_value(value)
                
                # Armazenar no Redis com TTL
                success = self.redis_client.setex(
                    key, 
                    ttl, 
                    serialized_value
                )
                
                if success:
                    logger.debug(f"🔥 Redis SET: {key} (TTL: {ttl}s)")
                    return True
                    
            except Exception as e:
                logger.warning(f"Erro ao armazenar no Redis: {e}")
                # Fallback para cache local
                self.is_redis_available = False
        
        # Cache local como fallback
        self.local_cache[key] = (value, timestamp + ttl)
        logger.debug(f"💾 Local SET: {key}")
        return True
    
    def get(self, key: str) -> Optional[Any]:
        """Recupera valor do cache"""
        
        if self.is_redis_available and self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    value = self._deserialize_value(data)
                    logger.debug(f"🚀 Redis HIT: {key}")
                    return value
                else:
                    logger.debug(f"❌ Redis MISS: {key}")
                    return None
                    
            except Exception as e:
                logger.warning(f"Erro ao ler do Redis: {e}")
                # Fallback para cache local
                self.is_redis_available = False
        
        # Cache local
        if key in self.local_cache:
            value, expiry = self.local_cache[key]
            if time.time() < expiry:
                logger.debug(f"💾 Local HIT: {key}")
                return value
            else:
                # Expirado
                del self.local_cache[key]
                logger.debug(f"⏰ Local EXPIRED: {key}")
        
        return None
    
    def delete(self, key: str) -> bool:
        """Remove chave do cache"""
        deleted = False
        
        if self.is_redis_available and self.redis_client:
            try:
                result = self.redis_client.delete(key)
                deleted = bool(result)
                logger.debug(f"🗑️ Redis DELETE: {key}")
            except Exception as e:
                logger.warning(f"Erro ao deletar do Redis: {e}")
        
        # Cache local
        if key in self.local_cache:
            del self.local_cache[key]
            deleted = True
            logger.debug(f"🗑️ Local DELETE: {key}")
        
        return deleted
    
    def clear(self) -> bool:
        """Limpa todo o cache"""
        cleared = False
        
        if self.is_redis_available and self.redis_client:
            try:
                self.redis_client.flushdb()
                cleared = True
                logger.info("🧹 Redis cache limpo!")
            except Exception as e:
                logger.warning(f"Erro ao limpar Redis: {e}")
        
        # Cache local
        self.local_cache.clear()
        logger.info("🧹 Cache local limpo!")
        return cleared
    
    def get_stats(self) -> Dict[str, Any]:
        """Estatísticas do cache"""
        stats = {
            'redis_available': self.is_redis_available,
            'local_cache_size': len(self.local_cache),
            'type': 'redis' if self.is_redis_available else 'local'
        }
        
        if self.is_redis_available and self.redis_client:
            try:
                info = self.redis_client.info('memory')
                stats.update({
                    'redis_memory_used': info.get('used_memory_human', 'N/A'),
                    'redis_keys': self.redis_client.dbsize(),
                })
            except Exception as e:
                logger.warning(f"Erro ao obter stats Redis: {e}")
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Verifica saúde do cache"""
        health = {
            'redis_connected': False,
            'local_cache_active': True,
            'total_keys': len(self.local_cache),
            'timestamp': datetime.now().isoformat()
        }
        
        if self.redis_client:
            try:
                self.redis_client.ping()
                health['redis_connected'] = True
                health['total_keys'] += self.redis_client.dbsize()
                self.is_redis_available = True
            except Exception as e:
                health['redis_error'] = str(e)
                self.is_redis_available = False
        
        return health

# Instância global do cache
cache_manager = RedisCache()


# Funções de conveniência para compatibilidade
def get_from_cache(key: str) -> Optional[Any]:
    """Função helper para obter do cache"""
    return cache_manager.get(key)


def set_in_cache(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """Função helper para armazenar no cache"""
    return cache_manager.set(key, value, ttl)


def clear_cache() -> bool:
    """Função helper para limpar cache"""
    return cache_manager.clear()