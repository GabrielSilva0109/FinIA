import os
from typing import Optional

class Settings:
    """Configurações da aplicação."""
    
    # API Keys e URLs
    YAHOO_FINANCE_URL: str = "https://query1.finance.yahoo.com/v8/finance/chart"
    COINGECKO_URL: str = "https://api.coingecko.com/api/v3"
    
    # Timeouts
    REQUEST_TIMEOUT: int = 10
    
    # CORS
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8080"]
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Cache
    CACHE_TTL: int = 300  # 5 minutos
    
    # Exchange API (opcional para crypto)
    EXCHANGE_API_KEY: Optional[str] = os.getenv("EXCHANGE_API_KEY")
    EXCHANGE_SECRET: Optional[str] = os.getenv("EXCHANGE_SECRET")
    EXCHANGE_SECRET_KEY: Optional[str] = os.getenv("EXCHANGE_SECRET_KEY")
    
    # Limites de dados
    MAX_HISTORICAL_DAYS: int = 365
    MIN_VOLUME_THRESHOLD: float = 10000
    
    # Modelos de ML
    ML_MODELS_PATH: str = "./models"
    DEFAULT_TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    
    # Performance
    MAX_CONCURRENT_REQUESTS: int = 10
    
    # Redis Cache
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_ENABLED: bool = True
    CACHE_TTL: int = 3600  # 1 hora em segundos
    REDIS_TIMEOUT: int = 5  # timeout para conexões Redis

settings = Settings()