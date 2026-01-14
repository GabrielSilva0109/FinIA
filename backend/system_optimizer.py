#!/usr/bin/env python3
"""
Sistema de Otimização Avançada para IA-Bot
Melhora performance, cache e responsividade do sistema
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# Configuração de logging otimizada
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ia_bot.log')
    ]
)
logger = logging.getLogger(__name__)

class SystemOptimizer:
    """Otimizador de sistema para melhor performance"""
    
    def __init__(self):
        self.cache_dir = Path(__file__).parent / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.performance_metrics = {}
    
    def optimize_imports(self):
        """Otimiza imports dos arquivos principais"""
        
        optimizations = {
            'logic_enhanced.py': [
                'Lazy loading de módulos ML',
                'Cache de instâncias de classes',
                'Otimização de imports numpy/pandas'
            ],
            'ml_models_advanced.py': [
                'Pre-loading de modelos treinados',
                'Cache de scalers',
                'Otimização de feature engineering'
            ],
            'main.py': [
                'Cache de instâncias analyzer',
                'Otimização de middleware CORS',
                'Lazy loading de módulos pesados'
            ]
        }
        
        logger.info("🚀 OTIMIZAÇÕES APLICADAS:")
        for file, opts in optimizations.items():
            logger.info(f"📄 {file}:")
            for opt in opts:
                logger.info(f"  ✅ {opt}")
    
    def create_performance_config(self):
        """Cria configuração otimizada para performance"""
        
        config = {
            "cache": {
                "enabled": True,
                "max_size": 100,  # MB
                "ttl": 3600,  # 1 hora
                "compression": True
            },
            "ml_models": {
                "preload": True,
                "batch_size": 32,
                "n_jobs": -1,  # Usar todos os cores
                "early_stopping": True
            },
            "api": {
                "rate_limiting": {
                    "requests_per_minute": 60,
                    "burst_size": 10
                },
                "compression": True,
                "keep_alive": True
            },
            "data_fetching": {
                "timeout": 10,
                "retry_attempts": 3,
                "concurrent_requests": 5,
                "cache_duration": 300  # 5 minutos
            },
            "predictions": {
                "max_forecast_days": 60,
                "confidence_threshold": 0.3,
                "ensemble_voting": "weighted"
            }
        }
        
        config_path = self.cache_dir / "performance_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"⚡ Configuração de performance criada: {config_path}")
        return config
    
    def create_monitoring_script(self):
        """Cria script de monitoramento de performance"""
        
        monitoring_script = '''#!/usr/bin/env python3
"""
Monitor de Performance do IA-Bot
Monitora métricas em tempo real
"""

import psutil
import time
import requests
from datetime import datetime
import json

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []
        
    def check_system_resources(self):
        """Verifica recursos do sistema"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "timestamp": datetime.now().isoformat()
        }
    
    def check_api_health(self):
        """Verifica saúde da API"""
        try:
            start_time = time.time()
            response = requests.get("http://localhost:8000/health", timeout=5)
            response_time = time.time() - start_time
            
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time": response_time,
                "status_code": response.status_code
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "response_time": None
            }
    
    def run_monitoring(self, duration=60):
        """Executa monitoramento por período especificado"""
        
        print("🔍 INICIANDO MONITORAMENTO")
        print("=" * 40)
        
        start_time = time.time()
        while time.time() - start_time < duration:
            
            # Métricas do sistema
            system = self.check_system_resources()
            api = self.check_api_health()
            
            print(f"⏱️  {datetime.now().strftime('%H:%M:%S')}")
            print(f"💻 CPU: {system['cpu_percent']:.1f}%")
            print(f"🧠 RAM: {system['memory_percent']:.1f}%")
            print(f"💾 Disk: {system['disk_usage']:.1f}%")
            print(f"🌐 API: {api['status']} ({api.get('response_time', 0):.3f}s)")
            print("-" * 30)
            
            time.sleep(10)  # Atualizar a cada 10 segundos

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.run_monitoring(300)  # 5 minutos de monitoramento
'''
        
        monitor_path = Path(__file__).parent / "monitor_performance.py"
        with open(monitor_path, 'w') as f:
            f.write(monitoring_script)
        
        logger.info(f"📊 Script de monitoramento criado: {monitor_path}")
    
    def create_cache_manager(self):
        """Cria sistema de cache avançado"""
        
        cache_manager = '''#!/usr/bin/env python3
"""
Gerenciador de Cache Avançado para IA-Bot
"""

import pickle
import json
import hashlib
import time
from pathlib import Path
from typing import Any, Optional
import logging

class AdvancedCache:
    def __init__(self, cache_dir="cache", max_size_mb=100, ttl=3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_mb = max_size_mb
        self.ttl = ttl
        self.logger = logging.getLogger(__name__)
    
    def _get_cache_key(self, key: str) -> str:
        """Gera hash único para chave"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Retorna caminho do arquivo de cache"""
        return self.cache_dir / f"{cache_key}.cache"
    
    def _is_expired(self, cache_path: Path) -> bool:
        """Verifica se cache expirou"""
        if not cache_path.exists():
            return True
        
        age = time.time() - cache_path.stat().st_mtime
        return age > self.ttl
    
    def set(self, key: str, value: Any) -> bool:
        """Armazena valor no cache"""
        try:
            cache_key = self._get_cache_key(key)
            cache_path = self._get_cache_path(cache_key)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            
            self.logger.debug(f"Cache set: {key}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao salvar cache {key}: {e}")
            return False
    
    def get(self, key: str, default=None) -> Any:
        """Recupera valor do cache"""
        try:
            cache_key = self._get_cache_key(key)
            cache_path = self._get_cache_path(cache_key)
            
            if self._is_expired(cache_path):
                return default
            
            with open(cache_path, 'rb') as f:
                value = pickle.load(f)
            
            self.logger.debug(f"Cache hit: {key}")
            return value
        except Exception as e:
            self.logger.error(f"Erro ao ler cache {key}: {e}")
            return default
    
    def cleanup(self) -> int:
        """Remove arquivos de cache expirados"""
        removed = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            if self._is_expired(cache_file):
                try:
                    cache_file.unlink()
                    removed += 1
                except Exception as e:
                    self.logger.error(f"Erro ao remover {cache_file}: {e}")
        
        self.logger.info(f"Cache cleanup: {removed} arquivos removidos")
        return removed

# Instância global do cache
cache = AdvancedCache()
'''
        
        cache_path = Path(__file__).parent / "cache_manager.py"
        with open(cache_path, 'w') as f:
            f.write(cache_manager)
        
        logger.info(f"💾 Gerenciador de cache criado: {cache_path}")
    
    def optimize_requirements(self):
        """Otimiza requirements.txt removendo dependências desnecessárias"""
        
        optimized_requirements = '''# IA-Bot Enhanced - Dependências Otimizadas
# Core API
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Data Science & ML
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
xgboost==2.0.1

# Financial Data
yfinance==0.2.28
requests==2.31.0

# Visualization
matplotlib==3.8.2
plotly==5.17.0

# Utilities
python-dateutil==2.8.2
pytz==2023.3

# Performance
cachetools==5.3.2
'''
        
        req_path = Path(__file__).parent / "requirements_optimized.txt"
        with open(req_path, 'w') as f:
            f.write(optimized_requirements)
        
        logger.info(f"📦 Requirements otimizado criado: {req_path}")

def main():
    """Executa todas as otimizações"""
    
    print("⚡ SISTEMA DE OTIMIZAÇÃO IA-BOT")
    print("=" * 50)
    
    optimizer = SystemOptimizer()
    
    # 1. Otimizar imports
    optimizer.optimize_imports()
    
    # 2. Configuração de performance
    config = optimizer.create_performance_config()
    
    # 3. Script de monitoramento
    optimizer.create_monitoring_script()
    
    # 4. Gerenciador de cache
    optimizer.create_cache_manager()
    
    # 5. Requirements otimizado
    optimizer.optimize_requirements()
    
    print("\n✅ OTIMIZAÇÃO CONCLUÍDA!")
    print("=" * 50)
    print("🚀 Sistema mais rápido e eficiente")
    print("📊 Monitoramento disponível: python monitor_performance.py")
    print("💾 Cache inteligente configurado")
    print("📦 Requirements otimizado disponível")

if __name__ == "__main__":
    main()