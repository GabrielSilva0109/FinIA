#!/usr/bin/env python3
"""
Teste do Sistema Redis Cache
Verifica performance e funcionalidade
"""

import sys
sys.path.append('.')

import time
import requests
from redis_cache import cache_manager

def test_redis_system():
    print('🔥 TESTE COMPLETO DO SISTEMA REDIS')
    print('=' * 50)
    
    # 1. Testar Redis Cache diretamente
    print('1️⃣ TESTANDO REDIS CACHE DIRETO...')
    
    # Health check
    health = cache_manager.health_check()
    print(f"🏥 Redis Health: {health}")
    
    # Stats
    stats = cache_manager.get_stats()
    print(f"📊 Cache Stats: {stats}")
    
    # Teste básico de set/get
    test_key = "test_redis_performance"
    test_value = {"test": "performance", "timestamp": time.time()}
    
    start_time = time.time()
    cache_manager.set(test_key, test_value, ttl=60)
    set_time = (time.time() - start_time) * 1000
    
    start_time = time.time()
    retrieved_value = cache_manager.get(test_key)
    get_time = (time.time() - start_time) * 1000
    
    print(f"⚡ SET tempo: {set_time:.2f}ms")
    print(f"🚀 GET tempo: {get_time:.2f}ms")
    print(f"✅ Valor recuperado corretamente: {retrieved_value == test_value}")
    
    # 2. Testar Health Endpoint
    print('\n2️⃣ TESTANDO HEALTH ENDPOINT...')
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health OK: {health_data.get('cache', {}).get('redis_connected', False)}")
            print(f"📊 Features: {health_data.get('features_active', [])}")
        else:
            print(f"❌ Health endpoint erro: {response.status_code}")
    except Exception as e:
        print(f"❌ Erro ao acessar health: {e}")
    
    # 3. Testar Cache Stats Endpoint
    print('\n3️⃣ TESTANDO CACHE STATS ENDPOINT...')
    
    try:
        response = requests.get("http://localhost:8000/cache/stats", timeout=10)
        if response.status_code == 200:
            stats_data = response.json()
            print(f"📈 Cache Type: {stats_data.get('cache_stats', {}).get('type', 'unknown')}")
            print(f"🔥 Redis Available: {stats_data.get('cache_stats', {}).get('redis_available', False)}")
        else:
            print(f"❌ Stats endpoint erro: {response.status_code}")
    except Exception as e:
        print(f"❌ Erro ao acessar stats: {e}")
    
    # 4. Testar Performance da API com Redis
    print('\n4️⃣ TESTANDO PERFORMANCE API COM REDIS...')
    
    ticker_data = {'ticker': 'PETR4.SA', 'days_forecast': 15}
    
    # Primeira chamada (sem cache)
    print('🔥 Primeira chamada (cold cache)...')
    start_time = time.time()
    try:
        response = requests.post("http://localhost:8000/chart-data", json=ticker_data, timeout=30)
        first_call_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"⏱️  Tempo primeira chamada: {first_call_time:.2f}s")
        else:
            print(f"❌ Erro na primeira chamada: {response.status_code}")
            return
            
    except Exception as e:
        print(f"❌ Erro na primeira chamada: {e}")
        return
    
    # Segunda chamada (com Redis cache)
    print('🚀 Segunda chamada (Redis cache)...')
    start_time = time.time()
    try:
        response = requests.post("http://localhost:8000/chart-data", json=ticker_data, timeout=30)
        second_call_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"⚡ Tempo segunda chamada: {second_call_time:.2f}s")
            
            if second_call_time > 0:
                speedup = first_call_time / second_call_time
                print(f"🚀 SPEEDUP REDIS: {speedup:.1f}x mais rápido!")
                
                if speedup > 2:
                    print("🏆 REDIS FUNCIONANDO PERFEITAMENTE!")
                elif speedup > 1.2:
                    print("✅ Redis funcionando bem")
                else:
                    print("⚠️ Redis pode não estar sendo usado corretamente")
            
        else:
            print(f"❌ Erro na segunda chamada: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Erro na segunda chamada: {e}")
    
    # 5. Comparação final
    print('\n📊 RESUMO FINAL:')
    print(f"🔥 Cache direto SET: {set_time:.2f}ms")
    print(f"⚡ Cache direto GET: {get_time:.2f}ms")
    print(f"🐌 API sem cache: {first_call_time:.2f}s")
    print(f"🚀 API com Redis: {second_call_time:.2f}s")
    
    if 'first_call_time' in locals() and 'second_call_time' in locals() and second_call_time > 0:
        total_speedup = first_call_time / second_call_time
        if total_speedup > 3:
            print(f"🎯 RESULTADO: EXCELENTE! {total_speedup:.1f}x speedup")
        elif total_speedup > 2:
            print(f"✅ RESULTADO: BOM! {total_speedup:.1f}x speedup")
        else:
            print(f"⚠️ RESULTADO: OK. {total_speedup:.1f}x speedup")

if __name__ == "__main__":
    test_redis_system()