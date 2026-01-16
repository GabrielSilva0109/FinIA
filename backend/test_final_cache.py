#!/usr/bin/env python3
"""
Teste Final do Sistema Redis/Cache Local
"""
import requests
import time

def test_final_system():
    print('🎯 TESTE FINAL - Sistema Redis/Cache Local')
    print('=' * 50)
    
    # 1. Health Check
    print('1️⃣ HEALTH CHECK...')
    try:
        health = requests.get('http://localhost:8000/health').json()
        cache_info = health.get('cache', {})
        
        print(f"✅ Status: {health.get('status', 'unknown')}")
        print(f"🔥 Redis: {cache_info.get('redis_connected', False)}")
        print(f"💾 Local Cache: {cache_info.get('local_cache_active', False)}")
        print(f"📊 Total Keys: {cache_info.get('total_keys', 0)}")
        
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # 2. Cache Stats
    print('\n2️⃣ CACHE STATS...')
    try:
        stats = requests.get('http://localhost:8000/cache/stats').json()
        cache_stats = stats.get('cache_stats', {})
        
        print(f"📈 Cache Type: {cache_stats.get('type', 'unknown')}")
        print(f"🔥 Redis Available: {cache_stats.get('redis_available', False)}")
        print(f"💾 Local Entries: {cache_stats.get('local_cache_size', 0)}")
        
    except Exception as e:
        print(f"❌ Cache stats error: {e}")
    
    # 3. Performance Test Detalhado
    print('\n3️⃣ PERFORMANCE TEST DETALHADO...')
    
    test_cases = [
        {'ticker': 'PETR4.SA', 'days_forecast': 15},
        {'ticker': 'VALE3.SA', 'days_forecast': 10},
        {'ticker': 'ITUB4.SA', 'days_forecast': 20}
    ]
    
    results = []
    
    for i, case in enumerate(test_cases):
        print(f"\n📊 Teste {i+1}: {case['ticker']}")
        
        # Primeira chamada (cold)
        start_time = time.time()
        response1 = requests.post('http://localhost:8000/chart-data', json=case)
        cold_time = time.time() - start_time
        
        if response1.status_code == 200:
            data1 = response1.json()
            print(f"⏱️  Cold: {cold_time:.2f}s")
            print(f"🎯 Rec: {data1.get('analysis', {}).get('recommendation', 'N/A')}")
            
            # Segunda chamada (warm) - pouco delay para garantir diferença
            time.sleep(0.1)
            start_time = time.time()
            response2 = requests.post('http://localhost:8000/chart-data', json=case)
            warm_time = time.time() - start_time
            
            if response2.status_code == 200:
                speedup = cold_time / warm_time if warm_time > 0 else 1
                print(f"⚡ Warm: {warm_time:.2f}s")
                print(f"🚀 Speedup: {speedup:.1f}x")
                
                results.append({
                    'ticker': case['ticker'],
                    'cold': cold_time,
                    'warm': warm_time,
                    'speedup': speedup
                })
        else:
            print(f"❌ Error: {response1.status_code}")
    
    # 4. Resumo Final
    print('\n📊 RESUMO FINAL:')
    if results:
        avg_cold = sum(r['cold'] for r in results) / len(results)
        avg_warm = sum(r['warm'] for r in results) / len(results)
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        
        print(f"🐌 Avg Cold: {avg_cold:.2f}s")
        print(f"⚡ Avg Warm: {avg_warm:.2f}s")
        print(f"🚀 Avg Speedup: {avg_speedup:.1f}x")
        
        if avg_speedup > 5:
            performance = "🏆 EXCELENTE"
        elif avg_speedup > 2:
            performance = "✅ BOM"
        elif avg_speedup > 1.2:
            performance = "👍 OK"
        else:
            performance = "⚠️ PRECISA MELHORAR"
            
        print(f"📈 PERFORMANCE: {performance}")
    
    print(f"\n🎉 TESTE CONCLUÍDO!")
    
if __name__ == "__main__":
    test_final_system()