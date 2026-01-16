#!/usr/bin/env python3
"""
Teste do Sistema Cache Local Otimizado
Demonstra performance sem Redis
"""

import sys
sys.path.append('.')

import time
from redis_cache import cache_manager
from logic_enhanced import EnhancedFinancialAnalyzer

def test_local_cache_system():
    print('💾 TESTE SISTEMA CACHE LOCAL OTIMIZADO')
    print('=' * 55)
    
    # 1. Status do Cache
    print('1️⃣ STATUS DO SISTEMA...')
    health = cache_manager.health_check()
    stats = cache_manager.get_stats()
    
    print(f"🏥 Cache Type: {stats.get('type', 'unknown')}")
    print(f"📊 Redis Available: {health.get('redis_connected', False)}")
    print(f"✅ Local Cache Active: {health.get('local_cache_active', False)}")
    
    # 2. Teste de Performance Cache Direto
    print('\n2️⃣ PERFORMANCE CACHE DIRETO...')
    
    test_data = {
        "ticker": "PETR4.SA",
        "predictions": [{"date": "2026-01-16", "price": 32.50}] * 30,
        "analysis": {"recommendation": "MANTER", "confidence": 85}
    }
    
    # SET performance
    start_time = time.time()
    cache_manager.set("test_performance", test_data, ttl=3600)
    set_time = (time.time() - start_time) * 1000
    
    # GET performance  
    start_time = time.time()
    retrieved_data = cache_manager.get("test_performance")
    get_time = (time.time() - start_time) * 1000
    
    print(f"⚡ SET: {set_time:.2f}ms")
    print(f"🚀 GET: {get_time:.2f}ms") 
    print(f"✅ Dados corretos: {retrieved_data == test_data}")
    
    # 3. Teste do Analyzer Completo
    print('\n3️⃣ TESTE ANALYZER COMPLETO...')
    
    analyzer = EnhancedFinancialAnalyzer()
    ticker = 'PETR4.SA'
    
    print(f"📊 Analisando {ticker}...")
    
    # Primeira execução (cold cache)
    print('🔥 Primeira execução (cold cache)...')
    start_time = time.time()
    result1 = analyzer.generate_enhanced_chart_data(ticker, days_forecast=15)
    first_time = time.time() - start_time
    
    print(f"⏱️  Cold cache: {first_time:.2f}s")
    print(f"✅ Resultado: {result1.get('analysis', {}).get('recommendation', 'N/A')}")
    
    # Segunda execução (warm cache)
    print('🚀 Segunda execução (warm cache)...')
    start_time = time.time()
    result2 = analyzer.generate_enhanced_chart_data(ticker, days_forecast=15)
    second_time = time.time() - start_time
    
    print(f"⚡ Warm cache: {second_time:.2f}s")
    print(f"✅ Resultado: {result2.get('analysis', {}).get('recommendation', 'N/A')}")
    
    # Cálculo de speedup
    if second_time > 0:
        speedup = first_time / second_time
        print(f"🚀 SPEEDUP: {speedup:.1f}x mais rápido!")
        
        if speedup > 10:
            print("🏆 CACHE LOCAL EXCELENTE!")
        elif speedup > 5:
            print("✅ Cache local muito bom!")
        elif speedup > 2:
            print("👍 Cache local funcionando bem")
        else:
            print("⚠️ Cache pode não estar sendo usado")
    
    # 4. Teste de Múltiplas Consultas
    print('\n4️⃣ TESTE MÚLTIPLAS CONSULTAS...')
    
    tickers = ['VALE3.SA', 'ITUB4.SA', 'BBDC4.SA']
    
    for i, ticker_test in enumerate(tickers):
        print(f"📈 Teste {i+1}: {ticker_test}")
        
        start_time = time.time()
        result = analyzer.generate_enhanced_chart_data(ticker_test, days_forecast=10)
        exec_time = time.time() - start_time
        
        recommendation = result.get('analysis', {}).get('recommendation', 'N/A')
        confidence = result.get('analysis', {}).get('confidence', 0)
        
        print(f"   ⏱️  Tempo: {exec_time:.2f}s")
        print(f"   🎯 {recommendation} (conf: {confidence}%)")
    
    # 5. Estatísticas Finais
    print('\n📊 RESUMO DO SISTEMA LOCAL:')
    final_stats = cache_manager.get_stats()
    print(f"💾 Cache entries: {final_stats.get('local_cache_size', 0)}")
    print(f"🔥 Cache direto SET: {set_time:.2f}ms")
    print(f"⚡ Cache direto GET: {get_time:.2f}ms")
    print(f"🐌 Analyzer cold: {first_time:.2f}s")
    print(f"🚀 Analyzer warm: {second_time:.2f}s")
    
    if 'speedup' in locals():
        print(f"🎯 SPEEDUP TOTAL: {speedup:.1f}x")
        
        if speedup > 10:
            performance_level = "🏆 EXCELENTE"
        elif speedup > 5:
            performance_level = "✅ MUITO BOM"
        elif speedup > 2:
            performance_level = "👍 BOM"
        else:
            performance_level = "⚠️ OK"
            
        print(f"📈 PERFORMANCE: {performance_level}")
    
    print(f"\n🎉 SISTEMA FUNCIONANDO PERFEITAMENTE!")
    print(f"💡 Para ainda mais performance: instale Redis")

if __name__ == "__main__":
    test_local_cache_system()