#!/usr/bin/env python3
"""
Teste Específico de Cache - Mesmo Ticker
Verifica se cache está sendo reutilizado corretamente
"""
import requests
import time

def test_cache_efficiency():
    print('🔥 TESTE ESPECÍFICO DE CACHE')
    print('=' * 40)
    
    ticker_data = {'ticker': 'PETR4.SA', 'days_forecast': 15}
    
    print(f"📊 Testando cache para {ticker_data['ticker']}")
    print(f"📅 Forecast: {ticker_data['days_forecast']} dias")
    
    times = []
    
    # Fazer 5 chamadas sequenciais para o mesmo ticker
    for i in range(5):
        print(f"\n🔄 Chamada {i+1}/5...")
        
        start_time = time.time()
        response = requests.post('http://localhost:8000/chart-data', json=ticker_data)
        exec_time = time.time() - start_time
        
        if response.status_code == 200:
            times.append(exec_time)
            data = response.json()
            
            print(f"⏱️  Tempo: {exec_time:.2f}s")
            print(f"🎯 Rec: {data.get('analysis', {}).get('recommendation', 'N/A')}")
            
            # Verificar se indica cache hit
            performance = data.get('performance', {})
            cache_hit = performance.get('cache_hit', False)
            print(f"💾 Cache Hit: {cache_hit}")
        else:
            print(f"❌ Erro: {response.status_code}")
            break
        
        # Pequeno delay para separar as chamadas
        time.sleep(0.2)
    
    # Análise dos resultados
    print('\n📊 ANÁLISE DE CACHE:')
    if len(times) >= 2:
        print(f"🐌 1ª chamada: {times[0]:.2f}s")
        print(f"⚡ 2ª chamada: {times[1]:.2f}s")
        
        if len(times) >= 3:
            avg_cached = sum(times[1:]) / (len(times) - 1)
            print(f"💾 Média cached: {avg_cached:.2f}s")
            
            speedup = times[0] / avg_cached if avg_cached > 0 else 1
            print(f"🚀 Speedup médio: {speedup:.1f}x")
            
            if speedup > 3:
                print("🏆 CACHE EXCELENTE!")
            elif speedup > 1.5:
                print("✅ Cache funcionando bem")
            else:
                print("⚠️ Cache precisa melhorar")
        
        # Verificar consistência dos tempos cached
        if len(times) >= 3:
            cached_times = times[1:]
            max_cached = max(cached_times)
            min_cached = min(cached_times)
            
            print(f"\n📈 CONSISTÊNCIA:")
            print(f"⚡ Tempo mín cached: {min_cached:.2f}s")
            print(f"🐌 Tempo máx cached: {max_cached:.2f}s")
            
            if max_cached - min_cached < 0.5:
                print("✅ Cache muito consistente!")
            elif max_cached - min_cached < 1.0:
                print("👍 Cache razoavelmente consistente")
            else:
                print("⚠️ Cache inconsistente")

if __name__ == "__main__":
    test_cache_efficiency()