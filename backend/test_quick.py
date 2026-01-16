#!/usr/bin/env python3
"""
Teste Rápido do Sistema Corrigido
"""

import sys
sys.path.append('.')

import time
import requests

def test_quick():
    print('🔧 TESTE RÁPIDO - Sistema Corrigido')
    print('=' * 40)
    
    # Teste simples da API
    url = "http://localhost:8000/chart-data"
    data = {'ticker': 'PETR4.SA', 'days_forecast': 10}
    
    print('🚀 Testando API corrigida...')
    
    try:
        start_time = time.time()
        response = requests.post(url, json=data, timeout=20)
        exec_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f'✅ SUCESSO!')
            print(f'⏱️  Tempo: {exec_time:.2f}s')
            print(f'🎯 Recomendação: {result.get("analysis", {}).get("recommendation", "N/A")}')
            print(f'📊 Previsões: {len(result.get("prediction_data", []))} dias')
            print(f'🔧 Método: {result.get("prediction_data", [{}])[0].get("method", "N/A") if result.get("prediction_data") else "N/A"}')
            
            # Segunda chamada para testar cache
            print('\n🔥 Segunda chamada (cache test)...')
            start_time = time.time()
            response2 = requests.post(url, json=data, timeout=20)
            exec_time2 = time.time() - start_time
            
            if response2.status_code == 200:
                speedup = exec_time / exec_time2 if exec_time2 > 0 else 1
                print(f'⚡ Cache time: {exec_time2:.2f}s')
                print(f'🚀 Speedup: {speedup:.1f}x')
                
                if speedup > 2:
                    print('🎉 CACHE FUNCIONANDO!')
                else:
                    print('⚠️ Cache pode não estar ativo')
            
        else:
            print(f'❌ Erro: {response.status_code}')
            print(f'📝 Response: {response.text[:200]}...')
            
    except requests.exceptions.Timeout:
        print('⏰ Timeout - servidor pode estar lento')
    except requests.exceptions.ConnectionError:
        print('🚫 Erro de conexão - servidor não está rodando?')
    except Exception as e:
        print(f'❌ Erro: {e}')

if __name__ == "__main__":
    test_quick()