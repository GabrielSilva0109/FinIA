#!/usr/bin/env python3
"""
Teste Final - HTTP com Previsões Realísticas
"""

import requests
import time
import json

def test_http_realistic():
    print('🌐 TESTE FINAL - HTTP com Previsões Realísticas')
    print('=' * 55)

    # Teste com PETR4
    url = "http://localhost:8000/chart-data"
    data = {
        'ticker': 'PETR4.SA',
        'days_forecast': 15
    }
    
    print(f'🚀 Testando {data["ticker"]} via HTTP...')
    
    start_time = time.time()
    
    try:
        response = requests.post(url, json=data, timeout=30)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f'⏱️  TEMPO: {processing_time:.2f} segundos')
        
        if response.status_code == 200:
            result = response.json()
            predictions = result.get('prediction_data', [])
            
            print(f'✅ SUCESSO! Método: {predictions[0].get("method", "unknown") if predictions else "N/A"}')
            
            # Analisar oscilações
            if len(predictions) >= 10:
                print('\n📈 PRIMEIRAS 10 PREVISÕES:')
                
                last_price = result.get('historical_data', [{}])[-1].get('close', 0)
                changes = []
                
                for i, pred in enumerate(predictions[:10]):
                    price = pred['predicted_price']
                    date = pred['date']
                    
                    if i == 0:
                        change = price - last_price
                        change_pct = (change / last_price) * 100
                    else:
                        prev_price = predictions[i-1]['predicted_price']
                        change = price - prev_price
                        change_pct = (change / prev_price) * 100
                    
                    changes.append(change)
                    direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    print(f'{date}: R$ {price:.2f} {direction} {change_pct:+.1f}%')
                
                # Estatísticas
                positive = sum(1 for c in changes if c > 0)
                negative = sum(1 for c in changes if c < 0)
                neutral = sum(1 for c in changes if abs(c) < 0.01)
                
                print(f'\n📊 RESULTADO:')
                print(f'🔼 Subiu: {positive} dias')
                print(f'🔽 Desceu: {negative} dias')
                print(f'➡️ Estável: {neutral} dias')
                
                if positive > 0 and negative > 0:
                    print(f'🎯 ✅ OSCILAÇÕES REALÍSTICAS!')
                else:
                    print(f'⚠️ Ainda muito linear')
                    
        else:
            print(f'❌ Erro: {response.status_code}')
            
    except Exception as e:
        print(f'❌ Erro: {e}')

if __name__ == "__main__":
    test_http_realistic()