#!/usr/bin/env python3
"""
Teste para verificar se a API está funcionando corretamente
"""

import requests
import json

def test_api():
    print('=== TESTE DA API IA-BOT v3.0 ===')
    
    API_BASE_URL = 'http://localhost:8000'
    
    # Teste 1: Health check
    print('1. Testando /health:')
    try:
        response = requests.get(f'{API_BASE_URL}/health', timeout=5)
        print(f'   Status: {response.status_code}')
        if response.ok:
            print('   ✅ API disponível!')
            data = response.json()
            print(f'   Service: {data.get("service")}')
            print(f'   Version: {data.get("version")}')
        else:
            print('   ❌ API não disponível')
            return False
    except Exception as e:
        print(f'   ❌ Erro: {str(e)}')
        return False
    
    print('')
    
    # Teste 2: Chart-data (formato frontend)
    print('2. Testando /chart-data (formato frontend):')
    try:
        frontend_data = {
            'ticker': 'AAPL',
            'days_forecast': 5
        }
        
        response = requests.post(
            f'{API_BASE_URL}/chart-data',
            json=frontend_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        print(f'   Status: {response.status_code}')
        
        if response.ok:
            apiData = response.json()
            print('   ✅ Sucesso com dados da API!')
            print(f'   Ticker: {apiData.get("ticker")}')
            print(f'   Historical: {len(apiData.get("historical_data", []))} pontos')
            print(f'   Predictions: {len(apiData.get("prediction_data", []))} pontos')
            print(f'   API Version: {apiData.get("api_version")}')
            
            # Verificar campos essenciais para o frontend
            has_historical = bool(apiData.get('historical_data'))
            has_predictions = bool(apiData.get('prediction_data'))
            has_analysis = bool(apiData.get('analysis'))
            
            print(f'   Historical data: {"✅" if has_historical else "❌"}')
            print(f'   Prediction data: {"✅" if has_predictions else "❌"}')
            print(f'   Analysis data: {"✅" if has_analysis else "❌"}')
            
            if has_historical and has_predictions and has_analysis:
                print('   🎉 ESTRUTURA COMPATÍVEL COM FRONTEND!')
            else:
                print('   ⚠️ Estrutura incompleta')
                
        else:
            print(f'   ❌ Erro: {response.status_code}')
            print(f'   Texto: {response.text}')
            
    except Exception as error:
        print(f'   ❌ Erro na requisição: {str(error)}')
    
    print('')
    
    # Teste 3: Testar ticker brasileiro
    print('3. Testando ticker brasileiro (PETR4):')
    try:
        response = requests.post(
            f'{API_BASE_URL}/chart-data',
            json={'ticker': 'PETR4', 'days_forecast': 3},
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.ok:
            data = response.json()
            print('   ✅ Ticker brasileiro funcionando!')
            print(f'   Ticker processado: {data.get("processed_ticker")}')
            print(f'   Auto-correção: {"✅" if "PETR4.SA" in data.get("processed_ticker", "") else "❌"}')
        else:
            print(f'   ❌ Erro: {response.status_code}')
            
    except Exception as e:
        print(f'   ❌ Erro: {str(e)}')
    
    print('')
    print('=== TESTE COMPLETO ===')

if __name__ == '__main__':
    test_api()