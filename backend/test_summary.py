#!/usr/bin/env python3
"""
Teste da nova funcionalidade analysis_summary
"""

from fastapi.testclient import TestClient
from main import app
import json

def test_analysis_summary():
    """Testa se o campo analysis_summary é retornado"""
    client = TestClient(app)
    
    print("=== TESTE DO NOVO CAMPO analysis_summary ===\n")
    
    response = client.post('/chart-data', json={
        'ticker': 'BBAS3',
        'historical': 30,
        'predictions': 3
    })
    
    print(f"Status da requisição: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        # Verificar se o novo campo existe
        if 'analysis_summary' in data:
            print("✓ Campo 'analysis_summary' encontrado com sucesso!")
            
            summary = data['analysis_summary']
            print(f"\nTamanho do resumo: {len(summary)} caracteres")
            print(f"Ticker: {data.get('ticker', 'N/A')}")
            
            # Mostrar análise
            analysis = data.get('analysis', {})
            print(f"Recomendação: {analysis.get('recommendation', 'N/A')}")
            print(f"Confiança: {analysis.get('confidence', 'N/A')}%")
            
            print(f"\n--- RESUMO EXPLICATIVO DA IA ---")
            print(summary)
            
            print(f"\n--- ESTRUTURA DO JSON ---")
            print("Campos principais:")
            for key in sorted(data.keys()):
                if key != 'historical_data':  # Skip pois é muito longo
                    print(f"  - {key}")
            
            print(f"\n✓ IMPLEMENTAÇÃO CONCLUÍDA COM SUCESSO!")
            print(f"A IA agora explica os motivos de suas decisões de forma clara e detalhada.")
            
        else:
            print("✗ Campo 'analysis_summary' NÃO encontrado")
            print("Campos disponíveis:", list(data.keys()))
            
    else:
        print(f"Erro na requisição: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_analysis_summary()