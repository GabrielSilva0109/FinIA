#!/usr/bin/env python3
"""
Teste da Nova Previsão Realística
Compara previsão linear vs oscilações realísticas
"""

import sys
sys.path.append('.')

from logic_enhanced import EnhancedFinancialAnalyzer
import time

def test_realistic_predictions():
    print('🔄 TESTE DE PREVISÕES REALÍSTICAS')
    print('=' * 50)
    
    analyzer = EnhancedFinancialAnalyzer()
    
    # Teste com PETR4 (mesmo exemplo do usuário)
    ticker = 'PETR4.SA'
    days = 10  # Teste menor primeiro
    
    print(f'📊 Testando {ticker} - {days} dias...')
    
    start_time = time.time()
    
    try:
        result = analyzer.generate_enhanced_chart_data(ticker, days_forecast=days)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f'⏱️  TEMPO: {processing_time:.2f} segundos')
        
        # Mostrar previsões
        predictions = result.get('prediction_data', [])
        historical = result.get('historical_data', [])
        
        if len(historical) > 0:
            last_real_price = historical[-1]['close']
            print(f'\n💰 Preço atual: R$ {last_real_price:.2f}')
            
            print(f'\n🔮 PREVISÕES REALÍSTICAS:')
            for i, pred in enumerate(predictions[:7]):  # Primeiros 7 dias
                date = pred['date']
                price = pred['predicted_price']
                method = pred.get('method', 'unknown')
                
                # Calcular mudança vs preço anterior
                if i == 0:
                    change = price - last_real_price
                    change_pct = (change / last_real_price) * 100
                else:
                    prev_price = predictions[i-1]['predicted_price']
                    change = price - prev_price
                    change_pct = (change / prev_price) * 100
                
                direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                
                print(f'{date}: R$ {price:.2f} {direction} {change_pct:+.1f}% ({method})')
            
            # Verificar se há oscilações
            prices = [p['predicted_price'] for p in predictions]
            changes = []
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                changes.append(change)
            
            positive_moves = sum(1 for c in changes if c > 0)
            negative_moves = sum(1 for c in changes if c < 0)
            
            print(f'\n📊 ANÁLISE DE OSCILAÇÕES:')
            print(f'🔼 Dias subindo: {positive_moves}')
            print(f'🔽 Dias descendo: {negative_moves}')
            
            if positive_moves > 0 and negative_moves > 0:
                print(f'✅ SUCESSO: Previsão com oscilações realísticas!')
            elif positive_moves == 0:
                print(f'⚠️  PROBLEMA: Só desce (muito pessimista)')
            elif negative_moves == 0:
                print(f'⚠️  PROBLEMA: Só sobe (muito otimista)')
            
        else:
            print('❌ Não conseguiu obter dados históricos')
            
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        print(f'❌ ERRO: {str(e)}')
        print(f'⏱️  Tempo até erro: {processing_time:.2f}s')

if __name__ == "__main__":
    test_realistic_predictions()