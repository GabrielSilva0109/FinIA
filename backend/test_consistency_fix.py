#!/usr/bin/env python3
"""
Teste específico para verificar se a inconsistência foi corrigida
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic_enhanced import EnhancedFinancialAnalyzer

def test_consistency_fix():
    """Testa se a inconsistência entre previsões e recomendações foi corrigida"""
    print("🧪 Testando correção da inconsistência lógica...")
    
    ticker = "BBAS3.SA"
    analyzer = EnhancedFinancialAnalyzer()
    
    # Limpar cache para forçar novo cálculo
    try:
        analyzer.cache.delete(f"analysis_{ticker}_25")
        analyzer.cache.delete(f"predictions_*")
        if hasattr(analyzer.cache, 'local_cache'):
            analyzer.cache.local_cache.clear()
    except:
        pass
    
    print(f"\n📊 Analisando {ticker} com lógica corrigida...")
    result = analyzer.generate_enhanced_chart_data(ticker, days_forecast=25)
    
    # Extrair dados principais
    current_price = result.get('analysis', {}).get('current_price', 0)
    predicted_price = result.get('analysis', {}).get('predicted_price', 0)
    price_change_pct = result.get('analysis', {}).get('price_change_percent', 0)
    recommendation = result.get('analysis', {}).get('recommendation', 'N/A')
    
    print(f"   💰 Preço atual: R$ {current_price:.2f}")
    print(f"   🔮 Previsão usada: R$ {predicted_price:.2f}")
    print(f"   📈 Mudança prevista: {price_change_pct:.1f}%")
    print(f"   🎯 Recomendação: {recommendation}")
    
    # Analisar previsões individuais para entender a tendência
    predictions = result.get('prediction_data', [])
    if len(predictions) >= 3:
        print(f"\n🔍 Análise das previsões:")
        
        first_pred = predictions[0]['predicted_price']
        mid_pred = predictions[len(predictions)//2]['predicted_price'] 
        last_pred = predictions[-1]['predicted_price']
        
        short_trend = ((first_pred - current_price) / current_price) * 100
        medium_trend = ((mid_pred - current_price) / current_price) * 100
        long_trend = ((last_pred - current_price) / current_price) * 100
        
        print(f"   📅 Primeira previsão: R$ {first_pred:.2f} ({short_trend:+.1f}%)")
        print(f"   📅 Previsão média: R$ {mid_pred:.2f} ({medium_trend:+.1f}%)")
        print(f"   📅 Última previsão: R$ {last_pred:.2f} ({long_trend:+.1f}%)")
        
        # Verificar consistência lógica
        print(f"\n🎯 Verificação de consistência:")
        
        # Se queda significativa prevista (>10%), deve recomendar VENDA
        if long_trend <= -10 and recommendation != "VENDER":
            print(f"   ❌ INCONSISTÊNCIA: Queda de {long_trend:.1f}% prevista, mas recomenda {recommendation}")
            return False
        elif long_trend <= -5 and recommendation not in ["VENDER"]:
            print(f"   ❌ INCONSISTÊNCIA: Queda moderada de {long_trend:.1f}% prevista, mas recomenda {recommendation}")
            return False
        elif long_trend >= 10 and recommendation != "COMPRAR":
            print(f"   ⚠️ POSSÍVEL INCONSISTÊNCIA: Alta de {long_trend:.1f}% prevista, mas recomenda {recommendation}")
        else:
            print(f"   ✅ CONSISTÊNCIA: Recomendação '{recommendation}' coerente com previsões")
            return True
    
    return True

if __name__ == "__main__":
    success = test_consistency_fix()
    exit(0 if success else 1)