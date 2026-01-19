#!/usr/bin/env python3
"""
Debug profundo da inconsistência: IA recomenda COMPRA mas ativo vai CAIR
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic_enhanced import EnhancedFinancialAnalyzer

def debug_recommendation_logic():
    """Debug detalhado da lógica de recomendação"""
    print("🔬 DEBUG: Por que IA recomenda COMPRA quando ativo vai CAIR?")
    
    ticker = "BBAS3.SA"
    analyzer = EnhancedFinancialAnalyzer()
    
    # Limpar cache completamente
    try:
        analyzer.cache.delete(f"analysis_{ticker}_25")
        analyzer.cache.delete(f"predictions_*")
        if hasattr(analyzer.cache, 'local_cache'):
            analyzer.cache.local_cache.clear()
            print("Cache limpo")
    except:
        pass
    
    print(f"\n📊 Analisando {ticker} com debug completo...")
    result = analyzer.generate_enhanced_chart_data(ticker, days_forecast=25)
    
    # ===== 1. DADOS BÁSICOS =====
    current_price = result.get('analysis', {}).get('current_price', 0)
    predicted_price = result.get('analysis', {}).get('predicted_price', 0)
    price_change_pct = result.get('analysis', {}).get('price_change_percent', 0)
    recommendation = result.get('analysis', {}).get('recommendation', 'N/A')
    recommendation_score = result.get('analysis', {}).get('recommendation_score', 0)
    
    print(f"\n📈 DADOS PRINCIPAIS:")
    print(f"   💰 Preço atual: R$ {current_price:.2f}")
    print(f"   🔮 Previsão final: R$ {predicted_price:.2f}")
    print(f"   📊 Mudança prevista: {price_change_pct:.1f}%")
    print(f"   🎯 Recomendação: {recommendation}")
    print(f"   📊 Score: {recommendation_score:.2f}")
    
    # ===== 2. ANÁLISE DAS PREVISÕES =====
    predictions = result.get('prediction_data', [])
    print(f"\n🔮 ANÁLISE DE PREVISÕES ({len(predictions)} dias):")
    
    if predictions:
        first_pred = predictions[0]['predicted_price']
        mid_pred = predictions[len(predictions)//2]['predicted_price']
        last_pred = predictions[-1]['predicted_price']
        
        short_trend = ((first_pred - current_price) / current_price) * 100
        medium_trend = ((mid_pred - current_price) / current_price) * 100
        long_trend = ((last_pred - current_price) / current_price) * 100
        
        print(f"   📅 1º dia: R$ {first_pred:.2f} ({short_trend:+.1f}%)")
        print(f"   📅 Meio: R$ {mid_pred:.2f} ({medium_trend:+.1f}%)")
        print(f"   📅 Último: R$ {last_pred:.2f} ({long_trend:+.1f}%)")
        
        # Mostrar várias previsões para entender a tendência
        print(f"\n   📊 Progressão das previsões:")
        for i, pred in enumerate(predictions[:10]):  # Primeiros 10 dias
            pred_price = pred['predicted_price']
            change = ((pred_price - current_price) / current_price) * 100
            print(f"      Dia {i+1}: R$ {pred_price:.2f} ({change:+.1f}%)")
        
        # Detectar se há tendência de queda
        declining_count = 0
        for i in range(len(predictions)):
            pred_price = predictions[i]['predicted_price']
            change = ((pred_price - current_price) / current_price) * 100
            if change < -2:  # Queda > 2%
                declining_count += 1
        
        decline_ratio = declining_count / len(predictions)
        print(f"\n   📉 Dias com queda > 2%: {declining_count}/{len(predictions)} ({decline_ratio:.1%})")
        
        if long_trend <= -10:
            print(f"   ❌ TENDÊNCIA DE QUEDA SIGNIFICATIVA: {long_trend:.1f}%")
        elif decline_ratio > 0.6:
            print(f"   ⚠️ MAIORIA DAS PREVISÕES NEGATIVAS: {decline_ratio:.1%}")
    
    # ===== 3. INDICADORES TÉCNICOS =====
    indicators = result.get('indicators', {})
    print(f"\n📈 INDICADORES TÉCNICOS:")
    print(f"   RSI: {indicators.get('RSI', 0):.1f}")
    print(f"   MACD: {indicators.get('MACD', 0):.3f}")
    print(f"   MA20: R$ {indicators.get('ma20', 0):.2f}")
    
    # ===== 4. ANÁLISE DA INCONSISTÊNCIA =====
    print(f"\n🚨 ANÁLISE DE INCONSISTÊNCIA:")
    
    # Verificar se há inconsistência clara
    is_inconsistent = False
    
    if long_trend <= -10 and recommendation == "COMPRAR":
        print(f"   ❌ INCONSISTÊNCIA CRÍTICA: Queda de {long_trend:.1f}% prevista → mas recomenda COMPRA!")
        is_inconsistent = True
    elif medium_trend <= -10 and recommendation == "COMPRAR":
        print(f"   ❌ INCONSISTÊNCIA MÉDIA: Queda média de {medium_trend:.1f}% → mas recomenda COMPRA!")
        is_inconsistent = True
    elif decline_ratio > 0.7 and recommendation == "COMPRAR":
        print(f"   ⚠️ INCONSISTÊNCIA POSSÍVEL: {decline_ratio:.1%} das previsões negativas → mas recomenda COMPRA!")
        is_inconsistent = True
    
    if is_inconsistent:
        print(f"\n🔧 POSSÍVEIS CAUSAS:")
        print(f"   1. Score de recomendação dominado por indicadores técnicos")
        print(f"   2. Lógica de tendência não aplicada corretamente") 
        print(f"   3. Thresholds de decisão inadequados")
        print(f"   4. Cache corrompido ou dados inconsistentes")
        
        # Analisar technical_analysis
        tech = result.get('technical_analysis', {})
        print(f"\n   📊 Análise técnica:")
        print(f"      RSI signal: {tech.get('rsi_signal', 'N/A')}")
        print(f"      MACD signal: {tech.get('macd_signal', 'N/A')}")
        print(f"      Trend: {tech.get('trend', 'N/A')}")
        
        return False
    else:
        print(f"   ✅ Recomendação parece consistente com previsões")
        return True

if __name__ == "__main__":
    debug_recommendation_logic()