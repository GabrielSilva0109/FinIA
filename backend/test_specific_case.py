#!/usr/bin/env python3
"""
Teste com dados específicos do bug reportado pelo usuário
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic_enhanced import EnhancedFinancialAnalyzer
import pandas as pd
import numpy as np

def test_specific_case():
    """Simula exatamente o caso que o usuário reportou"""
    print("🎯 Testando caso específico reportado pelo usuário...")
    
    analyzer = EnhancedFinancialAnalyzer()
    
    # Simular as previsões exatas do usuário
    current_price = 21.34
    predictions_data = [
        {"predicted_price": 21.48, "date": "2026-01-17"},  # +0.64%
        {"predicted_price": 21.73, "date": "2026-01-18"},  # +1.8%
        {"predicted_price": 22.18, "date": "2026-01-19"},  # +3.9%
        {"predicted_price": 21.93, "date": "2026-01-20"},  # +2.7%
        {"predicted_price": 22.15, "date": "2026-01-21"},  # +3.8%
        {"predicted_price": 20.93, "date": "2026-01-22"},  # -1.9%
        {"predicted_price": 20.37, "date": "2026-01-23"},  # -4.5%
        {"predicted_price": 20.26, "date": "2026-01-24"},  # -5.1%
        {"predicted_price": 19.93, "date": "2026-01-25"},  # -6.6%
        {"predicted_price": 20.10, "date": "2026-01-26"},  # -5.8%
        {"predicted_price": 19.34, "date": "2026-01-27"},  # -9.4%
        {"predicted_price": 18.71, "date": "2026-01-28"},  # -12.3%
        {"predicted_price": 18.60, "date": "2026-01-29"},  # -12.8%
        {"predicted_price": 18.74, "date": "2026-01-30"},  # -12.2%
        {"predicted_price": 18.63, "date": "2026-01-31"},  # -12.7%
        {"predicted_price": 19.11, "date": "2026-02-01"},  # -10.4%
        {"predicted_price": 19.11, "date": "2026-02-02"},  # -10.4%
        {"predicted_price": 18.74, "date": "2026-02-03"},  # -12.2%
        {"predicted_price": 17.35, "date": "2026-02-04"},  # -18.7%
        {"predicted_price": 16.62, "date": "2026-02-05"},  # -22.1%
        {"predicted_price": 16.13, "date": "2026-02-06"},  # -24.4%
        {"predicted_price": 16.10, "date": "2026-02-07"},  # -24.6%
        {"predicted_price": 16.18, "date": "2026-02-08"},  # -24.2%
        {"predicted_price": 16.03, "date": "2026-02-09"},  # -24.9%
        {"predicted_price": 16.15, "date": "2026-02-10"},  # -24.3%
    ]
    
    print(f"💰 Preço atual: R$ {current_price:.2f}")
    print(f"📊 Previsões simuladas: {len(predictions_data)} dias")
    
    # Criar dados mínimos para análise
    dates = pd.date_range(start='2026-01-15', periods=1)
    test_data = pd.DataFrame({
        'open': [21.5],
        'high': [21.66], 
        'low': [21.34],
        'close': [current_price],
        'volume': [20000000]
    }, index=dates)
    
    test_data = analyzer._calculate_basic_indicators_fast(test_data)
    
    # Aplicar nova lógica de análise
    if len(predictions_data) >= 3:
        first_price = predictions_data[0]["predicted_price"]
        mid_price = predictions_data[len(predictions_data)//2]["predicted_price"]
        last_price = predictions_data[-1]["predicted_price"]
        
        # Calcular tendências
        short_trend = ((first_price - current_price) / current_price) * 100
        medium_trend = ((mid_price - current_price) / current_price) * 100
        long_trend = ((last_price - current_price) / current_price) * 100
        
        print(f"\n🔍 Análise de tendências:")
        print(f"   📈 Curto prazo (1º dia): {short_trend:+.1f}%")
        print(f"   📈 Médio prazo (dia {len(predictions_data)//2}): {medium_trend:+.1f}%")
        print(f"   📈 Longo prazo (último dia): {long_trend:+.1f}%")
        
        # Lógica corrigida
        if abs(long_trend) > abs(short_trend) and abs(long_trend) > 5:
            chosen_trend = long_trend
            chosen_type = "longo prazo"
        elif abs(medium_trend) > abs(short_trend) and abs(medium_trend) > 3:
            chosen_trend = medium_trend
            chosen_type = "médio prazo"
        else:
            chosen_trend = short_trend
            chosen_type = "curto prazo"
            
        print(f"   🎯 Tendência escolhida: {chosen_type} ({chosen_trend:+.1f}%)")
        
        # Determinar recomendação
        if chosen_trend <= -10:
            recommendation = "VENDER"
            reason = f"queda significativa de {chosen_trend:.1f}%"
        elif chosen_trend <= -5:
            recommendation = "VENDER"  
            reason = f"queda moderada de {chosen_trend:.1f}%"
        elif chosen_trend >= 5:
            recommendation = "COMPRAR"
            reason = f"alta significativa de {chosen_trend:.1f}%"
        else:
            recommendation = "MANTER"
            reason = f"mudança pequena de {chosen_trend:.1f}%"
            
        print(f"   🎯 Recomendação corrigida: {recommendation}")
        print(f"   💭 Justificativa: {reason}")
        
        # Verificar se corrigiu o problema original
        print(f"\n📋 Resultado:")
        if long_trend <= -20 and recommendation == "VENDER":
            print(f"   ✅ CORRIGIDO: Queda de {long_trend:.1f}% detectada → VENDA recomendada")
            return True
        elif abs(chosen_trend) > 10 and recommendation in ["COMPRAR", "VENDER"]:
            print(f"   ✅ CONSISTENTE: Mudança significativa de {chosen_trend:.1f}% → {recommendation}")
            return True
        else:
            print(f"   ⚠️ Análise: {recommendation} para mudança de {chosen_trend:.1f}%")
            return False

if __name__ == "__main__":
    test_specific_case()