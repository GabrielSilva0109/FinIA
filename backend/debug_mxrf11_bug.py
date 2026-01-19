#!/usr/bin/env python3
"""
🐛 Debug: Investigar bug de previsão absurda no MXRF11.SA
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic_enhanced import EnhancedFinancialAnalyzer
import yfinance as yf
import pandas as pd
import numpy as np

def debug_mxrf11_predictions():
    """Debug específico para MXRF11.SA"""
    print("🐛 Debug: Investigando bug de previsão absurda MXRF11.SA...")
    print("=" * 70)
    
    # Obter dados reais
    ticker = "MXRF11.SA"
    stock = yf.Ticker(ticker)
    data = stock.history(period="6mo")
    
    if data.empty:
        print("❌ Erro: Sem dados históricos")
        return
    
    current_price = data['Close'].iloc[-1]
    max_price_6m = data['Close'].max()
    min_price_6m = data['Close'].min()
    
    print(f"📊 ANÁLISE HISTÓRICA REAL:")
    print(f"   💰 Preço atual: R${current_price:.2f}")
    print(f"   📈 Máximo 6 meses: R${max_price_6m:.2f}")
    print(f"   📉 Mínimo 6 meses: R${min_price_6m:.2f}")
    print(f"   📊 Variação máxima: {((max_price_6m - min_price_6m) / min_price_6m * 100):+.1f}%")
    
    # Testar o analyzer
    print(f"\n🧪 TESTE DO ANALYZER:")
    analyzer = EnhancedFinancialAnalyzer()
    
    # Obter dados do analyzer
    analyzer_data = analyzer.get_stock_data(ticker)
    analyzer_current = analyzer_data['close'].iloc[-1]
    
    print(f"   💰 Preço analyzer: R${analyzer_current:.2f}")
    print(f"   📊 Diferença: {abs(current_price - analyzer_current):.6f}")
    
    # Testar previsões diretamente
    print(f"\n🔮 TESTE DE PREVISÕES DIRETAS:")
    predictions = analyzer._fallback_predictions(analyzer_data, 5)  # Só 5 dias para debug
    
    for i, pred in enumerate(predictions[:5]):
        pred_price = pred['predicted_price']
        ratio = pred_price / analyzer_current
        print(f"   Dia {i+1}: R${pred_price:.2f} (ratio: {ratio:.3f}) - {pred.get('method', 'unknown')}")
        
        if ratio > 1.5 or ratio < 0.5:
            print(f"      🚨 RATIO ANORMAL DETECTADO: {ratio:.3f}")
    
    # Testar análise completa
    print(f"\n📊 TESTE DE ANÁLISE COMPLETA:")
    result = analyzer.generate_enhanced_chart_data(ticker, 5)
    
    analysis = result['analysis']
    recommendations = result['recommendations']
    
    print(f"   💰 Preço atual: R${analysis['current_price']:.2f}")
    print(f"   🔮 Previsão: R${analysis['predicted_price']:.2f}")
    print(f"   📈 Variação: {analysis['price_change_percent']:+.1f}%")
    print(f"   🎯 Target: R${recommendations['target_price']:.2f}")
    
    target_ratio = recommendations['target_price'] / analysis['current_price']
    print(f"   📊 Target Ratio: {target_ratio:.3f}")
    
    if target_ratio > 2.0:
        print(f"      🚨 TARGET PRICE ABSURDO: {target_ratio:.3f}x o preço atual!")
        print(f"      📊 Isso seria um ganho de {((target_ratio - 1) * 100):.1f}%!")
    
    # Examinar prediction_data
    print(f"\n🔍 PRIMEIRA PREVISÃO DETALHADA:")
    if result['prediction_data']:
        first_pred = result['prediction_data'][0]
        first_ratio = first_pred['predicted_price'] / analysis['current_price']
        print(f"   💰 Preço previsto: R${first_pred['predicted_price']:.2f}")
        print(f"   📊 Ratio: {first_ratio:.3f}")
        print(f"   🔧 Método: {first_pred.get('method', 'unknown')}")
        print(f"   ✅ Validation ratio: {first_pred.get('validation_ratio', 'N/A')}")
        
        if first_ratio > 2.0:
            print(f"      🚨 PRIMEIRA PREVISÃO JÁ ESTÁ ABSURDA!")

def debug_prediction_math():
    """Debug do cálculo matemático das previsões"""
    print("\n🧮 DEBUG MATEMÁTICO DAS PREVISÕES:")
    print("=" * 50)
    
    # Simular dados do MXRF11
    current_price = 9.45
    print(f"💰 Preço base: R${current_price:.2f}")
    
    # Simular volatilidade (baseado no MXRF11)
    volatility = 0.02485  # ~2.5% como no resultado real
    print(f"📊 Volatilidade: {volatility:.4f} ({volatility*100:.1f}%)")
    
    # Testar componentes de oscilação
    cycle_amplitude = volatility * current_price * 1.5
    print(f"🌊 Amplitude ciclo: R${cycle_amplitude:.4f}")
    
    # Testar componente aleatório
    random_component = volatility * current_price * 0.7
    print(f"🎲 Componente aleatório: ±R${random_component:.4f}")
    
    # Simular 1 dia de previsão
    i = 1
    trend_factor = 0.001 * i * 0.3  # Muito pequeno
    cycle_factor = np.sin(i * np.pi / 6) * cycle_amplitude * 0.6
    random_factor = random_component  # Máximo positivo
    rsi_correction = volatility * current_price * 0.5  # RSI < 30
    
    predicted_price = current_price + trend_factor + cycle_factor + random_factor + rsi_correction
    
    print(f"\n🔧 COMPONENTES DO CÁLCULO:")
    print(f"   Trend factor: +R${trend_factor:.6f}")
    print(f"   Cycle factor: +R${cycle_factor:.6f}")
    print(f"   Random factor: +R${random_factor:.6f}")
    print(f"   RSI correction: +R${rsi_correction:.6f}")
    print(f"   TOTAL SOMA: +R${trend_factor + cycle_factor + random_factor + rsi_correction:.6f}")
    print(f"   PREVISÃO: R${predicted_price:.2f}")
    print(f"   RATIO: {predicted_price/current_price:.3f}")
    
    if predicted_price > current_price * 1.2:
        print(f"🚨 PROBLEMA: Previsão 1 dia já é {((predicted_price/current_price-1)*100):.1f}% maior!")

if __name__ == "__main__":
    debug_mxrf11_predictions()
    debug_prediction_math()