#!/usr/bin/env python3
"""
🧪 Teste: Verificar melhorias nas explicações da análise
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic_enhanced import EnhancedFinancialAnalyzer

def test_improved_explanations():
    """Testa as explicações melhoradas"""
    print("🧪 Testando explicações melhoradas da análise...")
    print("=" * 60)
    
    analyzer = EnhancedFinancialAnalyzer()
    
    # Testar com VALE3.SA
    print("📊 Analisando VALE3.SA...")
    result = analyzer.generate_enhanced_chart_data("VALE3.SA", 15)
    
    print(f"🎯 Recomendação: {result['analysis']['recommendation']}")
    print(f"💰 Preço atual: R${result['analysis']['current_price']:.2f}")
    print(f"🔮 Previsão: R${result['analysis']['predicted_price']:.2f}")
    print(f"📈 Variação: {result['analysis']['price_change_percent']:+.1f}%")
    print(f"📊 Confiança: {result['confidence_analysis']['confidence_percentage']}%")
    
    print(f"\n📝 RESUMO DA ANÁLISE:")
    print("=" * 50)
    print(result['analysis_summary'])
    
    print(f"\n🔍 ANÁLISE TÉCNICA:")
    print("=" * 50)
    tech = result['technical_analysis']
    for key, value in tech.items():
        if 'explanation' in key:
            print(f"• {key.replace('_explanation', '').upper()}: {value}")

if __name__ == "__main__":
    test_improved_explanations()