#!/usr/bin/env python3
"""
🔧 Patch: Melhorar resumo da análise temporariamente
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic_enhanced import EnhancedFinancialAnalyzer

# Função melhorada para resumo
def enhanced_analysis_summary(self, analysis, confidence_data, technical_analysis, recommendations):
    """Gera resumo ULTRA-DETALHADO da análise com explicações específicas dos motivos"""
    try:
        # Extrair dados principais
        recommendation = analysis.get('recommendation', 'MANTER')
        confidence = confidence_data.get('confidence_percentage', 50)
        trend = analysis.get('trend', 'neutro')
        price_change_pct = analysis.get('price_change_percent', 0)
        current_price = analysis.get('current_price', 0)
        predicted_price = analysis.get('predicted_price', 0)
        
        # Dados técnicos detalhados
        rsi_signal = technical_analysis.get('rsi_signal', 'NEUTRO')
        rsi_value = technical_analysis.get('rsi_value', 50)
        rsi_explanation = technical_analysis.get('rsi_explanation', '')
        macd_signal = technical_analysis.get('macd_signal', 'NEUTRO')
        macd_explanation = technical_analysis.get('macd_explanation', '')
        trend_strength = technical_analysis.get('trend_strength', 'LATERAL')
        ma_explanation = technical_analysis.get('ma_explanation', '')
        bollinger_explanation = technical_analysis.get('bollinger_explanation', '')
        volume_analysis = technical_analysis.get('volume_analysis', '')
        support_level = technical_analysis.get('support_level', 0)
        resistance_level = technical_analysis.get('resistance_level', 0)
        
        # Construir explicação ULTRA-DETALHADA
        summary_parts = []
        
        # 1. CABEÇALHO COM RECOMENDAÇÃO E JUSTIFICATIVA PRINCIPAL
        if recommendation == 'COMPRAR':
            summary_parts.append(f"🟢 RECOMENDAÇÃO: COMPRAR com {confidence}% de confiança")
            summary_parts.append(f"\n\n📈 MOTIVO PRINCIPAL: O ativo apresenta perspectiva de alta de {price_change_pct:.1f}% (de R${current_price:.2f} para R${predicted_price:.2f}).")
            
            # Explicar POR QUE recomenda comprar
            buy_reasons = []
            if trend_strength in ['ALTA_FORTE', 'ALTA_MODERADA']:
                buy_reasons.append(f"Tendência de alta confirmada")
            if macd_signal in ['COMPRA', 'COMPRA_FORTE']:
                buy_reasons.append(f"MACD em sinal positivo")
            if rsi_signal == 'SOBREVENDIDO':
                buy_reasons.append(f"RSI indica sobrevenda (oportunidade)")
            if price_change_pct > 5:
                buy_reasons.append(f"Forte potencial de valorização")
                
            if buy_reasons:
                summary_parts.append(f"\n🎯 INDICADORES FAVORÁVEIS: {', '.join(buy_reasons)}.")
                
        elif recommendation == 'VENDER':
            summary_parts.append(f"🔴 RECOMENDAÇÃO: VENDER com {confidence}% de confiança")
            summary_parts.append(f"\n\n📉 MOTIVO PRINCIPAL: O ativo apresenta perspectiva de queda de {price_change_pct:.1f}% (de R${current_price:.2f} para R${predicted_price:.2f}).")
            
            # Explicar POR QUE recomenda vender
            sell_reasons = []
            if trend_strength in ['BAIXA_FORTE', 'BAIXA_MODERADA']:
                sell_reasons.append(f"Tendência de baixa confirmada")
            if macd_signal in ['VENDA', 'VENDA_FORTE']:
                sell_reasons.append(f"MACD em sinal negativo")
            if rsi_signal == 'SOBRECOMPRADO':
                sell_reasons.append(f"RSI indica sobrecompra")
            if price_change_pct < -3:
                sell_reasons.append(f"Alto risco de desvalorização")
                
            if sell_reasons:
                summary_parts.append(f"\n🎯 INDICADORES DE RISCO: {', '.join(sell_reasons)}.")
                
        else:  # MANTER
            summary_parts.append(f"🟡 RECOMENDAÇÃO: MANTER com {confidence}% de confiança")
            if abs(price_change_pct) < 3:
                summary_parts.append(f"\n\n⚖️ MOTIVO PRINCIPAL: Expectativa de movimento lateral com variação pequena de {price_change_pct:+.1f}%.")
            else:
                summary_parts.append(f"\n\n⚖️ MOTIVO PRINCIPAL: Sinais técnicos conflitantes não justificam compra ou venda no momento.")
            
            # Explicar por que manter
            hold_reasons = []
            if trend_strength == 'LATERAL':
                hold_reasons.append("Movimento lateral predominante")
            if rsi_signal == 'NEUTRO':
                hold_reasons.append(f"RSI equilibrado ({rsi_value:.0f})")
            if macd_signal == 'NEUTRO':
                hold_reasons.append("MACD sem direção clara")
                
            if hold_reasons:
                summary_parts.append(f"\n🎯 MOTIVO: {', '.join(hold_reasons)}.")
        
        # 2. ANÁLISE TÉCNICA DETALHADA
        summary_parts.append(f"\n\n📊 ANÁLISE TÉCNICA DETALHADA:")
        
        # RSI explicação
        if rsi_explanation:
            summary_parts.append(f"\n• RSI: {rsi_explanation}")
        
        # MACD explicação
        if macd_explanation:
            summary_parts.append(f"\n• MACD: {macd_explanation}")
        
        # Médias móveis
        if ma_explanation:
            summary_parts.append(f"\n• Médias Móveis: {ma_explanation}")
            
        # Bollinger Bands
        if bollinger_explanation:
            summary_parts.append(f"\n• Bollinger Bands: {bollinger_explanation}")
        
        # Volume
        if volume_analysis and volume_analysis != "Volume não disponível":
            summary_parts.append(f"\n• Volume: {volume_analysis}")
        
        # Suporte e Resistência
        if support_level > 0 and resistance_level > 0:
            summary_parts.append(f"\n• Suporte/Resistência: Suporte em R${support_level:.2f}, Resistência em R${resistance_level:.2f}")
            if current_price <= support_level * 1.02:
                summary_parts.append(" (próximo ao suporte - possível reversão)")
            elif current_price >= resistance_level * 0.98:
                summary_parts.append(" (próximo à resistência - possível correção)")
        
        # 3. NÍVEL DE CONFIANÇA E EXPLICAÇÃO
        summary_parts.append(f"\n\n🏆 CONFIANÇA DA ANÁLISE ({confidence}%):")
        if confidence >= 80:
            summary_parts.append(f"\n• ALTA CONFIANÇA: Múltiplos indicadores convergem na mesma direção.")
        elif confidence >= 60:
            summary_parts.append(f"\n• CONFIANÇA MODERADA: Maioria dos indicadores convergem, mas há alguns conflitantes.")
        elif confidence >= 40:
            summary_parts.append(f"\n• CONFIANÇA BAIXA: Sinais técnicos mistos - monitorar evolução.")
        else:
            summary_parts.append(f"\n• CONFIANÇA MUITO BAIXA: Alta volatilidade - aguardar definição.")
        
        # 4. RECOMENDAÇÃO PRÁTICA 
        target_price = recommendations.get('target_price', 0)
        stop_loss = recommendations.get('stop_loss', 0)
        
        summary_parts.append(f"\n\n⚡ AÇÃO PRÁTICA:")
        if recommendation == 'COMPRAR':
            summary_parts.append(f"\n• 🎯 META: R${target_price:.2f} (ganho potencial: {((target_price/current_price-1)*100):+.1f}%)")
            if stop_loss > 0:
                summary_parts.append(f"\n• 🛡️ STOP LOSS: R${stop_loss:.2f}")
            summary_parts.append(f"\n• ⏰ PRAZO: {recommendations.get('timeframe', '1-2 semanas')}")
        elif recommendation == 'VENDER':
            summary_parts.append(f"\n• 🎯 OBJETIVO: Proteger capital da queda prevista")
            summary_parts.append(f"\n• ⏰ URGÊNCIA: {recommendations.get('timeframe', 'Imediato')}")
        else:
            summary_parts.append(f"\n• 🎯 MONITORAR: Aguardar sinais mais claros")
            summary_parts.append(f"\n• ⏰ REVISÃO: {recommendations.get('timeframe', '1-2 semanas')}")
            if resistance_level > current_price:
                summary_parts.append(f"\n• 📈 COMPRA SE: Romper R${resistance_level:.2f}")
            if support_level < current_price:
                summary_parts.append(f"\n• 📉 VENDA SE: Perder R${support_level:.2f}")
        
        return "".join(summary_parts)
        
    except Exception as e:
        return f"📊 ANÁLISE PARA {analysis.get('recommendation', 'MANTER')}\n\nConfiança: {confidence_data.get('confidence_percentage', 50)}%\nVariação esperada: {analysis.get('price_change_percent', 0):+.1f}%\n\nConsulte os indicadores técnicos para análise detalhada."

# Aplicar o patch temporariamente
def apply_summary_patch():
    """Aplica o patch melhorado"""
    # Substituir o método original
    EnhancedFinancialAnalyzer._generate_analysis_summary = enhanced_analysis_summary
    print("✅ Patch aplicado - resumos de análise melhorados!")

def test_patched_explanations():
    """Testa as explicações com o patch aplicado"""
    print("🧪 Testando explicações ULTRA-MELHORADAS...")
    print("=" * 60)
    
    # Aplicar patch
    apply_summary_patch()
    
    analyzer = EnhancedFinancialAnalyzer()
    
    # Testar com VALE3.SA
    print("📊 Analisando VALE3.SA com explicações melhoradas...")
    result = analyzer.generate_enhanced_chart_data("VALE3.SA", 15)
    
    print(f"🎯 Recomendação: {result['analysis']['recommendation']}")
    print(f"💰 Preço atual: R${result['analysis']['current_price']:.2f}")
    print(f"🔮 Previsão: R${result['analysis']['predicted_price']:.2f}")
    print(f"📈 Variação: {result['analysis']['price_change_percent']:+.1f}%")
    print(f"📊 Confiança: {result['confidence_analysis']['confidence_percentage']}%")
    
    print(f"\n📝 RESUMO DA ANÁLISE MELHORADO:")
    print("=" * 50)
    print(result['analysis_summary'])

if __name__ == "__main__":
    test_patched_explanations()