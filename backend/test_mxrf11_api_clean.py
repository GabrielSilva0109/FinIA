#!/usr/bin/env python3
"""
🧪 Teste: API real para MXRF11.SA com cache limpo
"""

import requests
import json

def test_mxrf11_api():
    """Testa API real para MXRF11.SA"""
    print("🧪 Teste API real para MXRF11.SA...")
    print("=" * 50)
    
    try:
        url = "http://localhost:8000/analise/acao?ticker=MXRF11.SA"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Análise básica
            analysis = result['analysis']
            recommendations = result['recommendations']
            predictions = result['prediction_data']
            
            print(f"🎯 Recomendação: {analysis['recommendation']}")
            print(f"💰 Preço atual: R${analysis['current_price']:.2f}")
            print(f"🔮 Previsão: R${analysis['predicted_price']:.2f}")
            print(f"📈 Variação: {analysis['price_change_percent']:+.1f}%")
            print(f"🎯 Target: R${recommendations['target_price']:.2f}")
            
            # Verificar ratio
            target_ratio = recommendations['target_price'] / analysis['current_price']
            print(f"📊 Target Ratio: {target_ratio:.3f}")
            
            if target_ratio > 2.0:
                print(f"🚨 BUG DETECTADO! Target {target_ratio:.1f}x o preço atual!")
            else:
                print(f"✅ Target ratio normal")
            
            # Examinar primeiras previsões
            print(f"\n🔍 PRIMEIRAS 5 PREVISÕES:")
            for i, pred in enumerate(predictions[:5]):
                pred_price = pred['predicted_price']
                ratio = pred_price / analysis['current_price']
                print(f"   Dia {i+1}: R${pred_price:.2f} (ratio: {ratio:.3f})")
                
                if ratio > 1.5:
                    print(f"      🚨 RATIO ANORMAL: {ratio:.3f}")
            
            # Salvar para análise
            with open("mxrf11_test_result.json", "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n📁 Resultado salvo em mxrf11_test_result.json")
            
        else:
            print(f"❌ Erro HTTP: {response.status_code}")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    test_mxrf11_api()