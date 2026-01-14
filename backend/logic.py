"""
Módulo principal de análise financeira integrada.
Combina análise técnica, machine learning e análise de sentimento.
"""
import yfinance as yf
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

# Importar módulos locais
from technical_indicators import compute_technical_indicators, compute_additional_indicators
from ml_models import FinancialMLModels, train_advanced_model
from sentiment_analysis import enhanced_sentiment_analysis
from config import settings

logger = logging.getLogger(__name__)


class FinancialAnalyzer:
    """Classe principal para análise financeira."""
    
    def __init__(self):
        self.ml_models = FinancialMLModels()
        self.cache = {}  # Cache simples para resultados
    
    def _get_stock_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Obtém dados históricos do Yahoo Finance com validações.
        
        Args:
            ticker: Símbolo da ação
            period: Período dos dados
            
        Returns:
            DataFrame com dados OHLCV
        """
        try:
            data = yf.download(ticker, period=period, interval="1d", auto_adjust=False)
            
            if data.empty:
                raise ValueError(f"Nenhum dado encontrado para o ticker {ticker}")
            
            # Validar volume mínimo
            avg_volume = data['Volume'].mean()
            if avg_volume < settings.MIN_VOLUME_THRESHOLD:
                logger.warning(f"Volume médio baixo para {ticker}: {avg_volume}")
            
            return data
            
        except Exception as e:
            logger.error(f"Erro ao obter dados para {ticker}: {e}")
            raise
    
    def _calculate_trend(self, ma7: float, ma15: float, ma30: float) -> str:
        """Calcula tendência baseada em médias móveis."""
        if ma7 > ma15 > ma30:
            return "alta"
        elif ma7 < ma15 < ma30:
            return "baixa"
        else:
            return "neutra"
    
    def _get_trading_strategy(self, last_price: float, forecast: float, 
                             rsi: float, trend: str, sentiment: str) -> str:
        """
        Gera estratégia de trading baseada em múltiplos indicadores.
        
        Args:
            last_price: Preço atual
            forecast: Previsão do modelo
            rsi: Valor do RSI
            trend: Tendência das médias móveis
            sentiment: Sentimento de mercado
            
        Returns:
            Recomendação de estratégia
        """
        strategy = "Observar"
        
        # Análise de previsão
        price_diff = (forecast - last_price) / last_price
        
        if price_diff > 0.02:  # 2% acima
            strategy = "Possível Compra (Previsão Otimista)"
        elif price_diff < -0.02:  # 2% abaixo
            strategy = "Possível Venda (Previsão Pessimista)"
        
        # RSI Override
        if rsi < 30:
            strategy = "Sinal de Compra (Sobrevendido)"
        elif rsi > 70:
            strategy = "Sinal de Venda (Sobrecomprado)"
        
        # Tendência
        if trend == "alta" and "Compra" not in strategy:
            strategy = "Sinal de Compra (Tendência de Alta)"
        elif trend == "baixa" and "Venda" not in strategy:
            strategy = "Sinal de Venda (Tendência de Baixa)"
        
        # Confirmação por sentiment
        sentiment_lower = sentiment.lower() if isinstance(sentiment, str) else "neutro"
        
        if "fortemente positivo" in sentiment_lower and "compra" in strategy.lower():
            strategy += " - Confirmado por Sentimento"
        elif "fortemente negativo" in sentiment_lower and "venda" in strategy.lower():
            strategy += " - Confirmado por Sentimento"
        
        return strategy
    
    def _generate_smart_alerts(self, analysis: Dict[str, Any]) -> List[str]:
        """Gera alertas inteligentes baseados na análise."""
        alerts = []
        
        rsi = analysis.get('RSI', 50)
        volume = analysis.get('Volume', 0)
        avg_volume = analysis.get('avg_volume', volume)
        trend = analysis.get('tendencia', 'neutra')
        sentiment = analysis.get('sentimento', 'neutro')
        
        # Alertas de RSI
        if rsi > 70:
            alerts.append("⚠️ RSI acima de 70 (sobrecomprado)")
        elif rsi < 30:
            alerts.append("📈 RSI abaixo de 30 (sobrevendido)")
        
        # Alertas de volume
        if avg_volume > 0 and volume > 2 * avg_volume:
            alerts.append(f"📊 Volume alto: {volume / avg_volume:.1f}x a média")
        
        # Alertas de divergência
        if "positivo" in str(sentiment).lower() and trend == "baixa":
            alerts.append("🔄 Divergência: Sentimento positivo vs tendência baixa")
        elif "negativo" in str(sentiment).lower() and trend == "alta":
            alerts.append("🔄 Divergência: Sentimento negativo vs tendência alta")
        
        return alerts
    
    def _calculate_risk_level(self, volatility: float, rsi: float, 
                             volume_ratio: float) -> str:
        """Calcula nível de risco do investimento."""
        risk_score = 0
        
        # Volatilidade
        if volatility > 0.05:  # 5%
            risk_score += 2
        elif volatility > 0.03:  # 3%
            risk_score += 1
        
        # RSI extremos
        if rsi > 80 or rsi < 20:
            risk_score += 2
        elif rsi > 70 or rsi < 30:
            risk_score += 1
        
        # Volume baixo
        if volume_ratio < 0.5:
            risk_score += 1
        
        if risk_score >= 4:
            return "ALTO"
        elif risk_score >= 2:
            return "MÉDIO"
        else:
            return "BAIXO"
    
    def analyze_single_stock(self, ticker: str) -> Dict[str, Any]:
        """
        Análise completa de uma ação individual.
        
        Args:
            ticker: Símbolo da ação
            
        Returns:
            Dicionário com análise completa
        """
        try:
            # Verificar cache
            cache_key = f"{ticker}_{datetime.now().strftime('%Y-%m-%d_%H')}"
            if cache_key in self.cache:
                logger.info(f"Retornando análise em cache para {ticker}")
                return self.cache[cache_key]
            
            # Obter dados
            data = self._get_stock_data(ticker)
            
            # Calcular indicadores técnicos
            data = compute_technical_indicators(data)
            data = compute_additional_indicators(data)
            
            # Análise de sentimento
            try:
                sentiment = enhanced_sentiment_analysis(ticker)
            except Exception as e:
                logger.warning(f"Erro na análise de sentimento para {ticker}: {e}")
                sentiment = "neutro"
            
            # Preparar dados para ML
            features = data.drop(['Close', 'retorno'], axis=1, errors='ignore')
            target = data['Close']
            
            if len(features) < 30:
                return {"erro": "Dados insuficientes para análise completa"}
            
            # Treinamento do modelo
            ml_results = self.ml_models.train_models(features, target)
            model = ml_results['best_model']
            
            if model is None:
                return {"erro": "Falha no treinamento dos modelos de ML"}
            
            # Previsão
            last_features = features.iloc[-1:].values
            forecast = float(model.predict(last_features)[0])
            
            # Métricas atuais
            last_data = data.iloc[-1]
            last_price = float(last_data['Close'])
            avg_volume = float(data['Volume'].mean())
            
            # Cálculos
            trend = self._calculate_trend(
                float(last_data['MA7']),
                float(last_data['MA15']),
                float(last_data['MA30'])
            )
            
            strategy = self._get_trading_strategy(
                last_price, forecast, float(last_data['RSI']), trend, sentiment
            )
            
            volatility = float(last_data.get('volatilidade', 0))
            volume_ratio = float(last_data['Volume']) / avg_volume if avg_volume > 0 else 1
            
            risk_level = self._calculate_risk_level(
                volatility, float(last_data['RSI']), volume_ratio
            )
            
            # Resultado final
            result = {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "preco_atual": last_price,
                "previsao": forecast,
                "variacao_prevista": ((forecast - last_price) / last_price) * 100,
                "estrategia": strategy,
                "tendencia": trend,
                "nivel_risco": risk_level,
                "indicadores": {
                    "RSI": float(last_data['RSI']),
                    "MACD": float(last_data.get('MACD', 0)),
                    "MA7": float(last_data['MA7']),
                    "MA15": float(last_data['MA15']),
                    "MA30": float(last_data['MA30']),
                    "volatilidade": volatility,
                    "volume": float(last_data['Volume']),
                    "volume_medio": avg_volume\n                },\n                \"sentimento\": sentiment,\n                \"alertas\": self._generate_smart_alerts({\n                    'RSI': float(last_data['RSI']),\n                    'Volume': float(last_data['Volume']),\n                    'avg_volume': avg_volume,\n                    'tendencia': trend,\n                    'sentimento': sentiment\n                }),\n                \"modelo\": {\n                    \"tipo\": \"Ensemble\",\n                    \"melhor_score\": ml_results['best_score'],\n                    \"feature_importance\": ml_results.get('feature_importance', {})\n                }\n            }\n            \n            # Cache do resultado\n            self.cache[cache_key] = result\n            \n            return result\n            \n        except Exception as e:\n            logger.error(f\"Erro na análise de {ticker}: {e}\")\n            return {\n                \"erro\": f\"Falha na análise: {str(e)}\",\n                \"ticker\": ticker,\n                \"timestamp\": datetime.now().isoformat()\n            }\n\n\n# Instância global do analisador\nanalyzer = FinancialAnalyzer()\n\n\n# Funções de compatibilidade com a API existente\ndef analyze(ticker: str) -> Dict[str, Any]:\n    \"\"\"Função de compatibilidade para análise individual.\"\"\"\n    return analyzer.analyze_single_stock(ticker)\n\n\ndef analyze_all() -> Dict[str, Any]:\n    \"\"\"Análise de múltiplas ações populares.\"\"\"\n    tickers = [\n        \"AAPL\", \"GOOGL\", \"MSFT\", \"AMZN\", \"TSLA\",  # US\n        \"PETR4.SA\", \"VALE3.SA\", \"ITUB4.SA\", \"BBDC4.SA\"  # BR\n    ]\n    \n    results = {}\n    errors = []\n    \n    for ticker in tickers:\n        try:\n            result = analyzer.analyze_single_stock(ticker)\n            if \"erro\" not in result:\n                results[ticker] = result\n            else:\n                errors.append(f\"{ticker}: {result['erro']}\")\n        except Exception as e:\n            errors.append(f\"{ticker}: {str(e)}\")\n    \n    return {\n        \"timestamp\": datetime.now().isoformat(),\n        \"resultados\": results,\n        \"total_analisados\": len(results),\n        \"erros\": errors\n    }\n\n\ndef price_ticker(ticker: str) -> Dict[str, Any]:\n    \"\"\"Obtém apenas o preço atual de um ticker.\"\"\"\n    try:\n        stock = yf.Ticker(ticker)\n        hist = stock.history(period=\"1d\")\n        \n        if hist.empty:\n            return {\"erro\": \"Ticker inválido ou sem dados\"}\n        \n        current_price = float(hist['Close'].iloc[-1])\n        \n        return {\n            \"ticker\": ticker,\n            \"preco\": current_price,\n            \"timestamp\": datetime.now().isoformat()\n        }\n        \n    except Exception as e:\n        logger.error(f\"Erro ao obter preço de {ticker}: {e}\")\n        return {\"erro\": str(e)}\n\n\n# Função auxiliar para análise setorial (implementação básica)\ndef sector_correlation_analysis(ticker: str) -> Dict[str, Any]:\n    \"\"\"Análise setorial básica (placeholder para implementação futura).\"\"\"\n    return {\n        \"sector\": \"Desconhecido\",\n        \"correlacao_sp500\": 0.5,\n        \"beta\": 1.0\n    }