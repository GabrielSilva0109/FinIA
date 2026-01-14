import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import requests
from typing import Dict, Any, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialAnalyzer:
    """Classe principal para análise financeira avançada."""
    
    def __init__(self):
        """Inicializa o analisador com componentes."""
        self.cache = {}
        self.news_cache = {}
        
    def _get_stock_data(self, ticker: str, period: str = "3mo") -> pd.DataFrame:
        """
        Obtém dados históricos da ação com indicadores técnicos.
        
        Args:
            ticker: Símbolo da ação
            period: Período dos dados ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
        
        Returns:
            DataFrame com dados históricos e indicadores técnicos
        """
        try:
            # Cache key baseado no ticker e período
            cache_key = f"{ticker}_{period}_{datetime.now().date()}"
            
            # Verificar cache
            if cache_key in self.cache:
                logger.info(f"Dados em cache encontrados para {ticker}")
                return self.cache[cache_key]
            
            # Buscar dados do yfinance
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                logger.warning(f"Nenhum dado encontrado para {ticker}")
                return pd.DataFrame()
            
            # Calcular indicadores técnicos
            logger.info(f"Calculando indicadores técnicos para {ticker}")
            
            # Médias móveis
            data['MA7'] = data['Close'].rolling(window=7).mean()
            data['MA15'] = data['Close'].rolling(window=15).mean()
            data['MA30'] = data['Close'].rolling(window=30).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = data['Close'].ewm(span=12).mean()
            ema26 = data['Close'].ewm(span=26).mean()
            data['MACD'] = ema12 - ema26
            data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_histogram'] = data['MACD'] - data['MACD_signal']
            
            # Bollinger Bands
            data['BB_middle'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
            data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
            data['BB_width'] = data['BB_upper'] - data['BB_lower']
            data['BB_position'] = (data['Close'] - data['BB_lower']) / data['BB_width']
            
            # Estocástico
            low14 = data['Low'].rolling(window=14).min()
            high14 = data['High'].rolling(window=14).max()
            data['%K'] = 100 * ((data['Close'] - low14) / (high14 - low14))
            data['%D'] = data['%K'].rolling(window=3).mean()
            
            # Volatilidade
            data['volatilidade'] = data['Close'].rolling(window=20).std()
            
            # Armazenar em cache
            self.cache[cache_key] = data
            
            logger.info(f"Dados processados para {ticker}: {len(data)} períodos")
            return data
            
        except Exception as e:
            logger.error(f"Erro ao obter dados para {ticker}: {e}")
            return pd.DataFrame()
    
    def _technical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Realiza análise técnica dos dados.
        
        Args:
            data: DataFrame com dados históricos
            
        Returns:
            Dicionário com análise técnica
        """
        try:
            if data.empty or len(data) < 30:
                return {"erro": "Dados insuficientes para análise técnica"}
            
            last_row = data.iloc[-1]
            
            # Indicadores atuais
            current_rsi = float(last_row['RSI'])
            current_macd = float(last_row['MACD'])
            current_macd_signal = float(last_row['MACD_signal'])
            current_bb_position = float(last_row['BB_position'])
            current_k = float(last_row['%K'])
            current_d = float(last_row['%D'])
            
            # Tendência baseada em médias móveis
            ma7 = float(last_row['MA7'])
            ma15 = float(last_row['MA15'])
            ma30 = float(last_row['MA30'])
            current_price = float(last_row['Close'])
            
            if ma7 > ma15 > ma30 and current_price > ma7:
                tendencia = "Forte Alta"
                score_tendencia = 2
            elif ma7 > ma15 and current_price > ma15:
                tendencia = "Alta"
                score_tendencia = 1
            elif ma7 < ma15 < ma30 and current_price < ma7:
                tendencia = "Forte Baixa"
                score_tendencia = -2
            elif ma7 < ma15 and current_price < ma15:
                tendencia = "Baixa"
                score_tendencia = -1
            else:
                tendencia = "Lateral"
                score_tendencia = 0
            
            # Score RSI
            if current_rsi > 80:
                score_rsi = -1  # Sobrecomprado
            elif current_rsi > 70:
                score_rsi = -0.5
            elif current_rsi < 20:
                score_rsi = 1  # Sobrevendido
            elif current_rsi < 30:
                score_rsi = 0.5
            else:
                score_rsi = 0
            
            # Score MACD
            if current_macd > current_macd_signal:
                score_macd = 1 if current_macd > 0 else 0.5
            else:
                score_macd = -1 if current_macd < 0 else -0.5
            
            # Score Bollinger Bands
            if current_bb_position > 0.8:
                score_bb = -0.5  # Próximo da banda superior
            elif current_bb_position < 0.2:
                score_bb = 0.5   # Próximo da banda inferior
            else:
                score_bb = 0
            
            # Score Estocástico
            if current_k > 80 and current_d > 80:
                score_stoch = -0.5  # Sobrecomprado
            elif current_k < 20 and current_d < 20:
                score_stoch = 0.5   # Sobrevendido
            else:
                score_stoch = 0
            
            # Score técnico combinado
            score_total = score_tendencia + score_rsi + score_macd + score_bb + score_stoch
            
            # Recomendação
            if score_total >= 1.5:
                recomendacao = "COMPRAR"
            elif score_total >= 0.5:
                recomendacao = "COMPRAR FRACO"
            elif score_total <= -1.5:
                recomendacao = "VENDER"
            elif score_total <= -0.5:
                recomendacao = "VENDER FRACO"
            else:
                recomendacao = "MANTER"
            
            # Suporte e resistência básicos
            recent_data = data.tail(20)
            resistencia = float(recent_data['High'].max())
            suporte = float(recent_data['Low'].min())
            
            return {
                "tendencia": tendencia,
                "recomendacao": recomendacao,
                "score_tecnico": float(score_total),
                "indicadores": {
                    "RSI": current_rsi,
                    "MACD": current_macd,
                    "MACD_signal": current_macd_signal,
                    "BB_position": current_bb_position,
                    "Estocástico_K": current_k,
                    "Estocástico_D": current_d,
                    "MA7": ma7,
                    "MA15": ma15,
                    "MA30": ma30
                },
                "niveis": {
                    "suporte": suporte,
                    "resistencia": resistencia,
                    "preco_atual": current_price
                }
            }
            
        except Exception as e:
            logger.error(f"Erro na análise técnica: {e}")
            return {"erro": f"Falha na análise técnica: {str(e)}"}
    
    def _fundamental_analysis(self, ticker: str) -> Dict[str, Any]:
        """Análise fundamentalista básica."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Métricas fundamentais básicas
            pe_ratio = info.get('trailingPE')
            pb_ratio = info.get('priceToBook')
            div_yield = info.get('dividendYield')
            market_cap = info.get('marketCap')
            debt_to_equity = info.get('debtToEquity')
            
            # Score fundamentalista básico
            score = 0
            
            if pe_ratio and 5 <= pe_ratio <= 25:
                score += 1
            if pb_ratio and pb_ratio <= 3:
                score += 1
            if div_yield and div_yield > 0.02:  # > 2%
                score += 1
            if debt_to_equity and debt_to_equity < 100:
                score += 1
            
            return {
                "metricas": {
                    "P/E": pe_ratio,
                    "P/B": pb_ratio,
                    "Dividend_Yield": div_yield,
                    "Market_Cap": market_cap,
                    "Debt_to_Equity": debt_to_equity
                },
                "score_fundamental": score,
                "max_score": 4
            }
            
        except Exception as e:
            logger.error(f"Erro na análise fundamental: {e}")
            return {"erro": str(e)}
    
    def _ml_analysis(self, data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """Análise com machine learning simples."""
        try:
            if len(data) < 50:
                return {"error": "Dados insuficientes para ML"}
            
            # Features simples para previsão
            data['returns'] = data['Close'].pct_change()
            data['volatility'] = data['returns'].rolling(window=20).std()
            data['momentum'] = data['Close'] / data['Close'].shift(10) - 1
            
            # Previsão simples baseada em momentum
            last_price = float(data['Close'].iloc[-1])
            last_momentum = float(data['momentum'].iloc[-1]) if not pd.isna(data['momentum'].iloc[-1]) else 0
            last_rsi = float(data['RSI'].iloc[-1])
            
            # Modelo simples baseado em regras
            prediction_change = 0
            confidence = 0.5
            
            if last_momentum > 0.05 and last_rsi < 70:
                prediction_change = 0.02  # +2%
                confidence = 0.7
                trend = "positivo"
            elif last_momentum < -0.05 and last_rsi > 30:
                prediction_change = -0.02  # -2%
                confidence = 0.7
                trend = "negativo"
            else:
                prediction_change = 0
                confidence = 0.5
                trend = "neutro"
            
            predicted_price = last_price * (1 + prediction_change)
            
            return {
                "price_prediction": {
                    "next_day": float(predicted_price),
                    "next_week": float(predicted_price * (1 + prediction_change * 0.5)),
                    "confidence": float(confidence)
                },
                "trend_prediction": trend,
                "risk_score": float(min(1.0, abs(prediction_change) * 10)),
                "features": {
                    "momentum": float(last_momentum),
                    "volatility": float(data['volatility'].iloc[-1]) if not pd.isna(data['volatility'].iloc[-1]) else 0,
                    "rsi": float(last_rsi)
                }
            }
            
        except Exception as e:
            logger.error(f"Erro na análise ML para {ticker}: {e}")
            return {"error": str(e)}
    
    def _sentiment_analysis(self, ticker: str) -> str:
        """Análise de sentimento básica."""
        try:
            # Análise simples baseada no ticker
            # Em implementação real, usaria APIs de notícias
            import random
            sentiments = ["positivo", "negativo", "neutro"]
            return random.choice(sentiments)
        except Exception as e:
            logger.warning(f"Erro na análise de sentimento: {e}")
            return "neutro"
    
    def _combine_analyses(self, ticker: str, data: pd.DataFrame, 
                         technical: Dict, fundamental: Dict, 
                         ml: Dict, sentiment: str) -> Dict[str, Any]:
        """Combina todas as análises em uma recomendação final."""
        try:
            last_row = data.iloc[-1]
            current_price = float(last_row['Close'])
            
            # Scores individuais
            tech_score = technical.get('score_tecnico', 0)
            fund_score = fundamental.get('score_fundamental', 0) - 2  # Normalizar para -2 a +2
            
            # Ajuste por sentimento
            sentiment_multiplier = 1.0
            if sentiment == "positivo":
                sentiment_multiplier = 1.2
            elif sentiment == "negativo":
                sentiment_multiplier = 0.8
            
            # Score combinado
            combined_score = (tech_score + fund_score) * sentiment_multiplier
            
            # Recomendação final
            if combined_score >= 1.5:
                final_recommendation = "FORTE COMPRA"
            elif combined_score >= 0.5:
                final_recommendation = "COMPRA"
            elif combined_score <= -1.5:
                final_recommendation = "FORTE VENDA"
            elif combined_score <= -0.5:
                final_recommendation = "VENDA"
            else:
                final_recommendation = "MANTER"
            
            return {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "preco_atual": current_price,
                "recomendacao_final": final_recommendation,
                "score_combinado": float(combined_score),
                "analise_tecnica": technical,
                "analise_fundamental": fundamental,
                "analise_ml": ml,
                "sentimento": sentiment,
                "confianca": min(100, max(0, (abs(combined_score) / 3) * 100))
            }
            
        except Exception as e:
            logger.error(f"Erro ao combinar análises: {e}")
            return self._error_response(ticker, str(e))
    
    def _error_response(self, ticker: str, error_msg: str) -> Dict[str, Any]:
        """Retorna resposta padrão para erros."""
        return {
            "ticker": ticker,
            "erro": error_msg,
            "timestamp": datetime.now().isoformat(),
            "recomendacao_final": "ERRO"
        }
    
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
            stock_data = self._get_stock_data(ticker)
            
            if stock_data.empty:
                return self._error_response(ticker, "Dados não encontrados")
            
            # Realizar análises
            technical_analysis = self._technical_analysis(stock_data)
            fundamental_analysis = self._fundamental_analysis(ticker)
            ml_analysis = self._ml_analysis(stock_data, ticker)
            sentiment_analysis = self._sentiment_analysis(ticker)
            
            # Combinar análises para recomendação final
            final_analysis = self._combine_analyses(
                ticker, stock_data, technical_analysis, 
                fundamental_analysis, ml_analysis, sentiment_analysis
            )
            
            # Cache do resultado
            self.cache[cache_key] = final_analysis
            
            return final_analysis
            
        except Exception as e:
            logger.error(f"Erro na análise de {ticker}: {e}")
            return self._error_response(ticker, str(e))


# Instância global do analisador
analyzer = FinancialAnalyzer()


# Funções de compatibilidade com a API existente
def analyze(ticker: str) -> Dict[str, Any]:
    """Função de compatibilidade para análise individual."""
    return analyzer.analyze_single_stock(ticker)


def analyze_all() -> Dict[str, Any]:
    """Análise de múltiplas ações populares."""
    tickers = [
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA",  # US
        "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA"  # BR
    ]
    
    results = {}
    errors = []
    
    for ticker in tickers:
        try:
            result = analyzer.analyze_single_stock(ticker)
            if "erro" not in result:
                results[ticker] = result
            else:
                errors.append(f"{ticker}: {result['erro']}")
        except Exception as e:
            errors.append(f"{ticker}: {str(e)}")
    
    return {
        "timestamp": datetime.now().isoformat(),
        "resultados": results,
        "total_analisados": len(results),
        "erros": errors
    }


def price_ticker(ticker: str) -> Dict[str, Any]:
    """Obtém apenas o preço atual de um ticker."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        
        if hist.empty:
            return {"erro": "Ticker inválido ou sem dados"}
        
        current_price = float(hist['Close'].iloc[-1])
        
        return {
            "ticker": ticker,
            "preco": current_price,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro ao obter preço de {ticker}: {e}")
        return {"erro": str(e)}