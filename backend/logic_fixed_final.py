"""
Módulo principal de análise financeira FUNCIONAL e CORRIGIDO.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Union
from functools import lru_cache
import warnings
import time

from technical_indicators import (
    compute_rsi, compute_macd, compute_bollinger_bands, 
    compute_stochastic_oscillator
)
from ml_models import FinancialMLModels
from sentiment_analysis import enhanced_sentiment_analysis
from config import settings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class FinancialAnalyzer:
    """Analisador financeiro principal com ML e cache."""
    
    def __init__(self):
        self.ml_models = FinancialMLModels()
        self.cache = {}
        self.cache_ttl = 300  # 5 minutos
        
        # Configurações de análise
        self.analysis_config = {
            'technical_weight': 0.4,
            'ml_weight': 0.3,
            'sentiment_weight': 0.2,
            'fundamental_weight': 0.1
        }
    
    def analyze_single_stock(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """Análise completa de uma ação com cache."""
        try:
            cache_key = f"{ticker}_{period}_{int(time.time() / self.cache_ttl)}"
            
            if cache_key in self.cache:
                logger.info(f"Retornando análise em cache para {ticker}")
                return self.cache[cache_key]
            
            # Buscar dados da ação
            stock_data = self._fetch_stock_data(ticker, period)
            
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
            
            # Armazenar no cache
            self.cache[cache_key] = final_analysis
            
            return final_analysis
            
        except Exception as e:
            logger.error(f"Erro na análise de {ticker}: {e}")
            return self._error_response(ticker, str(e))
    
    def _fetch_stock_data(self, ticker: str, period: str) -> pd.DataFrame:
        """Buscar dados históricos da ação."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                logger.warning(f"Nenhum dado encontrado para {ticker}")
            else:
                logger.info(f"Dados carregados para {ticker}: {len(data)} registros")
            
            return data
            
        except Exception as e:
            logger.error(f"Erro ao buscar dados para {ticker}: {e}")
            return pd.DataFrame()
    
    def _technical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Análise técnica completa."""
        try:
            if len(data) < 50:
                return {"error": "Dados insuficientes para análise técnica"}
            
            current_price = float(data['Close'].iloc[-1])
            
            # Calcular indicadores
            rsi = compute_rsi(data['Close'])
            macd = compute_macd(data['Close'])
            bb_upper, bb_middle, bb_lower = compute_bollinger_bands(data['Close'])
            stoch_k, stoch_d = compute_stochastic_oscillator(data['High'], data['Low'], data['Close'])
            
            # Médias móveis
            ma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
            ma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
            ma_200 = data['Close'].rolling(window=200).mean().iloc[-1] if len(data) >= 200 else None
            
            # Análise de tendência
            trend_analysis = self._analyze_trend(data, ma_20, ma_50, ma_200)
            
            # Suporte e resistência
            support_resistance = self._calculate_support_resistance(data)
            
            return {
                "current_price": current_price,
                "rsi": {
                    "value": float(rsi.iloc[-1]) if not rsi.empty else 50.0,
                    "signal": self._rsi_signal(rsi.iloc[-1] if not rsi.empty else 50.0)
                },
                "macd": {
                    "macd_line": float(macd['MACD'].iloc[-1]),
                    "signal_line": float(macd['Signal'].iloc[-1]),
                    "histogram": float(macd['Histogram'].iloc[-1]),
                    "signal": self._macd_signal(macd['Histogram'].iloc[-1])
                },
                "bollinger_bands": {
                    "upper": float(bb_upper.iloc[-1]),
                    "middle": float(bb_middle.iloc[-1]),
                    "lower": float(bb_lower.iloc[-1]),
                    "position": self._bb_position(current_price, bb_upper.iloc[-1], bb_lower.iloc[-1])
                },
                "stochastic": {
                    "k": float(stoch_k.iloc[-1]) if not stoch_k.empty else 50.0,
                    "d": float(stoch_d.iloc[-1]) if not stoch_d.empty else 50.0,
                    "signal": self._stoch_signal(stoch_k.iloc[-1] if not stoch_k.empty else 50.0)
                },
                "moving_averages": {
                    "ma_20": float(ma_20),
                    "ma_50": float(ma_50),
                    "ma_200": float(ma_200) if ma_200 else None
                },
                "trend": trend_analysis,
                "support_resistance": support_resistance
            }
            
        except Exception as e:
            logger.error(f"Erro na análise técnica: {e}")
            return {"error": str(e)}
    
    def _fundamental_analysis(self, ticker: str) -> Dict[str, Any]:
        """Análise fundamentalista básica."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                "market_cap": info.get('marketCap'),
                "pe_ratio": info.get('trailingPE'),
                "dividend_yield": info.get('dividendYield'),
                "debt_to_equity": info.get('debtToEquity'),
                "return_on_equity": info.get('returnOnEquity'),
                "price_to_book": info.get('priceToBook'),
                "sector": info.get('sector'),
                "industry": info.get('industry')
            }
            
        except Exception as e:
            logger.error(f"Erro na análise fundamentalista de {ticker}: {e}")
            return {"error": str(e)}
    
    def _ml_analysis(self, data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """Análise com machine learning."""
        try:
            # Preparar features simples
            if len(data) < 60:
                return {"error": "Dados insuficientes para ML"}
            
            # Features básicas para ML
            features = pd.DataFrame({
                'returns': data['Close'].pct_change(),
                'volume': data['Volume'],
                'high_low_pct': (data['High'] - data['Low']) / data['Close'],
                'close_open_pct': (data['Close'] - data['Open']) / data['Open']
            }).dropna()
            
            if len(features) < 30:
                return {"error": "Features insuficientes para ML"}
            
            # Treinar modelo simples se possível
            try:
                self.ml_models.prepare_features(features)
                self.ml_models.train_models()
                predictions = self.ml_models.predict()
                
                return {
                    "price_prediction": {
                        "next_day": float(predictions.get('next_day_prediction', data['Close'].iloc[-1])),
                        "next_week": float(predictions.get('next_week_prediction', data['Close'].iloc[-1])),
                        "confidence": float(predictions.get('confidence', 0.5))
                    },
                    "trend_prediction": predictions.get('trend_direction', 'neutral'),
                    "risk_score": float(predictions.get('risk_score', 0.5))
                }
            except Exception as ml_error:
                logger.warning(f"Erro no ML para {ticker}: {ml_error}")
                return {"error": f"ML training failed: {str(ml_error)}"}
                
        except Exception as e:
            logger.error(f"Erro na análise ML para {ticker}: {e}")
            return {"error": str(e)}
    
    def _sentiment_analysis(self, ticker: str) -> Dict[str, Any]:
        """Análise de sentimento."""
        try:
            sentiment = enhanced_sentiment_analysis(ticker)
            
            return {
                "sentiment": sentiment,
                "impact": "positivo" if sentiment == "positivo" else "negativo" if sentiment == "negativo" else "neutro"
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de sentimento para {ticker}: {e}")
            return {"sentiment": "neutro", "impact": "neutro", "error": str(e)}
    
    def _combine_analyses(self, ticker: str, data: pd.DataFrame, technical: Dict[str, Any], 
                         fundamental: Dict[str, Any], ml_analysis: Dict[str, Any], 
                         sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """Combinar todas as análises em uma recomendação final."""
        try:
            # Calcular scores
            tech_score = self._calculate_technical_score(technical)
            ml_score = self._calculate_ml_score(ml_analysis)
            sentiment_score = self._calculate_sentiment_score(sentiment)
            fundamental_score = self._calculate_fundamental_score(fundamental)
            
            # Score ponderado final
            final_score = (
                tech_score * self.analysis_config['technical_weight'] +
                ml_score * self.analysis_config['ml_weight'] +
                sentiment_score * self.analysis_config['sentiment_weight'] +
                fundamental_score * self.analysis_config['fundamental_weight']
            )
            
            # Gerar recomendação
            recommendation = self._generate_recommendation(final_score)
            risk_level = self._calculate_risk_level(technical, ml_analysis, data)
            price_targets = self._calculate_price_targets(data, technical, ml_analysis)
            
            return {
                "ticker": ticker,
                "analysis_timestamp": datetime.now().isoformat(),
                "current_price": float(data['Close'].iloc[-1]),
                "recommendation": recommendation['action'],
                "confidence": recommendation['confidence'],
                "final_score": final_score,
                "risk_level": risk_level,
                "price_targets": price_targets,
                "detailed_scores": {
                    "technical": tech_score,
                    "machine_learning": ml_score,
                    "sentiment": sentiment_score,
                    "fundamental": fundamental_score
                },
                "technical_analysis": technical,
                "fundamental_analysis": fundamental,
                "ml_analysis": ml_analysis,
                "sentiment_analysis": sentiment
            }
            
        except Exception as e:
            logger.error(f"Erro ao combinar análises para {ticker}: {e}")
            return self._error_response(ticker, str(e))
    
    def _calculate_technical_score(self, technical: Dict[str, Any]) -> float:
        """Calcular score técnico (-1 a 1)."""
        try:
            if 'error' in technical:
                return 0.0
            
            score = 0.0
            
            # RSI
            rsi_value = technical.get('rsi', {}).get('value', 50)
            if rsi_value < 30:
                score += 0.3  # Oversold - positivo
            elif rsi_value > 70:
                score -= 0.3  # Overbought - negativo
            
            # MACD
            macd_hist = technical.get('macd', {}).get('histogram', 0)
            if macd_hist > 0:
                score += 0.2
            else:
                score -= 0.2
            
            # Bollinger Bands
            bb_position = technical.get('bollinger_bands', {}).get('position', 'middle')
            if bb_position == 'below':
                score += 0.2
            elif bb_position == 'above':
                score -= 0.2
            
            # Trend
            trend = technical.get('trend', {}).get('direction', 'sideways')
            if trend == 'upward':
                score += 0.3
            elif trend == 'downward':
                score -= 0.3
            
            return max(-1.0, min(1.0, score))
            
        except Exception:
            return 0.0
    
    def _calculate_ml_score(self, ml_analysis: Dict[str, Any]) -> float:
        """Calcular score de ML (-1 a 1)."""
        try:
            if 'error' in ml_analysis:
                return 0.0
            
            trend = ml_analysis.get('trend_prediction', 'neutral')
            confidence = ml_analysis.get('price_prediction', {}).get('confidence', 0.5)
            
            if trend == 'bullish':
                return confidence
            elif trend == 'bearish':
                return -confidence
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_sentiment_score(self, sentiment: Dict[str, Any]) -> float:
        """Calcular score de sentiment (-1 a 1)."""
        try:
            sentiment_value = sentiment.get('sentiment', 'neutro')
            
            if sentiment_value == 'positivo':
                return 0.5
            elif sentiment_value == 'negativo':
                return -0.5
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_fundamental_score(self, fundamental: Dict[str, Any]) -> float:
        """Calcular score fundamentalista (-1 a 1)."""
        try:
            if 'error' in fundamental:
                return 0.0
            
            score = 0.0
            
            # P/E Ratio
            pe = fundamental.get('pe_ratio')
            if pe and pe < 15:
                score += 0.3
            elif pe and pe > 25:
                score -= 0.3
            
            # Dividend Yield
            div_yield = fundamental.get('dividend_yield')
            if div_yield and div_yield > 0.03:  # > 3%
                score += 0.2
            
            return max(-1.0, min(1.0, score))
            
        except Exception:
            return 0.0
    
    def _generate_recommendation(self, score: float) -> Dict[str, Any]:
        """Gerar recomendação baseada no score final."""
        if score > 0.3:
            return {"action": "COMPRAR", "confidence": min(90, 50 + score * 40)}
        elif score < -0.3:
            return {"action": "VENDER", "confidence": min(90, 50 + abs(score) * 40)}
        else:
            return {"action": "MANTER", "confidence": 30}
    
    def _calculate_risk_level(self, technical: Dict[str, Any], ml_analysis: Dict[str, Any], 
                             data: pd.DataFrame) -> str:
        """Calcular nível de risco."""
        try:
            # Volatilidade histórica
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Anualizada
            
            # Risk score do ML
            ml_risk = ml_analysis.get('risk_score', 0.5)
            
            # Combinar fatores
            if volatility > 0.4 or ml_risk > 0.7:
                return "ALTO"
            elif volatility > 0.2 or ml_risk > 0.4:
                return "MÉDIO"
            else:
                return "BAIXO"
                
        except Exception:
            return "MÉDIO"
    
    def _calculate_price_targets(self, data: pd.DataFrame, technical: Dict[str, Any], 
                                ml_analysis: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Calcular alvos de preço."""
        try:
            current_price = float(data['Close'].iloc[-1])
            
            # Target baseado em ML
            ml_prediction = ml_analysis.get('price_prediction', {}).get('next_week')
            
            # Target baseado em suporte/resistência
            support_resistance = technical.get('support_resistance', {})
            resistance = support_resistance.get('resistance')
            support = support_resistance.get('support')
            
            return {
                "target_price": ml_prediction if ml_prediction else resistance,
                "stop_loss": support if support else current_price * 0.95
            }
            
        except Exception:
            return {"target_price": None, "stop_loss": None}
    
    def _error_response(self, ticker: str, error_message: str) -> Dict[str, Any]:
        """Resposta padrão para erro."""
        return {
            "ticker": ticker,
            "error": error_message,
            "recommendation": "ERRO",
            "confidence": 0,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    # Métodos auxiliares para sinais
    def _rsi_signal(self, rsi: float) -> str:
        if rsi < 30:
            return "oversold"
        elif rsi > 70:
            return "overbought"
        else:
            return "neutral"
    
    def _macd_signal(self, histogram: float) -> str:
        return "bullish" if histogram > 0 else "bearish"
    
    def _bb_position(self, price: float, upper: float, lower: float) -> str:
        if price > upper:
            return "above"
        elif price < lower:
            return "below"
        else:
            return "middle"
    
    def _stoch_signal(self, k_value: float) -> str:
        if k_value < 20:
            return "oversold"
        elif k_value > 80:
            return "overbought"
        else:
            return "neutral"
    
    def _analyze_trend(self, data: pd.DataFrame, ma_20: float, ma_50: float, ma_200: Optional[float]) -> Dict[str, Any]:
        """Analisar tendência dos preços."""
        try:
            current_price = data['Close'].iloc[-1]
            
            # Direção baseada em médias móveis
            if current_price > ma_20 > ma_50:
                direction = "upward"
            elif current_price < ma_20 < ma_50:
                direction = "downward"
            else:
                direction = "sideways"
            
            # Força da tendência
            price_change_20d = (current_price - data['Close'].iloc[-20]) / data['Close'].iloc[-20]
            
            if abs(price_change_20d) > 0.1:
                strength = "strong"
            elif abs(price_change_20d) > 0.05:
                strength = "moderate"
            else:
                strength = "weak"
            
            return {
                "direction": direction,
                "strength": strength,
                "change_20d": float(price_change_20d * 100)
            }
            
        except Exception:
            return {"direction": "sideways", "strength": "weak", "change_20d": 0.0}
    
    def _calculate_support_resistance(self, data: pd.DataFrame) -> Dict[str, Optional[float]]:
        """Calcular níveis de suporte e resistência."""
        try:
            # Últimos 50 dias para cálculo
            recent_data = data.tail(50)
            
            # Resistência - máxima dos últimos dias
            resistance = float(recent_data['High'].max())
            
            # Suporte - mínima dos últimos dias
            support = float(recent_data['Low'].min())
            
            return {
                "resistance": resistance,
                "support": support
            }
            
        except Exception:
            return {"resistance": None, "support": None}


# Instância global
financial_analyzer = FinancialAnalyzer()


# Função de compatibilidade
def analyze_single_stock(ticker: str, period: str = "1y") -> Dict[str, Any]:
    """Função de compatibilidade para análise de ação."""
    return financial_analyzer.analyze_single_stock(ticker, period)