import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import requests
from typing import Dict, Any, List, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import io
import base64
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import io
import base64

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
            
            # Indicadores avançados
            # Williams %R
            high14 = data['High'].rolling(window=14).max()
            low14 = data['Low'].rolling(window=14).min()
            data['Williams_R'] = -100 * (high14 - data['Close']) / (high14 - low14)
            
            # CCI (Commodity Channel Index)
            tp = (data['High'] + data['Low'] + data['Close']) / 3
            data['CCI'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean()))))
            
            # OBV (On-Balance Volume)
            data['OBV'] = 0.0
            for i in range(1, len(data)):
                if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                    data['OBV'].iloc[i] = data['OBV'].iloc[i-1] + data['Volume'].iloc[i]
                elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                    data['OBV'].iloc[i] = data['OBV'].iloc[i-1] - data['Volume'].iloc[i]
                else:
                    data['OBV'].iloc[i] = data['OBV'].iloc[i-1]
            
            # Rate of Change (ROC)
            data['ROC'] = data['Close'].pct_change(periods=10) * 100
            
            # VWAP (Volume Weighted Average Price)
            data['VWAP'] = (data['Close'] * data['Volume']).rolling(window=20).sum() / data['Volume'].rolling(window=20).sum()
            
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
            current_williams_r = float(last_row['Williams_R'])
            current_cci = float(last_row['CCI'])
            current_roc = float(last_row['ROC'])
            current_obv = float(last_row['OBV'])
            current_vwap = float(last_row['VWAP'])
            
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
            
            # Score Williams %R
            if current_williams_r > -20:
                score_williams = -0.5  # Sobrecomprado
            elif current_williams_r < -80:
                score_williams = 0.5   # Sobrevendido
            else:
                score_williams = 0
            
            # Score CCI
            if current_cci > 100:
                score_cci = -0.5  # Sobrecomprado
            elif current_cci < -100:
                score_cci = 0.5   # Sobrevendido
            else:
                score_cci = 0
            
            # Score ROC (momentum)
            if current_roc > 5:
                score_roc = 1  # Momentum forte positivo
            elif current_roc > 2:
                score_roc = 0.5  # Momentum positivo
            elif current_roc < -5:
                score_roc = -1  # Momentum forte negativo
            elif current_roc < -2:
                score_roc = -0.5  # Momentum negativo
            else:
                score_roc = 0
            
            # Score VWAP
            if current_price > current_vwap * 1.02:
                score_vwap = 0.5  # Acima do VWAP
            elif current_price < current_vwap * 0.98:
                score_vwap = -0.5  # Abaixo do VWAP
            else:
                score_vwap = 0
            
            # Peso por volume (mais volume = mais confiável)
            volume_weight = min(2.0, max(0.5, float(last_row['Volume']) / data['Volume'].mean()))
            
            # Score técnico combinado com pesos inteligentes
            base_score = score_tendencia + score_rsi + score_macd + score_bb + score_stoch
            advanced_score = score_williams + score_cci + score_roc + score_vwap
            score_total = (base_score * 0.7 + advanced_score * 0.3) * volume_weight
            
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
                    "Williams_R": current_williams_r,
                    "CCI": current_cci,
                    "ROC": current_roc,
                    "VWAP": current_vwap,
                    "MA7": ma7,
                    "MA15": ma15,
                    "MA30": ma30
                },
                "volume_analysis": {
                    "OBV": current_obv,
                    "volume_ratio": float(last_row['Volume']) / data['Volume'].mean(),
                    "volume_weight": volume_weight
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
            
            # Features avançadas para previsão
            data['returns'] = data['Close'].pct_change()
            data['volatility'] = data['returns'].rolling(window=20).std()
            data['momentum'] = data['Close'] / data['Close'].shift(10) - 1
            data['momentum_5d'] = data['Close'] / data['Close'].shift(5) - 1
            data['momentum_20d'] = data['Close'] / data['Close'].shift(20) - 1
            data['volume_momentum'] = data['Volume'] / data['Volume'].shift(10) - 1
            
            # Features técnicas
            data['rsi_momentum'] = data['RSI'].diff(5)
            data['macd_momentum'] = data['MACD'].diff(3)
            data['bb_squeeze'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']
            
            # Previsão baseada em múltiplos fatores
            last_price = float(data['Close'].iloc[-1])
            last_momentum = float(data['momentum'].iloc[-1]) if not pd.isna(data['momentum'].iloc[-1]) else 0
            last_momentum_5d = float(data['momentum_5d'].iloc[-1]) if not pd.isna(data['momentum_5d'].iloc[-1]) else 0
            last_momentum_20d = float(data['momentum_20d'].iloc[-1]) if not pd.isna(data['momentum_20d'].iloc[-1]) else 0
            last_rsi = float(data['RSI'].iloc[-1])
            last_rsi_momentum = float(data['rsi_momentum'].iloc[-1]) if not pd.isna(data['rsi_momentum'].iloc[-1]) else 0
            last_volume_momentum = float(data['volume_momentum'].iloc[-1]) if not pd.isna(data['volume_momentum'].iloc[-1]) else 0
            
            # Modelo avançado multi-dimensional
            momentum_score = 0
            technical_score = 0
            volume_score = 0
            
            # Score de momentum (múltiplos timeframes)
            if last_momentum > 0.1:
                momentum_score += 2
            elif last_momentum > 0.05:
                momentum_score += 1
            elif last_momentum < -0.1:
                momentum_score -= 2
            elif last_momentum < -0.05:
                momentum_score -= 1
            
            if last_momentum_5d > 0.03:
                momentum_score += 1
            elif last_momentum_5d < -0.03:
                momentum_score -= 1
            
            # Score técnico
            if last_rsi < 30 and last_rsi_momentum > 0:
                technical_score += 1  # RSI oversold e recuperando
            elif last_rsi > 70 and last_rsi_momentum < 0:
                technical_score -= 1  # RSI overbought e caindo
            
            # Score de volume
            if last_volume_momentum > 0.2:
                volume_score += 1  # Volume crescente
            elif last_volume_momentum < -0.2:
                volume_score -= 1  # Volume declinante
            
            # Score total e predição
            total_score = momentum_score + technical_score + volume_score
            
            # Predição baseada no score total
            if total_score >= 3:
                prediction_change = 0.04  # +4%
                confidence = 0.85
                trend = "muito_positivo"
            elif total_score >= 1:
                prediction_change = 0.02  # +2%
                confidence = 0.7
                trend = "positivo"
            elif total_score <= -3:
                prediction_change = -0.04  # -4%
                confidence = 0.85
                trend = "muito_negativo"
            elif total_score <= -1:
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
                    "momentum_5d": float(last_momentum_5d),
                    "momentum_20d": float(last_momentum_20d),
                    "volatility": float(data['volatility'].iloc[-1]) if not pd.isna(data['volatility'].iloc[-1]) else 0,
                    "rsi": float(last_rsi),
                    "rsi_momentum": float(last_rsi_momentum),
                    "volume_momentum": float(last_volume_momentum),
                    "total_score": total_score
                },
                "risk_management": {
                    "stop_loss": float(last_price * (1 - min(0.15, max(0.03, data['volatility'].iloc[-1] * 3)))),
                    "take_profit": float(last_price * (1 + abs(prediction_change) * 2)),
                    "position_size": min(100, max(10, 50 / (data['volatility'].iloc[-1] * 100))),
                    "risk_reward_ratio": abs(prediction_change) * 2 / min(0.15, max(0.03, data['volatility'].iloc[-1] * 3))
                }
            }
            
        except Exception as e:
            logger.error(f"Erro na análise ML para {ticker}: {e}")
            return {"error": str(e)}
    
    def _sentiment_analysis(self, ticker: str) -> Dict[str, Any]:
        """Análise de sentimento avançada."""
        try:
            # Análise baseada em dados técnicos como proxy
            stock = yf.Ticker(ticker)
            
            # Obter dados básicos para sentiment
            recent_data = stock.history(period="5d")
            if recent_data.empty:
                return {"sentiment": "neutro", "score": 0, "confidence": 0.3}
            
            # Calcular sentiment baseado em performance recente
            recent_change = (recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[0] - 1) * 100
            volume_trend = recent_data['Volume'].iloc[-3:].mean() / recent_data['Volume'].mean()
            
            sentiment_score = 0
            
            # Performance dos últimos dias
            if recent_change > 5:
                sentiment_score += 2
            elif recent_change > 2:
                sentiment_score += 1
            elif recent_change < -5:
                sentiment_score -= 2
            elif recent_change < -2:
                sentiment_score -= 1
            
            # Volume como indicador de interesse
            if volume_trend > 1.5:
                sentiment_score += 1 if recent_change > 0 else -1
            
            # Determinar sentiment
            if sentiment_score >= 2:
                sentiment = "muito_positivo"
                confidence = 0.8
            elif sentiment_score >= 1:
                sentiment = "positivo"
                confidence = 0.7
            elif sentiment_score <= -2:
                sentiment = "muito_negativo"
                confidence = 0.8
            elif sentiment_score <= -1:
                sentiment = "negativo"
                confidence = 0.7
            else:
                sentiment = "neutro"
                confidence = 0.5
            
            return {
                "sentiment": sentiment,
                "score": sentiment_score,
                "confidence": confidence,
                "recent_change": recent_change,
                "volume_trend": volume_trend
            }
            
        except Exception as e:
            logger.warning(f"Erro na análise de sentimento: {e}")
            return {"sentiment": "neutro", "score": 0, "confidence": 0.3}
    
    def _combine_analyses(self, ticker: str, data: pd.DataFrame, 
                         technical: Dict, fundamental: Dict, 
                         ml: Dict, sentiment: Dict) -> Dict[str, Any]:
        """Combina todas as análises em uma recomendação final."""
        try:
            last_row = data.iloc[-1]
            current_price = float(last_row['Close'])
            
            # Scores individuais
            tech_score = technical.get('score_tecnico', 0)
            fund_score = fundamental.get('score_fundamental', 0) - 2  # Normalizar para -2 a +2
            ml_score = ml.get('features', {}).get('total_score', 0) * 0.5  # Score ML normalizado
            
            # Ajuste por sentimento com pesos dinâmicos
            sentiment_str = sentiment.get('sentiment', 'neutro')
            sentiment_score = sentiment.get('score', 0)
            sentiment_confidence = sentiment.get('confidence', 0.5)
            
            # Multiplicador baseado na confiança do sentiment
            sentiment_weight = 0.1 + (sentiment_confidence - 0.3) * 0.4  # 0.1 a 0.5
            sentiment_multiplier = 1.0 + (sentiment_score * sentiment_weight * 0.1)
            
            # Score combinado com pesos inteligentes
            # Peso maior para análise técnica em mercados voláteis
            volatility = data['volatilidade'].iloc[-1]
            tech_weight = 0.5 + min(0.3, volatility * 20)  # 0.5 a 0.8
            fund_weight = 0.3 - min(0.1, volatility * 10)   # 0.2 a 0.3
            ml_weight = 0.2
            
            combined_score = (tech_score * tech_weight + 
                            fund_score * fund_weight + 
                            ml_score * ml_weight) * sentiment_multiplier
            
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
            
            # Calcular confiança baseada em convergência de sinais
            convergence_score = 0
            if (tech_score > 0 and ml_score > 0 and sentiment_score > 0) or \
               (tech_score < 0 and ml_score < 0 and sentiment_score < 0):
                convergence_score = 1  # Todos concordam
            elif abs(tech_score - ml_score) < 0.5:
                convergence_score = 0.7  # Técnica e ML concordam
            else:
                convergence_score = 0.3  # Sinais divergentes
            
            confianca = min(95, max(30, convergence_score * 80 + sentiment_confidence * 20))
            
            return {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "preco_atual": current_price,
                "recomendacao_final": final_recommendation,
                "score_combinado": float(combined_score),
                "confianca": float(confianca),
                "convergencia_sinais": float(convergence_score),
                "analise_tecnica": technical,
                "analise_fundamental": fundamental,
                "analise_ml": ml,
                "analise_sentimento": sentiment,
                "pesos_analise": {
                    "tecnica": float(tech_weight),
                    "fundamental": float(fund_weight),
                    "ml": float(ml_weight),
                    "sentimento": float(sentiment_weight)
                },
                "risk_metrics": {
                    "volatilidade": float(volatility),
                    "beta_estimado": float(min(2.0, max(0.5, volatility * 50))),
                    "sharpe_estimado": float(max(-2, min(2, combined_score / max(0.1, volatility))))
                }
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

    def _plot_candlesticks(self, ax, dates, data):
        """Plota candlesticks no gráfico."""
        for i, date in enumerate(dates):
            open_price = data['Open'].iloc[i]
            high_price = data['High'].iloc[i] 
            low_price = data['Low'].iloc[i]
            close_price = data['Close'].iloc[i]
            
            # Cor da vela
            color = 'green' if close_price >= open_price else 'red'
            alpha = 0.8
            
            # Corpo da vela
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            rect = Rectangle((mdates.date2num(date) - 0.3, body_bottom), 
                           0.6, body_height, 
                           facecolor=color, edgecolor='black', 
                           alpha=alpha, linewidth=0.5)
            ax.add_patch(rect)
            
            # Sombras (wicks)
            ax.plot([mdates.date2num(date), mdates.date2num(date)], 
                   [low_price, high_price], 
                   color='black', linewidth=1, alpha=0.8)
    
    def _generate_future_predictions(self, data: pd.DataFrame, ml_data: Dict, days: int) -> Dict:
        """Gera previsões futuras para o gráfico."""
        try:
            last_price = float(data['Close'].iloc[-1])
            volatility = float(data['volatilidade'].iloc[-1])
            
            # Obter previsão base do ML
            base_prediction = ml_data.get('price_prediction', {})
            next_day_pred = base_prediction.get('next_day', last_price)
            confidence = base_prediction.get('confidence', 0.5)
            
            # Gerar datas futuras (apenas dias úteis)
            last_date = data.index[-1]
            future_dates = []
            current_date = last_date
            
            for i in range(days):
                current_date += timedelta(days=1)
                # Pular fins de semana
                while current_date.weekday() >= 5:
                    current_date += timedelta(days=1)
                future_dates.append(current_date)
            
            # Gerar previsões com trend e noise
            trend_prediction = ml_data.get('trend_prediction', 'neutro')
            
            if trend_prediction == 'muito_positivo':
                daily_trend = 0.015  # 1.5% por dia
            elif trend_prediction == 'positivo': 
                daily_trend = 0.007  # 0.7% por dia
            elif trend_prediction == 'muito_negativo':
                daily_trend = -0.015  # -1.5% por dia
            elif trend_prediction == 'negativo':
                daily_trend = -0.007  # -0.7% por dia
            else:
                daily_trend = 0.001  # 0.1% por dia (neutro)
            
            # Calcular previsões
            prices = []
            confidence_upper = []
            confidence_lower = []
            
            current_pred_price = next_day_pred
            
            for i in range(days):
                # Aplicar trend com diminuição ao longo do tempo
                trend_factor = daily_trend * (0.95 ** i)  # Decaimento
                noise_factor = np.random.normal(0, volatility * 0.3)  # Ruído baseado na volatilidade
                
                current_pred_price *= (1 + trend_factor + noise_factor * 0.1)
                prices.append(current_pred_price)
                
                # Intervalo de confiança cresce com o tempo
                confidence_range = current_pred_price * volatility * (1 + i * 0.1) * (2 - confidence)
                confidence_upper.append(current_pred_price + confidence_range)
                confidence_lower.append(current_pred_price - confidence_range)
            
            return {
                'dates': future_dates,
                'prices': prices,
                'confidence_upper': confidence_upper,
                'confidence_lower': confidence_lower,
                'days_forecast': days,
                'base_prediction': next_day_pred,
                'trend': trend_prediction
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar previsões futuras: {e}")
            # Retorno de fallback
            future_dates = [data.index[-1] + timedelta(days=i+1) for i in range(days)]
            flat_price = float(data['Close'].iloc[-1])
            return {
                'dates': future_dates,
                'prices': [flat_price] * days,
                'confidence_upper': [flat_price * 1.05] * days,
                'confidence_lower': [flat_price * 0.95] * days,
                'days_forecast': days,
                'base_prediction': flat_price,
                'trend': 'neutro'
            }
    
    def generate_chart_data(self, ticker: str, days_forecast: int = 10) -> Dict[str, Any]:
        """
        Gera dados estruturados para gráficos no frontend (sem imagem).
        
        Returns:
            Dict com dados históricos, previsões e análises
        """
        try:
            # Obter dados históricos estendidos
            stock_data = self._get_stock_data(ticker, period="6mo")
            
            if stock_data.empty:
                return {"erro": "Dados não encontrados"}
            
            # Realizar análise para obter previsões
            technical_analysis = self._technical_analysis(stock_data)
            fundamental_analysis = self._fundamental_analysis(ticker)
            ml_analysis = self._ml_analysis(stock_data, ticker)
            sentiment_analysis = self._sentiment_analysis(ticker)
            
            analysis = self._combine_analyses(
                ticker, stock_data, technical_analysis, 
                fundamental_analysis, ml_analysis, sentiment_analysis
            )
            ml_data = analysis.get('analise_ml', {})
            
            # Gerar previsões futuras
            future_predictions = self._generate_future_predictions(stock_data, ml_data, days_forecast)
            
            # Preparar dados históricos (últimos 90 dias)
            days_to_show = min(90, len(stock_data))
            historical_data = stock_data.iloc[-days_to_show:].copy()
            
            # Converter dados para formato JSON-friendly
            chart_data = []
            for i in range(len(historical_data)):
                chart_data.append({
                    "date": historical_data.index[i].strftime('%Y-%m-%d'),
                    "timestamp": int(historical_data.index[i].timestamp() * 1000),
                    "open": float(historical_data['Open'].iloc[i]),
                    "high": float(historical_data['High'].iloc[i]),
                    "low": float(historical_data['Low'].iloc[i]),
                    "close": float(historical_data['Close'].iloc[i]),
                    "volume": int(historical_data['Volume'].iloc[i]),
                    "ma7": float(historical_data['MA7'].iloc[i]) if 'MA7' in historical_data.columns else None,
                    "ma15": float(historical_data['MA15'].iloc[i]) if 'MA15' in historical_data.columns else None,
                    "ma30": float(historical_data['MA30'].iloc[i]) if 'MA30' in historical_data.columns else None,
                    "rsi": float(historical_data['RSI'].iloc[i]) if 'RSI' in historical_data.columns else None,
                    "macd": float(historical_data['MACD'].iloc[i]) if 'MACD' in historical_data.columns else None,
                    "bb_upper": float(historical_data['BB_upper'].iloc[i]) if 'BB_upper' in historical_data.columns else None,
                    "bb_lower": float(historical_data['BB_lower'].iloc[i]) if 'BB_lower' in historical_data.columns else None,
                    "bb_middle": float(historical_data['BB_middle'].iloc[i]) if 'BB_middle' in historical_data.columns else None,
                })
            
            # Preparar dados de previsão
            prediction_data = []
            for i in range(len(future_predictions['dates'])):
                prediction_data.append({
                    "date": future_predictions['dates'][i].strftime('%Y-%m-%d'),
                    "timestamp": int(future_predictions['dates'][i].timestamp() * 1000),
                    "predicted_price": float(future_predictions['prices'][i]),
                    "confidence_upper": float(future_predictions['confidence_upper'][i]),
                    "confidence_lower": float(future_predictions['confidence_lower'][i])
                })
            
            # Preparar indicadores atuais
            current_indicators = {}
            if len(historical_data) > 0:
                last_row = historical_data.iloc[-1]
                current_indicators = {
                    "RSI": float(last_row['RSI']) if 'RSI' in last_row and pd.notna(last_row['RSI']) else None,
                    "MACD": float(last_row['MACD']) if 'MACD' in last_row and pd.notna(last_row['MACD']) else None,
                    "Williams_R": float(last_row['Williams_R']) if 'Williams_R' in last_row and pd.notna(last_row['Williams_R']) else None,
                    "CCI": float(last_row['CCI']) if 'CCI' in last_row and pd.notna(last_row['CCI']) else None,
                    "BB_position": float(last_row['BB_position']) if 'BB_position' in last_row and pd.notna(last_row['BB_position']) else None,
                    "volatility": float(last_row['volatilidade']) if 'volatilidade' in last_row and pd.notna(last_row['volatilidade']) else None,
                    "volume": int(last_row['Volume']) if pd.notna(last_row['Volume']) else None
                }
            
            # Dados de preço
            current_price = float(historical_data['Close'].iloc[-1])
            predicted_price = float(future_predictions['prices'][-1])
            price_change = ((predicted_price / current_price) - 1) * 100
            
            return {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "historical_data": chart_data,
                "prediction_data": prediction_data,
                "analysis": {
                    "recommendation": analysis.get("recomendacao_final"),
                    "confidence": analysis.get("confianca"),
                    "current_price": current_price,
                    "predicted_price": predicted_price,
                    "price_change_percent": float(price_change),
                    "trend": ml_data.get('trend_prediction', 'neutro')
                },
                "indicators": current_indicators,
                "risk_management": ml_data.get('risk_management', {}),
                "days_forecast": days_forecast,
                "data_points": len(chart_data)
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar dados do gráfico para {ticker}: {e}")
            return {"erro": f"Falha ao gerar dados: {str(e)}"}

    # Manter o método antigo para compatibilidade
    def generate_financial_chart(self, ticker: str, days_forecast: int = 10, chart_type: str = "candlestick") -> Dict[str, Any]:
        """
        Gera gráfico financeiro com previsões futuras.
        
        Args:
            ticker: Símbolo da ação
            days_forecast: Dias de previsão para frente
            chart_type: Tipo de gráfico ("candlestick", "line", "ohlc")
        
        Returns:
            Dicionário com gráfico em base64 e dados
        """
        try:
            # Obter dados históricos estendidos
            stock_data = self._get_stock_data(ticker, period="6mo")
            
            if stock_data.empty:
                return {"erro": "Dados não encontrados para gerar gráfico"}
            
            # Realizar análise para obter previsões
            # Usar métodos internos para evitar recursão
            technical_analysis = self._technical_analysis(stock_data)
            fundamental_analysis = self._fundamental_analysis(ticker)
            ml_analysis = self._ml_analysis(stock_data, ticker)
            sentiment_analysis = self._sentiment_analysis(ticker)
            
            analysis = self._combine_analyses(
                ticker, stock_data, technical_analysis, 
                fundamental_analysis, ml_analysis, sentiment_analysis
            )
            ml_data = analysis.get('analise_ml', {})
            
            # Gerar previsões futuras
            future_predictions = self._generate_future_predictions(stock_data, ml_data, days_forecast)
            
            # Configurar figura
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), 
                                         gridspec_kw={'height_ratios': [3, 1]})
            
            # Dados históricos - Últimos 3 meses (~90 dias)
            days_to_show = min(90, len(stock_data))  # Até 90 dias ou todos os disponíveis
            dates = stock_data.index[-days_to_show:]
            data_subset = stock_data.iloc[-days_to_show:]
            
            # Gráfico principal de preços
            if chart_type == "candlestick":
                self._plot_candlesticks(ax1, dates, data_subset)
            elif chart_type == "line":
                ax1.plot(dates, data_subset['Close'], 'b-', linewidth=2, label='Preço')
            
            # Adicionar médias móveis
            ax1.plot(dates, data_subset['MA7'], 'orange', linewidth=1, alpha=0.7, label='MA7')
            ax1.plot(dates, data_subset['MA15'], 'red', linewidth=1, alpha=0.7, label='MA15')
            ax1.plot(dates, data_subset['MA30'], 'purple', linewidth=1, alpha=0.7, label='MA30')
            
            # Adicionar Bollinger Bands
            ax1.fill_between(dates, data_subset['BB_upper'], data_subset['BB_lower'], 
                           alpha=0.1, color='gray', label='Bollinger Bands')
            ax1.plot(dates, data_subset['BB_upper'], '--', color='gray', linewidth=0.8)
            ax1.plot(dates, data_subset['BB_lower'], '--', color='gray', linewidth=0.8)
            
            # Adicionar previsões futuras
            future_dates = future_predictions['dates']
            future_prices = future_predictions['prices']
            confidence_upper = future_predictions['confidence_upper']
            confidence_lower = future_predictions['confidence_lower']
            
            # Conectar último preço histórico com primeira previsão
            connection_dates = [dates[-1], future_dates[0]]
            connection_prices = [data_subset['Close'].iloc[-1], future_prices[0]]
            ax1.plot(connection_dates, connection_prices, '--', color='blue', linewidth=2, alpha=0.7)
            
            # Linha de previsão
            ax1.plot(future_dates, future_prices, 'g--', linewidth=3, label='Previsão ML', alpha=0.8)
            
            # Área de confiança
            ax1.fill_between(future_dates, confidence_lower, confidence_upper,
                           alpha=0.2, color='green', label='Intervalo de Confiança')
            
            # Adicionar linha vertical separando histórico de previsão
            ax1.axvline(x=dates[-1], color='red', linestyle=':', alpha=0.7, label='Hoje')
            
            # Configurações do gráfico principal
            ax1.set_title(f'{ticker} - Análise Técnica com Previsões ML\n'
                         f'Recomendação: {analysis.get("recomendacao_final", "N/A")} '
                         f'(Confiança: {analysis.get("confianca", 0):.1f}%)', 
                         fontsize=14, fontweight='bold')
            ax1.set_ylabel('Preço (USD)', fontsize=12)
            ax1.legend(loc='upper left', fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # Gráfico de volume
            volume_dates = dates
            volumes = data_subset['Volume']
            colors = ['green' if data_subset['Close'].iloc[i] >= data_subset['Open'].iloc[i] 
                     else 'red' for i in range(len(data_subset))]
            
            ax2.bar(volume_dates, volumes, color=colors, alpha=0.7, width=0.8)
            ax2.set_ylabel('Volume', fontsize=12)
            ax2.set_xlabel('Data', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Formatação de datas
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
            ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
            ax2.xaxis.set_major_locator(mdates.WeekdayLocator())
            
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # Adicionar texto com métricas
            current_price = data_subset['Close'].iloc[-1]
            predicted_price = future_prices[-1]
            price_change = ((predicted_price / current_price) - 1) * 100
            
            metrics_text = f"""MÉTRICAS ATUAIS:
Preço Atual: ${current_price:.2f}
RSI: {data_subset['RSI'].iloc[-1]:.1f}
MACD: {data_subset['MACD'].iloc[-1]:.3f}
Volatilidade: {data_subset['volatilidade'].iloc[-1]:.3f}

PREVISÃO {days_forecast} DIAS:
Preço Alvo: ${predicted_price:.2f}
Variação: {price_change:+.1f}%
Tendência: {ml_data.get('trend_prediction', 'N/A')}"""
            
            ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Converter para base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return {
                "chart_base64": chart_base64,
                "chart_url": f"data:image/png;base64,{chart_base64}",
                "predictions": future_predictions,
                "analysis_summary": {
                    "current_price": float(current_price),
                    "predicted_price": float(predicted_price),
                    "price_change_percent": float(price_change),
                    "recommendation": analysis.get("recomendacao_final"),
                    "confidence": analysis.get("confianca"),
                    "trend": ml_data.get('trend_prediction')
                },
                "technical_indicators": {
                    "RSI": float(data_subset['RSI'].iloc[-1]),
                    "MACD": float(data_subset['MACD'].iloc[-1]),
                    "BB_position": float(data_subset['BB_position'].iloc[-1]),
                    "Williams_R": float(data_subset['Williams_R'].iloc[-1]),
                    "volatility": float(data_subset['volatilidade'].iloc[-1])
                }
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico para {ticker}: {e}")
            return {"erro": f"Falha ao gerar gráfico: {str(e)}"}
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
def generate_chart(ticker: str, days_forecast: int = 10, chart_type: str = "candlestick") -> Dict[str, Any]:
    """Função de conveniência para gerar gráficos."""
    return analyzer.generate_financial_chart(ticker, days_forecast, chart_type)


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


def generate_chart_data(ticker: str, days_forecast: int = 10) -> Dict[str, Any]:
    """Função global para gerar apenas dados (sem imagem)."""
    analyzer = FinancialAnalyzer()
    return analyzer.generate_chart_data(ticker, days_forecast)

def generate_chart(ticker: str, days_forecast: int = 10, chart_type: str = "candlestick") -> Dict[str, Any]:
    """Função de conveniência para gerar gráficos com imagem."""
    analyzer = FinancialAnalyzer()
    return analyzer.generate_financial_chart(ticker, days_forecast, chart_type)