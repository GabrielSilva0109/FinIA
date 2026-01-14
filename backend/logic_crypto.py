"""
Módulo de análise de criptomoedas melhorado.
"""
import ccxt
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
from functools import lru_cache

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class CryptoSignal:
    """Classe para representar sinais de trading."""
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0-100
    reasons: List[str]
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    

class CryptoAnalyzer:
    """Analisador de criptomoedas com múltiplas exchanges."""
    
    def __init__(self, exchange_id: str = 'binance'):
        self.exchange_id = exchange_id
        self.exchange = None
        self.cache = {}
        self.cache_ttl = 300  # 5 minutos
        
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Inicializar conexão com exchange."""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'apiKey': settings.EXCHANGE_API_KEY,
                'secret': settings.EXCHANGE_SECRET_KEY,
                'timeout': 30000,
                'enableRateLimit': True,
                'sandbox': False,
            })
            logger.info(f"Exchange {self.exchange_id} inicializada")
        except Exception as e:
            logger.error(f"Erro ao inicializar exchange: {e}")
            self.exchange = None
    
    @lru_cache(maxsize=100)
    def get_crypto_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Buscar dados OHLCV de criptomoeda."""
        try:
            if not self.exchange:
                return pd.DataFrame()
            
            # Tentar buscar dados da exchange
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao buscar dados para {symbol}: {e}")
            return pd.DataFrame()
    
    def analyze_crypto_pair(self, symbol: str) -> Dict[str, Any]:
        """Análise completa de par de criptomoeda."""
        try:
            # Buscar dados históricos
            df = self.get_crypto_data(symbol)
            
            if df.empty:
                return {
                    "symbol": symbol,
                    "status": "error",
                    "message": "Dados não disponíveis"
                }
            
            # Análise técnica
            technical_analysis = self._technical_analysis(df)
            
            # Análise de volatilidade
            volatility_analysis = self._volatility_analysis(df)
            
            # Gerar sinal de trading
            signal = self._generate_signal(technical_analysis, volatility_analysis)
            
            return {
                "symbol": symbol,
                "status": "success",
                "price": float(df['close'].iloc[-1]),
                "change_24h": self._calculate_24h_change(df),
                "technical_analysis": technical_analysis,
                "volatility": volatility_analysis,
                "signal": {
                    "action": signal.action,
                    "confidence": signal.confidence,
                    "reasons": signal.reasons,
                    "price_target": signal.price_target,
                    "stop_loss": signal.stop_loss
                },
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de {symbol}: {e}")
            return {
                "symbol": symbol,
                "status": "error",
                "message": str(e)
            }
    
    def _technical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcular indicadores técnicos."""
        try:
            if len(df) < 14:
                return {"error": "Dados insuficientes"}
            
            # RSI
            rsi = ta.momentum.RSIIndicator(df['close'], window=14)
            current_rsi = rsi.rsi().iloc[-1]
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            macd_line = macd.macd().iloc[-1]
            signal_line = macd.macd_signal().iloc[-1]
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20)
            bb_upper = bb.bollinger_hband().iloc[-1]
            bb_lower = bb.bollinger_lband().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            return {
                "rsi": float(current_rsi),
                "macd": {
                    "macd_line": float(macd_line),
                    "signal_line": float(signal_line),
                    "histogram": float(macd_line - signal_line)
                },
                "bollinger_bands": {
                    "upper": float(bb_upper),
                    "lower": float(bb_lower),
                    "position": "above" if current_price > bb_upper else "below" if current_price < bb_lower else "middle"
                },
                "moving_averages": {
                    "ma_20": float(df['close'].rolling(20).mean().iloc[-1]),
                    "ma_50": float(df['close'].rolling(50).mean().iloc[-1]) if len(df) >= 50 else None
                }
            }
            
        except Exception as e:
            logger.error(f"Erro no cálculo de indicadores técnicos: {e}")
            return {"error": str(e)}
    
    def _volatility_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise de volatilidade."""
        try:
            # Volatilidade histórica
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(24 * 365)  # Anualizada para crypto (24/7)
            
            # ATR (Average True Range)
            atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
            current_atr = atr.average_true_range().iloc[-1]
            
            return {
                "historical_volatility": float(volatility),
                "atr": float(current_atr),
                "volatility_rank": self._classify_volatility(volatility),
                "price_range_24h": {
                    "high": float(df['high'].tail(24).max()),
                    "low": float(df['low'].tail(24).min())
                }
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de volatilidade: {e}")
            return {"error": str(e)}
    
    def _classify_volatility(self, volatility: float) -> str:
        """Classificar nível de volatilidade."""
        if volatility < 0.5:
            return "baixa"
        elif volatility < 1.0:
            return "média"
        elif volatility < 2.0:
            return "alta"
        else:
            return "extrema"
    
    def _calculate_24h_change(self, df: pd.DataFrame) -> float:
        """Calcular mudança percentual em 24h."""
        try:
            if len(df) < 24:
                return 0.0
            
            current_price = df['close'].iloc[-1]
            price_24h_ago = df['close'].iloc[-24]
            
            return float((current_price - price_24h_ago) / price_24h_ago * 100)
            
        except Exception as e:
            logger.error(f"Erro no cálculo de mudança 24h: {e}")
            return 0.0
    
    def _generate_signal(self, technical: Dict[str, Any], volatility: Dict[str, Any]) -> CryptoSignal:
        """Gerar sinal de trading baseado nas análises."""
        try:
            reasons = []
            score = 0
            
            # Análise RSI
            rsi = technical.get('rsi', 50)
            if rsi < 30:
                score += 2
                reasons.append("RSI oversold (< 30)")
            elif rsi > 70:
                score -= 2
                reasons.append("RSI overbought (> 70)")
            
            # Análise MACD
            macd_data = technical.get('macd', {})
            if macd_data.get('histogram', 0) > 0:
                score += 1
                reasons.append("MACD bullish")
            else:
                score -= 1
                reasons.append("MACD bearish")
            
            # Determinar ação
            if score >= 2:
                action = "buy"
                confidence = min(80, 50 + score * 10)
            elif score <= -2:
                action = "sell"
                confidence = min(80, 50 + abs(score) * 10)
            else:
                action = "hold"
                confidence = 30
            
            return CryptoSignal(
                action=action,
                confidence=confidence,
                reasons=reasons
            )
            
        except Exception as e:
            logger.error(f"Erro na geração de sinal: {e}")
            return CryptoSignal(
                action="hold",
                confidence=0,
                reasons=[f"Erro: {str(e)}"]
            )


# Instância global
crypto_analyzer = CryptoAnalyzer()


# Função de compatibilidade
def analyze_crypto_pair(symbol: str) -> Dict[str, Any]:
    """Função de compatibilidade para análise de crypto."""
    return crypto_analyzer.analyze_crypto_pair(symbol)