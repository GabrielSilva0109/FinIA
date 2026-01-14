"""
Advanced Technical Indicators for Financial Analysis
Implementa indicadores avançados sem dependência TA-Lib para compatibilidade Windows
"""

import pandas as pd
import numpy as np

class AdvancedIndicators:
    """Classe para calcular indicadores técnicos avançados sem dependência TA-Lib"""
    
    @staticmethod
    def ichimoku_cloud(high, low, close, period1=9, period2=26, period3=52):
        """Calcula Nuvem de Ichimoku"""
        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(window=period1).max() + low.rolling(window=period1).min()) / 2
        
        # Kijun-sen (Base Line)  
        kijun_sen = (high.rolling(window=period2).max() + low.rolling(window=period2).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(period2)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = ((high.rolling(window=period3).max() + low.rolling(window=period3).min()) / 2).shift(period2)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-period2)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def stochastic_oscillator(high, low, close, k_period=14, d_period=3):
        """Calcula Oscilador Estocástico"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }
    
    @staticmethod
    def adx(high, low, close, period=14):
        """Calcula Average Directional Index (ADX)"""
        # Calculate True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)
        
        # Calculate ATR
        atr = tr.rolling(window=period).mean()
        
        # Calculate DI+ and DI-
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX
        adx_value = dx.rolling(window=period).mean()
        
        return {
            'adx': adx_value,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'atr': atr
        }
    
    @staticmethod
    def atr(high, low, close, period=14):
        """Calcula Average True Range"""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def fibonacci_retracement(high, low):
        """Calcula níveis de retração de Fibonacci"""
        max_price = high.max()
        min_price = low.min()
        diff = max_price - min_price
        
        levels = {
            'level_0': max_price,
            'level_236': max_price - 0.236 * diff,
            'level_382': max_price - 0.382 * diff,
            'level_500': max_price - 0.500 * diff,
            'level_618': max_price - 0.618 * diff,
            'level_100': min_price
        }
        
        return levels
    
    @staticmethod
    def obv(close, volume):
        """Calcula On-Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def vwap(high, low, close, volume):
        """Calcula Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def williams_r(high, low, close, period=14):
        """Calcula Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        return williams_r
    
    @staticmethod
    def cci(high, low, close, period=20):
        """Calcula Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        
        return cci
    
    @staticmethod
    def roc(close, period=12):
        """Calcula Rate of Change"""
        roc = ((close - close.shift(period)) / close.shift(period)) * 100
        return roc
    
    def calculate_all_indicators(self, data):
        """Calcula todos os indicadores avançados"""
        high = data['high']
        low = data['low'] 
        close = data['close']
        volume = data['volume']
        
        indicators = {}
        
        try:
            # Ichimoku Cloud
            ichimoku = self.ichimoku_cloud(high, low, close)
            indicators.update({f'ichimoku_{k}': v for k, v in ichimoku.items()})
            
            # Stochastic Oscillator
            stoch = self.stochastic_oscillator(high, low, close)
            indicators.update({f'stoch_{k}': v for k, v in stoch.items()})
            
            # ADX
            adx_data = self.adx(high, low, close)
            indicators.update({f'adx_{k}': v for k, v in adx_data.items()})
            
            # ATR
            indicators['atr'] = self.atr(high, low, close)
            
            # Fibonacci Retracement
            fib = self.fibonacci_retracement(high, low)
            indicators.update({f'fib_{k}': v for k, v in fib.items()})
            
            # Volume indicators
            indicators['obv'] = self.obv(close, volume)
            indicators['vwap'] = self.vwap(high, low, close, volume)
            
            # Williams %R
            indicators['williams_r'] = self.williams_r(high, low, close)
            
            # CCI
            indicators['cci'] = self.cci(high, low, close)
            
            # ROC
            indicators['roc'] = self.roc(close)
            
        except Exception as e:
            print(f"Erro ao calcular indicadores: {e}")
            
        return indicators