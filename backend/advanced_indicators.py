"""
Advanced Technical Indicators for Financial Analysis
Implementa indicadores avançados: Ichimoku, Stochastic, ADX, ATR, etc.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Tuple, Optional

class AdvancedIndicators:
    """Classe com indicadores técnicos avançados"""
    
    @staticmethod
    def ichimoku_cloud(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict:
        """
        Ichimoku Kinko Hyo - Sistema completo de análise
        """
        # Tenkan-sen (linha de conversão) - 9 períodos
        tenkan_high = high.rolling(window=9).max()
        tenkan_low = low.rolling(window=9).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (linha base) - 26 períodos  
        kijun_high = high.rolling(window=26).max()
        kijun_low = low.rolling(window=26).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (nuvem líder A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (nuvem líder B) - 52 períodos
        senkou_high = high.rolling(window=52).max()
        senkou_low = low.rolling(window=52).min()
        senkou_span_b = ((senkou_high + senkou_low) / 2).shift(26)
        
        # Chikou Span (linha de atraso)
        chikou_span = close.shift(-26)
        
        # Sinais
        signal = "NEUTRO"
        if close.iloc[-1] > senkou_span_a.iloc[-1] and close.iloc[-1] > senkou_span_b.iloc[-1]:
            if tenkan_sen.iloc[-1] > kijun_sen.iloc[-1]:
                signal = "COMPRA_FORTE"
            else:
                signal = "COMPRA"
        elif close.iloc[-1] < senkou_span_a.iloc[-1] and close.iloc[-1] < senkou_span_b.iloc[-1]:
            if tenkan_sen.iloc[-1] < kijun_sen.iloc[-1]:
                signal = "VENDA_FORTE"
            else:
                signal = "VENDA"
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span,
            'signal': signal,
            'value': f"{tenkan_sen.iloc[-1]:.2f}"
        }
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_period: int = 14, d_period: int = 3) -> Dict:
        """
        Estocástico - Momentum oscillator
        """
        # %K
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        
        # %D (média móvel de %K)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        # Sinais
        k_current = k_percent.iloc[-1]
        d_current = d_percent.iloc[-1]
        
        if k_current < 20 and d_current < 20:
            signal = "SOBREVENDA"
        elif k_current > 80 and d_current > 80:
            signal = "SOBRECOMPRA"
        elif k_current > d_current and k_current > 50:
            signal = "COMPRA"
        elif k_current < d_current and k_current < 50:
            signal = "VENDA"
        else:
            signal = "NEUTRO"
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent,
            'signal': signal,
            'value': f"K:{k_current:.1f} D:{d_current:.1f}"
        }
    
    @staticmethod
    def adx_indicator(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict:
        """
        ADX - Average Directional Index (força da tendência)
        """
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = np.where((high - high.shift(1)) > (low.shift(1) - low),
                          np.maximum(high - high.shift(1), 0), 0)
        minus_dm = np.where((low.shift(1) - low) > (high - high.shift(1)),
                           np.maximum(low.shift(1) - low, 0), 0)
        
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)
        
        # Smoothed values
        tr_smooth = tr.rolling(window=period).mean()
        plus_dm_smooth = plus_dm.rolling(window=period).mean()
        minus_dm_smooth = minus_dm.rolling(window=period).mean()
        
        # Directional Indicators
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        # Interpretação
        adx_current = adx.iloc[-1]
        plus_di_current = plus_di.iloc[-1]
        minus_di_current = minus_di.iloc[-1]
        
        if adx_current > 25:
            if plus_di_current > minus_di_current:
                signal = "TENDENCIA_ALTA_FORTE"
            else:
                signal = "TENDENCIA_BAIXA_FORTE"
        elif adx_current > 15:
            signal = "TENDENCIA_MODERADA"
        else:
            signal = "SEM_TENDENCIA"
        
        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'signal': signal,
            'value': f"ADX:{adx_current:.1f}"
        }
    
    @staticmethod
    def atr_indicator(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict:
        """
        ATR - Average True Range (volatilidade)
        """
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR
        atr = tr.rolling(window=period).mean()
        
        # Interpretação
        atr_current = atr.iloc[-1]
        atr_avg = atr.tail(50).mean()
        
        if atr_current > atr_avg * 1.5:
            signal = "VOLATILIDADE_ALTA"
        elif atr_current < atr_avg * 0.7:
            signal = "VOLATILIDADE_BAIXA"
        else:
            signal = "VOLATILIDADE_NORMAL"
        
        return {
            'atr': atr,
            'signal': signal,
            'value': f"{atr_current:.2f}"
        }
    
    @staticmethod
    def fibonacci_retracement(high: pd.Series, low: pd.Series, period: int = 50) -> Dict:
        """
        Fibonacci Retracement levels
        """
        # Pegar high e low do período
        high_price = high.tail(period).max()
        low_price = low.tail(period).min()
        
        # Calcular níveis
        diff = high_price - low_price
        
        levels = {
            'level_0': high_price,
            'level_236': high_price - (diff * 0.236),
            'level_382': high_price - (diff * 0.382),
            'level_500': high_price - (diff * 0.500),
            'level_618': high_price - (diff * 0.618),
            'level_786': high_price - (diff * 0.786),
            'level_100': low_price
        }
        
        # Determinar zona atual
        current_price = high.iloc[-1] if hasattr(high.iloc[-1], 'values') else high.iloc[-1]
        
        signal = "NEUTRO"
        for level_name, level_value in levels.items():
            if abs(current_price - level_value) / current_price < 0.02:  # 2% de tolerância
                signal = f"PRÓXIMO_FIB_{level_name.split('_')[1]}"
                break
        
        return {
            'levels': levels,
            'signal': signal,
            'value': f"Zona: {current_price:.2f}"
        }
    
    @staticmethod
    def volume_indicators(volume: pd.Series, close: pd.Series, period: int = 20) -> Dict:
        """
        Indicadores baseados em volume
        """
        # Volume Price Trend
        vpt = (close.pct_change() * volume).cumsum()
        
        # On Balance Volume
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        
        # Volume Rate of Change
        volume_roc = volume.pct_change(period) * 100
        
        # Chaikin Money Flow
        high = close * 1.02  # Estimativa se não tiver high
        low = close * 0.98   # Estimativa se não tiver low
        
        mf_multiplier = ((close - low) - (high - close)) / (high - low)
        mf_volume = mf_multiplier * volume
        cmf = mf_volume.rolling(period).sum() / volume.rolling(period).sum()
        
        # Sinais
        obv_trend = "ALTA" if obv.diff(5).iloc[-1] > 0 else "BAIXA"
        volume_trend = "ALTA" if volume_roc.iloc[-1] > 10 else "BAIXA" if volume_roc.iloc[-1] < -10 else "NORMAL"
        
        return {
            'vpt': vpt,
            'obv': obv,
            'volume_roc': volume_roc,
            'cmf': cmf,
            'signal': f"Vol:{volume_trend}_OBV:{obv_trend}",
            'value': f"CMF:{cmf.iloc[-1]:.3f}"
        }
    
    @staticmethod
    def momentum_indicators(close: pd.Series) -> Dict:
        """
        Indicadores de Momentum
        """
        # Rate of Change
        roc_5 = close.pct_change(5) * 100
        roc_10 = close.pct_change(10) * 100
        
        # Money Flow Index (simplificado)
        high = close * 1.02
        low = close * 0.98
        volume = pd.Series(np.random.randint(1000000, 10000000, len(close)), index=close.index)
        
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price.diff() > 0, 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price.diff() < 0, 0).rolling(14).sum()
        
        mfi = 100 - (100 / (1 + (positive_flow / negative_flow)))
        
        # Williams %R
        high_14 = high.rolling(14).max()
        low_14 = low.rolling(14).min()
        williams_r = -100 * (high_14 - close) / (high_14 - low_14)
        
        # Sinais
        signal = "NEUTRO"
        if mfi.iloc[-1] > 80:
            signal = "SOBRECOMPRA_MFI"
        elif mfi.iloc[-1] < 20:
            signal = "SOBREVENDA_MFI"
        elif williams_r.iloc[-1] > -20:
            signal = "SOBRECOMPRA_WR"
        elif williams_r.iloc[-1] < -80:
            signal = "SOBREVENDA_WR"
        
        return {
            'roc_5': roc_5,
            'roc_10': roc_10,
            'mfi': mfi,
            'williams_r': williams_r,
            'signal': signal,
            'value': f"MFI:{mfi.iloc[-1]:.1f} WR:{williams_r.iloc[-1]:.1f}"
        }

def calculate_all_advanced_indicators(data: pd.DataFrame) -> Dict:
    """
    Calcula todos os indicadores avançados
    """
    indicators = AdvancedIndicators()
    
    high = data['High'] if 'High' in data.columns else data['Close'] * 1.02
    low = data['Low'] if 'Low' in data.columns else data['Close'] * 0.98
    close = data['Close']
    volume = data['Volume'] if 'Volume' in data.columns else pd.Series(np.random.randint(1000000, 10000000, len(data)), index=data.index)
    
    results = {
        'ichimoku': indicators.ichimoku_cloud(high, low, close),
        'stochastic': indicators.stochastic_oscillator(high, low, close),
        'adx': indicators.adx_indicator(high, low, close),
        'atr': indicators.atr_indicator(high, low, close),
        'fibonacci': indicators.fibonacci_retracement(high, low),
        'volume': indicators.volume_indicators(volume, close),
        'momentum': indicators.momentum_indicators(close)
    }
    
    return results