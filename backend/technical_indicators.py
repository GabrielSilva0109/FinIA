"""
Módulo de indicadores técnicos para análise financeira.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Union


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calcula o Relative Strength Index (RSI).
    
    Args:
        series: Série de preços
        period: Período para cálculo (padrão: 14)
    
    Returns:
        Série com valores de RSI
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Evitar divisão por zero
    rs = np.where(loss != 0, gain / loss, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=series.index)


def compute_macd(df: pd.DataFrame, short: int = 12, long: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """
    Calcula o MACD (Moving Average Convergence Divergence).
    
    Args:
        df: DataFrame com coluna 'Close'
        short: Período EMA rápido (padrão: 12)
        long: Período EMA lento (padrão: 26)
        signal: Período da linha de sinal (padrão: 9)
    
    Returns:
        Tupla com (MACD, Sinal)
    """
    exp1 = df['Close'].ewm(span=short, adjust=False).mean()
    exp2 = df['Close'].ewm(span=long, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def compute_bollinger_bands(price: pd.Series, window: int = 20, num_std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcula as Bandas de Bollinger.
    
    Args:
        price: Série de preços
        window: Janela para média móvel (padrão: 20)
        num_std: Número de desvios padrão (padrão: 2)
    
    Returns:
        Tupla com (banda_superior, banda_média, banda_inferior)
    """
    rolling_mean = price.rolling(window=window).mean()
    rolling_std = price.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band


def compute_stochastic_oscillator(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Calcula o Oscilador Estocástico.
    
    Args:
        df: DataFrame com colunas 'High', 'Low', 'Close'
        k_window: Janela para %K (padrão: 14)
        d_window: Janela para %D (padrão: 3)
    
    Returns:
        Tupla com (%K, %D)
    """
    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()
    k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calcula o Average True Range (ATR).
    
    Args:
        df: DataFrame com colunas 'High', 'Low', 'Close'
        window: Período para média (padrão: 14)
    
    Returns:
        Série com valores de ATR
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calcula o Volume Weighted Average Price (VWAP).
    
    Args:
        df: DataFrame com colunas 'High', 'Low', 'Close', 'Volume'
    
    Returns:
        Série com valores de VWAP
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    return (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos os indicadores técnicos principais.
    
    Args:
        df: DataFrame com dados OHLCV
    
    Returns:
        DataFrame com indicadores técnicos adicionados
    """
    df = df.copy()
    
    # Retornos e médias móveis
    df['retorno'] = df['Close'].pct_change()
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA15'] = df['Close'].rolling(window=15).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    
    # Indicadores principais
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'], df['MACD_Signal'] = compute_macd(df)
    df['volatilidade'] = df['retorno'].rolling(window=7).std()
    
    # Bandas de Bollinger
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = compute_bollinger_bands(df['Close'])
    
    # Oscilador Estocástico
    df['%K'], df['%D'] = compute_stochastic_oscillator(df)
    
    # ATR e VWAP
    df['ATR'] = compute_atr(df)
    df['VWAP'] = compute_vwap(df)
    
    # Garantir que Volume seja float
    df['Volume'] = df['Volume'].astype(float)
    
    return df.dropna()


def compute_additional_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula indicadores técnicos adicionais.
    
    Args:
        df: DataFrame com dados básicos
    
    Returns:
        DataFrame com indicadores adicionais
    """
    df = compute_technical_indicators(df)
    
    # Indicadores adicionais podem ser adicionados aqui
    # Por exemplo: Williams %R, CCI, etc.
    
    return df