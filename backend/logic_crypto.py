import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import ta
import requests
import numpy as np

def fetch_binance_ohlcv(symbol='BTC/USDT', timeframe='1d', limit=365):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def fetch_coingecko_info(coin_id='bitcoin'):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}'
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        info = {
            'name': data['name'],
            'symbol': data['symbol'],
            'market_cap': data['market_data']['market_cap']['usd'],
            'current_price': data['market_data']['current_price']['usd'],
            'total_volume': data['market_data']['total_volume']['usd'],
            'description': data['description']['en'][:500]  # resumo
        }
        return info
    return {}

def add_technical_indicators(df):
    # Médias móveis
    df['SMA20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['EMA20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['SMA50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['EMA50'] = ta.trend.ema_indicator(df['close'], window=50)
    # Indicadores de momentum
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['StochRSI'] = ta.momentum.stochrsi(df['close'])
    df['MACD'] = ta.trend.macd_diff(df['close'])
    df['MOM'] = ta.momentum.roc(df['close'], window=10)
    df['ROC'] = ta.momentum.roc(df['close'], window=12)
    df['UO'] = ta.momentum.ultimate_oscillator(df['high'], df['low'], df['close'])
    # Volatilidade
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    df['Bollinger_h'] = ta.volatility.bollinger_hband(df['close'])
    df['Bollinger_l'] = ta.volatility.bollinger_lband(df['close'])
    # Tendência
    df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'])
    # Volume
    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    # Outros
    df['CCI'] = ta.trend.cci(df['high'], df['low'], df['close'])
    df['WILLR'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
    # VWAP (simples)
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    return df

def detect_candle_patterns(df):
    # Martelo
    df['Hammer'] = ((df['high'] - df['low'] > 3 * (df['open'] - df['close'])) &
                    ((df['close'] - df['low']) / (.001 + df['high'] - df['low']) > 0.6) &
                    ((df['open'] - df['low']) / (.001 + df['high'] - df['low']) > 0.6))
    # Engolfo de alta
    df['Bullish_Engulfing'] = (df['close'].shift(1) < df['open'].shift(1)) & \
                              (df['close'] > df['open']) & \
                              (df['close'] > df['open'].shift(1)) & \
                              (df['open'] < df['close'].shift(1))
    # Doji
    df['Doji'] = (abs(df['close'] - df['open']) <= (df['high'] - df['low']) * 0.1)
    return df

def plot_all(df, symbol, save=False):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Candles'
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='blue'), name='SMA20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='orange'), name='EMA20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='purple'), name='SMA50'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='brown'), name='EMA50'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_h'], line=dict(color='green', dash='dot'), name='Bollinger High'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_l'], line=dict(color='red', dash='dot'), name='Bollinger Low'))
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='black', dash='dash'), name='VWAP'))
    fig.update_layout(title=f'Análise Gráfica de {symbol}', xaxis_title='Data', yaxis_title='Preço')
    if save:
        fig.write_html(f'{symbol}_plotly.html')
    fig.show()



    plt.figure(figsize=(16, 12))
    plt.subplot(4, 1, 1)
    plt.plot(df['close'], label='Close')
    plt.plot(df['SMA20'], label='SMA20')
    plt.plot(df['EMA20'], label='EMA20')
    plt.plot(df['SMA50'], label='SMA50')
    plt.plot(df['EMA50'], label='EMA50')
    plt.legend()
    plt.title('Médias Móveis')

    plt.subplot(4, 1, 2)
    plt.plot(df['RSI'], label='RSI')
    plt.plot(df['StochRSI'], label='StochRSI')
    plt.axhline(70, color='red', linestyle='--')
    plt.axhline(30, color='green', linestyle='--')
    plt.legend()
    plt.title('RSI e StochRSI')

    plt.subplot(4, 1, 3)
    plt.plot(df['MACD'], label='MACD')
    plt.plot(df['ADX'], label='ADX')
    plt.legend()
    plt.title('MACD e ADX')

    plt.subplot(4, 1, 4)
    plt.plot(df['ATR'], label='ATR')
    plt.plot(df['OBV'], label='OBV')
    plt.legend()
    plt.title('ATR e OBV')
    plt.tight_layout()
    if save:
        plt.savefig(f'{symbol}_matplotlib.png')
    plt.show()

def print_statistical_summary(df):
    print("\nResumo Estatístico dos Preços:")
    print(df['close'].describe())
    print("\nRetorno diário médio: {:.2f}%".format(100 * df['close'].pct_change().mean()))
    print("Volatilidade (desvio padrão dos retornos): {:.2f}%".format(100 * df['close'].pct_change().std()))

def print_pattern_signals(df):
    print("\nÚltimos padrões de candle identificados:")
    last = df.iloc[-10:]
    for idx, row in last.iterrows():
        patterns = []
        if row['Hammer']:
            patterns.append('Martelo')
        if row['Bullish_Engulfing']:
            patterns.append('Engolfo de Alta')
        if row['Doji']:
            patterns.append('Doji')
        if patterns:
            print(f"{idx.date()}: {', '.join(patterns)}")

def fetch_sentiment_analysis():
    """
    Busca o índice Fear & Greed do mercado cripto.
    Retorna um dicionário com o valor e classificação.
    """
    fg_url = 'https://api.alternative.me/fng/?limit=1'
    try:
        r = requests.get(fg_url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if 'data' in data and data['data']:
            value = data['data'][0]['value']
            classification = data['data'][0]['value_classification']
            return {
                'fear_greed_index': value,
                'classification': classification
            }
    except Exception as e:
        print(f"Erro ao buscar sentimento: {e}")
    return {}

def print_sentiment(sentiment):
    print("\nSentimento de Mercado (Fear & Greed Index):")
    if sentiment:
        print(f"Índice: {sentiment['fear_greed_index']} - {sentiment['classification']}")
    else:
        print("Não foi possível obter o sentimento de mercado.")

def main():
    symbol = 'BTC/USDT'
    coin_id = 'bitcoin'
    print(f'Buscando informações de {symbol}...')
    info = fetch_coingecko_info(coin_id)
    for k, v in info.items():
        print(f'{k}: {v}\n')

    print('Buscando dados históricos...')
    df = fetch_binance_ohlcv(symbol)
    df = add_technical_indicators(df)
    df = detect_candle_patterns(df)
    print(df.tail())

    print_statistical_summary(df)
    print_pattern_signals(df)

    print('Gerando gráficos...')
    plot_all(df, symbol, save=True)

if __name__ == '__main__':
    main()