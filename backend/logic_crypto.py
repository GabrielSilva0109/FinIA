import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import ta
import requests

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
    df['SMA20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['EMA20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['close'])
    df['Bollinger_h'] = ta.volatility.bollinger_hband(df['close'])
    df['Bollinger_l'] = ta.volatility.bollinger_lband(df['close'])
    df['StochRSI'] = ta.momentum.stochrsi(df['close'])
    return df

def plot_all(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Candles'
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='blue'), name='SMA20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='orange'), name='EMA20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_h'], line=dict(color='green', dash='dot'), name='Bollinger High'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_l'], line=dict(color='red', dash='dot'), name='Bollinger Low'))
    fig.update_layout(title=f'Análise Gráfica de {symbol}', xaxis_title='Data', yaxis_title='Preço')
    fig.show()

    plt.figure(figsize=(14, 7))
    plt.subplot(3, 1, 1)
    plt.plot(df['close'], label='Close')
    plt.plot(df['SMA20'], label='SMA20')
    plt.plot(df['EMA20'], label='EMA20')
    plt.legend()
    plt.title('Médias Móveis')

    plt.subplot(3, 1, 2)
    plt.plot(df['RSI'], label='RSI')
    plt.axhline(70, color='red', linestyle='--')
    plt.axhline(30, color='green', linestyle='--')
    plt.legend()
    plt.title('RSI')

    plt.subplot(3, 1, 3)
    plt.plot(df['MACD'], label='MACD')
    plt.legend()
    plt.title('MACD')
    plt.tight_layout()
    plt.show()

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
    print(df.tail())

    print('Gerando gráficos...')
    plot_all(df, symbol)

if __name__ == '__main__':
    main()