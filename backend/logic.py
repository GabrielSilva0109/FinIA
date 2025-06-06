import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import re
from transformers import pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import logging
from sentiment_analysis import enhanced_sentiment_analysis

# Configuração básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= INDICADORES TECNICOS ================= #
def compute_technical_indicators(df):
    df['retorno'] = df['Close'].pct_change()
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA15'] = df['Close'].rolling(window=15).mean() # Added 15-day MA
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean() # Renamed to MA30 for consistency
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'], df['MACD_Signal'] = compute_macd(df)
    df['volatilidade'] = df['retorno'].rolling(window=7).std()
    df['Volume'] = df['Volume'].astype(float)
    df = df.dropna()
    
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(df, short=12, long=26, signal=9):
    exp1 = df['Close'].ewm(span=short, adjust=False).mean()
    exp2 = df['Close'].ewm(span=long, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def train_advanced_model(features, target):
    target = target.values.ravel() if isinstance(target, pd.DataFrame) else target.values
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    models = {
        "XGBoost": XGBRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

    best_model, best_score = None, float('inf')

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            if mae < best_score:
                best_score = mae
                best_model = model
        except Exception as e:
            logger.warning(f"Erro ao treinar modelo {name}: {e}")

    return best_model, best_score

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Evitar divisão por zero
    rs = np.where(loss != 0, gain / loss, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_additional_indicators(df):
    # Bollinger Bands
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = compute_bollinger_bands(df['Close'])
    
    # Stochastic Oscillator
    df['%K'], df['%D'] = compute_stochastic_oscillator(df)
    
    # Volume Weighted Average Price
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    
    # ATR (Average True Range)
    df['ATR'] = compute_atr(df)
    
    return df.dropna()

def compute_bollinger_bands(price, window=20, num_std=2):
    rolling_mean = price.rolling(window=window).mean()
    rolling_std = price.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

def compute_stochastic_oscillator(df, k_window=14, d_window=3):
    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()
    df['%K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['%D'] = df['%K'].rolling(window=d_window).mean()
    return df['%K'], df['%D']

def compute_atr(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def sector_correlation_analysis(ticker):
    try:
        info = yf.Ticker(ticker).info
        sector = info.get('sector', None)

        sector_etfs = {
            "Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV",
            "Industrials": "XLI", "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
            "Energy": "XLE", "Utilities": "XLU", "Real Estate": "XLRE",
            "Communication Services": "XLC", "Basic Materials": "XLB"
        }

        if not sector or sector not in sector_etfs:
            return None

        etf = sector_etfs[sector]
        stock_data = yf.download(ticker, period="60d", interval="1d")["Close"].pct_change().dropna()
        etf_data = yf.download(etf, period="60d", interval="1d")["Close"].pct_change().dropna()
        
        min_len = min(len(stock_data), len(etf_data))
        correlation = np.corrcoef(stock_data[-min_len:], etf_data[-min_len:])[0, 1]

        return {
            "sector": sector,
            "correlation_with_sector": round(correlation, 2),
            "sector_trend": "alta" if etf_data.mean() > 0 else "baixa"
        }
    except:
        return None
    
# Função para calcular indicadores técnicos
def time_series_validation(df):
    train_size = int(0.7 * len(df))
    val_size = int(0.2 * len(df))
    train = df.iloc[:train_size]
    val = df.iloc[train_size:train_size+val_size]
    test = df.iloc[train_size+val_size:]
    
    model = XGBRegressor()
    model.fit(train.drop('Close', axis=1), train['Close'])

    val_preds = model.predict(val.drop('Close', axis=1))
    test_preds = model.predict(test.drop('Close', axis=1))

    return {
        "validation_mae": mean_absolute_error(val['Close'], val_preds),
        "test_mae": mean_absolute_error(test['Close'], test_preds),
        "model": model
    }

# ================= ALERTAS INTELIGENTES ================= #
def generate_smart_alerts(analysis_result):
    alerts = []

    rsi = float(analysis_result['RSI'])
    ma7 = float(analysis_result['MA7'])
    ma15 = float(analysis_result['MA15'])
    ma30 = float(analysis_result['MA30'])
    volume = float(analysis_result['Volume'])
    avg_volume = float(analysis_result['avg_volume'])
    tendencia = analysis_result['tendencia']
    sentimento = analysis_result['sentimento']

    if rsi > 70 and tendencia == 'alta':
        alerts.append("Alerta: RSI acima de 70 (sobrecomprado) com tendência de alta - possível correção")
    elif rsi < 30 and tendencia == 'baixa':
        alerts.append("Alerta: RSI abaixo de 30 (sobrevendido) com tendência de baixa - possível reversão")

    if ma7 > ma15 > ma30:
        alerts.append("Tendência de alta forte: MA7 > MA15 > MA30")
    elif ma7 < ma15 < ma30:
        alerts.append("Tendência de baixa forte: MA7 < MA15 < MA30")

    if volume > 2 * avg_volume:
        alerts.append(f"Volume alto: {volume / avg_volume:.1f}x a média")

    if sentimento == 'fortemente positivo' and tendencia == 'baixa':
        alerts.append("Divergência: Sentimento fortemente positivo com tendência de baixa")
    elif sentimento == 'fortemente negativo' and tendencia == 'alta':
        alerts.append("Divergência: Sentimento fortemente negativo com tendência de alta")

    return alerts

# Função para calcular indicadores técnicos
def backtest_strategy(df, initial_capital=10000):
    signals = []
    position = 0.0  # Initialize as float
    capital = float(initial_capital) # Ensure capital is also float
    portfolio_value = [capital]
    
    for i in range(1, len(df)):
        # Estratégia simples: compra quando RSI < 30 e vende quando RSI > 70
        if df['RSI'].iloc[i] < 30 and position == 0.0: # Compare with float
            position = capital / float(df['Close'].iloc[i]) # Explicitly convert to float
            capital = 0.0
            signals.append(('buy', df.index[i], df['Close'].iloc[i]))
        elif df['RSI'].iloc[i] > 70 and position > 0.0: # Compare with float
            capital = position * float(df['Close'].iloc[i]) # Explicitly convert to float
            position = 0.0
            signals.append(('sell', df.index[i], df['Close'].iloc[i]))
        
        # Calcula valor do portfólio
        if position > 0.0: # Compare with float
            portfolio_value.append(position * float(df['Close'].iloc[i]))
        else:
            portfolio_value.append(capital)
    
    returns = (portfolio_value[-1] - initial_capital) / initial_capital * 100
    max_drawdown = compute_drawdown(portfolio_value)
    
    return {
        "final_value": portfolio_value[-1],
        "return_percent": returns,
        "max_drawdown": max_drawdown,
        "signals": signals
    }

# Função para calcular indicadores técnicos
def compute_drawdown(values):
    peak = values[0]
    max_dd = 0
    for value in values:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd * 100

# Function to determine overall trend based on moving averages
def get_ma_trend(ma7, ma15, ma30):
    if ma7 > ma15 and ma15 > ma30:
        return "alta"
    elif ma7 < ma15 and ma15 < ma30:
        return "baixa"
    else:
        return "neutra"

# Function to suggest a strategy
def get_strategy(last_price, forecast, rsi, ma7, ma15, ma30, sentiment):
    strategy = "Observar"

    # Se sentiment for dict, pegar o texto da chave 'final_sentiment'
    if isinstance(sentiment, dict):
        sentiment = sentiment.get("final_sentiment", "neutro")

    # Price vs. Forecast
    if forecast > last_price * 1.02:  # If forecast is significantly higher
        strategy = "Possível Compra (Previsão Otimista)"
    elif forecast < last_price * 0.98:  # If forecast is significantly lower
        strategy = "Possível Venda (Previsão Pessimista)"

    # RSI based strategy
    if rsi < 30:
        strategy = "Sinal de Compra (Sobrecomprado)"
    elif rsi > 70:
        strategy = "Sinal de Venda (Sobrevendido)"
    
    # Moving Average Cross Strategy (simplified Golden/Death Cross)
    if ma7 > ma15 and ma15 > ma30:  # Strong bullish alignment
        if strategy != "Sinal de Compra (Sobrecomprado)":  # Avoid conflicting RSI alert
            strategy = "Sinal de Compra (Tendência de Alta)"
    elif ma7 < ma15 and ma15 < ma30:  # Strong bearish alignment
        if strategy != "Sinal de Venda (Sobrevendido)":  # Avoid conflicting RSI alert
            strategy = "Sinal de Venda (Tendência de Baixa)"
    
    # Incorporate sentiment (trata o texto em lowercase para evitar erros)
    sentiment_lower = sentiment.lower()
    strategy_lower = strategy.lower()

    if "fortemente positivo" in sentiment_lower and "compra" in strategy_lower:
        strategy += " - Confirmado por Sentimento Positivo"
    elif "fortemente negativo" in sentiment_lower and "venda" in strategy_lower:
        strategy += " - Confirmado por Sentimento Negativo"
    
    # Neutral/Uncertainty
    if "observar" in strategy_lower and (sentiment_lower == "neutro" or abs(forecast - last_price) < last_price * 0.01):
        strategy = "Manter Posição / Neutro"

    return strategy

def adjust_forecast_with_indicators(
    forecast, last_price, rsi, macd, ma7, ma15, ma30,
    bb_upper=None, bb_lower=None, vwap=None, atr=None
):
    adjustment = 0

    # Ajuste forte para tendência de baixa
    if ma7 < ma15 < ma30:
        adjustment -= 0.02 * last_price  # Reduz 2% se tendência de baixa forte
        if macd < 0:
            adjustment -= 0.01 * last_price  # Reduz mais 1% se MACD negativo
        if vwap and last_price < vwap:
            adjustment -= 0.005 * last_price  # Reduz mais se abaixo do VWAP

    # Ajuste forte para tendência de alta
    if ma7 > ma15 > ma30:
        adjustment += 0.02 * last_price  # Aumenta 2% se tendência de alta forte
        if macd > 0:
            adjustment += 0.01 * last_price  # Aumenta mais 1% se MACD positivo
        if vwap and last_price > vwap:
            adjustment += 0.005 * last_price  # Aumenta mais se acima do VWAP

    # RSI
    if rsi > 70:
        adjustment -= 0.01 * last_price
    elif rsi < 30:
        adjustment += 0.01 * last_price

    # MACD isolado (caso não tenha entrado acima)
    if macd > 0 and not (ma7 > ma15 > ma30):
        adjustment += 0.003 * last_price
    elif macd < 0 and not (ma7 < ma15 < ma30):
        adjustment -= 0.003 * last_price

    # Bollinger Bands
    if bb_upper and last_price > bb_upper:
        adjustment -= 0.005 * last_price
    if bb_lower and last_price < bb_lower:
        adjustment += 0.005 * last_price

    # ATR (volatilidade)
    if atr and atr > last_price * 0.03:
        adjustment -= 0.002 * last_price

    # Garante que, se tendência for de baixa forte, a previsão não fique acima do preço atual
    if ma7 < ma15 < ma30 and (forecast + adjustment) > last_price:
        adjustment = min(adjustment, last_price - forecast - 0.01)

    # Garante que, se tendência for de alta forte, a previsão não fique abaixo do preço atual
    if ma7 > ma15 > ma30 and (forecast + adjustment) < last_price:
        adjustment = max(adjustment, last_price - forecast + 0.01)

    return forecast + adjustment

def analyze(ticker):
    try:
        # Coleta de dados com auto_adjust explícito
        dados = yf.download(ticker, period="1y", interval="1d", auto_adjust=False)
        
        if dados.empty:
            return {"erro": "Ticker inválido ou sem dados"}
        
        # Indicadores técnicos
        dados = compute_technical_indicators(dados)
        dados = compute_additional_indicators(dados)
        
        # Análise setorial
        sector_analysis = sector_correlation_analysis(ticker)
        
        # Análise de sentimento (com fallback)
        try:
            sentiment = enhanced_sentiment_analysis(ticker)
        except Exception as e:
            logger.error(f"Erro na análise de sentimento para {ticker}: {e}")
            sentiment = "neutro"
        
        # Preparação dos dados para ML
        features = dados.drop(['Close', 'retorno'], axis=1, errors='ignore')
        target = dados[['Close']]
        
        # Modelagem
        if len(features) < 30:
            return {"erro": "Dados insuficientes para análise profunda e treinamento de modelo."}
        
        model, val_mae = train_advanced_model(features, target)
        
        if len(features) >= 30:
            test_preds = model.predict(features.iloc[-30:])
            test_mae = mean_absolute_error(target.iloc[-30:].values.ravel(), test_preds)
        else:
            test_preds = []
            test_mae = np.nan
        
        # Previsão do modelo
        last_features = features.iloc[-1].values.reshape(1, -1)
        forecast = float(model.predict(last_features)[0])
        last_price = float(dados['Close'].iloc[-1])

        # Indicadores para ajuste
        ma7 = float(dados['MA7'].iloc[-1])
        ma15 = float(dados['MA15'].iloc[-1])
        ma30 = float(dados['SMA_30'].iloc[-1])
        rsi = float(dados['RSI'].iloc[-1])
        macd = float(dados['MACD'].iloc[-1])

        # Ajuste da previsão com indicadores técnicos
        forecast_adjusted = adjust_forecast_with_indicators(forecast, last_price, rsi, macd, ma7, ma15, ma30)

        # Tendência
        tendencia = get_ma_trend(ma7, ma15, ma30)

        # Estratégia
        strategy = get_strategy(last_price, forecast_adjusted, rsi, ma7, ma15, ma30, sentiment)
        
        # Geração de alertas
        analysis_data = {
            'RSI': rsi,
            'MA7': ma7,
            'MA15': ma15,
            'MA20': float(dados['MA20'].iloc[-1]),
            'MA30': ma30,
            'Volume': float(dados['Volume'].iloc[-1]),
            'avg_volume': float(dados['Volume'].mean()),
            'tendencia': tendencia,
            'sentimento': sentiment
        }
        alerts = generate_smart_alerts(analysis_data)
        
        # Backtesting
        backtest_results = backtest_strategy(dados)

        return {
            "ticker": ticker,
            "preco_atual": round(last_price, 2),
            "previsao": round(forecast_adjusted, 2),
            "diferenca_preco_previsao": round(forecast_adjusted - last_price, 2),
            "media_movel_7": round(ma7, 2),
            "media_movel_15": round(ma15, 2),
            "media_movel_30": round(ma30, 2),
            "tendencia": tendencia,
            "estrategia": strategy,
            "indicadores": {
                "RSI": round(rsi, 2),
                "MACD": round(macd, 2),
                "Bollinger": {
                    "upper": round(float(dados['BB_upper'].iloc[-1]), 2),
                    "middle": round(float(dados['BB_middle'].iloc[-1]), 2),
                    "lower": round(float(dados['BB_lower'].iloc[-1]), 2)
                },
                "VWAP": round(float(dados['VWAP'].iloc[-1]), 2),
                "ATR": round(float(dados['ATR'].iloc[-1]), 2)
            },
            "model_performance": {
                "validation_mae": round(val_mae, 2),
                "test_mae": round(test_mae, 2) if not np.isnan(test_mae) else "N/A"
            },
            "sector_analysis": sector_analysis,
            "sentiment_analysis": sentiment,
            "alerts": alerts,
            "backtest_results": {
                "return_percent": round(float(backtest_results["return_percent"]), 2),
                "max_drawdown": round(float(backtest_results["max_drawdown"]), 2)
            },
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro ao analisar {ticker}: {str(e)}", exc_info=True)
        return {"erro": str(e)}
    
def analyze_all():
    try:
        # Fallback para pegar as ações do Ibovespa via scraping
        url = "https://finance.yahoo.com/quote/%5EBVSP/components?p=%5EBVSP"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        tickers_lista = []

        for a in soup.select("table tbody tr td:nth-child(1) a"):
            ticker = a.text.strip()
            if not ticker.endswith(".SA"):
                ticker += ".SA"
            tickers_lista.append(ticker)

        tickers_lista = tickers_lista[:30]  # Limita às 30 maiores

        resultados = []

        for ticker in tickers_lista:
            print(f"Analisando {ticker}...")
            resultado = analyze(ticker)

            if "erro" in resultado:
                continue

            preco_atual = resultado["preco_atual"]
            previsao = resultado["previsao"]
            diferenca = previsao - preco_atual
            media_movel_7 = resultado["media_movel_7"]
            media_movel_15 = resultado["media_movel_15"]
            media_movel_30 = resultado["media_movel_30"]

            resultados.append({
                "ticker": ticker,
                "preco_atual": preco_atual,
                "previsao": previsao,
                "media_movel_7": media_movel_7,
                "media_movel_15": media_movel_15,
                "media_movel_30": media_movel_30,
                "diferenca": diferenca,
                "tendencia": resultado["tendencia"],
                "sentimento": resultado["sentiment_analysis"], # Changed to sentiment_analysis
                "estrategia": resultado["estrategia"]
            })

        # Ordena por diferença de previsão - preço atual
        oportunidades = sorted(resultados, key=lambda x: x["diferenca"], reverse=True)[:5]
        quedas = sorted(resultados, key=lambda x: x["diferenca"])[:5]

        return {
            "melhores_oportunidades": oportunidades,
            "maiores_quedas": quedas
        }

    except Exception as e:
        logger.error(f"Erro ao analisar todas as ações: {str(e)}", exc_info=True)
        return {"erro": str(e)}