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

def compute_macd(df, short=12, long=26, signal=9):
    exp1 = df['Close'].ewm(span=short, adjust=False).mean()
    exp2 = df['Close'].ewm(span=long, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def train_advanced_model(features, target):
    # Validação básica dos dados
    if features is None or target is None:
        raise ValueError("Parâmetros 'features' e 'target' não podem ser None.")
    
    if features.empty or len(target) == 0:
        raise ValueError("Parâmetros 'features' e 'target' não podem estar vazios.")

    # Garantir que o target seja um array 1D
    target = target.values.ravel() if isinstance(target, pd.DataFrame) else target.values

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    models = {
        "XGBoost": XGBRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

    best_model = None
    best_score = float('inf')

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            print(f"{name}: MAE = {mae:.4f}")
            if mae < best_score:
                best_score = mae
                best_model = model
        except Exception as e:
            print(f"Erro ao treinar o modelo {name}: {e}")

    print(f"\nMelhor modelo selecionado com MAE = {best_score:.4f}")

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
        stock_info = yf.Ticker(ticker).info
        sector = stock_info.get('sector', None)
        
        if not sector:
            return None
            
        sector_etfs = {
            "Technology": "XLK",
            "Financial Services": "XLF",
            "Healthcare": "XLV",
            "Industrials": "XLI",
            "Consumer Cyclical": "XLY",
            "Consumer Defensive": "XLP",
            "Energy": "XLE",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Communication Services": "XLC",
            "Basic Materials": "XLB"
        }
        
        if sector not in sector_etfs:
            return None
            
        etf_ticker = sector_etfs[sector]
        stock_data = yf.download(ticker, period="60d", interval="1d", auto_adjust=False)['Close'].pct_change().dropna()
        etf_data = yf.download(etf_ticker, period="60d", interval="1d", auto_adjust=False)['Close'].pct_change().dropna()
        
        # Garantir que temos dados suficientes
        if len(stock_data) < 2 or len(etf_data) < 2:
            return None
            
        # Ajustar tamanho dos arrays
        min_len = min(len(stock_data), len(etf_data))
        stock_data = stock_data[-min_len:]
        etf_data = etf_data[-min_len:]
        
        # Calcular correlação com verificação
        try:
            correlation = np.corrcoef(stock_data, etf_data)[0,1]
            if np.isnan(correlation):
                return None
        except:
            return None
        
        return {
            "sector": sector,
            "correlation_with_sector": round(correlation, 2),
            "sector_trend": "alta" if etf_data.mean() > 0 else "baixa"
        }
    except:
        return None

# Função para calcular indicadores técnicos
def time_series_validation(df):
    # Divide os dados em treino (70%), validação (20%) e teste (10%)
    train_size = int(0.7 * len(df))
    val_size = int(0.2 * len(df))
    
    train = df.iloc[:train_size]
    val = df.iloc[train_size:train_size+val_size]
    test = df.iloc[train_size+val_size:]
    
    # Treina o modelo
    model = XGBRegressor()
    model.fit(train.drop('Close', axis=1), train['Close'])
    
    # Avalia no conjunto de validação
    val_preds = model.predict(val.drop('Close', axis=1))
    val_mae = mean_absolute_error(val['Close'], val_preds)
    
    # Avalia no conjunto de teste (últimos dados)
    test_preds = model.predict(test.drop('Close', axis=1))
    test_mae = mean_absolute_error(test['Close'], test_preds)
    
    return {
        "validation_mae": val_mae,
        "test_mae": test_mae,
        "model": model
    }

def generate_smart_alerts(analysis_result):
    alerts = []
    
    # Certifique-se de que estamos comparando valores únicos, não Series
    rsi = float(analysis_result['RSI'])
    ma7 = float(analysis_result['MA7'])
    ma20 = float(analysis_result['MA20'])
    ma30 = float(analysis_result['MA30']) # Changed from MA50 to MA30
    volume = float(analysis_result['Volume'])
    avg_volume = float(analysis_result['avg_volume'])
    tendencia = analysis_result['tendencia']
    sentimento = analysis_result['sentimento']
    
    # Alerta de divergência RSI-preço
    if rsi > 70 and tendencia == 'alta':
        alerts.append("Alerta: RSI acima de 70 (sobrecomprado) com tendência de alta - possível correção")
    elif rsi < 30 and tendencia == 'baixa':
        alerts.append("Alerta: RSI abaixo de 30 (sobrevendido) com tendência de baixa - possível reversão")
    
    # Alerta de cruzamento de médias móveis
    # Re-evaluating the MA alerts based on 7, 15, 30
    ma15 = float(analysis_result['MA15'])
    if ma7 > ma15 and ma15 > ma30:
        alerts.append("Tendência de alta forte: MA7 > MA15 > MA30")
    elif ma7 < ma15 and ma15 < ma30:
        alerts.append("Tendência de baixa forte: MA7 < MA15 < MA30")
    elif ma7 > ma30 and ma15 < ma30:
        alerts.append("Atenção: MA7 acima de MA30, mas MA15 abaixo - tendência incerta ou reversão")

    
    # Alerta de volume
    if volume > 2 * avg_volume:
        alerts.append(f"Volume alto: {volume/avg_volume:.1f}x a média")
    
    # Alerta de sentimento
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
        features = dados.drop(['Close', 'retorno'], axis=1, errors='ignore') # 'retorno_diario' not defined
        target = dados[['Close']]  # Mantém como DataFrame para usar .iloc
        
        # Modelagem
        # Check if there are enough data points for training after dropping NaNs
        if len(features) < 30: # Arbitrary minimum, adjust as needed
            return {"erro": "Dados insuficientes para análise profunda e treinamento de modelo."}
        
        model, val_mae = train_advanced_model(features, target)
        
        # Ensure there are enough data points for test_preds
        if len(features) >= 30:
            test_preds = model.predict(features.iloc[-30:])
            test_mae = mean_absolute_error(target.iloc[-30:].values.ravel(), test_preds)
        else:
            test_preds = []
            test_mae = np.nan # Not applicable if not enough data
        
        # Previsão - usando .iloc[0] para evitar FutureWarning
        last_features = features.iloc[-1].values.reshape(1, -1)
        forecast = float(model.predict(last_features)[0])
        last_price = float(dados['Close'].iloc[-1])

        # Get moving averages for current data
        ma7 = float(dados['MA7'].iloc[-1])
        ma15 = float(dados['MA15'].iloc[-1])
        ma30 = float(dados['SMA_30'].iloc[-1]) # Use SMA_30 as MA30

        # Determine trend
        tendencia = get_ma_trend(ma7, ma15, ma30)

        # Determine strategy
        strategy = get_strategy(last_price, forecast, float(dados['RSI'].iloc[-1]), ma7, ma15, ma30, sentiment)
        
        # Geração de alertas
        analysis_data = {
            'RSI': float(dados['RSI'].iloc[-1]),
            'MA7': ma7,
            'MA15': ma15, # Pass MA15 to alerts
            'MA20': float(dados['MA20'].iloc[-1]),
            'MA30': ma30, # Pass MA30 to alerts
            'Volume': float(dados['Volume'].iloc[-1]),
            'avg_volume': float(dados['Volume'].mean()),
            'tendencia': tendencia, # Use the new trend
            'sentimento': sentiment
        }
        alerts = generate_smart_alerts(analysis_data)
        
        # Backtesting
        backtest_results = backtest_strategy(dados)

        return {
            "ticker": ticker,
            "preco_atual": round(last_price, 2),
            "previsao": round(forecast, 2),
            "diferenca_preco_previsao": round(forecast - last_price, 2), # New field for clarity
            "media_movel_7": round(ma7, 2),
            "media_movel_15": round(ma15, 2),
            "media_movel_30": round(ma30, 2),
            "tendencia": tendencia, # Use the new trend classification
            "estrategia": strategy, # Include the determined strategy
            "indicadores": {
                "RSI": round(float(dados['RSI'].iloc[-1]), 2),
                "MACD": round(float(dados['MACD'].iloc[-1]), 2),
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
                # "confianca_modelo_r2": resultado["confianca_modelo_r2"], # This was causing an error as it's not in the 'analyze' output
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