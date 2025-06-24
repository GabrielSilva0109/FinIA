# import yfinance as yf
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# import requests
# from bs4 import BeautifulSoup
# import re

# def compute_technical_indicators(df):
#     df['retorno'] = df['Close'].pct_change()
#     df['MA7'] = df['Close'].rolling(window=7).mean()
#     df['MA20'] = df['Close'].rolling(window=20).mean()
#     df['RSI'] = compute_rsi(df['Close'])
#     df['volatilidade'] = df['retorno'].rolling(window=7).std()
#     df = df.dropna()
#     return df

# def compute_rsi(series, period=14):
#     delta = series.diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
#     rs = gain / loss
#     rsi = 100 - (100 / (1 + rs))
#     return rsi

# def compute_macd(df, short=12, long=26, signal=9):
#     exp1 = df['Close'].ewm(span=short, adjust=False).mean()
#     exp2 = df['Close'].ewm(span=long, adjust=False).mean()
#     macd = exp1 - exp2
#     signal_line = macd.ewm(span=signal, adjust=False).mean()
#     return macd, signal_line

# def analyze(ticker):
#     try:
#         dados = yf.download(ticker, period="180d", interval="1d")

#         if dados.empty:
#             return {"erro": "Ticker inválido ou sem dados"}

#         volume_medio = dados['Volume'].mean()
#         if isinstance(volume_medio, (float, int)) and volume_medio < 10000:
#             return {"erro": "Volume médio muito baixo, ação com pouca liquidez"}

#         # Indicadores técnicos
#         dados['retorno'] = dados['Close'].pct_change()
#         dados['MA7'] = dados['Close'].rolling(window=7).mean()
#         dados['MA20'] = dados['Close'].rolling(window=20).mean()
#         dados['EMA12'] = dados['Close'].ewm(span=12, adjust=False).mean()
#         dados['EMA26'] = dados['Close'].ewm(span=26, adjust=False).mean()
#         dados['RSI'] = compute_rsi(dados['Close'])
#         dados['MACD'], dados['MACD_Signal'] = compute_macd(dados)
#         dados['volatilidade'] = dados['retorno'].rolling(window=7).std()

#         dados = dados.dropna()

#         # Médias móveis adicionais
#         dados['SMA_7'] = dados['Close'].rolling(window=7).mean()
#         dados['SMA_15'] = dados['Close'].rolling(window=15).mean()
#         dados['SMA_30'] = dados['Close'].rolling(window=30).mean()
#         media_movel_7 = dados['SMA_7'].iloc[-1]
#         media_movel_15 = dados['SMA_15'].iloc[-1]
#         media_movel_30 = dados['SMA_30'].iloc[-1]

#         if np.isnan(media_movel_7):
#             media_movel_7 = None
#         if np.isnan(media_movel_15):
#             media_movel_15 = None
#         if np.isnan(media_movel_30):
#             media_movel_30 = None

#         # Volatilidade: desvio padrão dos retornos diários (%)
#         dados['retorno_diario'] = dados['Close'].pct_change()
#         volatilidade = dados['retorno_diario'].std()

#         # Normalização dos dados para o modelo
#         features = dados[[
#             'Open', 'High', 'Low', 'Volume',
#             'RSI', 'MACD', 'MACD_Signal',
#             'volatilidade', 'SMA_7', 'SMA_15', 'SMA_30'
#         ]].copy()

#         features.fillna(0, inplace=True)

#         minimos = features.min()
#         maximos = features.max()
#         denominador = maximos - minimos
#         denominador[denominador == 0] = 1

#         features_normalizadas = (features - minimos) / denominador

#         precos = dados['Close'].values

#         modelo = LinearRegression()
#         modelo.fit(features_normalizadas, precos)

#         ultima_linha = features_normalizadas.iloc[-1].values.reshape(1, -1)
#         previsao = float(modelo.predict(ultima_linha)[0])

#         ultimo_preco = float(dados['Close'].iloc[-1])

#         tendencia = "alta" if previsao >= ultimo_preco * 0.995 else "baixa"

#         r2 = modelo.score(features_normalizadas, precos) * 100

#         sentimento = sentimentAnalysis(ticker)

#         return {
#             "ticker": ticker,
#             "preco_atual": round(ultimo_preco, 2),
#             "previsao": round(previsao, 2),
#             "media_movel_7": round(media_movel_7, 2) if media_movel_7 is not None else None,
#             "media_movel_15": round(media_movel_15, 2) if media_movel_15 is not None else None,
#             "media_movel_30": round(media_movel_30, 2) if media_movel_30 is not None else None,
#             "diferenca": round(previsao - ultimo_preco, 2),
#             "volatilidade": f"{volatilidade*100:.2f}%",
#             "tendencia": tendencia,
#             "confianca_modelo_r2": f"{r2:.2f}%",
#             "sentimento": sentimento,
#             "estrategia": gerar_estrategia(tendencia)
#         }

#     except Exception as e:
#         return {"erro": str(e)}

# def sentimentAnalysis(ticker):
#     try:
#         # Pega nome da empresa pelo yfinance (fallback para ticker sem sufixo)
#         ticker_base = ticker.split('.')[0]
#         try:
#             nome_empresa = yf.Ticker(ticker).info.get('shortName', None)
#         except:
#             nome_empresa = None

#         termos_busca = [ticker, ticker_base]
#         if nome_empresa:
#             termos_busca.append(nome_empresa)

#         headlines = []

#         headers = {"User-Agent": "Mozilla/5.0"}

#         for termo in termos_busca:
#             url = f"https://news.google.com/search?q={requests.utils.quote(termo)}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
#             response = requests.get(url, headers=headers)
#             soup = BeautifulSoup(response.text, 'html.parser')

#             # Tenta capturar vários seletores possíveis para headlines
#             novos_headlines = []
#             for selector in ['article h3', 'article h4', 'div > a > h3']:
#                 novos_headlines.extend([h.get_text() for h in soup.select(selector)])

#             # Limpa texto: remove urls, espaços extras e caracteres estranhos
#             def limpar_texto(texto):
#                 texto = re.sub(r"http\S+", "", texto)
#                 texto = re.sub(r"\s+", " ", texto).strip()
#                 return texto

#             novos_headlines = [limpar_texto(h) for h in novos_headlines if h.strip() != '']
#             headlines.extend(novos_headlines)

#         # Remover duplicatas e limitar para, por exemplo, 30 headlines
#         headlines = list(dict.fromkeys(headlines))[:30]

#         print(f"Headlines encontradas: {len(headlines)}")

#         if not headlines:
#             return "neutro"

#         analyzer = SentimentIntensityAnalyzer()
#         scores = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]

#         media_score = sum(scores) / len(scores)

#         if media_score >= 0.05:
#             return "positivo"
#         elif media_score <= -0.05:
#             return "negativo"
#         else:
#             return "neutro"

#     except Exception as e:
#         print(f"Erro na análise de sentimento: {e}")
#         return "neutro"

# def gerar_estrategia(tendencia):
#     if tendencia == "alta":
#         return "Considere manter ou comprar mais, a tendência é de valorização 📈"
#     else:
#         return "Considere vender ou aguardar nova entrada, tendência de queda 📉"

# def analyze_all():
#     try:
#         # Fallback para pegar as ações do Ibovespa via scraping
#         url = "https://finance.yahoo.com/quote/%5EBVSP/components?p=%5EBVSP"
#         headers = {"User-Agent": "Mozilla/5.0"}
#         response = requests.get(url, headers=headers)
#         soup = BeautifulSoup(response.text, "html.parser")
#         tickers_lista = []

#         for a in soup.select("table tbody tr td:nth-child(1) a"):
#             ticker = a.text.strip()
#             if not ticker.endswith(".SA"):
#                 ticker += ".SA"
#             tickers_lista.append(ticker)

#         tickers_lista = tickers_lista[:30]  # Limita às 30 maiores

#         resultados = []

#         for ticker in tickers_lista:
#             print(f"Analisando {ticker}...")
#             resultado = analyze(ticker)

#             if "erro" in resultado:
#                 continue

#             preco_atual = resultado["preco_atual"]
#             previsao = resultado["previsao"]
#             diferenca = previsao - preco_atual
#             media_movel_7 = resultado["media_movel_7"]
#             media_movel_15 = resultado["media_movel_15"]
#             media_movel_30 = resultado["media_movel_30"]

#             resultados.append({
#                 "ticker": ticker,
#                 "preco_atual": preco_atual,
#                 "previsao": previsao,
#                 "media_movel_7": media_movel_7,
#                 "media_movel_15": media_movel_15,
#                 "media_movel_30": media_movel_30,
#                 "diferenca": diferenca,
#                 "tendencia": resultado["tendencia"],
#                 "confianca_modelo_r2": resultado["confianca_modelo_r2"],
#                 "sentimento": resultado["sentimento"],
#                 "estrategia": resultado["estrategia"]
#             })

#         # Ordena por diferença de previsão - preço atual
#         oportunidades = sorted(resultados, key=lambda x: x["diferenca"], reverse=True)[:5]
#         quedas = sorted(resultados, key=lambda x: x["diferenca"])[:5]

#         return {
#             "melhores_oportunidades": oportunidades,
#             "maiores_quedas": quedas
#         }

#     except Exception as e:
#         return {"erro": str(e)}