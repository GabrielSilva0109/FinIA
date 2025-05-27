import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup

def analyze(ticker):
    try:
        dados = yf.download(ticker, period="180d", interval="1d")

        if dados.empty:
            return {"erro": "Ticker inválido ou sem dados"}

        volume_medio = dados['Volume'].mean()
        if isinstance(volume_medio, (float, int)) and volume_medio < 10000:
            return {"erro": "Volume médio muito baixo, ação com pouca liquidez"}

        # Médias móveis
        dados['SMA_7'] = dados['Close'].rolling(window=7).mean()
        dados['SMA_15'] = dados['Close'].rolling(window=15).mean()
        dados['SMA_30'] = dados['Close'].rolling(window=30).mean()

        media_movel_7 = dados['SMA_7'].iloc[-1]
        media_movel_15 = dados['SMA_15'].iloc[-1]
        media_movel_30 = dados['SMA_30'].iloc[-1]

        if np.isnan(media_movel_7):
            media_movel_7 = None
        if np.isnan(media_movel_15):
            media_movel_15 = None
        if np.isnan(media_movel_30):
            media_movel_30 = None

        # Volatilidade: desvio padrão dos retornos diários (%)
        dados['retorno_diario'] = dados['Close'].pct_change()
        volatilidade = dados['retorno_diario'].std()

        # Normalização dos dados para o modelo
        features = dados[['Open', 'High', 'Low', 'Volume']].copy()
        features.fillna(0, inplace=True)

        minimos = features.min()
        maximos = features.max()
        denominador = maximos - minimos
        denominador[denominador == 0] = 1

        features_normalizadas = (features - minimos) / denominador

        precos = dados['Close'].values

        modelo = LinearRegression()
        modelo.fit(features_normalizadas, precos)

        ultima_linha = features_normalizadas.iloc[-1].values.reshape(1, -1)
        previsao = float(modelo.predict(ultima_linha)[0])

        ultimo_preco = float(dados['Close'].iloc[-1])

        tendencia = "alta" if previsao > ultimo_preco else "baixa"

        r2 = modelo.score(features_normalizadas, precos) * 100

        sentimento = sentimentAnalysis(ticker)

        return {
            "ticker": ticker,
            "preco_atual": round(ultimo_preco, 2),
            "previsao_proximo_dia": round(previsao, 2),
            "media_movel_7": round(media_movel_7, 2) if media_movel_7 is not None else None,
            "media_movel_15": round(media_movel_15, 2) if media_movel_15 is not None else None,
            "media_movel_30": round(media_movel_30, 2) if media_movel_30 is not None else None,
            "volatilidade": f"{volatilidade*100:.2f}%",
            "tendencia": tendencia,
            "confianca_modelo_r2": f"{r2:.2f}%",
            "sentimento": sentimento,
            "estrategia": gerar_estrategia(tendencia)
        }

    except Exception as e:
        return {"erro": str(e)}


def sentimentAnalysis(ticker):
    try:
        # Pega nome da empresa pelo yfinance (fallback para ticker sem sufixo)
        ticker_base = ticker.split('.')[0]
        try:
            nome_empresa = yf.Ticker(ticker).info.get('shortName', None)
        except:
            nome_empresa = None

        termos_busca = [ticker, ticker_base]
        if nome_empresa:
            termos_busca.append(nome_empresa)

        headlines = []

        headers = {"User-Agent": "Mozilla/5.0"}

        for termo in termos_busca:
            url = f"https://news.google.com/search?q={requests.utils.quote(termo)}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Tenta capturar vários seletores possíveis para headlines
            novos_headlines = []
            for selector in ['article h3', 'article h4', 'div > a > h3']:
                novos_headlines.extend([h.get_text() for h in soup.select(selector)])

            # Limpa texto: remove urls, espaços extras e caracteres estranhos
            def limpar_texto(texto):
                texto = re.sub(r"http\S+", "", texto)
                texto = re.sub(r"\s+", " ", texto).strip()
                return texto

            novos_headlines = [limpar_texto(h) for h in novos_headlines if h.strip() != '']
            headlines.extend(novos_headlines)

        # Remover duplicatas e limitar para, por exemplo, 30 headlines
        headlines = list(dict.fromkeys(headlines))[:30]

        print(f"Headlines encontradas: {len(headlines)}")

        if not headlines:
            return "neutro"

        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]

        media_score = sum(scores) / len(scores)

        if media_score >= 0.05:
            return "positivo"
        elif media_score <= -0.05:
            return "negativo"
        else:
            return "neutro"

    except Exception as e:
        print(f"Erro na análise de sentimento: {e}")
        return "neutro"

def gerar_estrategia(tendencia):
    if tendencia == "alta":
        return "Considere manter ou comprar mais, a tendência é de valorização 📈"
    else:
        return "Considere vender ou aguardar nova entrada, tendência de queda 📉"


# import yfinance as yf
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import numpy as np

# def analyze(ticker):
#     try:
#         dados = yf.download(ticker, period="180d", interval="1d")
#         if dados.empty:
#             return {"erro": "Ticker inválido ou sem dados"}

#         dias = np.arange(len(dados)).reshape(-1, 1)
#         precos = dados['Close'].values.reshape(-1)

#         modelo = LinearRegression()
#         modelo.fit(dias, precos)

#         previsao = float(modelo.predict(np.array([[len(dados)]]))[0])
#         ultimo_preco = float(dados['Close'].iloc[-1])

#         tendencia = "alta" if previsao > ultimo_preco else "baixa"

#         return {
#             "ticker": ticker,
#             "preco_atual": round(ultimo_preco, 2),
#             "previsao_proximo_dia": round(previsao, 2),
#             "tendencia": tendencia,
#             "estrategia": gerar_estrategia(tendencia)
#         }

#     except Exception as e:
#         return {"erro": str(e)}

# def gerar_estrategia(tendencia):
#     if tendencia == "alta":
#         return "Considere manter ou comprar mais, a tendência é de valorização 📈"
#     else:
#         return "Considere vender ou aguardar nova entrada, tendência de queda 📉"