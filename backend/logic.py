import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression

def analyze(ticker):
    try:
        dados = yf.download(ticker, period="180d", interval="1d")

        if dados.empty:
            return {"erro": "Ticker inválido ou sem dados"}

        volume_medio = dados['Volume'].mean()
        if isinstance(volume_medio, (float, int)) and volume_medio < 10000:
            return {"erro": "Volume médio muito baixo, ação com pouca liquidez"}

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

        r2 = modelo.score(features_normalizadas, precos) 

        ultima_linha = features_normalizadas.iloc[-1].values.reshape(1, -1)
        previsao = float(modelo.predict(ultima_linha)[0])

        ultimo_preco = float(dados['Close'].iloc[-1])

        tendencia = "alta" if previsao > ultimo_preco else "baixa"

        r2_formatado = f"{r2 * 100:,.2f}".replace('.', ',') + '%'

        return {
            "ticker": ticker,
            "preco_atual": round(ultimo_preco, 2),
            "previsao_proximo_dia": round(previsao, 2),
            "tendencia": tendencia,
            "confianca_modelo_r2": r2_formatado,
            "estrategia": gerar_estrategia(tendencia)
        }

    except Exception as e:
        return {"erro": str(e)}

def gerar_estrategia(tendencia):
    if tendencia == "alta":
        return "Considere manter ou comprar mais, a tendência é de valorização 📈"
    else:
        return "Considere vender ou aguardar nova entrada, tendência de queda 📉"


if __name__ == "__main__":

    print(analyze("AAPL"))



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