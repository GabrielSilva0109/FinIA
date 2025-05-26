import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def analisar_acao(ticker):
    try:
        dados = yf.download(ticker, period="60d", interval="1d")
        if dados.empty:
            return {"erro": "Ticker inválido ou sem dados"}

        dados['Dias'] = np.arange(len(dados)).reshape(-1, 1)
        modelo = LinearRegression()
        modelo.fit(dados['Dias'], dados['Close'])

        previsao = modelo.predict([[len(dados)]])[0]
        ultimo_preco = dados['Close'].iloc[-1]

        tendencia = "alta" if previsao > ultimo_preco else "baixa"

        return {
            "ticker": ticker,
            "preco_atual": round(ultimo_preco, 2),
            "previsao_proximo_dia": round(previsao, 2),
            "tendencia": tendencia,
            "estrategia": gerar_estrategia(tendencia)
        }

    except Exception as e:
        return {"erro": str(e)}

def gerar_estrategia(tendencia):
    if tendencia == "alta":
        return "Considere manter ou comprar mais, a tendência é de valorização 📈"
    else:
        return "Considere vender ou aguardar nova entrada, tendência de queda 📉"
