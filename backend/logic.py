import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

def analyze(ticker):
    try:
        dados = yf.download(ticker, period="180d", interval="1d")
        if dados.empty:
            return {"erro": "Ticker inválido ou sem dados"}

        dados.dropna(inplace=True)
        dados['Dias'] = np.arange(len(dados))
        dados['MM20'] = dados['Close'].rolling(window=20).mean()

        features = ['Dias', 'Volume', 'High', 'Low', 'Open']
        X = dados[features].copy()
        y = dados['Close'].values.ravel()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values)  # <-- aqui

        modelo = LinearRegression()
        modelo.fit(X_scaled, y)

        novo_dia = pd.DataFrame([{
            'Dias': len(dados),
            'Volume': dados['Volume'].iloc[-1],
            'High': dados['High'].iloc[-1],
            'Low': dados['Low'].iloc[-1],
            'Open': dados['Open'].iloc[-1]
        }])
        novo_dia_scaled = scaler.transform(novo_dia.values)  # <-- aqui

        r2 = float(r2_score(y, modelo.predict(X_scaled)))

        previsao = float(modelo.predict(novo_dia_scaled).item())
        ultimo_preco = dados['Close'].iloc[-1].item()  # <-- aqui

        tendencia = "alta" if previsao > ultimo_preco else "baixa"

        volatilidade = dados['Close'].pct_change().std() * 100
        if volatilidade > 3:
            risco = "alto"
        elif volatilidade > 1.5:
            risco = "moderado"
        else:
            risco = "baixo"

        mm20_valor = dados['MM20'].iloc[-1]
        media_movel_20d = round(float(mm20_valor), 2) if not pd.isna(mm20_valor) else None

        return {
            "ticker": ticker.upper(),
            "preco_atual": round(ultimo_preco, 2),
            "previsao_proximo_dia": round(previsao, 2),
            "tendencia": tendencia,
            "precisao_modelo_r2": round(r2, 3),
            "media_movel_20d": media_movel_20d,
            "volatilidade_%": round(volatilidade, 2),
            "classificacao_risco": risco,
            "estrategia": gerar_estrategia(tendencia, risco)
        }

    except Exception as e:
        return {"erro": str(e)}


def gerar_estrategia(tendencia, risco):
    if tendencia == "alta":
        if risco == "baixo":
            return "Tendência de alta com baixo risco. Boa oportunidade de compra 📈✅"
        elif risco == "moderado":
            return "Alta com risco moderado. Avalie sua estratégia antes de comprar 📈⚠️"
        else:
            return "Alta com risco alto. Cuidado com volatilidade 📈⚠️"
    else:
        if risco == "alto":
            return "Tendência de baixa e risco alto. Melhor evitar por enquanto 📉🚫"
        else:
            return "Tendência de baixa. Aguardar pode ser melhor 📉"
