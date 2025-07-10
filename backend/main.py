from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse  # <- faltava isso
import requests  # <- e isso também
import yfinance as yf
from typing import Optional
from logic import analyze, analyze_all

app = FastAPI()

# Adicione isto para permitir CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "FinAI backend está funcionando 🚀"}

@app.get("/analise/acao")
def analisar_ativo(ticker: str = Query(..., description="Código da ação, ex: AAPL ou PETR4.SA")):
    resultado = analyze(ticker)
    return resultado

@app.get("/analise/acoes")
def analisar_todos():
    resultado = analyze_all()
    return resultado


@app.get("/api/yahoo/{symbol}")
def yahoo_finance(symbol: str):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(content={"error": "Erro ao buscar dados"}, status_code=500)

