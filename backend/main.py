from fastapi import FastAPI, Query
import yfinance as yf
from typing import Optional
from logic import analyze, analyze_all

app = FastAPI()

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