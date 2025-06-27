from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
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