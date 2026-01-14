from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import yfinance as yf
from typing import Optional
import logging
from datetime import datetime
from models import TickerRequest, AnalysisResponse, ErrorResponse
from logic import analyze, analyze_all, price_ticker
from config import settings

# Configuração de logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinAI API",
    description="API para análise financeira inteligente com IA",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=dict)
def root():
    """Endpoint de health check."""
    return {
        "message": "FinAI backend está funcionando 🚀",
        "version": "1.0.0",
        "status": "active",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
def health_check():
    """Endpoint detalhado de health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "ml_models": "available",
            "external_apis": "connected"
        }
    }

@app.get("/analise/acao")
def analisar_ativo(ticker: str = Query(..., description="Código da ação, ex: AAPL ou PETR4.SA")):
    """Analisa uma ação específica retornando indicadores técnicos e previsões."""
    try:
        if not ticker or ticker.strip() == "":
            raise HTTPException(status_code=400, detail="Ticker não pode estar vazio")
        
        resultado = analyze(ticker.upper().strip())
        logger.info(f"Análise realizada para ticker: {ticker}")
        return resultado
    except Exception as e:
        logger.error(f"Erro ao analisar ticker {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.get("/analise/acoes")
def analisar_todos():
    """Analisa múltiplas ações em lote."""
    try:
        resultado = analyze_all()
        logger.info("Análise em lote realizada com sucesso")
        return resultado
    except Exception as e:
        logger.error(f"Erro na análise em lote: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro na análise em lote: {str(e)}")

@app.get("/api/yahoo/{symbol}")
def yahoo_finance(symbol: str):
    """Proxy para API do Yahoo Finance."""
    if not symbol or symbol.strip() == "":
        raise HTTPException(status_code=400, detail="Symbol não pode estar vazio")
        
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.upper().strip()}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Dados do Yahoo Finance obtidos para: {symbol}")
        return JSONResponse(content=data)
    except requests.exceptions.Timeout:
        logger.error(f"Timeout ao buscar dados para {symbol}")
        raise HTTPException(status_code=408, detail="Timeout na requisição")
    except requests.exceptions.RequestException as e:
        logger.error(f"Erro na requisição para {symbol}: {str(e)}")
        raise HTTPException(status_code=502, detail="Erro ao acessar Yahoo Finance")
    except Exception as e:
        logger.error(f"Erro inesperado para {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor")

@app.get("/preco/{symbol}")
def price(symbol: str):
    """Obtém o preço atual de um símbolo."""
    try:
        if not symbol or symbol.strip() == "":
            raise HTTPException(status_code=400, detail="Symbol não pode estar vazio")
        
        res = price_ticker(symbol.upper().strip())
        logger.info(f"Preço obtido para symbol: {symbol}")
        return res
    except Exception as e:
        logger.error(f"Erro ao obter preço para {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter preço: {str(e)}")

# Endpoint para documentação das funcionalidades
@app.get("/features")
def get_features():
    """Lista todas as funcionalidades disponíveis da API."""
    return {
        "endpoints": {
            "/analise/acao": "Análise completa de uma ação individual",
            "/analise/acoes": "Análise em lote de múltiplas ações",
            "/preco/{symbol}": "Preço atual de um símbolo",
            "/api/yahoo/{symbol}": "Proxy para dados do Yahoo Finance"
        },
        "indicators": [
            "RSI", "MACD", "Médias Móveis", "Bandas de Bollinger", 
            "Oscilador Estocástico", "ATR", "VWAP"
        ],
        "ml_models": ["XGBoost", "Random Forest", "Gradient Boosting"],
        "data_sources": ["Yahoo Finance", "Análise de Sentimento"]
    }