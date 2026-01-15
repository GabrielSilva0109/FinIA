from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import yfinance as yf
from typing import Optional
import logging
from datetime import datetime
from models import TickerRequest, TechnicalIndicators, SentimentAnalysis
from logic_enhanced import EnhancedFinancialAnalyzer
from logic_crypto import crypto_analyzer
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
    allow_origins=["*"],  # Permitir todas as origens temporariamente
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
        "service": "FinAI IA-Bot v3.0",
        "version": "3.0_intelligent",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "ml_models": "available",
            "external_apis": "connected"
        }
    }

@app.get("/analise/acao")
def analisar_ativo(ticker: str = Query(..., description="Código da ação, ex: AAPL ou PETR4.SA")):
    """Analisa uma ação específica usando sistema enhanced."""
    try:
        if not ticker or ticker.strip() == "":
            raise HTTPException(status_code=400, detail="Ticker não pode estar vazio")
        
        # Usar sistema enhanced unificado
        analyzer = EnhancedFinancialAnalyzer()
        resultado = analyzer.generate_enhanced_chart_data(ticker.upper().strip(), 15)
        
        logger.info(f"Análise enhanced realizada para ticker: {ticker}")
        return resultado
    except Exception as e:
        logger.error(f"Erro ao analisar ticker {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.get("/analise/acoes")
def analisar_todos():
    """Analisa múltiplas ações em lote."""
    try:
        # Análise em lote não implementada na nova versão
        raise HTTPException(status_code=501, detail="Análise em lote não implementada")
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
        
        # Buscar preço usando yfinance
        stock = yf.Ticker(symbol.upper().strip())
        info = stock.info
        current_price = info.get('regularMarketPrice') or info.get('currentPrice')
        
        if not current_price:
            # Tentar através de dados históricos
            hist = stock.history(period="1d")
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
            
        if not current_price:
            raise HTTPException(status_code=404, detail=f"Preço não encontrado para {symbol}")
            
        res = {
            "symbol": symbol.upper(),
            "price": current_price,
            "currency": info.get('currency', 'USD'),
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"Preço obtido para symbol: {symbol}")
        return res
    except Exception as e:
        logger.error(f"Erro ao obter preço para {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter preço: {str(e)}")

# Endpoint para documentação das funcionalidades
@app.get("/features")
def get_features():
    """Lista todas as funcionalidades disponíveis da API enhanced."""
    return {
        "endpoints": {
            "/analise/acao": "Análise enhanced de ação individual",
            "/analise/acoes": "Análise em lote (não implementado)",
            "/analise/crypto": "Análise completa de criptomoedas", 
            "/chart-data": "Análise completa e robusta com IA avançada",
            "/generate-chart": "Geração de gráficos com ML",
            "/crypto/symbols": "Lista símbolos de crypto disponíveis",
            "/preco/{symbol}": "Preço atual de um símbolo",
            "/api/yahoo/{symbol}": "Proxy para dados do Yahoo Finance"
        },
        "advanced_features": [
            "Machine Learning Ensemble (XGBoost + Random Forest + Gradient Boosting)",
            "Advanced Technical Indicators (Ichimoku, Stochastic, ADX, ATR)",
            "Intelligent Confidence System (Multi-factor)", 
            "Dynamic Risk Management (Stop-loss/Take-profit adaptativos)",
            "Auto Brazilian Ticker Correction (.SA)",
            "Real-time Predictions (30 days forecast)"
        ],
        "indicators": [
            "RSI", "MACD", "Médias Móveis", "Bandas de Bollinger", 
            "Williams %R", "CCI", "Oscilador Estocástico", "ATR", "VWAP", "ADX", 
            "Ichimoku Cloud", "Fibonacci Retracements", "Volume Analysis"
        ],
        "ml_models": ["XGBoost", "Random Forest", "Gradient Boosting", "Linear Regression"],
        "data_sources": ["Yahoo Finance", "Binance", "CoinGecko"],
        "api_version": "2.0 Enhanced"
    }

# === ENDPOINTS DE CRIPTOMOEDAS ===

@app.get("/crypto/symbols")
def get_crypto_symbols():
    """Lista símbolos de criptomoedas disponíveis."""
    try:
        symbols = crypto_analyzer.get_available_symbols()
        # Filtrar apenas os principais pares USDT
        usdt_pairs = [s for s in symbols if s.endswith('/USDT')][:50]  # Top 50
        
        return {
            "symbols": usdt_pairs,
            "total_available": len(symbols),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Erro ao obter símbolos de cripto: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter símbolos: {str(e)}")

@app.get("/analise/crypto")
def analisar_crypto(
    symbol: str = Query("BTC/USDT", description="Par de trading, ex: BTC/USDT"),
    timeframe: str = Query("1d", description="Timeframe: 1m, 5m, 1h, 1d")
):
    """Análise completa de criptomoedas."""
    try:
        if not symbol or symbol.strip() == "":
            raise HTTPException(status_code=400, detail="Symbol não pode estar vazio")
        
        # Validar timeframe
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
        if timeframe not in valid_timeframes:
            raise HTTPException(
                status_code=400, 
                detail=f"Timeframe inválido. Use: {', '.join(valid_timeframes)}"
            )
        
        resultado = crypto_analyzer.analyze_crypto(symbol.upper().strip(), timeframe)
        
        if 'error' in resultado:
            raise HTTPException(status_code=404, detail=resultado['error'])
        
        logger.info(f"Análise de cripto realizada para: {symbol}")
        return resultado
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na análise de cripto {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.get("/crypto/info/{coin_id}")
def get_crypto_info(coin_id: str):
    """Obtém informações detalhadas de uma criptomoeda do CoinGecko."""
    try:
        if not coin_id or coin_id.strip() == "":
            raise HTTPException(status_code=400, detail="Coin ID não pode estar vazio")
        
        info = crypto_analyzer.fetch_coingecko_info(coin_id.lower().strip())
        
        if not info:
            raise HTTPException(status_code=404, detail=f"Informações não encontradas para {coin_id}")
        
        return {
            "coin_id": coin_id,
            "info": info,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter info de {coin_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.post("/chart-data")
def get_chart_data(request: TickerRequest):
    """Gera análise financeira ULTRA-INTELIGENTE v3.0 com IA avançada, LSTM, Prophet e ensemble de ML."""
    try:
        # Extrair parâmetros do modelo validado
        raw_ticker = request.ticker
        days_forecast = request.predictions
        
        # Se days_forecast foi passado pelo frontend, usar ele
        if hasattr(request, 'days_forecast') and request.days_forecast:
            days_forecast = request.days_forecast
            
        if not raw_ticker:
            raise HTTPException(status_code=400, detail="Ticker não pode estar vazio")
        
        # LÓGICA INTELIGENTE: Auto-corrigir tickers brasileiros
        ticker = raw_ticker
        if raw_ticker and not '.' in raw_ticker:
            # Se é um ticker brasileiro sem sufixo, adicionar .SA
            brazilian_patterns = ['3', '4', '11']  # Terminações típicas de ações brasileiras
            if any(raw_ticker.endswith(pattern) for pattern in brazilian_patterns):
                ticker = f"{raw_ticker}.SA"
                logger.info(f"Auto-corrigido ticker brasileiro: {raw_ticker} -> {ticker}")
        
        # Validar days_forecast
        days_forecast = max(1, min(10, int(days_forecast)))  # Entre 1 e 10 dias para IA avançada
        
        logger.info(f"🤖 Iniciando análise INTELIGENTE v3.0 para: {ticker}")
        
        # USAR ANÁLISE INTELIGENTE v3.0 com todas as melhorias
        from logic_enhanced import generate_intelligent_analysis
        result = generate_intelligent_analysis(ticker, days_forecast)
        
        # Verificar se houve erro na obtenção de dados
        if result.get('error') or len(result.get('historical_data', [])) == 0:
            logger.warning(f"Dados não encontrados para {ticker}, tentando fallback...")
            
            # Tentar diferentes formatos
            fallback_tickers = []
            if ticker.endswith('.SA'):
                fallback_tickers.append(ticker.replace('.SA', ''))  # Remover .SA
            elif not '.' in ticker:
                fallback_tickers.extend([f"{ticker}.SA", f"{ticker}.SAO"])  # Adicionar sufixos
            
            for fallback_ticker in fallback_tickers:
                logger.info(f"Tentando fallback inteligente: {fallback_ticker}")
                result = generate_intelligent_analysis(fallback_ticker, days_forecast)
                if not result.get('error') and len(result.get('historical_data', [])) > 0:
                    ticker = fallback_ticker
                    break
        
        logger.info(f"🎯 Análise INTELIGENTE v3.0 concluída para {ticker}")
        
        # COMPATIBILIDADE COM FRONTEND: Adicionar campos esperados
        
        # Ajustar predictions para incluir campo confidence
        if 'prediction_data' in result:
            for pred in result['prediction_data']:
                if 'confidence' not in pred:
                    pred['confidence'] = result.get('analysis', {}).get('confidence', 80) / 100
        
        # Adicionar timestamp se não existir
        if 'timestamp' not in result:
            result['timestamp'] = datetime.now().isoformat()
        
        # Adicionar metadados da API v3.0
        result['api_version'] = '3.0_intelligent'
        result['original_ticker'] = raw_ticker  # Ticker original da requisição
        result['ticker'] = ticker  # Campo esperado pelo frontend
        result['ai_features'] = [
            'LSTM Neural Networks',
            'Prophet Time Series Forecasting',
            'XGBoost + Random Forest + GBM Ensemble',
            'Auto-Hyperparameter Tuning',
            'Market Regime Detection',
            'Smart Confidence System',
            'Fundamental Analysis Integration',
            'Dynamic Support/Resistance ML',
            'Price Pattern Recognition',
            'Intelligent Trading Signals',
            'Multi-timeframe Analysis',
            'Feature Importance Tracking',
            'Model Performance Monitoring',
            'Auto Brazilian Ticker Correction'
        ]
        
        # Adicionar estatísticas de IA
        if 'market_intelligence' in result:
            result['ai_stats'] = {
                'models_count': len(result.get('feature_importance', {})),
                'confidence_level': result.get('confidence_analysis', {}).get('confidence_level', 'MÉDIA'),
                'market_regime': result.get('market_intelligence', {}).get('market_regime', {}).get('regime', 'UNKNOWN'),
                'intelligence_version': result.get('intelligence_version', 'v3.0'),
                'analysis_timestamp': datetime.now().isoformat()
            }
        
        return result
    except ValueError as e:
        logger.error(f"Erro de validação para {raw_ticker}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Parâmetro inválido: {str(e)}")
    except Exception as e:
        logger.error(f"Erro na análise completa de {raw_ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro na análise: {str(e)}")

# Endpoint chart-data-enhanced removido - funcionalidade unificada em /chart-data

@app.post("/generate-chart")
def generate_financial_chart(request: dict):
    """Gera gráfico financeiro com previsões ML usando sistema enhanced."""
    try:
        analyzer = EnhancedFinancialAnalyzer()
        
        ticker = request.get('ticker', '').upper().strip()
        days_forecast = request.get('days_forecast', 15)
        
        if not ticker:
            raise HTTPException(status_code=400, detail="Ticker não pode estar vazio")
        
        logger.info(f"Gerando análise completa para: {ticker}")
        result = analyzer.generate_enhanced_chart_data(ticker, days_forecast)
        
        # Adicionar flag para indicar que é para gráfico
        result['chart_ready'] = True
        result['chart_type'] = request.get('chart_type', 'candlestick')
        
        logger.info(f"Análise para gráfico gerada para {ticker}")
        return result
    except Exception as e:
        logger.error(f"Erro ao gerar gráfico: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar gráfico: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)