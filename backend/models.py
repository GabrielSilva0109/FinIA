from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any
from datetime import datetime

class TickerRequest(BaseModel):
    """Modelo para requisição de análise de ticker."""
    symbol: str
    period: Optional[str] = "180d"
    interval: Optional[str] = "1d"
    
    @validator('symbol')
    def symbol_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Symbol não pode estar vazio')
        return v.upper().strip()

class TechnicalIndicators(BaseModel):
    """Indicadores técnicos calculados."""
    rsi: Optional[float] = None
    macd: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    volatility: Optional[float] = None
    volume: Optional[float] = None

class SentimentAnalysis(BaseModel):
    """Resultado da análise de sentimento."""
    score: float
    label: str  # positive, negative, neutral
    confidence: float
    sources_count: int

class PredictionResult(BaseModel):
    """Resultado de predição de preços."""
    predicted_price: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    model_used: str
    prediction_date: datetime

class AnalysisResponse(BaseModel):
    """Resposta completa da análise."""
    symbol: str
    current_price: float
    technical_indicators: TechnicalIndicators
    sentiment_analysis: Optional[SentimentAnalysis] = None
    prediction: Optional[PredictionResult] = None
    recommendation: str  # BUY, SELL, HOLD
    risk_level: str  # LOW, MEDIUM, HIGH
    last_updated: datetime
    
class ErrorResponse(BaseModel):
    """Modelo para respostas de erro."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = datetime.now()