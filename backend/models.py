from pydantic import BaseModel, validator, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class TickerRequest(BaseModel):
    """Modelo para requisição de análise de ticker."""
    ticker: str = Field(..., description="Símbolo do ativo (ex: PETR4, AAPL)")
    historical: int = Field(default=90, description="Número de dias históricos")
    predictions: int = Field(default=3, description="Número de dias para predição")
    days_forecast: Optional[int] = Field(default=None, description="Alias para predictions (compatibilidade frontend)")
    isMock: bool = Field(default=False, description="Usar dados simulados")
    
    # Campos antigos para compatibilidade
    symbol: Optional[str] = Field(default=None, description="Alias para ticker")
    period: Optional[str] = Field(default="180d", description="Período dos dados")
    interval: Optional[str] = Field(default="1d", description="Intervalo dos dados")
    
    @validator('ticker', pre=True, always=True)
    def set_ticker_from_symbol(cls, v, values):
        # Se ticker não estiver definido, usar symbol
        if not v and 'symbol' in values:
            return values['symbol']
        return v
    
    @validator('predictions', pre=True, always=True)
    def set_predictions_from_days_forecast(cls, v, values):
        # Se days_forecast estiver definido, usar ele
        if 'days_forecast' in values and values['days_forecast'] is not None:
            return values['days_forecast']
        return v
    
    @validator('ticker')
    def ticker_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Ticker não pode estar vazio')
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