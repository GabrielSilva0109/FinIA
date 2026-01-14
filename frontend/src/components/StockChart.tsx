import React, { useState, useEffect } from 'react';
import axios from 'axios';

// Substituindo os componentes ShadCN por componentes simples
const Card: React.FC<{children: React.ReactNode, className?: string}> = ({children, className = ''}) => (
  <div className={`card ${className}`}>{children}</div>
);

const CardContent: React.FC<{children: React.ReactNode, className?: string}> = ({children, className = ''}) => (
  <div className={`card-content ${className}`}>{children}</div>
);

const CardHeader: React.FC<{children: React.ReactNode, className?: string}> = ({children, className = ''}) => (
  <div className={`card-header ${className}`}>{children}</div>
);

const CardTitle: React.FC<{children: React.ReactNode, className?: string}> = ({children, className = ''}) => (
  <h3 className={`card-title ${className}`}>{children}</h3>
);

const CardDescription: React.FC<{children: React.ReactNode, className?: string}> = ({children, className = ''}) => (
  <p className={`card-description ${className}`}>{children}</p>
);

const Button: React.FC<{children: React.ReactNode, onClick?: () => void, disabled?: boolean, className?: string}> = 
  ({children, onClick, disabled = false, className = ''}) => (
  <button className={`btn btn-primary ${className}`} onClick={onClick} disabled={disabled}>
    {children}
  </button>
);

const Input: React.FC<{type?: string, placeholder?: string, value?: string | number, onChange?: (e: any) => void, className?: string}> = 
  ({type = 'text', placeholder, value, onChange, className = ''}) => (
  <input 
    type={type} 
    placeholder={placeholder} 
    value={value} 
    onChange={onChange}
    className={`input ${className}`}
  />
);

// Ícones simples usando SVG
const TrendingUp = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
  </svg>
);

const TrendingDown = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
  </svg>
);

const BarChart3 = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
  </svg>
);

const RefreshCw = ({className}: {className?: string}) => (
  <svg className={`h-4 w-4 ${className}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
  </svg>
);

const AlertCircle = () => (
  <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

interface AnalysisSummary {
  current_price: number;
  predicted_price: number;
  price_change_percent: number;
  recommendation: string;
  confidence: number;
  trend: string;
}

interface TechnicalIndicators {
  RSI: number;
  MACD: number;
  BB_position: number;
  Williams_R: number;
  volatility: number;
}

interface ChartData {
  chart_base64: string;
  chart_url: string;
  predictions: any;
  analysis_summary: AnalysisSummary;
  technical_indicators: TechnicalIndicators;
  erro?: string;
}

const StockChart: React.FC = () => {
  const [ticker, setTicker] = useState<string>('AAPL');
  const [daysForecast, setDaysForecast] = useState<number>(15);
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const fetchChart = async () => {
    if (!ticker.trim()) return;
    
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.post('http://localhost:8000/generate-chart', {
        ticker: ticker.toUpperCase(),
        days_forecast: daysForecast,
        chart_type: 'candlestick'
      });
      
      setChartData(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Erro ao carregar gráfico');
      console.error('Erro:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (ticker.trim()) {
      fetchChart();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticker, daysForecast]);

  const getRecommendationColor = (recommendation: string) => {
    switch (recommendation?.toUpperCase()) {
      case 'COMPRA':
      case 'COMPRA FORTE':
        return 'text-green-600 bg-green-50';
      case 'VENDA':
      case 'VENDA FORTE':
        return 'text-red-600 bg-red-50';
      default:
        return 'text-yellow-600 bg-yellow-50';
    }
  };

  const getRecommendationIcon = (recommendation: string) => {
    switch (recommendation?.toUpperCase()) {
      case 'COMPRA':
      case 'COMPRA FORTE':
        return <span style={{color: 'green', fontSize: '18px'}}>📈</span>;
      case 'VENDA':
      case 'VENDA FORTE':
        return <span style={{color: 'red', fontSize: '18px'}}>📉</span>;
      default:
        return <span style={{color: '#ca8a04', fontSize: '18px'}}>📊</span>;
    }
  };

  return (
    <div className="container">
      <div style={{display: 'flex', flexDirection: 'column', gap: '24px'}}>
        
        {/* Header */}
        <Card className="bg-white/80 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="header-title">
              <span style={{fontSize: '24px', marginRight: '12px'}}>📊</span>
              IA-Bot Financial Analysis
            </CardTitle>
            <CardDescription className="text-lg text-gray-600">
              Análise técnica avançada com previsões de Machine Learning
            </CardDescription>
          </CardHeader>
        </Card>

        {/* Controls */}
        <Card className="bg-white/80 backdrop-blur-sm">
          <CardContent>
            <div className="form-group">
              <div className="input-group">
                <label className="label">Símbolo da Ação</label>
                <Input
                  type="text"
                  placeholder="Ex: AAPL, TSLA, MSFT..."
                  value={ticker}
                  onChange={(e) => setTicker(e.target.value.toUpperCase())}
                  className="input"
                />
              </div>
              <div className="input-group">
                <label className="label">Dias de Previsão</label>
                <input
                  type="number"
                  min="1"
                  max="30"
                  value={daysForecast}
                  onChange={(e) => setDaysForecast(Number(e.target.value))}
                  className="input"
                />
              </div>
              <button 
                onClick={fetchChart}
                disabled={loading}
                className="btn btn-primary"
              >
                {loading ? (
                  <span className="loading-spinner">⟳</span>
                ) : (
                  <span>📊</span>
                )}
                {loading ? 'Gerando...' : 'Analisar'}
              </button>
            </div>
          </CardContent>
        </Card>

        {/* Error */}
        {error && (
          <div className="error-card">
            <span>⚠️</span>
            <span>{error}</span>
          </div>
        )}

        {/* Chart and Analysis */}
        {chartData && !chartData.erro && (
          <div className="grid grid-cols-3">
            
            {/* Main Chart */}
            <div>
              <Card>
                <CardHeader>
                  <CardTitle>
                    {ticker} - Análise Técnica com Previsões ML
                  </CardTitle>
                  <CardDescription>
                    Últimos 3 meses + {daysForecast} dias de previsão
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="chart-container">
                    <img 
                      src={chartData.chart_url}
                      alt={`Gráfico ${ticker}`}
                    />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Analysis Panel */}
            <div className="analysis-panel">
              
              {/* Recommendation */}
              <Card>
                <CardHeader>
                  <CardTitle>Recomendação</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className={`recommendation-card ${getRecommendationColor(chartData.analysis_summary.recommendation)}`}>
                    {getRecommendationIcon(chartData.analysis_summary.recommendation)}
                    <div>
                      <div style={{fontWeight: 'bold', fontSize: '1.1rem'}}>
                        {chartData.analysis_summary.recommendation}
                      </div>
                      <div style={{fontSize: '0.875rem', opacity: 0.75}}>
                        Confiança: {chartData.analysis_summary.confidence}%
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Price Analysis */}
              <Card>
                <CardHeader>
                  <CardTitle>Análise de Preços</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="metrics-grid">
                    <div className="metric-card" style={{background: '#dbeafe'}}>
                      <div className="metric-label">Preço Atual</div>
                      <div className="metric-value" style={{color: '#3b82f6'}}>
                        ${chartData.analysis_summary.current_price.toFixed(2)}
                      </div>
                    </div>
                    <div className="metric-card" style={{background: '#dcfce7'}}>
                      <div className="metric-label">Previsão {daysForecast}d</div>
                      <div className="metric-value" style={{color: '#16a34a'}}>
                        ${chartData.analysis_summary.predicted_price.toFixed(2)}
                      </div>
                    </div>
                  </div>
                  
                  <div className="metric-card" style={{background: '#f9fafb', marginTop: '12px'}}>
                    <div className="metric-label">Variação Esperada</div>
                    <div className={`metric-value`} style={{
                      color: chartData.analysis_summary.price_change_percent >= 0 ? '#16a34a' : '#dc2626'
                    }}>
                      {chartData.analysis_summary.price_change_percent > 0 ? '+' : ''}
                      {chartData.analysis_summary.price_change_percent.toFixed(1)}%
                    </div>
                  </div>

                  <div className="metric-card" style={{background: '#faf5ff', marginTop: '12px'}}>
                    <div className="metric-label">Tendência ML</div>
                    <div className="metric-value" style={{color: '#9333ea'}}>
                      {chartData.analysis_summary.trend}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Technical Indicators */}
              <Card>
                <CardHeader>
                  <CardTitle>Indicadores Técnicos</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="indicators-list">
                    <div className="indicator-row">
                      <span className="indicator-label">RSI</span>
                      <span className="indicator-value" style={{
                        color: chartData.technical_indicators.RSI > 70 ? '#dc2626' :
                               chartData.technical_indicators.RSI < 30 ? '#16a34a' : '#374151'
                      }}>
                        {chartData.technical_indicators.RSI.toFixed(1)}
                      </span>
                    </div>
                    
                    <div className="indicator-row">
                      <span className="indicator-label">MACD</span>
                      <span className="indicator-value" style={{
                        color: chartData.technical_indicators.MACD > 0 ? '#16a34a' : '#dc2626'
                      }}>
                        {chartData.technical_indicators.MACD.toFixed(3)}
                      </span>
                    </div>
                    
                    <div className="indicator-row">
                      <span className="indicator-label">Williams %R</span>
                      <span className="indicator-value">
                        {chartData.technical_indicators.Williams_R.toFixed(1)}
                      </span>
                    </div>
                    
                    <div className="indicator-row">
                      <span className="indicator-label">Bollinger Position</span>
                      <span className="indicator-value">
                        {chartData.technical_indicators.BB_position.toFixed(2)}
                      </span>
                    </div>
                    
                    <div className="indicator-row">
                      <span className="indicator-label">Volatilidade</span>
                      <span className="indicator-value">
                        {(chartData.technical_indicators.volatility * 100).toFixed(2)}%
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default StockChart;