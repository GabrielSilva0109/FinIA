import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { TrendingUp, TrendingDown, BarChart3, RefreshCw, AlertCircle } from 'lucide-react';

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
        return <TrendingUp className="h-4 w-4" />;
      case 'VENDA':
      case 'VENDA FORTE':
        return <TrendingDown className="h-4 w-4" />;
      default:
        return <BarChart3 className="h-4 w-4" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <Card className="bg-white/80 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-3xl font-bold text-gray-800 flex items-center gap-3">
              <BarChart3 className="h-8 w-8 text-blue-600" />
              IA-Bot Financial Analysis
            </CardTitle>
            <CardDescription className="text-lg text-gray-600">
              Análise técnica avançada com previsões de Machine Learning
            </CardDescription>
          </CardHeader>
        </Card>

        {/* Controls */}
        <Card className="bg-white/80 backdrop-blur-sm">
          <CardContent className="p-6">
            <div className="flex flex-wrap gap-4 items-end">
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">Símbolo da Ação</label>
                <Input
                  type="text"
                  placeholder="Ex: AAPL, TSLA, MSFT..."
                  value={ticker}
                  onChange={(e) => setTicker(e.target.value.toUpperCase())}
                  className="w-40"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">Dias de Previsão</label>
                <Input
                  type="number"
                  min="1"
                  max="30"
                  value={daysForecast}
                  onChange={(e) => setDaysForecast(Number(e.target.value))}
                  className="w-32"
                />
              </div>
              <Button 
                onClick={fetchChart}
                disabled={loading}
                className="bg-blue-600 hover:bg-blue-700"
              >
                {loading ? (
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <BarChart3 className="h-4 w-4 mr-2" />
                )}
                {loading ? 'Gerando...' : 'Analisar'}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Error */}
        {error && (
          <Card className="bg-red-50 border-red-200">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 text-red-600">
                <AlertCircle className="h-5 w-5" />
                <span className="font-medium">{error}</span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Chart and Analysis */}
        {chartData && !chartData.erro && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            {/* Main Chart */}
            <div className="lg:col-span-2">
              <Card className="bg-white/90 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="text-xl">
                    {ticker} - Análise Técnica com Previsões ML
                  </CardTitle>
                  <CardDescription>
                    Últimos 3 meses + {daysForecast} dias de previsão
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="w-full">
                    <img 
                      src={chartData.chart_url}
                      alt={`Gráfico ${ticker}`}
                      className="w-full h-auto rounded-lg shadow-lg"
                    />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Analysis Panel */}
            <div className="space-y-4">
              
              {/* Recommendation */}
              <Card className="bg-white/90 backdrop-blur-sm">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg">Recomendação</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className={`flex items-center gap-3 p-4 rounded-lg ${getRecommendationColor(chartData.analysis_summary.recommendation)}`}>
                    {getRecommendationIcon(chartData.analysis_summary.recommendation)}
                    <div>
                      <div className="font-bold text-lg">
                        {chartData.analysis_summary.recommendation}
                      </div>
                      <div className="text-sm opacity-75">
                        Confiança: {chartData.analysis_summary.confidence}%
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Price Analysis */}
              <Card className="bg-white/90 backdrop-blur-sm">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg">Análise de Preços</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 bg-blue-50 rounded-lg">
                      <div className="text-sm text-gray-600">Preço Atual</div>
                      <div className="text-xl font-bold text-blue-600">
                        ${chartData.analysis_summary.current_price.toFixed(2)}
                      </div>
                    </div>
                    <div className="text-center p-3 bg-green-50 rounded-lg">
                      <div className="text-sm text-gray-600">Previsão {daysForecast}d</div>
                      <div className="text-xl font-bold text-green-600">
                        ${chartData.analysis_summary.predicted_price.toFixed(2)}
                      </div>
                    </div>
                  </div>
                  
                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <div className="text-sm text-gray-600">Variação Esperada</div>
                    <div className={`text-xl font-bold ${
                      chartData.analysis_summary.price_change_percent >= 0 
                        ? 'text-green-600' 
                        : 'text-red-600'
                    }`}>
                      {chartData.analysis_summary.price_change_percent > 0 ? '+' : ''}
                      {chartData.analysis_summary.price_change_percent.toFixed(1)}%
                    </div>
                  </div>

                  <div className="text-center p-3 bg-purple-50 rounded-lg">
                    <div className="text-sm text-gray-600">Tendência ML</div>
                    <div className="font-semibold text-purple-600">
                      {chartData.analysis_summary.trend}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Technical Indicators */}
              <Card className="bg-white/90 backdrop-blur-sm">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg">Indicadores Técnicos</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">RSI</span>
                    <span className={`font-medium ${
                      chartData.technical_indicators.RSI > 70 ? 'text-red-600' :
                      chartData.technical_indicators.RSI < 30 ? 'text-green-600' : 'text-gray-800'
                    }`}>
                      {chartData.technical_indicators.RSI.toFixed(1)}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">MACD</span>
                    <span className={`font-medium ${
                      chartData.technical_indicators.MACD > 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {chartData.technical_indicators.MACD.toFixed(3)}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Williams %R</span>
                    <span className="font-medium text-gray-800">
                      {chartData.technical_indicators.Williams_R.toFixed(1)}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Bollinger Position</span>
                    <span className="font-medium text-gray-800">
                      {chartData.technical_indicators.BB_position.toFixed(2)}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Volatilidade</span>
                    <span className="font-medium text-gray-800">
                      {(chartData.technical_indicators.volatility * 100).toFixed(2)}%
                    </span>
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