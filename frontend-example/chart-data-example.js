// Exemplo de como consumir a nova API de dados para componentes de gráfico ShadCN
// Esta API retorna apenas dados estruturados, sem imagens

const BACKEND_URL = 'http://localhost:8000';

/**
 * Busca dados de análise técnica formatados para componentes de gráfico
 * @param {string} ticker - Símbolo da ação (ex: 'AAPL', 'MSFT')
 * @param {number} daysForecast - Dias de previsão (padrão: 10)
 * @returns {Promise<Object>} Dados estruturados para gráficos
 */
async function getChartData(ticker, daysForecast = 10) {
    try {
        const response = await fetch(`${BACKEND_URL}/chart-data`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                ticker: ticker,
                days_forecast: daysForecast
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Erro ao buscar dados:', error);
        throw error;
    }
}

/**
 * Formata dados para uso com bibliotecas como Recharts, Chart.js, etc.
 */
function formatForChartLibrary(apiData) {
    return {
        // Dados históricos para o gráfico principal
        historicalData: apiData.historical_data.map(point => ({
            date: point.date,
            price: point.price,
            volume: point.volume || 0,
            // Indicadores técnicos
            sma_20: point.sma_20,
            sma_50: point.sma_50,
            rsi: point.rsi,
            macd: point.macd,
            bollinger_upper: point.bollinger_upper,
            bollinger_lower: point.bollinger_lower
        })),
        
        // Previsões para destacar no gráfico
        predictions: apiData.prediction_data.map(point => ({
            date: point.date,
            price: point.price,
            isPrediction: true
        })),
        
        // Métricas para cards/widgets
        metrics: {
            currentPrice: apiData.analysis.current_price,
            priceChange: apiData.analysis.price_change,
            percentChange: apiData.analysis.percent_change,
            recommendation: apiData.analysis.recommendation,
            trend: apiData.analysis.trend
        },
        
        // Indicadores para widgets separados
        technicalIndicators: apiData.indicators.map(indicator => ({
            name: indicator.name,
            value: indicator.value,
            signal: indicator.signal
        }))
    };
}

// Exemplo de uso com async/await
async function exampleUsage() {
    try {
        console.log('🔄 Buscando dados da AAPL...');
        
        const rawData = await getChartData('AAPL', 15);
        const formattedData = formatForChartLibrary(rawData);
        
        console.log('📊 Dados históricos:', formattedData.historicalData.length, 'pontos');
        console.log('📈 Previsões:', formattedData.predictions.length, 'pontos');
        console.log('💰 Preço atual: $', formattedData.metrics.currentPrice);
        console.log('🎯 Recomendação:', formattedData.metrics.recommendation);
        
        return formattedData;
    } catch (error) {
        console.error('❌ Erro:', error.message);
    }
}

// Exemplo de estrutura para componente React
const ChartComponent = {
    data: `
        // Com os dados formatados, você pode usar em qualquer biblioteca:
        
        // 1. Para Recharts (ShadCN padrão):
        <LineChart data={formattedData.historicalData}>
            <Line dataKey="price" stroke="#8884d8" />
            <Line dataKey="sma_20" stroke="#82ca9d" />
        </LineChart>
        
        // 2. Para mostrar previsões:
        <LineChart data={[...formattedData.historicalData, ...formattedData.predictions]}>
            <Line 
                dataKey="price" 
                stroke={(entry) => entry.isPrediction ? "#ff7300" : "#8884d8"} 
            />
        </LineChart>
        
        // 3. Para cards de métricas:
        <Card>
            <CardContent>
                <p>Preço: ${formattedData.metrics.currentPrice}</p>
                <p>Mudança: {formattedData.metrics.percentChange}%</p>
                <Badge variant={formattedData.metrics.recommendation === 'COMPRA' ? 'success' : 'destructive'}>
                    {formattedData.metrics.recommendation}
                </Badge>
            </CardContent>
        </Card>
    `
};

// Exportar para uso em módulos
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { getChartData, formatForChartLibrary };
}

// Executar exemplo se rodado diretamente
if (typeof window !== 'undefined') {
    // Browser environment
    window.getChartData = getChartData;
    window.formatForChartLibrary = formatForChartLibrary;
    console.log('✅ Funções disponíveis globalmente: getChartData, formatForChartLibrary');
} else {
    // Node.js environment
    exampleUsage();
}