# 📊 IA-Bot - API de Dados para Frontend

## ✅ Implementação Concluída

Seu backend agora possui um endpoint **`/chart-data`** que retorna **apenas dados estruturados** (não imagens), perfeito para usar com componentes de gráfico do ShadCN/React.

## 🎯 Endpoint Principal

```
POST http://localhost:8000/chart-data
Content-Type: application/json

{
    "ticker": "AAPL",
    "days_forecast": 10
}
```

## 📋 Estrutura de Resposta

```json
{
  "historical_data": [
    {
      "date": "2024-01-15",
      "price": 185.5,
      "volume": 45678900,
      "sma_20": 180.25,
      "sma_50": 175.8,
      "rsi": 65.2,
      "macd": 2.15,
      "bollinger_upper": 190.5,
      "bollinger_lower": 170.3,
      "williams_r": -25.8,
      "cci": 125.5,
      "obv": 1250000000,
      "roc": 3.2,
      "vwap": 182.4
    }
    // ... mais 89 pontos históricos
  ],
  "prediction_data": [
    {
      "date": "2024-01-16",
      "price": 187.25
    }
    // ... 9 previsões futuras
  ],
  "analysis": {
    "current_price": 185.5,
    "price_change": 2.15,
    "percent_change": 1.17,
    "recommendation": "COMPRA",
    "trend": "ALTA",
    "confidence": 0.85
  },
  "indicators": [
    {
      "name": "RSI",
      "value": "65.20",
      "signal": "NEUTRO"
    },
    {
      "name": "MACD",
      "value": "2.15",
      "signal": "COMPRA"
    }
    // ... todos os indicadores técnicos
  ]
}
```

## 🛠️ Como Usar no Frontend

### 1. Fetch dos Dados

```javascript
async function getChartData(ticker = "AAPL", days = 10) {
  const response = await fetch("http://localhost:8000/chart-data", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker, days_forecast: days }),
  });
  return await response.json();
}
```

### 2. Usar com Recharts (ShadCN)

```jsx
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

function StockChart({ data }) {
  // Combinar dados históricos + previsões
  const chartData = [
    ...data.historical_data,
    ...data.prediction_data.map((p) => ({ ...p, isPrediction: true })),
  ];

  return (
    <LineChart data={chartData}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="date" />
      <YAxis />
      <Tooltip />
      <Legend />

      {/* Linha principal de preço */}
      <Line dataKey="price" stroke="#2563eb" strokeWidth={2} />

      {/* Médias móveis */}
      <Line dataKey="sma_20" stroke="#10b981" strokeDasharray="5 5" />
      <Line dataKey="sma_50" stroke="#f59e0b" strokeDasharray="5 5" />

      {/* Bollinger Bands */}
      <Line dataKey="bollinger_upper" stroke="#ef4444" strokeWidth={1} />
      <Line dataKey="bollinger_lower" stroke="#ef4444" strokeWidth={1} />
    </LineChart>
  );
}
```

### 3. Cards de Métricas

```jsx
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

function MetricsCards({ analysis }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
      <Card>
        <CardHeader>
          <CardTitle>Preço Atual</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-bold">${analysis.current_price}</p>
          <p
            className={`text-sm ${
              analysis.percent_change >= 0 ? "text-green-600" : "text-red-600"
            }`}
          >
            {analysis.percent_change >= 0 ? "+" : ""}
            {analysis.percent_change}%
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Recomendação</CardTitle>
        </CardHeader>
        <CardContent>
          <Badge
            variant={
              analysis.recommendation === "COMPRA" ? "default" : "destructive"
            }
          >
            {analysis.recommendation}
          </Badge>
          <p className="text-sm text-muted-foreground mt-2">
            Confiança: {(analysis.confidence * 100).toFixed(1)}%
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
```

## 📁 Arquivos Criados

1. **`/chart-data-example.js`** - Exemplo completo de como consumir a API
2. **`/test-api.html`** - Interface HTML para testar a API diretamente

## 🚀 Próximos Passos

1. **Iniciar o servidor**: `cd backend && python -m uvicorn main:app --reload`
2. **Testar a API**: Abrir `test-api.html` no navegador
3. **Integrar no React**: Usar os exemplos de código fornecidos
4. **Personalizar**: Adicionar mais indicadores ou modificar a análise

## 🎯 Vantagens da Nova Implementação

- ✅ **Dados estruturados** (JSON) ao invés de imagens
- ✅ **90 pontos históricos** + previsões personalizáveis
- ✅ **13 indicadores técnicos** incluídos
- ✅ **Análise com recomendação** automática
- ✅ **Compatível** com todas as bibliotecas de gráfico
- ✅ **Frontend agnóstico** - funciona com React, Vue, Angular, etc.

Agora você tem uma API robusta que retorna dados estruturados perfeitos para usar com componentes de gráfico modernos! 🎉
