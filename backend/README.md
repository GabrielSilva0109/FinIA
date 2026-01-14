# 🚀 FinAI - Plataforma de Análise Financeira Inteligente

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Uma plataforma moderna e abrangente que utiliza **Inteligência Artificial** para análise financeira, combinando análise técnica, machine learning e análise de sentimento para fornecer insights precisos sobre investimentos.

## ✨ Funcionalidades Principais

### 📈 Análise de Ações

- **Análise Técnica Avançada**: RSI, MACD, Bandas de Bollinger, médias móveis
- **Modelos de Machine Learning**: XGBoost, Random Forest, Gradient Boosting
- **Análise de Sentimento**: Processamento de notícias financeiras
- **Alertas Inteligentes**: Sinais automáticos de compra/venda

### 🪙 Análise de Criptomoedas

- **Múltiplas Exchanges**: Binance, Coinbase Pro
- **Indicadores Especializados**: Padrões de candlestick, volume, momentum
- **Métricas de Risco**: Volatilidade, Sharpe ratio, Value at Risk
- **Dados em Tempo Real**: Integração com CoinGecko e exchanges

### 🤖 IA e Machine Learning

- **Previsões de Preços**: Modelos ensemble para predição
- **Análise de Sentimento NLP**: BERT multilíngue para notícias
- **Risk Assessment**: Cálculo automático de níveis de risco
- **Feature Engineering**: Indicadores técnicos automatizados

## 🛠️ Tecnologias

### Backend

- **FastAPI**: API REST moderna e performática
- **Python 3.9+**: Linguagem principal
- **Pandas & NumPy**: Processamento de dados financeiros
- **Scikit-learn & XGBoost**: Machine Learning
- **Transformers**: NLP e análise de sentimento
- **yfinance**: Dados de ações
- **ccxt**: Dados de criptomoedas

### Qualidade e Testes

- **Pytest**: Testes unitários e de integração
- **Docker**: Containerização
- **Logging**: Monitoramento estruturado
- **Rate Limiting**: Controle de requisições
- **Cache**: Otimização de performance

## 📦 Instalação

### Pré-requisitos

- Python 3.9+
- pip

### Setup Local

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/IA-Bot.git
cd IA-Bot/backend

# Instale as dependências
pip install -r requirements.txt

# Execute a API
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build e run com Docker
docker build -t finai-backend .
docker run -p 8000:8000 finai-backend
```

## 🚀 Uso da API

### Endpoints Principais

#### 📊 Análise de Ações

```http
GET /analise/acao?ticker=AAPL
```

**Resposta:**

```json
{
  "ticker": "AAPL",
  "preco_atual": 185.25,
  "previsao": 190.5,
  "estrategia": "Sinal de Compra (Tendência de Alta)",
  "nivel_risco": "MÉDIO",
  "indicadores": {
    "RSI": 45.2,
    "MACD": 2.1,
    "volatilidade": 0.025
  },
  "sentimento": "positivo",
  "alertas": ["📈 Volume alto: 2.3x a média"]
}
```

#### 🪙 Análise de Criptomoedas

```http
GET /analise/crypto?symbol=BTC/USDT&timeframe=1d
```

#### 📈 Múltiplas Análises

```http
GET /analise/acoes
```

### Documentação Interativa

Acesse `/docs` para Swagger UI ou `/redoc` para documentação ReDoc.

## 🧪 Testes

```bash
# Executar todos os testes
python -m pytest test_improved.py -v

# Validação completa do projeto
python validate_project.py
```

## 📁 Estrutura do Projeto

```
backend/
├── main.py                    # API principal FastAPI
├── config.py                  # Configurações centralizadas
├── models.py                  # Modelos Pydantic
├── logic.py                   # Análise principal de ações
├── logic_crypto.py           # Análise de criptomoedas
├── technical_indicators.py   # Indicadores técnicos
├── ml_models.py              # Modelos de Machine Learning
├── sentiment_analysis.py     # Análise de sentimento
├── test_improved.py          # Testes unitários
├── validate_project.py       # Script de validação
├── requirements.txt          # Dependências
├── Dockerfile               # Container Docker
└── .gitignore              # Arquivos ignorados
```

## 🎯 Exemplos de Uso

### Python Client

```python
import requests

# Analisar uma ação
response = requests.get("http://localhost:8000/analise/acao?ticker=TSLA")
data = response.json()

print(f"Preço atual: ${data['preco_atual']}")
print(f"Estratégia: {data['estrategia']}")
print(f"Risco: {data['nivel_risco']}")
```

### JavaScript/React

```javascript
const response = await fetch("/analise/crypto?symbol=ETH/USDT");
const data = await response.json();

console.log("Sinais de trading:", data.trading_signals);
console.log("Métricas de risco:", data.risk_metrics);
```

## 📊 Indicadores Suportados

### Análise Técnica

- **Momentum**: RSI, StochRSI, Williams %R, ROC
- **Tendência**: MACD, ADX, Parabolic SAR, Médias Móveis
- **Volatilidade**: Bandas de Bollinger, ATR, True Range
- **Volume**: OBV, VWAP, Chaikin Money Flow

### Padrões de Candlestick

- Hammer, Doji, Engulfing (alta/baixa)
- Morning Star, Evening Star
- Detecção automática de reversões

## 🔧 Configuração Avançada

### Variáveis de Ambiente

```bash
# config.py
LOG_LEVEL=INFO
REQUEST_TIMEOUT=10
CACHE_TTL=300
MAX_CONCURRENT_REQUESTS=10
```

### Cache e Performance

- Cache automático de resultados (5min TTL)
- Rate limiting para APIs externas
- Processamento assíncrono para múltiplas análises

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

### Guidelines

- Escreva testes para novas funcionalidades
- Mantenha cobertura de testes > 80%
- Use type hints em Python
- Siga PEP 8 para estilo de código

## 📈 Roadmap

### Próximas Versões

- [ ] **v2.0**: Interface web React
- [ ] **v2.1**: Backtesting de estratégias
- [ ] **v2.2**: Alertas via email/webhook
- [ ] **v2.3**: Suporte a forex
- [ ] **v2.4**: Portfolio tracking
- [ ] **v2.5**: Análise fundamentalista

### Melhorias Técnicas

- [ ] Redis para cache distribuído
- [ ] PostgreSQL para dados históricos
- [ ] Kubernetes deployment
- [ ] GraphQL API alternativa
- [ ] Streaming de dados real-time

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- **yfinance**: Dados de ações gratuitos
- **ccxt**: Biblioteca unificada de exchanges
- **FastAPI**: Framework web moderno
- **scikit-learn**: Machine Learning
- **Transformers**: NLP state-of-the-art

## 📞 Suporte

- 📧 Email: suporte@finai.com
- 💬 Discord: [FinAI Community](https://discord.gg/finai)
- 📖 Docs: [docs.finai.com](https://docs.finai.com)
- 🐛 Issues: [GitHub Issues](https://github.com/seu-usuario/IA-Bot/issues)

---

**Feito com ❤️ para a comunidade de investidores e desenvolvedores**
