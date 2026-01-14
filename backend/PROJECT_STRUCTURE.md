# 🏗️ ESTRUTURA DO PROJETO IA-BOT

## 📁 Arquivos Principais (Ativos)

### 🚀 Core Sistema
- **`main.py`** - Servidor FastAPI principal com todos os endpoints
- **`logic_enhanced.py`** - Sistema de análise financeira avançada (PRINCIPAL)
- **`config.py`** - Configurações centralizadas da aplicação

### 🤖 Machine Learning
- **`ml_models_advanced.py`** - Modelos ML avançados (XGBoost, RF, GB)
- **`intelligent_confidence.py`** - Sistema de confiança inteligente multi-fator

### 📊 Indicadores Técnicos  
- **`advanced_indicators.py`** - Indicadores técnicos avançados (Ichimoku, Stochastic, etc)

### 🪙 Criptomoedas
- **`logic_crypto.py`** - Análise especializada em criptomoedas

### 🔧 Utilitários
- **`models.py`** - Modelos Pydantic para validação da API
- **`logic.py`** - Sistema legado (mantido apenas para generate_chart)
- **`sentiment_analysis.py`** - Análise de sentimento (uso parcial)

## 📋 Dependências entre Módulos

```
main.py
├── logic_enhanced.py (PRINCIPAL)
│   ├── ml_models_advanced.py
│   ├── advanced_indicators.py  
│   └── intelligent_confidence.py
├── logic_crypto.py
├── models.py
└── config.py
```

## 🎯 Funcionalidades Principais

### 1. **Sistema Enhanced** (`logic_enhanced.py`)
- ✅ Machine Learning Ensemble (XGBoost + Random Forest + Gradient Boosting)
- ✅ Indicadores Técnicos Avançados (15+ indicadores)
- ✅ Sistema de Confiança Inteligente (multi-fator)
- ✅ Risk Management Dinâmico
- ✅ Correção Automática de Tickers Brasileiros
- ✅ Previsões para 30+ dias

### 2. **API Endpoints** (`main.py`)
- `POST /chart-data` - Análise completa e robusta (PRINCIPAL)
- `GET /analise/acao` - Análise de ação individual  
- `GET /analise/crypto` - Análise de criptomoedas
- `POST /generate-chart` - Geração de gráficos
- `GET /features` - Lista de funcionalidades

### 3. **Machine Learning** (`ml_models_advanced.py`)
- 🧠 XGBoost Regressor
- 🌳 Random Forest Regressor  
- 📈 Gradient Boosting Regressor
- 📊 Linear Regression
- 🎯 Ensemble Predictions com pesos adaptativos

## 📊 Estatísticas do Projeto

- **Arquivos ativos**: 10
- **Linhas de código**: ~3000+
- **Endpoints API**: 8
- **Modelos ML**: 4
- **Indicadores técnicos**: 15+
- **Confiança média**: 65-80%

## 🔄 Fluxo de Execução

1. **Requisição** → `main.py` 
2. **Processamento** → `logic_enhanced.py`
3. **ML Analysis** → `ml_models_advanced.py`
4. **Indicadores** → `advanced_indicators.py` 
5. **Confiança** → `intelligent_confidence.py`
6. **Resposta** → JSON estruturado

## ✅ Status de Qualidade

- **Performance**: 🟢 Otimizada
- **Precisão**: 🟢 65-80% de confiança
- **Robustez**: 🟢 Fallbacks inteligentes
- **Organização**: 🟢 Bem estruturada
- **Documentação**: 🟢 Atualizada

---

*Última atualização: Janeiro 2026*
*Versão do sistema: Enhanced v2.0*
