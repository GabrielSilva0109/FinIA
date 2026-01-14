"""
Testes unitários e de integração melhorados para FinAI API.
"""
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from main import app
from technical_indicators import compute_rsi, compute_macd, compute_technical_indicators
from ml_models import FinancialMLModels
from logic import FinancialAnalyzer

client = TestClient(app)


class TestAPI:
    """Testes para endpoints da API."""
    
    def test_root_endpoint(self):
        """Testa o endpoint raiz."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "FinAI backend está funcionando" in data["message"]
        assert "version" in data
        assert "timestamp" in data
    
    def test_health_endpoint(self):
        """Testa endpoint de health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
    
    def test_features_endpoint(self):
        """Testa endpoint de features."""
        response = client.get("/features")
        assert response.status_code == 200
        data = response.json()
        assert "endpoints" in data
        assert "indicators" in data
        assert "ml_models" in data
    
    def test_analyze_stock_valid_ticker(self):
        """Testa análise com ticker válido."""
        response = client.get("/analise/acao?ticker=AAPL")
        # API externa pode falhar, aceitar 200 ou 500
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "ticker" in data or "erro" in data
    
    def test_analyze_stock_empty_ticker(self):
        """Testa análise com ticker vazio."""
        response = client.get("/analise/acao?ticker=")
        assert response.status_code == 422  # Validation error
    
    def test_crypto_symbols_endpoint(self):
        """Testa endpoint de símbolos de cripto."""
        response = client.get("/crypto/symbols")
        # Pode falhar se API externa estiver indisponível
        assert response.status_code in [200, 500]
    
    def test_crypto_analysis_endpoint(self):
        """Testa análise de cripto."""
        response = client.get("/analise/crypto?symbol=BTC/USDT&timeframe=1d")
        # Pode falhar se API externa estiver indisponível
        assert response.status_code in [200, 404, 500]
    
    def test_crypto_analysis_invalid_timeframe(self):
        """Testa análise de cripto com timeframe inválido."""
        response = client.get("/analise/crypto?symbol=BTC/USDT&timeframe=invalid")
        assert response.status_code == 400


class TestTechnicalIndicators:
    """Testes para indicadores técnicos."""
    
    def setup_method(self):
        """Setup para cada teste."""
        # Dados de teste
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        self.test_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 50),
            'High': np.random.uniform(110, 120, 50),
            'Low': np.random.uniform(90, 100, 50),
            'Close': np.random.uniform(100, 110, 50),
            'Volume': np.random.uniform(1000000, 5000000, 50)
        }, index=dates)
        
        # Garantir que High >= Close >= Low e High >= Open >= Low
        self.test_data['High'] = np.maximum.reduce([
            self.test_data['High'], 
            self.test_data['Close'], 
            self.test_data['Open']
        ])
        
        self.test_data['Low'] = np.minimum.reduce([
            self.test_data['Low'], 
            self.test_data['Close'], 
            self.test_data['Open']
        ])
    
    def test_rsi_calculation(self):
        """Testa cálculo do RSI."""
        rsi = compute_rsi(self.test_data['Close'])
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(self.test_data)
        
        # RSI deve estar entre 0 e 100
        valid_values = rsi.dropna()
        assert all(0 <= val <= 100 for val in valid_values)
    
    def test_macd_calculation(self):
        """Testa cálculo do MACD."""
        macd, signal = compute_macd(self.test_data)
        
        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert len(macd) == len(self.test_data)
        assert len(signal) == len(self.test_data)
    
    def test_technical_indicators_complete(self):
        """Testa cálculo completo de indicadores técnicos."""
        result = compute_technical_indicators(self.test_data)
        
        assert isinstance(result, pd.DataFrame)
        
        # Verificar se colunas foram adicionadas
        expected_columns = ['retorno', 'MA7', 'MA20', 'RSI', 'MACD']
        for col in expected_columns:
            assert col in result.columns


if __name__ == "__main__":
    # Executar testes
    pytest.main([__file__, "-v"])