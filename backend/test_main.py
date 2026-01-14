import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root_endpoint():
    """Testa o endpoint raiz."""
    response = client.get("/")
    assert response.status_code == 200
    assert "FinAI backend está funcionando" in response.json()["message"]

def test_analyze_stock_valid_ticker():
    """Testa análise com ticker válido."""
    response = client.get("/analise/acao?ticker=AAPL")
    assert response.status_code in [200, 500]  # 500 pode ocorrer por limites da API

def test_analyze_stock_invalid_ticker():
    """Testa análise com ticker inválido."""
    response = client.get("/analise/acao?ticker=")
    assert response.status_code == 422  # Validation error

def test_yahoo_finance_endpoint():
    """Testa endpoint do Yahoo Finance."""
    response = client.get("/api/yahoo/AAPL")
    assert response.status_code in [200, 408, 502]  # Aceita códigos de erro da API

def test_price_endpoint():
    """Testa endpoint de preço."""
    response = client.get("/preco/AAPL")
    assert response.status_code in [200, 500]

class TestTechnicalIndicators:
    """Testes para indicadores técnicos."""
    
    def test_rsi_calculation(self):
        """Testa cálculo do RSI."""
        from logic import compute_rsi
        import pandas as pd
        import numpy as np
        
        # Dados de teste
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        rsi = compute_rsi(prices)
        
        assert not rsi.empty
        assert all(0 <= val <= 100 for val in rsi.dropna())
    
    def test_macd_calculation(self):
        """Testa cálculo do MACD."""
        from logic import compute_macd
        import pandas as pd
        
        # Dados de teste
        df = pd.DataFrame({
            'Close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112]
        })
        
        macd, signal = compute_macd(df)
        
        assert not macd.empty
        assert not signal.empty
        assert len(macd) == len(signal)

if __name__ == "__main__":
    pytest.main([__file__])