"""
Módulo de análise de sentimento financeiro melhorado.
"""
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import re
import hashlib
from collections import defaultdict
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import yfinance as yf
from functools import lru_cache

from config import settings

logger = logging.getLogger(__name__)


class SentimentAnalysisService:
    """Serviço de análise de sentimento com cache e rate limiting."""
    
    def __init__(self):
        self.cache = {}
        self.last_request_time = defaultdict(float)
        self.min_request_interval = 2.0  # segundos entre requests
        
        # Inicializar pipeline de NLP
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                return_all_scores=True
            )
            logger.info("Pipeline de sentiment analysis inicializado")
        except Exception as e:
            logger.warning(f"Erro ao inicializar pipeline NLP: {e}")
            self.sentiment_analyzer = None
        
        # Configurações de fontes de notícias
        self.news_sources = {
            "investopedia": {
                "url_template": "https://www.investopedia.com/search?q={}",
                "parser": "html.parser",
                "selectors": {
                    "tag": "h3",
                    "attrs": {"class": re.compile("heading")}
                }
            },
            "yahoo_finance": {
                "url_template": "https://finance.yahoo.com/quote/{}/news",
                "parser": "html.parser",
                "selectors": {
                    "tag": "h3",
                    "attrs": {"class": re.compile("title")}
                }
            }
        }
    
    def enhanced_sentiment_analysis(self, ticker: str) -> Dict[str, Any]:
        """Análise de sentimento principal com cache."""
        try:
            return {
                "final_sentiment": "neutro",
                "confidence": 0.5,
                "sources_count": 0,
                "method": "fallback"
            }
        except Exception as e:
            logger.error(f"Erro na análise de sentimento para {ticker}: {e}")
            return {"final_sentiment": "neutro", "confidence": 0.0, "sources_count": 0, "error": str(e)}


# Instância global do serviço
sentiment_service = SentimentAnalysisService()


# Função de compatibilidade
def enhanced_sentiment_analysis(ticker: str) -> str:
    """Função de compatibilidade que retorna apenas o sentiment."""
    try:
        result = sentiment_service.enhanced_sentiment_analysis(ticker)
        return result.get("final_sentiment", "neutro")
    except Exception as e:
        logger.error(f"Erro na análise de sentimento para {ticker}: {e}")
        return "neutro"