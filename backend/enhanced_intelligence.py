"""
Sistema de Inteligência Avançada para IA-Bot
Melhora significativa na precisão e riqueza dos dados
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import requests
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import ta  # Technical Analysis library

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedIntelligence:
    """Sistema de inteligência aprimorada para análise financeira"""
    
    def __init__(self):
        self.cache = {}
        self.market_regime_cache = {}
        self.performance_history = {}
        
    def detect_market_regime(self, data: pd.DataFrame) -> Dict:
        """Detecta regime de mercado atual (alta, baixa, lateral)"""
        try:
            if data.empty:
                return {'regime': 'UNKNOWN', 'confidence': 0.5}
            
            # Garantir colunas padronizadas
            data = data.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Volume': 'volume'
            })
            
            if 'close' not in data.columns:
                return {'regime': 'UNKNOWN', 'confidence': 0.5}
            
            # Calcular tendências de diferentes períodos
            returns_5d = data['close'].pct_change(5).iloc[-1] if len(data) > 5 else 0
            returns_20d = data['close'].pct_change(20).iloc[-1] if len(data) > 20 else 0
            returns_60d = data['close'].pct_change(60).iloc[-1] if len(data) > 60 else 0
            
            # Volatilidade realizada
            volatility = data['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) if len(data) > 20 else 0
            
            # Volume médio vs atual
            volume_ratio = 1.0
            if 'volume' in data.columns and len(data) > 20:
                avg_volume = data['volume'].rolling(20).mean().iloc[-1]
                current_volume = data['volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Determinar regime
            if returns_20d > 0.05 and returns_60d > 0.1 and volatility < 0.3:
                regime = "BULL_MARKET"
                confidence = 0.85
            elif returns_20d < -0.05 and returns_60d < -0.1:
                regime = "BEAR_MARKET" 
                confidence = 0.80
            elif abs(returns_20d) < 0.03 and volatility < 0.2:
                regime = "SIDEWAYS"
                confidence = 0.70
            elif volatility > 0.4:
                regime = "HIGH_VOLATILITY"
                confidence = 0.75
            else:
                regime = "TRANSITIONAL"
                confidence = 0.60
                
            return {
                'regime': regime,
                'confidence': confidence,
                'volatility': float(volatility),
                'volume_ratio': float(volume_ratio),
                'momentum_5d': float(returns_5d),
                'momentum_20d': float(returns_20d),
                'momentum_60d': float(returns_60d)
            }
            
        except Exception as e:
            logger.warning(f"Erro na detecção de regime: {e}")
            return {
                'regime': 'UNKNOWN',
                'confidence': 0.5,
                'volatility': 0.0,
                'volume_ratio': 1.0
            }
    
    def get_fundamental_data(self, ticker: str) -> Dict:
        """Obtém dados fundamentalistas enriquecidos"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Dados fundamentais relevantes
            fundamentals = {
                'pe_ratio': info.get('trailingPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'roe': info.get('returnOnEquity', 0),
                'profit_margin': info.get('profitMargins', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'free_cashflow': info.get('freeCashflow', 0),
                'beta': info.get('beta', 1.0)
            }
            
            # Score fundamentalista
            score = 0
            if fundamentals['pe_ratio'] and 5 <= fundamentals['pe_ratio'] <= 25:
                score += 1
            if fundamentals['roe'] and fundamentals['roe'] > 0.15:
                score += 1  
            if fundamentals['debt_to_equity'] and fundamentals['debt_to_equity'] < 0.5:
                score += 1
            if fundamentals['revenue_growth'] and fundamentals['revenue_growth'] > 0.05:
                score += 1
                
            fundamentals['fundamental_score'] = score / 4  # Normalizado 0-1
            
            return fundamentals
            
        except Exception as e:
            logger.warning(f"Erro ao obter dados fundamentais: {e}")
            return {'fundamental_score': 0.5}
    
    def calculate_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula features avançadas para ML"""
        try:
            # Garantir colunas padronizadas
            data = data.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Volume': 'volume'
            })
            
            if any(col not in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                return data  # Retorna dados originais se não tem colunas necessárias
            
            # Features de price action
            data['price_range'] = (data['high'] - data['low']) / data['close']
            data['gap_up'] = ((data['open'] - data['close'].shift(1)) / data['close'].shift(1)).clip(lower=0)
            data['gap_down'] = ((data['close'].shift(1) - data['open']) / data['close'].shift(1)).clip(lower=0)
            
            # Features de volume
            data['volume_sma'] = data['volume'].rolling(20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma']
            data['price_volume'] = data['close'] * data['volume']
            data['vwap'] = data['price_volume'].cumsum() / data['volume'].cumsum()
            
            # Features de volatilidade
            data['volatility_10'] = data['close'].pct_change().rolling(10).std() * np.sqrt(252)
            data['volatility_30'] = data['close'].pct_change().rolling(30).std() * np.sqrt(252)
            data['volatility_ratio'] = data['volatility_10'] / data['volatility_30']
            
            # Features de momentum multi-timeframe
            for period in [3, 5, 10, 15, 20, 30]:
                data[f'momentum_{period}'] = data['close'].pct_change(period)
                data[f'rsi_{period}'] = ta.momentum.RSIIndicator(data['close'], window=period).rsi()
            
            # Features de mean reversion
            for window in [10, 20, 50]:
                sma = data['close'].rolling(window).mean()
                data[f'mean_reversion_{window}'] = (data['close'] - sma) / sma
                
            # Features de breakout
            data['breakout_high'] = data['high'] / data['high'].rolling(20).max()
            data['breakout_low'] = data['low'] / data['low'].rolling(20).min()
            
            # Features sazonais
            data['day_of_week'] = data.index.dayofweek
            data['month'] = data.index.month
            data['quarter'] = data.index.quarter
            
            # Features de correlação com mercado (simulado)
            market_proxy = data['close'].rolling(5).mean().pct_change()
            data['beta_5d'] = data['close'].pct_change().rolling(20).corr(market_proxy)
            
            return data
            
        except Exception as e:
            logger.warning(f"Erro no cálculo de features avançadas: {e}")
            return data
    
    def calculate_dynamic_support_resistance(self, data: pd.DataFrame) -> Dict:
        """Calcula suporte e resistência dinâmicos usando ML"""
        try:
            # Usar pivot points e clusters de preços
            highs = data['high'].rolling(5).max()
            lows = data['low'].rolling(5).min()
            
            # Identificar níveis de concentração de preços
            price_range = np.linspace(data['low'].min(), data['high'].max(), 50)
            price_density = []
            
            for price in price_range:
                density = np.sum((data['close'] >= price * 0.99) & (data['close'] <= price * 1.01))
                price_density.append(density)
            
            price_density = np.array(price_density)
            
            # Encontrar picos (suporte/resistência)
            from scipy import signal
            peaks, _ = signal.find_peaks(price_density, height=np.mean(price_density))
            
            support_resistance = []
            for peak in peaks:
                level = price_range[peak]
                strength = price_density[peak] / np.max(price_density)
                
                # Classificar como suporte ou resistência baseado na posição atual
                current_price = data['close'].iloc[-1]
                if level < current_price:
                    sr_type = "SUPPORT"
                else:
                    sr_type = "RESISTANCE"
                    
                support_resistance.append({
                    'level': float(level),
                    'type': sr_type,
                    'strength': float(strength),
                    'distance_pct': float(abs(level - current_price) / current_price * 100)
                })
            
            # Ordenar por proximidade do preço atual
            support_resistance.sort(key=lambda x: x['distance_pct'])
            
            return {
                'levels': support_resistance[:6],  # Top 6 níveis
                'current_price': float(data['close'].iloc[-1])
            }
            
        except Exception as e:
            logger.warning(f"Erro no cálculo de S/R: {e}")
            return {'levels': [], 'current_price': float(data['close'].iloc[-1])}
    
    def analyze_price_patterns(self, data: pd.DataFrame) -> Dict:
        """Analisa padrões de preço usando reconhecimento de padrões"""
        try:
            patterns = {}
            
            # Padrão Doji
            body_size = abs(data['close'] - data['open']) / (data['high'] - data['low'])
            doji_threshold = 0.1
            patterns['doji_count'] = int(np.sum(body_size.tail(10) < doji_threshold))
            
            # Padrão Hammer/Hanging Man
            lower_shadow = data['open'] - data['low']
            upper_shadow = data['high'] - data['close']
            body = abs(data['close'] - data['open'])
            
            hammer_condition = (lower_shadow > 2 * body) & (upper_shadow < body)
            patterns['hammer_count'] = int(np.sum(hammer_condition.tail(10)))
            
            # Padrão Engolfo
            engulfing = ((data['close'] > data['open']) & 
                        (data['close'].shift(1) < data['open'].shift(1)) &
                        (data['close'] > data['open'].shift(1)) &
                        (data['open'] < data['close'].shift(1)))
            patterns['bullish_engulfing'] = int(np.sum(engulfing.tail(10)))
            
            # Gaps
            gap_up = data['open'] > data['close'].shift(1) * 1.01
            gap_down = data['open'] < data['close'].shift(1) * 0.99
            patterns['gaps_up'] = int(np.sum(gap_up.tail(20)))
            patterns['gaps_down'] = int(np.sum(gap_down.tail(20)))
            
            # Volume patterns
            above_avg_volume = data['volume'] > data['volume'].rolling(20).mean()
            patterns['high_volume_days'] = int(np.sum(above_avg_volume.tail(10)))
            
            # Padrão de força/fraqueza
            strong_closes = data['close'] > (data['high'] + data['low']) / 2
            patterns['strong_closes'] = int(np.sum(strong_closes.tail(10)))
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Erro na análise de padrões: {e}")
            return {}
    
    def calculate_smart_confidence(self, data: pd.DataFrame, predictions: List, 
                                 market_regime: Dict, fundamentals: Dict) -> Dict:
        """Calcula confiança inteligente baseada em múltiplos fatores"""
        try:
            confidence_factors = []
            
            # Fator 1: Consistência dos dados (20%)
            data_quality = 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
            confidence_factors.append(('data_quality', data_quality * 0.2))
            
            # Fator 2: Regime de mercado (15%)
            regime_confidence = market_regime.get('confidence', 0.5)
            confidence_factors.append(('market_regime', regime_confidence * 0.15))
            
            # Fator 3: Volatilidade (10%)
            volatility = market_regime.get('volatility', 0.3)
            vol_score = max(0, 1 - volatility)  # Menor volatilidade = maior confiança
            confidence_factors.append(('volatility', vol_score * 0.1))
            
            # Fator 4: Volume confirmation (10%)
            volume_ratio = market_regime.get('volume_ratio', 1.0)
            vol_confirmation = min(1.0, volume_ratio / 2)  # Volume alto = mais confiança
            confidence_factors.append(('volume', vol_confirmation * 0.1))
            
            # Fator 5: Fundamentals (15%)
            fund_score = fundamentals.get('fundamental_score', 0.5)
            confidence_factors.append(('fundamentals', fund_score * 0.15))
            
            # Fator 6: Indicadores técnicos (20%)
            rsi = data['rsi'].iloc[-1] if 'rsi' in data.columns else 50
            rsi_score = 1 - abs(rsi - 50) / 50  # Próximo de 50 = neutro = mais confiável
            confidence_factors.append(('technical', rsi_score * 0.2))
            
            # Fator 7: Consistência das predições (10%)
            if predictions and len(predictions) > 1:
                pred_values = [p.get('predicted_price', 0) for p in predictions[:3]]
                pred_std = np.std(pred_values) / np.mean(pred_values) if np.mean(pred_values) > 0 else 1
                pred_consistency = max(0, 1 - pred_std * 5)
            else:
                pred_consistency = 0.5
            confidence_factors.append(('predictions', pred_consistency * 0.1))
            
            # Calcular confiança final
            total_confidence = sum(score for _, score in confidence_factors)
            
            # Aplicar boost baseado em condições especiais
            if market_regime.get('regime') == 'BULL_MARKET':
                total_confidence *= 1.1
            elif market_regime.get('regime') == 'HIGH_VOLATILITY':
                total_confidence *= 0.85
                
            # Normalizar entre 30% e 95%
            final_confidence = max(0.3, min(0.95, total_confidence))
            
            return {
                'confidence_percentage': int(final_confidence * 100),
                'confidence_level': self._get_confidence_level(final_confidence),
                'confidence_factors': {name: round(score, 3) for name, score in confidence_factors},
                'market_regime': market_regime.get('regime', 'UNKNOWN')
            }
            
        except Exception as e:
            logger.warning(f"Erro no cálculo de confiança: {e}")
            return {
                'confidence_percentage': 60,
                'confidence_level': 'MÉDIA',
                'confidence_factors': {},
                'market_regime': 'UNKNOWN'
            }
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Converte confiança numérica em nível"""
        if confidence >= 0.85:
            return 'MUITO_ALTA'
        elif confidence >= 0.75:
            return 'ALTA'
        elif confidence >= 0.65:
            return 'MÉDIA_ALTA'
        elif confidence >= 0.50:
            return 'MÉDIA'
        elif confidence >= 0.35:
            return 'BAIXA'
        else:
            return 'MUITO_BAIXA'
    
    def generate_trading_signals(self, data: pd.DataFrame, market_regime: Dict) -> Dict:
        """Gera sinais de trading inteligentes baseados no contexto"""
        try:
            signals = []
            
            # Signal 1: RSI divergence
            if 'rsi' in data.columns:
                rsi = data['rsi'].iloc[-1]
                if rsi > 70:
                    signals.append({'type': 'SELL', 'reason': 'RSI Overbought', 'strength': 0.7})
                elif rsi < 30:
                    signals.append({'type': 'BUY', 'reason': 'RSI Oversold', 'strength': 0.7})
            
            # Signal 2: Volume breakout
            if market_regime.get('volume_ratio', 1) > 2:
                price_change = data['close'].pct_change().iloc[-1]
                if price_change > 0.02:
                    signals.append({'type': 'BUY', 'reason': 'Volume Breakout Up', 'strength': 0.8})
                elif price_change < -0.02:
                    signals.append({'type': 'SELL', 'reason': 'Volume Breakout Down', 'strength': 0.8})
            
            # Signal 3: Moving average crossover
            if 'ma7' in data.columns and 'ma30' in data.columns:
                ma7_trend = data['ma7'].diff().iloc[-1]
                ma30_trend = data['ma30'].diff().iloc[-1]
                
                if ma7_trend > 0 and data['close'].iloc[-1] > data['ma7'].iloc[-1]:
                    signals.append({'type': 'BUY', 'reason': 'MA Golden Cross', 'strength': 0.6})
                elif ma7_trend < 0 and data['close'].iloc[-1] < data['ma7'].iloc[-1]:
                    signals.append({'type': 'SELL', 'reason': 'MA Death Cross', 'strength': 0.6})
            
            # Signal 4: Market regime context
            regime = market_regime.get('regime', 'UNKNOWN')
            if regime == 'BULL_MARKET':
                signals.append({'type': 'BUY', 'reason': 'Bull Market Regime', 'strength': 0.5})
            elif regime == 'BEAR_MARKET':
                signals.append({'type': 'SELL', 'reason': 'Bear Market Regime', 'strength': 0.5})
            
            # Consolidar sinais
            buy_signals = [s for s in signals if s['type'] == 'BUY']
            sell_signals = [s for s in signals if s['type'] == 'SELL']
            
            buy_strength = sum(s['strength'] for s in buy_signals)
            sell_strength = sum(s['strength'] for s in sell_signals)
            
            # Determinar sinal principal
            if buy_strength > sell_strength + 0.3:
                main_signal = 'BUY'
                signal_strength = min(1.0, buy_strength)
            elif sell_strength > buy_strength + 0.3:
                main_signal = 'SELL'
                signal_strength = min(1.0, sell_strength)
            else:
                main_signal = 'HOLD'
                signal_strength = 0.5
            
            return {
                'main_signal': main_signal,
                'signal_strength': round(signal_strength, 2),
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'total_signals': len(signals)
            }
            
        except Exception as e:
            logger.warning(f"Erro na geração de sinais: {e}")
            return {
                'main_signal': 'HOLD',
                'signal_strength': 0.5,
                'buy_signals': [],
                'sell_signals': [],
                'total_signals': 0
            }

    def generate_analysis_summary(self, analysis_data: Dict) -> str:
        """Gera um resumo explicativo da análise da IA"""
        try:
            # Extrair dados principais
            analysis = analysis_data.get('analysis', {})
            indicators = analysis_data.get('indicators', {})
            market_intel = analysis_data.get('market_intelligence', {})
            confidence_analysis = analysis_data.get('confidence_analysis', {})
            
            recommendation = analysis.get('recommendation', 'MANTER')
            confidence = analysis.get('confidence', 50)
            trend = analysis.get('trend', 'neutro')
            current_price = analysis.get('current_price', 0)
            predicted_price = analysis.get('predicted_price', 0)
            
            # Indicadores técnicos
            rsi = indicators.get('RSI', 50)
            macd = indicators.get('MACD', 0)
            bb_position = indicators.get('BB_position', 0.5)
            williams_r = indicators.get('Williams_R', -50)
            
            # Dados de mercado
            market_regime = market_intel.get('market_regime', {}).get('regime', 'SIDEWAYS')
            fundamentals = market_intel.get('fundamentals', {})
            support_resistance = market_intel.get('support_resistance', {})
            
            # Construir o resumo
            summary_parts = []
            
            # Introdução com recomendação
            if recommendation == 'COMPRAR':
                summary_parts.append(f"A IA recomenda COMPRAR com {confidence}% de confiança.")
            elif recommendation == 'VENDER':
                summary_parts.append(f"A IA recomenda VENDER com {confidence}% de confiança.")
            else:
                summary_parts.append(f"A IA recomenda MANTER posição com {confidence}% de confiança.")
            
            # Análise do preço e tendência
            price_change = ((predicted_price - current_price) / current_price * 100) if current_price > 0 else 0
            if abs(price_change) > 2:
                direction = "alta" if price_change > 0 else "baixa"
                summary_parts.append(f"O modelo projeta uma {direction} de {abs(price_change):.1f}% no preço.")
            else:
                summary_parts.append("O modelo projeta estabilidade no preço.")
            
            # Análise técnica
            technical_signals = []
            
            # RSI
            if rsi > 70:
                technical_signals.append("RSI indica sobrecompra (acima de 70)")
            elif rsi < 30:
                technical_signals.append("RSI indica sobrevenda (abaixo de 30)")
            elif 45 <= rsi <= 55:
                technical_signals.append("RSI neutro")
            
            # MACD
            if macd > 0.1:
                technical_signals.append("MACD positivo sugere momento de alta")
            elif macd < -0.1:
                technical_signals.append("MACD negativo sugere momento de baixa")
            else:
                technical_signals.append("MACD neutro")
            
            # Bollinger Bands
            if bb_position > 0.8:
                technical_signals.append("preço próximo ao topo das Bandas de Bollinger")
            elif bb_position < 0.2:
                technical_signals.append("preço próximo ao fundo das Bandas de Bollinger")
            
            if technical_signals:
                summary_parts.append(f"Tecnicamente: {', '.join(technical_signals[:3])}.")
            
            # Regime de mercado
            if market_regime == 'BULLISH':
                summary_parts.append("O mercado está em regime de alta.")
            elif market_regime == 'BEARISH':
                summary_parts.append("O mercado está em regime de baixa.")
            else:
                summary_parts.append("O mercado está em movimento lateral.")
            
            # Análise de suporte e resistência
            levels = support_resistance.get('levels', [])
            if levels:
                support_levels = [l for l in levels if l.get('type') == 'SUPPORT']
                resistance_levels = [l for l in levels if l.get('type') == 'RESISTANCE']
                
                if support_levels and resistance_levels:
                    nearest_support = min(support_levels, key=lambda x: x.get('distance_pct', float('inf')))
                    nearest_resistance = min(resistance_levels, key=lambda x: x.get('distance_pct', float('inf')))
                    summary_parts.append(f"Próximo suporte em R${nearest_support.get('level', 0):.2f} e resistência em R${nearest_resistance.get('level', 0):.2f}.")
            
            # Análise fundamentalista
            pe_ratio = fundamentals.get('pe_ratio', 0)
            roe = fundamentals.get('roe', 0)
            dividend_yield = fundamentals.get('dividend_yield', 0)
            
            fundamental_notes = []
            if pe_ratio > 0:
                if pe_ratio < 10:
                    fundamental_notes.append("P/L baixo (possível subvalorização)")
                elif pe_ratio > 25:
                    fundamental_notes.append("P/L alto (possível sobrevalorização)")
            
            if roe > 0.15:
                fundamental_notes.append("ROE sólido (acima de 15%)")
            elif roe < 0.05:
                fundamental_notes.append("ROE baixo")
            
            if dividend_yield > 0.05:
                fundamental_notes.append(f"dividend yield atrativo ({dividend_yield*100:.1f}%)")
            
            if fundamental_notes:
                summary_parts.append(f"Fundamentalmente: {', '.join(fundamental_notes[:2])}.")
            
            # Fatores de confiança
            conf_factors = confidence_analysis.get('confidence_factors', {})
            main_factors = sorted(conf_factors.items(), key=lambda x: x[1], reverse=True)[:2]
            
            if main_factors:
                factor_names = {
                    'technical': 'análise técnica',
                    'data_quality': 'qualidade dos dados',
                    'market_regime': 'regime de mercado',
                    'fundamentals': 'fundamentos',
                    'volatility': 'volatilidade',
                    'volume': 'volume',
                    'predictions': 'previsões'
                }
                top_factors = [factor_names.get(f[0], f[0]) for f in main_factors]
                summary_parts.append(f"Os principais fatores de confiança são: {' e '.join(top_factors)}.")
            
            # Conclusão
            if confidence >= 80:
                summary_parts.append("A análise apresenta alta confiabilidade.")
            elif confidence >= 60:
                summary_parts.append("A análise apresenta confiabilidade moderada.")
            else:
                summary_parts.append("A análise requer cautela devido à menor confiabilidade.")
            
            return " ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Erro na geração do resumo da análise: {e}")
            return "Resumo indisponível devido a erro no processamento."

def enhance_analysis_data(ticker: str, base_data: Dict) -> Dict:
    """Função principal para enriquecer dados de análise"""
    try:
        intelligence = EnhancedIntelligence()
        
        # Obter dados históricos
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period='6mo')
        
        if hist_data.empty:
            return base_data
        
        # Detectar regime de mercado
        market_regime = intelligence.detect_market_regime(hist_data)
        
        # Obter fundamentals
        fundamentals = intelligence.get_fundamental_data(ticker)
        
        # Calcular features avançadas
        enhanced_data = intelligence.calculate_advanced_features(hist_data)
        
        # Calcular suporte e resistência
        support_resistance = intelligence.calculate_dynamic_support_resistance(enhanced_data)
        
        # Analisar padrões
        price_patterns = intelligence.analyze_price_patterns(enhanced_data)
        
        # Gerar sinais de trading
        trading_signals = intelligence.generate_trading_signals(enhanced_data, market_regime)
        
        # Calcular confiança inteligente
        predictions = base_data.get('prediction_data', [])
        smart_confidence = intelligence.calculate_smart_confidence(
            enhanced_data, predictions, market_regime, fundamentals
        )
        
        # Enriquecer dados base
        enhanced_base = base_data.copy()
        
        # Adicionar dados inteligentes
        enhanced_base['market_intelligence'] = {
            'market_regime': market_regime,
            'fundamentals': fundamentals,
            'support_resistance': support_resistance,
            'price_patterns': price_patterns,
            'trading_signals': trading_signals,
            'smart_confidence': smart_confidence
        }
        
        # Atualizar confiança principal
        enhanced_base['analysis']['confidence'] = smart_confidence['confidence_percentage']
        enhanced_base['confidence_analysis'] = smart_confidence
        
        # Gerar resumo da análise
        analysis_summary = intelligence.generate_analysis_summary(enhanced_base)
        enhanced_base['analysis_summary'] = analysis_summary
        
        # Adicionar metadados de inteligência
        enhanced_base['intelligence_version'] = 'v3.0_enhanced'
        enhanced_base['features'].extend([
            'Market Regime Detection',
            'Smart Confidence System',
            'Dynamic Support/Resistance',
            'Price Pattern Recognition',
            'Intelligent Trading Signals',
            'Fundamental Analysis Integration',
            'AI Analysis Summary'
        ])
        
        logger.info(f"Análise inteligente aplicada para {ticker}")
        return enhanced_base
        
    except Exception as e:
        logger.error(f"Erro na inteligência avançada para {ticker}: {e}")
        return base_data