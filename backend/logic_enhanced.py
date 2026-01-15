"""
Enhanced Financial Analysis Logic - Versão Avançada
Integra modelos ML avançados, indicadores técnicos aprimorados e sistema de confiança inteligente
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings

# Importar módulos avançados
try:
    from ml_models_advanced import AdvancedMLModels, train_advanced_models
    from advanced_indicators import AdvancedIndicators
    from intelligent_confidence import IntelligentConfidence
    from enhanced_intelligence import EnhancedIntelligence, enhance_analysis_data
    from ml_models_ultimate import AdvancedMLModels as UltimateMLModels
    ADVANCED_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Módulos avançados não encontrados: {e}")
    ADVANCED_MODULES_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFinancialAnalyzer:
    """Analisador financeiro aprimorado com IA avançada"""
    
    def __init__(self):
        self.cache = {}
        self.ml_models = AdvancedMLModels() if ADVANCED_MODULES_AVAILABLE else None
        self.confidence_system = IntelligentConfidence() if ADVANCED_MODULES_AVAILABLE else None
        
    def get_stock_data(self, ticker: str, period: str = "6mo") -> pd.DataFrame:
        """Obtém dados históricos da ação com tratamento robusto de erros"""
        try:
            logging.info(f"Buscando dados para {ticker} com período {period}")
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                logging.warning(f"Nenhum dado encontrado para {ticker}")
                return pd.DataFrame()
            
            logging.info(f"Dados obtidos para {ticker}: {len(data)} períodos")
                
            # Garantir colunas padronizadas
            data = data.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Volume': 'volume'
            })
            
            # Garantir que todas as colunas necessárias existem
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in data.columns:
                    logging.error(f"Coluna {col} não encontrada nos dados")
                    return pd.DataFrame()
            
            return data
            
        except Exception as e:
            logging.error(f"Erro ao obter dados para {ticker}: {e}")
            return pd.DataFrame()
    
    def _calculate_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores técnicos básicos"""
        data = data.copy()
        
        # Médias móveis
        for period in [7, 15, 20, 30, 50]:
            data[f'ma{period}'] = data['close'].rolling(window=period).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['close'].ewm(span=12).mean()
        exp2 = data['close'].ewm(span=26).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_ma = data['close'].rolling(bb_period).mean()
        bb_std_dev = data['close'].rolling(bb_period).std()
        data['bb_upper'] = bb_ma + (bb_std_dev * bb_std)
        data['bb_lower'] = bb_ma - (bb_std_dev * bb_std)
        data['bb_middle'] = bb_ma
        
        return data
    
    def _make_enhanced_predictions(self, data: pd.DataFrame, days_forecast: int) -> List[Dict]:
        """Faz previsões usando modelos avançados"""
        predictions = []
        
        if ADVANCED_MODULES_AVAILABLE and self.ml_models:
            try:
                # Usar modelos ML avançados
                features = self.ml_models.create_features(data)
                features_clean = features.dropna()
                
                if features_clean.empty:
                    return self._fallback_predictions(data, days_forecast)
                
                X, y = self.ml_models.prepare_data(features_clean)
                
                # Treinar modelos
                results, X_test, y_test = self.ml_models.train_models(X, y)
                
                # Gerar previsões ensemble
                last_features = features_clean.tail(1)
                if not last_features.empty:
                    ensemble_result = self.ml_models.create_ensemble_prediction(last_features)
                    
                    # Criar previsões para os próximos dias
                    current_price = data['close'].iloc[-1]
                    base_date = data.index[-1]
                    
                    # Base prediction do ensemble
                    base_prediction = ensemble_result.get('ensemble_prediction', [current_price])[0]
                    std_dev = ensemble_result.get('std_dev', current_price * 0.05)
                    
                    for i in range(1, days_forecast + 1):
                        pred_date = base_date + timedelta(days=i)
                        
                        # Usar ensemble prediction como base com variação temporal
                        trend_factor = 1 + (np.random.normal(0, 0.01) * i)  # Variação menor
                        predicted_price = float(base_prediction * trend_factor)
                        
                        # Intervalos de confiança
                        confidence_upper = float(predicted_price + (std_dev * 1.96))
                        confidence_lower = float(predicted_price - (std_dev * 1.96))
                        
                        predictions.append({
                            'date': pred_date.strftime('%Y-%m-%d'),
                            'timestamp': int(pred_date.timestamp() * 1000),
                            'predicted_price': predicted_price,
                            'confidence_upper': confidence_upper,
                            'confidence_lower': confidence_lower
                        })
                    
            except Exception as e:
                logging.warning(f"Erro nos modelos avançados: {e}")
                predictions = self._fallback_predictions(data, days_forecast)
        else:
            predictions = self._fallback_predictions(data, days_forecast)
        
        return predictions
    
    def _fallback_predictions(self, data: pd.DataFrame, days_forecast: int) -> List[Dict]:
        """Previsões de fallback usando métodos simples"""
        predictions = []
        current_price = data['close'].iloc[-1]
        base_date = data.index[-1]
        
        # Calcular tendência simples
        recent_prices = data['close'].tail(10)
        trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / len(recent_prices)
        
        for i in range(1, days_forecast + 1):
            pred_date = base_date + timedelta(days=i)
            predicted_price = current_price + (trend * i)
            
            predictions.append({
                'date': pred_date.strftime('%Y-%m-%d'),
                'timestamp': int(pred_date.timestamp() * 1000),
                'predicted_price': predicted_price,
                'confidence_upper': predicted_price * 1.1,
                'confidence_lower': predicted_price * 0.9
            })
        
        return predictions
    
    def _analyze_advanced(self, data: pd.DataFrame, predictions: List[Dict], 
                         advanced_indicators: Dict = None) -> Dict:
        """Análise avançada com múltiplos fatores"""
        current_price = data['close'].iloc[-1]
        
        # Previsão final - usar primeira predição válida
        valid_predictions = [p for p in predictions if p.get('predicted_price', 0) > 0]
        if valid_predictions:
            final_prediction = valid_predictions[0]['predicted_price']  # Primeira predição
            price_change_pct = ((final_prediction / current_price) - 1) * 100
        else:
            # Fallback: calcular baseado em tendência simples
            recent_prices = data['close'].tail(5)
            trend_change = ((recent_prices.iloc[-1] / recent_prices.iloc[0]) - 1) * 100
            final_prediction = current_price * (1 + (trend_change / 100) * 0.5)  # Metade da tendência
            price_change_pct = ((final_prediction / current_price) - 1) * 100
        
        # Análise de tendência
        trend = "positivo" if price_change_pct > 2 else "negativo" if price_change_pct < -2 else "neutro"
        
        # Recomendação baseada em múltiplos fatores
        recommendation_score = 0
        
        # Fator 1: Previsão de preço
        if price_change_pct > 5:
            recommendation_score += 2
        elif price_change_pct > 2:
            recommendation_score += 1
        elif price_change_pct < -5:
            recommendation_score -= 2
        elif price_change_pct < -2:
            recommendation_score -= 1
        
        # Fator 2: RSI
        current_rsi = data['rsi'].iloc[-1] if 'rsi' in data.columns else 50
        if current_rsi < 30:
            recommendation_score += 1  # Sobrevenda = oportunidade
        elif current_rsi > 70:
            recommendation_score -= 1  # Sobrecompra = cuidado
        
        # Fator 3: MACD
        current_macd = data['macd'].iloc[-1] if 'macd' in data.columns else 0
        if current_macd > 0:
            recommendation_score += 0.5
        else:
            recommendation_score -= 0.5
        
        # Fator 4: Indicadores avançados
        if advanced_indicators:
            for indicator_name, indicator_data in advanced_indicators.items():
                if isinstance(indicator_data, dict) and 'signal' in indicator_data:
                    signal = indicator_data['signal']
                    if 'COMPRA' in signal or 'ALTA' in signal:
                        recommendation_score += 0.5
                    elif 'VENDA' in signal or 'BAIXA' in signal:
                        recommendation_score -= 0.5
        
        # Converter score em recomendação
        if recommendation_score >= 2:
            recommendation = "COMPRA"
        elif recommendation_score >= 0.5:
            recommendation = "MANTER"
        elif recommendation_score <= -2:
            recommendation = "VENDA"
        else:
            recommendation = "MANTER"
        
        return {
            'recommendation': recommendation,
            'confidence': min(int(abs(recommendation_score) * 20 + 30), 95),
            'current_price': current_price,
            'predicted_price': final_prediction,
            'price_change_percent': price_change_pct,
            'trend': trend,
            'recommendation_score': recommendation_score
        }
    
    def _calculate_enhanced_confidence(self, data: pd.DataFrame, predictions: List[Dict], 
                                     advanced_indicators: Dict = None) -> Dict:
        """Calcula confiança usando sistema inteligente"""
        try:
            # Base confidence
            base_confidence = 60
            
            # Verificar se temos predições válidas
            valid_predictions = [p.get('predicted_price', 0) for p in predictions if p.get('predicted_price', 0) > 0]
            
            if not valid_predictions:
                return {'confidence_percentage': base_confidence, 'confidence_level': 'MÉDIA'}
            
            # Calcular fatores de confiança
            confidence_factors = []
            
            # Fator 1: Qualidade dos dados
            data_quality = min(100, len(data) / 90 * 100)  # Máximo para 90+ dias
            confidence_factors.append(data_quality)
            
            # Fator 2: Consistência das predições
            pred_std = np.std(valid_predictions[:5])  # Desvio das primeiras 5 predições
            pred_consistency = max(0, 100 - (pred_std / np.mean(valid_predictions) * 100))
            confidence_factors.append(pred_consistency)
            
            # Fator 3: Indicadores técnicos
            if not data.empty and 'rsi' in data.columns:
                latest_rsi = data['rsi'].iloc[-1]
                rsi_confidence = 100 - abs(50 - latest_rsi)  # Quanto mais próximo de 50, maior incerteza
                confidence_factors.append(rsi_confidence)
            
            # Fator 4: Volume
            if not data.empty and 'volume' in data.columns:
                volume_data = data['volume'].tail(10)
                volume_consistency = 100 - (volume_data.std() / volume_data.mean() * 50)
                confidence_factors.append(max(0, min(100, volume_consistency)))
            
            # Calcular confiança final
            final_confidence = np.mean(confidence_factors) if confidence_factors else base_confidence
            final_confidence = max(30, min(95, final_confidence))  # Entre 30-95%
            
            # Determinar nível
            if final_confidence >= 80:
                level = 'MUITO_ALTA'
            elif final_confidence >= 65:
                level = 'ALTA'
            elif final_confidence >= 50:
                level = 'MÉDIA-ALTA'
            elif final_confidence >= 35:
                level = 'MÉDIA'
            else:
                level = 'BAIXA'
            
            return {
                'confidence_percentage': int(final_confidence),
                'confidence_level': level,
                'factors_used': len(confidence_factors)
            }
            
        except Exception as e:
            logging.warning(f"Erro no cálculo de confiança: {e}")
            return {'confidence_percentage': 65, 'confidence_level': 'MÉDIA-ALTA'}
    
    def generate_enhanced_chart_data(self, ticker: str, days_forecast: int = 30) -> Dict:
        """Gera análise completa com recursos avançados"""
        try:
            logging.info(f"Iniciando análise avançada para {ticker}")
            
            # Obter dados
            data = self.get_stock_data(ticker)
            if data.empty:
                return self._create_empty_response(ticker)
            
            # Calcular indicadores básicos
            data_with_indicators = self._calculate_basic_indicators(data)
            
            # Indicadores avançados
            advanced_indicators = {}
            if ADVANCED_MODULES_AVAILABLE:
                try:
                    advanced_indicators = AdvancedIndicators().calculate_all_indicators(data_with_indicators)
                except Exception as e:
                    logging.warning(f"Erro nos indicadores avançados: {e}")
            
            # Previsões avançadas
            predictions = self._make_enhanced_predictions(data_with_indicators, days_forecast)
            
            # Análise aprimorada
            analysis = self._analyze_advanced(data_with_indicators, predictions, advanced_indicators)
            
            # Sistema de confiança
            confidence_data = self._calculate_enhanced_confidence(
                data_with_indicators, predictions, advanced_indicators
            )
            
            # Verificar se houve erro na confiança e usar fallback
            if not confidence_data or not isinstance(confidence_data, dict):
                confidence_data = {'confidence_percentage': 65, 'confidence_level': 'MÉDIA-ALTA'}
            
            # Formatação dos dados históricos (últimos 90 pontos)
            historical_data = self._format_historical_data(data_with_indicators.tail(90))
            
            # Indicadores para frontend
            indicators = self._format_indicators(data_with_indicators)
            
            # Risk management avançado
            risk_management = self._calculate_advanced_risk_management(
                data_with_indicators, analysis, confidence_data
            )
            
            logging.info(f"Análise avançada concluída para {ticker}")            
            return {
                'ticker': ticker,
                'timestamp': datetime.now().isoformat(),
                'historical_data': historical_data,
                'prediction_data': predictions,
                'analysis': analysis,
                'indicators': indicators,
                'risk_management': risk_management,
                'confidence_analysis': confidence_data,
                'advanced_indicators': self._format_advanced_indicators_summary(advanced_indicators),
                'days_forecast': days_forecast,
                'data_points': len(historical_data),
                'model_version': 'enhanced_v2.0',
                'api_version': '2.0',
                'original_ticker': ticker.replace('.SA', '') if '.SA' in ticker else ticker,
                'features': [
                    'Machine Learning Ensemble',
                    'Advanced Technical Indicators', 
                    'Intelligent Confidence System',
                    'Dynamic Risk Management',
                    'Multi-Model Predictions',
                    'Auto Brazilian Ticker Correction'
                ]
            }
            
        except Exception as e:
            logging.error(f"Erro na análise avançada para {ticker}: {e}")
            return self._create_empty_response(ticker)
    
    def _format_historical_data(self, data: pd.DataFrame) -> List[Dict]:
        """Formata dados históricos para o frontend"""
        historical_data = []
        
        for idx, row in data.iterrows():
            point = {
                'date': idx.strftime('%Y-%m-%d'),
                'timestamp': int(idx.timestamp() * 1000),
                'open': float(row.get('open', row.get('Open', 0))),
                'high': float(row.get('high', row.get('High', 0))),
                'low': float(row.get('low', row.get('Low', 0))),
                'close': float(row.get('close', row.get('Close', 0))),
                'volume': int(row.get('volume', row.get('Volume', 0))),
            }
            
            # Adicionar indicadores se disponíveis
            for indicator in ['ma7', 'ma15', 'ma30', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'bb_middle']:
                if indicator in row:
                    point[indicator] = float(row[indicator])
            
            historical_data.append(point)
        
        return historical_data
    
    def _format_indicators(self, data: pd.DataFrame) -> Dict:
        """Formata indicadores para o frontend"""
        latest = data.iloc[-1]
        
        indicators = {
            'RSI': float(latest.get('rsi', 50)),
            'MACD': float(latest.get('macd', 0)),
            'Williams_R': self._calculate_williams_r(data),
            'CCI': self._calculate_cci(data),
            'BB_position': self._calculate_bb_position(data),
            'volatility': float(data['close'].pct_change().std() * np.sqrt(252)),
            'volume': int(latest.get('volume', 0))
        }
        
        return indicators
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calcula Williams %R"""
        try:
            high_14 = data['high'].rolling(period).max().iloc[-1]
            low_14 = data['low'].rolling(period).min().iloc[-1]
            close = data['close'].iloc[-1]
            
            williams_r = -100 * (high_14 - close) / (high_14 - low_14)
            return float(williams_r)
        except:
            return -50.0
    
    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calcula Commodity Channel Index"""
        try:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            ma_typical = typical_price.rolling(period).mean()
            mean_deviation = typical_price.rolling(period).apply(
                lambda x: np.mean(np.abs(x - x.mean()))
            )
            
            cci = (typical_price.iloc[-1] - ma_typical.iloc[-1]) / (0.015 * mean_deviation.iloc[-1])
            return float(cci)
        except:
            return 0.0
    
    def _calculate_bb_position(self, data: pd.DataFrame) -> float:
        """Calcula posição dentro das Bollinger Bands"""
        try:
            latest = data.iloc[-1]
            bb_upper = latest.get('bb_upper', latest['close'] * 1.02)
            bb_lower = latest.get('bb_lower', latest['close'] * 0.98)
            close = latest['close']
            
            position = (close - bb_lower) / (bb_upper - bb_lower)
            return float(position)
        except:
            return 0.5
    
    def _calculate_advanced_risk_management(self, data: pd.DataFrame, analysis: Dict, 
                                          confidence_data: Dict) -> Dict:
        """Calcula risk management avançado"""
        current_price = data['close'].iloc[-1]
        
        # Volatilidade para ajuste dinâmico
        volatility = data['close'].pct_change().std() * np.sqrt(252)
        
        # Stop loss dinâmico baseado na volatilidade
        stop_loss_pct = max(0.03, min(0.10, volatility * 0.5))  # Entre 3% e 10%
        stop_loss = current_price * (1 - stop_loss_pct)
        
        # Take profit baseado na confiança
        confidence_pct = confidence_data.get('confidence_percentage', 50) / 100
        take_profit_pct = 0.02 + (confidence_pct * 0.03)  # Entre 2% e 5%
        take_profit = current_price * (1 + take_profit_pct)
        
        # Position size baseado na confiança e volatilidade
        base_position = 50  # 50% base
        confidence_adjustment = (confidence_pct - 0.5) * 20  # ±10%
        volatility_adjustment = -volatility * 30  # Reduzir com alta volatilidade
        
        position_size = max(10, min(80, base_position + confidence_adjustment + volatility_adjustment))
        
        # Risk/Reward ratio
        risk = abs(current_price - stop_loss) / current_price
        reward = abs(take_profit - current_price) / current_price
        risk_reward_ratio = reward / risk if risk > 0 else 1
        
        return {
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'position_size': float(position_size),
            'risk_reward_ratio': float(risk_reward_ratio),
            'volatility': float(volatility),
            'stop_loss_pct': float(stop_loss_pct * 100),
            'take_profit_pct': float(take_profit_pct * 100)
        }
    
    def _format_advanced_indicators_summary(self, advanced_indicators: Dict) -> List[Dict]:
        """Formata resumo dos indicadores avançados"""
        if not advanced_indicators:
            return []
        
        summary = []
        for name, data in advanced_indicators.items():
            if isinstance(data, dict) and 'signal' in data:
                summary.append({
                    'name': name.title(),
                    'signal': data['signal'],
                    'value': data.get('value', 'N/A'),
                    'description': self._get_indicator_description(name)
                })
        
        return summary
    
    def _get_indicator_description(self, indicator_name: str) -> str:
        """Retorna descrição do indicador"""
        descriptions = {
            'ichimoku': 'Sistema Ichimoku - Análise de tendência completa',
            'stochastic': 'Estocástico - Momentum e sobrecompra/sobrevenda',
            'adx': 'ADX - Força da tendência',
            'atr': 'ATR - Volatilidade e amplitude de movimento',
            'fibonacci': 'Fibonacci - Níveis de suporte e resistência',
            'volume': 'Análise de Volume - Confirmação de movimentos',
            'momentum': 'Indicadores de Momentum - Força direcional'
        }
        return descriptions.get(indicator_name, 'Indicador técnico')
    
    def _create_empty_response(self, ticker: str) -> Dict:
        """Cria resposta vazia para casos de erro"""
        return {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'historical_data': [],
            'prediction_data': [],
            'analysis': {
                'recommendation': 'NEUTRO',
                'confidence': 0,
                'current_price': 0,
                'predicted_price': 0,
                'price_change_percent': 0,
                'trend': 'indefinido'
            },
            'indicators': {},
            'risk_management': {},
            'confidence_analysis': {'confidence_percentage': 0, 'confidence_level': 'MUITO_BAIXA'},
            'advanced_indicators': [],
            'days_forecast': 0,
            'data_points': 0,
            'model_version': 'enhanced_v2.0',
            'error': 'Dados não encontrados'
        }

# Função wrapper principal para compatibilidade e robustez
def generate_chart_data(ticker: str, days_forecast: int = 30) -> Dict:
    """
    Função principal unificada para análise financeira completa e robusta.
    
    Esta função integra:
    - Machine Learning Ensemble (XGBoost, Random Forest, Gradient Boosting)
    - Indicadores Técnicos Avançados (Ichimoku, Stochastic, ADX, ATR, etc)
    - Sistema de Confiança Inteligente Multi-fator
    - Risk Management Dinâmico
    - Previsões Estabilizadas
    
    Args:
        ticker (str): Símbolo da ação (ex: BBAS3.SA, AAPL)
        days_forecast (int): Dias de previsão (1-60)
    
    Returns:
        Dict: Análise completa com dados históricos, previsões e indicadores
    """
    try:
        analyzer = EnhancedFinancialAnalyzer()
        return analyzer.generate_enhanced_chart_data(ticker, days_forecast)
    except Exception as e:
        logging.error(f"Erro na análise de {ticker}: {e}")
        # Retorno robusto em caso de erro
        return EnhancedFinancialAnalyzer()._create_empty_response(ticker)

def generate_intelligent_analysis(ticker: str, days_forecast: int = 3) -> Dict:
    """
    Análise inteligente com IA avançada v3.0
    
    Esta função integra todos os sistemas de inteligência:
    - Machine Learning Ultimate com LSTM + Prophet + Ensemble
    - Detecção automática de regime de mercado
    - Sistema de confiança multi-fatorial inteligente
    - Análise fundamentalista integrada
    - Reconhecimento de padrões avançado
    - Sinais de trading contextuais
    
    Args:
        ticker (str): Símbolo da ação
        days_forecast (int): Dias de previsão
        
    Returns:
        Dict: Análise ultra-inteligente com dados enriquecidos
    """
    try:
        # Análise base usando o sistema enhanced existente
        analyzer = EnhancedFinancialAnalyzer()
        base_analysis = analyzer.generate_enhanced_chart_data(ticker, days_forecast)
        
        # Aplicar inteligência avançada
        if ADVANCED_MODULES_AVAILABLE:
            # Sistema de ML Ultimate
            ultimate_ml = UltimateMLModels()
            
            # Obter dados para análise avançada
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period='1y')  # Mais dados para IA
            
            if not hist_data.empty:
                # Treinar modelos avançados
                ml_predictions = ultimate_ml.train_ensemble_models(hist_data, [1, 3, 5, 10])
                
                # Aplicar inteligência contextual
                intelligent_analysis = enhance_analysis_data(ticker, base_analysis)
                
                # Integrar predições ML Ultimate
                if ml_predictions:
                    for day, pred_data in ml_predictions.items():
                        # Encontrar predição correspondente na base
                        day_num = int(day.split('_')[1])
                        for i, pred in enumerate(intelligent_analysis.get('prediction_data', [])):
                            if pred.get('day') == day_num:
                                # Enriquecer com dados do ML Ultimate
                                pred['ml_ultimate'] = {
                                    'predicted_price': pred_data['predicted_price'],
                                    'confidence': pred_data['confidence'],
                                    'models_used': pred_data['models_used'],
                                    'ensemble_weights': pred_data['ensemble_weights']
                                }
                                
                                # Usar predição mais confiável
                                if pred_data['confidence'] > pred.get('confidence', 0):
                                    pred['predicted_price'] = pred_data['predicted_price']
                                    pred['confidence'] = pred_data['confidence']
                                    pred['ml_version'] = 'ultimate_v3.0'
                
                # Feature importance
                feature_importance = {}
                for day in [1, 3, 5]:
                    importance = ultimate_ml.get_feature_importance(f'day_{day}')
                    if importance:
                        feature_importance[f'day_{day}'] = importance
                
                intelligent_analysis['feature_importance'] = feature_importance
                
                # Salvar modelos para cache
                ultimate_ml.save_models(ticker.split('.')[0])
                
                # Metadados de versão
                intelligent_analysis['ai_version'] = 'ultimate_v3.0'
                intelligent_analysis['features'].extend([
                    'LSTM Neural Networks',
                    'Prophet Time Series',
                    'Auto-Hyperparameter Tuning',
                    'Feature Importance Analysis',
                    'Model Performance Tracking'
                ])
                
                logger.info(f"Análise inteligente v3.0 aplicada para {ticker}")
                return intelligent_analysis
        
        # Fallback para análise base se módulos avançados não estiverem disponíveis
        logger.warning("Módulos avançados não disponíveis, usando análise base")
        return base_analysis
        
    except Exception as e:
        logger.error(f"Erro na análise inteligente para {ticker}: {e}")
        # Fallback para análise base
        try:
            analyzer = EnhancedFinancialAnalyzer()
            return analyzer.generate_enhanced_chart_data(ticker, days_forecast)
        except Exception as fallback_error:
            logger.error(f"Erro no fallback para {ticker}: {fallback_error}")
            return EnhancedFinancialAnalyzer()._create_empty_response(ticker)
    analyzer = EnhancedFinancialAnalyzer()
    return analyzer.generate_enhanced_chart_data(ticker, days_forecast)