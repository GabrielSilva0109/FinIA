"""
Intelligent Confidence System for Financial Analysis
Sistema avançado para calcular confiança nas previsões baseado em múltiplos fatores
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class IntelligentConfidence:
    """Sistema inteligente de cálculo de confiança"""
    
    def __init__(self):
        self.confidence_factors = {}
        self.weights = {
            'model_agreement': 0.25,      # Concordância entre modelos
            'prediction_stability': 0.20,  # Estabilidade das previsões
            'historical_accuracy': 0.20,   # Precisão histórica
            'indicator_consensus': 0.15,   # Consenso dos indicadores
            'market_volatility': 0.10,     # Volatilidade do mercado
            'data_quality': 0.10          # Qualidade dos dados
        }
    
    def calculate_model_agreement(self, predictions: Dict) -> float:
        """
        Calcula a concordância entre diferentes modelos
        Quanto menor a divergência, maior a confiança
        """
        if len(predictions) < 2:
            return 0.5
        
        pred_values = np.array(list(predictions.values()))
        
        # Coefficient of variation (std/mean)
        cv = np.std(pred_values) / np.mean(pred_values) if np.mean(pred_values) != 0 else 1
        
        # Converter para score de confiança (0-1)
        agreement_score = 1 / (1 + cv * 2)  # Quanto menor CV, maior a confiança
        
        return min(max(agreement_score, 0), 1)
    
    def calculate_prediction_stability(self, predictions_history: List[float], 
                                     current_prediction: float) -> float:
        """
        Avalia a estabilidade das previsões ao longo do tempo
        """
        if len(predictions_history) < 5:
            return 0.6  # Confiança média para poucos dados
        
        # Calcular trend das previsões
        predictions_series = pd.Series(predictions_history + [current_prediction])
        
        # Volatilidade das previsões
        pred_volatility = predictions_series.std() / predictions_series.mean()
        
        # Tendência consistente
        trend_consistency = np.corrcoef(range(len(predictions_series)), predictions_series)[0, 1]
        trend_consistency = abs(trend_consistency) if not np.isnan(trend_consistency) else 0
        
        # Score combinado
        stability_score = (1 - min(pred_volatility, 1)) * 0.7 + trend_consistency * 0.3
        
        return min(max(stability_score, 0), 1)
    
    def calculate_historical_accuracy(self, actual_prices: List[float], 
                                    predicted_prices: List[float]) -> float:
        """
        Calcula a precisão histórica do modelo
        """
        if len(actual_prices) < 3 or len(predicted_prices) < 3:
            return 0.5
        
        # MAPE (Mean Absolute Percentage Error)
        actual = np.array(actual_prices)
        predicted = np.array(predicted_prices)
        
        # Evitar divisão por zero
        non_zero_mask = actual != 0
        actual_safe = actual[non_zero_mask]
        predicted_safe = predicted[non_zero_mask]
        
        if len(actual_safe) == 0:
            return 0.5
        
        mape = np.mean(np.abs((actual_safe - predicted_safe) / actual_safe)) * 100
        
        # Converter MAPE para score de confiança
        accuracy_score = max(0, 1 - (mape / 100))  # 0% error = 100% confidence
        
        return min(accuracy_score, 1)
    
    def calculate_indicator_consensus(self, indicators: Dict) -> float:
        """
        Avalia o consenso entre diferentes indicadores técnicos
        """
        signals = []
        
        # Extrair sinais dos indicadores
        for indicator_name, indicator_data in indicators.items():
            if isinstance(indicator_data, dict) and 'signal' in indicator_data:
                signal = indicator_data['signal']
                
                # Converter sinais para scores numéricos
                if 'COMPRA' in signal or 'ALTA' in signal or 'BULLISH' in signal:
                    signals.append(1)
                elif 'VENDA' in signal or 'BAIXA' in signal or 'BEARISH' in signal:
                    signals.append(-1)
                else:
                    signals.append(0)  # Neutro
        
        if len(signals) < 3:
            return 0.5
        
        # Calcular consenso
        signals_array = np.array(signals)
        consensus = np.abs(np.mean(signals_array))  # Quanto mais próximo de ±1, maior o consenso
        
        return min(consensus, 1)
    
    def calculate_market_volatility_factor(self, price_data: pd.Series, 
                                         period: int = 20) -> float:
        """
        Avalia o impacto da volatilidade na confiança
        Alta volatilidade = menor confiança
        """
        if len(price_data) < period:
            return 0.5
        
        # Volatilidade realizada (desvio padrão dos retornos)
        returns = price_data.pct_change().dropna()
        volatility = returns.rolling(period).std().iloc[-1]
        
        # Volatilidade anualizada
        annual_volatility = volatility * np.sqrt(252)  # 252 trading days
        
        # Score de confiança baseado na volatilidade
        # Volatilidade normal para ações: 15-25%
        if annual_volatility <= 0.15:
            volatility_score = 0.9
        elif annual_volatility <= 0.25:
            volatility_score = 0.7
        elif annual_volatility <= 0.40:
            volatility_score = 0.5
        else:
            volatility_score = 0.3
        
        return volatility_score
    
    def calculate_data_quality(self, data: pd.DataFrame) -> float:
        """
        Avalia a qualidade dos dados de entrada
        """
        quality_factors = []
        
        # Completude dos dados (% de valores não nulos)
        completeness = data.notna().mean().mean()
        quality_factors.append(completeness)
        
        # Consistência temporal (sem gaps grandes)
        if 'timestamp' in data.columns or isinstance(data.index, pd.DatetimeIndex):
            date_index = pd.to_datetime(data.index) if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data['timestamp'])
            time_gaps = date_index.diff().dt.days.fillna(0)
            max_gap = time_gaps.max()
            
            # Penalizar gaps maiores que 7 dias
            gap_score = max(0, 1 - (max_gap - 1) / 10) if max_gap > 1 else 1
            quality_factors.append(gap_score)
        
        # Volume de dados (mais dados = melhor qualidade)
        data_volume_score = min(len(data) / 100, 1)  # 100+ pontos = score máximo
        quality_factors.append(data_volume_score)
        
        # Ausência de outliers extremos
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_scores = []
        
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            outlier_ratio = len(outliers) / len(data)
            outlier_score = max(0, 1 - outlier_ratio * 5)  # Penalizar muitos outliers
            outlier_scores.append(outlier_score)
        
        if outlier_scores:
            quality_factors.append(np.mean(outlier_scores))
        
        return np.mean(quality_factors)
    
    def calculate_comprehensive_confidence(self, 
                                         predictions: Dict = None,
                                         predictions_history: List[float] = None,
                                         current_prediction: float = None,
                                         actual_prices: List[float] = None,
                                         predicted_prices: List[float] = None,
                                         indicators: Dict = None,
                                         price_data: pd.Series = None,
                                         data: pd.DataFrame = None) -> Dict:
        """
        Calcula confiança abrangente baseada em todos os fatores
        """
        confidence_scores = {}
        
        # 1. Concordância entre modelos
        if predictions:
            confidence_scores['model_agreement'] = self.calculate_model_agreement(predictions)
        else:
            confidence_scores['model_agreement'] = 0.5
        
        # 2. Estabilidade das previsões
        if predictions_history and current_prediction is not None:
            confidence_scores['prediction_stability'] = self.calculate_prediction_stability(
                predictions_history, current_prediction
            )
        else:
            confidence_scores['prediction_stability'] = 0.5
        
        # 3. Precisão histórica
        if actual_prices and predicted_prices:
            confidence_scores['historical_accuracy'] = self.calculate_historical_accuracy(
                actual_prices, predicted_prices
            )
        else:
            confidence_scores['historical_accuracy'] = 0.5
        
        # 4. Consenso dos indicadores
        if indicators:
            confidence_scores['indicator_consensus'] = self.calculate_indicator_consensus(indicators)
        else:
            confidence_scores['indicator_consensus'] = 0.5
        
        # 5. Volatilidade do mercado
        if price_data is not None:
            confidence_scores['market_volatility'] = self.calculate_market_volatility_factor(price_data)
        else:
            confidence_scores['market_volatility'] = 0.5
        
        # 6. Qualidade dos dados
        if data is not None:
            confidence_scores['data_quality'] = self.calculate_data_quality(data)
        else:
            confidence_scores['data_quality'] = 0.5
        
        # Calcular confiança final ponderada
        weighted_confidence = sum(
            confidence_scores[factor] * self.weights[factor]
            for factor in confidence_scores
        )
        
        # Converter para percentual
        final_confidence = int(weighted_confidence * 100)
        
        # Classificação qualitativa
        if final_confidence >= 80:
            confidence_level = "MUITO_ALTA"
        elif final_confidence >= 65:
            confidence_level = "ALTA"
        elif final_confidence >= 50:
            confidence_level = "MÉDIA"
        elif final_confidence >= 35:
            confidence_level = "BAIXA"
        else:
            confidence_level = "MUITO_BAIXA"
        
        return {
            'confidence_percentage': final_confidence,
            'confidence_level': confidence_level,
            'confidence_factors': confidence_scores,
            'factor_weights': self.weights,
            'recommendations': self._generate_confidence_recommendations(confidence_scores)
        }
    
    def _generate_confidence_recommendations(self, scores: Dict) -> List[str]:
        """
        Gera recomendações baseadas nos scores de confiança
        """
        recommendations = []
        
        if scores['model_agreement'] < 0.6:
            recommendations.append("Modelos divergem - considere validação adicional")
        
        if scores['prediction_stability'] < 0.5:
            recommendations.append("Previsões instáveis - aguarde mais dados")
        
        if scores['historical_accuracy'] < 0.6:
            recommendations.append("Precisão histórica baixa - revise modelo")
        
        if scores['indicator_consensus'] < 0.5:
            recommendations.append("Indicadores divergem - análise manual necessária")
        
        if scores['market_volatility'] < 0.5:
            recommendations.append("Alta volatilidade - reduza exposição")
        
        if scores['data_quality'] < 0.7:
            recommendations.append("Qualidade dos dados questionável - verifique fontes")
        
        if not recommendations:
            recommendations.append("Confiança adequada para decisão")
        
        return recommendations

# Função helper para uso fácil
def calculate_intelligent_confidence(**kwargs) -> Dict:
    """
    Wrapper function para calcular confiança inteligente facilmente
    """
    confidence_system = IntelligentConfidence()
    return confidence_system.calculate_comprehensive_confidence(**kwargs)