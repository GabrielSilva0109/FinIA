"""
Módulo para modelos de Machine Learning aplicados em análise financeira.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


class FinancialMLModels:
    """Classe para modelos de ML financeiros."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            "XGBoost": XGBRegressor(random_state=random_state, n_estimators=100),
            "RandomForest": RandomForestRegressor(random_state=random_state, n_estimators=100),
            "GradientBoosting": GradientBoostingRegressor(random_state=random_state, n_estimators=100)
        }
        self.best_model = None
        self.best_score = float('inf')
        self.feature_importance = None
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara features para treinamento.
        
        Args:
            df: DataFrame com indicadores técnicos
        
        Returns:
            DataFrame com features preparadas
        """
        features = []
        
        # Features básicas
        if 'retorno' in df.columns:
            features.append('retorno')
        if 'volatilidade' in df.columns:
            features.append('volatilidade')
        
        # Médias móveis
        ma_cols = [col for col in df.columns if col.startswith('MA') or col.startswith('SMA')]
        features.extend(ma_cols)
        
        # Indicadores técnicos
        tech_indicators = ['RSI', 'MACD', 'MACD_Signal', '%K', '%D', 'ATR']
        for indicator in tech_indicators:
            if indicator in df.columns:
                features.append(indicator)
        
        # Bandas de Bollinger
        bb_cols = [col for col in df.columns if col.startswith('BB_')]
        features.extend(bb_cols)
        
        # Volume
        if 'Volume' in df.columns:
            features.append('Volume')
            # Features derivadas do volume
            if len(df) > 20:
                df['Volume_MA'] = df['Volume'].rolling(20).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
                features.extend(['Volume_MA', 'Volume_Ratio'])
        
        # Remover NaN e infinitos
        feature_df = df[features].replace([np.inf, -np.inf], np.nan).dropna()
        
        return feature_df
    
    def train_models(self, features: pd.DataFrame, target: pd.Series, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Treina múltiplos modelos e retorna o melhor.
        
        Args:
            features: DataFrame com features
            target: Série target
            test_size: Proporção para teste
        
        Returns:
            Dicionário com resultados do treinamento
        """
        # Preparar dados
        X = features.values
        y = target.values if isinstance(target, pd.Series) else target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        results = {}
        best_model = None
        best_score = float('inf')
        
        for name, model in self.models.items():
            try:
                # Treinar modelo
                model.fit(X_train, y_train)
                
                # Fazer predições
                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)
                
                # Calcular métricas
                train_mae = mean_absolute_error(y_train, train_preds)
                test_mae = mean_absolute_error(y_test, test_preds)
                train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
                
                results[name] = {
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'model': model
                }
                
                # Atualizar melhor modelo
                if test_mae < best_score:
                    best_score = test_mae
                    best_model = model
                    self.best_model = model
                    self.best_score = best_score
                
                logger.info(f"Modelo {name} - Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}")
                
            except Exception as e:
                logger.warning(f"Erro ao treinar modelo {name}: {e}")
                results[name] = {'error': str(e)}
        
        # Feature importance do melhor modelo
        if best_model and hasattr(best_model, 'feature_importances_'):
            self.feature_importance = dict(zip(features.columns, best_model.feature_importances_))
        
        return {
            'best_model': best_model,
            'best_score': best_score,
            'all_results': results,
            'feature_importance': self.feature_importance
        }
    
    def predict(self, features: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Faz predições usando o melhor modelo.
        
        Args:
            features: Features para predição
        
        Returns:
            Array com predições ou None se não houver modelo
        """
        if self.best_model is None:
            logger.warning("Nenhum modelo treinado disponível")
            return None
        
        try:
            return self.best_model.predict(features.values)
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return None
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Retorna a importância das features."""
        return self.feature_importance


def train_advanced_model(features: pd.DataFrame, target: pd.Series) -> Tuple[Any, float]:
    """
    Função de compatibilidade com código existente.
    
    Args:
        features: DataFrame com features
        target: Série target
    
    Returns:
        Tupla com (melhor_modelo, melhor_score)
    """
    ml_models = FinancialMLModels()
    results = ml_models.train_models(features, target)
    
    return results['best_model'], results['best_score']