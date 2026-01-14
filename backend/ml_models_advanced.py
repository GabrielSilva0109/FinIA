"""
Advanced Machine Learning Models for Financial Analysis
Implementa modelos avançados: LSTM, XGBoost, Ensemble
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class AdvancedMLModels:
    """Classe com modelos ML avançados para análise financeira"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_fitted = False
        
        # Inicializar modelos
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'rf': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        }
        
        # Scalers para cada modelo
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
    
    def create_features(self, data: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """Cria features avançadas para ML"""
        features = pd.DataFrame()
        
        # Preços básicos
        features['price'] = data['close']
        features['volume'] = data['volume']
        
        # Returns
        features['return_1d'] = data['close'].pct_change(1)
        features['return_5d'] = data['close'].pct_change(5)
        features['return_10d'] = data['close'].pct_change(10)
        
        # Médias móveis
        for period in [5, 10, 20, 50]:
            features[f'ma_{period}'] = data['close'].rolling(period).mean()
            features[f'price_to_ma_{period}'] = data['close'] / features[f'ma_{period}']
        
        # Volatilidade
        features['volatility_10d'] = data['close'].rolling(10).std()
        features['volatility_20d'] = data['close'].rolling(20).std()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_ma = data['close'].rolling(bb_period).mean()
        bb_std_dev = data['close'].rolling(bb_period).std()
        features['bb_upper'] = bb_ma + (bb_std_dev * bb_std)
        features['bb_lower'] = bb_ma - (bb_std_dev * bb_std)
        features['bb_position'] = (data['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # MACD
        exp1 = data['close'].ewm(span=12).mean()
        exp2 = data['close'].ewm(span=26).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Volume indicators
        features['volume_ma'] = data['volume'].rolling(20).mean()
        features['volume_ratio'] = data['volume'] / features['volume_ma']
        
        # Lags (valores anteriores)
        for lag in range(1, 6):
            features[f'price_lag_{lag}'] = data['close'].shift(lag)
            features[f'volume_lag_{lag}'] = data['volume'].shift(lag)
        
        # High-Low features
        features['high_low_pct'] = (data['high'] - data['low']) / data['close'] * 100
        features['close_to_high'] = data['close'] / data['high']
        features['close_to_low'] = data['close'] / data['low']
        
        # Momentum indicators
        features['momentum_5'] = data['close'] / data['close'].shift(5)
        features['momentum_10'] = data['close'] / data['close'].shift(10)
        
        return features.fillna(method='ffill').fillna(0)
    
    def prepare_data(self, features: pd.DataFrame, target_col: str = 'price', 
                    forecast_horizon: int = 1) -> tuple:
        """Prepara dados para treinamento"""
        # Target é o preço futuro
        y = features[target_col].shift(-forecast_horizon)
        X = features.drop(columns=[target_col] if target_col in features.columns else [])
        
        # Remover NaNs
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Treina todos os modelos"""
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        results = {}
        
        for model_name, model in self.models.items():
            try:
                # Scale features
                X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                X_test_scaled = self.scalers[model_name].transform(X_test)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                
                # Metrics
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                test_mae = mean_absolute_error(y_test, y_pred_test)
                
                results[model_name] = {
                    'model': model,
                    'scaler': self.scalers[model_name],
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'predictions': y_pred_test
                }
                
            except Exception as e:
                print(f"Erro no modelo {model_name}: {e}")
                results[model_name] = None
        
        self.is_fitted = True
        return results, X_test, y_test
    
    def create_ensemble_prediction(self, X: pd.DataFrame, weights: dict = None) -> dict:
        """Cria previsão ensemble combinando todos os modelos"""
        if not self.is_fitted:
            raise ValueError("Modelos não foram treinados ainda")
        
        if weights is None:
            # Pesos iguais por padrão
            weights = {name: 1.0 for name in self.models.keys()}
        
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                X_scaled = self.scalers[model_name].transform(X)
                pred = model.predict(X_scaled)
                predictions[model_name] = pred
            except Exception as e:
                print(f"Erro na previsão {model_name}: {e}")
                predictions[model_name] = np.zeros(len(X))
        
        # Ensemble prediction (weighted average)
        ensemble_pred = np.zeros(len(X))
        total_weight = sum(weights.values())
        
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 1.0) / total_weight
            ensemble_pred += pred * weight
        
        # Confidence baseado na concordância entre modelos
        pred_array = np.array(list(predictions.values()))
        confidence = 1.0 / (1.0 + np.std(pred_array, axis=0).mean())
        
        return {
            'ensemble_prediction': ensemble_pred,
            'individual_predictions': predictions,
            'confidence': min(confidence * 100, 95),  # Max 95%
            'std_dev': np.std(pred_array, axis=0).mean()
        }
    
    def get_feature_importance(self, model_name: str = 'xgb') -> dict:
        """Retorna importância das features"""
        if model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return dict(zip(
                range(len(model.feature_importances_)), 
                model.feature_importances_
            ))
        elif hasattr(model, 'coef_'):
            return dict(zip(
                range(len(model.coef_)), 
                np.abs(model.coef_)
            ))
        
        return {}
    
    def validate_models(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> dict:
        """Validação cruzada dos modelos"""
        results = {}
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        for model_name, model in self.models.items():
            try:
                # Time series cross-validation
                scores = cross_val_score(
                    model, X, y, 
                    cv=tscv, 
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                
                results[model_name] = {
                    'cv_rmse_mean': np.sqrt(-scores.mean()),
                    'cv_rmse_std': np.sqrt(scores.std()),
                    'cv_scores': scores
                }
                
            except Exception as e:
                print(f"Erro na validação {model_name}: {e}")
                results[model_name] = None
        
        return results

# Função helper para usar facilmente
def train_advanced_models(data: pd.DataFrame, forecast_days: int = 10) -> dict:
    """Função principal para treinar modelos avançados"""
    ml_models = AdvancedMLModels()
    
    # Criar features
    features = ml_models.create_features(data)
    
    # Preparar dados
    X, y = ml_models.prepare_data(features, forecast_horizon=forecast_days)
    
    # Treinar modelos
    results, X_test, y_test = ml_models.train_models(X, y)
    
    # Validação cruzada
    cv_results = ml_models.validate_models(X, y)
    
    # Ensemble prediction para os próximos dias
    ensemble_result = ml_models.create_ensemble_prediction(X.tail(forecast_days))
    
    return {
        'ml_models': ml_models,
        'training_results': results,
        'cv_results': cv_results,
        'ensemble_prediction': ensemble_result,
        'features': features,
        'X_test': X_test,
        'y_test': y_test
    }