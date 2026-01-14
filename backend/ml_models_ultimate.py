"""
Sistema de Machine Learning Avançado v3.0
Modelos de última geração com auto-tuning e ensemble inteligente
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
import joblib
import os

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Prophet for time series
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet não disponível. Instale com: pip install prophet")

# LSTM dependencies
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')  # Reduzir logs do TensorFlow
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow não disponível para LSTM")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class AdvancedMLModels:
    """Sistema ML avançado com auto-tuning e ensemble inteligente"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.ensemble_weights = {}
        self.model_cache_dir = 'model_cache'
        self._create_cache_dir()
        
    def _create_cache_dir(self):
        """Cria diretório para cache dos modelos"""
        try:
            if not os.path.exists(self.model_cache_dir):
                os.makedirs(self.model_cache_dir)
        except Exception as e:
            logger.warning(f"Erro ao criar diretório de cache: {e}")
    
    def prepare_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepara features avançadas para ML"""
        try:
            df = data.copy()
            
            # Features básicas
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # Features de preço
            for window in [5, 10, 20, 50]:
                df[f'sma_{window}'] = df['close'].rolling(window).mean()
                df[f'price_to_sma_{window}'] = df['close'] / df[f'sma_{window}']
                df[f'sma_slope_{window}'] = df[f'sma_{window}'].diff(5)
                
            # Features de volatilidade
            df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50).mean()
            df['atr'] = self._calculate_atr(df)
            
            # Features de momentum
            for period in [5, 10, 14, 20]:
                df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / 
                                     df['close'].shift(period)) * 100
                df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
                
            # Features de volume
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['price_volume'] = df['close'] * df['volume']
            df['obv'] = self._calculate_obv(df)
            
            # Features de padrões
            df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
            df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
            df['body_size'] = np.abs(df['close'] - df['open'])
            df['range_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # Features de gap
            df['gap'] = df['open'] - df['close'].shift(1)
            df['gap_pct'] = df['gap'] / df['close'].shift(1)
            
            # Features sazonais e temporais
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['is_month_end'] = (df.index.day > 25).astype(int)
            df['is_friday'] = (df.index.dayofweek == 4).astype(int)
            
            # Features de correlação lag
            for lag in [1, 2, 3, 5]:
                df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                
            # Features de regime de mercado
            df['high_volatility'] = (df['volatility'] > df['volatility'].rolling(50).quantile(0.8)).astype(int)
            df['trending'] = (abs(df['returns'].rolling(5).mean()) > df['volatility']).astype(int)
            
            # Remover NaN
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            logger.error(f"Erro na preparação de features: {e}")
            return data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calcula RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calcula Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window).mean()
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calcula On-Balance Volume"""
        obv = np.zeros(len(df))
        obv[0] = df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv[i] = obv[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv[i] = obv[i-1] - df['volume'].iloc[i]
            else:
                obv[i] = obv[i-1]
                
        return pd.Series(obv, index=df.index)
    
    def create_lstm_model(self, input_shape: Tuple) -> Optional[object]:
        """Cria modelo LSTM para séries temporais"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        try:
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(LSTM(50, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(50))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            return model
            
        except Exception as e:
            logger.warning(f"Erro ao criar modelo LSTM: {e}")
            return None
    
    def train_prophet_model(self, data: pd.DataFrame) -> Optional[object]:
        """Treina modelo Prophet para séries temporais"""
        if not PROPHET_AVAILABLE:
            return None
            
        try:
            # Preparar dados para Prophet
            prophet_data = pd.DataFrame({
                'ds': data.index,
                'y': data['close']
            })
            
            # Configurar modelo com parâmetros otimizados
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05
            )
            
            model.fit(prophet_data)
            return model
            
        except Exception as e:
            logger.warning(f"Erro no treinamento do Prophet: {e}")
            return None
    
    def auto_tune_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                                 model_type: str) -> Dict:
        """Auto-tuning de hiperparâmetros com validação temporal"""
        try:
            tscv = TimeSeriesSplit(n_splits=3)
            
            if model_type == 'xgboost':
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.8, 1.0]
                }
                base_model = xgb.XGBRegressor(random_state=42)
                
            elif model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
                base_model = RandomForestRegressor(random_state=42)
                
            elif model_type == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0]
                }
                base_model = GradientBoostingRegressor(random_state=42)
                
            elif model_type == 'svr':
                param_grid = {
                    'C': [0.1, 1, 10],
                    'epsilon': [0.01, 0.1],
                    'kernel': ['rbf', 'linear']
                }
                base_model = SVR()
                
            else:
                return {}
            
            # Grid search com validação temporal
            grid_search = GridSearchCV(
                base_model, 
                param_grid, 
                cv=tscv, 
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            return {
                'best_params': grid_search.best_params_,
                'best_score': -grid_search.best_score_,
                'model': grid_search.best_estimator_
            }
            
        except Exception as e:
            logger.warning(f"Erro no auto-tuning para {model_type}: {e}")
            return {}
    
    def train_ensemble_models(self, data: pd.DataFrame, target_days: List[int] = [1, 3, 5]) -> Dict:
        """Treina ensemble de modelos com auto-tuning"""
        try:
            # Preparar features
            feature_data = self.prepare_advanced_features(data)
            
            # Selecionar features para ML
            feature_columns = [col for col in feature_data.columns if col not in 
                             ['open', 'high', 'low', 'close', 'volume']]
            
            X = feature_data[feature_columns].values
            predictions = {}
            
            for target_day in target_days:
                # Preparar target
                y = feature_data['close'].shift(-target_day).values
                
                # Remover NaN
                valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X_clean = X[valid_idx]
                y_clean = y[valid_idx]
                
                if len(X_clean) < 50:  # Mínimo de dados necessários
                    continue
                
                # Split temporal
                split_idx = int(len(X_clean) * 0.8)
                X_train, X_test = X_clean[:split_idx], X_clean[split_idx:]
                y_train, y_test = y_clean[:split_idx], y_clean[split_idx:]
                
                # Scaling
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Treinar modelos com auto-tuning
                models = {}
                performances = {}
                
                model_types = ['xgboost', 'random_forest', 'gradient_boosting', 'svr']
                
                for model_type in model_types:
                    tuning_result = self.auto_tune_hyperparameters(
                        X_train_scaled, y_train, model_type
                    )
                    
                    if tuning_result and 'model' in tuning_result:
                        model = tuning_result['model']
                        model.fit(X_train_scaled, y_train)
                        
                        # Avaliar performance
                        y_pred = model.predict(X_test_scaled)
                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        models[model_type] = model
                        performances[model_type] = {
                            'mse': mse,
                            'mae': mae,
                            'r2': r2,
                            'score': r2 - (mse / np.var(y_test))  # Score customizado
                        }
                
                # Treinar modelos especiais
                if TENSORFLOW_AVAILABLE and len(X_train_scaled) > 100:
                    lstm_model = self._train_lstm_model(X_train_scaled, y_train, X_test_scaled, y_test)
                    if lstm_model:
                        models['lstm'] = lstm_model
                
                if PROPHET_AVAILABLE:
                    prophet_model = self.train_prophet_model(feature_data)
                    if prophet_model:
                        models['prophet'] = prophet_model
                
                # Calcular pesos do ensemble baseado em performance
                if performances:
                    total_score = sum(p['score'] for p in performances.values())
                    weights = {model: max(0.1, p['score'] / total_score) 
                              for model, p in performances.items()}
                else:
                    weights = {model: 1/len(models) for model in models}
                
                # Fazer predições ensemble
                current_features = X_clean[-1:] 
                current_scaled = scaler.transform(current_features)
                
                ensemble_pred = 0
                total_weight = 0
                
                for model_name, model in models.items():
                    try:
                        if model_name == 'prophet':
                            # Prophet precisa de tratamento especial
                            continue  # Implementar se necessário
                        elif model_name == 'lstm':
                            if current_scaled.shape[1] >= 50:
                                pred = model.predict(current_scaled.reshape(1, 1, -1))[0][0]
                            else:
                                continue
                        else:
                            pred = model.predict(current_scaled)[0]
                        
                        weight = weights.get(model_name, 0.1)
                        ensemble_pred += pred * weight
                        total_weight += weight
                        
                    except Exception as e:
                        logger.warning(f"Erro na predição do modelo {model_name}: {e}")
                
                if total_weight > 0:
                    ensemble_pred = ensemble_pred / total_weight
                    
                    # Calcular métricas de confiança
                    recent_error = np.mean([p['mae'] for p in performances.values()]) if performances else 0
                    confidence = max(0.3, min(0.95, 1 - (recent_error / np.mean(y_test))))
                    
                    predictions[f'day_{target_day}'] = {
                        'predicted_price': float(ensemble_pred),
                        'confidence': float(confidence),
                        'models_used': list(models.keys()),
                        'ensemble_weights': weights,
                        'performance_metrics': performances
                    }
                
                # Cache modelos e scalers
                self.models[f'day_{target_day}'] = models
                self.scalers[f'day_{target_day}'] = scaler
                self.model_performance[f'day_{target_day}'] = performances
                
            return predictions
            
        except Exception as e:
            logger.error(f"Erro no treinamento do ensemble: {e}")
            return {}
    
    def _train_lstm_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_test: np.ndarray, y_test: np.ndarray) -> Optional[object]:
        """Treina modelo LSTM específico"""
        try:
            if X_train.shape[1] < 50:  # LSTM precisa de features suficientes
                return None
            
            # Reshape para LSTM
            X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
            
            model = self.create_lstm_model((1, X_train.shape[1]))
            if model is None:
                return None
            
            # Treinar com early stopping
            from tensorflow.keras.callbacks import EarlyStopping
            
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            model.fit(
                X_train_lstm, y_train,
                validation_data=(X_test_lstm, y_test),
                epochs=50,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )
            
            return model
            
        except Exception as e:
            logger.warning(f"Erro no treinamento LSTM: {e}")
            return None
    
    def get_feature_importance(self, target_day: str) -> Dict:
        """Obtém importância das features dos modelos"""
        try:
            importance_data = {}
            
            models = self.models.get(target_day, {})
            
            for model_name, model in models.items():
                if hasattr(model, 'feature_importances_'):
                    importance_data[model_name] = model.feature_importances_.tolist()
                elif hasattr(model, 'coef_'):
                    importance_data[model_name] = np.abs(model.coef_).tolist()
                    
            return importance_data
            
        except Exception as e:
            logger.warning(f"Erro ao obter feature importance: {e}")
            return {}
    
    def save_models(self, ticker: str):
        """Salva modelos treinados para reuso"""
        try:
            cache_path = os.path.join(self.model_cache_dir, f'{ticker}_models.joblib')
            
            cache_data = {
                'models': self.models,
                'scalers': self.scalers,
                'performance': self.model_performance,
                'timestamp': datetime.now()
            }
            
            joblib.dump(cache_data, cache_path)
            logger.info(f"Modelos salvos para {ticker}")
            
        except Exception as e:
            logger.warning(f"Erro ao salvar modelos: {e}")
    
    def load_models(self, ticker: str) -> bool:
        """Carrega modelos salvos se disponíveis"""
        try:
            cache_path = os.path.join(self.model_cache_dir, f'{ticker}_models.joblib')
            
            if os.path.exists(cache_path):
                cache_data = joblib.load(cache_path)
                
                # Verificar se cache não está muito antigo (1 hora)
                if datetime.now() - cache_data['timestamp'] < timedelta(hours=1):
                    self.models = cache_data['models']
                    self.scalers = cache_data['scalers']
                    self.model_performance = cache_data['performance']
                    logger.info(f"Modelos carregados para {ticker}")
                    return True
                    
            return False
            
        except Exception as e:
            logger.warning(f"Erro ao carregar modelos: {e}")
            return False