from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
from backend.core.strategy_base import StrategyBase

@dataclass
class MLEnsembleParams:
    """Machine Learning Ensemble Strategy Parameters"""
    # Feature Parameters
    lookback_periods: List[int] = (5, 10, 20, 50)  # Multiple timeframes
    feature_window: int = 100  # Window for feature calculation
    prediction_horizon: int = 5  # Predict n periods ahead
    
    # Model Parameters
    n_estimators: int = 100
    max_depth: int = 5
    min_samples_split: int = 50
    
    # Ensemble Parameters
    confidence_threshold: float = 0.6  # Minimum ensemble confidence
    model_weights: Dict[str, float] = {
        'random_forest': 0.4,
        'gradient_boost': 0.4,
        'trend_model': 0.2
    }
    
    # Training Parameters
    train_size: float = 0.8
    retrain_interval: int = 1000  # Retrain every n periods
    min_train_samples: int = 1000
    
    # Signal Parameters
    signal_threshold: float = 0.5
    min_prediction_confidence: float = 0.6
    
class MLEnsembleStrategy(StrategyBase):
    def __init__(self,
                 name: str = "ML_Ensemble",
                 params: Optional[MLEnsembleParams] = None,
                 **kwargs):
        """
        Machine Learning Ensemble strategy combining multiple models.
        
        Args:
            name: Strategy name
            params: Strategy parameters
            **kwargs: Base strategy parameters
        """
        super().__init__(name, **kwargs)
        self.params = params or MLEnsembleParams()
        
        # Initialize models
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=self.params.n_estimators,
                max_depth=self.params.max_depth,
                min_samples_split=self.params.min_samples_split,
                random_state=42
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=self.params.n_estimators,
                max_depth=self.params.max_depth,
                min_samples_split=self.params.min_samples_split,
                random_state=42
            )
        }
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.samples_since_training = 0
        
    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features for ML models."""
        df = data.copy()
        
        # Price-based features
        for period in self.params.lookback_periods:
            # Returns
            df[f'returns_{period}'] = df['close'].pct_change(period)
            
            # Moving averages
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # Volatility
            df[f'volatility_{period}'] = df['returns_{period}'].rolling(window=period).std()
            
            # Price distance from MA
            df[f'ma_dist_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
            
        # Volume features
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Momentum features
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
        
        # Trend features
        df['trend_strength'] = self._calculate_trend_strength(df)
        
        return df
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd(self,
                       prices: pd.Series,
                       fast: int = 12,
                       slow: int = 26,
                       signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        fast_ema = prices.ewm(span=fast).mean()
        slow_ema = prices.ewm(span=slow).mean()
        macd = fast_ema - slow_ema
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
        
    def _calculate_trend_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend strength indicator."""
        df = data.copy()
        ema_short = df['close'].ewm(span=20).mean()
        ema_long = df['close'].ewm(span=50).mean()
        return (ema_short - ema_long) / ema_long
        
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training."""
        df = self._calculate_features(data)
        
        # Create target variable (1 for price increase, 0 for decrease)
        df['target'] = (df['close'].shift(-self.params.prediction_horizon) > df['close']).astype(int)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Select features
        feature_columns = [col for col in df.columns if col.startswith(('returns_', 'ma_dist_', 'volatility_', 'volume_', 'rsi', 'macd', 'trend_'))]
        X = df[feature_columns].values
        y = df['target'].values
        
        return X, y
        
    def train_models(self, data: pd.DataFrame) -> None:
        """Train all models in the ensemble."""
        X, y = self._prepare_training_data(data)
        
        if len(X) < self.params.min_train_samples:
            return
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            train_size=self.params.train_size,
            shuffle=False
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Train models
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Print model performance
            print(f"{name} Performance:")
            print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
            print(f"Precision: {precision_score(y_test, y_pred):.3f}")
            print(f"Recall: {recall_score(y_test, y_pred):.3f}\n")
            
        self.is_trained = True
        self.samples_since_training = 0
        
    def _get_ensemble_prediction(self, features: np.ndarray) -> Tuple[int, float]:
        """Get weighted prediction from all models."""
        predictions = {}
        probabilities = {}
        
        # Get predictions from each model
        scaled_features = self.scaler.transform(features)
        for name, model in self.models.items():
            predictions[name] = model.predict(scaled_features)
            probabilities[name] = model.predict_proba(scaled_features)[:, 1]
            
        # Calculate weighted ensemble prediction
        weighted_prob = sum(
            prob * self.params.model_weights[name]
            for name, prob in probabilities.items()
        )
        
        # Convert to signal
        if weighted_prob > self.params.signal_threshold + self.params.min_prediction_confidence:
            signal = 1
        elif weighted_prob < self.params.signal_threshold - self.params.min_prediction_confidence:
            signal = -1
        else:
            signal = 0
            
        return signal, weighted_prob
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on ML ensemble predictions."""
        df = data.copy()
        
        # Initialize signals
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Check if we need to train/retrain models
        if not self.is_trained or self.samples_since_training >= self.params.retrain_interval:
            self.train_models(df)
            
        # Calculate features
        df = self._calculate_features(df)
        
        # Drop NaN values
        df = df.dropna()
        
        if len(df) == 0:
            return df
            
        # Select features
        feature_columns = [col for col in df.columns if col.startswith(('returns_', 'ma_dist_', 'volatility_', 'volume_', 'rsi', 'macd', 'trend_'))]
        features = df[feature_columns].values
        
        # Get ensemble predictions
        signals = []
        probabilities = []
        for i in range(len(features)):
            signal, prob = self._get_ensemble_prediction(features[i:i+1])
            signals.append(signal)
            probabilities.append(prob)
            
        df['signal'] = signals
        df['signal_strength'] = [abs(p - 0.5) * 2 for p in probabilities]
        
        self.samples_since_training += len(df)
        
        return df
    
    def calculate_position_size(self,
                              account_balance: float,
                              current_price: float,
                              volatility: float) -> float:
        """
        Calculate position size based on prediction confidence.
        
        Args:
            account_balance: Current account balance
            current_price: Current asset price
            volatility: Current market volatility
            
        Returns:
            Position size in base currency
        """
        # Get latest signal strength
        signal_strength = self.current_signal_strength if hasattr(self, 'current_signal_strength') else 0.5
        
        # Base position size from risk parameters
        base_size = account_balance * self.risk_per_trade
        
        # Adjust for volatility
        vol_factor = 1 / (1 + volatility)
        
        # Adjust for prediction confidence
        confidence_factor = 0.5 + (signal_strength * 0.5)
        
        # Calculate final position size
        position_size = base_size * vol_factor * confidence_factor / current_price
        
        # Apply maximum position size limit
        max_size = account_balance * self.max_position_size / current_price if hasattr(self, 'max_position_size') else position_size
        position_size = min(position_size, max_size)
        
        return position_size
        
    def should_update_stops(self,
                          current_price: float,
                          position_data: Dict) -> Dict[str, float]:
        """
        Update stops based on prediction confidence and volatility.
        
        Args:
            current_price: Current asset price
            position_data: Current position information
            
        Returns:
            Dict with updated stop levels
        """
        if not position_data:
            return {}
            
        entry_price = position_data['entry_price']
        side = position_data['side']
        current_stop = position_data.get('stop_loss')
        
        # Get current market data
        if not hasattr(self, 'market_data'):
            return {}
            
        # Get volatility measure
        volatility = self.market_data['volatility_20'].iloc[-1]  # 20-period volatility
        
        updates = {}
        
        # Calculate stop distance based on volatility and prediction confidence
        confidence = self.current_signal_strength if hasattr(self, 'current_signal_strength') else 0.5
        base_stop_distance = volatility * current_price * (1 + confidence)
        
        if side == 'buy':
            # Initial stop
            if not current_stop:
                updates['stop_loss'] = entry_price - base_stop_distance
                updates['take_profit'] = entry_price + (base_stop_distance * 2)
            else:
                # Trail stop based on volatility and confidence
                new_stop = current_price - base_stop_distance
                if new_stop > current_stop:
                    updates['stop_loss'] = new_stop
                    
        else:  # sell position
            # Initial stop
            if not current_stop:
                updates['stop_loss'] = entry_price + base_stop_distance
                updates['take_profit'] = entry_price - (base_stop_distance * 2)
            else:
                # Trail stop based on volatility and confidence
                new_stop = current_price + base_stop_distance
                if new_stop < current_stop:
                    updates['stop_loss'] = new_stop
                    
        return updates
        
    def save_models(self, path: str) -> None:
        """Save trained models to disk."""
        if not self.is_trained:
            return
            
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f"{path}/{name}.joblib")
            
        # Save scaler
        joblib.dump(self.scaler, f"{path}/scaler.joblib")
        
    def load_models(self, path: str) -> None:
        """Load trained models from disk."""
        # Load models
        for name in self.models.keys():
            self.models[name] = joblib.load(f"{path}/{name}.joblib")
            
        # Load scaler
        self.scaler = joblib.load(f"{path}/scaler.joblib")
        self.is_trained = True 