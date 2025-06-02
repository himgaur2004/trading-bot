from typing import Dict, Optional, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from backend.core.strategy_base import StrategyBase

@dataclass
class MLStrategyParams:
    """Machine Learning Strategy Parameters"""
    lookback_period: int = 20
    prediction_horizon: int = 5
    min_samples: int = 1000
    feature_window: int = 10
    confidence_threshold: float = 0.6
    ensemble_size: int = 100
    volume_ma_period: int = 20
    min_volume_factor: float = 1.5
    retraining_interval: int = 100
    
class MLStrategy(StrategyBase):
    def __init__(self,
                 name: str = "ML_Strategy",
                 params: Optional[MLStrategyParams] = None,
                 **kwargs):
        """
        Machine Learning strategy using ensemble methods.
        
        Args:
            name: Strategy name
            params: Strategy parameters
            **kwargs: Base strategy parameters
        """
        super().__init__(name, **kwargs)
        self.params = params or MLStrategyParams()
        
        # Initialize models
        self.rf_model = RandomForestClassifier(
            n_estimators=self.params.ensemble_size,
            max_depth=5,
            random_state=42
        )
        
        self.gb_model = GradientBoostingClassifier(
            n_estimators=self.params.ensemble_size,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.samples_since_training = 0
        
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for machine learning."""
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # Rolling statistics
        for window in [5, 10, 20]:
            # Returns features
            df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
            df[f'returns_skew_{window}'] = df['returns'].rolling(window).skew()
            
            # Price features
            df[f'price_momentum_{window}'] = df['close'].pct_change(window)
            df[f'price_roc_{window}'] = df['close'].diff(window) / df['close'].shift(window)
            
            # Volume features
            df[f'volume_momentum_{window}'] = df['volume'].pct_change(window)
            df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
            
        # Technical indicators
        df['rsi_ratio'] = df['rsi'] / 100
        df['macd_ratio'] = (df['macd'] - df['macd_signal']) / df['close']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volatility features
        df['atr_ratio'] = df['atr'] / df['close']
        df['volatility_ratio'] = df['returns'].rolling(20).std() / df['returns'].rolling(100).std()
        
        return df
        
    def _prepare_training_data(self, data: pd.DataFrame) -> tuple:
        """Prepare data for model training."""
        # Create features
        df = self._create_features(data)
        
        # Create target variable (1 for price increase, 0 for decrease)
        df['target'] = (df['close'].shift(-self.params.prediction_horizon) > df['close']).astype(int)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Select features
        feature_columns = [col for col in df.columns if any(
            prefix in col for prefix in ['returns_', 'price_', 'volume_', 'rsi', 'macd', 'bb_', 'atr']
        )]
        
        X = df[feature_columns].values
        y = df['target'].values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        return X, y
        
    def _train_models(self, data: pd.DataFrame):
        """Train the ensemble models."""
        if len(data) < self.params.min_samples:
            return
            
        X, y = self._prepare_training_data(data)
        
        # Train models
        self.rf_model.fit(X, y)
        self.gb_model.fit(X, y)
        
        self.is_trained = True
        self.samples_since_training = 0
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using ML predictions."""
        df = data.copy()
        
        # Initialize signals
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Check if we need to train/retrain models
        self.samples_since_training += 1
        if not self.is_trained or self.samples_since_training >= self.params.retraining_interval:
            self._train_models(df)
        
        # Create features for prediction
        features_df = self._create_features(df)
        feature_columns = [col for col in features_df.columns if any(
            prefix in col for prefix in ['returns_', 'price_', 'volume_', 'rsi', 'macd', 'bb_', 'atr']
        )]
        
        # Get latest features
        latest_features = features_df[feature_columns].iloc[-1].values.reshape(1, -1)
        latest_features = self.scaler.transform(latest_features)
        
        # Get predictions from both models
        rf_pred_proba = self.rf_model.predict_proba(latest_features)[0]
        gb_pred_proba = self.gb_model.predict_proba(latest_features)[0]
        
        # Ensemble predictions with equal weights
        ensemble_proba = (rf_pred_proba + gb_pred_proba) / 2
        
        # Generate signals based on prediction confidence
        if ensemble_proba[1] > self.params.confidence_threshold:
            df.iloc[-1, df.columns.get_loc('signal')] = 1
            df.iloc[-1, df.columns.get_loc('signal_strength')] = ensemble_proba[1]
        elif ensemble_proba[0] > self.params.confidence_threshold:
            df.iloc[-1, df.columns.get_loc('signal')] = -1
            df.iloc[-1, df.columns.get_loc('signal_strength')] = ensemble_proba[0]
        
        return df
    
    def calculate_position_size(self,
                              account_balance: float,
                              current_price: float,
                              volatility: float) -> float:
        """
        Calculate position size based on prediction confidence and volatility.
        
        Args:
            account_balance: Current account balance
            current_price: Current asset price
            volatility: Current market volatility
            
        Returns:
            Position size in base currency
        """
        # Get latest signal strength (prediction confidence)
        signal_strength = self.current_signal_strength if hasattr(self, 'current_signal_strength') else 0.5
        
        # Base position size from risk parameters
        base_size = account_balance * self.risk_per_trade
        
        # Adjust for volatility
        vol_factor = 1 / (1 + volatility * 2)
        
        # Adjust for prediction confidence
        confidence_factor = signal_strength  # Direct use of model confidence
        
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
        Update stops based on model predictions and volatility.
        
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
        
        # Get current ATR if available
        atr = self.market_data['atr'].iloc[-1] if hasattr(self, 'market_data') else None
        if atr is None:
            return {}
            
        updates = {}
        
        # Get latest prediction confidence
        confidence = self.current_signal_strength if hasattr(self, 'current_signal_strength') else 0.5
        
        # Adjust stop distance based on prediction confidence
        stop_multiplier = 2 - confidence  # Tighter stops with higher confidence
        
        if side == 'buy':
            # Initial stop
            if not current_stop:
                stop_distance = atr * stop_multiplier
                updates['stop_loss'] = entry_price - stop_distance
                updates['take_profit'] = entry_price + (stop_distance * 2)
            else:
                # Trail stop based on ATR and confidence
                new_stop = current_price - (atr * stop_multiplier)
                if new_stop > current_stop:
                    updates['stop_loss'] = new_stop
                    
        else:  # sell position
            # Initial stop
            if not current_stop:
                stop_distance = atr * stop_multiplier
                updates['stop_loss'] = entry_price + stop_distance
                updates['take_profit'] = entry_price - (stop_distance * 2)
            else:
                # Trail stop based on ATR and confidence
                new_stop = current_price + (atr * stop_multiplier)
                if new_stop < current_stop:
                    updates['stop_loss'] = new_stop
                    
        return updates 