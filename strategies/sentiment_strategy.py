from typing import Dict, Optional, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from backend.core.strategy_base import StrategyBase

@dataclass
class SentimentParams:
    """Sentiment Analysis Strategy Parameters"""
    # Sentiment Sources
    use_news: bool = True
    use_social: bool = True
    use_market: bool = True
    
    # Sentiment Thresholds
    strong_positive_threshold: float = 0.6
    strong_negative_threshold: float = -0.6
    neutral_threshold: float = 0.2
    
    # Time Parameters
    sentiment_lookback: int = 24  # Hours to look back
    sentiment_decay: float = 0.95  # Exponential decay factor
    
    # Volume Parameters
    volume_ma_period: int = 20
    min_volume_factor: float = 1.5
    
    # Market Parameters
    market_sentiment_period: int = 14  # Period for market indicators
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    
    # Signal Parameters
    min_confidence: float = 0.6  # Minimum confidence for signals
    max_position_value: float = 0.2  # Maximum 20% of portfolio per trade
    
class SentimentStrategy(StrategyBase):
    def __init__(self,
                 name: str = "Sentiment_Analysis",
                 params: Optional[SentimentParams] = None,
                 **kwargs):
        """
        Sentiment Analysis strategy combining multiple sources.
        
        Args:
            name: Strategy name
            params: Strategy parameters
            **kwargs: Base strategy parameters
        """
        super().__init__(name, **kwargs)
        self.params = params or SentimentParams()
        self.vader = SentimentIntensityAnalyzer()
        self.active_positions: Dict[str, Dict] = {}
        
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using multiple models."""
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        vader_compound = vader_scores['compound']
        
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        
        # Combine scores (weighted average)
        combined_sentiment = (vader_compound * 0.6) + (textblob_polarity * 0.4)
        return combined_sentiment
        
    def _calculate_news_sentiment(self, news_data: List[Dict]) -> pd.Series:
        """Calculate sentiment from news articles."""
        sentiments = []
        timestamps = []
        
        for article in news_data:
            # Extract text and timestamp
            text = f"{article['title']} {article['description']}"
            timestamp = pd.Timestamp(article['published_at'])
            
            # Calculate sentiment
            sentiment = self._analyze_text_sentiment(text)
            
            # Apply source credibility weight
            credibility = article.get('source_credibility', 0.5)
            weighted_sentiment = sentiment * credibility
            
            sentiments.append(weighted_sentiment)
            timestamps.append(timestamp)
            
        # Create time series
        sentiment_series = pd.Series(sentiments, index=timestamps)
        return sentiment_series
        
    def _calculate_social_sentiment(self, social_data: List[Dict]) -> pd.Series:
        """Calculate sentiment from social media data."""
        sentiments = []
        timestamps = []
        
        for post in social_data:
            # Extract text and timestamp
            text = post['text']
            timestamp = pd.Timestamp(post['created_at'])
            
            # Calculate sentiment
            sentiment = self._analyze_text_sentiment(text)
            
            # Apply influence weight
            influence = post.get('user_influence', 0.5)
            weighted_sentiment = sentiment * influence
            
            sentiments.append(weighted_sentiment)
            timestamps.append(timestamp)
            
        # Create time series
        sentiment_series = pd.Series(sentiments, index=timestamps)
        return sentiment_series
        
    def _calculate_market_sentiment(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate sentiment from market indicators."""
        df = market_data.copy()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.params.market_sentiment_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.params.market_sentiment_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Calculate market sentiment score
        rsi_sentiment = (df['rsi'] - 50) / 50  # Normalize to [-1, 1]
        macd_sentiment = np.where(
            df['macd'] > df['signal_line'],
            (df['macd'] - df['signal_line']) / df['close'],
            -(df['signal_line'] - df['macd']) / df['close']
        )
        
        # Combine indicators
        market_sentiment = (rsi_sentiment * 0.4) + (macd_sentiment * 0.6)
        return pd.Series(market_sentiment, index=df.index)
        
    def _combine_sentiment_sources(self,
                                 news_sentiment: pd.Series,
                                 social_sentiment: pd.Series,
                                 market_sentiment: pd.Series) -> pd.Series:
        """Combine different sentiment sources with time decay."""
        # Resample all series to same frequency
        current_time = pd.Timestamp.now()
        lookback_start = current_time - pd.Timedelta(hours=self.params.sentiment_lookback)
        
        # Create time decay weights
        time_index = pd.date_range(lookback_start, current_time, freq='H')
        decay_weights = np.power(
            self.params.sentiment_decay,
            np.arange(len(time_index))[::-1]
        )
        
        # Resample and apply decay
        news = news_sentiment.reindex(time_index).fillna(0) * decay_weights
        social = social_sentiment.reindex(time_index).fillna(0) * decay_weights
        market = market_sentiment.reindex(time_index).fillna(0) * decay_weights
        
        # Weighted combination
        combined_sentiment = pd.Series(0, index=time_index)
        
        if self.params.use_news:
            combined_sentiment += news * 0.3
        if self.params.use_social:
            combined_sentiment += social * 0.3
        if self.params.use_market:
            combined_sentiment += market * 0.4
            
        return combined_sentiment
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate trading signals based on sentiment analysis."""
        # Initialize output DataFrame
        df = pd.DataFrame(index=data['market'].index)
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Calculate sentiment from different sources
        if self.params.use_news:
            news_sentiment = self._calculate_news_sentiment(data.get('news', []))
        else:
            news_sentiment = pd.Series()
            
        if self.params.use_social:
            social_sentiment = self._calculate_social_sentiment(data.get('social', []))
        else:
            social_sentiment = pd.Series()
            
        if self.params.use_market:
            market_sentiment = self._calculate_market_sentiment(data['market'])
        else:
            market_sentiment = pd.Series()
            
        # Combine sentiment sources
        combined_sentiment = self._combine_sentiment_sources(
            news_sentiment,
            social_sentiment,
            market_sentiment
        )
        
        # Volume confirmation
        volume = data['market']['volume']
        volume_ma = volume.rolling(window=self.params.volume_ma_period).mean()
        volume_confirmed = volume > volume_ma * self.params.min_volume_factor
        
        # Generate signals
        strong_positive = combined_sentiment > self.params.strong_positive_threshold
        strong_negative = combined_sentiment < self.params.strong_negative_threshold
        
        # Long signals
        long_conditions = (
            strong_positive &
            volume_confirmed &
            (abs(combined_sentiment) > self.params.min_confidence)
        )
        
        # Short signals
        short_conditions = (
            strong_negative &
            volume_confirmed &
            (abs(combined_sentiment) > self.params.min_confidence)
        )
        
        # Set signals
        df.loc[long_conditions, 'signal'] = 1
        df.loc[short_conditions, 'signal'] = -1
        
        # Calculate signal strength based on sentiment strength
        df['signal_strength'] = abs(combined_sentiment).clip(0, 1)
        
        return df
    
    def calculate_position_size(self,
                              account_balance: float,
                              current_price: float,
                              volatility: float) -> float:
        """
        Calculate position size based on sentiment strength.
        
        Args:
            account_balance: Current account balance
            current_price: Current asset price
            volatility: Current market volatility
            
        Returns:
            Position size in base currency
        """
        # Get latest signal strength
        signal_strength = self.current_signal_strength if hasattr(self, 'current_signal_strength') else 0.5
        
        # Base position size
        base_size = account_balance * self.params.max_position_value
        
        # Adjust for sentiment strength
        sentiment_factor = 0.5 + (signal_strength * 0.5)
        
        # Adjust for volatility
        vol_factor = 1 / (1 + volatility)
        
        # Calculate final position size
        position_size = base_size * sentiment_factor * vol_factor / current_price
        
        # Apply maximum position size limit
        max_size = account_balance * self.max_position_size / current_price if hasattr(self, 'max_position_size') else position_size
        position_size = min(position_size, max_size)
        
        return position_size
        
    def should_update_stops(self,
                          current_price: float,
                          position_data: Dict) -> Dict[str, float]:
        """
        Update stops based on sentiment changes.
        
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
        
        # Get current sentiment strength
        sentiment_strength = self.current_signal_strength if hasattr(self, 'current_signal_strength') else 0.5
        
        updates = {}
        
        # Calculate stop distance based on sentiment strength
        stop_distance = current_price * (0.02 + (0.02 * sentiment_strength))
        
        if side == 'buy':
            # Initial stop
            if not current_stop:
                updates['stop_loss'] = entry_price - stop_distance
                updates['take_profit'] = entry_price + (stop_distance * 2)
            else:
                # Trail stop based on sentiment
                new_stop = current_price - stop_distance
                if new_stop > current_stop:
                    updates['stop_loss'] = new_stop
                    
        else:  # sell position
            # Initial stop
            if not current_stop:
                updates['stop_loss'] = entry_price + stop_distance
                updates['take_profit'] = entry_price - (stop_distance * 2)
            else:
                # Trail stop based on sentiment
                new_stop = current_price + stop_distance
                if new_stop < current_stop:
                    updates['stop_loss'] = new_stop
                    
        return updates
        
    def update_active_positions(self,
                              position_id: str,
                              position_data: Dict) -> None:
        """
        Update active positions dictionary.
        
        Args:
            position_id: Unique position identifier
            position_data: Position information
        """
        if position_data.get('status') == 'closed':
            self.active_positions.pop(position_id, None)
        else:
            self.active_positions[position_id] = position_data 