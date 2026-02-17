"""
Sentiment Analysis Module
Analyze news and social media sentiment for trading signals
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

# Optional NLP libraries
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Sentiment Analysis for Trading
    
    Features:
    - News sentiment analysis
    - Social media sentiment
    - Sentiment scoring
    - Signal generation
    """
    
    def __init__(self):
        self.news_api_key = None
        self.twitter_api_key = None
    
    def set_news_api_key(self, api_key: str):
        """Set News API key"""
        self.news_api_key = api_key
    
    def set_twitter_api_key(self, api_key: str):
        """Set Twitter API key"""
        self.twitter_api_key = api_key
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of text
        
        Returns:
            Dictionary with sentiment scores
        """
        if not HAS_TEXTBLOB:
            logger.warning("TextBlob not installed. Install with: pip install textblob")
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'sentiment': 'neutral'
            }
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Classify sentiment
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': sentiment
        }
    
    def fetch_news(
        self,
        symbol: str,
        days: int = 7
    ) -> List[Dict]:
        """
        Fetch news articles for a symbol
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
        
        Returns:
            List of news articles
        """
        if not HAS_REQUESTS:
            logger.warning("requests not installed")
            return []
        
        # Mock news data (replace with real API)
        # Real implementation would use NewsAPI, Finnhub, or Alpha Vantage
        
        mock_news = [
            {
                'title': f'{symbol} Reports Strong Quarterly Earnings',
                'description': 'Company beats expectations with robust growth',
                'publishedAt': datetime.now() - timedelta(days=1),
                'source': 'Financial Times'
            },
            {
                'title': f'{symbol} Announces New Product Launch',
                'description': 'Innovative product expected to drive revenue',
                'publishedAt': datetime.now() - timedelta(days=2),
                'source': 'Reuters'
            },
            {
                'title': f'Analysts Upgrade {symbol} Rating',
                'description': 'Multiple analysts raise price targets',
                'publishedAt': datetime.now() - timedelta(days=3),
                'source': 'Bloomberg'
            }
        ]
        
        return mock_news
    
    def analyze_news_sentiment(
        self,
        symbol: str,
        days: int = 7
    ) -> Dict:
        """
        Analyze news sentiment for a symbol
        
        Returns:
            Aggregated sentiment metrics
        """
        news = self.fetch_news(symbol, days)
        
        if not news:
            return {
                'overall_sentiment': 'neutral',
                'polarity': 0.0,
                'article_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for article in news:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            sentiment = self.analyze_text(text)
            sentiments.append(sentiment['polarity'])
            
            if sentiment['sentiment'] == 'positive':
                positive_count += 1
            elif sentiment['sentiment'] == 'negative':
                negative_count += 1
            else:
                neutral_count += 1
        
        avg_polarity = np.mean(sentiments) if sentiments else 0.0
        
        # Determine overall sentiment
        if avg_polarity > 0.1:
            overall = 'positive'
        elif avg_polarity < -0.1:
            overall = 'negative'
        else:
            overall = 'neutral'
        
        return {
            'overall_sentiment': overall,
            'polarity': avg_polarity,
            'article_count': len(news),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'sentiment_score': (avg_polarity + 1) / 2 * 100  # Scale to 0-100
        }
    
    def get_social_sentiment(self, symbol: str) -> Dict:
        """
        Get social media sentiment
        
        Mock implementation - real version would use Twitter API
        """
        # Mock data
        return {
            'twitter_mentions': np.random.randint(1000, 50000),
            'sentiment_score': np.random.uniform(40, 80),
            'positive_ratio': np.random.uniform(0.4, 0.7),
            'engagement_rate': np.random.uniform(0.02, 0.08)
        }
    
    def generate_sentiment_signal(
        self,
        symbol: str,
        threshold_positive: float = 0.6,
        threshold_negative: float = 0.4
    ) -> Dict:
        """
        Generate trading signal based on sentiment
        
        Args:
            symbol: Stock symbol
            threshold_positive: Threshold for positive signal
            threshold_negative: Threshold for negative signal
        
        Returns:
            Trading signal dictionary
        """
        news_sentiment = self.analyze_news_sentiment(symbol)
        social_sentiment = self.get_social_sentiment(symbol)
        
        # Combined sentiment score (weighted average)
        combined_score = (
            news_sentiment['sentiment_score'] * 0.6 +
            social_sentiment['sentiment_score'] * 0.4
        )
        
        # Generate signal
        if combined_score >= threshold_positive * 100:
            signal = 'BUY'
            confidence = (combined_score - threshold_positive * 100) / (100 - threshold_positive * 100)
        elif combined_score <= threshold_negative * 100:
            signal = 'SELL'
            confidence = (threshold_negative * 100 - combined_score) / (threshold_negative * 100)
        else:
            signal = 'HOLD'
            confidence = 0.0
        
        return {
            'symbol': symbol,
            'signal': signal,
            'confidence': min(confidence, 1.0),
            'combined_score': combined_score,
            'news_sentiment': news_sentiment['overall_sentiment'],
            'news_polarity': news_sentiment['polarity'],
            'social_mentions': social_sentiment['twitter_mentions']
        }
    
    def get_sentiment_history(
        self,
        symbol: str,
        days: int = 30
    ) -> pd.DataFrame:
        """
        Get sentiment history over time
        
        Returns DataFrame with daily sentiment scores
        """
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Mock data - real implementation would fetch historical sentiment
        data = {
            'date': dates,
            'sentiment_score': np.random.uniform(40, 80, days),
            'news_volume': np.random.randint(5, 50, days),
            'social_mentions': np.random.randint(1000, 20000, days)
        }
        
        df = pd.DataFrame(data)
        
        # Add moving averages
        df['sentiment_ma7'] = df['sentiment_score'].rolling(7).mean()
        df['sentiment_ma30'] = df['sentiment_score'].rolling(30).mean()
        
        return df
    
    def detect_sentiment_divergence(
        self,
        symbol: str,
        price_data: pd.Series,
        days: int = 30
    ) -> Dict:
        """
        Detect divergence between sentiment and price
        
        Contrarian signal: price down but sentiment up (buy)
        Contrarian signal: price up but sentiment down (sell)
        """
        sentiment_history = self.get_sentiment_history(symbol, days)
        
        # Calculate trends
        price_trend = (price_data.iloc[-1] - price_data.iloc[0]) / price_data.iloc[0]
        sentiment_trend = (
            sentiment_history['sentiment_score'].iloc[-1] - 
            sentiment_history['sentiment_score'].iloc[0]
        ) / 100
        
        # Detect divergence
        if price_trend < -0.05 and sentiment_trend > 0.05:
            divergence = 'bullish'  # Price down, sentiment up
            signal = 'BUY'
        elif price_trend > 0.05 and sentiment_trend < -0.05:
            divergence = 'bearish'  # Price up, sentiment down
            signal = 'SELL'
        else:
            divergence = 'none'
            signal = 'HOLD'
        
        return {
            'divergence': divergence,
            'signal': signal,
            'price_trend': price_trend,
            'sentiment_trend': sentiment_trend
        }
    
    def get_sentiment_report(self, symbol: str) -> str:
        """Generate formatted sentiment report"""
        
        news = self.analyze_news_sentiment(symbol)
        social = self.get_social_sentiment(symbol)
        signal = self.generate_sentiment_signal(symbol)
        
        report = f"""
╔═══════════════════════════════════════════════════════════╗
║            SENTIMENT ANALYSIS REPORT - {symbol:4}              ║
╚═══════════════════════════════════════════════════════════╝

NEWS SENTIMENT
--------------
Overall Sentiment:        {news['overall_sentiment'].upper()}
Sentiment Score:          {news['sentiment_score']:.1f}/100
Article Count:            {news['article_count']}
  Positive:               {news['positive_count']}
  Negative:               {news['negative_count']}
  Neutral:                {news['neutral_count']}

SOCIAL MEDIA
------------
Twitter Mentions:         {social['twitter_mentions']:,}
Sentiment Score:          {social['sentiment_score']:.1f}/100
Positive Ratio:           {social['positive_ratio']:.1%}
Engagement Rate:          {social['engagement_rate']:.2%}

TRADING SIGNAL
--------------
Signal:                   {signal['signal']}
Confidence:               {signal['confidence']:.1%}
Combined Score:           {signal['combined_score']:.1f}/100

RECOMMENDATION
--------------
"""
        
        if signal['signal'] == 'BUY':
            report += "✅ POSITIVE - Consider buying on positive sentiment\n"
        elif signal['signal'] == 'SELL':
            report += "❌ NEGATIVE - Consider selling on negative sentiment\n"
        else:
            report += "⚪ NEUTRAL - Hold position, sentiment not strong\n"
        
        return report


if __name__ == "__main__":
    # Test sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Test text analysis
    if HAS_TEXTBLOB:
        text = "The company reported excellent earnings and beat expectations!"
        result = analyzer.analyze_text(text)
        print("Text Analysis:")
        print(f"  Text: {text}")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Polarity: {result['polarity']:.2f}")
    
    # Test news sentiment
    print("\n" + analyzer.get_sentiment_report('AAPL'))
    
    # Test signal generation
    signal = analyzer.generate_sentiment_signal('AAPL')
    print(f"\nTrading Signal: {signal['signal']} (Confidence: {signal['confidence']:.1%})")
