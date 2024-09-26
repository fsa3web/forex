import os
from newsapi import NewsApiClient
from textblob import TextBlob
import logging

logger = logging.getLogger(__name__)

# Initialize NewsApiClient
newsapi = NewsApiClient(api_key=os.environ.get('NEWS_API_KEY'))

def fetch_forex_news(currency_pair, num_articles=5):
    """
    Fetch news related to a specific currency pair.
    
    :param currency_pair: The currency pair (e.g., "EURUSD")
    :param num_articles: Number of articles to fetch (default: 5)
    :return: List of dictionaries containing article information
    """
    logger.info(f"Fetching news for {currency_pair}")
    
    # Extract individual currencies from the pair
    base_currency = currency_pair[:3]
    quote_currency = currency_pair[3:]
    
    # Construct the query
    query = f"{base_currency} OR {quote_currency} OR forex OR 'foreign exchange'"
    
    try:
        news = newsapi.get_everything(q=query,
                                      language='en',
                                      sort_by='publishedAt',
                                      page_size=num_articles)
        
        logger.debug(f"Fetched {len(news['articles'])} articles for {currency_pair}")
        return news['articles']
    except Exception as e:
        logger.error(f"Error fetching news for {currency_pair}: {str(e)}")
        return []

def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text.
    
    :param text: The text to analyze
    :return: Sentiment polarity (-1 to 1)
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity

def get_news_sentiment(currency_pair, num_articles=5):
    """
    Get news articles and their sentiment for a specific currency pair.
    
    :param currency_pair: The currency pair (e.g., "EURUSD")
    :param num_articles: Number of articles to analyze (default: 5)
    :return: List of dictionaries containing article information and sentiment
    """
    logger.info(f"Getting news sentiment for {currency_pair}")
    
    articles = fetch_forex_news(currency_pair, num_articles)
    
    sentiment_results = []
    for article in articles:
        sentiment = analyze_sentiment(article['title'] + ' ' + (article['description'] or ''))
        sentiment_results.append({
            'title': article['title'],
            'description': article['description'],
            'url': article['url'],
            'publishedAt': article['publishedAt'],
            'sentiment': sentiment
        })
    
    logger.debug(f"Analyzed sentiment for {len(sentiment_results)} articles")
    return sentiment_results

def calculate_overall_sentiment(sentiment_results):
    """
    Calculate the overall sentiment based on multiple articles.
    
    :param sentiment_results: List of dictionaries containing article information and sentiment
    :return: Overall sentiment score (-1 to 1)
    """
    if not sentiment_results:
        return 0
    
    total_sentiment = sum(article['sentiment'] for article in sentiment_results)
    return total_sentiment / len(sentiment_results)