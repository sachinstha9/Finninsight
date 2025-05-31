import os

DUCKDB_DATABASE_FILE = os.path.join(os.path.dirname(__file__), 'data', 'fininsight.duckdb')
DUCKDB_TABLE_HISTORICAL_STOCK = 'historical_stock_data'
DUCKDB_TABLE_NEWS_ARTICLE = 'news_article'
DUCKDB_TABLE_SENTIMENT_SCORES = 'sentiment_scores'
DUCKDB_TABLE_FINANCIAL_INSIGHTS = 'financial_insights'

FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY', 'API')
