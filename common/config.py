import os

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "d0ir93pr01qnehifshrgd0ir93pr01qnehifshs0")
SENTIMENT_ANALYSIS_MODEL_NAME = "ProsusAI/finbert"

DUCKDB_DATABASE_FILE = "/home/amirul/finninsight/data/finninsight.duckdb" 
DUCKDB_TABLE_NAME_HISTORICAL_STOCK = "historical_stock_data"
DUCKDB_TABLE_NAME_NEWS_ARTICLES = "news_articles"

DUCKDB_TABLE_HISTORICAL_STOCK_DDL = f"""
CREATE TABLE IF NOT EXISTS {DUCKDB_TABLE_NAME_HISTORICAL_STOCK} (
    ticker VARCHAR,
    date DATE,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume BIGINT,
    data_source VARCHAR,
    ingestion_timestamp TIMESTAMP
);
"""

DUCKDB_TABLE_NEWS_ARTICLES_DDL = f"""
CREATE TABLE IF NOT EXISTS {DUCKDB_TABLE_NAME_NEWS_ARTICLES} (
    ticker VARCHAR,
    title VARCHAR,
    description VARCHAR,
    url VARCHAR,
    published_at TIMESTAMP,
    source VARCHAR,
    ingestion_timestamp TIMESTAMP
);
"""

PROCESSED_TABLE_NAME_HISTORICAL_STOCK = "processed_historical_stock"
PROCESSED_TABLE_NAME_NEWS_SENTIMENT = "processed_news_sentiment"

PROCESSED_TABLE_HISTORICAL_STOCK_DDL = f"""
CREATE TABLE IF NOT EXISTS {PROCESSED_TABLE_NAME_HISTORICAL_STOCK} (
    ticker VARCHAR,
    date DATE,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume BIGINT,
    "7_day_ma" DOUBLE,
    "20_day_ma" DOUBLE,
    processing_timestamp TIMESTAMP
);
"""

PROCESSED_TABLE_NEWS_SENTIMENT_DDL = f"""
CREATE TABLE IF NOT EXISTS {PROCESSED_TABLE_NAME_NEWS_SENTIMENT} (
    ticker VARCHAR,
    processed_article_id VARCHAR PRIMARY KEY,
    title VARCHAR,
    description VARCHAR,
    url VARCHAR,
    published_at TIMESTAMP,
    source VARCHAR,
    positive_sentiment DOUBLE,
    negative_sentiment DOUBLE,
    neutral_sentiment DOUBLE,
    compound_sentiment DOUBLE,
    processing_timestamp TIMESTAMP
);
"""