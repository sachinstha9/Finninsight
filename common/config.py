import os

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "d0ir93pr01qnehifshrgd0ir93pr01qnehifshs0")
DUCKDB_DATABASE_FILE = "finninsight.duckdb" 

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