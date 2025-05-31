import yfinance as yf
import pandas as pd
import duckdb
import finnhub
import json
import logging
from datetime import datetime, timedelta, timezone
from common.config import (
    DUCKDB_DATABASE_FILE, DUCKDB_TABLE_HISTORICAL_STOCK, DUCKDB_TABLE_NEWS_ARTICLE, FINNHUB_API_KEY
)
from common.utils import get_duckdb_connection, simulate_pubsub_publish, get_utc_timestamp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_finhub_client = None

def get_finnhub_client():
    global _finhub_client
    
    if _finhub_client is None:
        if not FINNHUB_API_KEY or FINNHUB_API_KEY == "API":
            logging.error("Finnhub Api cannot be found.")
            raise ValueError("Finnhub API key is missing.")
        _finhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
        return _finhub_client
    
HISTORICAL_STOCK_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {DUCKDB_TABLE_HISTORICAL_STOCK} (
    ticker VARCHAR NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(18, 4),
    high DECIMAL(18, 4),
    low DECIMAL(18, 4),
    close DECIMAL(18, 4),
    volume GITINT,
    date_source VARCHAR,
    ingestio_timestamp TIMESTAMP
);
"""

NEWS_ARTICLES_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {DUCKDB_TABLE_NEWS_ARTICLE} (
    ticker VARCHAR NOT NULL,
    title VARCHAR,
    description VARCHAR,
    utl VARCHAR,
    published_at TIMESTAMP,
    source VARCHAR,
    ingestion_timestamp TIMESTAMP
);
"""

def initialize_duckdb_tables():
    logging.info("Intializing DuckDB tables...")
    try:
        with get_duckdb_connection() as con:
            con.execute(HISTORICAL_STOCK_SCHEMA)
            con.execute(NEWS_ARTICLES_SCHEMA)
            logging.info("DuckDB tables initialized.")
    except Exception as e:
        logging.error(f"Error initializing DuckDB tables: {e}")
        raise
    
def fetch_historical_stock_data(ticker, period='1mo'):
    logging.info(f"Fetching historical dara for {ticker} for period '{period}'...")
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            logging.warning(f"No historical data found for {ticker} for period {period}.")
            return pd.DataFrame()
        
        df = df.reset_index(drop=True)
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        if 'date' not in df.columns and 'index' in df.columns:
            df.rename(columns={'index': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        df['ticker'] = ticker
        df['data_source'] = 'Yahoo Finance'
        df['ingestion_timestamp'] = get_utc_timestamp()
        
        required_cols = [
            'ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'data_source', 'ingestion_timestamp'
        ]
        df = df.reindex(columns=required_cols)
        
        logging.info(f"Successfully fetched {len(df)} rows of historical stock data for {ticker}")
        return df
    except Exception as e:
        logging.error(f"Error fetching historical stock data for {ticker}: {e}", exc_info=True)
        return pd.DataFrame()
    
    
def fetch_news_data(ticker, days_ago=7):
    logging.info(f"Fetching company news for {ticker} for last {days_ago} days from Finnhub...")
    
    try:
        finnhub_client = get_finnhub_client()
        
        from_date = (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        to_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        
        news_data = finnhub_client.company_news(ticker, _from=from_date, to=to_date)
        
        if not news_data:
            logging.warning(f"No company news found for {ticker} from Finnhub for the period {from_date} to {to_date}.")
            return pd.DataFrame()
        
        records = []
        for article in news_data:
            records.append({
                'ticker': ticker,
                'title': article.get('headline'),
                'description': article.get('summary'),
                'url': article.get('url'),
                'published_at': datetime.fromtimestamp(article.get('datetime'), tz=timezone.utc) if article.get('datetime') else None,
                'source': article.get('source'),
                'ingestion_timestamp': get_utc_timestamp()
            })
            
            df = pd.DataFrame(records)
            logging.info(f"Successfully fetched {len(df)} company news articles for {ticker} from Finnhub.")
            return df
        
    except ValueError as ve:
        logging.error(f"Finnhub client error: {ve}")
        return pd.DataFrame()
    
    except finnhub.FinnhubAPIException as e:
        logging.error(f"Finnhub API error for {ticker}: {e}", exc_info=True)
        if "not within free plan" in str(e).lower() or "too many requests" in str(e).lower() or "api limit reached" in str(e).lower():
            logging.error("Something wrong in finnhub api")
        return pd.DataFrame()
    
    except Exception as e:
        logging.error(f"An unexpected error occured while fetching news for {ticker} from Finnhub: {e}", exc_info=True)
        return pd.DataFrame() 
    
def load_dataframe_to_duckdb(dataframe, table_name):
    if dataframe.empty:
        logging.info(f"No data to load into DuckDB table '{table_name}'.")
        return 
    
    try:
        with get_duckdb_connection() as con:
            con.execute(f"INSERT INTO {table_name} SELECT * FROM dataframe")
            logging.info(f"Loaded {len(dataframe)} rows into DuckDB table '{table_name}'.")
    except Exception as e:
        logging.error(f"Error loading data to DuckDB table '{table_name}': {e}", exc_info=True)