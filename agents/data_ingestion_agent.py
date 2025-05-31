import sys
import os
import yfinance as yf
import pandas as pd
import duckdb
import finnhub
import json
import logging
import time
from datetime import datetime, timedelta, timezone

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.config import (
    DUCKDB_DATABASE_FILE,
    DUCKDB_TABLE_HISTORICAL_STOCK_DDL,
    DUCKDB_TABLE_NEWS_ARTICLES_DDL,
    DUCKDB_TABLE_NAME_HISTORICAL_STOCK,
    DUCKDB_TABLE_NAME_NEWS_ARTICLES,  
    FINNHUB_API_KEY
)
from common.utils import get_duckdb_connection, simulate_pubsub_publish, get_utc_timestamp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_finnhub_client = None

def get_finnhub_client():
    global _finnhub_client
    if _finnhub_client is None:
        if not FINNHUB_API_KEY or FINNHUB_API_KEY == "YOUR_FINNHUB_API_KEY_HERE_IF_NOT_ENV_VAR":
            error_msg = "Finnhub API key is not set or is invalid. Cannot initialize client."
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            _finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
            if _finnhub_client is None:
                logging.error("Finnhub client initialization returned None unexpectedly.")
                raise ValueError("Finnhub client initialization failed (returned None).")
            logging.info("Finnhub client successfully initialized.")
        except Exception as e:
            logging.error(f"Error during Finnhub client instantiation: {e}")
            raise ValueError(f"Finnhub client instantiation failed due to: {e}")
    return _finnhub_client

def initialize_duckdb_tables():
    logging.info("Initializing DuckDB tables...")
    try:
        with get_duckdb_connection() as con:
            con.execute(DUCKDB_TABLE_HISTORICAL_STOCK_DDL)
            con.execute(DUCKDB_TABLE_NEWS_ARTICLES_DDL)
            logging.info("DuckDB tables initialized.")
    except Exception as e:
        logging.error(f"Error initializing DuckDB tables: {e}", exc_info=True)

def fetch_historical_stock_data(ticker: str, period: str = "1mo"):
    logging.info(f"Fetching historical stock data for {ticker} for period '{period}'...")
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            logging.warning(f"No historical data found for {ticker} for period '{period}'.")
            return pd.DataFrame()

        df = df.reset_index()

        first_col_name = df.columns[0]
        df.rename(columns={first_col_name: 'date'}, inplace=True)
        
        new_columns = []
        for col in df.columns:
            if col == 'date':
                new_columns.append(col)
            else:
                new_columns.append(col.lower().replace(' ', '_').replace('.', ''))
        df.columns = new_columns
        
        df['date'] = pd.to_datetime(df['date']).dt.date

        df['ticker'] = ticker
        df['data_source'] = 'Yahoo Finance'
        df['ingestion_timestamp'] = get_utc_timestamp()

        required_columns = [
            'ticker', 'date', 'open', 'high', 'low', 'close', 'volume',
            'data_source', 'ingestion_timestamp'
        ]
        df = df.reindex(columns=required_columns)

        logging.info(f"Successfully fetched {len(df)} rows of historical stock data for {ticker}.")
        return df

    except Exception as e:
        logging.error(f"Error fetching historical stock data for {ticker}: {e}", exc_info=True)
        return pd.DataFrame()
    
def fetch_news_data(ticker: str, days_ago: int = 7):
    logging.info(f"Fetching company news for {ticker} for last {days_ago} days from Finnhub...")

    try:
        finnhub_client = get_finnhub_client()
        
        time.sleep(1) # Pause for 1 second before making the API call because of API limits
        
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
        logging.info(f"Successfully fetched {len(df)} company news articles for {ticker}.")
        return df

    except ValueError as ve:
        logging.error(f"Finnhub client error: {ve}")
        return pd.DataFrame()
    except finnhub.FinnhubAPIException as e:
        logging.error(f"Finnhub API error for {ticker}: {e}", exc_info=True)
        if "not within the free plan" in str(e).lower() or "too many requests" in str(e).lower() or "api limit reached" in str(e).lower():
            logging.error("Finnhub limits reached.")
        return pd.DataFrame()
    except AttributeError as e:
        logging.error(f"Finnhub client object is not valid for {ticker}: {e}.")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"An unexpected error occurred while fetching news for {ticker} from Finnhub: {e}", exc_info=True)
        return pd.DataFrame()
    
def load_dataframe_to_duckdb(dataframe: pd.DataFrame, table_name: str):
    if dataframe.empty:
        logging.info(f"No data in '{table_name}'.")
        return 
    
    try:
        with get_duckdb_connection() as con:
            con.append(table_name, dataframe)
            logging.info(f"Loaded {len(dataframe)} rows into DuckDB table '{table_name}'.")
    except Exception as e:
        logging.error(f"Error loading data to DuckDB table '{table_name}': {e}", exc_info=True)

def ingest_data_agent_main(ticker: str, stock_period: str = '1mo', news_days_ago: int = 7):
    logging.info(f"---Starting data ingestion for ticker: {ticker}---")
    
    try:
        df_stock = fetch_historical_stock_data(ticker, period=stock_period)
        if not df_stock.empty:
            load_dataframe_to_duckdb(df_stock, DUCKDB_TABLE_NAME_HISTORICAL_STOCK)
        else:
            logging.warning(f"No stock data fetched for {ticker}.")
        
        df_news = fetch_news_data(ticker, days_ago=news_days_ago)
        if not df_news.empty:
            load_dataframe_to_duckdb(df_news, DUCKDB_TABLE_NAME_NEWS_ARTICLES)
        else:
            logging.warning(f"No news data fetched for {ticker}.")
        
        message_for_next_agent = {
            'ticker': ticker,
            'data_types_ingested': ['historical_stock_data', 'news_articles'],
            'ingestion_timestamp_uts': str(get_utc_timestamp()) # Convert to string
        }
        
        simulate_pubsub_publish("new_data_topic", message_for_next_agent)
        
        logging.info(f"---Data ingestion complete for {ticker}.---")
        return f"Successfully ingested data for {ticker}"
    except Exception as e:
        logging.info(f"An unhandled error occurred in Data ingestion agent for {ticker}: {e}", exc_info=True)
        return f"Error during data ingestion for {ticker}: {e}"
    
if __name__ == "__main__":
    logging.info("--Starting local testing of data ingestion agent---")

    tickers_to_ingest = [
        {"ticker": "AAPL", "historical_period": "1y", "news_days_ago": 30},
        {"ticker": "GOOGL", "historical_period": "6mo", "news_days_ago": 7},
        {"ticker": "MSFT", "historical_period": "3mo", "news_days_ago": 5},
    ]

    initialize_duckdb_tables() 

    results = {}
    for i, ticker_config in enumerate(tickers_to_ingest):
        ticker = ticker_config["ticker"]
        historical_period = ticker_config["historical_period"]
        news_days_ago = ticker_config["news_days_ago"]

        logging.info(f"\n--- Ingesting data for {ticker} ({historical_period} historical, {news_days_ago} days news) ---")
        
        historical_df = fetch_historical_stock_data(ticker, historical_period)
        if not historical_df.empty:
            load_dataframe_to_duckdb(historical_df, DUCKDB_TABLE_NAME_HISTORICAL_STOCK)
        else:
            logging.warning(f"No historical data fetched for {ticker}.")

        news_df = fetch_news_data(ticker, news_days_ago)
        if not news_df.empty:
            load_dataframe_to_duckdb(news_df, DUCKDB_TABLE_NAME_NEWS_ARTICLES)
        else:
            logging.warning(f"No news data fetched for {ticker}.")
        
        data_types_ingested = []
        if not historical_df.empty:
            data_types_ingested.append("historical_stock_data")
        if not news_df.empty:
            data_types_ingested.append("news_articles")

        if data_types_ingested:
            simulate_pubsub_publish(
                topic_id="new_data_topic",
                message_data={
                    "ticker": ticker,
                    "data_types_ingested": data_types_ingested,
                    "ingestion_timestamp_uts": str(get_utc_timestamp())
                }
            )

        logging.info(f"---Data ingestion complete for {ticker}.---")
        results[ticker] = "Successfully ingested data" if data_types_ingested else "No data ingested"

        if i < len(tickers_to_ingest) - 1:
            print(f"Pausing for 5 seconds before ingesting data for the next ticker because of API limits...")
            time.sleep(5) 
            
    logging.info("\n--- Verifying Data in DuckDB (using a separate connection) ---")

    with get_duckdb_connection() as con:
        logging.info(f"\nLast 5 rows of Historical Stock Data for AAPL from '{DUCKDB_TABLE_NAME_HISTORICAL_STOCK}':")
        aapl_hist_data = con.execute(f"SELECT * FROM {DUCKDB_TABLE_NAME_HISTORICAL_STOCK} WHERE ticker = 'AAPL' ORDER BY date DESC LIMIT 5;").fetchdf()
        print(aapl_hist_data.to_string())

    with get_duckdb_connection() as con:
        logging.info(f"\nLast 5 rows of News Articles for GOOGL from '{DUCKDB_TABLE_NAME_NEWS_ARTICLES}':")
        googl_news_data = con.execute(f"SELECT * FROM {DUCKDB_TABLE_NAME_NEWS_ARTICLES} WHERE ticker = 'GOOGL' ORDER BY published_at DESC LIMIT 5;").fetchdf()
        print(googl_news_data.to_string())
        
    logging.info("\n--- All ingestion and verification complete. ---")