import duckdb
import json
import logging
import time
from datetime import datetime, timezone
from common.config import (
    DUCKDB_DATABASE_FILE, DUCKDB_TABLE_HISTORICAL_STOCK_DDL, DUCKDB_TABLE_NEWS_ARTICLES_DDL, DUCKDB_TABLE_NAME_HISTORICAL_STOCK, DUCKDB_TABLE_NAME_NEWS_ARTICLES,
    PROCESSED_TABLE_HISTORICAL_STOCK_DDL, PROCESSED_TABLE_NEWS_SENTIMENT_DDL, PROCESSED_TABLE_NAME_HISTORICAL_STOCK, PROCESSED_TABLE_NAME_NEWS_SENTIMENT
    )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_duckdb_connection():
    return duckdb.connect(database=DUCKDB_DATABASE_FILE)

def create_table(table_ddl, table_name):
    try:
        with get_duckdb_connection() as con:
            con.execute(table_ddl)
        logging.info(f"Table '{table_name}' created.")
    except Exception as e:
        logging.error(f"Error creating table '{table_name}': {e}", exc_info=True)
        raise
    
def init_database():
    logging.info("Initializing DuckDB database for raw data...")
    create_table(DUCKDB_TABLE_HISTORICAL_STOCK_DDL, DUCKDB_TABLE_NAME_HISTORICAL_STOCK)
    create_table(DUCKDB_TABLE_NEWS_ARTICLES_DDL, DUCKDB_TABLE_NAME_NEWS_ARTICLES)
    logging.info("Raw data tables initialized.")
    
def init_database_for_processed_data():
    logging.info("Initializing DuckDB database for processed data...")
    create_table(PROCESSED_TABLE_HISTORICAL_STOCK_DDL, PROCESSED_TABLE_NAME_HISTORICAL_STOCK)
    create_table(PROCESSED_TABLE_NEWS_SENTIMENT_DDL, PROCESSED_TABLE_NAME_NEWS_SENTIMENT)
    logging.info("Processed data tables initialized.")

def get_utc_timestamp():
    return datetime.now(timezone.utc)

def simulate_pubsub_publish(topic_id, message_data):
    try:
        for key,value in message_data.items():
            if isinstance(value, datetime):
                message_data[key] = str(value)
        logging.info(f"[SIMULATED PUB/SUB] Publishing to topic '{topic_id}': {json.dumps(message_data, indent=2)}")
    except TypeError as e:
        logging.error(f"Error serializing message for topic '{topic_id}': {e}, Message: {message_data}", exc_info=True)
    except Exception as e:
        logging.error(f"Error simulating Pub/Sub publish to topic '{topic_id}': {e}", exc_info=True)
    
def simulate_pubsub_subscribe(topic_id, callback_function):
    logging.info(f"[SIMULATED PUB/SUB] Agent subscribed to topic '{topic_id}'. Waiting for messages (simulated)...")
    
    test_messages = [
        {
            'ticker': 'AAPL',
            'data_types_ingested': ['historical_stock_data', 'news_articles'],
            'ingestion_timestamp_uts': str(get_utc_timestamp())
        }
    ]
    
    print("\n[SIMULATED PUB/SUB] Simulating message reception and calling callback...")
    for i, message in enumerate(test_messages):
        logging.info(f"[SIMULATED PUB/SUB] Received meessage {i+1} for topic '{topic_id}': {message}")
        try:
            callback_function(message)
        except Exception as e:
            logging.error(f"[SIMULATDE PUB/SUB] Error in callback for message {message}: {e}", exc_info=True)
            
        if i < len(test_messages) - 1:
            print("[SIMULATED PUB/SUB] Simulating a pause before next message...")
            time.sleep(3)
            
    logging.info(f"[SIMULATED PUB/SUB] Simulated message reception finished for topic '{topic_id}'.")