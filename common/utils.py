import duckdb
import json
import logging
import time
from datetime import datetime, timezone
from common.config import DUCKDB_DATABASE_FILE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_duckdb_connection():
    return duckdb.connect(database=DUCKDB_DATABASE_FILE)

def get_utc_timestamp():
    return datetime.now(timezone.utc)

def simulate_pubsub_publish(topic_id, message_data):
    logging.info(f"[SIMULATED PUB/SUB] Publishing to topic '{topic_id}': {json.dumps(message_data)}")
    
def simulate_pubsub_subscribe(topic_id, callback_function):
    logging.info(f"[SIMULATED PUB/SUB] Agent subscribed to topic '{topic_id}'. Waiting for messages (simulated)...")
    
    test_messages = [
        {
            'ticker': 'AAPL',
            'data_types_ingested': ['historical_stock_data', 'news_articles'],
            'ingestion_timestamp_uts': str(get_utc_timestamp())
        },
        {
            'ticker': 'GOOGL',
            'data_types_ingested': ['historical_stock_data', 'news_article'],
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