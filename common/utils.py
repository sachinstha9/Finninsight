import duckdb
import json
import logging
from datetime import datetime, timezone
from common.config import DUCKDB_DATABASE_FILE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_duckdb_connection():
    return duckdb.connect(database=DUCKDB_DATABASE_FILE)

def get_utc_timestamp():
    return datetime.now(timezone.utc)

def simulate_pubsub_publish(topic_id, message_data):
    logging.info(f"[SIMULATED PUB/SUB] Publishing to topic '{topic_id}': {json.dumps(message_data)}")