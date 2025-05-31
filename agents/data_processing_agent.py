import sys
import os
import json
import logging
import pandas as pd
from datetime import datetime, timezone
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from common.config import (
    DUCKDB_DATABASE_FILE,
    DUCKDB_TABLE_NAME_HISTORICAL_STOCK,
    DUCKDB_TABLE_NAME_NEWS_ARTICLES,
    SENTIMENT_ANALYSIS_MODEL_NAME
)
from common.utils import get_duckdb_connection, simulate_pubsub_subscribe, simulate_pubsub_publish, get_utc_timestamp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_sentiment_analysis_tokenizer = None
_sentiment_analysis_model = None
_sentiment_analysis_device = "cuda" if torch.cuda.is_available() else "cpu"

def get_sentiment_analysis_model_and_tokenizer():
    global _sentiment_analysis_tokenizer, _sentiment_analysis_model
    
    if _sentiment_analysis_model is None or _sentiment_analysis_tokenizer is None:
        logging.info(f"Logging FinBERT model to {_sentiment_analysis_device}...")
        
        try:
            model_name = SENTIMENT_ANALYSIS_MODEL_NAME
            _sentiment_analysis_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _sentiment_analysis_device = AutoModelForSequenceClassification.from_pretrained(model_name)
            _sentiment_analysis_model.to(_sentiment_analysis_device)
            _sentiment_analysis_model.eval()
            logging.info(f"{SENTIMENT_ANALYSIS_MODEL_NAME} loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load FinBERT model: {e}", exc_info=True)
            
            _sentiment_analysis_tokenizer = None
            _sentiment_analysis_model = None
            raise
    return _sentiment_analysis_tokenizer, _sentiment_analysis_model