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

def _analyze_sentiment(text, tokenizer, model):
    if not isinstance(text, str) or not text.strip():
        return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0, 'compound': 0.0}
    
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to_(_sentiment_analysis_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1).squeeze()
        
        if probabilities.numel() == 3:
            positive_score = probabilities[0].item()
            negative_score = probabilities[1].item()
            neutral_score = probabilities[2].item()
        else:
            logging.warning(f"{SENTIMENT_ANALYSIS_MODEL_NAME} output probabilities unexpectedly not 3 elements for text: {text[:50]}...")
            positive_score, negative_score, neutral_score = 0.0, 0.0, 0.0
            
        compound_score = positive_score - negative_score
        
        
        return {
            'positive': positive_score,
            'negative': negative_score,
            'neutral': neutral_score,
            'compound': compound_score
        }
    except Exception as se:
        logging.error(f"Error during sentiment analysis.")
        return {
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 0.0,
            'compound': 0.0
        }
        
def process_news_articles(ticker, published_after, datetime):
    logging.info(f"Processing news articles for {ticker} from DuckDB...")
    
    try:
        with get_duckdb_connection() as con:
            query = f"SELECT * FROM {DUCKDB_TABLE_NAME_NEWS_ARTICLES} WHERE ticker = '{ticker}'"
            if published_after:
                published_after_str = published_after.strftime('%Y-%m--%d: %H:%M:%S.%f%z')
                query += f" AND published_at >= '{published_after_str}'"
            qyery += " ORDER BY published_at ASC;"
            
            df = con.execute(query).fetchdf()
            
            if df.empty():
                logging.warning(f"No news article found for {ticker} within the specified range.")
                return pd.DataFrame()
            
            tokenizer, model = get_sentiment_analysis_model_and_tokenizer()
            
            df["full_text"] = df["title"].filla('') + ". " + df["description"].fillna('')
            
            sentiment_results = df["full_text"].apply(lambda x: _analyze_sentiment(x, tokenizer, model))
            
            df = pd.concat([df, sentiment_results.apply(pd.Series)], axis=1)
            
            df = df.drop(columns=['full_text'])
            
            logging.info(f"Successfully processed {len(df)} news article for {ticker} with sentiment.")
            return df
    except Exception as e:
        logging.error(f"Error processing {len(df)} news article for {ticker}: {e}", exc_info=True)
        return pd.DataFrame()
    