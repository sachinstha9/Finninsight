import sys
import os
import json
import logging
import pandas as pd
from datetime import datetime, timezone
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import hashlib
import torch.nn.functional as F

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from common.config import (
    DUCKDB_DATABASE_FILE,
    DUCKDB_TABLE_NAME_HISTORICAL_STOCK,
    DUCKDB_TABLE_NAME_NEWS_ARTICLES,
    SENTIMENT_ANALYSIS_MODEL_NAME,
    PROCESSED_TABLE_NAME_HISTORICAL_STOCK, 
    PROCESSED_TABLE_NAME_NEWS_SENTIMENT
)
from common.utils import get_duckdb_connection, simulate_pubsub_subscribe, simulate_pubsub_publish, get_utc_timestamp, init_database_for_processed_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_sentiment_analysis_tokenizer = None
_sentiment_analysis_model = None
_sentiment_analysis_device = "cuda" if torch.cuda.is_available() else "cpu"

def get_sentiment_analysis_model_and_tokenizer():
    global _sentiment_analysis_tokenizer, _sentiment_analysis_model
    
    if _sentiment_analysis_model is None or _sentiment_analysis_tokenizer is None:
        logging.info(f"Loading {SENTIMENT_ANALYSIS_MODEL_NAME} model to {_sentiment_analysis_device}...")
        
        try:
            model_name = SENTIMENT_ANALYSIS_MODEL_NAME
            _sentiment_analysis_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _sentiment_analysis_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            _sentiment_analysis_model.to(_sentiment_analysis_device)
            _sentiment_analysis_model.eval()
            logging.info(f"{SENTIMENT_ANALYSIS_MODEL_NAME} loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load {SENTIMENT_ANALYSIS_MODEL_NAME} model: {e}", exc_info=True)
            _sentiment_analysis_tokenizer = None
            _sentiment_analysis_model = None
            raise
    return _sentiment_analysis_tokenizer, _sentiment_analysis_model

def _analyze_sentiment(text, tokenizer, model):
    if not isinstance(text, str) or not text.strip():
        return {'positive_sentiment': 0.0, 'negative_sentiment': 0.0, 'neutral_sentiment': 0.0, 'compound_sentiment': 0.0}
    
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(_sentiment_analysis_device) for k, v in inputs.items()}
        
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
            'positive_sentiment': positive_score,
            'negative_sentiment': negative_score,
            'neutral_sentiment': neutral_score,
            'compound_sentiment': compound_score
        }
    except Exception as se:
        logging.error(f"Error during sentiment analysis.", exc_info=True) # Added exc_info for more details
        return {'positive_sentiment': 0.0, 'negative_sentiment': 0.0, 'neutral_sentiment': 0.0, 'compound_sentiment': 0.0}

def process_historical_stock_data(ticker, start_date=None, end_date=None):
    logging.info(f"Processing historical stock data for {ticker} from DuckDB...")
    processed_df = pd.DataFrame()
    
    try:
        with get_duckdb_connection() as con:
            query = f"SELECT * FROM {DUCKDB_TABLE_NAME_HISTORICAL_STOCK} WHERE ticker = '{ticker}'"
            if start_date:
                query += f" AND date >= '{start_date}'"
            if end_date:
                query += f" AND date <= '{end_date}'"
            query += " ORDER BY date ASC;"
            
            raw_df = con.execute(query).fetchdf()
            
            if raw_df.empty:
                logging.warning(f"No raw historical data found for {ticker}.")
                return pd.DataFrame()
            
            processed_df = raw_df.copy()
            processed_df['7_day_ma'] = processed_df['close'].rolling(window=7).mean()
            processed_df['20_day_ma'] = processed_df['close'].rolling(window=20).mean()
            
            processed_df['processing_timestamp'] = get_utc_timestamp()
            
            columns_to_insert = [
                'ticker', 'date', 'open', 'high', 'low', 'close', 'volume', '7_day_ma', '20_day_ma', 'processing_timestamp'
            ]
            
            processed_df = processed_df[columns_to_insert] # Correctly removed .fillna()
            
            logging.info(f"DataFrame info for {PROCESSED_TABLE_NAME_HISTORICAL_STOCK} before append:")
            processed_df.info()
            logging.info(f"DataFrame head for {PROCESSED_TABLE_NAME_HISTORICAL_STOCK} before append:\n{processed_df.head()}")

            con.append(PROCESSED_TABLE_NAME_HISTORICAL_STOCK, processed_df)
            logging.info(f"Successfully stored {len(processed_df)} rows of data.") 
            
            return processed_df
    except Exception as e:
        logging.error(f"Error processing and storing historical stock data for {ticker}: {e}.", exc_info=True)
        return pd.DataFrame()
        
def process_news_articles(ticker, published_after=None):
    logging.info(f"Processing news articles for {ticker} from DuckDB...")
    
    try:
        with get_duckdb_connection() as con:
            query = f"SELECT * FROM {DUCKDB_TABLE_NAME_NEWS_ARTICLES} WHERE ticker = '{ticker}'"
            if published_after:
                published_after_str = published_after.strftime('%Y-%m-%d %H:%M:%S.%f%z')
                query += f" AND published_at >= '{published_after_str}'"
            query += " ORDER BY published_at ASC;"
            
            raw_df = con.execute(query).fetchdf()
            
            if raw_df.empty:
                logging.warning(f"No news article found for {ticker} within the specified range.")
                return pd.DataFrame()
            
            processed_df = raw_df.copy()
            
            tokenizer, model = get_sentiment_analysis_model_and_tokenizer()
            
            processed_df["full_text"] = processed_df["title"].fillna('') + ". " + processed_df["description"].fillna('')
            
            sentiment_results = processed_df["full_text"].apply(lambda x: _analyze_sentiment(x, tokenizer, model))
            
            processed_df = pd.concat([processed_df, sentiment_results.apply(pd.Series)], axis=1)
            
            processed_df['processed_article_id'] = processed_df.apply(
                lambda row: hashlib.sha256(f"{row['url']}{row['published_at']}".encode()).hexdigest(), axis=1
            )        
            
            processed_df['processing_timestamp'] = get_utc_timestamp()
            
            columns_to_insert = [
                'ticker', 'processed_article_id', 'title', 'description', 'url',
                'published_at', 'source', 'positive_sentiment', 'negative_sentiment',
                'neutral_sentiment', 'compound_sentiment', 'processing_timestamp'
            ]
            
            processed_df = processed_df[columns_to_insert]
            
            logging.info(f"DataFrame info for {PROCESSED_TABLE_NAME_NEWS_SENTIMENT} before append:")
            processed_df.info()
            logging.info(f"DataFrame head for {PROCESSED_TABLE_NAME_NEWS_SENTIMENT} before append:\n{processed_df.head()}")

            con.append(PROCESSED_TABLE_NAME_NEWS_SENTIMENT, processed_df)
            logging.info(f"Successfully stored {len(processed_df)} processed new articles.")
                
            return processed_df.drop(columns=['full_text'], errors='ignore')
    except Exception as e:
        logging.error(f"Error processing news article for {ticker}: {e}", exc_info=True)
        return pd.DataFrame()

def data_processing_agent_main(message):
    ticker = message.get('ticker')
    ingestion_timestamp_uts = message.get('ingestion_timestamp_uts')
    
    if not ticker:
        logging.error("Received message without a 'ticker. Skipping processing")
        return 
    
    logging.info(f"---Agent 2: Starting data processing for ticker: {ticker}---")
        
    last_ingestion_time = None
    
    if ingestion_timestamp_uts:
        try:
            if '.' in ingestion_timestamp_uts and ('+' in ingestion_timestamp_uts or '-' == ingestion_timestamp_uts[-5]):
                last_ingestion_time = datetime.fromisoformat(ingestion_timestamp_uts)
            elif '.' in ingestion_timestamp_uts:
                last_ingestion_time = datetime.strptime(ingestion_timestamp_uts, '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=timezone.utc)
            else:
                last_ingestion_time = datetime.strptime(ingestion_timestamp_uts, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
        except ValueError as e:
            logging.warning(f"Could not parse ingestion_timestamp_uts '{ingestion_timestamp_uts}': {e}. Processing all available data")
            last_ingestion_time = None
            
    processed_stock_df = process_historical_stock_data(
        ticker
        # start_date=last_ingestion_time.strftime('%Y-%m-%d') if last_ingestion_time else None,
        # end_date=datetime.now(timezone.utc).strftime('%Y-%m-%d')
    )
    
    processed_news_df = process_news_articles(ticker)

    summary_results = {
        'ticker': ticker,
        'processing_timestamp_uts': str(get_utc_timestamp()),
        'stock_data_stored_rows': len(processed_stock_df),
        'news_data_stored_articles': len(processed_news_df),
        'latest_stock_7_day_ma': processed_stock_df['7_day_ma'].iloc[-1] if not processed_stock_df.empty else None,
        'latest_stock_20_day_ma': processed_stock_df['20_day_ma'].iloc[-1] if not processed_stock_df.empty else None,
        'average_news_compound_sentiment': processed_news_df['compound_sentiment'].mean() if not processed_news_df.empty else None
    }
    
    simulate_pubsub_publish("processed_data_topic", summary_results)
    logging.info(f"---Agent 2: Data processing and storage compelete for {ticker}---")
    
if __name__ == "__main__":
    logging.info("---Starting agent 2 (Data Processing Agent) in subscription mode---")
    
    try:
        get_sentiment_analysis_model_and_tokenizer()
    except Exception as e:
        logging.critical(f"Failed to initialize {SENTIMENT_ANALYSIS_MODEL_NAME} model. Exiting error: {e}")
        sys.exit(1)
    
    try:
        init_database_for_processed_data()
    except Exception as e:
        logging.critical(f"Failed to initialize processed data tables in DuckDB. Exiting, Error: {e}")
        sys.exit(1)
    
    simulate_pubsub_subscribe("new_data_topic", data_processing_agent_main)
    logging.info("---Agent 2: Simulation complete---")