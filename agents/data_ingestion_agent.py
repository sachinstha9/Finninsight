import os
import sys
import pandas as pd
import finnhub
from firebase_admin import firestore
from datetime import datetime, timedelta

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.config import FINNHUB_API_KEY
from common.utils import initFirestore
from data_processing_agent import getSentimentAnalysisModelAndTokenizer, analyzeSentiment

def getFinnhubClient():
    try:
        return finnhub.Client(api_key=FINNHUB_API_KEY)
    except Exception as e:
        print(f"Finnhub initialization failed: {e}")

def getStockNews(ticker, _from, to, batchSize=7):
    try:
        finnhubClient = getFinnhubClient()
    except Exception as e:
        print(f"Cannot initialize finnhub: {e}")
        return pd.DataFrame()
    
    _from = datetime.strptime(_from, '%Y-%m-%d').date() if isinstance(_from, str) else _from
    to = datetime.strptime(to, '%Y-%m-%d').date() if isinstance(to, str) else to
    
    combinedNews = pd.DataFrame()
    
    delta = timedelta(batchSize)
    start = _from
    
    while start <= to:
        end = min(start + delta, to)
        
        print(f"News from {start} to {end}...")
        
        try:
            news = finnhubClient.company_news(ticker, start, end)
        except Exception as e:
            print(f"Cannot fetch news from finnhub: {e}")
            return pd.DataFrame()
        
        news = pd.DataFrame(news)
        
        combinedNews = pd.concat([news, combinedNews], axis=0)
        
        if end == to: break
        start = end
        
    combinedNews['datetime'] = pd.to_datetime(combinedNews['datetime'], unit='s', utc=True, errors = 'coerce')
    combinedNews = combinedNews.drop_duplicates()
    combinedNews = combinedNews.iloc[:, 1:]
    return combinedNews
    
def appendNewsWithSentimentAnalysisScore(database, username, ticker, to, tokenizer, model, device):
    docs = database.collection('news') \
            .order_by('datetime', direction=firestore.Query.DESCENDING) \
            .limit(1) \
            .stream()

    docs_list = list(docs)  
    if not docs_list:
        _from = datetime.strptime(to, '%Y-%m-%d').date() - timedelta(20)
    else:
        latest_doc = docs_list[0]
        doc_dict = latest_doc.to_dict()
        _from = doc_dict.get('datetime')
        if isinstance(_from, str): _from = datetime.fromisoformat(_from)
        
    news = getStockNews(ticker, _from.date(), to, 3)
    news['ticker'] = ticker
    
    unwanted_cols = ['id']
    news = news.drop(unwanted_cols, axis=1, errors="ignore")    

    news = news[news['datetime'] > _from]
    
    news["positive_sentiment"], news["negative_sentiment"], news["neutral_sentiment"] = analyzeSentiment(news['headline'], tokenizer, model, device)
    
    for _, row in news.iterrows():
        data = row.to_dict()
        if isinstance(data.get('datetime'), datetime): data["datetime"] = data["datetime"].isoformat()
        dataID = f"{data['datetime']} | {ticker}"
        database.collection(username).document(dataID).set(data)
        print(f"{dataID} appended successfully...")
        
db = initFirestore()
tokenizer, model, device = getSentimentAnalysisModelAndTokenizer()
appendNewsWithSentimentAnalysisScore(db, 'news', 'AAPL', '2025-05-10', tokenizer, model, device)