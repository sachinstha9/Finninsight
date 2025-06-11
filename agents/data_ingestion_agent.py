import os
import sys
import pandas as pd
import finnhub
import yfinance as yf
from firebase_admin import firestore
from datetime import datetime, timedelta, timezone
from google.cloud.firestore_v1.base_query import FieldFilter

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
    
def appendNewsWithSentimentAnalysisScore(database, collectionName, ticker, to, tokenizer, model, device):
    docs = database.collection(collectionName) \
            .order_by('datetime', direction=firestore.Query.DESCENDING) \
            .limit(1) \
            .stream()

    docs_list = list(docs)  
    if not docs_list:
        _from = datetime.strptime(to, '%Y-%m-%d').date() - timedelta(10)
    else:
        latest_doc = docs_list[0]
        doc_dict = latest_doc.to_dict()
        _from = doc_dict.get('datetime')
        if isinstance(_from, str): _from = datetime.fromisoformat(_from)
        
    news = getStockNews(ticker, _from.date() if docs_list else str(_from), to, 3)
    news['ticker'] = ticker
    
    unwanted_cols = ['id']
    news = news.drop(unwanted_cols, axis=1, errors="ignore")    

    if docs_list: news = news[news['datetime'] > _from]
    
    print(f"Sentiment Analysis started for {news.shape[0]} news...")
    news["positive_sentiment"], news["negative_sentiment"], news["neutral_sentiment"] = analyzeSentiment(news['headline'], tokenizer, model, device)
    print(f"Sentiment Analysis finished...")
    
    news = news[::-1].reset_index(drop=True)
    for _, row in news.iterrows():
        data = row.to_dict()
        if isinstance(data['datetime'], pd.Timestamp): data['datetime'] = data["datetime"].to_pydatetime()
        if isinstance(data['datetime'], datetime) and data['datetime'].tzinfo is None: data['datetime'] = data['datetime'].replace(tzinfo=timezone.utc)
        dataID = f"{data['datetime'].isoformat(timespec='seconds')} | {ticker}"
        database.collection(collectionName).document(dataID).set(data)
        print(f"{dataID} appended successfully...")
    
def getNewsFromFirebase(database, collectionName, ticker, window=7):
    cutoff = datetime.now(timezone.utc) - timedelta(days=window)
    docs = database.collection(collectionName) \
        .where(filter=FieldFilter('ticker', '==', ticker)) \
        .where(filter=FieldFilter('datetime', '>=', cutoff)) \
        .stream()
    return [doc.to_dict()['headline'] for doc in docs]

def getPriceVolumeHistory(ticker, period='7d', interval='1d'):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    return hist.reset_index()[['Date', 'Close', 'Volume']]

def getCompanyProfile(ticker):
    try:
        finnhubClient = getFinnhubClient()
    except Exception as e:
        print(f"Cannot initialize finnhub: {e}")
        return {}
    return finnhubClient.company_profile2(symbol=ticker)
    
# db = initFirestore()
# print("News from firebase...")
# print(getNewsFromFirebase(db, 'news', 'AAPL', 40))
# print("\n\nPrice History...")
# print(getPriceVolumeHistory("AAPL"))
# print("\n\nCompany Profile...")
# print(getCompanyProfile("GOOGL"))


# tokenizer, model, device = getSentimentAnalysisModelAndTokenizer()
# appendNewsWithSentimentAnalysisScore(db, 'news', 'AAPL', '2025-05-25', tokenizer, model, device)

