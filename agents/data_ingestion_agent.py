import sys
import os
import finnhub
import pandas as pd
import yfinance as yf
from firebase_admin import firestore
from datetime import date, timedelta, datetime, timezone

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.config import FINNHUB_API_KEY
from common.utils import initFirestore

def getFinnhubClient():
    try:
        return finnhub.Client(api_key=FINNHUB_API_KEY)
    except Exception as e:
        print(f"Finnhub initialization failed: {e}")

def getStockNews(ticker, _from=None, to=None):
    finnhubClient = getFinnhubClient()
    combinedNews = pd.DataFrame()
    delta = timedelta(days=7)

    _from = datetime.strptime(_from, '%Y-%m-%d').date()
    to = datetime.strptime(to, '%Y-%m-%d').date()
    start = _from
    
    while start <= to:
        end = min(start + delta, to)
        print(f"Fetching {ticker}'s news from {start} to {end}...")  

        try:
            news = finnhubClient.company_news(
                ticker, start, end
            )

            if news:
                combinedNews = pd.concat([combinedNews, pd.DataFrame(news)], ignore_index=True)
            
            start = end + timedelta(1) if start == end else end
        except Exception as e:
            print(f"News fetch failed. {e}")
            break
        
    combinedNewsUnique = combinedNews.drop_duplicates(subset='datetime', keep='first')
    return combinedNewsUnique

def getStockData(ticker, start=None, end=None, period=None, interval='1d'):
    t = yf.Ticker(ticker)
    
    if period:
        return pd.DataFrame(t.history(period=period, interval=interval)).reset_index()
    elif start or end:
        return pd.DataFrame(t.history(start=start, end=end, interval=interval)).reset_index()
    else:
        raise ValueError("You must provide either a period or start/end date.")
    

def uploadDataframeToFirestore(database, df, collectionName, docIdColName=None):    
    for index, row in df.iterrows():
        docData = row.to_dict()
        
        docId = str(index) if docIdColName is None else ' | '.join([str(row[col]).replace('/', '_') for col in docIdColName])
        
        database.collection(collectionName).document(docId).set(docData)

    print(f"Uploaded {len(df)} documents to collection '{collectionName}'.")


def updateNews(database, collectionName, ticker, to=None):
    docs = database.collection(collectionName)\
        .where('ticker', '==', ticker)\
        .order_by('datetime', direction=firestore.Query.DESCENDING)\
        .limit(1).stream()

    recentNews = None
    for doc in docs:
        recentNews = doc.to_dict()

    if recentNews is None:
        _from = datetime.now() - timedelta(days=150)
    else:
        firestore_dt = recentNews['datetime']
        if hasattr(firestore_dt, 'to_datetime'):
            _from = firestore_dt.to_datetime()
        else:
            _from = firestore_dt

    if _from.tzinfo is not None:
        _from = _from.replace(tzinfo=None)

    if to is None:
        to = datetime.now()
    else:
        to = datetime.strptime(to, '%Y-%m-%d')
        if to.tzinfo is not None:
            to = to.replace(tzinfo=None)

    news = getStockNews(ticker, str(_from.date()), str(to.date()))
    
    news['datetime'] = pd.to_datetime(news['datetime'], unit='s')
    news['ticker'] = ticker
    
    unwantedColumns = ['id', 'image']
    news = news.drop(unwantedColumns, axis=1)
    
    news = news[news['datetime'] > _from]

    uploadDataframeToFirestore(database, news, collectionName, docIdColName=['ticker', 'datetime', 'headline'])

    
db = initFirestore()
updateNews(db, 'news', 'AAPL', '2024-08-01')
