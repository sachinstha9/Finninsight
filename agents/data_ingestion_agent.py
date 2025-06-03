import sys
import os
import finnhub
import pandas as pd
import yfinance as yf
from datetime import date, timedelta, datetime

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.config import FINNHUB_API_KEY

def get_finnhub_client():
    import finnhub
    try:
        return finnhub.Client(api_key=FINNHUB_API_KEY)
    except Exception as e:
        print(f"Finnhub initialization failed: {e}")

def getStockNews(ticker, _from=None, to=None):
    if _from is None:
        _from = date.today() - timedelta(days=30)
    if to is None:
        to = date.today()
        
    try:
        if isinstance(_from, str):
            _from = datetime.strptime(_from, '%Y/%m/%d').date()
        if isinstance(to, str):
            to = datetime.strptime(to, '%Y/%m/%d').date()
    except ValueError as e:
        raise ValueError("Passed date does not match format %Y/%m/%d") from e

    if _from > to:
        raise ValueError("_from cannot be greater than to.")

    finnhub_client = get_finnhub_client()
    combined_news = pd.DataFrame()
    delta = timedelta(days=7)

    start = _from

    while start <= to:
        end = min(start + delta - timedelta(days=1), to)  # ensure we don't go beyond `to`
        
        print(f"Fetching news from {start} to {end}...")  # Optional debug print

        try:
            news = finnhub_client.company_news(
                ticker,
                start.strftime('%Y-%m-%d'),
                end.strftime('%Y-%m-%d')
            )

            if news:
                combined_news = pd.concat([combined_news, pd.DataFrame(news)], ignore_index=True)

            start = end + timedelta(days=1)
        except Exception as e:
            print(f"News fetch failed. {e}")
            break

    return combined_news

def getStockData(ticker, start=None, end=None, period=None, interval='1d'):
    t = yf.Ticker(ticker)
    
    if period:
        return pd.DataFrame(t.history(period=period, interval=interval)).reset_index()
    elif start or end:
        return pd.DataFrame(t.history(start=start, end=end, interval=interval)).reset_index()
    else:
        raise ValueError("You must provide either a period or start/end date.")
    
getStockNews('AAPL', '2025/01/01', '2025/01/20')