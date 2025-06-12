import sys
import os
import heapq
import yfinance as yf
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForCasualLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from fuzzywuzzy import process
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from datetime import datetime, timedelta

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_ingestion_agent import getNewsFromFirebase, getCompanyBasicFinancials
from common.utils import initFirestore
from common.config import FINNHUB_API_KEY, EMBEDDING_MODEL_NAME, SUMMARIZATION_MODEL_NAME, ZERO_SHOT_CLASSIFIER_MODEL_NAME, TEXT_GENERATION_MODEL_NAME

companyDict = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN"
}

FUZZY_THRESHOLD = 80
ZEROSHOT_THRESHOLD = 0.5
zeroShotClassifier = pipeline("zero-shot-classification", model=ZERO_SHOT_CLASSIFIER_MODEL_NAME)
def extractTicker(query, companyDict, topN=3):
    fuzzyMatch = process.extract(query, companyDict.keys(), limit=topN)
    fuzzyTickers = [companyDict[name] for name, score in fuzzyMatch if score >= FUZZY_THRESHOLD]
    
    if len(fuzzyTickers) < topN:
        candidateLabels = list(companyDict.keys())
        result = zeroShotClassifier(query, candidateLabels)
        zslTickers = [
            companyDict[label] for label, score in zip(result['labels'], result['scores'])
            if score >= ZEROSHOT_THRESHOLD
        ]        
        
        allTickers = list(dict.fromkeys(fuzzyTickers + zslTickers))
        return allTickers[:topN]

    return fuzzyTickers[:topN]


embeddingModel = SentenceTransformer(EMBEDDING_MODEL_NAME)
summarizer = pipeline("summarization", SUMMARIZATION_MODEL_NAME)

# Get top N most relevant news based on semantic similarity
def getRelevantNews(query, news, topN=10):
    if not news:
        return []
    
    queryEmbedding = embeddingModel.encode(query, convert_to_tensor=True)
    newsEmbeddings = embeddingModel.encode(news, convert_to_tensor=True)
    
    cosineScores = util.pytorch_cos_sim(queryEmbedding, newsEmbeddings)[0]
    
    topIndices = heapq.nlargest(topN, range(len(cosineScores)), key=lambda i: cosineScores[i])
    return [news[i] for i in topIndices]

def summarizeRelevantNews(query, news, topN=10, maxLength=150):
    news = getRelevantNews(query, news, topN)
    combinedText = " ".join(news)
    summary = summarizer(combinedText, max_length=maxLength, min_length=50, do_sample=False)
    return summary[0]['summary_text']

def summarizePriceVolumeChange(ticker):
    stock =yf.Ticker(ticker)
    end = datetime.today()
    start7d = end - timedelta(days=7)
    start30d = end - timedelta(days=30)
    
    hist7d = stock.history(start=start7d.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
    hist30d = stock.history(start=start30d.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))

    if hist7d.empty or hist30d.empty:
        return "No recent price of Volume data available."
    
    startPrice7d = hist7d['Close'].iloc[0]
    endPrice7d = hist7d['Close'].iloc[-1]
    change7d = endPrice7d - startPrice7d
    percentChange7d = (change7d / startPrice7d) * 100
    direction7d = "increased" if change7d > 0 else "decreased"

    startPrice30d = hist30d['Close'].iloc[0]
    endPrice30d = hist30d['Close'].iloc[-1]
    change30d = endPrice30d - startPrice30d
    percentChange30d = (change30d / startPrice30d) * 100
    direction30d = "increased" if change30d > 0 else "decreased"


    avgVolume7d = hist7d["Volume"].mean()
    avgVolume30d = hist30d["Volume"].mean()
    
    if avgVolume30d == 0:
        volumeTrend = "Volume trend not available."
    else:
        volumeRatio = avgVolume7d / avgVolume30d
        if volumeRatio > 1.2:
            volumeTrend = "Trading volume is above average, indicating high intrest."
        elif volumeRatio < 0.8:
            volumeTrend = "Trading volume is below average, indicating low intrest."
        else:
            volumeTrend = "Trading volume is within the normal range."
    
    summary = f"""
In the last 7 days, the stock price has {direction7d} by {abs(percentChange7d):.2f}%
(from ${startPrice7d:.2f} to ${endPrice7d:.2f}).
Average daily volume: {avgVolume7d:,.0f} shares

In the last 30 days, the stock price has {direction30d} by {abs(percentChange30d):.2f}%
(from ${startPrice30d:.2f} to ${endPrice30d:.2f}).
Average daily volume: {avgVolume30d:,.0f} shares

When comparing average volume of 7 day and average volume of 30 day, {volumeTrend}"""
    
    return summary

def buildContext(database, collectionName, query):
    tickers = extractTicker(query, companyDict)
    if not tickers:
        return "No valid ticker found in the query/"
    
    contextBlocks = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            companyName = info.get("shortName", ticker)
            sector = info.get("sector", "N/A")
            industry = info.get("industry", "N/A")
            marketCap = info.get("marketCap", "N/A")
            currentPrice = info.get("CurrentPrice", "N/A")
            
            news = getNewsFromFirebase(database, collectionName, ticker, 7)
            newsSummary = summarizeRelevantNews(query, news, 15, 150)
            
            priceVolSummary = summarizePriceVolumeChange(ticker)
            
            basicFinancials = getCompanyBasicFinancials(ticker)
            
            block = f"""
Company: {companyName}
Ticker: {ticker}
Sector: {sector}
Industry: {industry}
Market Capitalization: {marketCap}
Current Price: ${currentPrice}

Price and Volume Summary:
{priceVolSummary}

Recent News Summary:
{newsSummary}

Some Basic Financials:
P/E ratio: {basicFinancials.get("peNormalizedAnnual", "N/A")}
EPS (Annual): {basicFinancials.get("epsNormalizedAnnual", "N/A")}
Net Profit Margin: {basicFinancials.get("netProfitMarginAnnual", "N/A")}%
Debt/Equity: {basicFinancials.get("totalDebt/totalEquityAnnual", "N/A")}
Return on Equity: {basicFinancials.get("roeAnnual", "N/A")}
Revenue: {basicFinancials.get("revenuePerShareAnnual", "N/A")}"""
            
            contextBlocks.append(block.strip())
        except Exception as e:
            contextBlocks.append(f"Error loading data for {ticker}: {str(e)}")

    return "\n\n--\n\n".join(contextBlocks)
    return tickers

db = initFirestore()
collectionName = 'news'
query = "How is apple and microsoft stock doing?"
print(buildContext(db, collectionName, query))
print(extractTicker(query, companyDict))

tokenizer = AutoTokenizer.from_pretrained(TEXT_GENERATION_MODEL_NAME)
textGenModel = AutoModelForCasualLM.from_pretrained(TEXT_GENERATION_MODEL_NAME, device_map="auto", trust_remote_code=True)