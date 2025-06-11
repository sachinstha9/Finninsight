import sys
import os
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from fuzzywuzzy import process
from transformers import pipeline
from data_ingestion_agent import getNewsFromFirebase, getPriceVolumeHistory, getCompanyProfile

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.utils import initFirestore

companyDict = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN"
}

FUZZY_THRESHOLD = 80
ZEROSHOT_THRESHOLD = 0.5
zeroShotClassificationModelName = "facebook/bart-large-mnli"
classifier = pipeline("zero-shot-classification", model=zeroShotClassificationModelName)

def extractTicker(query, companyDict, topN=3):
    fuzzyMatch = process.extractOne(query, companyDict.keys(), limit=topN)
    fuzzyTickers = [companyDict[name] for name, score in fuzzyMatch if score >= FUZZY_THRESHOLD]
    
    if len(fuzzyTickers) < topN:
        candidateLabels = list(companyDict.keys())
        result = classifier(query, candidateLabels)
        zslTickers = [
            companyDict[label] for label, score in zip(result['labels'], result['scores'])
            if score >= ZEROSHOT_THRESHOLD
        ]        
        
        allTickers = list(dict.fromkeys(fuzzyTickers + zslTickers))
        return allTickers[:topN]

    return fuzzyTickers[:topN]

db = initFirestore()
collectionName = 'news'

def buildContext(database, collectionName, ticker):
    news = getNewsFromFirebase(database, collectionName, ticker, window=40)
    priceVolume = getPriceVolumeHistory(ticker, period='14d', interval='1d')
    profile = getCompanyProfile(ticker)
    
    priceVolumeStr = '\n'.join([f"{row.Date.date()}: ${row.Close:.2f} ${row.Volume:.2f}" for _, row in priceVolume.iterrows()])
    newsStr = '\n'.join([f"- {n}" for n in news]) if news else "No recent news found."

    context = f"""
    Company: {profile.get('name')}
    Industry: {profile.get('finnhubIndustry')}
    Country: {profile.get('country')}
    
    Stock Price and volumne (Last 14 days):
    {priceVolumeStr}
    
    Recent News:
    {newsStr}
    """
    
    return context.strip()

print(buildContext(db, collectionName, 'AAPL'))