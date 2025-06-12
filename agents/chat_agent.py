import sys
import os
import heapq
import yfinance as yf
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForCausalLM
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
from common.config import FINNHUB_API_KEY, EMBEDDING_MODEL_NAME, SUMMARIZATION_MODEL_NAME, ZERO_SHOT_CLASSIFIER_MODEL_NAME, TEXT_GENERATION_MODEL_NAME, HF_TOKEN

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
embeddingModel = embeddingModel.to('cuda')
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

chatHistory = [
    {
        "user": "How is apple and microsoft stock doing?", 
        "advisor": "As of today, Apple stock has shown steady growth with positive earnings reports driving investor confidence. Microsoft is also performing well, supported by its cloud services and enterprise software growth. Both stocks are considered strong in the tech sector."
    },
    {
        "user": "What do you think of investing in apple?", 
        "advisor": "Investing in Apple is generally considered a solid choice due to its strong brand, consistent revenue growth, and diversified product ecosystem. However, it's important to monitor market conditions and competition before making a decision."
    },
    {
        "user": "What do you think if microsoft?", 
        "advisor": "Microsoft has been a reliable performer, especially with its focus on cloud computing and enterprise solutions like Azure and Office 365. It offers good long-term growth potential, although investors should watch for market fluctuations."
    }
]

textGenTokenizer = AutoTokenizer.from_pretrained(TEXT_GENERATION_MODEL_NAME, token=HF_TOKEN)
textGenModel = AutoModelForCausalLM.from_pretrained(TEXT_GENERATION_MODEL_NAME, device_map="auto", trust_remote_code=True, token=HF_TOKEN)
textGenerator = pipeline("text-generation", model=textGenModel, tokenizer=textGenTokenizer)

RECENT_TURNS_TO_KEEP = 5
MAX_TOKENS = 2048
SAFETY_BUFFER = 512

SUMMARIZER_MAX_TOKENS = summarizer.tokenizer.model_max_length  # Or summarizer.tokenizer.model_max_length if accessible

def estimateTokens(text):
    return len(textGenTokenizer.encode(text))

def chunkTextForSummarizer(text, tokenizer, maxTokens=1024):
    words = text.split()
    chunks = []
    currentChunk = []

    for word in words:
        currentChunk.append(word)
        tokenized = tokenizer(" ".join(currentChunk), return_tensors='pt', truncation=False)
        tokenCount = len(tokenized['input_ids'][0])
        
        if tokenCount >= maxTokens:
            currentChunk.pop()
            chunks.append(" ".join(currentChunk))
            currentChunk = [word] 
            
    if currentChunk:
        chunks.append(" ".join(currentChunk))

    return chunks

def summarizeChatHistory(text):
    if not text.strip():
        return ""

    chunks = chunkTextForSummarizer(text, summarizer.tokenizer, maxTokens=1024)
    summaries = []

    for chunkText in chunks:
        if not chunkText or len(chunkText.split()) < 10:
            continue

        try:
            inputLen = len(chunkText.split())
            maxLen = max(10, min(256, inputLen // 2))

            summary = summarizer(chunkText, max_length=maxLen, min_length=10, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            print("Summarization error: ", e)
            continue

    return "\n".join(summaries)


def buildChatHistoryPrompt(chatHistory, query):
    fullHistoryText = ""
    for e in chatHistory:
        fullHistoryText += f"user: {e['user']}\nadvisor: {e['advisor']}\n\n"

    totalTokens = estimateTokens(fullHistoryText) + estimateTokens(query) + SAFETY_BUFFER

    if totalTokens > MAX_TOKENS:
        oldChats = chatHistory[:-RECENT_TURNS_TO_KEEP] if len(chatHistory) > RECENT_TURNS_TO_KEEP else []
        recentChats = chatHistory[-RECENT_TURNS_TO_KEEP:]

        oldText = ""
        for e in oldChats:
            oldText += f"user: {e['user']}\nadvisor: {e['advisor']}\n\n"

        oldChatSummarized = summarizeChatHistory(oldText) if oldText.strip() else "No prior context to summarize."

        recentText = ""
        for e in recentChats:
            recentText += f"user: {e['user']}\nadvisor: {e['advisor']}\n\n"

        return f"""
Summary of earlier conversation:
{oldChatSummarized}

Recent chat:
{recentText}"""

    return f"""
Recent chat:
{fullHistoryText}"""
        
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


def chatAgent(database, collectionName, query):    
    context = buildContext(database, collectionName, query)
    chatHistoryPrompt = buildChatHistoryPrompt(chatHistory, query)
    
    prompt = f"""
You are a expert financial advisor. Use the provided context to answer the user's question accurately and clearly.

Context:
{context}

{chatHistoryPrompt}

user: {query}
advisor:"""

    print(prompt)

    response = textGenerator(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)[0]["generated_text"]
    
    answer = response.split("advisor:")[-1].strip()
    return answer

db = initFirestore()
collectionName = 'news'
query = "How is apple and microsoft stock doing?"
print(chatAgent(db, collectionName, query))
