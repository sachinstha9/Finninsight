import os
from dotenv import load_dotenv
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
SENTIMENT_ANALYSIS_MODEL_NAME = "ProsusAI/finbert"