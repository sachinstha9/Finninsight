import os
from dotenv import load_dotenv
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
SENTIMENT_ANALYSIS_MODEL_NAME = "ProsusAI/finbert"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SUMMARIZATION_MODEL_NAME = "facebook/bart-large-cnn"
ZERO_SHOT_CLASSIFIER_MODEL_NAME = "facebook/bart-large-mnli"
TEXT_GENERATION_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
FIREBASE_KEY_JSON_PATH = "../firebase_key.json"
