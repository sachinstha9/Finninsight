import os
from dotenv import load_dotenv
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
SENTIMENT_ANALYSIS_MODEL_NAME = "ProsusAI/finbert"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# SUMMARIZATION_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
SUMMARIZATION_MODEL_NAME = "facebook/bart-large-cnn"
ZERO_SHOT_CLASSIFIER_MODEL_NAME = "facebook/bart-large-mnli"
# TEXT_GENERATION_MODEL_NAME = "EleutherAI/gpt-neo-125m"
TEXT_GENERATION_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
FIREBASE_KEY_JSON_PATH = "../firebase_key.json"