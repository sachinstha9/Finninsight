import sys
import os
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.config import SENTIMENT_ANALYSIS_MODEL_NAME

def getSentimentAnalysisModelAndTokenizer():
    try:
        sentiment_analysis_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_ANALYSIS_MODEL_NAME)
        sentiment_analysis_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_ANALYSIS_MODEL_NAME)
        
        sentiment_analysis_device = "cuda" if torch.cuda.is_available() else "cpu"
        sentiment_analysis_model.to(sentiment_analysis_device)
        sentiment_analysis_model.eval()
        
        return sentiment_analysis_tokenizer, sentiment_analysis_model, sentiment_analysis_device

    except Exception as e:
        print(f"[ERROR] Sentiment analysis model loading failed: {e}")
        return None, None, None

def analyzeSentiment(texts, tokenizer, model, device):
    if isinstance(texts, str):
        texts = [texts]
    
    valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    
    if not valid_texts:
        return [{'positive_sentiment': 0.0, 'negative_sentiment': 0.0, 'neutral_sentiment': 0.0}] * len(texts)

    try:
        inputs = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

        return probs[:, 0], probs[:, 1], probs[:, 2]

    except Exception as e:
        print("Batch sentiment analysis error:", e)
        return np.zeros(len(texts)), np.zeros(len(texts)), np.zeros(len(texts))

