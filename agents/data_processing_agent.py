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
    
    result_list = [{'positive_sentiment': 0.0, 'negative_sentiment': 0.0, 'neutral_sentiment': 0.0}] * len(texts)

    valid_texts = []
    valid_indices = []

    for i, t in enumerate(texts):
        if isinstance(t, str) and t.strip():
            valid_texts.append(t)
            valid_indices.append(i)

    if not valid_texts:
        return result_list

    try:
        inputs = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = F.softmax(logits, dim=1).cpu().numpy()

        for idx, sentiment in zip(valid_indices, probs):
            result_list[idx] = {
                'positive_sentiment': float(sentiment[0]),
                'negative_sentiment': float(sentiment[1]),
                'neutral_sentiment': float(sentiment[2]),
            }

        return result_list

    except Exception as e:
        print("Batch sentiment analysis error:", e)
        return result_list

