import yfinance as yf
import pandas as pd
import duckdb
import finnhub
import json
import logging
from datetime import datetime, timedelta, timezone

from common.config import (
    DUCKDB_DATABASE_FILE, DUCKDB_TABLE_HISTORICAL_STOCK, DUCKDB_TABLE_NEWS_ARTICLE, FINNHUB_API_KEY
)