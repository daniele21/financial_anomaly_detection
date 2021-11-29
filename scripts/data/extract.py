from typing import Text

from yfinance import Ticker


def extract_data(ticker: Text):
    ticker_obj = Ticker(ticker)
    return ticker_obj.history(period='max', interval="1d")


