from typing import Text

from yfinance import Ticker


def extract_data(ticker: Text):
    ticker_obj = Ticker(ticker)

    ticker_history = None
    while ticker_history is None:
        ticker_history = ticker_obj.history(period='max',
                                            interval="1d",
                                            timeout=5)
    return ticker_history
