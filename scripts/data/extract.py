from typing import Text

import pandas_datareader as pdr


def extract_data(ticker: Text,
                 start_date: Text,
                 end_date: Text):
    data = pdr.get_data_yahoo(ticker,
                              start=start_date,
                              end=end_date)
    return data
