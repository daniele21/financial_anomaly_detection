import logging
from typing import Text

import pandas as pd

from scripts.data.constants import DEFAULT_FEATURE, DEFAULT_ALPHA, DEFAULT_MIN_VARIANCE
from scripts.data.extract import extract_data
from scripts.data.features import add_pct_change, add_variance, add_log, add_volatility, add_return, add_frac_diff_FDD
from scripts.data.windowing import window_data
from scripts.labeling.pct_change import anomaly_pct_change
from scripts.utils.logger import setup_logger


class TickerData:

    def __init__(self, ticker: Text, window: int):
        self.name = ticker
        self.logger = setup_logger(f'{self.name} logger')
        self.data = self._init_data(ticker, window)
        self.labeled_data = None
        self.window = window

    def _init_data(self, ticker: Text, window: int) -> pd.DataFrame:
        self.logger.info(f' > Init Data: {self.name}')
        data = extract_data(ticker)

        lags = [1,3,5,10]
        volatility_windows = [21, 21*3, 21*6, 252]

        # Adding pct_change on close feature
        data = add_pct_change(data, feature='Close', windows=[window])
        data = add_pct_change(data, feature='Volume', windows=[window])
        data = add_variance(data, feature='Close', windows=[window])
        data = add_variance(data, feature='Volume', windows=[window])
        data = add_return(data, feature='Close', lags=lags)
        data = add_log(data, feature='Close')
        for l in lags:
            data = add_log(data, feature=f'return_{l}_Close')
        data = add_volatility(data, feature='return_1_Close', windows=volatility_windows)
        data = add_volatility(data, feature='log_return_1_Close', windows=volatility_windows)
        data = add_frac_diff_FDD(data, feature='log_return_1_Close', windows=volatility_windows)


        # Adding windowed variance
        data = add_variance(data, [window])

        return data

    def label_data(self,
                   feature: Text = DEFAULT_FEATURE,
                   alpha: float = DEFAULT_ALPHA,
                   variance: float = DEFAULT_MIN_VARIANCE):
        self.logger.info(f' > Labeling data')
        labeled_data = anomaly_pct_change(self.data, self.window, feature, alpha, variance)
        self.labeled_data = labeled_data

        return labeled_data

    def get_positive_anomalous_windows(self):

        if self.labeled_data is None:
            self.logger.error(f' > No labeled data found! Use .label_data()')
            return None

        windows_list = window_data(self.labeled_data, self.window, anomaly=1)
        return windows_list

    def get_negative_anomalous_windows(self):

        if self.labeled_data is None:
            self.logger.error(f' > No labeled data found! Use .label_data()')
            return None

        windows_list = window_data(self.labeled_data, self.window, anomaly=-1)
        return windows_list

    def get_normal_windows(self):
        if self.labeled_data is None:
            self.logger.error(f' > No labeled data found! Use .label_data()')
            return None

        windows_list = window_data(self.labeled_data, self.window, anomaly=0)
        return windows_list



