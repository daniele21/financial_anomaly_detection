import logging
from typing import Text

import pandas as pd

from scripts.analysis.feature_engineering import add_pct_change, add_variance
from scripts.data.constants import DEFAULT_FEATURE, DEFAULT_ALPHA, DEFAULT_MIN_VARIANCE
from scripts.data.extract import extract_data
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

        # Adding pct_change on close feature
        data = add_pct_change(data, [window])

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

        if not self.labeled_data:
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
        if not self.labeled_data:
            self.logger.error(f' > No labeled data found! Use .label_data()')
            return None

        windows_list = window_data(self.labeled_data, self.window, anomaly=0)
        return windows_list



