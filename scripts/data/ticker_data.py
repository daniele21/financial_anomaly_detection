from typing import Text

import pandas as pd

from scripts.data.constants import DEFAULT_FEATURE, DEFAULT_ALPHA, DEFAULT_MIN_VARIANCE
from scripts.data.extract import extract_data
from scripts.data.features import add_pct_change, add_variance, add_log, add_volatility
from scripts.dataset.preprocessing import preprocessing_version_1
from scripts.labeling.pct_change import anomaly_pct_change
from scripts.utils.exceptions import PreprocessingVersionException
from scripts.utils.logger import setup_logger


class TickerData:

    def __init__(self, ticker: Text, window: int):
        self.name = ticker
        self.logger = setup_logger(f'{self.name} logger')
        self.data = self._init_data(ticker, window)
        self.preprocessed_data = None
        self.labeled_data = None
        self.window = window

    def _init_data(self,
                   ticker: Text,
                   window: int,
                   start_date: Text = None,
                   end_date: Text = None,
                   ) -> pd.DataFrame:
        self.logger.info(f' > Init Data: {self.name}')
        data = extract_data(ticker, start_date=start_date, end_date=end_date)

        lags = [1, 3, 5, 10]
        volatility_windows = [21, 21 * 3, 21 * 6, 252]

        # Adding pct_change on close feature
        data = add_pct_change(data, feature='Close', windows=[window])
        data = add_pct_change(data, feature='Volume', windows=[window])
        data = add_variance(data, feature='Close', windows=[window])
        data = add_variance(data, feature='Volume', windows=[window])
        data = add_log(data, feature='Close')
        data = add_volatility(data, feature='Close', windows=volatility_windows)

        return data

    def preprocessing(self, version: int):
        if self.labeled_data is None:
            self.logger.warning(f' > No labeled data yet. Operation cancelled!')
            return None

        if version == 1:
            self.preprocessed_data = preprocessing_version_1(self.labeled_data)
            return self.preprocessed_data
        else:
            error = f' > No valid version for preprocessing of data'
            self.logger.error(error)
            raise PreprocessingVersionException(error)

    def label_data(self,
                   feature: Text = DEFAULT_FEATURE,
                   alpha: float = DEFAULT_ALPHA,
                   variance: float = DEFAULT_MIN_VARIANCE):
        self.logger.info(f' > Labeling data')
        labeled_data = anomaly_pct_change(self.data,
                                          self.window,
                                          feature,
                                          alpha,
                                          variance)
        self.labeled_data = labeled_data

        return labeled_data

    # def get_windows(self, window: int = None) -> Dict:
    #     if self.preprocessed_data is None:
    #         self.logger.error(f' > No preprocessed data found! Use .preprocessing()')
    #         return None
    #
    #     normal_windows = get_windows(self.preprocessed_data, window, anomaly=0)
    #     pos_anom_windows = get_windows(self.preprocessed_data, window, anomaly=1)
    #     neg_anom_windows = get_windows(self.preprocessed_data, window, anomaly=-1)
    #
    #     return {'pos': pos_anom_windows,
    #             'neg': neg_anom_windows,
    #             'norm': normal_windows}
