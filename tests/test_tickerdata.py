import unittest

from scripts.data.ticker_data import TickerData
from scripts.utils.logger import setup_logger

logger = setup_logger('Ticker Data test')


class TestTickerData(unittest.TestCase):
    ticker = 'ETH-EUR'
    window = 5

    ticker_data = TickerData(ticker=ticker,
                             window=window)

    def test_init(self):
        ticker_data = TickerData(ticker=self.ticker,
                                 window=self.window)

        self.assertIsNotNone(ticker_data)

    def test_labeling(self):
        labeled_data = self.ticker_data.label_data()

        self.assertIsNotNone(labeled_data)

    def test_get_anomalous_windows(self):
        self.ticker_data.label_data()
        neg_anomaly_df = self.ticker_data.get_negative_anomalous_windows()
        pos_anomaly_df = self.ticker_data.get_positive_anomalous_windows()

        self.assertIsNotNone(neg_anomaly_df)
        self.assertIsNotNone(pos_anomaly_df)


if __name__ == '__main__':
    unittest.main()
