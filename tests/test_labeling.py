import unittest

from scripts.analysis.feature_engineering import add_pct_change, add_variance
from scripts.data.extract import extract_data
from scripts.labeling.pct_change import anomaly_pct_change
from scripts.visualization.plot_labeling import plot_anomaly_labels


class TestLabeling(unittest.TestCase):
    ticker = 'ADA-EUR'
    ada_data = extract_data(ticker)
    windows = [5, 10]
    ada_data = add_pct_change(ada_data, windows)
    ada_data = add_variance(ada_data, windows)

    def test_anomaly_pct_change(self):
        alpha = 0.05
        w = 5
        min_variance_on_window = 0.005
        df = anomaly_pct_change(self.ada_data,
                                window=w,
                                feature='close',
                                alpha=alpha,
                                variance=min_variance_on_window)
        plot_anomaly_labels(df, title=self.ticker)

        # self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
