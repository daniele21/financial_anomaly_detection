import unittest
import pandas as pd
from yfinance import Ticker
import numpy as np
from scripts.data.extract import extract_data
from scripts.data.features import add_pct_change, add_variance, add_return, add_log, add_volatility
from scripts.data.frac_diff import get_weights, get_weights_FFD, frac_diff, frac_diff_FFD, get_opt_d, \
    get_min_frac_diff_ffd
from scripts.visualization.frac_diff_plot import plot_min_frac_diff


class TestAnalysis(unittest.TestCase):
    def test_get_weights(self):
        d = 0.5
        size = 10000
        w = get_weights(d, size)

        self.assertIsNotNone(w)

    def test_get_weights_FFD(self):
        d = 0.5
        size = 10000
        thres = 0.01
        w = get_weights_FFD(d, thres, size)

        self.assertIsNotNone(w)

    def test_frac_diff(self):
        d = 0.3
        thres = 0.01
        df = Ticker('ADA-EUR').history(period='max')

        fd = frac_diff(df['Close'].to_frame(), d, thres)

        self.assertIsNotNone(fd)

    def test_frac_diff_FFD(self):
        d = 0.3
        thres = 0.01
        df = Ticker('ADA-EUR').history(period='max')
        # close = numpy.log(df['Close'])
        close = df['Close']

        fd = frac_diff_FFD(close, d, thr=thres)

        self.assertIsNotNone(fd)

    def test_get_opt_d(self):
        d = 0.5
        thres = 0.01
        df = Ticker('ADA-EUR').history(period='max')

        output = get_opt_d(df['Close'])

        self.assertIsNotNone(output)

    def test_frac_diff_analysis(self):
        ticker = 'ADA-EUR'
        data = extract_data(ticker)
        window = 5
        lags = [1, 3, 5, 10]
        volatility_windows = [21, 21 * 3, 21 * 6, 252]

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

        feature = 'log_return_1_Close'
        thr = 0.01
        ds = np.linspace(0, 1, 10)

        results = {}
        for d in ds:
            frac_df = frac_diff_FFD(data[feature], d=d, thr=thr)
            results[f'frac_diff_{d}'] = frac_df.values

        result_df = pd.DataFrame(results)

        self.assertIsNotNone(result_df)

    def test_get_min_frac_diff_ffd(self):
        ticker = 'ADA-EUR'
        data = extract_data(ticker)
        window = 5
        lags = [1, 3, 5, 10]
        volatility_windows = [21, 21 * 3, 21 * 6, 252]

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

        feature = 'Close'
        thr = 0.01
        ds = np.linspace(0, 1, 11)

        adf_df = get_min_frac_diff_ffd(data, feature, ds)
        plot_min_frac_diff(adf_df, title=ticker)

        self.assertIsNotNone(adf_df)


if __name__ == '__main__':
    unittest.main()
