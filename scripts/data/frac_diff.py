from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm


def get_weights(d, size=10000):
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def get_weights_FFD(d: float,
                    thr: float,
                    max_size: Optional[int] = 10000):
    """Get coefficient for calculating fractional derivative

    Args:
        d (int): the degree of differentiation
        thr (float)
        max_size (int, optional) Defauts to 1e4.\
            Set the maximum size for stability

    Returns:
        array-like
    """
    w = [1.]
    for k in range(1, max_size):
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) <= thr:
            break
        w.append(w_)
    w = np.array(w)
    return w


# def get_weight_ffd(d, thres, lim):
#     w, k = [1.], 1
#     ctr = 0
#     while True:
#         w_ = -w[-1] / k * (d - k + 1)
#         if abs(w_) < thres:
#             break
#         w.append(w_)
#         k += 1
#         ctr += 1
#         if ctr == lim - 1:
#             break
#     w = np.array(w[::-1]).reshape(-1, 1)
#     return w

# def frac_diff_ffd(x, d, thres=1e-5):
#     w = get_weight_ffd(d, thres, len(x))
#     width = len(w) - 1
#     output = []
#     output.extend([0] * width)
#     for i in range(width, len(x)):
#         output.append(np.dot(w.T, x[i - width:i + 1])[0])
#     return np.array(output)


def frac_diff(series: pd.Series,
              d: float,
              thr: Optional[float] = 0.01,
              max_size: Optional[int] = 10000):
    """
    Increasing width window, with treatment of NaNs
    Note 1: For thrs=1, nothing is skipped
    Note 2: d can be any positive fractional, not necessarily
        bounded between [0,1]
    """
    # Compute weights for the longest series
    w = get_weights(d, max_size)

    # Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thr].shape[0]

    # Apply weights to values
    series = series.fillna(method='ffill').dropna() if series.isnull().sum() > 0 else series
    frac_diff_df = pd.Series()
    for iloc in range(skip, len(series)):
        loc = series.index[iloc]
        if not np.isfinite(series.loc[loc][0]).any():
            continue  # exclude NAs
        try:
            frac_diff_df.loc[loc] = np.dot(w[-(iloc + 1):, :].T, series.loc[:loc])[0, 0]
        except Exception as e:
            print(f' > Error: {e}')

    frac_diff_df = frac_diff_df.rename('frac_diff')
    return frac_diff_df


def frac_diff_FFD(series: pd.Series,
                  d: float,
                  lag: Optional[int] = 1,
                  thr: Optional[float] = 1e-5,
                  max_size: Optional[int] = 10000):
    """
    Constant width window (new solution)
    Note 1: thr determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    """
    # Compute weights for the longest series
    series_len = len(series) - 1
    max_size = int(max_size / lag)
    w = get_weights_FFD(d, thr, max_size)
    width = len(w) - 1

    # Apply weights to values
    series = series.fillna(method='ffill').dropna() if series.isnull().sum() > 0 else series
    frac_diff_df = pd.Series()
    for iloc in range(width, series_len):
        loc0, loc1 = series.index[iloc - width], series.index[iloc]
        if not np.isfinite(series.loc[loc1]).any():
            continue  # exclude NAs
        frac_diff_df[loc1] = np.dot(w.T, series.loc[loc0:loc1])
    frac_diff_df = frac_diff_df.rename('frac_diff_ffd')
    end = len(frac_diff_df) + 1
    frac_diff_df = pd.concat((series.iloc[:-end].to_frame(), frac_diff_df.to_frame()))['frac_diff_ffd']
    return frac_diff_df


def get_opt_d(series, ds=None, lag=1, thres=1e-5, max_size=10000,
              p_thres=1e-2, autolag=None, verbose=1, **kwargs):
    """Find minimum value of degree of stationary differntial

    Args:
        series (pd.Series)
        ds (array-like, optional): Defaults to np.linspace(0, 1, 100)\
            Search space of degree.
        lag (int, optional): Defaults to 1.\
            The lag scale when making differential like series.diff(lag)
        thres (float, optional): Defaults to 1e-5.\
            Threshold to determine fixed length window
        p_threds (float, optional): Defaults to 1e-2.\
        auto_lag (str, optional)
        verbose (int, optional): Defaults to 1.\
            If 1 or 2, show the progress bar. 2 for notebook

        kwargs (optional): paramters for ADF

    Returns:
        int: optimal degree
    """
    if ds is None:
        ds = np.linspace(0, 10, 100)
    # Sort to ascending order
    ds = np.array(ds)
    sort_idx = np.argsort(ds)
    ds = ds[sort_idx]
    iter_ds = ds
    opt_d = ds[-1]
    # Compute pval for each d
    for d in iter_ds:
        diff = frac_diff_FFD(series, d=d, thr=thres, max_size=max_size)
        clean_diff = diff.dropna()
        if len(clean_diff) > 0:
            pval = adfuller(clean_diff.values, autolag=autolag, **kwargs)[1]
            if pval < p_thres:
                opt_d = d
                break
        else:
            print(f'Empty diff for d={d}')
    return opt_d


def get_min_frac_diff_ffd(df: pd.DataFrame,
                          feature: str,
                          d_range=range(0, 11)):
    adf_df = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])

    for d in tqdm(d_range):
        frac_diff_df = frac_diff_FFD(df[feature], d, thr=.01)
        frac_diff_df = frac_diff_df.dropna()
        corr = np.corrcoef(df.loc[frac_diff_df.index, feature], frac_diff_df)[0, 1]
        adf_result = adfuller(frac_diff_df, maxlag=1, regression='c', autolag=None)
        adf_df.loc[d] = list(adf_result[:4]) + [adf_result[4]['5%']] + [corr]  # with critical value

    # plot_min_frac_diff(adf_df)

    return adf_df
