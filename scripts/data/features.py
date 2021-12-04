from typing import List
import numpy as np
import pandas as pd
import logging
from scripts.data.frac_diff import frac_diff_FFD


def add_variance(df: pd.DataFrame,
                 windows: List[int],
                 feature: str):
    for w in windows:
        df[f'variance_{w}_{feature}'] = df[feature].rolling(w).var()

    return df


def add_pct_change(df: pd.DataFrame,
                   windows: List[int],
                   feature: str):
    for w in windows:
        df[f'pct_change_{w}_{feature}'] = df[feature].pct_change(w)

    return df


def add_log(df: pd.DataFrame,
            feature: str):
    df[f'log_{feature}'] = np.log(df[feature])

    return df


def add_frac_diff_FDD(df: pd.DataFrame,
                      feature: str,
                      d: float,
                      thr: float):
    df[f'fd_{feature}'] = frac_diff_FFD(df[feature], d=d, thr=thr)
    return df


def add_return(df: pd.DataFrame,
               feature: str,
               lags: List[int] = [1]):
    for l in lags:
        df[f'return_{l}_{feature}'] = df['Close']/df['Close'].shift(l)

    return df


def add_volatility(df: pd.DataFrame,
                   feature: str,
                   windows: List[int]):

    for w in windows:
        df[f'volat_{w}_{feature}'] = df[feature].std() * w * 0.5

    return df
