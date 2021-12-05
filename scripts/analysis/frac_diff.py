import pandas as pd
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm
import numpy as np

from scripts.data.frac_diff import frac_diff_FFD


def frac_diff_ffd_analysis(df: pd.DataFrame,
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


def get_min_diff_ffd(adf_df: pd.DataFrame,
                     min_pvalue: float = 0.01):
    d_opt = adf_df[adf_df['pVal'] < min_pvalue].index[0]

    return d_opt

