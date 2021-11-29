from typing import Text

import pandas as pd


def anomaly_pct_change(df: pd.DataFrame,
                       window: int,
                       feature: Text,
                       alpha: float,
                       variance: float):
    pct_feature = f'pct_change_{window}_{feature}'
    variance_feature = f'variance_{window}_{feature}'

    upper_q = df[pct_feature].quantile(1 - alpha)
    lower_q = df[pct_feature].quantile(alpha)

    df['anomaly'] = None
    df.loc[(df[pct_feature] <= lower_q) & (df[variance_feature] > variance), 'anomaly'] = -1
    df.loc[(df[pct_feature] >= upper_q) & (df[variance_feature] > variance), 'anomaly'] = 1
    df['anomaly'] = df['anomaly'].backfill(limit=window)
    df['anomaly'] = df['anomaly'].fillna(0)

    return df
