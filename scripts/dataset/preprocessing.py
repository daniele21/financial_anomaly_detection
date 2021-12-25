import pandas as pd

from scripts.data.features import add_frac_diff_FDD


def preprocessing_version_1(df: pd.DataFrame):
    data = df.copy(deep=True)

    data = add_frac_diff_FDD(data, feature='Close', d=0.4, thr=0.01)
    data = add_frac_diff_FDD(data, feature='Open', d=0.5, thr=0.01)
    data = add_frac_diff_FDD(data, feature='Low', d=0.5, thr=0.01)
    data = add_frac_diff_FDD(data, feature='High', d=0.5, thr=0.01)
    data = add_frac_diff_FDD(data, feature='log_Close', d=0.5, thr=0.01)

    cols = ['fd_Close', 'fd_Open', 'fd_Low',
            'fd_High', 'fd_log_Close', 'Volume',
            'anomaly']

    return data[cols]
