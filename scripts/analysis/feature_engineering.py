from typing import List

import pandas as pd


def add_variance(df: pd.DataFrame,
                 windows: List[int]):
    for w in windows:
        df[f'variance_{w}_close'] = df['Close'].rolling(w).var()
        df[f'variance_{w}_volume'] = df['Volume'].rolling(w).var()

    return df


def add_pct_change(df: pd.DataFrame,
                   windows: List[int]):
    for w in windows:
        df[f'pct_change_{w}_close'] = df['Close'].pct_change(w)
        df[f'pct_change_{w}_volume'] = df['Volume'].pct_change(w)

    return df


