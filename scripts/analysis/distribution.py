import pandas as pd

def get_quantile(df: pd.DataFrame,
                 q: float):
    df.quantile(q)