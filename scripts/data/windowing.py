import pandas as pd


def window_data(df: pd.DataFrame,
                window: int,
                anomaly: int):
    assert anomaly in [-1, 0, 1], f' > No valid anomaly: {anomaly}'

    windows_list = []

    for i in range(len(df)):
        start = i
        end = i + window if i + window < len(df) else None

        if end is None:
            continue

        window_df = df.iloc[start: end]
        if (window_df['anomaly'] == anomaly).all():
            window_dict = {'date': window_df.index.to_list(),
                           'close': window_df['Close'].to_list()}
            windows_list.append(window_dict)

    return windows_list
