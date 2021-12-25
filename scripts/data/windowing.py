import pandas as pd


def get_windows(df: pd.DataFrame,
                window: int,
                anomaly: int):
    assert anomaly in [-1, 0, 1], f' > No valid anomaly: {anomaly}'

    windows_list = []

    k = 0
    for i in range(len(df)):
        start = i
        end = i + window if i + window < len(df) else None

        if end is None:
            continue

        window_df = df.iloc[start: end]

        # Normal window if all in windows are normal
        # Anomalous window if at least one is anomalous
        if (anomaly == 0 and (window_df['anomaly'] == anomaly).all()) or \
                (anomaly != 0 and (window_df['anomaly'] == anomaly).any()):
            windows_list.append(window_df)

    return windows_list
