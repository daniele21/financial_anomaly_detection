from typing import Text

import pandas as pd
import matplotlib.pyplot as plt


def plot_anomaly_labels(df: pd.DataFrame,
                        title: Text):

    pos_vector, neg_vector = [], []
    for i, row in df.iterrows():
        if row['anomaly'] == 1:
            neg_vector.append(None)
            pos_vector.append(row['Close'])
        elif row['anomaly'] == -1:
            pos_vector.append(None)
            neg_vector.append(row['Close'])
        else:
            pos_vector.append(None)
            neg_vector.append(None)

    pos_anomalies_df = pd.DataFrame(pos_vector, index=df.index)
    neg_anomalies_df = pd.DataFrame(neg_vector, index=df.index)

    plt.title(title, fontsize=20)
    plt.plot(df['Close'], c='b', label='Data')
    plt.plot(pos_anomalies_df, c='g', label='Positive Anomaly')
    plt.plot(neg_anomalies_df, c='r', label='Negative Anomaly')

    plt.legend()

    plt.show()
