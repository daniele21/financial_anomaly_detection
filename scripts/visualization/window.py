from typing import List, Text
from matplotlib import pyplot as plt
import matplotlib.dates as md


def plot_windows(windows_list: List,
                 title: Text):
    plt.title(title)
    ax = plt.gca()
    xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    plt.xticks(rotation=75)

    for window_dict in windows_list:
        index = window_dict['date']
        close = window_dict['close']

        plt.plot(x=index, y=close)
        plt.scatter(x=index[-1], y=close[-1], marker='|')

    plt.show()
