import pandas as pd
import matplotlib.pyplot as plt


def plot_min_frac_diff(adf_df: pd.DataFrame,
                       title: str):

    # adf_df['adfStat'].plot(secondary_y='adfStat', xlabel='Fractional Differentiation')
    # adf_df['corr'].plot(label='correlation', ylabel='Correlation', xlabel='Fractional Differentiation')

    adf_df[['adfStat', 'corr']].plot(ylabel='Correlation',
                                     secondary_y='adfStat',
                                     xlabel='Fractional Differentiation')
    plt.axhline(adf_df['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted',
                label='95% confidence')

    # plt.xlabel('Fractional Differentiation')
    # plt.ylabel('Correlation')
    plt.title(title)
    # plt.legend()
    plt.tight_layout()
    plt.show()
