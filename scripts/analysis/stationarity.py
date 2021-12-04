import statsmodels.api as sm


def adf_test(data,
             **kwargs):
    return sm.tsa.stattools.adfuller(data, **kwargs)


def is_stationary(data,
                  alpha: float = 0.05):
    test = adf_test(data)
    p_value = test[1]

    return p_value < alpha


