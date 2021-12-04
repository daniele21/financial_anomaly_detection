# FAD - Financial Anomaly Detection based on Nowcasting

![Licence](https://img.shields.io/badge/Licence-MIT-orange)

Libraries: 

![Pandas](https://img.shields.io/badge/Pandas-1.3.4-brightgreen)
![Statsmodels](https://img.shields.io/badge/Statsmodels-0.13.1-brightgreen)

Dependences:

[![Python](https://img.shields.io/badge/Python-3.8-yellow)](https://github.com/daniele21/Genre_Detection/blob/master/dependences.md)

## Contents
- [Description](#description)
- [Goal](#goal)
- [Dataset](#dataset)
- [Anomaly Definition](#anomaly-definition)
  - [Price Change](#price-change)
  - [Triple Barrier](#triple-barrier)
  - [Meta-Labeling](#meta-labeling)
- [Analysis](#analysis)
  - [Feature Engineering](#feature-engineering)
  - [Fractionally Differentiated Features](#fractionally-differentiated-features)
  - [Feature Selection](#feature-selection)
- [Models](#models)

## Description
In this project I want to develop a AI algorithm able to understand when an **anomaly** affects the _price trend_. Instead of solving the forecasting problem, I want to focus on the **nowcasting problem**, _i.e. prediction of the present, the very near future_. 

Forecasting problem are very difficult to be launch in real-case use, because they are not able to predict an external shock, given by a news or anything else, infact they are very poor againts the many external sources of shock. 

**Nowcasting** does not take the responsability to predict the future, as forecasting does, but it tries to predict the actual state or the near future. For this reason, I'm developing this project for **preventing some shock** that could affect the price trend of a financial product.

## Goal
Given a window, the **FAD model** detects if the window is **normal** or **anomalous**

## Dataset
The dataset is provided by the yahoo finance API _(yfinance)_. I focus mainly on **Crypto** data.

## Anomaly Definition
### Price Change Based
#### Assumptions
Let's define the **price change** *pc* on **window** *w* as following:

- <img src="https://bit.ly/3GfEP5u" align="center" border="0" alt="pc_w = \frac{Price_{i+w} - Price_i}{Price_i}" width="199" height="46" />

Now let's take the 95% quantile of **pc_w** and set the minimum variance to taking into account:

- <img src="https://bit.ly/3rBOwYa" align="center" border="0" alt="q = Quantile(pc_w, 0.95)" width="204" height="19" />


- <img src="https://bit.ly/3xWsU9H" align="center" border="0" alt="var_{min}" width="60" height="15" />


#### The anomaly is referred to the window **w**, that pct_change_w is computed on

Example of anomaly labeling:
![Labeling](https://raw.github.com/daniele21/financial_anomaly_detection/main/static/Anomaly%20labeling.png?raw=true)

### Triple Barrier
### Meta-Labeling

## Analysis

### Feature Engineering
### Fractionally Differentiated Features
De Prado in [1] explains the **'stationary vs memory dilemma'**: in finance the time-series are non-stationary, due to
the presence of the memory in the time-series itself. Transforming the time-series obtaining a stationary trend
(i.e. computing the return on price or the changes in volatility, ecc.), led to lose the memory from the time-series.
"**Memory is the basis of predictive power for models, but stationarity is a necessary property for inferencial purposes**.
The **dilemma** is that **returns** are **stationary**, however **memory-less**, and **prices have **memory**,
however they are **non-stationary**."

The idea is transforming the time-series in order to consider it as **stationary**, where **not all memory is erased**.

The **Fractionally Differentiation** corresponds to the difference operation but with non-integer steps:

Here it is an example of the analysis for **Cardano** Cryptocurrency (ADA-EUR)
![Labeling](https://raw.github.com/daniele21/financial_anomaly_detection/main/static/frac_diff.png?raw=true)

The dashed red line represents the **95% confidence** for the **Adfueller test**.

From this analysis you can see that with a **fractional differentiation** of ADA-EUR price between [**0.3**; **0.45**]
you get a transformed time-series that holds the **stationary** assumption with confidence around **95% confidence**, **maintaining** 
a still high **correlation** with the original time-series around **0.9**


### Feature Selection

## Models





# References

[1]. "Advances in Financial Machine Learning" by Marcos Lopez de Prado, 2018

