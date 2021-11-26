# FAD - Financial Anomaly Detection

## Description
In this project I want to develop a AI algorithm able to understand when an **anomaly** affects the _price trend_. Instead of solving the forecasting problem, I want to focus on the **nowcasting problem**, _i.e. prediction of the present, the very near future_. 

Forecasting problem are very difficult to be launch in real-case use, because they are not able to predict an external shock, given by a news or anything else, infact they are very poor againts the many external sources of shock. 

**Nowcasting** does not take the responsability to predict the future, as forecasting does, but it tries to predict the actual state or the near future. For this reason, I'm developing this project for **preventing some shock** that could affect the price trend of a financial product.

## Goal
Given a timestamps window, the **FAD model** has to detect if it is **anomalous**

## Dataset
The dataset is provided by the yahoo finance API _(yfinance)_. I focus mainly on **Crypto** data.

## Anomaly Definition

#### Assumptions
- Let's assume this feature in the dataset:

| Feature       | Description           |
| -------------- |:---------------------:|
| pct_change_w   | Percentage change of the **Close price** computed on **w** timestamps| 

- Let's consider the following **setup variables**:

| Variable       | Description           |
| -------------- |:---------------------:|
| w              | Timestamp Window size | 
| alpha          | Quantile of _pct_change_w_|
| min_variance   | Minimum variance of the _Close price_ |

- **Anomaly** is defined as the timestamp that holds the following conditions:
  - **pct_change_w** belongs to the tails at **(1-alpha) quantile** of _pct_change_w_  --> **Positive Anomaly**
  - **pct_change_w** belongs to the tails at **(alpha) quantile** of _pct_change_w_  --> **Negative Anomaly**  
    
#### The anomaly is referred to the window **w**, that pct_change_w is computed on



