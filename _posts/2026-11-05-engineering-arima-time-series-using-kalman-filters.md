---
layout: post
title:  "Engineering ARIMA for Time Series using Kalman Filters"
date:   2026-11-05 18:50:11 +0530
categories: ml
---
As a ML engineer, when I started working with time series modelling, popularity of ARIMA and similar class of algorithms/models were overshadowed by Recurrent Neural Networks (RNN) and its variants such as GRU and the more popular LSTM. Although, I did learn about ARIMA from various resources but mostly from the point of view of using ARIMA as a baseline for performance numbers because LSTM would outperform ARIMA on most occasions. But, while collaborating with data scientists, most DS would prefer ARIMA over LSTM/RNN for time series modelling. They usually worked on problems that were more analysis focussed for e.g. anomaly detection. And as such, simplicity and interpretability of the approach mattered more than the final performance of the model. ARIMA results are easier to interpret than that of LSTM.
<br/><br/>
But that does not mean that ARIMA does not have its share of problems. Some of them are:
```
1. ARIMA is a linear model whereas RNN/LSTM are non-linear. In ARIMA, the target variable is linearly related to the past observations and other covariates.
But in RNN/LSTM, the RNN layers have non-linearity in the form of ReLU, Tanh etc.

2. ARIMA does not model multiple time series together i.e. it cannot handle cross learning in different time series.

3. Computing the prediction intervals in ARIMA e.g. 95% interval, assumes that the error terms are normally distributed. With RNN/LSTM, we can use Pinball Loss
to train the model to learn the 2.5% and 97.5% quantile values without assuming any distribution of errors.

4. ARIMA needs different seasonal and non-seasonal parameters - (lag, differencing, moving average). RNN/LSTM only needs the lag parameter.
```
<br/><br/>
Working extensively with ARIMA for the past few months and scaling training and predictions for millions of time series, made me probe in details the math and engineering behind ARIMA. Contrary to my earlier notion, ARIMA is not just a simple regression problem.
<br/><br/>
Let's start off by describing a simple ARIMA equation with a lag=1 i.e. the target at time T is only dependent on the observation at time T-1.
<br/><br/>
![arima1](/docs/assets/arima1.png)
<br/><br/>
This indeed looks like a simple regression problem. The loss function and the update equation for the parameter(s) using gradient descent can be written simply as (similar update equation for the constant parameter):
<br/><br/>
![arima3](/docs/assets/arima2.png)
<br/><br/>
Adding more lag terms to the equation, uses more number of historical observations and the same update equation for each parameter, theta holds. Thus, we can see that with only lagged series, the equation is equivalent to a linear regression problem which we can either solve using matrix operations or iteratively using gradient descent as shown above. Apart from the lag terms, ARIMA also can have the moving average terms. The moving average terms are nothing but simply the error terms from the previous timesteps similar to the previous observations, and can be expressed as follows:
<br/><br/>
![arima4](/docs/assets/arima4.png)
<br/><br/>
But note that now the derivative of the loss L w.r.t. the parameters, theta and phi are not simple, because each moving average term (or the error terms) are also dependent on the same parameters and cannot be treated as constants unlike the past observation terms. The modified partial derivatives are as follows:
