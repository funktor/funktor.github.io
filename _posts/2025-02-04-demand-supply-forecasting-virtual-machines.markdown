---
layout: post
title:  "Practical lessons learnt from building a demand-supply forecasting model"
date:   2025-02-04 18:50:11 +0530
categories: ml
---

Sometimes back, I was leading a project at Microsoft to build their demand and supply forecasting models for `spot virtual machines` running in Azure. The idea behind using the demand-supply forecast models is to improve the `dynamic pricing` algorithm for spot instances.

For people who are unaware of spot virtual machines, almost all cloud service providers provide them. These virtual machines comes at a `discounted price` compared to the standard virtual machines (roughly `10% to 90%` discount) but with the gotcha that these spot VMs can be evicted (read: Job killed) anytime depending on the current demand and bids. (Think like an `auction`).

![Real Time Bidding for VMs (AI Generated)](/docs/assets/rtb.jpg)

An optimal pricing algorithm for these spot instances should take into account the projected demand and supply because we do not want to price these VMs too less such that the demand for these exceeds the projected supply. The exact dynamic pricing algorithm is a story for another post though.

The demand-supply forecasting deals with `700+ different virtual machines` e.g. `standard_d16s_v5`, running across `100+ regions` and running either `linux or windows` OS. Thus, there are approx. 700 * 100 * 2 = `140K` different time series for demand as well as supply or `280K` in total considering both demand and supply together.

Some lessons learnt and choices made while taking this project to completion are:

1. Using deep learning models from the word go instead of spending time on classical ML techniques such as `ARIMA`, `Gradient Boosting` for forecasting etc.

2. Modelling all the 280K time series together instead of individually modelling them or creating sub-groups based on VM or region or OS etc.

3. Using distributed `map-reduce` jobs wherever feasible to deal with large datasets and to reduce data processing times. But understanding where map-reduce will be beneficial vs. wherever not.

4. Using static features corresponding to virtual machines as well other time dependent features apart from demand and supply.

5. Using a granularity of 1 day for the time series and 30 days historical data to predict 30 days horizon for both demand and supply. Instead of `auto-regressive` model, the model used is a `multi-horizon` forecast model.

6. Using efficient data compression and time series `compression algorithms` as well as `sparse data structures` for distributed map-reduce jobs reduces processing times as well as memory usage.

7. Using `Conv1D` architectures instead of `LSTM` or `RNN` based deep learning models gives almost equivalent or better performances but much higher speed of training and inference.

8. There are lots of missing data in the time series'. Handling missing data using `exponential averaging` improves the overall model performance.

9. Encoding of `categorical` and `numerical` features efficiently so as not to explode the size of the matrices in memory as well as not sacrifice on performance.

10. Explicit `garbage collections` of large in-memory objects while working with notebooks, helps iterate faster with less resources.

11. Demand and supply data do not follow a normal distribution but the data is highly skewed with a long tail resembling a `Poisson` or a `Negative Binomial Distribution`. Thus, care must be taken while defining the loss function.

12. Different loss functions were tried such as `MAE`, `MAPE`, `Pinball Loss`, `Tweedie Loss` etc.

13. `Probabilistic forecasting` model to handle different quantiles at once instead of a single quantile regression model. A `negative binomial distribution` was assumed.

14. `Feature scaling` led to very small floating point numbers for demand as well as supply leading to `vanishing gradient` problem famously associated with deep neural networks. Care must be taken so as not to scale down features which are already very small.

15. Probabilistic forecasting model with a negative binomial distribution was tricky to handle due to `sigmoid` and `softplus` activations for the parameters. Used variable transformation from probability to mean of the distribution and adding epsilon to logarithmic functions so as to avoid nans during training.

16. Using `int32` instead of `float64` for demand and supply values reduces memory consumption by half. 
    
