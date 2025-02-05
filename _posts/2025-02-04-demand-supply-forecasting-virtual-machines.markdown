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

Each time series uses last 365 days worth of data i.e. approximately 102M values.

Some lessons learnt and choices made while taking this project to completion are:

1. **Using deep learning models from the word go instead of spending time on classical ML techniques such as `ARIMA`, `Gradient Boosting` for forecasting etc.**<br/><br/>
This is not to say that classical forecasting algorithms and gradient boosting are not good algorithms. But from our experience working with forecasting problems in similar domains, we have found that deep learning models often outperform ARIMA or XGBOOST and also we save time on manually creating features such as finding the periodicity using `Fast Fourier Transforms` etc.<br/><br/>
Another reason that we had in mind is that the predictions from our forecasting models are not consumed in real time, thus there is no SLA regarding inference times and thus we took the liberty to improve the accuracy of the forecasts.

2. **Modelling all the 280K time series together instead of individually modelling them or creating sub-groups based on VM or region or OS etc.**<br/><br/>
The advantages of a single model over multiple models are as follows:
   1. A significant number of (VM, region and OS) tuples have very less data and thus modelling them separately do not make much sense.
   2. Ease of maintenance. It is easier to build and maintain a single model.
   3. Better with cold start problem. New VMs will not have enough time series data.

4. **Using distributed `map-reduce` jobs wherever feasible to deal with large datasets and to reduce data processing times. But understanding where map-reduce will be beneficial vs. wherever not.**<br/><br/>
Running map-reduce over distributed nodes has an additional network overhead due to multiple I/Os over the network. The problem can be significant if the executor nodes are located very far away from the driver node or some executor nodes is crashing occassionally and the scheduler has to retry the task. This is advantageous when:
    1. Driver node has limited memory and cannot run with all data at once.
    2. Driver node has enough memory but CPU time taken is more as compared to solving it in parallel + network overhead.<br/><br/>
Map-reduce with `Pyspark`:
```python
def mapr(objects):
  def evaluate(object):
    result = do_something(object)
    return result

  rdd = spark.sparkContext.parallelize(objects, len(objects))
  out = rdd.map(evaluate)
  res = out.collect()
```

6. **Use `static features` corresponding to virtual machines as well other time dependent features apart from demand and supply instead of categorical features to identify virtual machines.**<br/><br/>
Static features and other time dependent features apart from the time series values helps to distinguish different time series i.e. different (VM, region, OS) tuples without actually using some categorical variables to identify these.<br/><br/>
The advantage is that if some new (VM, region, OS) tuple is added after the model is built, we can still get predictions for the next 30 days using the last 30 days values for this tuple as well as the static features and the time dependent features.<br/><br/>
If we had used some categorical feature to identify the tuple, we cannot get predictions from the trained model because the model has not seen those features.

8. **Instead of `auto-regressive` model, we chose to use a `multi-horizon` forecast model.**<br/><br/>
In auto-regressive model, we feed the prediction of day D as input to the network to generate prediction for day D+1 and so on. Thus to predict the demand forecast for day D+29, it would use last 29 days of predicted values and only 1 actual value.<br/><br/>
If each prediction has some error, then the errors will accumulate over all the 29 days. Instead we prefer to use a multi-horizon model where the model predicts the forecasts for days D to D+29 using only the actual values D-30 to D-1.

10. **Using efficient data compression and time series `compression algorithms` as well as `sparse data structures` for distributed map-reduce jobs reduces processing times as well as memory usage.**<br/><br/>
One of the tricky parts during the map-reduce operations is that the task results from each executor node could be well over few 100 GBs. Sending all the data over the network could consume significant network bandwidth and could be very slow. Moreover, the driver node that is collecting all the data from multiple executors will be holding M*100 GB of data where M is the number of executor nodes. This could easily go out-of-memory.

11. Using `Conv1D` architectures instead of `LSTM` or `RNN` based deep learning models gives almost equivalent or better performances but much higher speed of training and inference.

12. There are lots of missing data in the time series'. Handling missing data using `exponential averaging` improves the overall model performance.

13. Encoding of `categorical` and `numerical` features efficiently so as not to explode the size of the matrices in memory as well as not sacrifice on performance.

14. Explicit `garbage collections` of large in-memory objects while working with notebooks, helps iterate faster with less resources.

15. Demand and supply data do not follow a normal distribution but the data is highly skewed with a long tail resembling a `Poisson` or a `Negative Binomial Distribution`. Thus, care must be taken while defining the loss function.

16. Different loss functions were tried such as `MAE`, `MAPE`, `Pinball Loss`, `Tweedie Loss` etc.

17. `Probabilistic forecasting` model to handle different quantiles at once instead of a single quantile regression model. A `negative binomial distribution` was assumed.

18. `Feature scaling` led to very small floating point numbers for demand as well as supply leading to `vanishing gradient` problem famously associated with deep neural networks. Care must be taken so as not to scale down features which are already very small.

19. Probabilistic forecasting model with a negative binomial distribution was tricky to handle due to `sigmoid` and `softplus` activations for the parameters. Used variable transformation from probability to mean of the distribution and adding epsilon to logarithmic functions so as to avoid nans during training.

20. Using `int32` instead of `float64` for demand and supply values reduces memory consumption by half.

21. Handling cold start problem by not including VM specific features.

22. Choosing the offline metrics wisely. Just don't (only) use MAPE.
    
