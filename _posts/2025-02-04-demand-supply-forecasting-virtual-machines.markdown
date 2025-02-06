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
The advantages of a single model over multiple models are as follows:<br/><br/>
   1. A significant number of (VM, region and OS) tuples have very less data and thus modelling them separately do not make much sense.
   2. Ease of maintenance. It is easier to build and maintain a single model.
   3. Better with cold start problem. New VMs will not have enough time series data.<br/><br/>

4. **Using distributed `map-reduce` jobs wherever feasible to deal with large datasets and to reduce data processing times. But understanding where map-reduce will be beneficial vs. wherever not.**<br/><br/>
Running map-reduce over distributed nodes has an additional network overhead due to multiple I/Os over the network. The problem can be significant if the executor nodes are located very far away from the driver node or some executor nodes is crashing occassionally and the scheduler has to retry the task. This is advantageous when:<br/><br/>
    1. Driver node has limited memory and cannot run with all data at once.
    2. Driver node has enough memory but CPU time taken is more as compared to solving it in parallel + network overhead.<br/><br/>
Map-reduce with `Pyspark`:<br/><br/>
       ```python
       def mapr(objects):
         def evaluate(object):
           result = do_something(object)
           return result

         rdd = spark.sparkContext.parallelize(objects, len(objects))
         out = rdd.map(evaluate)
         res = out.collect()
       ```
<br/><br/>

6. **Use `static features` corresponding to virtual machines as well other time dependent features apart from demand and supply.**<br/><br/>
Static features and other time dependent features apart from the time series values helps to distinguish different time series i.e. different (VM, region, OS) tuples without actually using categorical variables to identify these.<br/><br/>
The advantage is that if some new (VM, region, OS) tuple is added after the model is built, we can still get predictions for the next 30 days using the last 30 days values for this tuple.<br/><br/>
If we had used some categorical feature to identify the tuple, we cannot get predictions from the trained model because the model has not seen those features.<br/><br/>
Some examples of static features are number of CPU cores in a VM, RAM and SSD size, set of hardwares where the VM can run and so on.<br/><br/>

8. **Instead of `auto-regressive` model, we chose to use a `multi-horizon` forecast model.**<br/><br/>
In auto-regressive model, we feed the prediction of day D as input to the network to generate prediction for day D+1 and so on. Thus to predict the demand forecast for day D+29, it would use last 29 days of predicted values and only 1 actual value.<br/><br/>
If each prediction has some error, then the errors will accumulate over all the 29 days. Instead we prefer to use a multi-horizon model where the model predicts the forecasts for days D to D+29 using only the actual values D-30 to D-1.<br/><br/>

10. **Using efficient data compression and time series `compression algorithms` as well as `sparse data structures` for distributed map-reduce jobs reduces processing times as well as memory usage.**<br/><br/>
One of the tricky parts during the map-reduce operations is that the task results from each executor node could be well over few 100 GBs. Sending all the data over the network could consume significant network bandwidth and could be very slow.<br/><br/>
Moreover, the driver node that is collecting all the data from multiple executors will be holding M*100 GB of data where M is the number of executor nodes. This could easily go out-of-memory.<br/><br/>
One possible way to reduce of the size of the data transferred is to use compression. Some strategies that we felt are useful for compressing the data:<br/><br/>
    1. If the result is a sparse NumPy matrix, then use scipy.sparse.csr_matrix format instead of numpy.array().<br/><br/>
    2. If the result are time series values, then encode each time series separately using delta encoding or XOR encoding techniques. [Gorilla](https://www.vldb.org/pvldb/vol8/p1816-teller.pdf) paper from Meta is a nice reference on how to do XOR encoding of time series data.<br/><br/>
    3. If the data is a dense matrix, then one can use histogram based encoding for each floating point columns. Note that these are lossy encoding and are useful only if you are going to use the histogram encoding as features to your model.<br/><br/>
    4. Another lossy encoding is to do dimensionality reduction using Truncated SVD or PCA. Again this is useful only when the encoded columns are being used as features to the model.
<br/><br/>
If all else fails, and you are still getting out-of-memory errors, one possible solution is to persist the resultant objects in a blob storage container and read them back from storage sequentially.
<br/><br/>

12. **Using `Conv1D` architectures instead of `LSTM` or `RNN` based deep learning models gives almost equivalent or better performances but much higher speed of training and inference.**<br/><br/>
Convolution operations such as Conv1D are nothing but multiple matrix dot product operations. Such operations can be easily parallelized across different regions of the matrix and each matrix dot product can leverage SIMD or GPU instructions for faster computations. <br/><br/>
On the other hand LSTM and RNN are sequential architectures. They cannot be parallelized.<br/><br/>
For our problem, Conv1D was at-least 8 times faster than LSTM architecture while giving similar results.<br/><br/>

13. **There are lots of missing data in the time series'. Handling missing data using `exponential averaging` improves the overall model performance.**<br/><br/>
Simple averaging of past N days values gives equal weightage to all the past N days but in a time series, usually the recent values are better predictor than the older values.<br/><br/>
Thus, we chose to use an exponentially weighted average, where the value for day D-H is given a weightage of exp(-H/K). H is the number of days backwards from current day D and K is a constant e.g. 30.<br/><br/>
    ```python
    import numpy as np
    def exponential_averaging(values, index_to_fill, past_N, K=30):
       s = 0
       u = 0
       for i in range(index_to_fill-1, max(-1, index_to_fill-past_N-1), -1):
           h = np.exp(-(index_to_fill-i)/K)
           s += h*values[i]
           u += h
       values[index_to_fill] = s/u if u > 0 else 0.0
    ```
<br/><br/>

15. **Encoding of `categorical` and `numerical` features efficiently so as not to explode the size of the matrices in memory as well as not sacrifice on performance.**<br/><br/>
Neural networks requires all input data to be numerical. Thus any categorical features such as region, OS etc. needs to be converted into numerical features. One such strategy is using one-hot encoding. But with 700+ products, the size of a one-hot encoded vector would be 700+ with lots of zeros.<br/><br/>
We did Truncated SVD on the sparse matrix to reduce dimensionality.<br/><br/>
For numerical features (floats etc.) we had used binning strategy to encode the features. The number of bins to use can be chosen by hyperparameter tuning.<br/><br/>
We had used the starting boundary value of each bin as encoded feature values.<br/><br/>

17. **Explicit `garbage collections` of large in-memory objects while working with notebooks, helps iterate faster with less resources.**<br/><br/>
While working with notebooks, if we do not take care, very quickly the size of the objects held in the memory would cause the notebook to crash with out-of-memory errors.<br/><br/>
One strategy is to do explicit garbage collection of large objects in memory.<br/><br/>
Another startegy is to persist large objects in the blob storage and read then back in another session or another job.<br/><br/>

19. **Demand and supply data do not follow a normal distribution but the data is highly skewed with a long tail resembling a `Gamma` or a `Negative Binomial Distribution`. Thus, care must be taken while defining the loss function.**<br/><br/>
Usual loss functions used in regression and time series forecasting are mean squared errors or mean absolute errors. But with skewed data, MSE or MAE may not be the best choice. Remember that the loss function is derived from the maximum likelihood principle of the distribution of the target variable. Usually we take the negative log likelihood.<br/><br/>
MSE or MAE works best when we assume that the target variable follows a normal distribution.<br/><br/>
In our case, our distribution for demand and supply follows a distrubtion similar to a Gamma or a Negative Binomial distribution.<br/><br/>
In Tensorflow, we can define custom loss for a gamma distribution as follows:<br/><br/>
    ```python
    def gamma_loss():
       def loss(y_true, y_pred):
           k, m = tf.unstack(y_pred, num=2, axis=-1)
   
           k = tf.expand_dims(k, -1)
           m = tf.expand_dims(m, -1)
   
           l = tf.math.lgamma(k) + k*tf.math.log(m) - k*tf.math.log(k) - (k-1)*tf.math.log(y_true+1e-7) + y_true*k/y_pred
           
           return tf.reduce_mean(l)
   
       return loss

    def negative_binomial_loss():
       def loss(y_true, y_pred):
           r, m = tf.unstack(y_pred, num=2, axis=-1)
   
           r = tf.expand_dims(r, -1)
           m = tf.expand_dims(m, -1)
   
           nll = (
               tf.math.lgamma(r) 
               + tf.math.lgamma(y_true + 1)
               - tf.math.lgamma(r + y_true)
               - r * tf.math.log(r) 
               + r * tf.math.log(r+m)
               - y_true * tf.math.log(m) 
               + y_true * tf.math.log(r+m)
           )
   
           return tf.reduce_mean(nll)       
       
       return loss
    ```

In gamma loss, k is the shape parameter > 0 and m is the mean of the gamma distribution.
In negative binomial loss, r is the parameter for number of successes and m is the mean of the distribution.
<br/><br/>

21. **`Probabilistic forecasting` model to handle different quantiles at once instead of a single quantile regression model.**<br/><br/>
Instead of only predicting the mean of the distribution of the demand and supply as forecasted values, our network also predicts the parameters of the distribution. This has the advantage that we can use the distribution to predict different quantiles fo the demand and supply forecast values. For e.g. 90% quantile implies that the predicted values are greater than the true values 90% of the time.<br/><br/>
For VM forecasting, we want to make sure that there is always sufficient buffer capacity available in case demand peaks, P90 or P99 forecast values can be useful.<br/><br/>

22. **`Feature scaling` led to very small floating point numbers for demand as well as supply leading to `vanishing gradient` problem famously associated with deep neural networks. Care must be taken so as not to scale down features which are already very small.**<br/><br/>
Using MinMaxScaler() to scale the values for demand and supply led to very small values because the actual range was in 1 to 1e7, and after scaling, the range shifted to 1e-7 to 1.<br/><br/>
During backpropagation, the feature values are multiplied with the gradient to obtain the updated weights. If both gradient and feature values are very small, then the weight updates are negligible and the network does not train properly.<br/><br/>
Better strategy was not to re-scale the demand and supply values as both these time series were of similar scales.<br/><br/>

23. Probabilistic forecasting model with a negative binomial distribution was tricky to handle due to `sigmoid` and `softplus` activations for the parameters. Used variable transformation from probability to mean of the distribution and adding epsilon to logarithmic functions so as to avoid nans during training.

24. Using `int32` instead of `float64` for demand and supply values reduces memory consumption by half.

25. Handling cold start problem by not including VM specific features.

26. Choosing the offline metrics wisely. Just don't (only) use MAPE.
    
