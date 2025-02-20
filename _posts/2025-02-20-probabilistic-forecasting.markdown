---
layout: post
title:  "Deep learning probabilistic forecasting with different distributions"
date:   2025-02-20 18:50:11 +0530
categories: ml
---

During our work on building a [demand forecasting model for spot virtual machines](https://funktor.github.io/ml/2025/02/04/demand-supply-forecasting-virtual-machines.html) we realized that a standard 'Mean Squared Error' loss function was not sufficient. Two main reasons being that:

1. Most often we are interested in understanding the 90th percentile or 99th percentile demand. <br/><br/>
The 'mean' or 'median' demand implies that the predicted demand would be greater than (or equal to) the actual demand approx. 50% of the time. The remaining 50% of the time, the predicted demand would be lower than actual. To allocate enough cores for running VMs in the cloud, knowing 90th percentile or 99th percentile demand is more useful.

2. The distribution of the demand is skewed. Instead of a symmetric normal distribution, the distribution resembles a gamma or a negative binomial distribution with a long tail.

For point no. 1, if one desires specific quantile such as P90 or P99 value, then there is a specific cost function associated, termed as the Pinball Loss.

```
PL(y_true, y_pred) = q*max(y_true-y_pred, 0) + (1-q)*max(y_pred-y_true, 0)

q = 0.9 for P90 or 0.99 for P99
```

If we set q=0.5, then we will get the P50 or the median forecast value.

The cost function is basically a 'penalty' function. If the actual value is higher than the predicted value by some X, the cost is 0.9 times X but if the actual value is lower than the predicted by X then the cost is 0.1 times X. This ensures that the predicted value stays 'above' the actual approx. 90% of the times.

But often we need to model for multiple quantiles such as P50, P90, P99 etc. This calls for retraining the model with each quantile. A better strategy in this case would be to learn the distribution of the demand values. Once we learn the distribution parameters of the demand values, then we can calculate any arbitrary quantile value by random sampling from the distribution.

For e.g. given the normal distribution:<br/><br/>
![image](https://github.com/user-attachments/assets/85439ae3-ac2d-45e8-9a6e-78d6f640b5af)

One can then find different quantile values. One strategy would be to randomly sample N values, sort the values and take the q*N-th value where q-quantile. But since sorting could be expensive if N is very large or we are doing random sampling quite often, another strategy would be to use a histogram based approach.

If the minimum value is a and max value is b and 
