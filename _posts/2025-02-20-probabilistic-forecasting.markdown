---
layout: post
title:  "Deep learning probabilistic forecasting with different distributions"
date:   2025-02-20 18:50:11 +0530
categories: ml
---

During our work on building a [demand forecasting model for spot virtual machines](https://funktor.github.io/ml/2025/02/04/demand-supply-forecasting-virtual-machines.html) we realized that a standard **Mean Squared Error** loss function was not sufficient. Two main reasons being that:

1. Most often we are interested in understanding the `90th percentile` or `99th percentile` demand. <br/><br/>
**Mean squared error predicts the mean demand.** <br/><br/>
The `mean` or `median` demand implies that the predicted demand would be greater than (or equal to) the actual demand approx. 50% of the time. The remaining 50% of the time, the predicted demand would be lower than actual. To allocate enough cores for running VMs in the cloud, knowing 90th percentile or 99th percentile demand is more useful.

3. The `distribution` of the demand is `skewed`. Instead of a symmetric normal distribution, the distribution is skewed resembling a `gamma` or a `negative binomial distribution` with a `long tail`.

For point no. 1, if one desires specific quantile such as P90 or P99 value, then there is a specific cost function associated, termed as the **Pinball Loss**.

```
PL(y_true, y_pred) = q*max(y_true-y_pred, 0) + (1-q)*max(y_pred-y_true, 0)

q = 0.9 for P90 or 0.99 for P99
```

If we set q=0.5, then we will get the `P50` or the median forecast value.

The cost function is basically a `penalty` function. If the actual value is higher than the predicted value by some X, the cost is 0.9 times X but if the actual value is lower than the predicted by X then the cost is 0.1 times X. This ensures that the predicted value stays 'above' the actual approx. 90% of the times.

But often we need to model for multiple quantiles such as P50, P90, P99 etc. 

This calls for retraining the model with each quantile. A better strategy in this case would be to learn the probability distribution of the demand values. Once we learn the distribution parameters of the demand values, we can calculate any arbitrary quantile value by random sampling from the distribution.

For e.g. given the normal distribution:<br/><br/>
<p align="center">
    <img src="https://github.com/user-attachments/assets/85439ae3-ac2d-45e8-9a6e-78d6f640b5af">
<p/>

One can then find different quantile values. 

One strategy would be to `randomly sample N values`, `sort` the values and take the `q*N` th value where q is the quantile. <br/>

```python
def get_quantile(u, s, n, q):
    # u - mean of distribution
    # s - sd of distribution

    a = np.random.normal(u, s, n)
    a = sorted(a.tolist())
    j = int(q*n)

    if j < n:
        return a[j]
    return None

# get the 99th percentile value of a normal distribution with mean of 0.0 and sd of 1.0
# comes around 2.325
print(get_quantile(0.0, 1.0, 1000000, 0.99))
```
<br/>

But since sorting is expensive if N is very large or we are doing sampling quite often, a better strategy would be to use a `histogram` based approach. If the minimum value is a and max value is b and say we are using 100 bins, then we divide the interval [a, b) into 100 bins s.t. a value D would go into bin index int((D-a)/c) where c = (b-a)/100.0.

To get the 99th percentile, continue to sum the size of the bins until the index int(q*N) lies inside a bin. Sort the values within the bin only and take the corresponding value as the quantile.<br/><br/>

```python
def get_quantile_hist(u, s, n, q, nbins):
    r = np.random.normal(u, s, n).tolist()

    a = np.min(r)
    b = np.max(r)
    c = (b-a)/nbins

    bins = [[] for _ in range(nbins)]

    for x in r:
        j = int((x-a)/c)
        j = min(j, nbins-1)
        bins[j] += [x]

    # index we want
    j = int(q*n)

    p = 0
    for b in bins:
        # continue to sum size of bins until the desired index lies inside a bin
        if p + len(b) > j:
            # sort the bin values only
            sb = sorted(b)
            return sb[j-p]
        
        p += len(b)
    
    return None

print(get_quantile_hist(0.0, 1.0, 1000000, 0.99, 100))
```
<br/>

Note that the histogram approach is not always efficient because the sizes of the bins can be skewed i.e. it is entirely possible that only the 1st bin has 99% of all the values. In that case the 1st approach is better.
