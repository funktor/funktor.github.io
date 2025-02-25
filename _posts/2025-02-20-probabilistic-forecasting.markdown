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
</p>

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

But since sorting is expensive if N is very large or we are doing sampling quite often, a better strategy would be to use a `histogram` based approach. If the minimum value is a and max value is b and say we are using 100 bins, then we divide the interval [a, b) into 100 bins s.t. a value D would go into bin index int((D-a)/c) where c = (b-a)/100.0.

To get the 99th percentile, continue to sum the size of the bins until the index int(q*N) lies inside a bin. Sort the values within the bin only and take the corresponding value as the quantile.<br/>

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

Note that the `histogram approach is not always efficient` because the sizes of the bins can be `skewed` i.e. it is entirely possible that only the 1st bin has 99% of all the values. In that case the 1st approach is better.

The above strategy to learn the distribution parameters of the demand instead of just the mean or median demand is also useful when the distribution is `not gaussian` but is skewed such as the `Gamma` or the `Negative Binomial Distribution`. **Most real world datasets are skewed**. 

But before showing how to incorporate gamma or negative binom. distributions in our demand forecasting problem, lets first understand loss functions with arbitrary distibutions of the target variable and why mean squared error is not appropriate for every problem.

Most standard loss functions are the **negative log likelihood of the target variable**.

Assuming that each output j is sampled from a `normal distribution` N(u<sub>j</sub>, 1.0) where u<sub>j</sub> is the model predicted value i.e. y<sup>j</sup><sub>pred</sub> (`standard deviation` is just a scaling factor hence it is assumed to be 1.0). The PDF of the output j is:

![image](https://github.com/user-attachments/assets/7b0af647-2410-45be-a9c4-20f5686d27c4)

The joint PDF of all outputs from j=0 to N-1 is just the `product of the individual PDFs`.

![image](https://github.com/user-attachments/assets/b91babd3-ec9e-43b9-a7c1-775d07a2f6db)

The `log likelihood` is the probability that given the parameter y<sub>pred</sub>, how likely the above joint distribution fits the data. It has the `same expression as the joint PDF`. To get the loss function, we take the log of the above joint distribution and `add a negative sign` before it (since its a loss or cost function).

![image](https://github.com/user-attachments/assets/46e74277-cdd1-4e78-a279-1b231582c39c)

Ignoring the constant terms, the above expression resembles the **mean squared error** term which is the loss function most commonly used for `linear regression` problems.

Similarly, for binary classification problems, the target variable is either 0 or 1 and assuming that the probability of 1 is p<sub>j</sub> (which is equal to the model prediction y<sup>j</sup><sub>pred</sub>), the probability distribution of the target is a `binomial distribution` as follows:

![image](https://github.com/user-attachments/assets/04693924-c0c7-4ae3-b3e1-3f03c7dca9af)

Finding the joint distribution by taking the product for all j=0 to N-1 and taking their negative log likelihood, we get the familiar form of the `logistic loss` used extensively for `binary classification problems`.

![image](https://github.com/user-attachments/assets/cf46a01d-f831-4dff-babe-7dfb29fc1c7c)

But it is **not a prerequisite to use the negative log likelihood as the loss function** everywhere. For linear regression, even if the target variable or the residual (y<sub>true</sub>-y<sub>pred</sub>) do not follow the normal distribution we can still use the mean squared error loss function to learn y<sub>pred</sub>. Similarly, for classification once can also use the mean squared error loss instead of the logistic loss and still get good results.

There are many loss functions such as the `contrastive loss` or `pinball loss` etc. which does not directly follow from negative log likelihood expressions.

One can use any loss function if the objective is just to learn y<sub>pred</sub>, but in our case, we want to find different **quantiles for y<sub>pred</sub> instead of learning y<sub>pred</sub>** and for that we must know the correct distribution for y and then use that distribution to sample values and get the quantile. If the distribution is not correct, then we will **sample incorrect values and quantiles will also be incorrect**.

Before proceeding with probabilistic forecasting, one important point to discuss is that most often the distribution is a known distribution and libraries such as numpy or scipy will have some implementation to do random sampling from the distribution. But sometimes, a distribution may not have an inbuilt library for sampling.

In such cases, a common technique applied is finding the inverse of the CDF function. CDF is the Cumulative Density Function i.e. sum of the PDF from -inf to x (or 0 to x if x is non-negative).

```
CDF(x) = [PDF(y) for y in -inf to x]
```

The q-th quantile value is the value 'x' at which CDF(x) = q.

For e.g. the PDF of the exponential distribution with mean of 0.5 is:

![image](https://github.com/user-attachments/assets/1fb5ea11-7c00-42bd-aadf-1867dc9b51ca)

The CDF is found by taking an integral of F(x) from 0 to x i.e.

![image](https://github.com/user-attachments/assets/b9f8461e-b53d-4c90-bd66-1b445cadbe4e)

The inverse function can be found out be:

![image](https://github.com/user-attachments/assets/1eead258-a7cf-4e03-9ee8-ee9249cca53b)

For e.g. to find the value at quantile q=0.99, we substitute q=0.99 in the above equation. The P99 value for exponential distribution is thus 2.30.

In the event, the CDF is not possible to express in a closed functional form, we can sample multiple values from the  distribution F(x), compute the corresponding CDF(x) values. Then using binary search, find the greatest value z (sampled) s.t. the CDF(z) is smaller than U.








