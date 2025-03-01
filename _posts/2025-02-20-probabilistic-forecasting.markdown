---
layout: post
title:  "Deep learning probabilistic forecasting with non-gaussian distributions"
date:   2025-02-20 18:50:11 +0530
categories: ml
---

During our work on building a [demand forecasting model for spot virtual machines](https://funktor.github.io/ml/2025/02/04/demand-supply-forecasting-virtual-machines.html) we realized that a standard **Mean Squared Error** loss function was not sufficient. Two main reasons being that:

1. Most often we are interested in understanding the `90th percentile` or `99th percentile` demand. <br/><br/>
**Mean squared error predicts the mean demand.** <br/><br/>
The `mean` or `median` demand implies that the predicted demand would be greater than (or equal to) the actual demand approx. 50% of the time. The remaining 50% of the time, the predicted demand would be lower than actual. To allocate enough cores for running VMs in the cloud, knowing 90th percentile or 99th percentile demand is more useful.<br/><br/>
Note that **mean and median may not be equal**.

3. The `distribution` of the demand is `skewed`. Instead of a symmetric normal distribution, the distribution is skewed resembling a `gamma` or a `negative binomial distribution` with a `long tail`.

For point no. 1, if one desires specific quantile such as P90 or P99 value, then there is a specific cost function associated, termed as the **Pinball Loss**.

```
PL(y_true, y_pred) = q*max(y_true-y_pred, 0) + (1-q)*max(y_pred-y_true, 0)

q = 0.9 for P90 or 0.99 for P99
```

If we set q=0.5, then we will get the `P50` or the median forecast value.

The cost function implies that if the actual value is higher than the predicted value by some X, the cost is 0.9 times X but if the actual value is lower than the predicted by X then the cost is 0.1 times X. This ensures that the predicted value stays 'above' the actual approx. 90% of the times.

But often we need to model for multiple quantiles such as P50, P90, P99 etc. 

This calls for retraining the model with each quantile. A better strategy in this case would be to learn the probability distribution of the demand values. Once we learn the distribution parameters of the demand values, we can calculate any arbitrary quantile value by random sampling from the distribution.

For e.g. given the normal distribution:<br/><br/>
<p align="center">
    <img src="https://github.com/user-attachments/assets/85439ae3-ac2d-45e8-9a6e-78d6f640b5af">
</p>

One can then find different quantile values. 

One strategy would be to `randomly sample N values`, `sort` the values and take the `q*N` th value where q is the quantile. <br/>

```python
def get_quantile_normal(u, s, n, q=0.5):
    # u - mean of distribution
    # s - sd of distribution

    assert 0 <= q <= 1, "quantile should be between 0 and 1"

    a = np.random.normal(u, s, n)
    a = sorted(a.tolist())

    j = int(q*n)

    if j < n:
        return a[j]
    return None

# get the 99th percentile value of a normal distribution with mean of 0.0 and sd of 1.0
# comes around 2.325
print(get_quantile_normal(0.0, 1.0, 1000000, 0.99))
```

But since sorting is expensive if N is very large or we are doing sampling quite often, a better strategy would be to use a `histogram` based approach. If the minimum value is `a` and max value is `b` and say we are using 100 bins, then we divide the interval [a, b) into 100 bins s.t. a value D would go into bin indexed: 

```
int((D-a)/c) where c = (b-a)/100.0
```

To get the 99th percentile, continue to sum the size of the bins until the index int(q*N) lies inside the current bin. Sort the values within the bin only and take the corresponding value as the quantile. Or we can recursively continue to create bins until the maximum size of a bin is less than some constant threshold e.g. 10 s.t. sorting overhead is minimal.<br/>

```python
def get_quantile_hist_normal(u, s, n, q=0.5, nbins=10):
    r = np.random.normal(u, s, n).tolist()

    a = np.min(r)
    b = np.max(r)
    c = (b-a)/nbins

    bins = [[] for _ in range(nbins)]

    for x in r:
        j = int((x-a)/c)

        # maximum value 'b' goes into the last bin
        j = min(j, nbins-1)
        bins[j] += [x]

    assert 0 <= q <= 1, "quantile should be between 0 and 1"

    # index we want
    j = int(q*n)

    p = 0
    for bn in bins:
        # continue to sum size of bins until the desired index lies inside a bin
        if p + len(bn) > j:
            # sort the bin values only
            sb = sorted(bn)
            return sb[j-p]
        
        p += len(bn)
    
    return None

print(get_quantile_hist_normal(0.0, 1.0, 1000000, 0.99, 100))
```

Note that the `histogram approach is not always efficient` because the sizes of the bins can be `skewed` i.e. it is entirely possible that only the 1st bin has 99% of all the values. In that case either use sorting approach or recursively create bins. Another strategy to find the quantiles from the distribution is by using the **inverse CDF**.

CDF is the `Cumulative Density Function` i.e. sum of the PDF from -inf to x (or 0 to x if x is non-negative).

```
CDF(x) = [PDF(y) for y in -inf to x]
```

**The q-th quantile value is the value 'x' at which CDF(x) = q**

For e.g. the PDF of the exponential distribution with mean of 0.5 is:

<p align="center">
    <img src="https://github.com/user-attachments/assets/1fb5ea11-7c00-42bd-aadf-1867dc9b51ca">
</p>

The CDF is found by taking an integral of F(x) from 0 to x i.e.

<p align="center">
    <img src="https://github.com/user-attachments/assets/b9f8461e-b53d-4c90-bd66-1b445cadbe4e">
</p>

The inverse function can be found out be:

<p align="center">
    <img src="https://github.com/user-attachments/assets/1eead258-a7cf-4e03-9ee8-ee9249cca53b">
</p>

In order to find the value at quantile q=0.99, we substitute q=0.99 in the above equation. The P99 value for exponential distribution is thus 2.30. If we know the closed form function for the CDF, then this is the most efficient approach to find the q-th quantile.

The above strategy to learn the distribution parameters of the demand instead of just the mean or median demand is also useful when the distribution is `not gaussian` but is skewed such as the `Gamma` or the `Negative Binomial Distribution`. **Most real world datasets are skewed**. 

But before showing how to incorporate gamma or negative binom. distributions in our demand forecasting problem, lets first understand loss functions with arbitrary distibutions of the target variable and why mean squared error is not appropriate for every problem.

Most standard loss functions are the **negative log likelihood of the target variable**.

Assuming that each output label j is sampled from a `normal distribution` N(u<sub>j</sub>, 1.0) where u<sub>j</sub> is the model predicted value i.e. y<sup>j</sup><sub>pred</sub> (`standard deviation` is just a scaling factor hence it is assumed to be 1.0). The PDF of the output j is:

<p align="center">
    <img src="https://github.com/user-attachments/assets/490c3cb5-9a75-438d-948d-40d9e285c9b1">
</p>

The joint PDF of all outputs from j=0 to N-1 is just the `product of the individual PDFs`.

<p align="center">
    <img src="https://github.com/user-attachments/assets/543b7874-8450-4742-9f7a-ba81d96b92b3">
</p>

The `log likelihood` is the probability that given the parameter y<sub>pred</sub>, how likely the above joint distribution fits the data. It has the `same expression as the joint PDF`. To get the loss function, we take the log of the above joint distribution and `add a negative sign` before it (since its a loss or cost function).

<p align="center">
    <img src="https://github.com/user-attachments/assets/d77653dc-0c0a-443c-a354-2878db14a84f">
</p>

Ignoring the constant terms, the above expression resembles the **mean squared error** term which is the loss function most commonly used for `linear regression` problems.

Similarly, for binary classification problems, the target variable is either 0 or 1 and assuming that the probability of 1 is p<sub>j</sub> (which is equal to the model prediction y<sup>j</sup><sub>pred</sub>), the probability distribution of the target is a `binomial distribution` as follows:

<p align="center">
    <img src="https://github.com/user-attachments/assets/708e0b8f-d5c0-49a5-a196-474798e04ba9">
</p>

Finding the joint distribution by taking the product for all j=0 to N-1 and taking their negative log likelihood, we get the familiar form of the `logistic loss` used extensively for `binary classification problems`.

<p align="center">
    <img src="https://github.com/user-attachments/assets/d81be952-7088-4102-86e7-56e4b9301ca6">
</p>

But it is **not a prerequisite to use the negative log likelihood as the loss function**. For linear regression, even if the target variable or the residual (y<sub>true</sub>-y<sub>pred</sub>) do not follow the normal distribution we can still use the mean squared error loss function to learn y<sub>pred</sub>. Similarly, for classification once can also use the mean squared error loss instead of the logistic loss and still get good results.

There are many loss functions such as the `contrastive loss` or `pinball loss` etc. which does not directly follow from negative log likelihood expressions.

One can use any loss function if the objective is just to learn y<sub>pred</sub>, but in our case, we want to find different **quantiles for y<sub>pred</sub> instead of learning y<sub>pred</sub>** and for that we must know the correct distribution for y and then use that distribution to either sample values or find the inverse CDF and get the quantile. If the distribution is not correct, then we will **sample incorrect values and quantiles will also be incorrect**.

In this post we will explore the `Negative Binomial Distribution` which is quite commonly encountered in demand forecasting problems.

**Negative Binomial Distribution**<br/>
Given a binary sequence of 1s and 0s where the probability of 1 is p and probability of 0 is 1-p. In the context of virtual machines,let 1 indicate that a VM was used and 0 indicate it was not used. Assume that each entry corresponds to an event at time stamp T<sub>i</sub>. For some given sequence length N, the `probability of having r 1's and ending with a 1` is given as (if we exclude the last entry which is fixed to 1, then we can `choose r-1 places from N-1 places for the 1s`):

<p align="center">
    <img src="https://github.com/user-attachments/assets/90e4ede5-a688-4aad-9794-3efcdc176bbf">
</p>

Here `N and p are parameters of the distribution`. If N was fixed, it would just be a `binomial distribution`. If N is a parameter, it is a negative binomial distribution. In our case N is a parameter because we do not know at what time the observations were taken. Our goal is to find N and probability p.

The `log likelihood` formula for the above distribution would be something like:

<p align="center">
    <img src="https://github.com/user-attachments/assets/3f560cf7-d595-417d-be7e-ea365c16e161">
</p>

The gamma function for integers are defined to be:

<p align="center">
    <img src="https://github.com/user-attachments/assets/4b446509-8dfa-4513-bb06-0f6968d67f7e">
</p>

In one of the [earlier posts](https://funktor.github.io/ml/2025/02/04/demand-supply-forecasting-virtual-machines.html), we introduced a deep learning architecture for time series forecasting of demand and supply. Referring back to the same architecture, we can define our own custom negative binomial layer and the negative binomial log loss as follows:

```python
def negative_binomial_layer(x):
    num_dims = len(x.get_shape())

    # assuming that the input to this layer is a concatenation of n and p, extract the 2 variables from input x
    n, p = tf.unstack(x, num=2, axis=-1)

    n = tf.expand_dims(n, -1)
    p = tf.expand_dims(p, -1)

    # adding small epsilon so that log or lgamma functions do not produce nans in loss function
    n = tf.keras.activations.softplus(n)+1e-5
    p = 1e-5 + (1-2e-5)*tf.keras.activations.sigmoid(p)
 
    out_tensor = tf.concat((n, p), axis=num_dims-1)
 
    return out_tensor

def negative_binomial_log_loss():
   def loss(y_true, y_pred):
       # Log loss of the negative binomial distribution
       # assuming that the input to this layer is a concatenation of n and p, extract the 2 variables from input x
       n, p = tf.unstack(y_pred, num=2, axis=-1)
   
       n = tf.expand_dims(n, -1)
       p = tf.expand_dims(p, -1)
   
       nll = (
           tf.math.lgamma(n-y_true+1)
           + tf.math.lgamma(y_true)
           - tf.math.lgamma(n)
           - n * tf.math.log(p)
           - y_true * tf.math.log(1-p) 
       )
   
       return tf.reduce_mean(nll)       
       
   return loss

class ProbabilisticModel():
    def __init__(
        self, 
        epochs=200, 
        batch_size=512, 
        model_path=None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_path = model_path

    def initialize(self, inp_shape, out_shape):

        # inp_shape - shape of the input time series
        # out_shape - shape of the output time series

        inp = Input(shape=inp_shape)

        x = \
            Conv1D(
                filters=16, 
                kernel_size=inp_shape[0], 
                activation='relu', 
                input_shape=inp_shape
            )(inp)
                
        out = Dense(out_shape[0]*2)(x)
        out = Reshape((out_shape[0], 2))(out)
        out = tf.keras.layers.Lambda(negative_binomial_layer)(out)
                
        self.model = Model(inp, out)
        
        self.model.compile(
            loss=negative_binomial_log_loss(), 
            optimizer=tf.keras.optimizers.Adam(0.001)
        )
    
    def fit(self, X:np.array, Y:np.array):

         # X - input time series
         # Y - output time series

         model_checkpoint_callback = \
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.model_path,
                monitor='loss',
                mode='min',
                save_best_only=True
            )
        
         self.model.fit\
            (
                X, Y, 
                epochs=self.epochs, 
                batch_size=self.batch_size, 
                validation_split=None, 
                verbose=1, 
                shuffle=True,
                callbacks=[model_checkpoint_callback]
            )
    
    def predict(self, X:np.array):
        return self.model.predict(X)
```

Few things to note in the above tensorflow implementation:

1. The forecasting model is a multi-horizon model and each input is a 3D Tensor (batch size, num timesteps, embedding size).
2. For N we are using `softplus` activation because N is a non-negative parameter for the distribution.
3. For p we are using `sigmoid` activation because it is probability and lies between 0 and 1.
4. For p we are adjusting the range in [0.00001, 0.99999] because we are taking log(p) and log(1-p) in the loss function. If p is 0 or p is 1, this will lead to invalid values such as nans.
5. For n we are adding a small epsilon 1e-5 so that it is always greater than 0 because the log gamma function will generate nans if N is 0.
6. In the formula for the log likelihood, the variable 'r' is the y_true value.

If you run the above code, you will most likely get `nan` as loss. This is because of this term in the loss function:
```
tf.math.lgamma(n-y_true+1)
```
If n < y_true-1, then the term inside log gamma function will be negative and the function is undefined at that point.

One solution to this problem is instead of using n (length of sequence) we will use a different variable k s.t. n = k+r. With this change, we will mitigate the nan issue. The updated code is as follows:

```python
def negative_binomial_layer(x):
    num_dims = len(x.get_shape())

    # assuming that the input to this layer is a concatenation of k and p, extract the 2 variables from input x
    k, p = tf.unstack(x, num=2, axis=-1)

    k = tf.expand_dims(k, -1)
    p = tf.expand_dims(p, -1)

    # adding small epsilon so that log or lgamma functions do not produce nans in loss function
    k = tf.keras.activations.softplus(k)+1e-5
    p = 1e-5 + (1-2e-5)*tf.keras.activations.sigmoid(p)
 
    out_tensor = tf.concat((k, p), axis=num_dims-1)
 
    return out_tensor

def negative_binomial_log_loss():
   def loss(y_true, y_pred):
       # Log loss of the negative binomial distribution
       # assuming that the input to this layer is a concatenation of n and p, extract the 2 variables from input x
       k, p = tf.unstack(y_pred, num=2, axis=-1)
   
       k = tf.expand_dims(k, -1)
       p = tf.expand_dims(p, -1)
   
       nll = (
           tf.math.lgamma(y_true+1)
           + tf.math.lgamma(k)
           - tf.math.lgamma(k+y_true)
           - k * tf.math.log(p)
           - y_true * tf.math.log(1-p) 
       )
   
       return tf.reduce_mean(nll)       
       
   return loss
```

One more issue even with the above change is that, when the demand values are large, the input to the (k, p)-layer are also large and since the 'p' output passes through a sigmoid acivation, the sigmoid saturates and learning stops. The issue does not occur when the demand values are small.

This issue can also be mitigated by using another variable change. Use mean instead of probability 'p'. 

The mean of the distribution is given as:

<p align="center">
    <img src="https://github.com/user-attachments/assets/9fc2d75a-2b41-4756-839a-72b0b3f91d36">
</p>

In such case, the probability p and 1-p can be expressed as:

<p align="center">
    <img src="https://github.com/user-attachments/assets/e931c54c-fe91-4309-9d69-ca22e5b55446">
</p>

The updated code is as follows:

```python
def negative_binomial_layer(x):
    num_dims = len(x.get_shape())

    # assuming that the input to this layer is a concatenation of k and m, extract the 2 variables from input x
    k, m = tf.unstack(x, num=2, axis=-1)

    k = tf.expand_dims(k, -1)
    m = tf.expand_dims(m, -1)

    # adding small epsilon so that log or lgamma functions do not produce nans in loss function
    k = tf.keras.activations.softplus(k)+1e-5
    m = tf.keras.activations.softplus(m)+1e-5
 
    out_tensor = tf.concat((k, m), axis=num_dims-1)
 
    return out_tensor

def negative_binomial_log_loss():
   def loss(y_true, y_pred):
       # Log loss of the negative binomial distribution
       # assuming that the input to this layer is a concatenation of k and m, extract the 2 variables from input x
       k, m = tf.unstack(y_pred, num=2, axis=-1)
   
       k = tf.expand_dims(k, -1)
       m = tf.expand_dims(m, -1)
   
       nll = (
           tf.math.lgamma(y_true+1)
           + tf.math.lgamma(k)
           - tf.math.lgamma(k+y_true)
           - k * tf.math.log(k)
           + k * tf.math.log(k+m)
           - y_true * tf.math.log(m) 
           + y_true * tf.math.log(k+m) 
       )
   
       return tf.reduce_mean(nll)       
       
   return loss
```

Lastly, we would need to write a module to extract the desired quantile values from the distribution. It is not straightforward to calculate the [CDF of the negative binomial distribution](https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function). Thus, we stick to the `sampling` approach as follows:

```python
def do_prediction(X, model, q=0.99, sample_size=10000):
    z = model.predict(X)
    res = []
    for i in range(len(z)):
        r1 = []
        for j in range(len(z[i])):
            k, m = z[i][j]
            p = k/(k+m)
            u = np.random.negative_binomial(k,p,sample_size).tolist()
            u = sorted(u)
            r1 += [u[int(q*len(u))]]
        res += [r1]
    
    res = np.array(res)
    res = res.reshape((res.shape[0], res.shape[1], 1))
    return res
```
<br/>

**99th percentile** actual vs. predicted demand values
<p align="center">
    <img src="https://github.com/user-attachments/assets/0212e8db-9900-45be-82a2-6b85db2522a0">
</p>

**50th percentile (median)** actual vs. predicted demand values
<p align="center">
    <img src="https://github.com/user-attachments/assets/765d84a2-f76c-446b-97e8-428c65823952">
</p>

Note that for the 50th percentile predictions, the predicted demand values almost coincides with the actual demand values whereas for 99th percentile, the predicted values are almost always higher than the actual. 

One might say that the median is a more accurate representation but in practical scenarios `99th percentile is more useful` because the supply or buffer we need to procure to cater all the demand is captured by the 99th percentile and thus ensure that demand will not exceed supply.
