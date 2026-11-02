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
But note that, now the derivative of the loss L w.r.t. the parameters, theta and phi are not simple, because each moving average term (or the error terms) are also dependent on the same parameters and cannot be treated as constants unlike the past observation terms. The modified partial derivatives are as follows:
<br/><br/>
![arima5](/docs/assets/arima5.png)
<br/><br/>
The above set of partial derivatives can be simplified by using recurrence relations F and G corresponding to the partial derivatives of the error at timestep t w.r.t. the parameters theta and phi respectively as follows:
<br/><br/>
![arima7](/docs/assets/arima7.png)
<br/><br/>
Either we explicitly compute the recurrence relations F and G and then the partial derivatives for gradient descent algorithm, or re-use the automatic differentiation implementation from the previous post. Using the automatic differentiation routine for scalars, we can very easily compute the loss as well as the updates to the parameters theta and phi corresponding to the lags and moving averages. But there was a gotcha in the prior implementation. In the ARIMA equation with moving average terms, each error term at timestep T is dependent recursively on the error terms from the last q timesteps. Thus a plain recursion implementation will have an exponential time complexity for automatic differentiation. Modified version shown below, caches all the node's forward and backward results for re-use.
<br/><br/>
```python
import numpy as np
import math

class node:
    def __init__(self):
        pass
    def forward(self, param_values):
        pass
    def backward(self):
        pass

class nn_context:
    param_nodes = {} # parameters node to id
    num_params = 0 # number of parameters
    fwd_cache = {}
    bwd_cache = {}

    @staticmethod
    def init():
        # init the settings
        nn_context.param_nodes = {}
        nn_context.num_params = 0
        nn_context.fwd_cache = {}
        nn_context.bwd_cache = {}

    @staticmethod
    def add_param_node(node:node):
        # add parameter node, increment id by 1
        nn_context.param_nodes[node] = nn_context.num_params
        nn_context.num_params += 1

    @staticmethod
    def get_param_node_id(node:node):
        # get node id for node
        return nn_context.param_nodes[node] if node in nn_context.param_nodes else None

    @staticmethod
    def get_num_params():
        # get number of parameters
        return nn_context.num_params

    @staticmethod
    def reset_cache():
        # reset fwd and bwd caches
        nn_context.fwd_cache = {}
        nn_context.bwd_cache = {}


class param_node(node):
    def __init__(self):
        nn_context.add_param_node(self)
        self.param_id = nn_context.get_param_node_id(self)

    def forward(self, param_vals):
        nn_context.fwd_cache[self] = param_vals[self.param_id] if self.param_id is not None else None
        return nn_context.fwd_cache[self]

    def backward(self):
        n = nn_context.get_num_params()
        out = np.zeros(shape=(1,n))
        out[0,self.param_id] = 1.0 # 1.0 only for self, 0.0 for all other parameters.
        nn_context.bwd_cache[self] = out
        return nn_context.bwd_cache[self]


class constant_node(node):
    def __init__(self, value):
        self.value = value

    def forward(self, param_vals):
        nn_context.fwd_cache[self] = self.value
        return nn_context.fwd_cache[self]

    def backward(self):
        n = nn_context.get_num_params() # all leaf nodes need to access nn_context
        nn_context.bwd_cache[self] = np.zeros(shape=(1,n))
        return nn_context.bwd_cache[self]


class sum_node(node):
    def __init__(self, u:node, v:node):
        self.u = u
        self.v = v

    def forward(self, param_values):
        if self not in nn_context.fwd_cache:
            u_f = self.u.forward(param_values)
            v_f = self.v.forward(param_values)
            nn_context.fwd_cache[self] = u_f + v_f

        return nn_context.fwd_cache[self]

    def backward(self):
        if self not in nn_context.bwd_cache:
            a:np.ndarray = self.u.backward()
            b:np.ndarray = self.v.backward()
            nn_context.bwd_cache[self] = np.array([[1.0, 1.0]]) @ np.concatenate((a, b), axis=0) # s = u_f + v_f, ds/du_f = 1 and ds/dv_f = 1

        return nn_context.bwd_cache[self]


class prod_node(node):
    def __init__(self, u:node, v:node):
        self.u = u
        self.v = v

    def forward(self, param_values):
        if self not in nn_context.fwd_cache:
            u_f = self.u.forward(param_values)
            v_f = self.v.forward(param_values)
            nn_context.fwd_cache[self] = u_f * v_f

        return nn_context.fwd_cache[self]

    def backward(self):
        if self not in nn_context.bwd_cache:
            u_f = nn_context.fwd_cache[self.u]
            v_f = nn_context.fwd_cache[self.v]

            a:np.ndarray = self.u.backward()
            b:np.ndarray = self.v.backward()

            nn_context.bwd_cache[self] = np.array([[v_f, u_f]]) @ np.concatenate((a, b), axis=0) # s = u_f * v_f, ds/du_f = v_f and ds/dv_f = u_f

        return nn_context.bwd_cache[self]
```
<br/><br/>
Using caching, the exponential complexity gets down to a quadratic time complexity algorithm.  To compute the loss function and the gradient updates, we need to write the routine to build the graph as follows:
<br/><br/>
```python
def get_predictions(inp, ser_len, p, q):
    # get the prediction nodes corresponding to each timestep
    theta = [param_node() for _ in range(p)] # theta corresponding to lag
    phi = [param_node() for _ in range(q)] # phi corresponding to moving averages
    bias = param_node()

    err_nodes = []
    pred_nodes = []

    for t in range(ser_len):
        pred = sum_node(constant_node(0.0), bias)

        for j in range(1, p+1): # contribution of the lagged observations.
            if t-j >= 0:
                g = prod_node(theta[j-1], constant_node(inp[t-j]))
                pred = sum_node(pred, g)
            else:
                break

        for k in range(1, q+1): # contribution of the moving averages.
            if t-k >= 0:
                h = prod_node(phi[k-1], err_nodes[-k])
                pred = sum_node(pred, h)
            else:
                break

        pred_nodes += [pred]
        err = sum_node(constant_node(-inp[t]), pred) # error node
        err_nodes += [err]

    return pred_nodes, err_nodes

nn_context.init()
pred_nodes, err_nodes = get_predictions(inp, ser_len, p, q)

# compute the cost (loss) due to all the error terms. 
cost = constant_node(0.0)
for err_node in err_nodes:
    cost = sum_node(cost, prod_node(constant_node(1.0/ser_len), prod_node(err_node, err_node)))
```
<br/><br/>
Let's generate a dummy time series of size 1000 and use lag=10 i.e. 10 past observations are used to predict current observation and number of moving average terms (q=3) and then run the training and predictions as follows:
<br/><br/>
```python
ser_len = 1000
p = 10
q = 3
learning_rate = 1e-3
num_epochs = 500

# generate time series data
steps = np.random.normal(loc=0, scale=0.5, size=ser_len)
random_walk = np.cumsum(steps)
inp = 1 / (1 + np.exp(-random_walk / 3))

# training for parameters.
theta_v = np.random.normal(0.0, 1e-3, size=(p,)) # initial theta
phi_v = np.random.normal(0.0, 1e-3, size=(q,)) # initial phi
bias_v = np.array([0.0]) # initial bias term

for epoch in range(num_epochs):
    # reset the fwd and bwd caches after each epoch so as to not use incorrect values.
    nn_context.reset_cache()
    d = cost.forward(np.concatenate((theta_v, phi_v, bias_v)))
    print(epoch, d)

    # compute the gradients
    grads = cost.backward()[0]

    # update the parameters
    theta_v -= learning_rate*grads[:p]
    phi_v -= learning_rate*grads[p:p+q]
    bias_v -= learning_rate*grads[p+q:]
```
<br/><br/>
