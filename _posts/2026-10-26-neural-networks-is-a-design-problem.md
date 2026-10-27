---
layout: post
title:  "Neural Networks is a Design Problem"
date:   2026-10-28 18:50:11 +0530
categories: ml
---
You might find a lot of well-written posts over the internet and popular books teaching how to build a neural network from scratch. Most of them treat neural networks as a special category of ML problem (tool or a framework whatever you want to call it). Implementing a neural network from scratch is basically equivalent to implementing backpropagation algorithm. That is how, I learnt it initially and that is how most tutorials teach it. As a matter of fact, that is not completely incorrect but the way it is taught makes one feel that backpropagation is some special algorithm and a one-off technique. But after implementing a series of ML models from scratch, it came to realisation that the common pattern across all ML models is finding the derivative of the loss function w.r.t. the variables.
<br/><br/>
Different ML models use different loss functions and as a result, one need to rewrite the forward pass and the derivative calculation steps from scratch every-time one writes a new ML model. For complicated loss functions or loss functions using composition of variables (such as in neural networks), the derivative calculation can be tiresome (and unscalable) if one tries to find the derivative analytically by hand or error prone if using some 1st order approximation technique.
<br/><br/>
In this post, I show that building a neural network from scratch is more of a design problem rather than spending time and energy solving for the derivatives in backpropagation. The design is based on autograd.
<br/><br/>
The core ideas behind designing a neural network are as follows:
```
1. Functions are 1st class citizens.
Everything in a neural network is considered to be a function : outputs, loss, gradients etc.
In fact functions are assumed to be 1st class citizens for any ML modelling problem.

2. Functions are modelled as graphs. For e.g. f(g,h) = g*h, g(x,y) = 5x+2y, h(x,y) = y*sin(x).
Can be modelled as a root node f, having 2 children nodes g and h and each of g and h having x and y as their children nodes.

3. Derivatives are decomposable: 
i.e. `df/dx = (df/dg).(dg/dx) + (df/dh).(dh/dx)` and
     `df/dy = (df/dg).(dg/dy) + (df/dh).(dh/dy)`.

The above 2 equations can be re-written using matrix operations as shown below:
```
<br/><br/>
In terms of matrix multiplication operations:
<br/><br/>
![Grad matmul](/docs/assets/grad1.png)
<br/><br/>
Each node of the graph (point no. 2 above) can be assumed to implement 2 functions - forward and backward. <br/><br/>
The forward function computes the value of the function for a set of values provided for the variables. For e.g. f(g,h) at (x,y)=(1,2) can be implemented as:<br/><br/>
`f(g(1,2), h(1,2)) = (5*1+2*2)*(2*sin(1))`
<br/><br/>
On the other hand, the backward function is assumed to implement the gradient calculations. But you might ask, how's the graph structure useful in all of these. Here's the trick, both the forward and backward pass (functions) leverage the graph structure to simplify the calculations. In the case of forward pass, this is more clear. For a given node e.g. Node f above, in order to compute the forward pass for f, it recursively calls the forward pass of its child nodes g and h, collects the results and just runs the multiplication operation g*h. Node f does not need to bother about how nodes g and h implements their own forward passes or what are the functional forms of g and h. Node f just implements `f=g*h` recursively.
<br/><br/>
The backward pass also leverages the recursive nature of the graph structure but is a bit more involved. Similar to the forward pass, the backward pass for a node f also need not bother about the derivatives of g and h w.r.t. the variables. Since we can represent the gradient i.e. the derivatives of f w.r.t. the variables x and y as matrix multiplication operations as shown above, thus the backward pass for node f only needs to know what are the derivatives of f w.r.t. g and h which is much more simpler than knowing the derivative for a full chain of operations. Once we know the derivative of f w.r.t. to g and h, we can use recursion to compute the gradients of g and h and then compute the dot product as shown above.
<br/><br/>
Thus, although the function f is a complicated function involving polynomial and sine terms, but the derivative calculations are simpler because the individual operations are elementary. Here is how one would design simple classes for solving the above problem.
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

    @staticmethod
    def init():
        # init the settings
        nn_context.param_nodes = {}
        nn_context.num_params = 0

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


class param_node(node):
    def __init__(self):
        nn_context.add_param_node(self)
        self.param_id = nn_context.get_param_node_id(self)

    def forward(self, param_vals):
        return param_vals[self.param_id] if self.param_id is not None else None

    def backward(self):
        n = nn_context.get_num_params()
        out = np.zeros(shape=(1,n))
        out[0,self.param_id] = 1.0 # 1.0 only for self, 0.0 for all other parameters.
        return out


class constant_node(node):
    def __init__(self, value):
        self.value = value

    def forward(self, param_vals):
        return self.value

    def backward(self):
        n = nn_context.get_num_params() # all leaf nodes need to access nn_context
        return np.zeros(shape=(1,n))


class sum_node(node):
    def __init__(self, u:node, v:node):
        self.u = u
        self.v = v

    def forward(self, param_values):
        u_f = self.u.forward(param_values)
        v_f = self.v.forward(param_values)

        return u_f + v_f

    def backward(self):
        a:np.ndarray = self.u.backward()
        b:np.ndarray = self.v.backward()

        return np.array([[1.0, 1.0]]) @ np.concatenate((a, b), axis=0) # s = u_f + v_f, ds/du_f = 1 and ds/dv_f = 1


class prod_node(node):
    def __init__(self, u:node, v:node):
        self.u = u
        self.v = v
        self.fwd_cache = {}

    def forward(self, param_values):
        u_f = self.u.forward(param_values)
        v_f = self.v.forward(param_values)

        self.fwd_cache['u'] = u_f
        self.fwd_cache['v'] = v_f

        return u_f * v_f

    def backward(self):
        u_f = self.fwd_cache['u']
        v_f = self.fwd_cache['v']

        a:np.ndarray = self.u.backward()
        b:np.ndarray = self.v.backward()

        return np.array([[v_f, u_f]]) @ np.concatenate((a, b), axis=0) # s = u_f * v_f, ds/du_f = v_f and ds/dv_f = u_f


class sine_node(node):
    def __init__(self, u:node):
        self.u = u
        self.fwd_cache = {}

    def forward(self, param_values):
        u_f = self.u.forward(param_values)
        self.fwd_cache['u'] = u_f

        return math.sin(u_f)

    def backward(self):
        u_f = self.fwd_cache['u']
        a:np.ndarray = self.u.backward()
        return np.array([[math.cos(u_f)]]) @ a # s = sin(u_f), ds/du_f = cos(u_f)
```
<br/><br/>
For each operation, define a class with its own forward and backward method. Note that we defined classes for parameters and constants too. This design makes it simpler and intuitive to work with variables and constants instead of explicitly tagging them in the class methods. Note that, each backward method just computes the derivative of its forward value w.r.t. each of its input values. Correlating against the matrix multiplication operation shown above, in `prod_node backward()` method `np.array([[v_f, u_f]])` corresponds to the left 1x2 matrix on the R.H.S and `np.concatenate((a, b), axis=0)` corresponds to the right 2x2 matrix on R.H.S.
<br/><br/>
For e.g. in `f=g*h` we just compute `df/dg` and `df/dh` in the backward method without expanding g or h. In order to call the above operations, we can simply do as follows:
<br/><br/>
```python
nn_context.init()

x = param_node()
y = param_node()
g = sum_node(prod_node(constant_node(5.0), x), prod_node(constant_node(2.0), y)) # g = 5x + 2y
h = prod_node(y, sine_node(x)) # h = y*sin(x)
f = prod_node(g, h) # f = g*h

print(f.forward([1.0, 2.0]))
print(f.backward())
```
<br/><br/>
The need for the static methods in `nn_context` is to maintain global variables for the parameters. One can also tweak the parameter class to do that but the api interface is more clean with the current approach. The call to `f.backward()` returns an array where the 1st element corresponds to the gradient `df/dx` and the 2nd element corresponds to `df/dy`. One can also verify these numbers by running a 1st order approximation.
<br/><br/>
```python
h = 1e-7
(f.forward([1.0+h, 2.0]) - f.forward([1.0, 2.0]))/h # df/dx = (f(x+h,y)-f(x,y))/h
(f.forward([1.0, 2.0+h]) - f.forward([1.0, 2.0]))/h # df/dy = (f(x,y+h)-f(x,y))/h
```
<br/><br/>
