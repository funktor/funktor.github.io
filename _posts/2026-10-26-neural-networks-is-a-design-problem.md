---
layout: post
title:  "Neural Networks is a Design Problem"
date:   2026-10-27 18:50:11 +0530
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
The graph structure:
<br/><br/>
![Autograd graph](/docs/assets/grad2.png)
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
So far we have been working with scalar variables x and y and scalar functions such as `f=g*h` or `g=5x+2y` and so on. But in practise, neural networks works with vectors, matrices and tensors. Here is an updated version of the classes which works with Numpy arrays as inputs instead of scalars. The core design principles remains unchanged, only how each class implements its own forward and backward methods gets changed. How the derivatives and gradients are computed are explained below.
<br/><br/>
```python
import numpy as np

def get_total_shape(shape:tuple):
    # returns product of all dimensions or in other words total number of elements
    n = 1
    for j in range(len(shape)):
        n *= shape[j]

    return n


def tensor_prod(a:list, b:list):
    # Custom dot product implementation with tensors/matrices of different shapes.
    # Numpy requires an array to have all elements of the same shape, but instead if we use a python list,
    # then we can use arrays of varying shapes in same list.
    n = len(a)
    m = len(a[0])
    p = len(b[0])

    out = []

    for i in range(n):
        r = []
        for j in range(p):
            x = a[i][0].shape[0]
            y = b[0][j].shape[1]
            w = np.zeros(shape=(x, y), dtype=np.float32)
            for k in range(m):
                u = a[i][k]
                v = b[k][j]
                w += np.dot(u, v)

            r += [w]
        out += [r]

    return out


class ParameterNode(Node):
    def __init__(self, shape:tuple):
        self.shape = shape
        nn_context.add_param_node(self)
        self.param_id = nn_context.get_param_node_id(self)

    def forward(self, param_vals:np.ndarray):
        return param_vals[self.param_id] if self.param_id is not None else None

    def backward(self):
        n = get_total_shape(self.shape)

        out = []
        for node in nn_context.param_nodes:
            m = get_total_shape(node.shape)

            if nn_context.param_nodes[node] == self.param_id: # self parameter return identity matrix.
                out += [np.eye(n, m, dtype=np.float32)]
            else: # others parameters return zero matrix.
                out += [np.zeros(shape=(n,m), dtype=np.float32)]

        return [out]


class ConstantNode(Node):
    def __init__(self):
        self.value = None

    def set_value(self, value:np.ndarray):
        self.value = value

    def forward(self, param_vals:np.ndarray):
        return self.value

    def backward(self):
        n = get_total_shape(self.value.shape)

        out = []
        for node in nn_context.param_nodes:
            m = get_total_shape(node.shape)
            out += [np.zeros(shape=(n,m), dtype=np.float32)]

        return [out]


class uDotvNode(Node):
    def __init__(self, u:Node, v:Node):
        self.u = u
        self.v = v
        self.fwd_cache = {}

    def forward(self, param_vals:np.ndarray):
        u_f = self.u.forward(param_vals)
        v_f = self.v.forward(param_vals)

        assert u_f.shape[-1] == v_f.shape[-2], "Shape mismatch for dot product"

        self.fwd_cache['u'] = u_f
        self.fwd_cache['v'] = v_f

        return np.dot(u_f, v_f)

    def backward(self):
        assert len(self.fwd_cache) > 0, "Forward pass should be called first"

        u_f = self.fwd_cache['u']
        v_f = self.fwd_cache['v']

        n_u = get_total_shape(u_f.shape)
        m_u = u_f.shape[-1]
        p_u = int(n_u/m_u)

        v_ft = np.swapaxes(v_f, -2, -1)
        n_v = get_total_shape(v_ft.shape)
        m_v = v_ft.shape[-1]
        p_v = int(n_v/m_v)

        a = np.kron(np.eye(p_u, p_u, dtype=np.float32), v_ft)
        b = np.kron(u_f, np.eye(p_v, p_v, dtype=np.float32))

        arr = [[a, b]]

        u_b = self.u.backward()
        v_b = self.v.backward()

        out = u_b + v_b

        return tensor_prod(arr, out)


class SumNode(Node):
    def __init__(self, u:Node, v:Node):
        self.u = u
        self.v = v
        self.fwd_cache = {}

    def forward(self, param_vals:np.ndarray):
        u_f = self.u.forward(param_vals)
        v_f = self.v.forward(param_vals)

        self.fwd_cache['u'] = u_f
        self.fwd_cache['v'] = v_f

        return u_f + v_f

    def backward(self):
        assert len(self.fwd_cache) > 0, "Forward pass should be called first"
        u_f:np.ndarray = self.fwd_cache['u']

        n = get_total_shape(u_f.shape)

        a = np.eye(n, n, dtype=np.float32)
        b = np.eye(n, n, dtype=np.float32)

        arr = [[a, b]]

        u_b = self.u.backward()
        v_b = self.v.backward()

        out = u_b + v_b # This is list concatenation not addition.

        return tensor_prod(arr, out)


class LeakyReluNode(Node): # Leaky ReLU function
    def __init__(self, u:Node, alpha:float=0.01):
        self.u = u
        self.alpha = alpha
        self.fwd_cache = {}

    def forward(self, param_vals:np.ndarray):
        u_f = self.u.forward(param_vals)

        self.fwd_cache['u'] = u_f
        out = np.where(u_f <= 0, self.alpha*u_f, u_f)

        return out

    def backward(self):
        assert len(self.fwd_cache) > 0, "Forward pass should be called first"
        u_f:np.ndarray = self.fwd_cache['u']

        n = get_total_shape(u_f.shape)

        a = np.eye(n, n, dtype=np.float32)
        b = np.ones(shape=u_f.shape, dtype=np.float32) # df/du = 1.0 if u > 0 else alpha
        b[u_f <= 0] = self.alpha

        np.fill_diagonal(a, b.flatten())

        arr = [[a]]
        out = self.u.backward()

        return tensor_prod(arr, out)


class SoftmaxNode(Node): # Softmax function
    def __init__(self, u:Node):
        self.u = u
        self.fwd_cache = {}

    def forward(self, param_vals:np.ndarray):
        u_f = self.u.forward(param_vals)

        self.fwd_cache['u'] = u_f

        max_vals = np.max(u_f, axis=-1, keepdims=True)
        u_f = np.exp(u_f - max_vals)
        sum_vals = np.sum(u_f, axis=-1, keepdims=True)
        out = u_f/sum_vals

        self.fwd_cache['out'] = out

        return out

    def backward(self):
        assert len(self.fwd_cache) > 0, "Forward pass should be called first"
        u_f:np.ndarray = self.fwd_cache['u']
        v:np.ndarray = self.fwd_cache['out']
        v = v.flatten()

        n = get_total_shape(u_f.shape)

        m = u_f.shape[-1]
        p = int(n/m)

        a = np.zeros(shape=(n, n), dtype=np.float32)

        # df_ij/du_kl = u_ij*(1-u_ij) if ij == kl
        # df_ij/du_kl = -u_ij*u_kl if i == k and j != l
        # else df_ij/du_kl = 0

        for i in range(p):
            for j in range(m):
                for k in range(m):
                    h = i*m + j
                    g = i*m + k

                    if j == k:
                        a[h,g] = v[h]*(1.0 - v[g])
                    else:
                        a[h,g] = -v[h]*v[g]

        arr = [[a]]
        out = self.u.backward()

        return tensor_prod(arr, out)


class CategoricalCrossEntropyNode(Node): # Categorical Cross Entropy loss function
    def __init__(self, u:Node, y:Node, epsilon:float=1e-10):
        self.u = u
        self.y = y
        self.epsilon = epsilon
        self.fwd_cache = {}

    def forward(self, param_vals:np.ndarray):
        u_f = self.u.forward(param_vals)
        y_f = self.y.forward(param_vals)

        self.fwd_cache['u'] = u_f
        self.fwd_cache['y'] = y_f

        a = -y_f * np.log(u_f + self.epsilon)
        out = np.sum(a)

        return out

    def backward(self):
        assert len(self.fwd_cache) > 0, "Forward pass should be called first"
        u_f:np.ndarray = self.fwd_cache['u']
        y_f:np.ndarray = self.fwd_cache['y']

        y_f = y_f.flatten()
        u_f = u_f.flatten()

        a = np.expand_dims(-y_f/(u_f + self.epsilon), axis=0) # df/du = -y/(u+eps)
        b = np.expand_dims(-np.log(u_f + self.epsilon), axis=0) # df/dy = -log(u+eps)

        arr = [[a, b]]

        u_b = self.u.backward()
        y_b = self.y.backward()

        out = u_b + y_b # This is list concatenation not addition.

        return tensor_prod(arr, out)


class MSENode(Node): # MSE loss function
    def __init__(self, u:Node, y:Node):
        self.u = u
        self.y = y
        self.fwd_cache = {}

    def forward(self, param_vals:np.ndarray):
        u_f = self.u.forward(param_vals)
        y_f = self.y.forward(param_vals)

        self.fwd_cache['u'] = u_f
        self.fwd_cache['y'] = y_f

        out = 0.5*np.sum((u_f-y_f)**2)

        return out

    def backward(self):
        assert len(self.fwd_cache) > 0, "Forward pass should be called first"
        u_f:np.ndarray = self.fwd_cache['u']
        y_f:np.ndarray = self.fwd_cache['y']

        y_f = y_f.flatten()
        u_f = u_f.flatten()

        a = np.expand_dims(u_f - y_f, axis=0) # df/du = u-y
        b = np.expand_dims(y_f - u_f, axis=0) # df/dy = y-u

        arr = [[a, b]]

        u_b = self.u.backward()
        y_b = self.y.backward()

        out = u_b + y_b # This is list concatenation not addition.

        return tensor_prod(arr, out)
```
<br/><br/>
Above we define all basic and necessary neural network operations and their corresponding forward and backward methods assuming that the inputs are tensors (matrices here) and not scalars. I will not go through each and every operation and their forward() and backward() method implementations but just give a high level overview. Assuming a given operation or function f takes inputs u and v which are both tensors of shapes `(u0, u1, ...)` and `(v0, v1, ...)` respectively and f computes a tensor of shape `(f0, f1, ...)`. The forward function is quite intuitive as the output is of shape `(f0, f1, ...)` but what is the shape of the output in the backward pass?
<br/><br/>
What we are computing in the backward pass are `df/du` and `df/dv`, so `df/du` here is a 2-D matrix or tensor where each element is `df[i0,i1,...]/du[j0,j1,...]`.
<br/><br/>
Basically, we flatten the forward output matrix f having number of elements as `n` and similarly flatten the input matrix u having number of elements as `m_u` and then find the derivative of each element of flattened f w.r.t. each element of flattened u. Each row of the derivative matrix `df/du` consists of same f element and each column consist of same u, thus number of rows equals `n` and number of rows equals `m_u`. Same computations repeat for df/dv too but now we flatten the tensor v instead of tensor u. Thus although the forward pass matrix is of shape `n` but each backward pass matrix are of shapes `n*m_u` and `n*m_v` respectively.
<br/><br/>
Below image shows an example of the derivative matrices where u, v, and f are assumed to be 2D tensors of shapes (32,32), (32,7) and (64,64) respectively. Note that we have not defined how f is expressed in terms of inputs u and v.
<br/><br/>
![Tensor gradients](/docs/assets/grad3.png)
<br/><br/>
We are going to deep dive into one of the operations - `uDotvNode`. This operation is basically dot product of tensors u and v. For scalars u and v, the backward pass is pretty straightforward as seen earlier. For tensors, the derivatives follow the above rules of differentiation. Algebraically, one could find the elements of the tensor (matrix) as follows:
<br/><br/>
![Tensor gradients](/docs/assets/grad4.png)
<br/><br/>
In the above schema, we can make the following substitutions and get a more simplified versions of the equations as follows. The tensor v' is just v with the last 2 axes swapped:
<br/><br/>
![Tensor gradients](/docs/assets/grad5.png)
<br/><br/>
We can either compute the derivatives using a for-loop or use tensor operations such as the ones implemented in the code above. The tensor operations can be written as follows:
<br/><br/>
![Tensor gradients](/docs/assets/grad6.png)
<br/><br/>
It will not be apparent immediately as to how one can write the above derivatives in terms of the kronecker product of tensors but one can verify this is indeed true with small 2D tensors.
<br/><br/>
Now, we have all the required classes to build a simple neural network from scratch. For that, we will generate some dummy data (inputs and targets) for a regression problem as well as a multi-class classification problem. First, we tackle a regression problem. Here is how I would generate some dummy data.
<br/><br/>
```python
num_examples = 64
num_features = 32

features = np.random.normal(loc=0.0, scale=1.0, size=(num_examples, num_features))
target = np.random.normal(loc=0.0, scale=1.0, size=(num_examples,))
```
<br/><br/>
Finally, we build and train a small neural network with the MSE loss function as follows:
<br/><br/>
```python
NNContext.init()

w1 = ParameterNode(shape=(num_features, 16))
w2 = ParameterNode(shape=(16, 1))

inp = ConstantNode()
oup = ConstantNode()

l1 = uDotvNode(inp, w1)
l2 = LeakyReluNode(l1)
l3 = uDotvNode(l2, w2)
l4 = MSENode(l3, oup)

learning_rate = 1e-4
num_epochs = 500

w1v = np.random.normal(loc=0.0, scale=1.0, size=w1.shape)
w2v = np.random.normal(loc=0.0, scale=1.0, size=w2.shape)

for epoch in range(num_epochs):
    inp.set_value(features)
    oup.set_value(target)
    loss = l4.forward([w1v, w2v]) # forward pass
    print(epoch, loss)
    grads = l4.backward()[0] # backward pass
    w1v = w1v - learning_rate*grads[0].reshape(w1.shape) # update weights using gradient descent
    w2v = w2v - learning_rate*grads[1].reshape(w2.shape) # update weights using gradient descent

```
<br/><br/>
Once we run the above code, it will print the epoch and its corresponding loss value. If the loss decrease over epochs, it means that the model is learning and implementation is (mostly) correct. Although it is a small and a dummy example but still one can verify if the neural network is implemented correctly because neural networks are universal function approximators and even if the data is bad or concocted, it will still learn the pattern and loss will decrease (assuming the loss function is implemented correctly).
<br/><br/>
Similar to the regression problem, we can also create a dummy multi-class classification problem as shown below:
<br/><br/>
```python
num_classes = 5 # we have 5 class classification problem.
num_examples = 64
num_features = 32

features = np.random.normal(loc=0.0, scale=1.0, size=(num_examples, num_features))
labels = np.random.randint(0, num_classes, size=(num_examples,)) # assign a random class label from [0,4] to each instance
target = np.zeros(shape=(num_examples, num_classes), dtype=np.float32)
target[np.arange(num_examples), labels] = 1.0 # do one hot encoding

NNContext.init()

w1 = ParameterNode(shape=(num_features, 16))
w2 = ParameterNode(shape=(16, num_classes))

inp = ConstantNode()
oup = ConstantNode()

l1 = uDotvNode(inp, w1)
l2 = LeakyReluNode(l1)
l3 = uDotvNode(l2, w2)
l4 = SoftmaxNode(l3)
l5 = CategoricalCrossEntropyNode(l4, oup)

learning_rate = 1e-4
num_epochs = 500

w1v = np.random.normal(loc=0.0, scale=1.0, size=w1.shape)
w2v = np.random.normal(loc=0.0, scale=1.0, size=w2.shape)

for epoch in range(num_epochs):
    inp.set_value(features)
    oup.set_value(target)
    loss = l5.forward([w1v, w2v]) # forward pass
    print(epoch, loss)
    grads = l5.backward()[0] # backward pass
    w1v = w1v - learning_rate*grads[0].reshape(w1.shape)
    w2v = w2v - learning_rate*grads[1].reshape(w2.shape)

```
<br/><br/>
The above neural network is similar to the regression one above with minor differences such as adding a softmax layer and using a categorical cross entropy loss function instead of a MSE loss function. In this example too, we see that the loss decreases over epoch implying that the neural network is learning.
<br/><br/>
So far we have only focussed on the designing aspects of the neural network and a bit on the tensor operations for gradient calculations. But a production grade neural networks would require much more than that, but that is not the goal of this post. This post was primarily targeted towards software engineers who would like to extend their system design skills to the machine learning domain. As a side note, the design is not neural network specific and it should work with all kinds of ML modelling algorithms. For the enthusiastic readers, I would urge them to extend the design and implementation to more operations such as adding different loss functions such as BCE, Pinball Loss etc., adding more non-linear functions such as GeLU, Tanh etc., experiment with complicated neural networks such as CNN or LSTM by extending the design, add residual layers and so on.
<br/><br/>
Also, note that on the performance side, the above implementation is not scalable or fast. Use PyTorch for all production grade requirements. As an exercise, one can take up the task to make the implementation faster and scalable by leveraging GPU and/or re-implementing the same with C/C++.
