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
