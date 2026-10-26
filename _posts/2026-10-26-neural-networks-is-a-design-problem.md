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
In fact functions are 1st class citizens for any ML modelling problem.

2. Functions are modelled as graphs. For e.g. f(g,h) = g*h, g(x,y) = 5x+2y, h(x,y) = 6x.
Can be modelled as a root node f, having 2 children nodes g and h and each of g and h having x and y as their children nodes.

3. Derivatives are decomposable: 
i.e. `df/dx = (df/dg).(dg/dx) + (df/dh).(dh/dx)` and
     `df/dy = (df/dg).(dg/dy) + (df/dh).(dh/dy)`.

The above 2 equations can be re-written using matrix operations as follows:
`[df/dx df/dy] = [df/dg df/dh].[dg/dx dg/dy
                                dh/dx dh/dy]`
```
<br/><br/>
