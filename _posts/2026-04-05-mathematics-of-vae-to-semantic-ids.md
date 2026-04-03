---
layout: post
title:  "Mathematics of VAE to Semantic IDs"
date:   2026-04-05 18:50:11 +0530
categories: math
---
Autoencoders are a commonly used technique to do dimensionality reduction or find a latent space. 
<br/><br/>
Given the input features `X=[x_0, x_1, ..., x_{m-1}]`, our goal is to find a latent space `Z=[z_0, z_1, ..., z_{h-1}]` where h < m, such that when we decode Z we should get back X or some X' which is very similar to X. Minimizing the difference between input X and output X' is handled by the MSE loss function `|X-X'|^2`. 
<br/><br/>
![mse](/docs/assets/loss_mse.png)
<br/><br/>
In variational autoencoders, instead of learning the latent space Z, we learn the probability distribution `P(Z|X, {w})` where `{w}` are the parameters of the probability distribution. Once we learn `{w}`, we can then determine `P(Z|X, {w})`. Z can be sampled from this distribution and decoded to get back X or some X' close to X. This enables one to learn variations in the input data and generate samples in the output with 
<br/><br/>
Note that the loss function cannot simply be the MSE between X and X' because in that case the network might learn arbitrary `{w}` that may not actually represent the distribution of the latent space Z or close to the actual distribution of Z given X. We will see how to formulate the loss function that handles both the encoder loss i.e. difference between actual and estimated distribution of Z and the decoder loss i.e. difference between the predicted X' and actual X.
<br/><br/>
Using Bayes' Theorem to compute the conditional probability of `z` given `x` as follows:
<br/><br/>
![cond](/docs/assets/cond.png)
<br/><br/>
The denominator is an integration over all possible `z`, which is intractable to compute analytically for most probability distributions. Instead of computing `p(z|x,{w})` using the Bayes' Theorem as above, we use some known and easy to compute probability distribution `q(z|x,{u})` as a proxy and then we minimize the difference between `p(z|x,{w})` and `q(z|x,{u})` in the loss function. Note that it is not necessary for `p(z|x,{w})` and `q(z|x,{u})` to be from the same family of distribution.
<br/><br/>
The difference between the probability distributions `p(z|x,{w})` and `q(z|x,{u})` can be captured using the KL Divergence as follows:
