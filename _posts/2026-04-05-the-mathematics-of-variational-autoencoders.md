---
layout: post
title:  "The Mathematics of Variational Autoencoders"
date:   2026-04-05 18:50:11 +0530
categories: math
---
Autoencoders are a commonly used technique to do dimensionality reduction or find a latent space. 
<br/><br/>
Given the input features `X=[x_0, x_1, ..., x_{m-1}]`, our goal is to find a latent space `Z=[z_0, z_1, ..., z_{h-1}]` where h < m, such that when we decode Z we should get back X or some X' which is very similar to X. Minimizing the difference between input X and output X' is handled by the MSE loss function `|X-X'|^2`. 
<br/><br/>
![mse](/docs/assets/loss_mse.png)
<br/><br/>
In variational autoencoders, instead of learning the latent space Z, we learn the probability distribution `P(Z|X, {w})` where `{w}` are the parameters of the probability distribution. Once we learn `{w}`, we can then determine `P(Z|X, {w})`. Z can be sampled from this distribution and decoded to get back X or some X' close to X. This enables us to generate samples in the output with variations, unlike autoencoder where we construct X' directly from Z. 
<br/><br/>
Note that the loss function cannot simply be the MSE between X and X' because in that case the network might learn arbitrary `{w}` that may not actually represent the distribution of the latent space Z or close to the actual distribution of Z. We will see how to formulate the loss function that handles both the encoder loss i.e. difference between actual and estimated distribution of Z and the decoder loss i.e. difference between the predicted X' and actual X.
<br/><br/>
To make the network learn one must ensure that the likelihood of the output is maximized (`Maximum Likelihood Estimation`). Usually one uses the `negative log likelihood (NLL)` of the output distribution as the loss function. For eg. assuming that the output is normally distributed with a constant variance, the NLL of the normal distribution would look like:
<br/><br/>
![mle_norm](/docs/assets/mle_norm.png)
<br/><br/>
Thus, we see that the MSE loss function is actually derived from the negative log likelihood value of the normal distribution where the variance is assumed to be a constant.
<br/><br/>
In our example since the input and output from VAE are the same i.e. `x`, one must maximize `p(x)`. We can write `p(x)` in terms of z as follows:
<br/><br/>
![px](/docs/assets/px.png)
<br/><br/>
But the above integral is difficult to compute since it is over all possible latent states `z`. It is even analytically difficult to compute the integral for simpler distributions. Hence we need a better method to approximate the negative log likelihood of `p(x)`. Let's see if we can come up with some easy to compute expression for `p(x)`.
<br/><br/>
Using Bayes' Theorem to compute the conditional probability of `z` given `x` as follows:
<br/><br/>
![cond](/docs/assets/cond2.png)
<br/><br/>
The denominator is `p(x)` which we saw above is intractable to compute. Instead of computing `p(z|x;{w})` using the Bayes' Theorem as above, we use some known and easy to compute probability distribution `q(z|x;{u})` as a proxy and then we minimize the difference between `p(z|x;{w})` and `q(z|x;{u})` in the loss function. Note that it is not necessary for `p(z|x;{w})` and `q(z|x;{u})` to be from the same family of distribution.
<br/><br/>
The difference between the probability distributions `p(z|x,{w})` and `q(z|x,{u})` can be captured using the KL Divergence as follows:
<br/><br/>
![kld](/docs/assets/kld.png)
<br/><br/>
The last inequality implies that the log likelihood of `p(x)` has a lower bound also famously known as `ELBO (Evidence Lower Bound)` which implies that maximizing the ELBO will also maximize the likelihood of `p(x)` which we means we can substitute the integration:
<br/><br/>
![px](/docs/assets/px.png)
<br/><br/>
with the following:
<br/><br/>
![elbo](/docs/assets/elbo.png)
<br/><br/>
As mentioned earlier, we don't need `q(z|x)` to be of the same family as the original distribution of `z`, which means we can assume `q(z|x)` to be normal distribution `N(z;mu, var)` where `mu` and `var` are the mean and variance of the normal distribution and these parameters are learnt using a neural network. The KL divergence term can be expanded as follows:
<br/><br/>
![kl](/docs/assets/kl2.png)
<br/><br/>
Note that `z`, `mu` and `var` are h-dimensional vectors/tensors. We assume that each dimension of the latent space `z` is independently sampled from the probability distribution `q(z|x)`.
It is not apparent how the expectation terms are derived in the above derivation. We can show the proof of how the expectation terms are derived as follows:
<br/><br/>
![ibp](/docs/assets/ibp.png)
<br/><br/>
<br/><br/>
![proof](/docs/assets/proof.png)
<br/><br/>
We have derived the expression for the KL divergence term in the ELBO equation above pertaining to the log likelihood of `p(x)`. Thus the substitited equation looks like below:
<br/><br/>
![logl](/docs/assets/logl.png)
<br/><br/>
Coming to the next part of the equation, `E[log(p(x|z)]` i.e. the expectation of the log likelihood of the output given the latent state `z`. If we assume a `normal distribution` with a constant variance, then the log likelihood translates into `MSE` loss function whereas if we assume a `Bernoulli distribution`, then the log likelihood translates into `Binary Cross Entropy (BCE)` loss function.
<br/><br/>
![msel](/docs/assets/msel2.png)
<br/><br/>
If we do not assume a constant variance in the above equation, then we have another parameter to learn and the log likelihood function should be accordingly modified as follows:
<br/><br/>
![msel](/docs/assets/msel3.png)
<br/><br/>
The above term is assuming a normal distribution wherease the below term is assuming a bernoulli distribution.
<br/><br/>
![bcel](/docs/assets/bcel2.png)
<br/><br/>
Thus the loss function to learn for training the variational autoencoder is the negative log likelihood of `p(x)`, which is as follows (for BCE loss):
<br/><br/>
![loss](/docs/assets/loss.png)
<br/><br/>
