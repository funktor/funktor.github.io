---
layout: post
title:  "Generative Neural Networks From Scratch"
date:   2026-05-30 18:50:11 +0530
categories: ml
---
In this post, we are going to deep dive and implement some of the most commonly used generative neural net architectures starting from `GANs (Generative Adversarial Networks)`. In the process we will also understand the differences between each architecture and their pros and cons. 
<br/><br/>
The code implementations will primarily be in PyTorch but we will not limit ourselves to Python only and will go ahead and implement the architectures in C/C++ and CUDA for parallel processing and speedup. Although do not expect the final implementations to be at par with bulit-in libraries. But we will try to make them as efficient as possible by the end of this post.
<br/><br/>
The focus of this post is going to be the following special architectures and generative AI/ML algorithms:
<br/><br/>
```
1. Generative Adversarial Network (GAN)
2. Variational Autoencoder (VAE)
3. Vector Quantized VAE (VQ-VAE)
4. Discrete VAE (d-VAE)
5. Residual Quantized VAE (RQ-VAE)
6. Diffusion Models
7. Flow Matching
8. Normalized Flows
9. Attention and Transformers
```
For each one of the above we will also look at how they are useful and what problems they solve apart from their pros and cons and implementation details.
<br/><br/>
