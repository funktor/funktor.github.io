---
layout: post
title:  "The GPU Notes - Part 1"
date:   2025-06-18 18:50:11 +0530
categories: software-engineering
---

My notes on learning and using GPU and CUDA for Machine Learning, Deep Learning and Software Engineering problems. Most of the content has been inspired from the book "Programming Massively Parallel Processors-4th Edition".

1. **GPU - TFLOPS on Steroids**<br/><br/>
FLOPS - Floating Point Operations Per Seconds. 1 TFLOPS = 10^12 FLOPS. FLOPS is the number of floating point operations that can be done per second.<br/><br/>
Intel i9 processor with 24 CPU cores (latest as of writing this) can reach a peak of 1.3 TFLOPS for single precision i.e. 32-bit floats (or FP32).<br/><br/>
Compare this to the H100 GPU which has 14592 CUDA Cores and 640 Tensor Cores (will come later to this) and has limit of 989 TFLOPS for FP32 and 67 TFLOPS for FP64 (double-precision floats also known as 'double' in C).<br/><br/>
Latest GPU models also support half precision floats i.e. FP16 with higher TFLOPS (1979 TFLOPS) and FP8 with 3958 TFLOPS. FP8 and FP16 are used in deep learning for mixed precision training (revisited later).<br/><br/>
For understanding floating point representations, refer to one of my earlier posts on floating point compression: [Floating Point Compression](https://funktor.github.io/software-engineering/2025/02/12/time-series-compression.html)
<br/><br/>

2. **Smaller but more number of cores**<br/><br/>
Most commercially available CPUs have 4 to 24 cores and have larger L2 and L3 cache sizes. CPUs also have larger area for control units managing branch prediction etc. Each core also have its own L1 cache (both data and instruction). CPUs also have fewer number of channels to the DRAM i.e. the main memory. CPU cores are optimized for low latency whereas GPU cores are optimized for high throughput.<br/><br/>
To achieve low latency, CPU needs more number of registers or flip-flops which requires more power and thus one cannot have too many cores inside a CPU. CPUs are good for low latency operations on sequential programs for e.g. computing the first 1 million fibonacci numbers.<br/><br/>
[Nice write up on CPU architecture](https://www.redhat.com/en/blog/cpu-components-functionality)<br/><br/>
GPUs on the other hand have smaller cores but more number of cores. For example e.g. H100 have 14592 CUDA cores in addition to 640 Tensor cores. They also have smaller caches and smaller control units and more number of channels to the DRAM. The goal is high throughput for parallel computations.<br/><br/>
![CPU vs GPU](/docs/assets/cpu_gpu.png)
