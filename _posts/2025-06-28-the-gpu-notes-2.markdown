---
layout: post
title:  "The GPU Notes - Part 2"
date:   2025-06-30 18:50:11 +0530
categories: software-engineering
---

In the [previous post](https://funktor.github.io/software-engineering/2025/06/21/the-gpu-notes-1.html), I started jotting down my learnings with GPU and CUDA programming and explored some of the fundamentals of GPU architecture and memory. Towards the end, we saw how we can speed up memory access in matrix multiplication in order increase TFLOPS by using shared memory tiling. In this part we will look at more GPU optimization techniques through more examples.

1. **Memory Coalescing**<br/><br/>
In the previous post we saw that reading from global memory in GPU is slow because firstly there are implemented off-chip and secondly they are implemented using the DRAM cells. Shared memory and caches on the other hand are implemented on-chip and using SRAM cells.<br/><br/>
Similar to cache lines in CPU, when a location in the global memory is accessed, "nearby" locations are also accessible in the same CPU cycle. This saves number of CPU cycles to read the data from global memory. Threads in a warp (group of 32 threads) follow the same instruction and as a result the threads in warp access consecutive memory locations in the global memory. Global memory addresses are 128-byte aligned and thus accessing 4-byte floats (fp32) by a warp of 32 threads can be done in a single pass (coalesced). Accessing with offset or strided access patterns are not coalesced.
One possible solution to speed-up access to non-consecutive memory locations is to read the data from the global memory to shared memory in coalesced manner and then read from shared memory in un-coalesced manner since shared memory is 100x faster than global memory. In our matrix multiplication using tiling example from previous part highlights this optimization.

2. **Thread Coarsening**<br/><br/>

3. **Minimize Control Divergence in Warps**<br/><br/>

4. **Convolution Kernel**<br/><br/>

5. **Stencils**<br/><br/>

6. **Parallel histogram, reduction and atomic operations**<br/><br/>

7. **Prefix Sums**<br/><br/>

