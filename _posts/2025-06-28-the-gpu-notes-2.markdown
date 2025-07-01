---
layout: post
title:  "The GPU Notes - Part 2"
date:   2025-07-05 18:50:11 +0530
categories: software-engineering
---

In the [previous post](https://funktor.github.io/software-engineering/2025/06/21/the-gpu-notes-1.html), I started jotting down my learnings with GPU and CUDA programming and explored some of the fundamentals of GPU architecture and memory. Towards the end, we saw how we can speed up memory access in matrix multiplication in order increase TFLOPS by using shared memory tiling. In this part we will look at more GPU optimization techniques through more examples.

1. **Memory Coalescing**<br/><br/>
In the previous post we saw that reading from global memory in GPU is slow because firstly they are implemented off-chip and secondly they are implemented using the DRAM cells. Shared memory and caches on the other hand are implemented on-chip and using SRAM cells. SRAM is much faster as compared to DRAM.<br/><br/>
Similar to cache lines in CPU, when a location in the global memory is accessed, "nearby" locations are also accessible in the same CPU cycle. This saves number of CPU cycles to read the data from global memory. In CPU, the cache line size is usually 64-bytes. Once read from RAM they are stored in either L1, L2 or L3 cache. <br/><br/>
Threads in a warp (group of 32 threads) follow the same instruction (SIMD model) and as a result the threads in warp access consecutive memory locations in the global memory. Global memory addresses are 128-byte aligned and thus accessing 4-byte floats (fp32) by a warp of 32 threads can be done in a single pass (coalesced).<br/><br/>
Accessing with offset or strided access patterns are not coalesced. Each 128-byte segment in global memory is termed as a burst.<br/><br/>
    ```cpp
    __global__
    void offset_add(float *inp, float *oup, int n, int offset) {
        // When offset=0, access is coalesced but if offset=1, then some threads in a warp will read data from burst i
        // and the remaining from burst i+1.
        // In the worst case, each thread reads two bursts from the global memory.
    
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        index += offset;
        oup[index] = inp[index] + 100.0;
    }
    
    __global__
    void strided_add(float *inp, float *oup, int n, int stride) {
        // When stride=1, access is coalesced but if stride=2,
        // then some threads in a warp will read data from burst i and the remaining from burst 2i.
        // In the worst case, each thread reads 32 bursts from the global memory
        // because when stride >= 32 each thread in a warp reads from a different burst.
    
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        index *= stride;
        oup[index] = inp[index] + 100.0;
    }
    ```
    <br/><br/>
Performance takes a major hit when using strided global memory access.<br/><br/>
In matrix multiplication using the below kernel:<br/><br/>
    ```cpp
    __global__ 
    void cuda_mul(float *a, float *b, float *c, int n, int m, int p) {
        int row = blockIdx.y*blockDim.y + threadIdx.y;
        int col = blockIdx.x*blockDim.x + threadIdx.x;
    
        if (row < n && col < p) {
            float res = 0.0;
            for (int i = 0; i < m; i++) res += a[row*m+i]*b[i*p+col];
            c[row*p+col] = res;
        }
    }
    ```
    <br/><br/>
Since each thread in warp is responsible for calculating each element of matrix c laid out in row-major order, thus threads at indices (x, y) and (x, y+1) calculates `c[x][y]` and `c[x][y+1]` respectively. Thus, access to matrix c is coalesced.<br/><br/>
Also, for consecutive threads at indices (x, y) and (x, y+1) reads the same elements from row x of matrix a and thus uses the same burst from the global memory and thus are coalesced. Consecutive threads at the edges such as (x, y+m-1) and (x+1, 0) reads 2 different rows of a with a stride of m (column width) and thus access is not coalesced in this case.<br/><br/>
For matrix b, the threads at indices (x, y) and (x, y+1) reads consecutive columns y and y+1 and are coalesced.<br/><br/>
But if instead of the multiplication `c=a.b`, it was transpose of b i.e. `c=a.bT`, then consecutive thread access to elements of b are not coalesced and are strided by size of m and thus would perform worse than `c=a.b`.<br/><br/>
    ```cpp
    __global__ 
    void cuda_mul_bt(float *a, float *b, float *c, int n, int m, int p) {
        int row = blockIdx.y*blockDim.y + threadIdx.y;
        int col = blockIdx.x*blockDim.x + threadIdx.x;
    
        if (row < n && col < p) {
            float res = 0.0;
            for (int i = 0; i < m; i++) res += a[row*m+i]*b[col*m+i];
            c[row*p+col] = res;
        }
    }
    ```
    <br/><br/>
One possible solution to speed-up access to non-consecutive memory locations is to read the data from the global memory to shared memory in coalesced manner and then read from shared memory in un-coalesced manner since shared memory is 100x faster than global memory. In our matrix multiplication using tiling example from previous part highlights this optimization.
[High Bandwidth Memory](https://en.wikipedia.org/wiki/High_Bandwidth_Memory)

[Global Memory Coalescing](https://giahuy04.medium.com/global-memory-coalescing-37a6f9d7e314)

[Memory Coalescing Techniques](https://homepages.math.uic.edu/~jan/mcs572/memory_coalescing.pdf)

[Memory Access Coalescing](https://cse.iitkgp.ac.in/~soumya/hp3/slides/mem-coalesce.pdf)

[GPU Performance](https://engineering.purdue.edu/~smidkiff/ece563/NVidiaGPUTeachingToolkit/Mod6/Lecture-6-2-memory-coalescing.pdf)

[Access Global Memory Efficiently in CUDA C/C++ Kernels](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)

[Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)

3. **Thread Coarsening**<br/><br/>

4. **Minimize Control Divergence in Warps**<br/><br/>

5. **Convolution Kernel**<br/><br/>

6. **Stencils**<br/><br/>

7. **Parallel histogram, reduction and atomic operations**<br/><br/>

8. **Prefix Sums**<br/><br/>

