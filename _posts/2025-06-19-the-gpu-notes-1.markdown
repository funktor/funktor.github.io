---
layout: post
title:  "The GPU Notes - Part 1"
date:   2025-06-25 18:50:11 +0530
categories: software-engineering
---

My notes on learning and using GPU and CUDA for Machine Learning, Deep Learning and Software Engineering problems. Most of the content has been inspired from the book "Programming Massively Parallel Processors-4th Edition".

1. **GPU - TFLOPS on Steroids**<br/><br/>
FLOPS - Floating Point Operations Per Seconds. 1 TFLOPS = 10^12 FLOPS. FLOPS is the number of floating point operations that can be done per second.<br/><br/>
Intel i9 processor with 24 CPU cores (latest as of writing this) can reach a peak of 1.3 TFLOPS for single precision i.e. 32-bit floats (or FP32).<br/><br/>
Compare this to the H100 GPU which has 16896 CUDA Cores and 640 Tensor Cores (will come later to this) and has limit of 989 TFLOPS for FP32 and 67 TFLOPS for FP64 (double-precision floats also known as 'double' in C).<br/><br/>
Latest GPU models also support half precision floats i.e. FP16 with higher TFLOPS (1979 TFLOPS) and FP8 with 3958 TFLOPS. FP8 and FP16 are used in deep learning for mixed precision training (revisited later).<br/><br/>
For understanding floating point representations, refer to one of my earlier posts on floating point compression:<br/><br/>
[Floating Point Compression](https://funktor.github.io/software-engineering/2025/02/12/time-series-compression.html)
<br/><br/>

2. **Smaller but more number of cores**<br/><br/>
Most commercially available CPUs have 4 to 24 cores and have larger L2 and L3 cache sizes. CPUs also have larger area for control units managing branch prediction etc. Each core also have its own L1 cache (both data and instruction). CPUs also have fewer number of channels to the DRAM i.e. the main memory. CPU cores are optimized for low latency whereas GPU cores are optimized for high throughput.<br/><br/>
To achieve low latency, CPU needs more number of registers or flip-flops which requires more power and thus one cannot have too many cores inside a CPU. CPUs are good for low latency operations on sequential programs for e.g. computing the first 1 million fibonacci numbers.<br/><br/>
Large cache size also allows significant re-use.<br/><br/>
[Nice write up on CPU architecture](https://www.redhat.com/en/blog/cpu-components-functionality)<br/><br/>
GPUs on the other hand have smaller cores but more number of cores. For example e.g. H100 have 14592 CUDA cores in addition to 640 Tensor cores. They also have smaller caches and smaller control units and more number of channels to the DRAM. The goal is high throughput for parallel computations.<br/><br/>
![CPU vs GPU](/docs/assets/cpu_gpu.png)
Smaller cache sizes may force certain applications to be memory bound i.e. most time taken goes into fetching data from DRAM into cache or register. To overcome this drawback more number of smaller caches and more channels enables more number of threads to fetch data in parallel. Thus GPUs have high memory thorughput as compared to CPU.<br/><br/>

3. **CUDA Matrix Multiplication**<br/><br/>
Basic 2D Matrix multiplication using CUDA C++.<br/><br/>
    ```cpp
    #include <stdio.h>
    #include <iostream>
    #include <math.h>
    #include <assert.h>
    #include <omp.h>
    #include <cuda.h>
    #include <cuda_runtime.h>
    
    void generate_data(float *x, int n, int m) {
        static std::random_device dev;
        static std::mt19937 rng(dev());
    
        std::uniform_real_distribution<float> dist(0.0, 1.0);
        for (int i = 0; i < n*m; i++) x[i] = dist(rng);
    }
    
    // Matrix multiplication on GPU device
    __global__ 
    void cuda_mul(float *a, float *b, float *c, int n, int m, int p) {
        // In CUDA, the ordering of dimensions are reversed i.e. a matrix of dim (N, M, P) will be represented as (P, M, N) in CUDA
        int row = blockIdx.y*blockDim.y + threadIdx.y;
        int col = blockIdx.x*blockDim.x + threadIdx.x;
    
        if (row < n && col < p) {
            float res = 0.0;
            for (int i = 0; i < m; i++) res += a[row*m+i]*b[i*p+col];
            c[row*p+col] = res;
        }
    }
    
    // Matrix multiplication on CPU
    void mat_mul(float *a, float *b, float *c, int n, int m, int p) {
        for (int i = 0; i < n*p; i++) c[i] = 0.0;
    
        omp_set_num_threads(8);
        #pragma omp parallel for shared(a, b, c)
        for(int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                for (int k = 0; k < p; k++) c[i*p+k] += a[i*m+j]*b[j*p+k];
            }
        }
    }
    
    int main(){
        int n = 2048;
        int m = 2048;
        int p = 2048;
    
        float *a, *b, *c, *d;
    
        size_t size_a = sizeof(float)*n*m;
        size_t size_b = sizeof(float)*m*p;
        size_t size_c = sizeof(float)*n*p;
    
        // a, b and c are defined for both CPU and GPU. Thus they can be accessed from both host code and device code
        cudaMallocManaged(&a, size_a);
        cudaMallocManaged(&b, size_b);
        cudaMallocManaged(&c, size_c);
    
        generate_data(a, n, m);
        generate_data(b, m, p);
    
        auto start = std::chrono::high_resolution_clock::now();
    
        // Launch grid and blocks. Each block is 3d and can have maximum of 1024 threads across all dimensions.
        // Each block has 32 threads in x-direction and 32 in y-direction.
        // Number of blocks in x direction = #columns in out matrix/#threads in x-direction
        // In CUDA, the ordering of dimensions are reversed i.e. a matrix of dim (N, M, P) will be represented as (P, M, N) in CUDA
    
        dim3 bd(32, 32, 1);
        dim3 gd(ceil(p/32.0), ceil(n/32.0), 1);
    
        // Launch kernel
        cuda_mul<<<gd, bd>>>(a, b, c, n, m, p);
    
        // Synchronize the host and device memory before accessing output
        cudaDeviceSynchronize();
    
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    
        std::cout << "CUDA Duration = " << duration.count() << " ms" << std::endl;
    
        // Compare with multi-threaded matrix multiplication on CPU
        start = std::chrono::high_resolution_clock::now();
        d = (float*)malloc(size_c);
        mat_mul(a, b, d, n, m, p);
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    
        std::cout << "Standard Duration = " << duration.count() << " ms" << std::endl;
    
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
        free(d);
    }
    ```
    <br/><br/>

4. **Streaming Multiprocessors and Warps**<br/><br/>
The GPU is arranged as an array of streaming multiprocessors (SM) where each SM contains several streaming processors or CUDA cores, has its own shared memory and control unit. H100 comes with 132 SMs where each SM contains 128 Cores thus totalling 16896 CUDA Cores.<br/><br/>
When we launch a grid of threads arranged in a block, the 3d block is linearized in row-major order and distributed across SMs. All threads in a block are assigned to the same SM. Multiple blocks can be assigned to the same SM. All active threads in a SM are run concurrently. H100 supports a maximum of 2048 active threads (that can run concurrently) per SM.<br/><br/>
Since all threads in a block run concurrently on the same SM, they would have similar running times. CUDA supports syncing threads using the _syncthreads() method but it is only applicable to threads in same block for the same reason that threads in same block have similar running times.<br/><br/>
Thus if the total number of threads in an application is 2^22 and each block has 1024 i.e 2^10 threads, then there are a total of 2^12 blocks of threads. Since each block has 1024 threads and each SM can support 2048 threads, thus each SM can accomodate 2 blocks at a time. Thus, the 1st 2 blocks are assigned to SM0, the next 2 blocks to SM1 and so on. The 132 SMs can accomodate 264 blocks at a time. Thus, the 1st 264 blocks runs concurrently and the remaining blocks are queued. Whenever any one block is completed, one of the queued block is assigned to the SM having capacity.<br/><br/>
Note that each SM has only 128 CUDA cores in H100 but can accomodate a maximum of 2048 threads.<br/><br/>
The more number of threads assigned per SM is in order to hide latency. If only 128 threads are assigned to a SM, not all threads will be working at all times. Some threads might be busy fetching data from memory and during that time the thread will remain idle. This allows some other thread in the queue to start processing.<br/><br/>
GPUs have smaller control units. Control units fetches instruction into the Instruction Register (IR). The instructions are then parsed. If multiple threads are working on different parts of a program, the amount of instructions to fetch during each instruction cycle will be quite large thus requiring larger control units with larger power requirements.<br/><br/>
In order to trade-off smaller control units for more number of cores, blocks of threads are partitioned into warps of 32 threads and the 128 cores in SM is partitioned into processing blocks. If each processing block is 16 cores, then there would be 128/16=8 processing blocks in each SM. Threads in the same warp are assigned to the same processing block.<br/><br/>
The control unit fetches instruction per warp instead of per thread thus reducing the number of instructions to fetch.<br/><br/>
The same instruction is applied to all threads in a SIMD or SIMT (Single Instruction Multiple Thread) fashion. The lane size for SIMD is 32 here.<br/><br/>
Since all threads in a warp follow same instruction, a CUDA program having an if-else statement which is dependent on thread index i.e. some threads will execute if statement whereas other threads will execute else statment, will be run twice by all the threads in the warp. First time all the threads will run the if block and next time will run the else block. Threads that should not run the if block will be deactivated during 1st run and similarly threads that should not run the else block will be deactivated during 2nd run.<br/><br/>
Control divergence happens here:<br/><br/>
    ```cpp
    __global__
    void sq_or_cube(float *a, float *out, int n) {
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        if (idx < n) {
            if (idx % 2 == 0) out[idx] = a[idx]*a[idx];
            else out[idx] = a[idx]*a[idx]*a[idx];
        }
    }
    ```
    <br/><br/>
Unlike context switching in CPU threads, threads and warps in GPU have low overhead for context switching. As mentioned above, a SM is assigned more threads and warps than the number of available cores because most often warps will sit idle and during that time other warps can run.<br/><br/>

6. **DRAM vs HBM**
