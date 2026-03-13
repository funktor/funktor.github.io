---
layout: post
title:  "CUDA GEMMS"
date:   2026-03-12 18:50:11 +0530
categories: software-engineering
---
Matrix multiplication is the building block of Deep Neural Networks which in turn are the building blocks of all AI models and applications. In order to scale AI models to billions of parameters, one must thus scale matrix multiplication. Given matrices A and B of shapes `MxK` and `KxM`, the dot product `C=A.B` is of shape `MxN`. This is what 90% of neural networks do. Time complexity of computing `C=A.B` is `O(MxNxK)` or if all dimensions are same then it is `O(M^3)`. Algorithmically, one cannot improve the time complexity much although some algorithms exists such as `Strassen's Algorithm` which does matrix multiplication in `O(M^2.8)` operations. Even the advanced algorithms today cannot achieve better than `O(M^2.371)`.<br/><br/>
So algorithmically one cannot improve the run-time performance too much. The other strategy is to use massive parallelization.<br/><br/>
On CPU, one can use multiple threads to calculate multiple output elements of C in parallel. For e.g. multithreading using `openmp` in C++ improves the run-time of multiplying 2 matrices of shapes `1024x1024` from `2688 ms` to `555 ms` using `8 threads` i.e. an improvement of around 5x.<br/><br/>
```cpp
void gemm_cpu(
    const float *a, 
    const float *b, 
    float *c, 
    const float alpha,
    const float beta,
    const unsigned int m, 
    const unsigned int n, 
    const unsigned int k
) {
    omp_set_num_threads(8);
    #pragma omp parallel for shared(a, b, c)
    for(auto i = 0; i < m; i++) {
        for (auto j = 0; j < n; j++) {
            float r = 0.0f;
            for (auto q = 0; q < k; q++) r += a[i*k+q]*b[q*n+j];
            c[i*n+j] = alpha*r + beta*c[i*n+j];
        }
    }
}
```
<br/><br/>
The above function computes the `GEMM` (General Matrix Multiply) where `D=alpha*A.B + beta*C`. For standard matrix multiplication A.B we can consider `alpha=1.0` and `beta=0.0`.<br/><br/>
CPU is limited by the number of threads because CPUs are optimized for lowering the latency of a single process instead of solving problems in parallel. Most commercially available CPUs have at-most `64 cores`. On the other hand modern GPUs have thousands of cores or threads to perform GEMM in parallel (`SIMD` or `SIMT` Single Instruction Multiple Data/Threads) and that would be the topic of this post. We will try to optimize GEMM on GPUs by leveraging different CUDA kernel optimization strategies.<br/><br/>
Before we begin exploring kernels, one should keep in mind that not all GPU architectures are built same and the same kernel A that performs better than kernel B on a GPU arch X, may perform worse than kernel B on another GPU arch Y. Importantly you should write your kernels keeping in mind the GPU architecture of your compute nodes or pods.<br/><br/>
All of the kernels I am going to show here are written on `L4 GPUs` and so the performance numbers are w.r.t. the L4 GPUs only. The numbers might change drastically if you run the same kernel on say `H100` or `A100` or `RTX`.<br/><br/>
All the codes are available at my [github repository](https://github.com/funktor/gemm).

## Kernel 1 - Standard CUDA
```cpp
__global__
void gemm_fp32_cuda(
    const float *a_fp32, 
    const float *b_fp32, 
    float *c_fp32, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float res = 0.0f;
        for (int i = 0; i < k; i++) res += a_fp32[row*k+i]*b_fp32[i*n+col];
        c_fp32[row*n+col] = alpha*res + beta*c_fp32[row*n+col];
    }
}

float *c_gpu_fp32_ccores;
cudaErrCheck(cudaMallocManaged(&c_gpu_fp32_ccores, m * n * sizeof(float)));

for (auto i = 0; i < m*n; i++) c_gpu_fp32_ccores[i] = 0.0f;

dim3 bd(32, 32, 1);
dim3 gd((n+31)/32, (m+31)/32, 1);

gemm_fp32_cuda<<<gd, bd>>>(a_fp32, b_fp32, c_gpu_fp32_ccores, 1.0, 0.0, m, n, k);
cudaDeviceSynchronize();
cudaErrCheck(cudaFree(c_gpu_fp32_ccores));
```
<br/><br/>
In CUDA, each block of threads can have at-most `1024 threads`. It is upto you how you want to distribute the threads across multiple dimensions. For e.g. in the above kernel each block is 2D and thus each dimension has 32 threads totalling `32*32=1024` threads. Note that in the dimensions defined above, the 1st dimension (x) corresponds to number of columns and 2nd dimension (y) corresponds to number of rows.
<br/><br/>
Total number of blocks required to populate the entire output matrix along the column dimension is `ceil(n/32)` or `(n+31)/32` where n is the number of columns in output matrix and along the row dimension is `(m+31)/32`. In the above kernel, each thread is responsible for computing one element of the output matrix.
<br/><br/>
Time taken to multiply two 4096x4096 matrices is around `40.4367 ms`.
<br/><br/>
To compile all the CUDA kernels on `L4 GPU`, I use the following command from the terminal. Make sure you have the necessary libraries such as TBB or OpenMP installed.<br/><br/>
```
nvcc -rdc=true *.cu -Xcompiler -fopenmp -o my_gemm -O3 -Xcompiler -O3 --gpu-code=sm_89 -arch=compute_89 -lcublas -lcurand -ltbb
```
<br/><br/>

## Kernel 2 - 1D Tiling + Thread Coarsening
```cpp
#define COARSE_FACTOR 4
#define TILE_WIDTH 32

__global__
void gemm_fp32_cuda_tiled(
    const float *a_fp32, 
    const float *b_fp32, 
    float *c_fp32, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    __shared__ float Mds[TILE_WIDTH*TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH*TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by*TILE_WIDTH + ty;
    int col_start = bx*TILE_WIDTH*COARSE_FACTOR + tx;

    float Pval[COARSE_FACTOR];
    for (int r = 0; r < COARSE_FACTOR; r++) Pval[r] = 0.0f;

    for (int ph = 0; ph < k; ph += TILE_WIDTH) {
        if (row < m && (ph + tx) < k) Mds[ty*TILE_WIDTH+tx] = a_fp32[row*k + ph + tx];
        else Mds[ty*TILE_WIDTH+tx] = 0.0f;

        for (int r = 0; r < COARSE_FACTOR; r++) {
            int col = col_start + r*TILE_WIDTH;

            if ((ph + ty) < k && col < n) Nds[ty*TILE_WIDTH+tx] = b_fp32[(ph + ty)*n + col];
            else Nds[ty*TILE_WIDTH+tx] = 0.0f;
            __syncthreads();

            for (int i = 0; i < TILE_WIDTH; i++) Pval[r] += Mds[ty*TILE_WIDTH+i]*Nds[i*TILE_WIDTH+tx];
            __syncthreads();
        }
    }

    for (int r = 0; r < COARSE_FACTOR; r++) {
        int col = col_start + r*TILE_WIDTH;
        if (row < m && col < n) c_fp32[row*n+col] = alpha*Pval[r] + beta*c_fp32[row*n+col];
    }
}

float *c_gpu_fp32_tiled;
cudaErrCheck(cudaMallocManaged(&c_gpu_fp32_tiled, m * n * sizeof(float)));

for (auto i = 0; i < m*n; i++) c_gpu_fp32_tiled[i] = 0.0f;

dim3 bd1(32, 32, 1);
dim3 gd1((n+32*COARSE_FACTOR-1)/(32*COARSE_FACTOR), (m+31)/32, 1);

gemm_fp32_cuda_tiled<<<gd1, bd1>>>(a_fp32, b_fp32, c_gpu_fp32_tiled, 1.0, 0.0, m, n, k);
cudaDeviceSynchronize();
cudaErrCheck(cudaFree(c_gpu_fp32_tiled));
```
<br/><br/>
As before, we have a blocks of threads where each block has 32 threads per row and there are 32 such rows totalling 1024 threads per block. But now each thread is responsible for computing 4 elements (`COARSE_FACTOR=4`). Thus each block now computes the output elements equivalent to 4 blocks as in the previous kernel. Thus number of blocks required will reduce along the column (x) dimension to `(n+127)/128`.
<br/><br/>
![1D Tile](/docs/assets/1d_tile.png)
<br/><br/>
Also, another important technique used to optimize the kernel is Tiling. In Tiling, instead of each thread reading a full row `i` of matrix A and a full column `j` of matrix B from the global memory to compute `C[i,j]`, each thread now reads `TILE_WIDTH=32` elements from row `i` in A and `TILE_WIDTH=32` elements from column `j` in B at a time, loads them from global memory to shared memory and computes the partial sum for `C[i,j]`. Once a tile from A and B has been read and partial sum is computed, the next tile from A and B is read by the thread to get the next 32 elements of row `i` in A and next 32 elements of column `j` in B and the process is repeated. To understand why this works:<br/><br/>
```
C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j] + ... + A[i,k]*B[k,j]

Assuming each tile is of size 32 and k=4096, then there would be 128 tiles.
C_tile_0[i,j]   = A[i,0]*B[0,j]       + A[i,1]*B[1,j]       +  ...  + A[i,31]*B[31,j]
C_tile_1[i,j]   = A[i,32]*B[32,j]     + A[i,33]*B[33,j]     +  ...  + A[i,63]*B[63,j]
C_tile_2[i,j]   = A[i,64]*B[64,j]     + A[i,65]*B[65,j]     +  ...  + A[i,95]*B[95,j]
....
C_tile_127[i,j] = A[i,4064]*B[4064,j] + A[i,4065]*B[4065,j] + ... + A[i,4095]*B[4095,j]

Then we have,
C[i,j] = C_tile_0[i,j] + C_tile_1[i,j] + ... + C_tile_127[i,j]
```
<br/><br/>
The reason for Tiling is to reduce the latency in fetching data from the global memory of the GPU. The process of Tiling is similar to caching where we pull the frequenctly accessed elements from RAM to L1/L2/L3 Cache.
<br/><br/>
Similar to memory hierarchy in CPU : Register > L1 > L2 > L3 > RAM, GPU has its own memory hierarchy which looks something like Register > Shared Memory > Global Memory. Similar to CPU, the higher performance memory are limited in size as compared to the lower performance memory i.e. shared memory is much smaller (`48KB` per block and `163KB` per SM) as compared to global memory (around `24GB`).<br/><br/>
Shared memory is accessible by all threads in the block. Thus if thread `T1` computes the element `C[i,j]` and thread `T2` computed `C[i,j+1]`, then note that we only need to read the row `i` from global memory to shared memory once for all columns corresponding to row `i` in the output matrix C. But since shared memory size is limited we resort to use tiling i.e. read 32 elements from row `i` at a time.<br/><br/>
In the above kernel, each thread computes 4 elements `C[i,j]`, `C[i,j+32]`, `C[i,j+64]` and `C[i,j+96]`. This is because consecutive threads compute consecutive elements. Threads T0 to T31 computes `C[i,j]` to `C[i,j+31]`. Then the same threads computes `C[i,j+32]` to `C[i,j+63]` and so on. A group of 32 consecutive threads is called a `Warp` and a Warp is scheduled to run simultaneously, thus threads T0 to T31 is accessing consecutive memory locations and thus require a single GPU cycle to read all 32 consecutive elements.
<br/><br/>
In matrix multiplication, multiplying two k length vectors requires 2*k operations (k multiplications + k additions). Multiplying two matrices of size `TILE_WIDTH*TILE_WIDTH` requires `2*TILE_WIDTH^3` operations. The number of bytes transferred in the above kernel from global memory is `8*TILE_WIDTH^2` bytes for `Mds` and `8*TILE_WIDTH^2` bytes for `Nds`. Thus the ratio of operations per byte transferred is `TILE_WIDTH/8` which is 4 i.e. for every byte transferred from global memory, we are doing 4 operations.
<br/><br/>
Without tiling, to compute each element of output matrix C, we required `2*k` operations (k columns) and transferred `16*k` bytes from global memory in total. Thus the number of operations per byte transferred was `0.25`. Thus with tiling we have improved the ratio by `TILE_WIDTH` times. Note that the amount of shared memory usage per block is currently `16*TILE_WIDTH^2` bytes or 16KB. If we double the TILE_WIDTH, the shared memory usage will become 4 times i.e. 64KB which exceeds 48KB available per block. Thus, we cannot increase shared memory arrays arbitrarily to improve the ratio of number of operations per byte transferred from global memory.
<br/><br/>
This number will be useful when we look at the next kernel.
<br/><br/>

## Kernel 3 - 2D Tiling + Thread Coarsening
```cpp
__global__
void gemm_fp32_cuda_tiled_2D(
    const float *a_fp32, 
    const float *b_fp32, 
    float *c_fp32, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    __shared__ float Mds[TILE_WIDTH*TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH*TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_start = by*TILE_WIDTH*COARSE_FACTOR_2D + ty;
    int col_start = bx*TILE_WIDTH*COARSE_FACTOR_2D + tx;

    float Pval[COARSE_FACTOR_2D*COARSE_FACTOR_2D];
    for (int r = 0; r < COARSE_FACTOR_2D*COARSE_FACTOR_2D; r++) Pval[r] = 0.0f;

    for (int ph = 0; ph < k; ph += TILE_WIDTH) {
        for (int r = 0; r < COARSE_FACTOR_2D; r++) {
            int row = row_start + r*TILE_WIDTH;

            if (row < m && ph + tx < k) Mds[ty*TILE_WIDTH+tx] = a_fp32[row*k + ph + tx];
            else Mds[ty*TILE_WIDTH+tx] = 0.0f;

            for (int c = 0; c < COARSE_FACTOR_2D; c++) {
                int col = col_start + c*TILE_WIDTH;

                if (ph + ty < k && col < n) Nds[ty*TILE_WIDTH+tx] = b_fp32[(ph + ty)*n + col];
                else Nds[ty*TILE_WIDTH+tx] = 0.0f;
                __syncthreads();

                for (int i = 0; i < TILE_WIDTH; i++) Pval[r*COARSE_FACTOR_2D + c] += Mds[ty*TILE_WIDTH+i]*Nds[i*TILE_WIDTH+tx];
                __syncthreads();
            }
        }
    }

    for (int r = 0; r < COARSE_FACTOR_2D; r++) {
        int row = row_start + r*TILE_WIDTH;
        for (int c = 0; c < COARSE_FACTOR_2D; c++) {
            int col = col_start + c*TILE_WIDTH;
            if (row < m && col < n) c_fp32[row*n+col] = alpha * Pval[r*COARSE_FACTOR_2D + c] + beta * c_fp32[row*n+col];
        }
    }
}
```
<br/><br/>
Instead of each thread computing 4 elements of the same row in the output matrix, in the above kernel each thread now computes 4x4 elements comprising of 4 rows and 4 columns of the output matrix.
