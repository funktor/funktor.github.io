---
layout: post
title:  "CUDA GEMMS"
date:   2026-03-20 18:50:11 +0530
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
Also, for the same architecture matrices of different dimensions shows different relative performance for different kernels. For e.g. say on L4 GPU, if kernel 1 performs better than kernel 2 on 1024x1024 matrices it does not imply that kernel 1 will still perform better than kernel 2 on 4096x4096 matrices.<br/><br/>
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
In matrix multiplication, multiplying two k length vectors requires 2*k operations (k multiplications + k additions). Multiplying two matrices of size `TILE_WIDTH*TILE_WIDTH` requires `2*TILE_WIDTH^3` operations. The number of bytes transferred in the above kernel from global memory per `ph` is `8*TILE_WIDTH^2` bytes for `Mds` and `8*COARSE_FACTOR*TILE_WIDTH^2` bytes for `Nds`. Thus the ratio of operations per byte transferred is `(2*COARSE_FACTOR*TILE_WIDTH^3)/(8*TILE_WIDTH^2 + 8*COARSE_FACTOR*TILE_WIDTH^2)` which is 6.4 i.e. for every byte transferred from global memory, we are doing 6.4 operations.
<br/><br/>
Without tiling, to compute each element of output matrix C, we required `2*k` operations (k columns) and transferred `16*k` bytes from global memory in total. Thus the number of operations per byte transferred was `0.25`. Thus with tiling we have improved the ratio by `constant*TILE_WIDTH` times. Note that the amount of shared memory usage per block is currently `16*TILE_WIDTH^2` bytes or 16KB. If we double the TILE_WIDTH, the shared memory usage will become 4 times i.e. 64KB which exceeds 48KB available per block. Thus, we cannot increase shared memory arrays arbitrarily to improve the ratio of number of operations per byte transferred from global memory.
<br/><br/>
Another important metric to look out for is that for a thread to compute a submatrix of size `1x4` for the output matrix C, it needs to load a sub-matrix of shape `1xk` from A and a sub-matrix of shape `kx4` from B. Thus in order to compute 4 output elements, the kernel needs to load `k + 4k = 5k` elements in total from global memory. The number of outputs per element loaded from global memory is `4/5k`. This number will be useful when we look at the next kernel.
<br/><br/>
Time taken to multiply two 4096x4096 matrices is around `27.349 ms`.
<br/><br/>

## Kernel 3 - 2D Tiling + Thread Coarsening
```cpp
#define COARSE_FACTOR_2D 4
#define TILE_WIDTH 32
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
![2D Tile](/docs/assets/2d_tile.png)
<br/><br/>
Instead of each thread computing 4 elements of the same row in the output matrix, in the above kernel each thread now computes `4x4` elements comprising of 4 rows and 4 columns of the output matrix. Apart from these most of the code is similar to the 1D Tiling kernel above. Let's compute the number of operations per byte transferred with 2D tiling kernel.
<br/><br/>
Total number of operations in 4x4 blocks each with 32x32 elements = `(2*COARSE_FACTOR_2D*COARSE_FACTOR_2D*TILE_WIDTH^3)`
<br/><br/>
Total number of bytes transferred = `(8*COARSE_FACTOR_2D*TILE_WIDTH^2 + 8*COARSE_FACTOR_2D*COARSE_FACTOR_2D*TILE_WIDTH^2)` bytes.
<br/><br/>
Ratio of number of operations per byte transferred = `(COARSE_FACTOR_2D*TILE_WIDTH)/(4*(1 + COARSE_FACTOR_2D)) = 6.4`
<br/><br/>
Thus, the ratio of number of operations per byte transferred remains same as the previous kernel.
<br/><br/>
But note that in order to compute 4x4=16 elements of C, the kernel loads `4xk` elements from A and `kx4` elements from B, thus totalling 8k elements. The number of outputs per element loaded from global memory is `16/8k=2/k`. Compare this to the previous kernel where number of outputs per element loaded from global memory was `0.8/k`. 
<br/><br/>
We can see that this kernel is more efficient because it computes more element for the same number of inputs loaded from global memory. This metric is useful because apart from shared memory, GPU also has L1/L2 cache similar to CPU and often elements fetched from global memory are cached for reuse. 
<br/><br/>
Time taken to multiply two 4096x4096 matrices is around `23.7353 ms`.
<br/><br/>

## Kernel 4 - 2D Tiling + Vectorization
```cpp
#define TILE_WIDTH 32
#define COARSE_FACTOR_2D 4

__global__
void gemm_fp32_cuda_tiled_2D_vectorize(
    float *a_fp32, 
    float *b_fp32, 
    float *c_fp32, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    __shared__ alignas(16) float Mds[TILE_WIDTH*TILE_WIDTH];
    __shared__ alignas(16) float Nds[TILE_WIDTH*TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_start = by*TILE_WIDTH*COARSE_FACTOR_2D + ty;
    int col_start = bx*TILE_WIDTH*COARSE_FACTOR_2D + tx*4;

    float Pval[COARSE_FACTOR_2D*COARSE_FACTOR_2D*4];
    for (int r = 0; r < COARSE_FACTOR_2D*COARSE_FACTOR_2D*4; r++) Pval[r] = 0.0f;

    for (int ph = 0; ph < k; ph += TILE_WIDTH) {
        for (int r = 0; r < COARSE_FACTOR_2D; r++) {
            int row = row_start + r*TILE_WIDTH;
            reinterpret_cast<float4 *>(&Mds[ty*TILE_WIDTH + tx*4])[0] = reinterpret_cast<float4 *>(&a_fp32[row*k + ph + tx*4])[0];

            for (int c = 0; c < COARSE_FACTOR_2D; c++) {
                int col = col_start + c*TILE_WIDTH;

                reinterpret_cast<float4 *>(&Nds[ty*TILE_WIDTH + tx*4])[0] = reinterpret_cast<float4 *>(&b_fp32[(ph + ty)*n + col])[0];
                __syncthreads();

                for (int i = 0; i < TILE_WIDTH; i++) {
                    Pval[r*COARSE_FACTOR_2D*4 + 4*c + 0] += Mds[ty*TILE_WIDTH+i]*Nds[i*TILE_WIDTH+tx*4+0];
                    Pval[r*COARSE_FACTOR_2D*4 + 4*c + 1] += Mds[ty*TILE_WIDTH+i]*Nds[i*TILE_WIDTH+tx*4+1];
                    Pval[r*COARSE_FACTOR_2D*4 + 4*c + 2] += Mds[ty*TILE_WIDTH+i]*Nds[i*TILE_WIDTH+tx*4+2];
                    Pval[r*COARSE_FACTOR_2D*4 + 4*c + 3] += Mds[ty*TILE_WIDTH+i]*Nds[i*TILE_WIDTH+tx*4+3];
                }
                __syncthreads();
            }
        }
    }

    for (int r = 0; r < COARSE_FACTOR_2D; r++) {
        int row = row_start + r*TILE_WIDTH;
        for (int c = 0; c < COARSE_FACTOR_2D; c++) {
            int col = col_start + c*TILE_WIDTH;

            c_fp32[row*n + col + 0] = alpha*Pval[r*COARSE_FACTOR_2D*4 + 4*c + 0] + beta*c_fp32[row*n + col + 0];
            c_fp32[row*n + col + 1] = alpha*Pval[r*COARSE_FACTOR_2D*4 + 4*c + 1] + beta*c_fp32[row*n + col + 1];
            c_fp32[row*n + col + 2] = alpha*Pval[r*COARSE_FACTOR_2D*4 + 4*c + 2] + beta*c_fp32[row*n + col + 2];
            c_fp32[row*n + col + 3] = alpha*Pval[r*COARSE_FACTOR_2D*4 + 4*c + 3] + beta*c_fp32[row*n + col + 3];
        }
    }
}

float *c_gpu_fp32_tiled_2d_vec;
cudaErrCheck(cudaMallocManaged(&c_gpu_fp32_tiled_2d_vec, m * n * sizeof(float)));

for (auto i = 0; i < m*n; i++) c_gpu_fp32_tiled_2d_vec[i] = 0.0f;

dim3 bd3(8, 32, 1);
dim3 gd3((n+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), (m+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), 1);

gemm_fp32_cuda_tiled_2D_vectorize<<<gd3, bd3>>>(a_fp32, b_fp32, c_gpu_fp32_tiled_2d_vec, 1.0, 0.0, m, n, k);
cudaDeviceSynchronize();
cudaErrCheck(cudaFree(c_gpu_fp32_tiled_2d_vec));
```
<br/><br/>
![2D Tile Vec](/docs/assets/2d_tile_vec.png)
<br/><br/>
In the above kernel we are using vectorization with `float4` data type i.e. instead of a thread loading a 32-bit `float` from global memory, a thread loads a 128-bit `float4` or 4 consecutive 32-bit addresses. Instead of 4 instructions, now we have to issue only one instruction.
<br/><br/>
```
reinterpret_cast<float4 *>(&Mds[ty*TILE_WIDTH + tx*4])[0] = reinterpret_cast<float4 *>(&a_fp32[row*k + ph + tx*4])[0];
reinterpret_cast<float4 *>(&Nds[ty*TILE_WIDTH + tx*4])[0] = reinterpret_cast<float4 *>(&b_fp32[(ph + ty)*n + col])[0];
```
<br/><br/>
In order to work with `float4` vectorization the x-dimension for each block of thread is reduced by a factor of 4 from the previous kernel i.e. instead of each thread computing `4x4=16` elements of the output matrix, now each each thread computes `4x4x4=64` elements.
<br/><br/>
Note that having many threads per block is not always good because GPU has what is known as the `occupancy problem`. Basically the resources such as number of registers, number of warps that can be simulataneously scheduled, maximum shared memory per block, maximum registers per thread are limited. When number of threads are higher, it can lead to smaller concurrency due to exceeding number of registers per thread or shared mempry size etc. So often you would see that lesser number of threads perform better than more number of threads.<br/><br/>
Note that the shared memory addresses needs to be 16 byte or 128-bit aligned using `alignas(16)`.
<br/><br/>
Time taken to multiply two 4096x4096 matrices is around `15.1583 ms`.
<br/><br/>

## Kernel 5 - 2D Tiling + Asynchronous Pipelining N-Stage
```cpp
#define TILE_WIDTH 32
#define COARSE_FACTOR_2D 4
#define NUM_STAGES_ASYNC_PIPELINE 4

__global__
void gemm_fp32_cuda_tiled_2D_async(
    const float *a_fp32, 
    const float *b_fp32, 
    float *c_fp32, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    __shared__ alignas(16) float Mds[NUM_STAGES_ASYNC_PIPELINE][TILE_WIDTH*TILE_WIDTH];
    __shared__ alignas(16) float Nds[NUM_STAGES_ASYNC_PIPELINE][TILE_WIDTH*TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_start = by*TILE_WIDTH*COARSE_FACTOR_2D + ty;
    int col_start = bx*TILE_WIDTH*COARSE_FACTOR_2D + tx*4;

    for (int r = 0; r < COARSE_FACTOR_2D; r++) {
        int row = row_start + r*TILE_WIDTH;
        for (int c = 0; c < COARSE_FACTOR_2D; c++) {
            int col = col_start + c*TILE_WIDTH;
            
            for (int s = 0; s < NUM_STAGES_ASYNC_PIPELINE; s++) {
                pipeline.producer_acquire();
                cuda::memcpy_async(Mds[s] + ty*TILE_WIDTH + tx*4, a_fp32 + row*k + s*TILE_WIDTH + tx*4, cuda::aligned_size_t<4>(sizeof(float)*4), pipeline);
                cuda::memcpy_async(Nds[s] + ty*TILE_WIDTH + tx*4, b_fp32 + (s*TILE_WIDTH + ty)*n + col, cuda::aligned_size_t<4>(sizeof(float)*4), pipeline);
                pipeline.producer_commit();
            }

            int s = NUM_STAGES_ASYNC_PIPELINE;
            float res[4] = {0.0f};

            for (int ph = 0; ph < k; ph += TILE_WIDTH) {
                int stage = s % NUM_STAGES_ASYNC_PIPELINE;

                constexpr size_t pending_batches = NUM_STAGES_ASYNC_PIPELINE - 1;
                cuda::pipeline_consumer_wait_prior<pending_batches>(pipeline);
                __syncthreads();

                for (int i = 0; i < TILE_WIDTH; i++) {
                    res[0] += Mds[stage][ty*TILE_WIDTH+i]*Nds[stage][i*TILE_WIDTH+tx*4+0];
                    res[1] += Mds[stage][ty*TILE_WIDTH+i]*Nds[stage][i*TILE_WIDTH+tx*4+1];
                    res[2] += Mds[stage][ty*TILE_WIDTH+i]*Nds[stage][i*TILE_WIDTH+tx*4+2];
                    res[3] += Mds[stage][ty*TILE_WIDTH+i]*Nds[stage][i*TILE_WIDTH+tx*4+3];
                }

                pipeline.consumer_release();
                __syncthreads();

                pipeline.producer_acquire();
                if (s*TILE_WIDTH < k) {
                    cuda::memcpy_async(Mds[stage] + ty*TILE_WIDTH + tx*4, a_fp32 + row*k + s*TILE_WIDTH + tx*4, cuda::aligned_size_t<4>(sizeof(float)*4), pipeline);
                    cuda::memcpy_async(Nds[stage] + ty*TILE_WIDTH + tx*4, b_fp32 + (s*TILE_WIDTH + ty)*n + col, cuda::aligned_size_t<4>(sizeof(float)*4), pipeline);
                }
                pipeline.producer_commit();

                s += 1;
            }

            c_fp32[row*n+col+0] = alpha * res[0] + beta * c_fp32[row*n+col+0];
            c_fp32[row*n+col+1] = alpha * res[1] + beta * c_fp32[row*n+col+1];
            c_fp32[row*n+col+2] = alpha * res[2] + beta * c_fp32[row*n+col+2];
            c_fp32[row*n+col+3] = alpha * res[3] + beta * c_fp32[row*n+col+3];
        }
    }
}

float *c_gpu_fp32_tiled_2d_async;
cudaErrCheck(cudaMallocManaged(&c_gpu_fp32_tiled_2d_async, m * n * sizeof(float)));

for (auto i = 0; i < m*n; i++) c_gpu_fp32_tiled_2d_async[i] = 0.0f;

dim3 bd21(8, 32, 1);
dim3 gd21((n+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), (m+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), 1);

gemm_fp32_cuda_tiled_2D_async<<<gd21, bd21>>>(a_fp32, b_fp32, c_gpu_fp32_tiled_2d_async, 1.0, 0.0, m, n, k);
cudaDeviceSynchronize();
cudaErrCheck(cudaFree(c_gpu_fp32_tiled_2d_async));
```
<br/><br/>
![Async](/docs/assets/async.png)
<br/><br/>
CUDA `pipeline` is something similar to a `FIFO Queue`. There is a `producer` pushing stages to the end of the queue while the `consumer` is reading the stages off the front of the queue. 
<br/><br/>
Recall that in the 2D kernel, each thread computes `4x4x4=64` elements of the output matrix. To compute each output element, we use tiles that slide over the matrix A along the columns and over matrix B along rows. For `k=4096` and tile dimension of `32x32`, to compute each element one needs to slide over `128 tiles` in both A (horizontally) and B (vertically). In the original 2D kernel, we slide over each tile one by one.
<br/><br/>
Instead what if in the meantime the threads that are multiplying the shared memory matrices `Mds` and `Nds`, the inactive threads asynchronpusly transfer the next tile from the global memory and by the time the computation is done, the next tiles should be ready to use. Thus we can basically overlap transfer of tiles from global memory to shared memory with actual matrix multiplication computations.
<br/><br/>
In the above kernel, we define shared memory matrices of size `4x32x32`. For each output element, initally we issue asynchronous copy command from global memory to shared memory of 4 tiles (or stages). The producer pushes the `4 stages` into the pipeline. This operation is `non-blocking` and it does not use registers during the copy process (during standard copy of global to shared, first the data is read into registers and then copied from registers to shared memory).
<br/><br/>
Next we check in a for-loop if at-least 1 stage has been completed (in FIFO order the first stage to be pushed is the first stage to complete). If not the consumer waits, else pulls the completed stage off the front of the pipeline and does the matrix multiplication of the 2 shared memory matrices corresponding to 1st stage. Now if there are any pending tiles, the producer again issues a asynchronous copy command and pushes this stage to the back of the pipeline. The loop continues until there are no more tiles.
<br/><br/>
Since data transfer is a time consuming operation as compared to matrix multiplication, thus we issue multiple asynchronous copy commands at the beginning so that there is a good overlap between data transfer and actual computation.
<br/><br/>
Note that the pipeline object is local to the thread (`cuda::pipeline<cuda::thread_scope_thread>`) because we are using a single pipeline to queue all the tiles required for computing a single output element by an individual thread. Having the pipeline visiblity at the block level, one cannot use the `cuda::pipeline_consumer_wait_prior<pending_batches>(pipeline)` command because the completed stage may come from any thread's job i.e. it could correspond to any other output element.
<br/><br/>
Time taken to multiply two 4096x4096 matrices is around `14.9248 ms`.
<br/><br/>

## Kernel 6 - 2D Tiling + Asynchronous Pipelining Warp Specialization
```cpp
#define TILE_WIDTH 32
#define COARSE_FACTOR_2D 4
#define NUM_STAGES_ASYNC_PIPELINE 4

__global__
void gemm_fp32_cuda_tiled_2D_async_warp_spl(
    const float *a_fp32, 
    const float *b_fp32, 
    float *c_fp32, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    auto block = cooperative_groups::this_thread_block();
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, NUM_STAGES_ASYNC_PIPELINE> shared_state;
    cuda::pipeline<cuda::thread_scope_block> pipe = cuda::make_pipeline(block, &shared_state, 32);

    __shared__ alignas(16) float Mds[NUM_STAGES_ASYNC_PIPELINE][TILE_WIDTH*TILE_WIDTH];
    __shared__ alignas(16) float Nds[NUM_STAGES_ASYNC_PIPELINE][TILE_WIDTH*TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_start = by*TILE_WIDTH*COARSE_FACTOR_2D + ty;
    int col_start = bx*TILE_WIDTH*COARSE_FACTOR_2D + tx*4;
    int tid = block.thread_rank();
    int warp_id = tid/32;

    for (int r = 0; r < COARSE_FACTOR_2D; r++) {
        int row = row_start + r*TILE_WIDTH;
        for (int c = 0; c < COARSE_FACTOR_2D; c++) {
            int col = col_start + c*TILE_WIDTH;

            if (warp_id == 0) {
                int row_off = by*TILE_WIDTH*COARSE_FACTOR_2D + r*TILE_WIDTH + tid;
                int col_off = bx*TILE_WIDTH*COARSE_FACTOR_2D + c*TILE_WIDTH;

                for (int ph = 0; ph < k; ph += TILE_WIDTH) {
                    int stage = (ph/TILE_WIDTH) % NUM_STAGES_ASYNC_PIPELINE;
                    pipe.producer_acquire();
                    cuda::memcpy_async(Mds[stage] + tid*TILE_WIDTH, a_fp32 + row_off*k + ph, cuda::aligned_size_t<4>(sizeof(float)*32), pipe);
                    cuda::memcpy_async(Nds[stage] + tid*TILE_WIDTH, b_fp32 + (ph + tid)*n + col_off, cuda::aligned_size_t<4>(sizeof(float)*32), pipe);
                    pipe.producer_commit();
                }
            }
            else {
                auto consumer_group = cooperative_groups::tiled_partition<32>(block);
                float res[8] = {0.0f};

                for (int ph = 0; ph < k; ph += TILE_WIDTH) {
                    int stage = (ph/TILE_WIDTH) % NUM_STAGES_ASYNC_PIPELINE;
                    pipe.consumer_wait();
                    for (int row_off=ty-4; row_off < TILE_WIDTH; row_off += 28) {
                        for (int i = 0; i < TILE_WIDTH; i++) {
                            res[4*(row_off/28) + 0] += Mds[stage][row_off*TILE_WIDTH+i]*Nds[stage][i*TILE_WIDTH+tx*4+0];
                            res[4*(row_off/28) + 1] += Mds[stage][row_off*TILE_WIDTH+i]*Nds[stage][i*TILE_WIDTH+tx*4+1];
                            res[4*(row_off/28) + 2] += Mds[stage][row_off*TILE_WIDTH+i]*Nds[stage][i*TILE_WIDTH+tx*4+2];
                            res[4*(row_off/28) + 3] += Mds[stage][row_off*TILE_WIDTH+i]*Nds[stage][i*TILE_WIDTH+tx*4+3];
                        }
                    }
                    cooperative_groups::sync(consumer_group);
                    pipe.consumer_release();
                }

                for (int row_off=ty-4; row_off < TILE_WIDTH; row_off += 28) {
                    c_fp32[(row+row_off-ty)*n+col + 0] = alpha * res[4*(row_off/28) +  0] + beta * c_fp32[(row+row_off-ty)*n+col + 0];
                    c_fp32[(row+row_off-ty)*n+col + 1] = alpha * res[4*(row_off/28) +  1] + beta * c_fp32[(row+row_off-ty)*n+col + 1];
                    c_fp32[(row+row_off-ty)*n+col + 2] = alpha * res[4*(row_off/28) +  2] + beta * c_fp32[(row+row_off-ty)*n+col + 2];
                    c_fp32[(row+row_off-ty)*n+col + 3] = alpha * res[4*(row_off/28) +  3] + beta * c_fp32[(row+row_off-ty)*n+col + 3];
                }
            }
        }
    }
}

float *c_gpu_fp32_tiled_2d_async_warp_spl;
cudaErrCheck(cudaMallocManaged(&c_gpu_fp32_tiled_2d_async_warp_spl, m * n * sizeof(float)));

for (auto i = 0; i < m*n; i++) c_gpu_fp32_tiled_2d_async_warp_spl[i] = 0.0f;

dim3 bd22(8, 32, 1);
dim3 gd22((n+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), (m+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), 1);

gemm_fp32_cuda_tiled_2D_async_warp_spl<<<gd22, bd22>>>(a_fp32, b_fp32, c_gpu_fp32_tiled_2d_async_warp_spl, 1.0, 0.0, m, n, k);
cudaDeviceSynchronize();
cudaErrCheck(cudaFree(c_gpu_fp32_tiled_2d_async_warp_spl));
```
<br/><br/>
In the previous kernel, all the threads are participating for both data transfer and actual computations. In this kernel, we divide the responsibilities. Assuming a block of thread of `8x32=256` threads, it will be divided up into warps of 32 threads each. Thus, there will be `8 warps` in total per block. We can use the `1st warp (producer)` for all data transfer jobs whereas the remaining `7 warps (consumers)` would be used for actual matrix multiplcation computations.
<br/><br/>

Without warp specialization, we can face the following challenges:<br/><br/>

### Warp Divergence
Some threads in a warp might be doing matrix multiplications whereas the other threads of the same warp might be involved in data transfer from global memory. One cannot achive full `SIMD` with this kind of setting. Warp specialization aims for full SIMD and removes `warp divergence` because all threads in warp are either doing data transefr or doing matmul.
<br/><br/>

### Redundant register usage
All threads in a warp use the same number of registers. Clearly the threads involved in matmul needs to use more number of registers than the threads doing data transfer but since threads do not have a separation of responsibilities, all threads are using additional registers.
<br/><br/>

### Unnecessary __syncthreads()
Using `__syncthreads()` to sync all threads is slow. Since we do not know which threads are doing data transfer and which matmul, we need to do __syncthreads() to sync all threads in a block. This could lead to wastage of bandwidth. On the other hand, with warp specialization, since we know which warps are doing matmul, we can explicitly sync only those warps using `cooperative_groups::sync(consumer_group)`.
<br/><br/>

Note that the pipeline is shared among all threads in the block unlike the previous kernel where the pipeline had visibility at the thread scope level because with warp specialization, all consumer warps needs to synchronize.
<br/><br/>

In the above kernel it might appear that the producer warp (warp_id=0) might overwrite the stages as it loops over the k-dimension because there are only 4 stages. Note that the pipeline has been created with `shared_state` with a maximum concurrency of 4 i.e. at any given point in time the pipeline can have a maximum of 4 stages in the queue. The consumer warps are waiting on the stages. So once 1st stage has been completed for e.g. stage=0 is pushed to pipeline by the producer warp and data is transferred, it is `locked` for writing until the consumer warps read it and releases the lock.
<br/><br/>
Time taken to multiply two 4096x4096 matrices is around `21.3606 ms`.
<br/><br/>
It is surprising to see that `warp specialization` takes more time as compared to N-stage asynchronous pipeline above. It could be due to multiple reasons.
<br/><br/>

### shared_state overhead
Maintaining the `shared_state` buffer in shared memory by both the producer and consumer warps could be additional overhead as it requires frequent updates once producer pushes a stage or consumer consumes a stage. shared_state was absent in the previous kernel as the pipeline was local to each thread.
<br/><br/>

### Uneven speed of producer and consumer
It could be that the consumer warps are slow to process the computations and the producer warp is waiting to push new stage or vice versa where the producer warp is slow in fetching data from global memory while consumer warps have finished the operations and waiting for new stages in the pipeline.
<br/><br/>

## Kernel 6 - WMMA
```cpp
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ 
void gemm_wmma(
    const half *a, 
    const half *b, 
    float *c, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    int lda = k;
    int ldb = n;
    int ldc = n;

    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int i = 0; i < k; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;

        int bRow = i;
        int bCol = warpN * WMMA_N;

        if (aRow < m && aCol < k && bRow < k && bCol < n) {
            wmma::load_matrix_sync(a_frag, a + aRow * lda + aCol, lda);
            wmma::load_matrix_sync(b_frag, b + bRow * ldb + bCol, ldb);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < m && cCol < n) {
        wmma::load_matrix_sync(c_frag, c + cRow * ldc + cCol, ldc, wmma::mem_row_major);
        #pragma unroll
        for(int i=0; i < c_frag.num_elements; i++) c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        wmma::store_matrix_sync(c + cRow * ldc + cCol, c_frag, ldc, wmma::mem_row_major);
    }
}

float *c_gpu_fp32_wmma;
cudaErrCheck(cudaMallocManaged(&c_gpu_fp32_wmma, m * n * sizeof(float)));

for (auto i = 0; i < m*n; i++) c_gpu_fp32_wmma[i] = 0.0f;

dim3 bd4(128, 4, 1);
dim3 gd4((n+WMMA_N*128/32-1)/(WMMA_N*128/32), (m+WMMA_M*4-1)/(WMMA_M*4), 1);

gemm_wmma<<<gd4, bd4>>>(a_fp16, b_fp16, c_gpu_fp32_wmma, 1.0, 0.0, m, n, k);
cudaDeviceSynchronize();
cudaErrCheck(cudaFree(c_gpu_fp32_wmma));
```
<br/><br/>

Till now we have been doing GEMM operations on the CUDA cores. From this kernel onwards we are going to leverage `Tensor Cores` for doing GEMM operations. Tensor Cores offers a lot of performance advantage over `CUDA cores` with regards to GEMM operations.
<br/><br/>

### FP16/BF16, FP8 and INT8 mixed precision training
CUDA cores is only capable of doing `FP32` or 32-bit floating point operations in GEMM. Whereas Tensor Cores are capable of doing GEMM with reduced precision such as `16-bit` and `8-bit`.
<br/><br/>

### More TFLOPS as compared to CUDA cores
In L4 GPU, number of tensor cores are `240` as compared to `7424` CUDA cores but Tensor Cores offer higher peak TFLOPs of `120` as compared to only `30.3 TFLOPs` for CUDA cores on 32-bit floats. With 16-bit floats Tensor Cores offers `242 peak TFLOPs`. Tensor Cores can do `FMA (Fused Multiply and Add)` operations on `4x4` matrices in a single cycle wherease CUDA cores takes multiple cycles for the same.
<br/><br/>

In the above kernel, we are declaring warp level fragments `a_frag`, `b_frag`, `c_frag` and `acc_frag`. Each fragment is of shape `16x16` and of type `half` which is `FP16` data type (16-bit floats).
<br/><br/>
Each block comprises of `512 threads` with `128 threads` along `4 rows`. Each row of 128 threads is divided up into 4 warps each of 32 threads. Thus each block comprises of `4x4=16 warps`.
Each warp computes a `16x16` tile in the output matrix. Thus with `4x4 warps`, each block computes `64x64 tile` in the output matrix.
<br/><br/>
Each warp copies a `16x16 tile` from matrix A in global memory into `a_frag` register and a `16x16 tile` from matrix B into b_frag repeated along the k-dimension using `wmma::load_matrix_sync` command. Since a warp contains 32 threads thus to copy `16x16=256` elements each thread copies 8 elements from global memory to the fragments in registers.
<br/><br/>
```
Thread  0 loads 8 FP16 elements from 1st  row and 1st col.
Thread  1 loads 8 FP16 elements from 2nd  row and 1st col.
...
Thread 15 loads 8 FP16 elements from 16th row and 1st col.
Thread 16 loads 8 FP16 elements from 1st  row and 8th col.
Thread 17 loads 8 FP16 elements from 2nd  row and 9th col.
...
Thread 31 loads 8 FP16 elements from 16th row and 8th col.
```
<br/><br/>
![Warp TC](/docs/assets/ldmatrix.png)
<br/><br/>
Next `wmma::mma_sync` command multiplies the 16x16 tile in a_frag with a 16x8 tile in b_frag repeated 2 times horizontally (because b_frag is 16x16) using Tensor Cores. The results of the matmul operations are stored in `acc_frag`. Tensor Cores does `FMA (Fused Multiply and Add)` operation on the fragments.
<br/><br/>
The 16x16 tile is divided into 4 `8x8` tiles and per 8x8 of 16-bit elements, each thread loads 32-bits or 2 consecutive elements.
<br/><br/>
![Warp TC](/docs/assets/warptc.png)
<br/><br/>
Once a warp is done with multiplying `16x16` fragments, the results of the `acc_frag` is updated to `c_frag`. Since it is a GEMM operation where we are computing `D = alpha * AxB + beta * C`, the results of `AxB` are stored in `acc_frag`, and instead of using two separate matrices C and D, we are updating the matrix C itself with the final result assuming that the original matrix C is not used later. Once we update the `c_frag` with elementwise multiplication and summation, we update the results in the output matrix C in global memory.
<br/><br/>
In the event we do not want GEMM but only say the product `AxB`, we can just do `wmma::store_matrix_sync(c + cRow * ldc + cCol, acc_frag, ldc, wmma::mem_row_major)`.
<br/><br/>
Time taken to multiply two 4096x4096 matrices is around `14.7231 ms`.
<br/><br/>
A slightly better solution would be to first copy the 64x64 tile of FP16 floats from global memory to shared memory and then use `wmma::load_matrix_sync` to transfer data from shared memory to registers for each 16x16 sub-tile within the 64x64 tile. Since copying from global memory to shared memory also involves register, we can use async copy as previously seen.
<br/><br/>
```cpp
__global__ 
void gemm_wmma_shmm(
    const half *a, 
    const half *b, 
    float *c, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    __shared__ alignas(16) half Mds[NUM_STAGES_ASYNC_PIPELINE][TILE_WIDTH_WMMA*TILE_WIDTH_WMMA];
    __shared__ alignas(16) half Nds[NUM_STAGES_ASYNC_PIPELINE][TILE_WIDTH_WMMA*TILE_WIDTH_WMMA];

    int ldc = n;

    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    int a_block_row = by * TILE_WIDTH_WMMA;
    int b_block_col = bx * TILE_WIDTH_WMMA;

    for (int s = 0; s < NUM_STAGES_ASYNC_PIPELINE; s++) {
        int h = s*TILE_WIDTH_WMMA;

        pipeline.producer_acquire();
        #pragma unroll
        for (int j = idx; j < TILE_WIDTH_WMMA*TILE_WIDTH_WMMA; j += blockDim.x * blockDim.y) {
            cuda::memcpy_async(Mds[s] + j, a + (a_block_row + j/TILE_WIDTH_WMMA)*k + h + (j % TILE_WIDTH_WMMA), cuda::aligned_size_t<2>(sizeof(half)), pipeline);
            cuda::memcpy_async(Nds[s] + j, b + (h + j/TILE_WIDTH_WMMA)*n + b_block_col + (j % TILE_WIDTH_WMMA), cuda::aligned_size_t<2>(sizeof(half)), pipeline);
        }
        pipeline.producer_commit();
    }

    int s = NUM_STAGES_ASYNC_PIPELINE;

    for (int i = 0; i < k; i += TILE_WIDTH_WMMA) {
        int stage = s % NUM_STAGES_ASYNC_PIPELINE;

        constexpr size_t pending_batches = NUM_STAGES_ASYNC_PIPELINE - 1;
        cuda::pipeline_consumer_wait_prior<pending_batches>(pipeline);
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < TILE_WIDTH_WMMA; j += WMMA_K) {
            int a_warp_row = threadIdx.y * WMMA_M;
            int a_warp_col = j;

            int b_warp_row = j;
            int b_warp_col = (threadIdx.x / 32) * WMMA_N;

            if (a_warp_row < TILE_WIDTH_WMMA && a_warp_col < TILE_WIDTH_WMMA && b_warp_row < TILE_WIDTH_WMMA && b_warp_col < TILE_WIDTH_WMMA) {
                wmma::load_matrix_sync(a_frag, Mds[stage] + a_warp_row * TILE_WIDTH_WMMA + a_warp_col, TILE_WIDTH_WMMA);
                wmma::load_matrix_sync(b_frag, Nds[stage] + b_warp_row * TILE_WIDTH_WMMA + b_warp_col, TILE_WIDTH_WMMA);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
        }

        pipeline.consumer_release();
        __syncthreads();

        pipeline.producer_acquire();
        int h = s*TILE_WIDTH_WMMA;
        if (h < k) {
            #pragma unroll
            for (int j = idx; j < TILE_WIDTH_WMMA*TILE_WIDTH_WMMA; j += blockDim.x * blockDim.y) {
                cuda::memcpy_async(Mds[stage] + j, a + (a_block_row + j/TILE_WIDTH_WMMA)*k + h + (j % TILE_WIDTH_WMMA), cuda::aligned_size_t<2>(sizeof(half)), pipeline);
                cuda::memcpy_async(Nds[stage] + j, b + (h + j/TILE_WIDTH_WMMA)*n + b_block_col + (j % TILE_WIDTH_WMMA), cuda::aligned_size_t<2>(sizeof(half)), pipeline);
            }
        }
        pipeline.producer_commit();

        s += 1;
    }

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < m && cCol < n) {
        wmma::load_matrix_sync(c_frag, c + cRow * ldc + cCol, ldc, wmma::mem_row_major);

        #pragma unroll
        for(int i=0; i < c_frag.num_elements; i++) c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];

        wmma::store_matrix_sync(c + cRow * ldc + cCol, c_frag, ldc, wmma::mem_row_major);
    }
}
```
<br/><br/>
Time taken to multiply two 4096x4096 matrices is around `14.1765 ms`. This is slightly better than the previous kernel.
<br/><br/>

## Kernel 6 - mma.sync custom
```cpp
__global__ 
void gemm_mma_sync_fp16(
    const half *a, 
    const half *b, 
    float *c, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    __shared__ alignas(16) half Mds[TILE_WIDTH_WMMA*TILE_WIDTH_WMMA];
    __shared__ alignas(16) half Nds[TILE_WIDTH_WMMA*TILE_WIDTH_WMMA];

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    int warp_row_id = idx/blockDim.x;
    int warp_col_id = (idx % blockDim.x)/32;
    int thread_id_in_warp = idx % 32;

    for (int i = 0; i < k; i += TILE_WIDTH_WMMA) {
        int a_row = blockIdx.y * TILE_WIDTH_WMMA;
        int a_col = i;

        for (int j = idx; j < TILE_WIDTH_WMMA*TILE_WIDTH_WMMA; j += blockDim.x * blockDim.y) {
            Mds[j] = a[(a_row + j/TILE_WIDTH_WMMA) * k + (a_col + j % TILE_WIDTH_WMMA)];
        }

        int b_row = i;
        int b_col = blockIdx.x * TILE_WIDTH_WMMA;

        for (int j = idx; j < TILE_WIDTH_WMMA*TILE_WIDTH_WMMA; j += blockDim.x * blockDim.y) {
            Nds[j] = b[(b_row + j/TILE_WIDTH_WMMA) * n + (b_col + j % TILE_WIDTH_WMMA)];
        }

        __syncthreads();

        for (int j = 0; j < TILE_WIDTH_WMMA; j += 16) {
            uint32_t regs_a[4];

            uint32_t regs_b_1[2];
            uint32_t regs_b_2[2];

            float regs_c_1[4] = {0.0f};
            float regs_c_2[4] = {0.0f};

            int m_row = warp_row_id * 16;
            int m_col = j;

            int n_row = j;
            int n_col_1 = warp_col_id * 16;
            int n_col_2 = n_col_1 + 8;

            uint32_t addr_a   = __cvta_generic_to_shared(&Mds[(m_row + thread_id_in_warp % 16) * TILE_WIDTH_WMMA + (thread_id_in_warp/16) * 8 + m_col]);
            uint32_t addr_b_1 = __cvta_generic_to_shared(&Nds[(n_row + thread_id_in_warp % 16) * TILE_WIDTH_WMMA + n_col_1]);
            uint32_t addr_b_2 = __cvta_generic_to_shared(&Nds[(n_row + thread_id_in_warp % 16) * TILE_WIDTH_WMMA + n_col_2]);

            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                "{%0, %1, %2, %3}, [%4];"
                : "=r"(regs_a[0]), "=r"(regs_a[1]), "=r"(regs_a[2]), "=r"(regs_a[3])
                : "r"(addr_a)
            );

            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.shared.trans.b16 "
                "{%0, %1}, [%2];"
                : "=r"(regs_b_1[0]), "=r"(regs_b_1[1])
                : "r"(addr_b_1)
            );

            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.shared.trans.b16 "
                "{%0, %1}, [%2];"
                : "=r"(regs_b_2[0]), "=r"(regs_b_2[1])
                : "r"(addr_b_2)
            );

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%0, %1, %2, %3};\n"
                : "+f"(regs_c_1[0]), "+f"(regs_c_1[1]), "+f"(regs_c_1[2]),"+f"(regs_c_1[3])
                : "r"(regs_a[0]), "r"(regs_a[1]), "r"(regs_a[2]), "r"(regs_a[3]), "r"(regs_b_1[0]), "r"(regs_b_1[1])
            );

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%0, %1, %2, %3};\n"
                : "+f"(regs_c_2[0]), "+f"(regs_c_2[1]), "+f"(regs_c_2[2]),"+f"(regs_c_2[3])
                : "r"(regs_a[0]), "r"(regs_a[1]), "r"(regs_a[2]), "r"(regs_a[3]), "r"(regs_b_2[0]), "r"(regs_b_2[1])
            );

            #pragma unroll
            for (int q = 0; q < 4; q++) {
                int rw = (thread_id_in_warp >> 2) + 8 * (q / 2);
                int cl = 2 * (thread_id_in_warp % 4) + (q % 2);
                c[(a_row + m_row + rw) * n + (b_col + n_col_1 + cl)] += regs_c_1[q];
                c[(a_row + m_row + rw) * n + (b_col + n_col_2 + cl)] += regs_c_2[q];
            }
        }

        __syncthreads();
    }
}

float *c_gpu_mma_sync_fp16;
cudaErrCheck(cudaMallocManaged(&c_gpu_mma_sync_fp16, m * n * sizeof(float)));

for (auto i = 0; i < m*n; i++) c_gpu_mma_sync_fp16[i] = 0.0f;

dim3 bd6(128, 4, 1);
dim3 gd6((n+TILE_WIDTH_WMMA-1)/TILE_WIDTH_WMMA, (m+TILE_WIDTH_WMMA-1)/TILE_WIDTH_WMMA, 1);

gemm_mma_sync_fp16<<<gd6, bd6>>>(a_fp16, b_fp16, c_gpu_mma_sync_fp16, 1.0, 0.0, m, n, k);
cudaDeviceSynchronize();
cudaErrCheck(cudaFree(c_gpu_mma_sync_fp16));
```
<br/><br/>
In this kernel we show how to write the `Tensor Core GEMM` without using `WMMA`. The crucial parts of understanding the above kernel is understanding how `ldmatrix` PTX instruction is used to copy from shared memory to registers. For e.g. the instruction `ldmatrix.sync.aligned.m8n8.x4.shared.b16` is used to copy `4 8x8` submatrices of 16-bit data types from shared memory to registers. We saw this earlier with WMMA too where the 16x16 a_frag was divided up into 4 8x8 sub-tiles and each thread in a warp then copies 8 FP16 elements.
<br/><br/>
The next crucial part is how to make the PTX instruction for `mma.sync`. For e.g. `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` does matrix-multiply-and-add operation on a 16x16 matrix A and 16x8 matrix B where A is row-major and B is column major. A and B are both FP16 and output matrix is FP32.
<br/><br/>
Each block of thread is divided into 4x4 warps where each warp computes 16x16 sub-matrix of the output. Given an output matrix of shape 1024x1024, each block computes a 64x64 submatrix. In the earlier kernels, we would directly compute the 64x64 output tile by sliding horizontally across A and vertically across B and loading 64x64 tiles from global to shared memory and doing matmul on each tile and summing up the results across the tiles. In this kernel, we further divide each 64x64 tile into 16x16 sub-tiles which are handled by warps because we want to leverage the Tensor Cores.
<br/><br/>
A warp or a group of 32 threads loads the 16x16 sub-tile from shared memory to registers as follows:
<br/><br/>
```
Thread  0 loads 8 FP16 elements from 1st  row and 1st col.
Thread  1 loads 8 FP16 elements from 2nd  row and 1st col.
...
Thread 15 loads 8 FP16 elements from 16th row and 1st col.
Thread 16 loads 8 FP16 elements from 1st  row and 8th col.
Thread 17 loads 8 FP16 elements from 2nd  row and 9th col.
...
Thread 31 loads 8 FP16 elements from 16th row and 8th col.
```
<br/><br/>
Each 8 FP16 elements is read into 4 FP32 registers using `ldmatrix.sync.aligned.m8n8.x4.shared.b16`
```
asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(regs_a[0]), "=r"(regs_a[1]), "=r"(regs_a[2]), "=r"(regs_a[3])
    : "r"(addr_a)
);
```
<br/><br/>
Note that since `mma.sync` can only multiply a 16x16 matrix with a 16x8 matrix at a time, so we do 2 `mma.sync` operations each multiplying a 16x16 matrix with a 16x8 matrix and then merging the results.
<br/><br/>
Finally we update the output matrix C with the final results. Note that I am directly updating the C matrix in the global memory. A better approach here would be to use `stmatrix.sync` to write the output 16x16 matrix from registers to shared memory and then from shared memory to global memory.
<br/><br/>
Note the thread id to index mapping in the output matrix C. The indices corresponding to each thread id is computed based on the thread assignment to a 8x8 sub-matrix as shown in the diagram above.
<br/><br/>
```
Thread  0 computes elements (0,0) (0,1) (15,0) and (15,1) for a 16x8 output submatrix.
Thread  1 computes elements (0,2) (0,3) (15,2) and (15,3) for a 16x8 output submatrix.
...
Thread 31 computes elements (7,6) (7,7) (15,6) and (15,7) for a 16x8 output submatrix.
```
<br/><br/>
