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
CPU is limited by the number of threads because CPUs are optimized for lowering the latency of a single process instead of solving problems in parallel. Most modern GPUs have thousands of cores or threads to perform GEMM in parallel and that would be the topic of this post. We will try to optimize GEMM on GPUs by exploring leveraging different CUDA kernel optimization strategies.
