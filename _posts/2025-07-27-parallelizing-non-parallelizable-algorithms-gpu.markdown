---
layout: post
title:  "Parallelizing Non-Parallelizable algorithms on GPU - Prefix Sum"
date:   2025-07-30 18:50:11 +0530
categories: software-engineering
---
GPUs are highly effective in parallelizing algorithms more importantly algorithms which are inherently parallelizable as the ones we saw previously such as vector addition, matrix multiplication, convolution, histogram reduction etc. We also saw a GPU implementation of summation of an array of numbers. Unlike matrix multiplication or convolution where each thread is responsible for calculating independent or disjoint set of output values, summation of an array of numbers required only 1 output value and thus required synchronization between multiple threads. But with reduction tree technique and atomic addition it was relatively straightforward to achieve better performance on a GPU as compared to a CPU.<br/><br/>
Given an input array A of N numbers, prefix sum return an array P of size N where `P[i]` is the summation from `A[0] to A[i]`. This is pretty straightforward to calculate using C/C++ as shown below:<br/><br/>
```cpp
void prefix_sum(float *arr, float *out, unsigned int n) {
    for (unsigned int i = 0; i < n; i++) {
        if (i == 0) out[i] = arr[i];
        else out[i] = out[i-1] + arr[i];
    }
}
```
<br/><br/>
The time complexity of the above algorithm is `O(N)`.<br/><br/>
If we use a separate thread to calculate the prefix sum for each index i in the output array P, then for the last index i.e. `P[N-1]`, the corresponding thread needs to calculate the sum of all numbers in the input array A. Thus the worst case time complexity is still `O(N)` assuming each thread is running in parallel. <br/><br/>
```cpp
__global__
void prefix_sum_cuda_unoptimized(float *arr, float *out, unsigned int n) {
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (unsigned int i = 0; i <= index; i++) {
        sum += arr[i];
    }
    if (index < n) out[index] = sum;
}
```
<br/><br/>
On the other hand total work done across all threads is `O(N^2)`. But with large N, not all threads will be running in parallel as blocks are scheduled in streaming multiprocessors according to available resources. Thus, worst case time complexity is greater than `O(N)`.<br/><br/>
Thus each thread working independently will have a performance that is worse than the sequential algorithm implementation. On the other hand if for each index i, all threads collaborate to calculate the sum from `A[0] to A[i]` just like the summation problem using a reduction tree, then although calculation for each index is optimized but overall we still need to calculate with all indices in a sequential manner leading to a performance that is again worse than the sequential algorithm.<br/><br/>
An alternative stratgey would be to use a combination of the 2 approaches above i.e. each thread independently calculates the prefix sums for a block of the array A and then later collaborate to merge the block prefix sums. Two popular algorithms exists for the same. One is known as the Kogge-Stone algorithm and the other is the Brent-Kung algorithm. Both the algorithms are used in digital circuit design for implementing adders.<br/><br/>
[Kogge Stone Adder](https://en.wikipedia.org/wiki/Kogge–Stone_adder)<br/><br/>
[Brent Kung Adder](https://en.wikipedia.org/wiki/Brent–Kung_adder)<br/><br/>
Let's start with the Kogge Stone adder first. The diagrammatic representation is shown below. The algorithm works in stages as follows:<br/><br/>
Copy the input array A in the output array P. Then for each stage S (starting from 0):<br/><br/>
    1. For each index i greater than equal to `(1<<S)` i.e. 2 to the power of S, in the output array P, calculate the sum `P[i] = P[i]+P[i-(1<<S)]`<br/><br/>
At the end of `log(N)` stages, each index i will contain the sum of `A[0] to A[i]`.<br/><br/>
Let's implement the above in CUDA as follows:
```cpp
__device__
void prefix_sum_kogge_stone(float *arr, float *out, int n) {
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n) out[index] = arr[index];

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp = 0.0f;
        if (threadIdx.x >= stride) temp = out[threadIdx.x-stride];
        __syncthreads();
        out[threadIdx.x] += temp;
        __syncthreads();
    }
}
```
<br/><br/>
