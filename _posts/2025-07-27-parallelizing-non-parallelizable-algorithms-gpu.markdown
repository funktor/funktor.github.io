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
Copy the input array A in the output array P. Then for each stage S (starting from 0), for each index i greater than equal to `(1<<S)` i.e. 2 to the power of S, in the output array P, calculate the sum `P[i] = P[i]+P[i-(1<<S)]`<br/><br/>
At the end of `log(N)` stages, each index i will contain the sum of `A[0] to A[i]`.<br/><br/>
Let's implement the above in CUDA as follows:<br/><br/>
```cpp
__global__
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
In the above code we are using a temp variable to store the results of `out[threadIdx.x-stride]` before updating `out[threadIdx.x]` because for e.g. if stride=2 and threadIdx.x=5, then the thread will add `out[3]` to `out[5]` and update `out[5]`. But since threads are running in parallel, it could be that the thread with threadIdx.x=3, is also updating `out[3] = out[3] + out[1]`. Now if threadIdx.x=3 updates `out[3]` before threadIdx.x=5 reads `out[3]` we will have incorrect value stored in `out[5]` as `out[5]` requires the older value of `out[3]` and not the current value. Hence we first need to read all the older values in thread specific registers (`float temp`) and then after all threads have stored these values, we update the values.<br/><br/>
But note that the above code will only work correctly if there is only 1 block of thread. But since a block can have a maximum upto 1024 threads thus the above code is only able to handle array sizes N <= 1024. But why this is so ?<br/><br/>
Assuming we are having multiple blocks and each block contains 1024 threads, now for the index say 1025 i.e. block index=1 and stride=4, the update equation will look like `out[1025] = out[1025] + out[1021]`.<br/><br/>
But note that index=1021 lies in block=0 while index=1025 in block=1 and `__syncthreads()` is only applicable at the block level i.e. the threads corresponding to indices 1021 and 1025 will not be synchronized and as a result `out[1025]` might read the updated value of `out[1021]` instead of the old value leading to incorrect results.<br/><br/>
Block synchronization is a tricky affair in CUDA as not all blocks will be running in parallel. If number of blocks are greater than the number of streaming multiprocessors, then only a subset of all blocks will be running in parallel. The rest of the blocks will wait for their turn.<br/><br/>
```cpp
__device__ unsigned int counter = 0

__global__
void prefix_sum_kogge_stone(float *arr, float *out, int n) {
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n) out[index] = arr[index];

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp = 0.0f;
        if (threadIdx.x >= stride) temp = out[threadIdx.x-stride];
        // synchronize all threads across all blocks
        while (atomicAdd(&counter, 1) < blockDim.x*gridDim.x) {}
        out[threadIdx.x] += temp;
        // synchronize all threads across all blocks
        while (atomicSub(&counter, 1) > 0) {}
    }
}
```
<br/><br/>
A common way to synchronize all threads across all blocks is to use a `while () {}` loop like the one shown above. Using a global variable `counter`, each threads takes turn to update its value and when all thread updates the value only then the current thread is able to break out of the while loop. A common danger in the above code is when number of SMs are smaller than the number of blocks in which case we might see a deadlock happening.<br/><br/>

