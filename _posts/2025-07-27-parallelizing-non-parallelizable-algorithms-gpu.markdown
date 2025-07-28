---
layout: post
title:  "Parallelizing Non-Parallelizable algorithms on GPU - Prefix Sum"
date:   2025-07-30 18:50:11 +0530
categories: software-engineering
---
GPUs are highly effective in parallelizing algorithms more importantly algorithms which are inherently parallelizable as the ones we saw previously such as vector addition, matrix multiplication, convolution, histogram reduction etc. We also saw a GPU implementation of summation of an array of numbers. Unlike matrix multiplication or convolution where each thread is responsible for calculating independent or disjoint set of output values, summation of an array of numbers required only 1 output value and thus required synchronization between multiple threads. But with reduction tree technique and atomic addition it was relatively straightforward to achieve better performance on a GPU as compared to a CPU.<br/><br/>
Given an input array A of N numbers, prefix sum return an array P of size N where `P[i]` is the summation from `A[0] to A[i]`. This is pretty straightforward to calculate using C/C++ as shown below:<br/><br/>
```cpp
void prefix_sum(float *A, float *P, unsigned int n) {
    for (unsigned int i = 0; i < n; i++) {
        if (i == 0) P[i] = A[i];
        else P[i] = P[i-1] + A[i];
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
void prefix_sum_kogge_stone(float *A, float *P, int n) {
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n) P[index] = A[index];

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp = 0.0f;
        if (threadIdx.x >= stride) temp = P[threadIdx.x-stride];
        __syncthreads();
        P[threadIdx.x] += temp;
        __syncthreads();
    }
}
```
<br/><br/>
In the above code we are using a temp variable to store the results of `P[threadIdx.x-stride]` before updating `P[threadIdx.x]` because for e.g. if stride=2 and threadIdx.x=5, then the thread will add `P[3]` to `P[5]` and update `P[5]`. But since threads are running in parallel, it could be that the thread with threadIdx.x=3, is also updating `P[3] = P[3] + P[1]`. Now if threadIdx.x=3 updates `P[3]` before threadIdx.x=5 reads `P[3]` we will have incorrect value stored in `P[5]` as `P[5]` requires the older value of `P[3]` and not the current value. Hence we first need to read all the older values in thread specific registers (`float temp`) and then after all threads have stored these values, we update the values.<br/><br/>
But note that the above code will only work correctly if there is only 1 block of thread. But since a block can have a maximum upto 1024 threads thus the above code is only able to handle array sizes N <= 1024. But why this is so ?<br/><br/>
Assuming we are having multiple blocks and each block contains 1024 threads, now for the index say 1025 i.e. block index=1 and stride=4, the update equation will look like `P[1025] = P[1025] + P[1021]`.<br/><br/>
But note that index=1021 lies in block=0 while index=1025 in block=1 and `__syncthreads()` is only applicable at the block level i.e. the threads corresponding to indices 1021 and 1025 will not be synchronized and as a result `P[1025]` might read the updated value of `P[1021]` instead of the old value leading to incorrect results.<br/><br/>
Block synchronization is a tricky affair in CUDA as not all blocks will be running in parallel. If number of blocks are greater than the number of streaming multiprocessors, then only a subset of all blocks will be running in parallel. The rest of the blocks will wait for their turn.<br/><br/>
```cpp
__device__ unsigned int counter = 0

__global__
void prefix_sum_kogge_stone(float *A, float *P, int n) {
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n) P[index] = A[index];

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp = 0.0f;
        if (threadIdx.x >= stride) temp = P[threadIdx.x-stride];
        // synchronize all threads across all blocks
        while (atomicAdd(&counter, 1) < blockDim.x*gridDim.x) {}
        P[threadIdx.x] += temp;
        // synchronize all threads across all blocks
        while (atomicSub(&counter, 1) > 0) {}
    }
}
```
<br/><br/>
A common way to synchronize all threads across all blocks is to use a `while () {}` loop like the one shown above. Using a global variable `counter`, each threads takes turn to update its value and when all thread updates the value only then the current thread is able to break out of the while loop. A common danger in the above code is when number of SMs are smaller than the number of blocks in which case we might see a deadlock happening.<br/><br/>
Synchronizing all threads across all blocks penalize performance heavily. An alternative way to implement the Kogge-Stone algorithm is to run the algorithm per block first. After this all blocks would have computed its own prefix sums. Except for the 1st block all other blocks will have only partial prefix sums.<br/><br/>
Using a global array S of length equal to the number of blocks, each index i in S stores the value of the prefix sum from the last index from each block. Thus each element of S corresponds to the sum of one block.<br/><br/>
Then run the prefix sum algorithm again but now on the global array S. <br/><br/>
Then for each block, for each index i add the value of `S[blockIdx.x-1]` i.e. the value of S corresponding to the previous block to itself. In this way each output element will have the correct value.<br/><br/>
Taking an example:
```
Input : A = [2,1,5,8,9,0,4,6,3,4,5,4,1,7,7,2]
block size = 4

STEP 1: Calculate P array for each block
A0 = [2,1,5,8] P0 = [2,3,8,16]
A1 = [9,0,4,6] P1 = [9,9,13,19]
A2 = [3,4,5,4] P2 = [3,7,12,16]
A3 = [1,7,7,2] P3 = [1,7,15,17]

STEP 2: Calculate S array from last elements of each P above
S  = [16, 19, 16, 17]

STEP 3: Calculate prefix sum array for S from Step 2
PS = [16, 35, 51, 68]

STEP 3: Update P array for each block using the prefix sum S array PS above
P0' = P0 = [2, 3, 8, 16]
P1' = P1 + PS[0] = [9+16, 9+16, 13+16, 19+16] = [25, 25, 29, 35]
P2' = P2 + PS[1] = [3+35, 7+35, 12+35, 16+35] = [38, 42, 47, 51]
P3' = P3 + PS[2] = [1+51, 7+51, 15+51, 17+51] = [52, 58, 66, 68]

STEP 5: Concatenate the P arrays
P = [2, 3, 8, 16, 25, 25, 29, 35, 38, 42, 47, 51, 52, 58, 66, 68]
```
<br/><br/>
Instead of running prefix sum algorithm twice, once on A and once on S in the above algorithm, one can modify the above algorithm as follows:<br/><br/>
After step 1, for each block check if the S element corresponding to the previous block i.e. `S[blockIdx.x-1]` has been set. If the S element in the previous block has been set, then update the S element of current block by adding `S[blockIdx.x-1]` to the last element of its P array. After that, update all indices for P in the current block by adding `S[blockIdx.x-1]`.<br/><br/>
```
Input : A = [2,1,5,8,9,0,4,6,3,4,5,4,1,7,7,2]
block size = 4

STEP 1: Calculate P array for each block
A0 = [2,1,5,8] P0 = [2,3,8,16]
A1 = [9,0,4,6] P1 = [9,9,13,19]
A2 = [3,4,5,4] P2 = [3,7,12,16]
A3 = [1,7,7,2] P3 = [1,7,15,17]

STEP 2: Calculate S array from last elements of P above and previous blocks' S value.
S0 = 0  + 16 = 16  P0 = [2, 3, 8, 16]
S1 = 16 + 19 = 35  P1 = [9+16, 9+16, 13+16, 19+16] = [25, 25, 29, 35]
S2 = 35 + 16 = 51  P2 = [3+35, 7+35, 12+35, 16+35] = [38, 42, 47, 51]
S3 = 51 + 17 = 68  P3 = [1+51, 7+51, 15+51, 17+51] = [52, 58, 66, 68]

STEP 3: Concatenate the P arrays
P = [2, 3, 8, 16, 25, 25, 29, 35, 38, 42, 47, 51, 52, 58, 66, 68]
```
<br/><br/>
The calculations using Kogge-Stone can be further optimized by using shared memory array. In order to identify whether the S element corresponding to previous block has been set or not, we use another `flags` array where `flags[blockIdx.x]=1` if S corresponding to `blockIdx.x` has been calculated, else `flags[blockIdx.x]=0`.<br/><br/>
In the below code we use the logic discussed above but with a small change that each block updates the S element for the next block by adding the S value of the current block to it. Similarly for the `flags` boolean array. This doesn't change the algorithm only the block indices are shifted. The code is as follows:<br/><br/>
```cpp
__device__
void prefix_sum_kogge_stone_block(float *arr, float *XY, int n) {
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index < n) XY[threadIdx.x] = arr[index]; 
    else XY[threadIdx.x] = 0.0f;

    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp = 0.0f;
        if (threadIdx.x >= stride) temp = XY[threadIdx.x-stride];
        __syncthreads();
        XY[threadIdx.x] += temp;
        __syncthreads();
    }
}

__global__
void prefix_sum(float *A, float *P, int *flags, float *S, int n, int m) {
    extern __shared__ float XY[];
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;

    // calculate kogge-stone algorithm per blcok and store the results in a shared memory array
    prefix_sum_kogge_stone_block(A, XY, n);

    // since S is updated once per block, thus the below code runs only for the 1st thread in each block
    if (blockIdx.x + 1 < m && threadIdx.x == 0) {
        // check if the previous block has set the S element corresponding to current block
        // if not then continue while loop else break loop.
        while (atomicAdd(&flags[blockIdx.x], 0) == 0) {}

        // update the S element for the next block by adding the S element of
        // current block with the last element of XY shared memory array
        S[blockIdx.x + 1] = S[blockIdx.x] + XY[min(blockDim.x-1, n-1-blockIdx.x*blockDim.x)];

        // __threadfence() indicates that any other block will see the change in S before the change in
        // flag set below.
        __threadfence();

        // set the flag for the next block
        atomicAdd(&flags[blockIdx.x + 1], 1);
    }

    // syncing threads is required here because we are reading one index of XY
    // above and updating a different index of XY below.
    __syncthreads();

    if (blockIdx.x < m && blockIdx.x > 0) {
        // check if the flag is set for current block which implies that the S element is also updated.
        while (atomicAdd(&flags[blockIdx.x], 0) == 0) {}

        // update XY shared memory array for all threads in the current block
        XY[threadIdx.x] += S[blockIdx.x];
    }

    // finally update the global memory array P with XY shared memory array.
    if (index < n) P[index] = XY[threadIdx.x];
}
```
<br/><br/>

