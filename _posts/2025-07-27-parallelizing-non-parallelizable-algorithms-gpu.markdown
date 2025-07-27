---
layout: post
title:  "Parallelizing Non-Parallelizable algorithms on GPU - Prefix Sum"
date:   2025-07-30 18:50:11 +0530
categories: software-engineering
---
GPUs are highly effective in parallelizing algorithms more importantly algorithms which are inherently parallelizable as the ones we saw previously such as vector addition, matrix multiplication, convolution, histogram reduction etc. We also saw a GPU implementation of summation of an array of numbers. Unlike matrix multiplication or convolution where each thread is responsible for calculating independent or disjoint set of output values, summation of an array of numbers required only 1 output value and thus required synchronization between multiple threads. But with reduction tree technique and atomic addition it was relatively straightforward to achieve better performance on a GPU as compared to a CPU.<br/><br/>
Given an input array A of N numbers, prefix sum return an array P of size N where `P[i]` is the summation from `A[0] to A[i]`. This is pretty straightforward to calculate using C/C++ as shown below:<br/><br/>
    ```cpp
    void prefix_sum(float *arr, float *out, int n) {
        for (int i = 0; i < n; i++) {
            if (i == 0) out[i] = arr[i];
            else out[i] = out[i-1] + arr[i];
        }
    }
    ```
    <br/><br/>
The time complexity of the above algorithm is `O(N)`.<br/><br/>
If we use a separate thread to calculate the prefix sum for each index i in the output array P, then for the last index i.e. `P[N-1]`, the corresponding thread needs to calculate the sum of all numbers in the input array A. Thus the worst case time complexity is still `O(N)` assuming each thread is running in parallel. On the other hand total work done across all threads is `O(N^2)`. But with large N, not all threads will be running in parallel as blocks are scheduled in streaming multiprocessors according to available resources. Thus, worst case time complexity is greater than `O(N)`.<br/><br/>
Another possible approach is to use a reduction tree like technique to calculate `P[i]` instead of a sequential sum 
