---
layout: post
title:  "Writing your own Softmax PyTorch operator"
date:   2025-09-02 18:50:11 +0530
categories: software-engineering
---
PyTorch offers convenient ways for writing your own custom operators and extensions in both `C++` and `Python`. One can also leverage `CUDA` to write extensions for GPU devices. In this post I am going to show how to develop your own custom `softmax` operator for both CPU and GPU devices using C++ and Python.<br/><br/>
Softmax is a common operation used in deep learning networks. They are used to turn prediction scores into probabilities in binary classification problems. It is also used in attention mechanism to compute the `attention scores` for a sequence.<br/><br/>
Before beginning to write our own softmax operator, lets see how to use the in-built softmax operator from PyTorch:<br/><br/>
```python
import torch
a = torch.randn(3, 4, dtype=torch.float32)
print(torch.nn.functional.softmax(a, dim=-1))
```
We created a 3x4 matrix with random numbers between 0 and 1 and then used softmax operator to turn each row into probabilities. Note the dim=-1 argument which says that the softmax should be computed w.r.t. the last dimension i.e. across columns in this case.<br/><br/> 
For an input array `[x0, x1, ... xn-1]` , the softmax values are computed as follows:<br/><br/>
But the problem with the above formulation is that when the values `xi` are high as say 500, `exp(xi)` can cause overflow and return `Infinity` which will cause the softmax calculation to fail. One possible solution is to multiply both numerator and denominator by the constant `exp(-max(x))`. Then the updated formula is:<br/><br/>
Now since we know how softmax should be computed, assuming our input tensors are 2D in shape, let's write the following C++ function to calculate the softmax. For each row compute the maximum values and then calculate the sum of all the exponentials (denominator) per row. Since each row can be computed in parallel, we will leverage multi-threading for this. One can use either `OpenMP`, or `TBB` (Thread Building Blocks) for this or explicit thread management. Here I am using TBB. <br/><br/>
```cpp
// File name : pytorch_c_ext.cpp

#include <Python.h>
#include <torch/extension.h>
#include <tbb/tbb.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>

namespace extension_cpp {
	void softmax(const float *inp, float *out, const unsigned long n, const unsigned long m) {
		float *max_per_row = new float[n];
		float *sum_per_row = new float[n];
	
		// Initialize max_per_row and sum_per_row
		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, n), 
			[&max_per_row, &sum_per_row](tbb::blocked_range<size_t> r) {
			for (auto i = r.begin(); i < r.end(); i++) {
				max_per_row[i] = -MAXFLOAT;
				sum_per_row[i] = 0.0;
			}
		});
	
		// Calculate max_per_row
		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, n), 
			[&max_per_row, &inp, m](tbb::blocked_range<size_t> r) {
			for (auto i = r.begin(); i < r.end(); i++) {
				for (unsigned long j = 0; j < m; j++) {
					max_per_row[i] = std::max(max_per_row[i], inp[i*m+j]);
				}
			}
		});
	
		// Calculate sum_per_row
		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, n), 
			[&sum_per_row, &max_per_row, &inp, m](tbb::blocked_range<size_t> r) {
			for (auto i = r.begin(); i < r.end(); i++) {
				for (unsigned long j = 0; j < m; j++) {
					sum_per_row[i] += exp(inp[i*m+j]-max_per_row[i]);
				}
			}
		});
	
		// Calculate softmax per row
		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, n), 
			[&sum_per_row, &max_per_row, &inp, &out, m](tbb::blocked_range<size_t> r) {
			for (auto i = r.begin(); i < r.end(); i++) {
				for (unsigned long j = 0; j < m; j++) {
					out[i*m+j] = exp(inp[i*m+j]-max_per_row[i])/sum_per_row[i];
				}
			}
		});
	}
}
```
But note that the above function cannot be directly used from PyTorch. For that first we need to define another C++ method that directly works with the PyTorch C++ Frontend. This method calls the above `softmax` method.<br/><br/>
```cpp
namespace extension_cpp {
	torch::Tensor softmax_cpu(const torch::Tensor &a) {
		// Input valiidation
        TORCH_CHECK(a.device().is_cpu(), "Input tensor a must be a CPU tensor");
        TORCH_CHECK(a.is_contiguous(), "Input tensor a must be contiguous");
        TORCH_CHECK(a.dtype() == torch::kFloat32, "Input tensor a must be float32");

		// Output Tensor
        torch::Tensor c = torch::empty_like(a);
        unsigned long n = a.size(0);
        unsigned long m = a.size(1);
    
        softmax(
            a.data_ptr<float>(),
            c.data_ptr<float>(),
            n, 
            m
        );
    
        return c;
    }
}
```
Next we need to export the above C++ function so that it can be called from Python. For that we will use PYBIND11. Our custom softmax function can be called in Python using `extension_cpp.mysoftmax_cpu(Tensor)`.<br/><br/>
```cpp
namespace extension_cpp {
	PYBIND11_MODULE(extension_cpp, m) {
        m.def("mysoftmax_cpu", &softmax_cpu, "Softmax CPU Forward");
    }
}
```
Notice the 1st three C++ header files included above `Python.h`, `torch/extension.h` and `tbb/tbb.h`. These files may not be automatically included in your path. To include the file `Python.h` requires you to specify your Python installation `include` directory. Also it requires `libpython-dev` to be installed. This path can be found by running the following commands on the Python shell.<br/><br/>
```python
import sysconfig
print(sysconfig.get_paths()['include'])
```
If `libpython-dev` is not installed, install them by running the command `apt-get install libpython-dev` (in Ubuntu).<br/><br/>
For the `torch/extension.h` file, this requires you to include the libtorch installation directory. If torch is automatically installed using `pip install torch` then it must be present in the site-packages folder (virtual environment if you are using one). The torch installation path can be found using:<br/><br/>
```python
import torch
print(torch.__file__)
```
You need to include two different include directories for the `torch/extension.h` file to work. For e.g. in Linux Ubuntu running Python 3.10, the following two paths are required to be included:<br/><br/>
```
/opt/python/3.10/lib/python3.10/site-packages/torch/include/torch/csrc/api/include,
/opt/python/3.10/lib/python3.10/site-packages/torch/include
```
To include `tbb/tbb.h` (for using multi-threading with TBB), you need to first install `tbb`. In MacOS it can be installed via `brew install tbb` and in Linux Ubuntu, it can be installed via `apt-get install libtbb-dev`. To build the C++ file into binary file and make our softmax functions callable from Python, we need to run setup.py script. Below is an example of setup.py script I am using for running in Linux/Ubuntu:<br/><br/>
```python
# File name : setup_ubuntu.py

import torch
from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    BuildExtension,
)

setup(
    name="extension_cpp",
    version="0.0.1",
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            "extension_cpp",
            ["pytorch_c_ext.cpp"],
            extra_compile_args={
                "cxx": ["-O3", "-ltbb", "-Wall"]
            },
            extra_link_args=["-ltbb"],
            include_dirs=[
                "/opt/python/3.10/include/python3.10", # for Python.h
                "/opt/python/3.10/lib/python3.10/site-packages/torch/include/torch/csrc/api/include", # for torch/extension.h
                "/opt/python/3.10/lib/python3.10/site-packages/torch/include", # for torch/extension.h
                "/usr/include" # for tbb/tbb.h
            ],
            library_dirs=["/usr/lib/x86_64-linux-gnu"]
        )
    ],
    install_requires=["torch"],
    description="Custom softmax implementation",
    cmdclass={"build_ext": BuildExtension}
)
```
Notice the `extra_compile_args` and `extra_link_args` in the above script. Also the `include_dirs` contains the include paths for the above mentioned header files. Note that these paths may vary depending on your OS and distribution. For e.g. in my MacOS, the setup.py script that works with the C++ file is:<br/><br/>
```python
# File name : setup_macos.py

import torch
from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    BuildExtension,
)

setup(
    name="extension_cpp",
    version="0.0.1",
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            "extension_cpp",
            ["pytorch_c_ext.cpp"],
            extra_compile_args={
                "cxx": ["-O3", "-ltbb", "-Wall"]
            },
            extra_link_args=["-ltbb"],
            include_dirs=[
                "/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/include/python3.13", # for Python.h
                "/Users/amondal/recsys/.venv/lib/python3.13/site-packages/torch/include/torch/csrc/api/include", # for torch/extension.h
                "/Users/amondal/recsys/.venv/lib/python3.13/site-packages/torch/include", # for torch/extension.h
                "/opt/homebrew/opt/tbb/include" # for tbb/tbb.h
            ],
            library_dirs=["/opt/homebrew/opt/tbb/lib"]
        )
    ],
    install_requires=["torch"],
    description="Custom softmax implementation",
    cmdclass={"build_ext": BuildExtension}
)
```
To build the C++ files in Ubuntu using setup.py, we can run the following command to build wheel files (similar command for MacOS too): `python3 setup_ubuntu.py bdist_wheel`<br/><br/>
This will create the wheel file with the package name and version inside the `dist` folder. To install the wheel file run pip install as follows: `python3 -m pip install dist/*.whl`<br/><br/>
This will install the python package in the site-packages folder. Once installed, you can use it in your Python code by importing `extension_cpp`. Following example shows how to use the above custom softmax operator.<br/><br/>
```python
# File name : mytest.py

import torch
import extension_cpp
import time

a_cpu = torch.randn(1000, 1024, dtype=torch.float32, requires_grad=False, device='cpu')

start = time.time()*1000
b1 = torch.nn.functional.softmax(a_cpu, dim=-1, dtype=torch.float32)
end = time.time()*1000
print("Torch CPU Forward Pass Duration = ", end-start)
print("Torch CPU Forward Pass Output\n", b1)
print()

start = time.time()*1000
b2 = extension_cpp.mysoftmax_cpu(a_cpu)
end = time.time()*1000
print("Custom CPU Forward Pass Duration = ", end-start)
print("Custom CPU Forward Pass Output\n", b2)
```
You can compare the outputs and the run-times with the built-in softmax vs the custom softmax operator. On my MacOS M4 ARM chip, I get the following performance numbers with a random matrix of shape 1000x1024.<br/><br/>
```
Torch  CPU Forward Pass Duration =  0.6728515625
Custom CPU Forward Pass Duration =  1.5710449218
```
On an Intel server running Ubuntu 22.04, I get the following performance numbers.<br/><br/>
```
Torch  CPU Forward Pass Duration =  1.87451171875
Custom CPU Forward Pass Duration =  6.12963867187
```
Usually the custom softmax operator is slower than the in-built one since the C++ code is not very optimized. You can further improve the performace of the custom C++ code by using SIMD (available on Intel CPU machines). The in-built softmax can perform exceptionally well for some special matrix dimensions and structure such as large number of columns than rows or number of rows and columns with powers of 2 and so on. The custom implemention performs better when number of rows are far greater than the number of columns because the multi-threading is applied along rows. <br/><br/>
So far we have only implemented a custom softmax version for the CPU, let's build the same for GPU using CUDA. Define a CUDA file with the below 2 functions as follows:<br/><br/>
```cpp
// File name : mysoftmax.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>

#define BLOCK_WIDTH_PER_DIM 32

__device__ __forceinline__ float atomicMaxF32(float *address, float val) {
    int ret = __float_as_int(*address);
    while(val > __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

__global__
void softmax_cuda(const float *inp, float *out, const unsigned long n, const unsigned long m) {
    unsigned long row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned long p = (m+BLOCK_WIDTH_PER_DIM-1)/BLOCK_WIDTH_PER_DIM;

    __shared__ float max_per_row[BLOCK_WIDTH_PER_DIM];
    __shared__ float sum_per_row[BLOCK_WIDTH_PER_DIM];

    if (row < n) {
        if (threadIdx.x == 0) {
            max_per_row[threadIdx.y] = -MAXFLOAT;
            sum_per_row[threadIdx.y] = 0.0f;
        }
        __syncthreads();

        for (unsigned long j = threadIdx.x; j < p*BLOCK_WIDTH_PER_DIM; j += BLOCK_WIDTH_PER_DIM) {
            if (j < m) {
                atomicMaxF32(&max_per_row[threadIdx.y], inp[row*m + j]);
            }
        }

        __syncthreads();

        for (unsigned long j = threadIdx.x; j < p*BLOCK_WIDTH_PER_DIM; j += BLOCK_WIDTH_PER_DIM) {
            if (j < m) {
                atomicAdd(&sum_per_row[threadIdx.y], exp(inp[row*m + j]-max_per_row[threadIdx.y]));
            }
        }

        __syncthreads();

        for (unsigned long j = threadIdx.x; j < p*BLOCK_WIDTH_PER_DIM; j += BLOCK_WIDTH_PER_DIM) {
            if (j < m) {
                out[row*m + j] = exp(inp[row*m + j]-max_per_row[threadIdx.y])/sum_per_row[threadIdx.y];
            }
        }
    }
}

void softmax_cuda_launcher(const float *inp, float *out, const unsigned long n, const unsigned long m) {
    dim3 bd(BLOCK_WIDTH_PER_DIM, BLOCK_WIDTH_PER_DIM, 1);
    dim3 gd(1, (n+BLOCK_WIDTH_PER_DIM-1)/BLOCK_WIDTH_PER_DIM, 1);

    softmax_cuda<<<gd, bd>>>(inp, out, n, m);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
}
```
We launch a 2D grid of threads and blocks where each block has 32 threads along x-dim and 32 threads along the y-dim (because maximum number of threads per block can be 1024). We launch only 1 block along the x-dim i.e. columns because we need to compute the maximum value and the summation along columns and having multiple blocks will require block synchronization if we are using shared memory to store the maximum and summation values. In the event we do not use shared memory but global memory for storing the maximum and summations, we can launch multiple blocks along the x-dim too.<br/><br/>
Since a block has 32 threads along x-dim, thus for `m` columns in the input matrix, each thread is responsbile for `ceil(m/32)` number of elements. The maximum and summation values are computed in shared memory to avoid global memory latencies. <br/><br/>
We update our C++ file to include the function declaration for our function `softmax_cuda_launcher` and as well define the torch C++ frontend for interacting with the CUDA launcher. Another modification needed is to update the PYBIND11 module to include the CUDA version of the softmax.<br/><br/>
```cpp
// File name : pytorch_c_ext.cpp

#include <Python.h>
#include <torch/extension.h>
#include <tbb/tbb.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>

void softmax_cuda_launcher(const float *inp, float *out, const unsigned long n, const unsigned long m);

namespace extension_cpp {
	torch::Tensor softmax_gpu(const torch::Tensor &a) {
        TORCH_CHECK(a.device().is_cuda(), "Input tensor a must be a CUDA tensor");
        TORCH_CHECK(a.is_contiguous(), "Input tensor a must be contiguous");
        TORCH_CHECK(a.dtype() == torch::kFloat32, "Input tensor a must be float32");
    
        torch::Tensor c = torch::empty_like(a);
        unsigned long n = a.size(0);
        unsigned long m = a.size(1);
    
        softmax_cuda_launcher(
            a.data_ptr<float>(),
            c.data_ptr<float>(),
            n, 
            m
        );
    
        return c;
    }

	PYBIND11_MODULE(extension_cpp, m) {
        m.def("mysoftmax_cpu", &softmax_cpu, "Softmax CPU Forward");
        m.def("mysoftmax_gpu", &softmax_gpu, "Softmax GPU Forward");
    }
}
```
Next we also need to modify the `setup.py` script to include CUDA specific configurations. The following is the updated `setup_ubuntu.py` script from above.<br/><br/>
```python
# File name : setup_ubuntu.py

import torch
from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
)

setup(
    name="extension_cpp",
    version="0.0.1",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            "extension_cpp",
            ["pytorch_c_ext.cpp", "mysoftmax.cu"],
            extra_compile_args={
                "nvcc": ["-arch=sm_89", "-Xcompiler=-O3"],
                "cxx": ["-O3", "-ltbb", "-Wall"]
            },
            extra_link_args=["-ltbb"],
            include_dirs=[
                "/opt/python/3.10/include/python3.10", 
                "/opt/python/3.10/lib/python3.10/site-packages/torch/include/torch/csrc/api/include",
                "/opt/python/3.10/lib/python3.10/site-packages/torch/include",
                "/usr/include"
            ],
            library_dirs=["/usr/lib/x86_64-linux-gnu"]
        )
    ],
    install_requires=["torch"],
    description="Custom CUDA softmax extension",
    cmdclass={"build_ext": BuildExtension}
)
```


