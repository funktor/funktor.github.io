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




