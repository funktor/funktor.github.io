---
layout: post
title:  "Experimenting with Time Series Floating Point Compression"
date:   2025-02-12 18:50:11 +0530
categories: software-engineering
---

While working on time series forecasting for [demand and supply of spot virtual machines in Azure](https://funktor.github.io/ml/2025/02/04/demand-supply-forecasting-virtual-machines.html), intermediate data processing was done in a distributed fashion using Spark. Often the data size in each executor node exceeded 50-100 GB. This caused some issues while doing map-reduce operations.

1. When the results from each executor node was collected at the driver node, the data size exceeded available RAM and caused the driver nodes to crash.

2. Transferring GBs of data over network has throughput implications. Network trasfers of large data often took long time.

One hack we did early on to mitigate issue number 1 above was to write the outputs of the executor nodes into blob storage in Azure. Then read back the files from the driver node sequentially and process the data without hogging a lot of memory. Once we mounted the blob storage using NFS, reading and writing became a lot faster.

For issue number 2, there was no fast hack but to compress the outputs of each executor node before sending them over the network. Sometimes the data structure was a dataframe with mixed data types, sometimes it was just a NumPy matrix. But most often they were time series data.

One very interesting property of time series data is that consecutive values in a time series are often closer to one another. But it does not imply that values that are far-apart are not close to one another. Most contemporary time series compression algorithms take advantage of this fact while coming up with a compression algorithm.

In this post, I will only discuss how to compress time series with floating point numbers instead of generalizing them to integers or timestamp values. While it is easier to compress integer values in a time series using either delta encoding or dictionary encoding, these strategies do not work with floating points because, difference of 2 64-bit floats is still a 64-bit float while a differences between 2 64-bit ints can also be a 8-bit int if the values are close to one another. Or, if there are very few unique integer values one can also use dictionary encoding. On the other hand, floats are "infinite".

Most common floating point compression algorithm assumes the fact that floats that are closer to one another in terms of absolute values, are also closer to one another in terms of their binary representations. While this might hold true in certain cases but in general this is not true. But anyways, our focus for this post will be on XOR compression.

But first lets understand how floating point numbers are represented in binary. We will take the example of 64-bit floats as these was the most commonly used format in our work.

The 1st bit is the sign bit 0 for +ve and 1 for -ve.
The next 11 bits represent the exponent.
The remaining 52 bits are used to represent the number called the significand or mantissa.

For e.g. if **N=316.9845**

1. Represent the number in binary. For e.g. 316 is represented in binary as 100111100. The fraction 0.9845 can be converted to binary by repeatedly multiplying by 2 and taking the integer part out.<br/><br/>
   ```
   0.9845 * 2 = 1.9690 - Take out 1
   0.9690 * 2 = 1.9380 - Take out 1
   0.9380 * 2 = 1.8760 - Take out 1
   ....and so on.
   ```
   Finally we get the binary for 0.9845 as 11111100000010000011000100100110111010010111100011011<br/><br/>
Representing 316.9845 in binary, we have:<br/><br/>
X=100111100.11111100000010000011000100100110111010010111100011011<br/><br/>

2. Next, write the representation in 1.xxxxx format i.e. <br/>
X=1.0011110011111100000010000011000100100110111010010111100011011 * 2^8<br/><br/>

   For 64 bits, the power of 2, is usually written as 1023+p. Here p=8, thus we have<br/><br/>
X=1.0011110011111100000010000011000100100110111010010111100011011 * 2^(1031-1023)<br/><br/>

3. The binary representation of the exponent 1031 is 10000000111. The final representation is the concatenation of the following:<br/><br/>
   a. 0 bit because the number is +ve<br/><br/>
   b. Binary representation of 1031 (the exponent) : 10000000111<br/><br/>
   c. The first 52 bits after the decimal point in the above representation i.e. 0011110011111100000010000011000100100110111010010111<br/><br/>

   The final representation is : 0100000001110011110011111100000010000011000100100110111010010111<br/><br/>
Since we have taken 52 relevant bits out of 61 bits, the final representation is rounded:
0100000001110011110011111100000010000011000100100110111010011000<br/><br/>

In Python, we can get the 64 bit representation from a float as follows:<br/>
```python
import bitstring
def float_2_bin(x):
  f = bitstring.BitArray(float=x, length=64)
  return str(f.bin)
```
<br/><br/>
Similarly, we can obtain the `floating point value from a 64-bit representation` as follows:
For e.g. given the following pattern:<br/>

**X = 0100000001011110110001111101111100111011011001000101101000011101**

1. The 1st bit is 0 indicating it is a +ve number.
2. The next 11 bits are : 1000000010 represents 1029. Subtracting off 1023 gives us 6. Thus the exponent is 2^6.
3. The remaining bits are 1110110001111101111100111011011001000101101000011101. Thus the decimal  format is (adding 1. before it): 1.1110110001111101111100111011011001000101101000011101<br/><br/>

Thus we have the number:<br/>
X = 1.1110110001111101111100111011011001000101101000011101 * 2^6, or<br/>
X = 1111011.0001111101111100111011011001000101101000011101.<br/><br/>

The integer part is 1111011 or 123.<br/>
The fractional part is 0001111101111100111011011001000101101000011101.<br/>

To get the base-10 of the fractional part, for each bit at position i we multiply it by 2^(-(i+1)). We get the base-10 of the fractional part as 0.12300000000000466.<br/>
```python
def get_integer_from_binary_fraction(s):
  c = 0
  for i in range(len(s)):
      c += 2**(-(i+1)) if s[i] == '1' else 0
  return c

get_integer_from_binary_fraction('0001111101111100111011011001000101101000011101')
# 0.12300000000000466
```
<br/>
Thus, the final value is 123.123 (rounded).<br/>

In Python, one can obtain the floating point value from a 64-bit representation as follows:
```python
import struct
def bin_to_float(binary):
    return struct.unpack('d', struct.pack('Q', int(f'0b{binary}', 0)))[0]
```
<br/>
