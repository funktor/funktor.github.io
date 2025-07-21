---
layout: post
title:  "Why Attention needs the three musketeers - Query, Key and Value"
date:   2025-07-22 18:50:11 +0530
categories: ml
---
In the landmark paper "Attention is all you need" we were introduced to the Transformer architecture which forever "transformed" how large language models are trained using deep neural networks. The "near-human" level performance of LLMs is possible due to terabytes and petabytes of data used in training which in turn is made possible due to the shift from RNN/LSTM to the Attention achitecture.<br/><br/>
RNN and its variants such as LSTM and GRU are sequential in nature i.e. for a given sequence of length L > 1, the hidden representation for the token (or word) at index i > 0 is dependent on the representations and some contexts learned from the tokens at indices 0 to i-1. Since each token at index i already "encodes" information from 0 to i-1, thus for calculating the representation for token at index i+1, we can only use the representation from index i instead of all previous indices.<br/><br/> 
Thus we see that it is not possible to calculate the representation for token say 10 without calculating the representation for token 5.<br/><br/>
On the other hand, Attention mechanism allows one to calculate the representation for token 10 before calculating the representation for token 5 and thus allowing the representations to be computed in parallel. This is in turn is made possible due to the 3 different vectors for each token namely, Query, Key and Value.<br/><br/>
The idea is to calculate the representation R(i) for the token at index i as a function of some vectors V(j) corresponding to each token j where the vectors V(j) does not get updated in the same forward pass. The simplest mechanism to calculate R(i) is to use a weighted sum of the V(j)'s i.e.<br/><br/>
  ```
  R(i,t) = w(0,t-1) * V(0,t-1) + w(1,t-1) * V(1,t-1) + ... + w(n-1,t-1) * V(n-1,t-1)
  ```
  <br/><br/>
The index 't' denotes the epoch number. The above equation denotes that the repreentation for current epoch t and token index i is dependent only on weights w and vectors V from the previous epoch t-1.<br/><br/>
The weights w(j) should be some function of the current token index i, else if w(j) was independent of i in the above equation, then for all indices i, R(i) would be same. One possible mechanism is to have another vector Q(j) corresponding to each token index j and then w(j) = <Q(i), Q(j)> i.e. dot product of Q(i) and Q(j). Thus,<br/><br/>
  ```
  R(i,t) = <Q(i, t-1), Q(0, t-1)> * V(0,t-1) + <Q(i, t-1), Q(1, t-1)> * V(1,t-1) + ... + <Q(i, t-1), Q(n-1, t-1)> * V(n-1,t-1)
  ```
  <br/><br/>
But note that the dot product <Q(i), Q(j)> is symmetric w.r.t i and j i.e.<br/><br/>
  ```
  <Q(i), Q(j)> = <Q(j), Q(i)>
  ```
  <br/><br/>
The weights may not be symmetric in a sequence of tokens. For e.g. for the sequence "My phone has a RAM of 6GB", the score of the token "phone" w.r.t. the token "RAM" need not be same as the score of the token "RAM" w.r.t. "phone" because "RAM" could also score highly with some token like "computer" or "laptop" etc. elsewhere in the input sequence. Thus, there is no need for the dot product to be symmetric.<br/><br/>
Thus, comes another vector K(j) corresponding to each token index j and the dot product is calculated as: <Q(i), K(j))>. Thus,<br/><br/>
  ```
  R(i,t) = <Q(i, t-1), K(0, t-1)> * V(0,t-1) + <Q(i, t-1), K(1, t-1)> * V(1,t-1) + ... + <Q(i, t-1), K(n-1, t-1)> * V(n-1,t-1)
  ```
  <br/><br/>
The dot products <Q(i), K(j))> are not symmetric i.e.
  ```
  <Q(i), K(j)> != <Q(j), K(i)>
  ```
  <br/><br/>
