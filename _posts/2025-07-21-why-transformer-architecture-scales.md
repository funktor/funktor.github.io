---
layout: post
title:  "Why Transformer architecture scales"
date:   2025-07-22 18:50:11 +0530
categories: ml
---
In the landmark paper "Attention is all you need" we were introduced to the Transformer architecture which forever "transformed" how large language models are trained using deep neural networks. The "near-human" level performance of LLMs is possible due to terabytes and petabytes of data used in training which in turn is made possible due to the shift from RNN/LSTM to the Attention achitecture.<br/><br/>
RNN and its variants such as LSTM and GRU are sequential in nature i.e. for a given sequence of length L > 1, the hidden representation for the token (or word) at index i > 0 is dependent on the representations and some contexts learned from the tokens at indices 0 to i-1. Since each token at index i already "encodes" information from 0 to i-1, thus for calculating the representation for token at index i+1, we can only use the representation from index i instead of all previous indices.<br/><br/> 
Thus we see that it is not possible to calculate the representation for token say 10 without calculating the representation for token 5.<br/><br/>
On the other hand, Attention mechanism allows one to calculate the representation for token 10 before calculating the representation for token 5 and thus allowing the representations to be computed in parallel. This is in turn is made possible due to the 3 different vectors for each token namely, Query, Key and Value.<br/><br/>
ri = (qi.q0)u0 + (qi.q1)u1 + ..... + ... + (qi.qj)uj
dL/dqi = dL/dri * dri/dqi + dL/dr0 *
coupled equations
symmetry breaking (DAG a -> b != b -> a)
"The cat ate the fish, it also ate the cookie"
