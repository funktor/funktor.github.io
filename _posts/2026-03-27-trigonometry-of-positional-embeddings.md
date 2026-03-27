---
layout: post
title:  "Trigonometry of Positional Embeddings"
date:   2026-03-29 18:50:11 +0530
categories: math
---
It is well known fact that `Transformer` models uses `positional embeddings` to encode positional information during the attention calculation for each pair of token embeddings. Prior to Transformers, we had `RNN` and `LSTM` which intrinsically handled the sequence information. But with Transformers and `attention` mechanism, standard dot product between pairs of token embeddings do not maintain the position information. For e.g. given a sentence such as "The apple fell from the tree on my apple macbook". The token embedding for "apple" at position 2 is the same as the token embedding at position 9 but they imply different objects given the sequence of words. Without positional encodings, attention mechanism will give the same dot product (query and key) for position 2 and 9 when we compute `Q.K^T`.
<br/><br/>
But how does one design "good" positional embeddings that meets the following conditions:
<br/><br/>
1. Should be able to handle arbitrary sequence lengths. Not just the maximum sequence length encoutered during training.
2. Same token at different positions should have different positional embeddings.
3. For any two positions `p` and `q` in the sequence, if `f` is the function for computing the positional embedding and `k` is some distance, then `|f(p+k)-f(p)| = |f(q+k)-f(q)|`
<br/><br/>

The last condition implies that the absolute distance of the positional embeddings between 2 positions separated by distance of `k` should remain same irrespective of which position in the sequence we are looking at. This is important because it implies that words or tokens are only affected by its neighboring words.
<br/><br/>
The last condition implies that the function `f` should be some form of `rotation` or `linear transformation`. For e.g. given the linear transformation `f(x) = ax + b`, we see that it satisfies the last condition where `a` and `b` can be learnable parameters. For Transformer use case, `x` must be a tensor of shape `(batch, seq_len, emb_dim)`. Thus the learnable parameter `a` must be a weight matrix of shape `(seq_len, seq_len)`. But note that this formulation violates condition 1 above because if `a` is of shape `(100, 100)` for a sequence length of 100 encoutered during training then we cannot use this to compute `f(x)` when x is of shape `(200, 16)` encountered during inference because the last dimension of `a` must match with the 1st dimension of `x` for dot product calculations.
<br/><br/>
The other possibility is `rotation` i.e. `f(x)=e^(iwx)` where `i` is the imaginary square root of unity `i*i=1` and `w` is the frequency of rotation. Expanding using Euler's formula, we can also write it as `e^(iwx) = cos(wx) + i*sin(wx)`. Note that none of the parameters in the equation of rotation depends on the sequence length. Let's prove that this formulation satisfies condition 3 above.
<br/><br/>
![Proof](/docs/assets/lagrida_latex_editor.png)
<br/><br/>
Thus, we see that the absolute distance between the positional embeddings separated by a distance `k` when f is a rotation, is agnostic of the positions `p` or `q`, which implies that:
<br/><br/>
`|f(p+k)-f(p)| = |f(q+k)-f(q)|` for any `p` and `q`
<br/><br/>
Now, let us represent the positional embedding vector of dimension `d` for position `p` as follows:
<br/><br/>
```
PE(pos) = [e^(i*w(pos, 0)*0), e^(i*w(pos, 1)*1), ... e^(i*w(pos, d-1)*(d-1)]
```
<br/><br/>
where `w(pos, j)` is the frequency of rotation for `pos`-th token and `j`-th value in the d-dimensional embedding i.e. each dimension of the positional embedding represents a different rotation.
<br/><br/>
But now we have another problem, the positional embeddings are all a bunch of complex numbers. If we take only the real part out of each rotation as follows i.e.
```
PE(p) = [cos(w(p, 0)*0), cos(w(p, 1)*1), ... cos(w(p, d-1)*(d-1)]
```
Then if we take something like:
```
PE(p+k) = [cos(w(p+k, 0)*0), cos(w(p+k, 1)*1), ... cos(w(p+k, d-1)*(d-1)]
```
and take the distance between these 2 vectors as the L2 norm:
```
|PE(p+k)-PE(p)|^2 = (cos(w(p+k, 0)*0)-cos(w(p, 0)*0))^2 + (cos(w(p+k, 1)*1)-cos(w(p, 1)*1))^2 + ...
```
We cannot reduce the above into an expression that is independent of p.
One way to resolve this is to group the embedding values in groups of 2.
