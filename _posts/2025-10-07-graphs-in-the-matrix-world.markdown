---
layout: post
title:  "Graphs in the Matrix world"
date:   2025-08-29 18:50:11 +0530
categories: software-engineering
---
Working with both graphs and matrices made me realize that a lot of problems in graphs can also be solved using matrices (and a bit of linear algebra).<br/><br/>
Although I have not explored all possible graph problems but few commonly used such as searching, connected components, shortest path, detect cycles, number of paths, topological sorting etc. This post is meant to be a fun exercise and not meant for replacing graph algorithms with matrix based algorithms as it will be evident later that most matrix based approaches are less efficient than existing graph algorithms which we will also explore and understand why.<br/><br/>
To work with matrix based algorithms one can create random graphs using the `networkx` package in python as shown below:<br/><br/>
```python
import networkx as nx

# Build a random directed graph with 100 nodes and add an edge between 2 nodes with a probability of 0.05
# graph is a list of edges [(x,y),...]
n = 100
p = 0.05
graph = nx.gnp_random_graph(n, p, directed=True)
graph = graph.edges()
```
<br/><br/>
For all problems shown below, we will assume an unweighted graph i.e. all edge weights are 1.<br/><br/>
To work with matrices we will assume sparse matrix format especially `csr_matrix` from scipy. The reason for using sparse matrices is that most often graphs can be represented using adjacency matrix which is a sparse matrix i.e. each node has only a few edges to other nodes. There would be exceptions to this but most practical graph problems will have similar sparsity structures. To create a sparse matrix from the list of edges created above:<br/><br/>
```python
import numpy as np
from scipy.sparse import csr_matrix

mat = [[0]*n for _ in range(n)]

for i, j in graph:
    mat[i][j] = 1
    # For undirected graphs uncomment the following 
    # mat[j][i] = 1

# full dense matrix
a = np.array(mat)

# sparse csr_matrix
a = csr_matrix(a, dtype=np.uint64)
```
<br/><br/>
While working with graph algorithms, we would want to represent the graph as an adjacency list or matrix. Adjacency list is the sparse version of adjacency matrix.<br/><br/>
```python
adj = {i:[] for i in range(n)}

for i, j in graph:
    adj[i] += [j]
    # For undirected graphs uncomment the following 
    # adj[j] += [i]
```
<br/><br/>

**Graph Search**<br/>
Search if there is a path from a source node to a destination node.<br/><br/>
This can be solved easily using breadth first search approach as shown below. Time complexity of the approach shown below is O(n) where n is the total number of nodes:<br/><br/>
```python
import collections

def search_graph(adj, n, src, dst):
    queue = collections.deque([src])
    visited = [0]*n
    visited[src] = 1

    while len(queue) > 0:
        node = queue.popleft()
        if node == dst:
            return True

        for b in adj[node]:
            if visited[b] == 0:
                queue.append(b)
                visited[b] = 1
    
    return False
```
<br/><br/>
To solve the same problem but using only matrices and matrix operations can be done in one way shown below. Remember that the input matrix `a` is a sparse matrix in csr format:<br/><br/>
```python
def search_matrix(a, n, src, dst):
    b = csr_matrix(a[src:src+1])

    for _ in range(n):
        if b[0,dst] != 0:
            return True
        b = b.dot(a)

    return False
```
<br/><br/>
The algorithm works as follows. The matrix `b` is a copy of the original adjacency matrix `a` in csr_matrix format. Thus the entry `b[i,j]` represents presence (`b[i,j] = 1`) or absence (`b[i,j] = 0`) of an edge for our problems since we have assumed that it is an unweighted graph. If it was a weighted graph then `b[i,j]` would have represented the weight of the edge `a->b`.<br/><br/>
The operation `b = b.dot(a)` represents exponentiation of b i.e. `b^2`, `b^3` and so on.<br/><br/>




