---
layout: post
title:  "Graphs in the Matrix world"
date:   2025-10-10 18:50:11 +0530
categories: software-engineering
---
Working with both graphs and matrices made me realize that a lot of problems in graphs can also be solved using matrices (and a bit of linear algebra).<br/><br/>
Although I have not explored all possible graph problems but few commonly used such as searching, connected components, shortest path, detect cycles, number of paths, topological sorting etc. This post is meant to be a fun exercise and not meant for replacing graph algorithms with matrix based algorithms as it will be evident later that most matrix based approaches are less efficient than existing graph algorithms which we will also explore and understand why.<br/><br/>
But matrix operations can be parallelized using either `SIMD` on CPU or `CUDA` on GPU. Even without SIMD or CUDA one can also use multi-threading in C++ using either `TBB` or `OpenMP` libraries. Moreover a lot of matrix algebra operations are optimized in the BLAS library available for C (`cblas`) and CUDA (`cuBLAS`) both.<br/><br/>
Another advantage of matrix based approaches is that they are useful for `single write multiple reads` i.e. if you are updating the graph less often or only once but reading several times, then the matrix based approaches is expensive only during writes but reads are highly optimized `O(1)`.<br/><br/>
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
To work with matrices we will mostly use the sparse matrix format especially `csr_matrix` from scipy. The reason for using sparse matrices is that graphs can be represented using adjacency matrix which is a sparse matrix i.e. each node has only a few edges to other nodes. There would be exceptions to this but most real world graph problems will have similar sparsity. To create a sparse matrix from the list of edges created above:<br/><br/>
```python
import numpy as np
from scipy.sparse import csr_matrix

mat = [[0]*n for _ in range(n)]

for i, j in graph:
    mat[i][j] = 1
    # For undirected graphs uncomment the following 
    # mat[j][i] = 1

# full dense matrix
a = np.array(mat, dtype=np.uint64)

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
## Graph Search
Search if there is a path from a source node to a destination node.<br/><br/>
This can be solved easily using breadth first search approach as shown below. Time complexity of the approach is O(n + e) where n is the total number of nodes and e is the total number of edges because we are exploring each node once and then all edges out of that node to get the adjacent nodes:<br/><br/>
```python
import collections

def search_graph(adj, n, src, dst):
    queue = collections.deque([src])
    visited = [0]*n
    visited[src] = 1

    while len(queue) > 0:
        node = queue.popleft()
        # found destination
        if node == dst:
            return True

        for b in adj[node]:
            # visit each node once
            if visited[b] == 0:
                queue.append(b)
                visited[b] = 1
    
    return False
```
<br/><br/>
To solve the same problem but using only matrices and matrix operations can be done in one way shown below. Note that the methods shown here may not be unique and possibly different approaches can be used to solve the same problem:<br/><br/>
```python
def search_matrix(a, n, src, dst):
    # a is a dense numpy array
    a = csr_matrix(a)
    b = csr_matrix(a[src:src+1])

    for _ in range(n):
        if b[0,dst] != 0:
            return True
        # b[i,j] > 0 implies that there is a path of length k from i to j
        b = b.dot(a)

    return False
```
<br/><br/>
The algorithm works as follows. The matrix `a` is the adjacency matrix in csr_matrix format. Thus the entry `a[i,j]` represents presence (`a[i,j]` = 1) or absence (`a[i,j]` = 0) of an edge for our problems since we have assumed that it is an unweighted graph. If it was a weighted graph then `a[i,j]` would have represented the weight of the edge `i->j`.<br/><br/>
If we compute the square of matrix a i.e. `a^2`, it represents the 2nd degree edges i.e. if `u=a^2` then `u[i,j] > 0` implies that there is a path of length 2 from i to j. In general if `u=a^k` and `u[i,j] > 0` implies that there is a path of length k from i to j. When `a` is a binary matrix then we will see that `u[i,j]` equals the number of paths of length k from i to j.<br/><br/>
In the above code, instead of exponentiating `a` again and again we are only multiplying the row vector corresponding to the `src` node with the `a` matrix since we are only concerned about the path from `src` to `dst`. The operation `b = b.dot(a)` represents exponentiation of b i.e. `b^2`, `b^3` and so on where `u=b^k` represents whether there is a path of length k from `src` to all other nodes.<br/><br/>
Note that we need to run the exponentiation from `k=0 to n-1` because the path lengths in a graph can range from 0 to n-1 (Imagine a linked list like graph where the distance from one end to the other end is n-1).<br/><br/>
The time complexity of the above code is `O(n^3)` in the worst case because the dot product is `O(n^2)` in the worst case (dense adjacency matrix). Note that realistically we might never hit the worst case because if the graph is linear like a linked list then the dot product using sparse matrix operations is O(1) and total time complexity is O(n).<br/><br/> 
On the other hand if the graph is fully connected `src` and `dst` are directly connected and thus we exit before any dot product. For cases somewhere in between e.g. `src` and `dst` are length k apart and each node is connected to m nodes on average where m << n, time complexity would be `O(k*m^2)`. Note that k and m are orthogonal i.e. higher m implies lower k and vice versa.<br/><br/>
This is also verified experimentally where we saw that for n=1000 and p=0.3, `search_matrix` was about 10x faster on average than `search_graph`. The gap reduces when p is smaller implying that the path length between `src` and `dst` increases and vice versa.<br/><br/>
Note that instead of exponentiation as we are doing above, another approach exists. The adjacency matrix `a` can be diagonalized as `a=P.D.P^-1` where columns of P are the eigenvectors of `a` and D is a diagonal matrix with the eigenvalues of `a` along the diagonal and `P^-1` is the inverse of P. Thus `a^2=P.D.P^-1.P.D.P^-1=P.D^2.P^-1`. In general we can write `a^k=P.D^k.P^-1`.<br/><br/>
```python
def search_eig(a, n, src, dst):
    b = np.copy(a)

    d, p = np.linalg.eig(b)
    d = np.diag(d)
    p_inv = np.linalg.inv(p)
    # b = p @ d @ p_inv
    f = d

    for _ in range(n):
        if b[src,dst] != 0:
            return True

        # f = d^k
        f *= d
        # To handle complex numbers we can either take absolute value or take real part
        b = np.abs(p @ f @ p_inv)

        # Truncate very small values to 0 as these are mostly precision errors
        b[b < 1e-10] = 0

    return False
```
<br/><br/>
Note that diagonalization is only possible when the matrix is full rank i.e. rank = n. In many real world graphs, the adjacency matrix is low rank as there could be nodes with no edges or cliques or multiple connected components which can reduce the rank of the matrix. Time complexity of eigenvalue computation as well as inverse computation is `O(n^3)` thus the overall time complexity remains unchanged.<br/><br/>
The eigenvalue approach is useful when we want to run the search for multiple pairs of source and destination nodes on the same graph. We need to compute the matrices p, d and p_inv only once and also run the for loop only once and compute b and use it for any pair of source and destination in `O(1)` time complexity. On the other hand the BFS or DFS based approach is not useful for repeated queries as it will take same O(n + e) time complexity for any pair of source and destination nodes.<br/><br/>
## Connected Components in Undirected Graph
Given an undirected graph, calculate the number of connected components.<br/><br/>
This can be solved using breadth first search and union find approach as shown below. Time complexity of the approach shown below is O(n + e) where n is the total number of nodes and e is the number of edges:<br/><br/>
```python
def num_components_graph(adj, n):
    visited = [0]*n
    ncomp = 0
    for i in range(n):
        if visited[i] == 0:
            ncomp += 1
            # BFS or Union Find
            queue = collections.deque([i])
            while len(queue) > 0:
                node = queue.popleft()
                for j in adj[node]:
                    if visited[j] == 0:
                        queue.append(j)
                        visited[j] = 1
    
    return ncomp
```
<br/><br/>
The same problem can be solved using matrix operations as follows:
```python
def num_components_matrix(a, n):
    a = csr_matrix(a, dtype=np.uint64)
    b = csr_matrix(a, dtype=np.uint64)
    c = csr_matrix(b, dtype=np.uint64)

    for _ in range(n):
        # b[i,j] > 0 implies a path of length k exists from i to j
        b = b.dot(a)

        # c[i,j] > 0 implies a path exists from i to j
        c += b
    
    c[c > 0] = 1
    return np.linalg.matrix_rank(c.toarray())
```
<br/><br/>
Note that we are using another matrix `c` to sum the results of b. This is because `b = b.dot(a)` in the k-th iteration of the loop finds whether there is a path of length k between any two nodes. Which implies that if there is a path between 2 nodes i and j of length k-1 but there is no path between them of length k, then `b[i,j] = 0` in the k-th iteration. Thus to aggergate the presence of path across all path lengths, we are using the matrix `c`. `c[i,j] > 0` implies that there is at-least one path from i to j of any length.<br/><br/>
To get the number of connected components using matrix `c` one can use various strategies. If you observe the final matrix `c` then for a connected component with m nodes `[n1, n2, ... nm]`, `c[ni,nj] = 1` where ni and nj corresponds to the nodes in the component and if there is another component of p nodes `[r1, r2, ... rp]` then `c[ri,rj] = 1` but `c[ni,rj] = 0` and `c[ri,nj] = 0` i.e. between any two nodes across components the value is 0 in the matrix `c`. With a csr_matrix format one can find the number of components from this observation as follows:
```python
def num_comps(a, n):
    visited = [0]*n
    c = 0
    for i in range(len(a.indptr)-1):
        s = a.indptr[i]
        e = a.indptr[i+1] if i+1 < len(a.indptr) else n
        flag = False
        for k in range(s, e):
            j = a.indices[k]
            if visited[j] == 0:
                flag = True
                visited[j] = 1
            else:
                break
        if flag:
            c += 1
    
    return c
```
<br/><br/>
Time complexity is total number of non-zero entries in the matrix a which is usually << n for sparse matrices. But note that matrix c above will be much less sparse as compared to the input matrix because in the input matrix only non-zero entries are the direct edges but in matrix c the non-zero entries are almost all cells for a single connected component.<br/><br/>
A more `matrix` way of doing things which is not specific to sparse format is using matrix rank.<br/><br/>
Since we have seen above that for nodes in the same component have the same row values in the matrix `c` (after setting all non-zero values to 1) and nodes from different components are disjoint or orthogonal, thus if we can calculate the number of linearly independent rows in the matrix `c` we will get the number of connected components and the number of linearly independent rows in the matrix can be computed from the `rank` of the matrix.<br/><br/>
Time complexity of the above `num_components_matrix` code is `O(n^3)`. Unlike search where it was unlikely to observe the worst case time complexity, for number of components problem this is not the case as for most cases we are going to observe `O(n^3)` running times which makes this approach computationally much more expensive than a standard union find operation.<br/><br/>
## Number of Paths from Source to Destination In DAG
Given a directed acyclic graph (DAG), calculate the number of paths from source node to destination node.<br/><br/>
This can be solved using recursion as shown below. Time complexity of the approach shown below is O(n + e) where n is the total number of nodes and e is the number of edges:<br/><br/>
```python
def num_paths_graph_recurse(adj, n, src, dst, dp):
    if src == dst:
        return 1
    
    if dp[src] != -1:
        return dp[src]
    
    paths = 0
    for j in adj[src]:
        paths += num_paths_graph_recurse(adj, n, j, dst, dp)
    
    dp[src] = paths
    return dp[src]

def num_paths_graph(adj, n, src, dst):
    dp = [-1]*n
    return num_paths_graph_recurse(adj, n, src, dst, dp)
```
<br/><br/>
Non-recursive algorithm using BFS:<br/><br/>
```python
def num_paths_graph(adj, n, src, dst):
    queue = collections.deque([src])
    dp = [0]*n

    while len(queue) > 0:
        node = queue.popleft()
        dp[node] += 1

        for b in adj[node]:
            queue.append(b)
    
    return dp[dst]
```
<br/><br/>
The same problem can be solved using matrix operations as we have seen earlier in the following manner:<br/><br/>
```python
def num_paths_matrix(a, n, src, dst):
    np.fill_diagonal(a, 0)

    a = csr_matrix(a)
    b = csr_matrix(a[src:src+1])
    out = csr_matrix(b)

    for _ in range(n):
        b = b.dot(a)
        out += b

    return out[0,dst]
```
<br/><br/>
In the above code input `a` is a dense matrix but transformed into sparse format but before that we set the diagonal elements to 0 since we are assuming that it is a DAG and there are no cycles. Non-zero diagonal element indicates there is path to self. Time complexity of the above code is `O(n^3)` as seen before also.<br/><br/>
## Single Source Shortest Path in Unweighted DAG
Given a directed acyclic graph (DAG), calculate the distance in terms of number of edges from a source node to all other nodes.<br/><br/>
This can be solved simply using BFS with a time complexity of O(n + e) where n is the total number of nodes and e is the number of edges. Not showing the code here as it is similar to BFS traversal shown above. For solving the shortest path problem using matrix operations, we can implement as shown below:<br/><br/>
```python
def single_source_shortest_dist_unweighted(a, n, src):
    a[a == 0] = n+1
    np.fill_diagonal(a, 0)

    a = csr_matrix(a)
    b = csr_matrix(a[src:src+1])

    for _ in range(n):
        # c[i,j] is the minimum distance from node i to node j for all nodes j within a distance k from node i  
        c = dist_mat_mul(b, a, 1, n, n)
        # b[i,j] is the minimum distance from node i to node j
        b = b.minimum(c)

    return b.toarray()
```
<br/><br/>
Again since this is a DAG we set the diagonal elements in the adjacency matrix to 0. Also since we are interested in minimum distance we set all non-zero values in the adjacency matrix to n+1 (indicating no path yet). For each path length k we find the distance from source to all other nodes. Here the distance is implemented using the function `dist_mat_mul` which is nothing but matrix product where the multiplication is replaced with addition and addition with minimum operator. For dense matrices `a` and `b` the `dist_mat_mul` method would look something like below:
```python
def dist_mat_mul(a, b, n, m, p):
    c = np.zeros((n,p), dtype=np.uint64)

    for i in range(n):
        for j in range(p):
            s = m+1
            for k in range(m):
                # In standard matrix dot product it would have been s += a[i,k]*b[k,j]
                s = minimum(s, a[i,k] + b[k,j])
            c[i,j] = s

    return c
```
<br/><br/>
For sparse matrices, a memory efficient sparse implementation is a bit involved and can be implemented as shown below:
```python
def dist_mat_mul(a, b, n, m, p):
    cdata = []
    cindices = []
    cindptr = [0]

    res = []
    for a_i in range(len(a.indptr)-1):
        i = a_i
        s_a = a.indptr[i]
        e_a = a.indptr[i+1] if i+1 < len(a.indptr) else n

        for f in range(s_a, e_a):
            k = a.indices[f]
            u = a.data[f]

            s_b = b.indptr[k]
            e_b = b.indptr[k+1] if k+1 < len(b.indptr) else m

            for q in range(s_b, e_b):
                j = b.indices[q]
                v = b.data[q]

                if u > 0 and v > 0:
                    res += [(int(i), int(j), int(u + v))]
    
    res = sorted(res, key=lambda k: (k[0], k[1]))

    curr = (-1, -1)
    curr_i = -1
    curr_j = -1
    curr_v = 1e300
    h = 0

    for i, j, v in res:
        if (i, j) != curr:
            if i > curr_i and curr_i != -1:
                if curr_j != -1:
                    cindices += [curr_j]
                    cdata += [curr_v]
                    h += 1
                    
                cindptr += [h]

            if j > curr_j and curr_j != -1:
                cindices += [curr_j]
                cdata += [curr_v]
                h += 1

            curr = (i, j)
            curr_i = i
            curr_j = j
            curr_v = v
        else:
            curr_v = min(curr_v, v)
        
    if curr_j != -1:
        cindices += [curr_j]
        cdata += [curr_v]
        h += 1
    
    cindptr += [h]
    cindptr += [h]*(n+1-len(cindptr))
    
    return csr_matrix((cdata, cindices, cindptr), shape=(n,p), dtype=np.uint64)

```
<br/><br/>
Time complexity of the sparse distance implementation is `O(G*H*log(GH)` where G is the total number of non-zero elements in input sparse matrix a and H is the average number of non-zero elements in each row of input sparse matrix b. In the worst case when it is a dense matrix, time complexity is `O(n^3*log(n))` which is worse than dense matrix multiplication. For high sparsity, the time complexity is usually much smaller of the order of `O(n^2*log(n))`.<br/><br/>
Time complexity of the `single_source_shortest_dist_unweighted` method is O(n^3). For weighted graphs there won't be any changes to the above code although for the standard graph algorithm we have to use either djikstra or bellman-ford rather than simple BFS. The time complexity of which is greater than O(n + e).<br/><br/>
<br/><br/>
## Detect presence of cycle in a directed graph
Given a directed graph, return True if there is a cycle present else return False.<br/><br/>
This can be solved using DFS or recursion as shown below. with a time complexity of O(n + e) where n is the total number of nodes and e is the number of edges:<br/><br/>
```python
def dfs_cycle(adj, n, i, visited):
    for j in adj[i]:
        if visited[j] == 0:
            visited[j] = 1
            h = dfs_cycle(adj, n, j, visited)
            visited[j] = 0
            if h:
                return True
        else:
            return True
        
    return False

def has_cycle_graph(adj, n):
    visited = [0]*n
    for i in range(n):
        visited[i] = 1
        h = dfs_cycle(adj, n, i, visited)
        visited[i] = 0
        if h:
            return True
        
    return False
```
<br/><br/>
The corresponding matrix solution is again a very much similar operation as seen above for some other problems:<br/><br/>
```python
def has_cycle(a, n):
    a = csr_matrix(a)
    b = csr_matrix(a)

    for _ in range(n):
        b = b.dot(a)
        # There is a path of length k from a node to itself
        if b.diagonal().max() > 0:
            return True

    return False
```
<br/><br/>
If there is a cycle in the graph, at-least one diagonal element will have a non-zero value i.e. number of paths to self is non-zero in case a cycle is present. Time complexity of this approach is `O(n^4)`. The time complexity can be brought down to `O(n^3)` using an approach which is a variant of the all pairs shortest path algorithm is shown in the next problem.<br/><br/>
One can also use the matrix diagonalization technique introduced above. But one needs to be careful while using this technique as it only works when the matrix is full rank i.e rank=n. Most real world matrices representing graphs will be low rank because there can multiple components or nodes with no edges etc. In case matrix diagonalization is possible time complexity can be brought down to `O(n^3)`.<br/><br/>
```python
def has_cycle(a, n):
    d, p = np.linalg.eig(a)
    d = np.diag(d)
    p_inv = np.linalg.inv(p)
    # a = p @ d @ p_inv
    f = d

    for _ in range(n):
        # f = d^k
        f *= d
        # Convert complex values to real values either by taking real part or taking the absolute value
        g = np.abs(p @ f @ p_inv)

        # Truncate very small values to 0
        g[g < 1e-10] = 0
        if g.diagonal().max() > 0:
            return True

    return False
```
<br/><br/>
In each iteration we are exponentiating the diagonal matrix. With an optimized implementation `p @ f @ p_inv` should be `O(n^2)` since f is a diagonal matrix. The overall time complexity thus would be `O(n^3)`.<br/><br/>
<br/><br/>
## Toplogical Sorting
Given a directed graph, return toplogical sorting of the nodes else return empty list if a cycle exists.<br/><br/>
Standard approach for solving toplogical sorting problems with an adjacency list. Time complexity of the below algorithm is O(n + e):<br/><br/>
```python
def topological_sort_graph(adj, n):
    # calculate in degrees of nodes
    in_degs = [0]*n
    for i in range(n):
        for j in adj[i]:
            in_degs[j] += 1

    # the initial nodes in toplogical sorting are the ones where in_deg is 0
    arr = [x for x in range(n) if in_degs[x] == 0]
    res = arr

    # iterate for each level and update in degrees
    while len(arr) > 0:
        new_arr = []
        for i in arr:
            for j in adj[i]:
                in_degs[j] -= 1
                if in_degs[j] == 0:
                    new_arr += [j]

        arr = new_arr[:]
        res += arr

    # detect cycle, if cycle is present in degrees of some nodes will never be 0
    if sum([in_degs[i] > 0 for i in range(n)]) > 0:
        return []

    return res
```
<br/><br/>
We can solve the same problem using matrix operations as shown below:<br/><br/>
```python
def topological_sort_matrix(a, n):
    b = np.copy(a)
    b[b == 0] = -(n+1)
    np.fill_diagonal(b, 0)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                b[i,j] = max(b[i,j], b[i,k] + b[k,j])
                # cycle detected
                if i == j and b[i,j] > 0:
                    return []
        
    max_d = np.max(b, axis=0)
    return np.argsort(max_d).tolist()
```
<br/><br/>
So what we are doing here is that we are finding the maximum distance between each pair of nodes and then the nodes are sorted based on maximum distance from any other node in increasing order. The node which is at a maximum distance from any node should come at the end of the topological sort.<br/><br/>
Time complexity of the `topological_sort_matrix` is `O(n^3)`.<br/><br/>
The cycle detection technique used here can also be used to detect cycles in general as described above.<br/><br/>
The above algorithm is a variant of the `Floyd-Warshall` all pairs shortest path algorithm.<br/><br/>
As seen above, for most problems matrix operations to solve graph problems has higher time complexity as compared to graph algorithms like BFS or DFS and recursion based approaches. But the advantage of matrix operations is that they can be parallelized using either SIMD or CUDA.<br/><br/>
