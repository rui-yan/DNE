import numpy as np
import networkx as nx
import math
from collections import defaultdict

def apply(graph, func, non_edges):
    return ((u, v, func(u, v)) for u, v in non_edges)

def CN_score(graph,u,v):
    return len(list(w for w in graph[u] if w in graph[v] and w not in (u, v)))

def common_neighbors_score(graph): 
    non_edges = nx.non_edges(graph)
    def score(u,v):
        return len(list(w for w in graph[u] if w in graph[v] and w not in (u, v)))

    return ((u, v, score(u, v)) for u, v in non_edges) 

def AA_score(graph, u, v):
    return sum(1 / math.log(graph.degree(w))for w in nx.common_neighbors(graph, u, v))

def adamic_adar_score(graph):
    non_edges = nx.non_edges(graph)
    def score(u, v):
        return sum(1 / math.log(graph.degree(w))for w in nx.common_neighbors(graph, u, v))
    return ((u, v, score(u, v)) for u, v in non_edges) 

def PA_score(graph, u, v):
    return (graph.degree(u) * graph.degree(v))

def preferential_attachment_score(graph):
    non_edges = nx.non_edges(graph)
    return ((u, v, graph.degree(u) * graph.degree(v)) for u, v in non_edges)

def RA_score(graph, u, v):
    return sum(1 / graph.degree(w) for w in nx.common_neighbors(graph, u, v))

def resource_allocation_score(graph):
    non_edges = nx.non_edges(graph)
    def score(u, v):
        return sum(1 / graph.degree(w) for w in nx.common_neighbors(graph, u, v))
    return apply(graph, score,non_edges) 

def JC_score(graph, u, v):
    union_size = len(set(graph[u]) | set(graph[v]))
    if union_size == 0:
        return 0
    else:
        return len(list(nx.common_neighbors(graph, u, v))) / union_size

def jaccard_coefficient_score(graph):
    non_edges = nx.non_edges(graph)
    def score(u,v):
        union_size = len(set(graph[u]) | set(graph[v]))
        if union_size == 0:
            return 0
        else:
            return len(list(nx.common_neighbors(graph, u, v))) / union_size

    return apply(graph,score,non_edges)    

def friends_measure(graph, u, v):  #heuristic for katz
    score = 0
    for x in graph[u]:
        for y in graph[v]:
            if x == y:
                score = score + 1
            elif graph.has_edge(x,y):
                score = score + 1
    return score

def rpr_matrix(graph, alpha=0.85):
    D = graph.to_directed()
    H = nx.stochastic_graph(D)
    H = nx.to_numpy_matrix(H).transpose()
    I = np.eye(H.shape[0])
    S = alpha*np.linalg.inv(I - (1-alpha)*H)
    return S

def katz_score(graph,beta=0.004):
    # non_edges = nx.non_edges(graph)
    A = nx.to_numpy_matrix(graph)
    # print(A)
    w, v = np.linalg.eigh(A)
    lambda1 = max([abs(x) for x in w])   # beta should be less than 1/lambda1
    # print(1/lambda1)
    if beta >= 1/lambda1 :
        raise ValueError('beta should be less than 1/lambda, lambda being the eigenvalue with largest magnitude')
    I = np.eye(A.shape[0])
    S = np.linalg.inv(I - beta * A) - I
    return S

def RP_score(graph, u, v):
    return rooted_pagerank_score(graph,u)[v] + rooted_pagerank_score(graph,v)[u]

def rooted_pagerank(graph, root_node, alpha = 0.85, max_iter = 100):
    if not graph.is_directed():
        D = graph.to_directed()
    else:
         D = graph
    H = nx.stochastic_graph(D)
    n = len(graph.nodes())
    H = nx.to_numpy_matrix(H).transpose()
    x = np.full((n,1),1.0 / n)
    v = np.full((n,1),0.0)
    nodes = D.nodes()
    index = 0
    for node in nodes:
        # print(node)
        # print(root_node)
        # print(index)
        if node == root_node:
            # print(1-alpha)
            v[index][0] = 1 - alpha
            break
        index = index +  1
    # print(v)

    for _ in range(max_iter):
        x = np.add(alpha*np.dot(H,x),v) # x = alpha*H*x + (1-alpha)*v

    x = np.squeeze(np.asarray(x))
    x = x /	np.sum(x)
    return dict(zip(graph.nodes(), x))

def rooted_pagerank_score(G, root, alpha=0.85):
    personalization = dict.fromkeys(G, 0)
    personalization[root] = 1

    return nx.pagerank(G, alpha, personalization)


def rpr_dict(G, k=None, alpha=0.85):  #rpr node pair values
    res = defaultdict(float)
    nbunch = G.nodes()
    for u in list(nbunch):
        if k is not None:
            # Restrict to the k-neighbourhood subgraph
            G = nx.ego_graph(G, u, radius=k)
        pagerank_scores = rooted_pagerank_score(G, u, alpha)
        for v, w in pagerank_scores.items():
            if w > 0 and u != v:
                res[(u, v)] += w
    return res

def list_rpr_scores(G, k=None, alpha=0.85):  # updated 
    non_edges = nx.non_edges(G)
    rpr_values = rpr_dict(G,k,alpha)
    def score(u,v):
        return rpr_values[(u,v)] + rpr_values[(v,u)]
    
    return apply(G,score,non_edges)

def katz_dict(G, beta = 0.001, max_power=5):
    nodelist = list(G.nodes) 
    adj = nx.to_scipy_sparse_matrix(G)
    res = defaultdict(float)
    # http://stackoverflow.com/questions/4319014/
    for k in range(1, max_power + 1):
        matrix = (adj ** k).tocoo()
        for i, j, d in zip(matrix.row, matrix.col, matrix.data):
            if i == j:
                continue
            u, v = nodelist[i], nodelist[j]
            w = d * (beta ** k)
            res[(u, v)] += w
    if not G.is_directed():
            for pair in res:
                res[pair] /= 2

    return res

def list_katz_scores(G, beta=0.001, max_power=5):  #updated for limited path length
    non_edges = nx.non_edges(G)
    katz_values = katz_dict(G, beta, max_power)
    def score(u,v):
        return katz_values[(u,v)] + katz_values[(v,u)]

    return apply(G,score,non_edges)

def simrank(G, nodelist=None, c=0.8, num_iterations=10):
    n = len(G)
    M = raw_google_matrix(G, nodelist=nodelist)
    sim = np.identity(n, dtype=np.float32)
    for i in range(num_iterations):
        temp = c * M.T * sim * M
        sim = temp + np.identity(n) - np.diag(np.diag(temp))
    return sim

def raw_google_matrix(G, nodelist=None, weight='weight'):
    """Calculate the raw Google matrix (stochastic without teleportation)"""
    import numpy as np
    M = nx.to_numpy_matrix(G, nodelist=nodelist, dtype=np.float32,
                                 weight=weight)
    n, m = M.shape  # should be square
    assert n == m and n > 0
    # Find 'dangling' nodes, i.e. nodes whose row's sum = 0
    dangling = np.where(M.sum(axis=1) == 0)
    # add constant to dangling nodes' row
    for d in dangling[0]:
        M[d] = 1.0 / n
    # Normalize. We now have the 'raw' Google matrix (cf. example on p. 11 of
    # Langville & Meyer (2006)).
    M = M / M.sum(axis=1)
    return M